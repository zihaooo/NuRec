# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import logging
import math
import weakref

from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type

import lietorch as lt
import numpy as np
import torch

from gsplat.scene import GaussianScene
from omegaconf import DictConfig
from torch import nn

import nurec.models.gaussians.collect as collect

from ncore.data import RowOffsetStructuredSpinningLidarModelParameters
from ncore.sensors import RowOffsetStructuredSpinningLidarModel
from nurec.config.model import (
    DeformableGaussiansLayerConfig,
    ElasticLoadedAHLayerConfig,
    RigidGaussiansLayerConfig,
    SHGaussiansLayerConfig,
)
from nurec.config.trainer import TrainerConfig
from nurec.datasets.summary import DataSourceSummary
from nurec.datasets.track_split import TrackSplitChild, TrackSplitMetadata
from nurec.datasets.tracks import CuboidTracks
from nurec.models.base import BaseModel
from nurec.models.composite import LayerTrackIds
from nurec.models.gaussians.initializations import BaseInitialization
from nurec.models.gaussians.utils import (
    Asset,
    RGBPostProcessing,
    apply_rgb_post_processing_to_features_dc,
    sh_degree_to_specular_dim,
    write_ply_3dgrt,
    write_ply_3dgs,
)
from nurec.models.input_embedding import (
    BaseInputEmbedding,
    HolisticRemapTimeInputEmbedding,
    IndividualRemapTimeInputEmbedding,
    IndividualStepTimeInputEmbedding,
)
from nurec.models.object_feature_volume import HashGridObjectFeatureVolume
from nurec.models.tracks_calib import (
    BaseTracksCalib,
    CompositeTracksCalib,
    DirectTracksCalib,
    ElevationAwareUnicycleTracksCalib,
    make_row_grad_hook,
)
from nurec.models.utils import ExpActivation, get_activation, update_module_step
from nurec.utils.batch import DataAndRenderingBatch, RenderingData
from nurec.utils.cuda_rng import get_cuda_generator
from nurec.utils.misc import all_gather_tensor_list, unpack_optional
from nurec.utils.morton import VOXEL_SIZE, morton_sort_order
from nurec.utils.optim import OptimizerLRSchedulerConfig, configure_optimizers
from nurec.utils.prober import get_global_prober
from nurec.utils.profiling import ScopedTimer
from nurec.utils.trainer import adjust_step_for_world_size
from nurec.utils.types import AABB3D, SceneContractor


log = logging.getLogger(__name__)
prober = get_global_prober()


class GaussianExportFormat(str, Enum):
    _3DGS = "3dgs"
    _3DGRT = "3dgrt"


@dataclass(frozen=True, slots=True)
class PreparedRigidMotion:
    """An owner's motion and autograd graph, shared for one sensor collection.

    ``poses`` is [num_tracks, 7] translation plus quaternion xyzw; ``keep_mask``
    is [num_tracks] bool; and ``timestamps_us`` is the [num_tracks] timestamp
    used to resolve each pose. Dependent Gaussian IDs index these same owner tensors.
    An owner may append tracks after binding, but must preserve existing indices
    so dependent Gaussian IDs continue to identify the same tracks.
    Both layers keep the same pose graph, so autograd sums their contributions
    into the owner's calibration. ``track_time_data`` describes the same owner
    timeline for sparse collection planning. A new collection prepares a new graph.
    """

    poses: torch.Tensor
    keep_mask: torch.Tensor
    timestamps_us: torch.Tensor
    track_time_data: collect.TrackTimeData | None = None


@dataclass(frozen=True, slots=True)
class PreparedDeformableTableReplacements:
    """Allocated replacements for every DeformNet table indexed by track."""

    instance_embedding_weight: nn.Parameter
    timestamps_us_ranges: torch.Tensor
    aabb_extent: torch.Tensor
    aabb_blb: torch.Tensor
    aabb_trf: torch.Tensor


@dataclass(frozen=True, slots=True)
class PreparedDeformableTrackInsertion:
    """Legacy DeformNet state staged for an inserted actor track.

    ``replacements`` is ``None`` when the live tables already cover the
    requested track count. Otherwise every replacement has been allocated
    before the caller starts publishing the scene-wide insertion.
    """

    inserted_track_indices: frozenset[int]
    num_embeddings: int
    replacements: PreparedDeformableTableReplacements | None


@dataclass(frozen=True, slots=True)
class PreparedRigidAssetInsertion:
    """Fully materialized rigid-layer state for one atomic asset insertion.

    Preparation performs every tensor construction and track-dependent factory call
    without changing the live model. Publishing this value therefore consists only
    of assigning objects that have already been validated and allocated.
    """

    track_id: str
    cuboid_tracks: CuboidTracks
    tracks_calib: BaseTracksCalib
    time_embed: BaseInputEmbedding | None
    positions: nn.Parameter
    scales: nn.Parameter
    rotations: nn.Parameter
    densities: nn.Parameter
    features_albedo: nn.Parameter
    features_specular: nn.Parameter
    extra_signal: nn.Parameter
    camera_extra_signal: nn.Parameter
    lidar_extra_signal: nn.Parameter
    gaussian_cuboid_ids: nn.Buffer
    track_albedos: nn.Parameter | None
    track_scales: nn.Parameter | None
    model_specific_state: object | None


@lru_cache(maxsize=16)
def compute_stable_merging_indices(full_length: int, world_size: int, device: torch.device) -> torch.Tensor:
    # Because the gaussian parameters are distributed as [world_rank::world_size], we might want to re-order the
    # parameters back to the original order after all-gather to produce a consistent state_dict. This is done by
    # creating a mapping from the original indices to the all-gathered indices.
    full_indices = torch.arange(full_length, device=device)
    local_indices = []
    for rank in range(world_size):
        local_indices.append(full_indices[rank::world_size])
    merged_indices = torch.cat(local_indices)
    return torch.argsort(merged_indices)


@ScopedTimer("distributed_all_gather_gaussian_parameters")
def distributed_all_gather_gaussian_parameters(
    local_parameters: dict[str, torch.Tensor],
    keys: list[str] | None = None,
    stable: bool = False,
    *,
    group: torch.distributed.ProcessGroup | None = None,
) -> dict[str, torch.Tensor]:
    # If distributed training is not initialized, or the local parameters are empty, we can just return the local parameters
    if not local_parameters or not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1:
        return local_parameters
    if group is None:
        raise RuntimeError("Distributed Gaussian gathering requires the dedicated process group")

    keys = list(local_parameters.keys()) if keys is None else keys
    assert len(keys) > 0, "keys must be non-empty"

    world_size = torch.distributed.get_world_size(group)

    # reshape all parameters to [num_gaussians, D] and record the shapes
    length = local_parameters[keys[0]].shape[0]
    for key in keys:
        assert local_parameters[key].shape[0] == length, f"{key} {local_parameters[key].shape[0]} != {length}"
    shapes = [local_parameters[key].shape for key in keys]
    dtypes = [local_parameters[key].dtype for key in keys]

    # Handle empty case - compute the feature dimension explicitly
    if length == 0:
        # For empty tensors, calculate feature_dim from the original shape
        # shape is [0, d1, d2, ...], so feature_dim is d1 * d2 * ...
        params = []
        for key in keys:
            feature_dim = 1
            for dim in local_parameters[key].shape[1:]:
                feature_dim *= dim
            params.append(local_parameters[key].reshape(0, feature_dim))
    else:
        params = [local_parameters[key].reshape(length, -1) for key in keys]  # local parameters after reshape

    # All gather the parameters from all ranks. `parameters_all_ranks` is a list of tensors,
    # each tensor in it is a GS attribute with the shape [all_gaussians, D]
    full_params = all_gather_tensor_list(world_size, params, group=group)
    full_length = len(full_params[0])

    # recover the original shape of the parameters, and overwrite the parameter dict.
    full_parameters = local_parameters
    for key, shape, dtype, param in zip(keys, shapes, dtypes, full_params):
        full_parameters[key] = param.reshape(full_length, *shape[1:]).to(dtype)

    if stable:
        indices = compute_stable_merging_indices(full_length, world_size, torch.device("cuda"))
        for key in keys:
            full_parameters[key] = full_parameters[key][indices, ...]

    return full_parameters


_NUREC_TO_MCMC_SPLAT = {
    "positions": "means",
    "rotations": "quats",
    "scales": "scales",
    "densities": "opacities",
}
_MCMC_TO_NUREC_PARAMETER = {splat: parameter for parameter, splat in _NUREC_TO_MCMC_SPLAT.items()}


class BaseGaussianModel(BaseModel, ABC):
    # Core Gaussian parameters
    # Positions of the 3D Gaussians (x, y, z) [n_gaussians, 3]
    positions: nn.Parameter | nn.UninitializedParameter
    # Rotation of each Gaussian represented as a unit quaternion [n_gaussians, 4]
    rotations: nn.Parameter | nn.UninitializedParameter
    # Anisotropic scale of each Gaussian [n_gaussians, 3]
    scales: nn.Parameter | nn.UninitializedParameter
    # Density of each Gaussian [n_gaussians, 1]
    densities: nn.Parameter | nn.UninitializedParameter

    # Extra signal parameters (sensor-specific)
    # Common extra signals for both camera and lidar [n_gaussians, extra_signal_dim]
    extra_signal: nn.Parameter | nn.UninitializedParameter
    # Camera-specific extra signals [n_gaussians, camera_extra_signal_dim]
    camera_extra_signal: nn.Parameter | nn.UninitializedParameter
    # Lidar-specific extra signals [n_gaussians, lidar_extra_signal_dim]
    lidar_extra_signal: nn.Parameter | nn.UninitializedParameter

    scene_extent: float

    GAUSSIANS_VARIANTS: dict[str, Type[BaseGaussianModel]] = {}

    config: SHGaussiansLayerConfig  # type: ignore[assignment] # All gaussian models have SH config fields

    @staticmethod
    def register_to_gaussians_factory(name: str, cls: Type[BaseGaussianModel]) -> None:
        if name in BaseGaussianModel.GAUSSIANS_VARIANTS:
            raise KeyError(f"{name=} already in GAUSSIANS_VARIANTS.")
        BaseGaussianModel.GAUSSIANS_VARIANTS[name] = cls

    # Currently we only have one variant, but we anticipate exploring more in the future
    @staticmethod
    def factory(
        name: str,
        config: SHGaussiansLayerConfig,
        trainer_config: TrainerConfig,
        datasource: DataSourceSummary,
        init_from_datasource: bool,
        initializer: Optional[BaseInitialization] = None,
        precision: Optional[int] = None,
        cuboid_tracks: Optional[CuboidTracks] = None,
        start_end_timestamp_us: Optional[Tuple[int, int]] = None,
    ) -> BaseGaussianModel:
        return BaseGaussianModel.GAUSSIANS_VARIANTS[name](
            config=config,
            trainer_config=trainer_config,
            datasource=datasource,
            init_from_datasource=init_from_datasource,
            initializer=initializer,
            precision=precision,
            cuboid_tracks=cuboid_tracks,
            start_end_timestamp_us=start_end_timestamp_us,
        )

    def __init__(
        self,
        config: SHGaussiansLayerConfig,
        trainer_config: TrainerConfig,
        datasource: DataSourceSummary,
        init_from_datasource: bool,
        initializer: Optional[BaseInitialization] = None,
        precision: Optional[int] = None,
        cuboid_tracks: Optional[CuboidTracks] = None,
        start_end_timestamp_us: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__(config=config.to_dictconfig())
        self.config = config  # type: ignore[assignment]  # Override with typed config
        self.trainer_config = trainer_config

        self.density_activation = get_activation(self.config.density_activation)
        self.density_activation_inv = get_activation(self.config.density_activation, inverse=True)
        self.scale_activation = get_activation(self.config.scale_activation)
        self.scale_activation_inv = get_activation(self.config.scale_activation, inverse=True)
        self.rotation_activation = get_activation(self.config.rotation_activation)
        self.scene_extent = datasource.get_aabb().get_extent().max().item()

        # The packing process creates mapping dictionaries that specify how signals are
        # arranged in the parameter tensors, enabling efficient rendering for different
        # sensor types.

        # Step 1: Initialize packing info dictionaries for Gaussian-level signals
        # These track how signals are packed and activated within each Gaussian's parameter tensors
        self.extra_signal_param_infos: tuple[dict[str, int], list[int]] = (
            {},
            [],
        )  # Common signals: ({signal_name: index}, [signal_dims])
        self.camera_extra_signal_param_infos: tuple[dict[str, int], list[int]] = (
            {},
            [],
        )  # Camera signals: ({signal_name: index}, [signal_dims])
        self.lidar_extra_signal_param_infos: tuple[dict[str, int], list[int]] = (
            {},
            [],
        )  # Lidar signals: ({signal_name: index}, [signal_dims])

        extra_signal_configs = self.config.extra_signal
        if extra_signal_configs is not None:
            # Track index for each signal type
            start_index = 0
            camera_start_index = 0
            lidar_start_index = 0

            # First pass: Pack signals into their respective parameter tensors
            for signal_name, params in extra_signal_configs.items():
                signal_dim = params["n_signal_dim"]
                sensor_type = params["sensor_type"]
                signal_activation = get_activation(params["activation"])

                if sensor_type == "common":
                    # Common signals are stored in the extra_signal parameter tensor
                    self.extra_signal_param_infos[0][str(signal_name)] = start_index
                    self.extra_signal_param_infos[1].append(signal_dim)
                    start_index += 1
                elif sensor_type == "camera":
                    # Camera-specific signals are stored in the camera_extra_signal parameter tensor
                    self.camera_extra_signal_param_infos[0][str(signal_name)] = camera_start_index
                    self.camera_extra_signal_param_infos[1].append(signal_dim)
                    camera_start_index += 1
                elif sensor_type == "lidar":
                    # Lidar-specific signals are stored in the lidar_extra_signal parameter tensor
                    self.lidar_extra_signal_param_infos[0][str(signal_name)] = lidar_start_index
                    self.lidar_extra_signal_param_infos[1].append(signal_dim)
                    lidar_start_index += 1
                else:
                    raise ValueError(f"Invalid sensor_type '{sensor_type}' for signal '{signal_name}'. ")

        # Step 2: Create ray-level signal packing info for rendering
        # These dictionaries specify how signals are arranged in the final rendered output
        # for each sensor type. This is different from Gaussian-level packing because:
        # - Common signals are duplicated for both camera and lidar rendering
        # - Sensor-specific signals are only included in their respective outputs
        self.camera_extra_ray_signal_infos: tuple[list[str], list[int], list[Callable]] = ([], [], [])
        self.lidar_extra_ray_signal_infos: tuple[list[str], list[int], list[Callable]] = ([], [], [])

        if extra_signal_configs is not None:
            # First pass: Add common signals to both camera and lidar ray outputs
            for signal_name, params in extra_signal_configs.items():
                if params["sensor_type"] == "common":
                    signal_dim = params["n_signal_dim"]
                    signal_activation = get_activation(params["activation"])
                    # Common signals appear in both camera and lidar ray outputs
                    self.camera_extra_ray_signal_infos[0].append(str(signal_name))
                    self.camera_extra_ray_signal_infos[1].append(signal_dim)
                    self.camera_extra_ray_signal_infos[2].append(signal_activation)

                    self.lidar_extra_ray_signal_infos[0].append(str(signal_name))
                    self.lidar_extra_ray_signal_infos[1].append(signal_dim)
                    self.lidar_extra_ray_signal_infos[2].append(signal_activation)

            # Second pass: Add sensor-specific signals to their respective ray outputs
            for signal_name, params in extra_signal_configs.items():
                sensor_type = params["sensor_type"]
                signal_dim = params["n_signal_dim"]
                signal_activation = get_activation(params["activation"])
                if sensor_type == "camera":
                    # Camera-specific signals only appear in camera ray outputs
                    self.camera_extra_ray_signal_infos[0].append(str(signal_name))
                    self.camera_extra_ray_signal_infos[1].append(signal_dim)
                    self.camera_extra_ray_signal_infos[2].append(signal_activation)
                elif sensor_type == "lidar":
                    # Lidar-specific signals only appear in lidar ray outputs
                    self.lidar_extra_ray_signal_infos[0].append(str(signal_name))
                    self.lidar_extra_ray_signal_infos[1].append(signal_dim)
                    self.lidar_extra_ray_signal_infos[2].append(signal_activation)

        # We need to wait for the datasource and torch.distributed context to be initialized. We will re-initializing
        # particles from datasource and validate the model's parameters later
        self.positions = nn.UninitializedParameter()
        self.rotations = nn.UninitializedParameter()
        self.scales = nn.UninitializedParameter()
        self.densities = nn.UninitializedParameter()
        self.extra_signal = nn.UninitializedParameter()
        self.camera_extra_signal = nn.UninitializedParameter()
        self.lidar_extra_signal = nn.UninitializedParameter()

        # Defer initialization of Gaussians buffers and validate the model's parameters.
        def _maybe_initialize_gaussians():
            if init_from_datasource:
                assert initializer is not None and datasource is not None, (
                    f"{self.__class__.__name__}: initializer and datasource must be provided when init_from_datasource is True"
                )
                self.initialize_gaussians_from_datasource(
                    initializer=unpack_optional(initializer),
                    datasource=datasource,
                    cuboid_tracks=cuboid_tracks,
                )
                self._validate_fields()

        self._maybe_initialize_gaussians = _maybe_initialize_gaussians

        # Misc
        self.overriden_parameters: Dict[str, nn.Parameter] = {}
        self.overriden_buffers: Dict[str, nn.Buffer] = {}
        self.overriden_state: Dict[str, Any] = {}

    def maybe_initialize_gaussians(self) -> None:
        self._maybe_initialize_gaussians()

    def freeze_tracks(self, frozen_track_indices: Sequence[int]) -> None:
        """Stop gradients for the given tracks (indices into this node's track
        subset). Default no-op — nodes without per-track parameters override.
        """
        return

    def validate_external_gaussian_topology_edit(self, operation: str) -> None:
        """Reject an unsupported edit before external code changes Gaussian rows.

        Regular Gaussian layers allow inference-time row insertion and
        replacement. Models with auxiliary row-aligned state override this
        preflight so callers can fail atomically, before mutating any layer.
        """
        return

    def _get_slice_boundaries(self) -> tuple[int, int]:
        # In distributed training, each rank only hosts a subset of Gaussians.
        if self.trainer_config.world_size > 1:
            assert torch.distributed.is_initialized()
            assert torch.distributed.get_world_size() == self.trainer_config.world_size, (
                f"{torch.distributed.get_world_size()} != {self.trainer_config.world_size}"
            )
            world_rank = torch.distributed.get_rank()
            world_size = self.trainer_config.world_size
        else:
            world_rank = 0
            world_size = 1
        return world_rank, world_size

    def configure_sharded_params_and_buffers(self) -> list[str]:
        ret = super().configure_sharded_params_and_buffers()
        BaseModel.mark_as_sharded(self.positions)
        BaseModel.mark_as_sharded(self.rotations)
        BaseModel.mark_as_sharded(self.scales)
        BaseModel.mark_as_sharded(self.densities)
        BaseModel.mark_as_sharded(self.extra_signal)
        BaseModel.mark_as_sharded(self.camera_extra_signal)
        BaseModel.mark_as_sharded(self.lidar_extra_signal)
        return ret + [
            "positions",
            "rotations",
            "scales",
            "densities",
            "extra_signal",
            "camera_extra_signal",
            "lidar_extra_signal",
        ]

    def initialize_gaussians_from_datasource(
        self,
        initializer: BaseInitialization,
        datasource: DataSourceSummary,
        cuboid_tracks: Optional[CuboidTracks] = None,
    ) -> None:
        with ScopedTimer(f"BaseGaussianModel/{initializer.__class__.__name__}/initialize_from_datasource"):
            # Because initializer is not attached to any module, we need to manually move it to the correct device.
            initializer.to(self.device).initialize_from_datasource(datasource, cuboid_tracks=cuboid_tracks)
        if getattr(initializer, "ground_mesh_artifact", None) is not None:
            self.ground_mesh_artifact = initializer.ground_mesh_artifact
            self.ground_mesh_artifact_settings = initializer.ground_mesh_artifact_settings

        world_rank, world_size = self._get_slice_boundaries()

        self.positions = nn.Parameter(initializer.positions[world_rank::world_size].contiguous())
        self.rotations = nn.Parameter(initializer.rotations[world_rank::world_size].contiguous())
        self.scales = nn.Parameter(initializer.scales[world_rank::world_size].contiguous())
        self.densities = nn.Parameter(initializer.densities[world_rank::world_size].contiguous())
        self.extra_signal = nn.Parameter(
            torch.rand(
                self.positions.shape[0],
                self.config.particle.extra_signal_dim,
                dtype=self.positions.dtype,
                device=self.device,
            )
        )
        self.camera_extra_signal = nn.Parameter(
            torch.rand(
                self.positions.shape[0],
                self.config.particle.camera_extra_signal_dim,
                dtype=self.positions.dtype,
                device=self.device,
            )
        )
        # When lidar_extra_signal_sph_degree > 0 the parameter holds SH coefficients laid out
        # coefficient-major as [coeff0_ch0..coeff0_chD, coeff1_ch0..]. Only the DC band (the first
        # lidar_extra_signal_dim columns) is initialized; the higher-order bands start at zero so
        # the signal is view-independent at init and view dependence is learned, mirroring the
        # specular SH features (see initializations.py). Random-initializing every band would make
        # SH evaluation would produce large, view-dependent garbage from step 0.
        lidar_signal_dim = self.config.particle.lidar_extra_signal_dim
        lidar_extra_signal = torch.zeros(
            self.positions.shape[0],
            lidar_signal_dim * self.get_lidar_extra_signal_sph_dim(),
            dtype=self.positions.dtype,
            device=self.device,
        )
        lidar_extra_signal[:, :lidar_signal_dim] = torch.rand(
            self.positions.shape[0],
            lidar_signal_dim,
            dtype=self.positions.dtype,
            device=self.device,
        )
        self.lidar_extra_signal = nn.Parameter(lidar_extra_signal)

    def configure_optimizers(self, name_prefix: str = "") -> list[OptimizerLRSchedulerConfig]:
        """
        Configure the optimizers of the Gaussian model. We keep a reference to the optimizers such that the strategies
        can modify them, but also pass it to the global optimizers to perform steps/backwards.
        """

        optimizers = configure_optimizers(self.config.to_dictconfig(), self.trainer_config, self, name_prefix)

        # pos_lr_scene_extent: an explicit extent for the position learning rate (e.g. a cell on part of a dataset)
        lr_extent = self.config.pos_lr_scene_extent or self.scene_extent
        if self.config.scale_pos_lr_by_scene_extent and lr_extent != self.scene_extent:
            log.info(
                "position learning rate scaled by extent %.1f m (dataset scene_extent %.1f m)",
                lr_extent,
                self.scene_extent,
            )
        for opt in optimizers:
            for param_group in opt["optimizer"].param_groups:
                if "positions" in param_group["name"] and self.config.scale_pos_lr_by_scene_extent:
                    param_group["lr"] *= 0.5 * lr_extent  # Multiply the position lr by the scene scale
                    if "lr_scheduler" in opt:
                        opt["lr_scheduler"]["scheduler"].base_lrs[0] *= 0.5 * lr_extent

        # Make a reference to the optimizer such that the strategies will be able to modify it
        self.optimizers = optimizers

        readable = []
        for entry in optimizers:
            optimizer = entry["optimizer"]
            opt_class = optimizer.__class__.__name__
            group_names = [pg.get("name", f"group{idx}") for idx, pg in enumerate(optimizer.param_groups)]
            readable.append(f"{opt_class}[" + ", ".join(map(str, group_names)) + "]")
        log.info(f"Create optimizer for {self.__class__.__name__}: " + "; ".join(readable))

        return optimizers

    def _validate_fields(self) -> None:
        """
        Validate that all the per-Gaussian fields have the correct shape
        """
        num_gaussians = self.get_num_gaussians()
        assert self.positions.shape == (num_gaussians, 3)
        assert self.densities.shape == (num_gaussians, 1)
        assert self.rotations.shape == (num_gaussians, 4)
        assert self.scales.shape == (num_gaussians, 3)
        assert self.extra_signal.shape == (
            num_gaussians,
            self.config.particle.extra_signal_dim,
        )
        assert self.camera_extra_signal.shape == (
            num_gaussians,
            self.config.particle.camera_extra_signal_dim,
        )
        assert self.lidar_extra_signal.shape == (
            num_gaussians,
            self.config.particle.lidar_extra_signal_dim * self.get_lidar_extra_signal_sph_dim(),
        )

    #
    # Implementation of what is necessary for parameter collection.
    #
    @staticmethod
    def get_frame_window(rendering_data: RenderingData) -> tuple[int, int]:
        """The frame's (start, end) timestamps in microseconds, read from the host in one go."""
        timestamps = rendering_data.timestamps_startend_us_cpu
        assert timestamps is not None
        assert timestamps.shape == (1, 2)
        assert timestamps.dtype == torch.int64
        start_us, end_us = (int(t) for t in timestamps[0].tolist())
        assert end_us >= start_us, f"frame window runs backwards: [{start_us}, {end_us}]"
        return start_us, end_us

    @staticmethod
    def get_frame_timestamp(rendering_data: RenderingData) -> int:
        start_us, end_us = BaseGaussianModel.get_frame_window(rendering_data)
        return (start_us + end_us) // 2

    @dataclass(slots=True)
    class CollectionContext:
        rendering_data: RenderingData
        is_training_batch: bool
        tracks_edit: Optional[LayerTrackIds.Edit]
        frustum_cull: Optional[collect.TracksFrustumCullData] = None
        # True asks a tracked layer to publish ``track_time_data`` (below) so the
        # composite can decide on the host which track windows to collect and render.
        build_segment_plan: bool = False
        prepared_motion: PreparedRigidMotion | None = None

        # Each value is ``(tensor, include_in_collected_output)``.
        additional_data: dict[str, Tuple[torch.Tensor, bool]] = field(default_factory=dict)

        # OUTPUT, written by a tracked layer when ``build_segment_plan`` is set: the
        # per-track pose timestamps the collector interpolates from, or ``None`` for a
        # layer whose tracks are always alive (``is_static``).
        track_time_data: Optional[collect.TrackTimeData] = None

        # OUTPUT, written by the layer during collection (see
        # RigidGaussianModel._filter_edited_gaussians): when ``tracks_edit`` row-filtered
        # the layer, the unfiltered row of each surviving collected row, in the same
        # (post-gather) index space as the parameters handed to that call. ``None`` means
        # the layer was collected whole. Consumers whose indices address the unfiltered
        # layer remap through this; the context is per-collect, so there is no way to
        # read a previous frame's mapping.
        kept_rows: Optional[torch.Tensor] = None

    def get_parameters(self) -> dict[str, torch.Tensor]:
        return {
            "positions": self.positions,
            "rotations": self.rotations,
            "scales": self.scales,
            "densities": self.densities,
            "extra_signal": self.extra_signal,
            "camera_extra_signal": self.camera_extra_signal,
            "lidar_extra_signal": self.lidar_extra_signal,
        }

    @staticmethod
    def is_mcmc_row_aligned_tensor(
        value: torch.Tensor,
        row_count: int,
        *,
        device: torch.device | None = None,
    ) -> bool:
        """Return whether ``value`` is materialized row state for an MCMC owner."""
        return (
            not isinstance(value, (nn.UninitializedParameter, nn.UninitializedBuffer))
            and value.ndim > 0
            and value.shape[0] == row_count
            and (device is None or value.device == device)
        )

    def get_mcmc_row_parameters(self) -> dict[str, nn.Parameter] | None:
        """Return all live floating row Parameters, rejecting structural drift."""
        parameters = self.get_parameters()
        if any(isinstance(value, (nn.UninitializedParameter, nn.UninitializedBuffer)) for value in parameters.values()):
            return None

        row_count = self.get_num_gaussians()
        row_parameters: dict[str, nn.Parameter] = {}
        for name, value in parameters.items():
            if not isinstance(value, nn.Parameter):
                continue
            if not value.is_floating_point():
                raise TypeError(f"{name} is not a floating-point MCMC row parameter")
            if not self.is_mcmc_row_aligned_tensor(value, row_count):
                raise ValueError(f"{name} is not row-aligned ({tuple(value.shape)} for {row_count} rows)")
            row_parameters[name] = value
        return row_parameters

    def refresh_mcmc_scene(self, scene_id: str) -> GaussianScene | None:
        """Adopt exact live row Parameters, or leave a lazy model unbound."""
        current_scene_id = getattr(self, "_mcmc_scene_id", None)
        if current_scene_id is not None and current_scene_id != scene_id:
            raise ValueError(f"MCMC scene id is immutable ({current_scene_id!r} != {scene_id!r})")
        self._mcmc_scene_id = scene_id

        row_parameters = self.get_mcmc_row_parameters()
        if row_parameters is None:
            self._mcmc_scene = None
            return None

        splats = {_NUREC_TO_MCMC_SPLAT.get(name, name): value for name, value in row_parameters.items()}
        if missing := sorted(set(_NUREC_TO_MCMC_SPLAT.values()) - splats.keys()):
            raise ValueError(f"missing MCMC row parameters: {missing}")

        self._mcmc_scene = GaussianScene.from_splats(nn.ParameterDict(splats), id=scene_id)
        return self._mcmc_scene

    def get_mcmc_scene(self) -> GaussianScene | None:
        """Return the current owner-local runtime scene, if materialized."""
        return getattr(self, "_mcmc_scene", None)

    def mark_mcmc_publication_failed(self) -> None:
        """Latch an unrecoverable post-gsplat publication failure."""
        self._mcmc_publication_failed = True

    def raise_if_mcmc_publication_failed(self) -> None:
        """Reject later refinement or checkpointing after partial publication."""
        if getattr(self, "_mcmc_publication_failed", False):
            raise RuntimeError("MCMC publication previously failed; model state is unsafe to continue or checkpoint")

    def register_mcmc_parameter_hooks(self, name: str, parameter: nn.Parameter) -> None:
        """Attach model-owned hooks before an MCMC topology update installs a replacement parameter."""

    def apply_mcmc_row_update(
        self,
        source_row_map: torch.Tensor,
        retained_count: int,
        recycled_row_indices: torch.Tensor | None,
    ) -> None:
        """Update non-row-shaped model state after MCMC rewrites Gaussian rows.

        Ordinary Gaussian models keep their mutable state in row-shaped
        parameters and buffers handled by the MCMC adapter. Representations
        such as temporal tables override this hook to update external row
        mappings and copy a donor's complete trajectory.
        """

    def validate_mcmc_row_update(self, recycled_row_indices: torch.Tensor | None) -> None:
        """Fail before mutation when a model cannot represent an MCMC row plan."""

    def get_layer_config(self) -> collect.LayerConfigBase:
        if self.config.rotation_activation == "normalize":
            rotation_activation = collect.RotationActivation.NORMALIZE
        else:
            raise NotImplementedError(f"Invalid rotation activation: {self.config.rotation_activation}")

        if self.config.scale_activation == "exp":
            scale_activation = collect.ScaleActivation.EXP
        else:
            raise NotImplementedError(f"Invalid scale activation: {self.config.scale_activation}")

        if self.config.density_activation == "sigmoid":
            density_activation = collect.DensityActivation.SIGMOID
        else:
            raise NotImplementedError(f"Invalid density activation: {self.config.density_activation}")

        return collect.LayerConfigBase(
            rotation_activation=rotation_activation,
            scale_activation=scale_activation,
            density_activation=density_activation,
        )

    def get_layer_data(
        self, context: CollectionContext, gathered_parameters: dict[str, torch.Tensor]
    ) -> collect.LayerDataBase:
        return collect.LayerDataBase(
            positions=gathered_parameters["positions"],
            rotations=gathered_parameters["rotations"],
            scales=gathered_parameters["scales"],
            densities=gathered_parameters["densities"],
            extra_signal=gathered_parameters["extra_signal"],
            camera_extra_signal=gathered_parameters["camera_extra_signal"],
            lidar_extra_signal=gathered_parameters["lidar_extra_signal"],
        )

    def post_process(
        self,
        offset: int,
        results: collect.CollectorResult,
        layer_data: collect.LayerDataBase,
        rows: Optional[np.ndarray] = None,
    ) -> None:
        """Transform this layer's rows in the collected buffers, after the collect wrote them.

        ``rows`` is the layer-local ``[count, 2]`` ``[start, count]`` list the collect
        actually wrote this frame (``segment_plan.collect_ranges``), or ``None`` for the
        whole layer. An implementation that reads a row back before writing it must
        honour it: on a windowed collect the other rows still hold the values a previous
        frame left, transform included, so touching them applies the transform twice.
        """

    #
    # End of implementation of what is necessary for parameter collection.
    #

    def get_scales(self, preactivation=False) -> torch.Tensor:
        assert isinstance(self.scales, nn.Parameter), "should be initialized"
        if preactivation:
            return self.scales
        else:
            return self.scale_activation(self.scales)

    # Most of the NuRec codebase assumes quaternions are in xyzw format, but the 3DGUT rasterizer
    # (and original 3DGS) assumes wxyz format
    def get_rotations(self, quaternion_format: Literal["xyzw", "wxyz"], preactivation=False) -> torch.Tensor:
        assert isinstance(self.rotations, nn.Parameter), "should be initialized"
        if preactivation:
            rotations = self.rotations
        else:
            rotations = self.rotation_activation(self.rotations)

        match quaternion_format:
            case "xyzw":
                # Rotations are stored internally in wxyz format, as required by 3DGUT/gsplat kernels
                xyz = rotations[:, 1:4]
                w = rotations[:, 0:1]
                return torch.cat([xyz, w], dim=-1)
            case "wxyz":
                return rotations
            case _:
                raise NotImplementedError(f"quaternion_format {quaternion_format} not supported")

    def get_positions(self) -> nn.Parameter:
        assert isinstance(self.positions, nn.Parameter), "should be initialized"
        return self.positions

    def get_densities(self, preactivation=False) -> nn.Parameter:
        assert isinstance(self.densities, nn.Parameter), "should be initialized"
        if preactivation:
            return self.densities
        else:
            return self.density_activation(self.densities)

    def get_lidar_extra_signal_sph_dim(self) -> int:
        return (self.config.particle.lidar_extra_signal_sph_degree + 1) ** 2

    def get_extra_signal(self, sensor_type: str = "common") -> torch.Tensor:
        if sensor_type == "common":
            return self.extra_signal
        elif sensor_type == "camera":
            return self.camera_extra_signal
        elif sensor_type == "lidar":
            return self.lidar_extra_signal
        else:
            raise ValueError(f"Invalid sensor type: {sensor_type}")

    def get_extra_signal_by_key(self, key: str) -> torch.Tensor:
        if key in self.extra_signal_param_infos[0]:
            assert isinstance(self.extra_signal, nn.Parameter), "extra_signal should be initialized"
            return torch.split(self.extra_signal, self.extra_signal_param_infos[1], dim=-1)[
                self.extra_signal_param_infos[0][key]
            ]
        elif key in self.camera_extra_signal_param_infos[0]:
            assert isinstance(self.camera_extra_signal, nn.Parameter), "camera_extra_signal should be initialized"
            return torch.split(self.camera_extra_signal, self.camera_extra_signal_param_infos[1], dim=-1)[
                self.camera_extra_signal_param_infos[0][key]
            ]
        elif key in self.lidar_extra_signal_param_infos[0]:
            assert isinstance(self.lidar_extra_signal, nn.Parameter), "lidar_extra_signal should be initialized"
            return torch.split(self.lidar_extra_signal, self.lidar_extra_signal_param_infos[1], dim=-1)[
                self.lidar_extra_signal_param_infos[0][key]
            ]
        else:
            raise ValueError(f"Invalid signal key: {key}")

    def get_num_gaussians(self) -> int:
        return self.positions.shape[0]

    def has_temporal_appearance_for_export(self) -> bool:
        """Whether exporters must author time-sampled SH for this model."""

        return False

    def get_additional_buffers(self) -> dict[str, torch.Tensor]:
        """Buffers this class owns beyond the standard Gaussian parameters, keyed by attribute name.

        NOTE: please add row-aligned per-Gaussian buffers here, chaining through super().
        Morton reorder read this to keep rows in step.
        """
        return {}

    @staticmethod
    def background_keep_mask(
        positions: torch.Tensor, trajectories: List[torch.Tensor], cuboid_dims: List[torch.Tensor]
    ) -> torch.Tensor:
        """The per-gaussian KEEP mask for trajectory-cuboid removal: a gaussian is **removed** when it
        lies inside the cuboid (within ``0.5 * cuboid_dim`` of the box center on **every** axis) at **any**
        pose of **any** trajectory; kept otherwise. Pure (reads ``positions``, no mutation).
        """
        include_mask = torch.ones_like(positions[..., 0], dtype=torch.bool)
        # We can probably calculate the mask in one batched operation, but this might cause OOMs
        # depending on the number of gaussians and trajectories.
        for trajectory, cuboid_dim in zip(trajectories, cuboid_dims, strict=True):
            positions_trajectory = (positions @ trajectory[:, :3, :3]) - (
                trajectory[:, :3, :3].permute(0, 2, 1) @ trajectory[:, :3, 3:]
            ).permute(0, 2, 1)
            include_mask = torch.logical_and(
                include_mask, (positions_trajectory.abs() > 0.5 * cuboid_dim).any(dim=-1).all(dim=0)
            )
        return include_mask

    def remove_gaussians_in_trajectories(
        self, trajectories: List[torch.Tensor], cuboid_dims: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Removes gaussians along trajectories within the specified cuboid dims.
        Note that the override is only meant to be used at inference time and should be reverted
        by calling restore_training_parameters before resuming training.
        Args:
            trajectories: List of N trajectories. Each trajectory has a variable number of 4x4 poses
            cuboid_dims: List of N cuboid dims. Gaussians within cuboid_dim of each trajectory pose will be removed
        Return:
            Mask indicating which of the original gaussians remain.
        """
        include_mask = self.background_keep_mask(self.positions, trajectories, cuboid_dims)

        self.save_training_parameters()
        self.positions = nn.Parameter(self.positions[include_mask])
        self.scales = nn.Parameter(self.scales[include_mask])
        self.rotations = nn.Parameter(self.rotations[include_mask])
        self.densities = nn.Parameter(self.densities[include_mask])
        self.extra_signal = nn.Parameter(self.extra_signal[include_mask])
        # The per/camera/lidar extra-signal tensors are also per-gaussian and must be filtered by the
        # same mask, else a model with `camera_extra_signal_dim`/`lidar_extra_signal_dim` > 0 ends up with a
        # row-count mismatch (collector asserts `nb_gaussians == camera_extra_signal.shape[0]`).
        self.camera_extra_signal = nn.Parameter(self.camera_extra_signal[include_mask])
        self.lidar_extra_signal = nn.Parameter(self.lidar_extra_signal[include_mask])

        return include_mask

    @torch.no_grad()
    def compute_morton_order(self, voxel_size: float = VOXEL_SIZE) -> torch.Tensor | None:
        """Return the Morton permutation for this node's rows, without applying it.

        Separate from apply_morton_order so a caller can compute every node's order before any rows move.
        Positions that cannot be sorted raise here, while nothing has been mutated yet.

        Returns None if there is nothing to sort.

        Raises:
            ValueError: If positions are non-finite or exceed the Morton range.
        """
        positions = self.positions
        if isinstance(positions, torch.nn.parameter.UninitializedParameter) or positions.shape[0] == 0:
            return None

        return morton_sort_order(positions, voxel_size, label=f"{type(self).__name__}.positions").order

    def validate_optimizer_state(self, order: torch.Tensor) -> None:
        """Raise if this node's optimizer state cannot be permuted by the given order.

        Runs in the compute phase: apply_morton_order permutes rows before it touches the optimizer, so
        discovering a problem there would leave the node half-permuted.

        Row alignment is read from the leading dimension rather than the state key, since key names differ
        per optimizer. Only state reachable through optimizer.state is seen, so an optimizer holding
        row-aligned data elsewhere would be missed. apex FusedAdam does exactly that under master_weights,
        shadowing every parameter in param_groups_master, and the fused kernel writes parameters back from
        that shadow. Refused rather than handled, since permuting rows without the shadow corrupts the model.

        Raises:
            ValueError: If a non-scalar state entry is not row-aligned, or master weights are present.
        """
        for optim_sched in getattr(self, "optimizers", None) or []:
            optimizer = optim_sched["optimizer"]
            # apex fills param_groups_master with None unless master_weights is on, so a non-None entry
            # is the only signal that a shadow copy exists.
            for master_group in getattr(optimizer, "param_groups_master", None) or []:
                if any(master is not None for master in master_group["params"]):
                    raise ValueError(
                        f"{type(optimizer).__name__} holds master weights outside optimizer.state, "
                        "which this reorder cannot permute. Disable master_weights to sort this layer."
                    )
            for param_group in optimizer.param_groups:
                for param in param_group["params"]:
                    if param.shape[:1] != order.shape:
                        continue
                    for key, value in optimizer.state.get(param, {}).items():
                        if not isinstance(value, torch.Tensor) or value.dim() == 0:
                            continue
                        if value.shape[:1] != order.shape:
                            raise ValueError(
                                f"Optimizer state '{key}' has leading dim {value.shape[0]}, "
                                f"expected {order.shape[0]} to match the Gaussian rows"
                            )

    @torch.no_grad()
    def apply_morton_order(self, order: torch.Tensor) -> None:
        """Permute this node's per-Gaussian rows, its optimizer state and its gradient hooks."""
        # Maps each replaced tensor to its permuted successor, keyed by tensor object the way optimizer.state is.
        self._permuted_params: dict[torch.Tensor, nn.Parameter] = {}
        try:
            self._permute_gaussian_rows(order)
            self._permute_optimizer_state(order)
        finally:
            self._permuted_params = {}
        self._reinstall_grad_hooks()

    @torch.no_grad()
    def morton_sort_gaussians(self, voxel_size: float = VOXEL_SIZE) -> torch.Tensor | None:
        """Reorder this node's Gaussians into Morton order, so that Gaussians close in space sit close in memory.

        Returns the permutation that was applied, or None if there was nothing to sort.
        """
        order = self.compute_morton_order(voxel_size)
        if order is not None:
            self.validate_optimizer_state(order)
            self.apply_morton_order(order)
        return order

    def _reinstall_grad_hooks(self) -> None:
        """Reinstall the per-row gradient hooks this node registered on its own parameters.

        Replacing the parameters drops the hooks, which is what lets their row masks be rebuilt. The composite
        reinstalls the hooks it owns, so this covers only what a node registers for itself.
        """

    def _permute_gaussian_rows(self, order: torch.Tensor) -> None:
        """Apply a row permutation to every per-Gaussian tensor owned by this class.

        Parameters are listed here and subclasses extend this for the ones they add.
        Row-aligned buffers are not listed, they come from get_additional_buffers instead.
        Anything row-aligned that neither path reaches silently desynchronizes from positions.
        """
        self.positions = self._reordered_rows(self.positions, order)
        self.scales = self._reordered_rows(self.scales, order)
        self.rotations = self._reordered_rows(self.rotations, order)
        self.densities = self._reordered_rows(self.densities, order)
        self.extra_signal = self._reordered_rows(self.extra_signal, order)
        self.camera_extra_signal = self._reordered_rows(self.camera_extra_signal, order)
        self.lidar_extra_signal = self._reordered_rows(self.lidar_extra_signal, order)
        self._permute_additional_buffers(order)

    def _permute_additional_buffers(self, order: torch.Tensor) -> None:
        """Permute the row-aligned sidecar buffers a subclass declares in get_additional_buffers.

        That method is already the per-class registry of these buffers, and the MCMC relocation path indexes
        the same dict, so driving the reorder from it keeps the two in step.

        Buffers whose leading dim does not match the order are left alone, the same rule _reordered_rows uses.
        """
        for name, buffer in self.get_additional_buffers().items():
            if isinstance(buffer, nn.UninitializedBuffer) or buffer.shape[:1] != order.shape:
                continue
            setattr(self, name, self._reordered_buffer_rows(buffer, order))

    def _reordered_rows(
        self, param: nn.Parameter | nn.UninitializedParameter, order: torch.Tensor
    ) -> nn.Parameter | nn.UninitializedParameter:
        """Permute the rows of a per-Gaussian parameter, keeping its sharded flag.

        Deliberately returns a new parameter rather than permuting in place. Per-row gradient hooks are bound to
        the tensor object and capture a row mask built for the old order. Replacing the object drops them, so
        the composite rebuilds them from the permuted rows via reapply_component_only_row_hooks.

        Tensors whose leading dim does not match the order are left alone. The camera and lidar extra-signal
        parameters are (0, dim) when the layer carries no such signal. Indexing one of those with a full order
        would raise.

        A live gradient moves with the rows. Lightning runs callbacks before the module hook, so a save can
        land between backward and densification, which reads positions.grad. Dropping it would abort the run,
        and carrying it unpermuted would credit each gradient to the wrong Gaussian.
        """
        if isinstance(param, nn.UninitializedParameter) or param.shape[:1] != order.shape:
            return param

        reordered = nn.Parameter(param[order], requires_grad=param.requires_grad)
        if param.grad is not None:
            reordered.grad = param.grad[order]
        if BaseModel.is_sharded(param):
            BaseModel.mark_as_sharded(reordered)
        self._permuted_params[param] = reordered
        return reordered

    @staticmethod
    def _reordered_buffer_rows(buffer: torch.Tensor, order: torch.Tensor) -> nn.Buffer:
        """Permute the rows of a per-Gaussian buffer, keeping its sharded flag.

        Buffers never sit in an optimizer, so unlike _reordered_rows there is nothing to repoint.
        """
        reordered = nn.Buffer(buffer[order])
        if BaseModel.is_sharded(buffer):
            BaseModel.mark_as_sharded(reordered)
        return reordered

    def _permute_optimizer_state(self, order: torch.Tensor) -> None:
        """Repoint this node's optimizers at the reordered parameters and permute their state.

        Adam's exp_avg and exp_avg_sq are row-aligned, so they have to move with the parameters. Left behind, a
        resumed run picks up moments belonging to other Gaussians. Scalar entries such as the step counter are
        per-optimizer bookkeeping and are left alone.

        Groups are matched by tensor identity against what _permute_gaussian_rows just replaced. Names cannot be
        used, since one group is nameless and holds every parameter of the model.
        """
        for optim_sched in getattr(self, "optimizers", None) or []:
            optimizer = optim_sched["optimizer"]
            for param_group in optimizer.param_groups:
                params = param_group["params"]
                if not any(p in self._permuted_params for p in params):
                    continue
                # A group can hold several parameters, so rebuild the whole list and leave the others in place.
                param_group["params"] = [self._permuted_params.get(p, p) for p in params]
                for old_param, new_param in zip(params, param_group["params"], strict=True):
                    if new_param is old_param:
                        continue
                    state = optimizer.state.pop(old_param, {})
                    for key, value in list(state.items()):
                        # Adam's step counter is scalar per-optimizer bookkeeping, not per-Gaussian,
                        # so it stays put. validate_optimizer_state has already rejected any non-scalar
                        # entry that is not row-aligned, so nothing here can fail partway through.
                        if isinstance(value, torch.Tensor) and value.shape[:1] == order.shape:
                            state[key] = value[order]
                    optimizer.state[new_param] = state

    def save_training_parameters(self):
        """
        Should be called before validation-time modifications such as remove_gaussians_in_trajectories
        and track_ply_override. restore_training_parameters should then be called before resuming
        training.
        """
        if len(self.overriden_parameters) > 0:
            return  # training parameters have already been stored
        self.overriden_parameters["positions"] = self.positions
        self.overriden_parameters["scales"] = self.scales
        self.overriden_parameters["rotations"] = self.rotations
        self.overriden_parameters["densities"] = self.densities
        self.overriden_parameters["extra_signal"] = self.extra_signal
        self.overriden_parameters["camera_extra_signal"] = self.camera_extra_signal
        self.overriden_parameters["lidar_extra_signal"] = self.lidar_extra_signal

    def restore_training_parameters(self) -> None:
        if len(self.overriden_parameters) > 0:
            self.positions = self.overriden_parameters["positions"]
            self.scales = self.overriden_parameters["scales"]
            self.rotations = self.overriden_parameters["rotations"]
            self.densities = self.overriden_parameters["densities"]
            self.extra_signal = self.overriden_parameters["extra_signal"]
            # `camera_extra_signal`/`lidar_extra_signal` were added to the override snapshot after this
            # save/restore pair already existed. Tolerate an older snapshot that predates them: such a
            # save also never masked these fields, so leaving the current tensors untouched is the
            # correct no-op, rather than raising KeyError.
            if "camera_extra_signal" in self.overriden_parameters:
                self.camera_extra_signal = self.overriden_parameters["camera_extra_signal"]
            if "lidar_extra_signal" in self.overriden_parameters:
                self.lidar_extra_signal = self.overriden_parameters["lidar_extra_signal"]
            self.overriden_parameters = {}
            self.overriden_buffers = {}
            self.overriden_state = {}

    def get_number_of_gaussians_per_track(self) -> dict[str, int]:
        """
        Returns the number of gaussians, if tracks are available this is returned per track.
        """
        return {
            "<NO_TRACK>": self.get_num_gaussians(),
        }

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        for p in self.parameters():
            if p.is_cuda:
                p.record_stream(stream)
        for b in self.buffers():
            if b.is_cuda:
                b.record_stream(stream)


class SHGaussianModel(BaseGaussianModel):
    # Feature vector of the 0th order SH coefficients [n_gaussians, 3] (We split it into two due to different learning rates)
    features_albedo: nn.Parameter | nn.UninitializedParameter
    # Features of the higher order SH coefficients [n_gaussians, 3]
    features_specular: nn.Parameter | nn.UninitializedParameter

    n_active_features: int

    config: SHGaussiansLayerConfig  # type: ignore[assignment] # Narrow type from BaseLayerConfig

    def __init__(
        self,
        config: SHGaussiansLayerConfig,
        trainer_config: TrainerConfig,
        datasource: DataSourceSummary,
        init_from_datasource: bool,
        initializer: Optional[BaseInitialization],
        precision: Optional[int] = None,
        cuboid_tracks: Optional[CuboidTracks] = None,
        start_end_timestamp_us: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.fourier_features_dim = unpack_optional(config.fourier_features_dim)
        self.start_end_timestamp_us = start_end_timestamp_us
        super().__init__(
            config=config,
            trainer_config=trainer_config,
            datasource=datasource,
            init_from_datasource=init_from_datasource,
            initializer=initializer,
            precision=precision,
            cuboid_tracks=cuboid_tracks,
        )
        self.n_active_features = 1

        # Tracks the latest global_step seen by update_step_train_batch_start; used by
        # the features_albedo backward hook when fourier_freeze_until_step > 0.
        self._global_step_for_freeze: int = 0

        self.config = copy.deepcopy(self.config)
        self.config.progressive_training.increase_frequency = adjust_step_for_world_size(
            trainer_config, self.config.progressive_training.increase_frequency
        )
        log.info(
            "SHGaussianModel/progressive_training: increase_frequency=%d increase_step=%d",
            self.config.progressive_training.increase_frequency,
            self.config.progressive_training.increase_step,
        )

        self.time_embed = None
        if unpack_optional(self.config.fourier_features_dim) > 1:
            log.info(
                f"SHGaussianModel/time_embed: Fourier features enabled with dim={unpack_optional(self.config.fourier_features_dim)}"
            )
            assert self.config.time_embed is not None, "time_embed must be provided when fourier_features_dim > 1"
            self.time_embed = BaseInputEmbedding.factory(
                self.config.time_embed.name,
                self.config.time_embed,
                trainer_config,
                torch.tensor(self.start_end_timestamp_us),
                cuboid_tracks,
            )

        # We need to wait for the datasource and torch.distributed context to be initialized. We will re-initializing
        # particles from datasource and validate the model's parameters later
        self.features_albedo = nn.UninitializedParameter()
        self.features_specular = nn.UninitializedParameter()

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        # Auto-attach the fourier-freeze backward hook every time features_albedo is
        # rebound (initialization, MCMC remove/add, PLY load). Hook captures self, so
        # it reads the current global_step and config threshold dynamically.
        if (
            name == "features_albedo"
            and isinstance(value, nn.Parameter)
            and not isinstance(value, nn.UninitializedParameter)
            and getattr(self, "fourier_features_dim", 1) > 1
            and getattr(getattr(self, "config", None), "fourier_freeze_until_step", 0) > 0
        ):
            value.register_hook(self._fourier_freeze_grad_hook)

    def _fourier_freeze_grad_hook(self, grad: torch.Tensor) -> torch.Tensor:
        # Zero gradient on temporal Fourier bands (features_albedo[..., 1:, :]) until
        # the DC band has had time to converge. The DC band [..., 0, :] continues to
        # train normally throughout. After the threshold, gradients flow normally.
        if self._global_step_for_freeze < self.config.fourier_freeze_until_step:
            masked = grad.clone()
            masked[..., 1:, :] = 0
            return masked
        return grad

    def configure_sharded_params_and_buffers(self) -> list[str]:
        ret = super().configure_sharded_params_and_buffers()
        BaseModel.mark_as_sharded(self.features_albedo)
        BaseModel.mark_as_sharded(self.features_specular)
        return ret + ["features_albedo", "features_specular"]

    def initialize_gaussians_from_datasource(
        self,
        initializer: BaseInitialization,
        datasource: DataSourceSummary,
        cuboid_tracks: Optional[CuboidTracks] = None,
    ) -> None:
        super().initialize_gaussians_from_datasource(
            initializer=initializer,
            datasource=datasource,
            cuboid_tracks=cuboid_tracks,
        )

        assert initializer is not None, (
            f"{self.__class__.__name__}: initializer must be provided when init_from_datasource is True"
        )

        world_rank, world_size = self._get_slice_boundaries()

        if self.fourier_features_dim > 1:
            features_albedo = torch.zeros(self.get_num_gaussians(), self.fourier_features_dim, 3, device=self.device)
            features_albedo[:, 0, :] = initializer.features_albedo[world_rank::world_size]
            self.features_albedo = nn.Parameter(features_albedo.contiguous())
        else:
            self.features_albedo = nn.Parameter(initializer.features_albedo[world_rank::world_size].contiguous())

        self.features_specular = nn.Parameter(initializer.features_specular[world_rank::world_size].contiguous())

    def get_albedo_sh_dim(self) -> int:
        return 3

    def get_specular_sh_dim(self) -> int:
        return sh_degree_to_specular_dim(self.config.progressive_training.max_n_features)

    def _validate_fields(self) -> None:
        """Validate the base parameters"""
        super()._validate_fields()
        num_gaussians = self.get_num_gaussians()

        albedo_sh_dim = self.get_albedo_sh_dim()
        if self.fourier_features_dim > 1:
            assert self.features_albedo.shape == (num_gaussians, self.fourier_features_dim, albedo_sh_dim)
        else:
            assert self.features_albedo.shape == (num_gaussians, albedo_sh_dim)
        specular_sh_dims = self.get_specular_sh_dim()
        assert self.features_specular.shape == (num_gaussians, specular_sh_dims)

    #
    # Implementation of what is necessary for parameter collection.
    #
    def get_parameters(self) -> dict[str, torch.Tensor]:
        return {
            **super().get_parameters(),
            "features_albedo": self.features_albedo,
            "features_specular": self.features_specular,
        }

    def get_layer_config(self) -> collect.LayerConfigSH:
        embed_config: collect.EmbeddingConfig | None = None
        fourier_features_dim = unpack_optional(self.config.fourier_features_dim)
        if fourier_features_dim > 1:
            if type(self.time_embed) is IndividualRemapTimeInputEmbedding:
                embed_config = collect.IndividualRemapTimeInputEmbeddingConfig(
                    timestamps_us_ranges=self.time_embed.timestamps_us_ranges,
                    remap_min=self.time_embed.remap_min,
                    remap_max=self.time_embed.remap_max,
                )
            elif type(self.time_embed) is HolisticRemapTimeInputEmbedding:
                if type(self.time_embed.timestamps_us_min) is not int:
                    raise ValueError(
                        f"timestamps_us_min must be an integer, got {type(self.time_embed.timestamps_us_min)}"
                    )
                if type(self.time_embed.timestamps_us_max) is not int:
                    raise ValueError(
                        f"timestamps_us_max must be an integer, got {type(self.time_embed.timestamps_us_max)}"
                    )
                embed_config = collect.HolisticRemapTimeInputEmbeddingConfig(
                    timestamps_us_min=self.time_embed.timestamps_us_min,
                    timestamps_us_max=self.time_embed.timestamps_us_max,
                    remap_min=self.time_embed.remap_min,
                    remap_max=self.time_embed.remap_max,
                )
            elif type(self.time_embed) is IndividualStepTimeInputEmbedding:
                embed_config = collect.IndividualStepTimeInputEmbeddingConfig(
                    timestamps_us_ranges=self.time_embed.timestamps_us_ranges,
                    n_steps=self.time_embed.n_steps,
                    n_dims=self.time_embed.n_dims,
                )
            else:
                raise NotImplementedError(f"Invalid time embed: {type(self.time_embed)}")

        layer_config = super().get_layer_config()
        return collect.LayerConfigSH(
            rotation_activation=layer_config.rotation_activation,
            scale_activation=layer_config.scale_activation,
            density_activation=layer_config.density_activation,
            fourier_features_dim=fourier_features_dim,
            embed_config=embed_config,
        )

    def get_layer_data(
        self, context: BaseGaussianModel.CollectionContext, gathered_parameters: dict[str, torch.Tensor]
    ) -> collect.LayerDataSH:
        embed_data: collect.EmbeddingData | None = None
        if unpack_optional(self.config.fourier_features_dim) > 1:
            if type(self.time_embed) is IndividualRemapTimeInputEmbedding:
                embed_data = collect.IndividualRemapTimeInputEmbeddingData(
                    instance_idx=gathered_parameters["gaussian_cuboid_ids"],
                )
            elif type(self.time_embed) is HolisticRemapTimeInputEmbedding:
                embed_data = None
            elif type(self.time_embed) is IndividualStepTimeInputEmbedding:
                embed_data = collect.IndividualStepTimeInputEmbeddingData(
                    instance_idx=gathered_parameters["gaussian_cuboid_ids"],
                    u=self.time_embed.u,
                    beta=self.time_embed.beta,
                )
            else:
                raise NotImplementedError(f"Invalid time embed: {type(self.time_embed)}")
        else:
            embed_data = None

        layer_data = super().get_layer_data(context, gathered_parameters)
        return collect.LayerDataSH(
            positions=layer_data.positions,
            rotations=layer_data.rotations,
            scales=layer_data.scales,
            densities=layer_data.densities,
            extra_signal=layer_data.extra_signal,
            camera_extra_signal=layer_data.camera_extra_signal,
            lidar_extra_signal=layer_data.lidar_extra_signal,
            features_albedo=gathered_parameters["features_albedo"],
            features_specular=gathered_parameters["features_specular"],
            embed_data=embed_data,
        )

    # This is a copy from the model's collector, it can be used to access certain
    # "global" functions.
    collector: collect.GaussianParameterCollector

    #
    # End of implementation of what is necessary for parameter collection.
    #

    def update_step_train_batch_start(self, epoch: int, global_step: int, system, **kwargs) -> dict[str, torch.Tensor]:
        self._global_step_for_freeze = global_step
        additional_parameters: dict[str, torch.Tensor] = {}
        if self.time_embed is not None:
            additional_parameters |= update_module_step(self.time_embed, epoch, global_step, system)
        return additional_parameters

    def update_step_train_batch_end(
        self, epoch: int, global_step: int, batch: DataAndRenderingBatch, system, **kwargs
    ) -> None:
        # Every N its we increase the levels of SH up to a maximum degree
        if global_step > 0 and global_step % self.config.progressive_training.increase_frequency == 0:
            self.increase_number_of_active_features()

        self.log_extra_parameters(system, epoch, global_step)

    @ScopedTimer("SHGaussianModel/log_extra_parameters")
    def log_extra_parameters(self, system, epoch: int, global_step: int) -> None:
        system.log("n_active_features", self.n_active_features)

        for optimizer in self.optimizers:
            for param_group in optimizer["optimizer"].param_groups:
                if "name" in param_group:
                    system.log(f"lr/{param_group['name']}", param_group["lr"])

    def configure_optimizers(self, name_prefix: str = "") -> list[OptimizerLRSchedulerConfig]:
        """Returns a list of module-owned configured optimizers (optimizers paired with an optional LR scheduler),
        which will be stepped in the main training loop, allowing the module to interact with it's owned optimizers"""

        optimizers = super().configure_optimizers(name_prefix=name_prefix)
        if self.time_embed is not None:
            optimizers += configure_optimizers(
                self.config.time_embed, self.trainer_config, self.time_embed, name_prefix
            )

        return optimizers

    @ScopedTimer("SHGaussianModel/increase_number_of_active_features")
    def increase_number_of_active_features(self) -> None:
        """Increase the number of optimizable featues (features_specular)"""
        self.n_active_features = min(
            self.config.progressive_training.max_n_features,
            self.n_active_features + self.config.progressive_training.increase_step,
        )

    def get_extra_state(self) -> dict[str, Any] | None:
        return {"n_active_features": self.n_active_features}

    def set_extra_state(self, state: dict[str, Any] | None) -> None:
        if state is not None and "n_active_features" in state:
            self.n_active_features = int(state["n_active_features"])

    def remove_gaussians_in_trajectories(
        self, trajectories: List[torch.Tensor], cuboid_dims: List[torch.Tensor]
    ) -> torch.Tensor:
        include_mask = super().remove_gaussians_in_trajectories(trajectories, cuboid_dims)
        self.features_albedo = nn.Parameter(self.features_albedo[include_mask])
        self.features_specular = nn.Parameter(self.features_specular[include_mask])
        return include_mask

    def _permute_gaussian_rows(self, order: torch.Tensor) -> None:
        super()._permute_gaussian_rows(order)
        self.features_albedo = self._reordered_rows(self.features_albedo, order)
        self.features_specular = self._reordered_rows(self.features_specular, order)

    def save_training_parameters(self):
        first_save = len(self.overriden_parameters) == 0
        super().save_training_parameters()
        if first_save:
            self.overriden_parameters["features_albedo"] = self.features_albedo
            self.overriden_parameters["features_specular"] = self.features_specular

    def restore_training_parameters(self) -> None:
        if len(self.overriden_parameters) > 0:
            self.features_albedo = self.overriden_parameters["features_albedo"]
            self.features_specular = self.overriden_parameters["features_specular"]
            super().restore_training_parameters()

    def export_ply(
        self,
        export_dir: Path,
        format: GaussianExportFormat = GaussianExportFormat._3DGS,
        percentage_gaussians: float = 100,
    ) -> None:
        """
        Export Gaussian model as PLY files in the specified format.

        For 3DGS format, it should be compatible with the original 3DGS implementation but differences
        between 3DGS/3DGUT/3DGRT rendering will cause slight differences when rendered with
        3rd-party 3DGS viewers.
        NB : extra_signal is not exported.

        Args:
            export_dir: Directory path where the PLY file will be saved as 'model.ply'.
            format: Export format for the Gaussian data. Available options:
                - GaussianExportFormat._3DGS: Original 3D Gaussian Splatting format
                - GaussianExportFormat._3DGRT: 3DGRT format for visualization
            percentage_gaussians: Percentage of Gaussians to export (0, 100]. Only used
                for _3DGRT format. Defaults to 100 (export all Gaussians).
        """
        if format == GaussianExportFormat._3DGS:
            write_ply_3dgs(
                export_dir / "model.ply",
                self.positions,
                self.rotations,
                self.scales,
                self.densities,
                self.features_albedo,
                self.features_specular,
            )
        elif format == GaussianExportFormat._3DGRT:
            write_ply_3dgrt(
                export_dir / "model.ply",
                self.get_positions(),
                self.get_rotations(quaternion_format="xyzw"),
                self.get_scales(),
                self.get_densities(),
                percentage_gaussians=percentage_gaussians,
            )


class RigidGaussianModel(SHGaussianModel):
    cuboid_tracks: CuboidTracks
    tracks_calib: BaseTracksCalib

    gaussian_cuboid_ids: nn.Buffer | nn.UninitializedBuffer
    _split_boundary_track_indices: torch.Tensor
    _split_internal_start_timestamps_us: torch.Tensor
    _split_internal_end_timestamps_us: torch.Tensor
    global_step: int

    config: RigidGaussiansLayerConfig  # type: ignore[assignment] # Narrow type from SHGaussiansLayerConfig

    @staticmethod
    def _get_config_value(config: Any, key: str, default: Any = None) -> Any:
        if config is None:
            return default
        if hasattr(config, "get"):
            return config.get(key, default)
        return getattr(config, key, default)

    @classmethod
    def _maybe_coalesce_short_timestamp_gaps(
        cls,
        config: RigidGaussiansLayerConfig,
        cuboid_tracks: CuboidTracks,
    ) -> CuboidTracks:
        short_gap_fusion = cls._get_config_value(config.tracks_calib, "short_gap_fusion", None)
        if not cls._get_config_value(short_gap_fusion, "enabled", False):
            return cuboid_tracks

        min_gap_us = int(cls._get_config_value(short_gap_fusion, "min_gap_us", 10000))
        return CuboidTracks.Ops.coalesce_short_timestamp_gaps(cuboid_tracks, min_gap_us)

    @staticmethod
    def _build_split_boundary_policy(
        cuboid_tracks: CuboidTracks,
        metadata: TrackSplitMetadata | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build sparse, node-local bounds for synthetic split boundaries.

        Adjacent split children keep their shared endpoint pose, but logically own
        right-biased half-open intervals. The first/last child remain unbounded on
        the original parent's outer sides so normal endpoint tolerance is preserved.
        """
        device = cuboid_tracks.device
        if metadata is None or metadata.is_empty:
            empty = torch.empty((0,), dtype=torch.int64, device=device)
            return empty, empty.clone(), empty.clone()

        children_by_parent: dict[str, list[TrackSplitChild]] = {}
        for child in metadata.children:
            children_by_parent.setdefault(child.parent_track_id, []).append(child)

        local_track_indices = {track_id: track_idx for track_idx, track_id in enumerate(cuboid_tracks.tracks_id)}
        boundary_track_indices: list[int] = []
        internal_start_timestamps_us: list[int] = []
        internal_end_timestamps_us: list[int] = []
        int64_info = torch.iinfo(torch.int64)

        for children in children_by_parent.values():
            children.sort(key=lambda child: (child.start_timestamp_us, child.end_timestamp_us, child.child_index))
            for child_position, child in enumerate(children):
                track_idx = local_track_indices.get(child.child_track_id)
                if track_idx is None:
                    continue
                internal_start_us = int64_info.min if child_position == 0 else int(child.start_timestamp_us)
                internal_end_us = int64_info.max if child_position == len(children) - 1 else int(child.end_timestamp_us)
                if internal_start_us == int64_info.min and internal_end_us == int64_info.max:
                    continue
                boundary_track_indices.append(track_idx)
                internal_start_timestamps_us.append(internal_start_us)
                internal_end_timestamps_us.append(internal_end_us)

        return (
            torch.tensor(boundary_track_indices, dtype=torch.int64, device=device),
            torch.tensor(internal_start_timestamps_us, dtype=torch.int64, device=device),
            torch.tensor(internal_end_timestamps_us, dtype=torch.int64, device=device),
        )

    def __init__(
        self,
        config: RigidGaussiansLayerConfig,
        trainer_config: TrainerConfig,
        datasource: DataSourceSummary,
        init_from_datasource: bool,
        initializer: Optional[BaseInitialization],
        precision: Optional[int] = None,
        cuboid_tracks: Optional[CuboidTracks] = None,
        start_end_timestamp_us: Optional[Tuple[int, int]] = None,
    ) -> None:
        assert cuboid_tracks is not None, "cuboid_tracks must be provided for DynamicGaussians"
        cuboid_tracks = self._maybe_coalesce_short_timestamp_gaps(config, cuboid_tracks)
        self.cuboid_tracks = cuboid_tracks
        self._motion_source_ref: weakref.ReferenceType[RigidGaussianModel] | None = None
        self.global_step = 0

        super().__init__(
            config=config,
            trainer_config=trainer_config,
            datasource=datasource,
            init_from_datasource=init_from_datasource,
            initializer=initializer,
            precision=precision,
            cuboid_tracks=cuboid_tracks,
            start_end_timestamp_us=start_end_timestamp_us,
        )

        split_boundary_policy = self._build_split_boundary_policy(
            cuboid_tracks,
            datasource.get_track_split_metadata(),
        )
        self.register_buffer(
            "_split_boundary_track_indices",
            split_boundary_policy[0],
            persistent=False,
        )
        self.register_buffer(
            "_split_internal_start_timestamps_us",
            split_boundary_policy[1],
            persistent=False,
        )
        self.register_buffer(
            "_split_internal_end_timestamps_us",
            split_boundary_policy[2],
            persistent=False,
        )

        self.tracks_calib = BaseTracksCalib.factory(
            config.tracks_calib.name, config.tracks_calib, trainer_config, cuboid_tracks
        )

        self.track_albedos = (
            nn.Parameter(
                torch.eye(4, device=self.device)[:3].unsqueeze(0).repeat(unpack_optional(cuboid_tracks).n_tracks, 1, 1)
            )
            if self.config.optimize_track_albedo
            else None
        )

        self.track_scales = (
            nn.Parameter(
                self.scale_activation_inv(torch.ones(unpack_optional(cuboid_tracks).n_tracks, 3, device=self.device))
            )
            if self.config.optimize_track_scale
            else None
        )

        # Config flag for nearest neighbor track selection (LiDAR only)
        self.nearest_neighbor_track_for_lidar = self.config.nearest_neighbor_track_for_lidar

        # Inference-time subsampling of the per-track timestamp-estimation ray bundle.
        self.tracks_timestamp_estimation_ray_stride = self.config.tracks_timestamp_estimation_ray_stride
        assert self.tracks_timestamp_estimation_ray_stride >= 1, "ray stride must be >= 1"

        # We need to wait for the datasource and torch.distributed context to be initialized. We will re-initializing
        # particles from datasource and validate the model's parameters later
        self.gaussian_cuboid_ids = nn.UninitializedBuffer()

        # Config to specify if the Gaussians are static.
        self.is_static = self.config.is_static

    @property
    def owns_tracks(self) -> bool:
        """Whether this layer owns and publishes its rigid-track calibration."""

        return getattr(getattr(self, "config", None), "motion_source_layer", None) is None

    def bind_motion_source(self, source: RigidGaussianModel) -> None:
        """Use an owner's motion while retaining this layer's appearance-frame metadata.

        IDs and order must match at binding. Subsequent owner edits may append
        tracks, but must not remove or reorder the dependent's original tracks.
        Static and LiDAR interpolation modes must match at binding. The owner
        supplies the shared timestamp stride, including serve-time overrides;
        this layer's local stride may differ and does not affect shared motion.
        Pose samples and timestamp packing may differ: the owner can coalesce its
        motion timeline while this layer retains its original appearance frames.
        With is_static=True, the pose data passed to interpolation after calibration
        or actor edits must contain two consecutive samples per owner track, in
        owner-track order.
        """

        if source is self:
            raise ValueError("A rigid Gaussian layer cannot be its own motion source")
        if self.owns_tracks:
            raise ValueError("A motion-dependent rigid layer must configure motion_source_layer")
        if not source.owns_tracks:
            raise ValueError("A rigid motion source must own its tracks")
        if self.cuboid_tracks.tracks_id != source.cuboid_tracks.tracks_id:
            raise ValueError(
                "Motion-source tracks must match target tracks in id and order: "
                f"target={self.cuboid_tracks.tracks_id}, source={source.cuboid_tracks.tracks_id}"
            )
        if self.is_static != source.is_static:
            raise ValueError("Motion-source and target layers must use the same static interpolation mode")
        if self.nearest_neighbor_track_for_lidar != source.nearest_neighbor_track_for_lidar:
            raise ValueError("Motion-source and target layers must use the same LiDAR track interpolation mode")
        self._motion_source_ref = weakref.ref(source)

    def get_motion_source(self) -> RigidGaussianModel:
        """Return the live owner of the tracks, calibration and motion-sampling policy."""

        motion_source_ref = self._motion_source_ref
        if motion_source_ref is None:
            if not self.owns_tracks:
                raise RuntimeError("motion_source_layer is configured but bind_motion_source was not called")
            return self
        source = motion_source_ref()
        if source is None:
            raise RuntimeError("The configured rigid motion-source layer is no longer available")
        return source

    def get_motion_tracks_calib(self) -> BaseTracksCalib:
        """Return the sole calibration provider used for this layer's rigid motion."""

        return self.get_motion_source().tracks_calib

    def get_calibrated_cuboid_tracks(self) -> CuboidTracks:
        source = self.get_motion_source()
        return source.tracks_calib(source.cuboid_tracks)

    def create_tracks_edit_from_cuboid_tracks(self, edited_cuboid_tracks: CuboidTracks) -> LayerTrackIds.Edit:
        # Edits use the owner's current track indices, including inserted actors.
        # A light layer may have no Gaussians for an inserted track; its Gaussian
        # rows are filtered later without consulting its appearance-frame table.
        motion_tracks = self.get_motion_source().cuboid_tracks
        assert motion_tracks.tracks_id is not None, "track_ids should be initialized before enabling editing"
        owner_index_by_id: dict[str, int] = {}
        for index, track_id in enumerate(motion_tracks.tracks_id):
            owner_index_by_id.setdefault(track_id, index)
        existing_cuboid_tracks = CuboidTracks.Ops.subset_from_mask(
            edited_cuboid_tracks,
            torch.tensor([t in owner_index_by_id for t in edited_cuboid_tracks.tracks_id], dtype=torch.bool),
        )

        # Preserve edited-track order while mapping each entry to the owner's
        # original index, which both body and light Gaussian IDs still address.
        edited_mapping = torch.tensor(
            [owner_index_by_id[t] for t in existing_cuboid_tracks.tracks_id],
            dtype=torch.long,
            device=existing_cuboid_tracks.device,
        )

        return LayerTrackIds.Edit(cuboid_tracks=existing_cuboid_tracks, mapping=edited_mapping)

    def configure_sharded_params_and_buffers(self) -> list[str]:
        ret = super().configure_sharded_params_and_buffers()
        BaseModel.mark_as_sharded(self.gaussian_cuboid_ids)
        return ret + ["gaussian_cuboid_ids"]

    def initialize_gaussians_from_datasource(
        self,
        initializer: BaseInitialization,
        datasource: DataSourceSummary,
        cuboid_tracks: Optional[CuboidTracks] = None,
    ) -> None:
        super().initialize_gaussians_from_datasource(
            initializer=initializer,
            datasource=datasource,
            cuboid_tracks=cuboid_tracks,
        )

        assert initializer is not None, (
            f"{self.__class__.__name__}: initializer must be provided when init_from_datasource is True"
        )

        world_rank, world_size = self._get_slice_boundaries()

        self.gaussian_cuboid_ids = nn.Buffer(
            torch.vstack(initializer.gaussian_cuboid_ids).data.squeeze(-1)[world_rank::world_size].contiguous()
        )

    def _validate_fields(self) -> None:
        """Validate the base parameters"""
        super()._validate_fields()
        num_gaussians = self.get_num_gaussians()

        assert self.gaussian_cuboid_ids.shape == (num_gaussians,)

    def get_additional_buffers(self) -> dict[str, torch.Tensor]:
        additional_buffers = super().get_additional_buffers()
        additional_buffers |= {
            "gaussian_cuboid_ids": self.gaussian_cuboid_ids,
        }
        return additional_buffers

    def freeze_tracks(self, frozen_track_indices: Sequence[int]) -> None:
        """Stop gradients for the given tracks' parameters on this node.

        Freezes what this node owns and cannot be reached by the composite's
        per-Gaussian row hook: the per-track direct params
        (``track_albedos``/``track_scales``, indexed by track) and the
        calibration params (delegated to ``tracks_calib.freeze_tracks``,
        which handles its own pose/track indexing). ``frozen_track_indices``
        are indices into this node's track subset.
        """
        frozen = [int(t) for t in frozen_track_indices]
        if not frozen or self.cuboid_tracks is None:
            return
        packinfo = self.cuboid_tracks.tracks_packinfo
        n_tracks = int(packinfo.shape[0])
        track_trainable = torch.ones(n_tracks, dtype=torch.bool)
        track_trainable[torch.tensor(frozen, dtype=torch.long)] = False
        for param in (self.track_albedos, self.track_scales):
            if param is not None and param.requires_grad:
                param.register_hook(make_row_grad_hook(track_trainable))
        self.tracks_calib.freeze_tracks(frozen, packinfo)

    def remove_gaussians_in_trajectories(
        self, trajectories: List[torch.Tensor], cuboid_dims: List[torch.Tensor]
    ) -> torch.Tensor:
        include_mask = super().remove_gaussians_in_trajectories(trajectories, cuboid_dims)
        self.gaussian_cuboid_ids = nn.Buffer(self.gaussian_cuboid_ids[include_mask])
        return include_mask

    @torch.no_grad()
    def compute_morton_order(self, voxel_size: float = VOXEL_SIZE) -> torch.Tensor | None:
        """Return an order that partitions the rows by track, Morton-ordered inside each track.

        Rigid positions are stored in each cuboid's local frame, so a plain Morton key mixes the tracks
        together. Whether an object is on screen at all is the dominant early exit in projection, and that is
        per track. So track is the outer key, with Morton inside a track still giving world-space locality.
        Sorting by the inner key then stable-sorting by the outer leaves the inner order intact inside each
        group.

        This is deliberately not the order the compression codec wants. That codec groups equal Morton codes
        with unique_consecutive. A (track, morton) order makes the same voxel reappear once per track, which breaks
        that grouping. Do not wire this into the codec.

        Returns None if there is nothing to sort.

        Raises:
            ValueError: If positions are non-finite, exceed the Morton range, or the track buffer is not
                row-aligned with them.
        """
        positions = self.positions
        track_ids = self.gaussian_cuboid_ids
        if isinstance(positions, torch.nn.parameter.UninitializedParameter) or positions.shape[0] == 0:
            return None
        if isinstance(track_ids, torch.nn.parameter.UninitializedBuffer):
            return None
        # Indexing a longer track buffer with N indices is legal and would yield a plausible order, then
        # silently truncate the buffer against the wrong tracks. Refuse both lengths here instead.
        if track_ids.shape[:1] != positions.shape[:1]:
            raise ValueError(
                f"gaussian_cuboid_ids has {track_ids.shape[0]} rows, expected {positions.shape[0]} "
                "to match the Gaussians"
            )

        morton = morton_sort_order(positions, voxel_size, label=f"{type(self).__name__}.positions").order
        return morton[track_ids[morton].argsort(stable=True)]

    #
    # Implementation of what is necessary for parameter collection.
    #
    def get_parameters(self) -> dict[str, torch.Tensor]:
        return {
            **super().get_parameters(),
            "gaussian_cuboid_ids": self.gaussian_cuboid_ids,
        }

    def get_layer_config(self) -> collect.LayerConfigRigid:
        layer_config = super().get_layer_config()
        assert isinstance(layer_config, collect.LayerConfigSH)
        return collect.LayerConfigRigid(
            rotation_activation=layer_config.rotation_activation,
            scale_activation=layer_config.scale_activation,
            density_activation=layer_config.density_activation,
            fourier_features_dim=layer_config.fourier_features_dim,
            embed_config=layer_config.embed_config,
        )

    def get_layer_data(
        self, context: BaseGaussianModel.CollectionContext, gathered_parameters: dict[str, torch.Tensor]
    ) -> collect.LayerDataRigid:
        context.kept_rows = None
        if context.tracks_edit is not None and context.tracks_edit.mapping.shape[0] > 0:
            # Both layers use the same edited track motion, but only keep their
            # own Gaussians for the selected tracks. An inserted owner track may
            # have body Gaussians and no light Gaussians.
            self._filter_edited_gaussians(context, gathered_parameters)
        tracks_ids = gathered_parameters["gaussian_cuboid_ids"]
        prepared_motion = context.prepared_motion
        motion = (
            prepared_motion
            if prepared_motion is not None
            else self.prepare_motion(
                context.rendering_data,
                context.frustum_cull,
                context.is_training_batch,
                tracks_edit=context.tracks_edit,
            )
        )

        layer_data = super().get_layer_data(context, gathered_parameters)
        assert isinstance(layer_data, collect.LayerDataSH)

        # Apply track scale.  Note that this could be done in the merged kernel,
        # but it's not enabled by default, so we might optimize it later.
        if self.track_scales is not None:
            assert type(self.scale_activation) is ExpActivation
            track_scales = self.track_scales[tracks_ids]
            activated_track_scales = self.scale_activation(track_scales)
            layer_data.positions = layer_data.positions * activated_track_scales
            # scales are not activated here, we can just add them
            # exp(s1) * exp(s2) = exp(s1 + s2)
            layer_data.scales = layer_data.scales + track_scales

        context.additional_data["interpolated_mask"] = (motion.keep_mask, False)
        context.additional_data["interpolated_track_timestamps"] = (motion.timestamps_us, False)
        if context.build_segment_plan:
            context.track_time_data = motion.track_time_data

        return collect.LayerDataRigid(
            positions=layer_data.positions,
            rotations=layer_data.rotations,
            scales=layer_data.scales,
            densities=layer_data.densities,
            extra_signal=layer_data.extra_signal,
            camera_extra_signal=layer_data.camera_extra_signal,
            lidar_extra_signal=layer_data.lidar_extra_signal,
            features_albedo=layer_data.features_albedo,
            features_specular=layer_data.features_specular,
            embed_data=layer_data.embed_data,
            poses=motion.poses,
            keep_mask=motion.keep_mask,
            tracks_ids=tracks_ids,
        )

    def prepare_motion(
        self,
        rendering_data: RenderingData,
        frustum_cull: Optional[collect.TracksFrustumCullData],
        is_training_batch: bool,
        tracks_edit: LayerTrackIds.Edit | None = None,
    ) -> PreparedRigidMotion:
        """Prepare fresh motion using the owner's tracks, backend and interpolation policy.

        The composite passes this result to every dependent for one sensor
        collection, including edits. Supplied edited poses already describe the
        requested motion and must not receive learned calibration again. An empty
        edit mapping retains the existing unedited behavior. Standalone collection
        prepares a fresh result in the caller's grad/inference context.
        """
        source = self.get_motion_source()
        motion_data = (
            source._get_edited_motion_data(tracks_edit)
            if tracks_edit is not None and tracks_edit.mapping.shape[0] > 0
            else source._get_calibrated_motion_data()
        )
        poses, keep_mask, timestamps_us = source._get_interpolated_poses(
            rendering_data,
            frustum_cull,
            *motion_data,
            is_training_batch,
        )
        track_time_data = None
        if not source.is_static:
            track_time_data = collect.TrackTimeData(
                tracks_timestamps_us=motion_data[1],
                tracks_packinfo=motion_data[2],
                max_positive_timestamp_gap_us=motion_data[4],
            )
        return PreparedRigidMotion(
            poses=poses,
            keep_mask=keep_mask,
            timestamps_us=timestamps_us,
            track_time_data=track_time_data,
        )

    def _get_calibrated_motion_data(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calibrate this owner's motion without reading or filtering Gaussian rows."""
        cuboid_tracks = self.cuboid_tracks
        tracks_calib = self.tracks_calib
        if type(tracks_calib) is DirectTracksCalib:
            direct_tracks_calib_data = collect.DirectTracksCalibData(
                gradient_mask=tracks_calib.gradient_mask,
                tracks_delta_q=tracks_calib.tracks_delta_q,
                tracks_delta_t=tracks_calib.tracks_delta_t,
                left_multiply=tracks_calib.left_multiply,
            )
            tracks_poses = self.collector.calibrate_tracks_poses(
                cuboid_tracks.tracks_poses.vec(), direct_tracks_calib_data
            )
        elif type(tracks_calib) is ElevationAwareUnicycleTracksCalib:
            unicycle_tracks_calib_data = tracks_calib.collect_data()
            tracks_poses = tracks_calib._get_pose_from_collector(
                lambda: self.collector.calibrate_tracks_poses(
                    cuboid_tracks.tracks_poses.vec(), unicycle_tracks_calib_data
                )
            ).vec()
        else:
            cuboid_tracks = tracks_calib(cuboid_tracks)
            tracks_poses = cuboid_tracks.tracks_poses.vec()
        return (
            tracks_poses,
            cuboid_tracks.tracks_timestamps_us,
            cuboid_tracks.tracks_packinfo,
            cuboid_tracks.cuboids_dims,
            cuboid_tracks.max_positive_timestamp_gap_us,
        )

    def _get_edited_motion_data(
        self, tracks_edit: LayerTrackIds.Edit
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare owner-indexed metadata without filtering any layer's Gaussians."""
        motion_tracks = self.get_motion_source().cuboid_tracks
        # Gaussian IDs continue to address the owner's original track indices,
        # including tracks appended by insertion. The edit may select/reorder
        # those tracks, so scatter its metadata into the full owner table while
        # leaving poses and timestamps packed in edited-track order.
        new_pack_info = torch.zeros_like(motion_tracks.tracks_packinfo)
        new_pack_info[tracks_edit.mapping] = tracks_edit.cuboid_tracks.tracks_packinfo
        new_cuboid_dims = torch.zeros_like(motion_tracks.cuboids_dims)
        new_cuboid_dims[tracks_edit.mapping] = tracks_edit.cuboid_tracks.cuboids_dims
        new_max_positive_timestamp_gap_us = torch.zeros_like(motion_tracks.max_positive_timestamp_gap_us)
        new_max_positive_timestamp_gap_us[tracks_edit.mapping] = tracks_edit.cuboid_tracks.max_positive_timestamp_gap_us
        return (
            tracks_edit.cuboid_tracks.tracks_poses.vec(),
            tracks_edit.cuboid_tracks.tracks_timestamps_us,
            new_pack_info,
            new_cuboid_dims,
            new_max_positive_timestamp_gap_us,
        )

    def _filter_edited_gaussians(
        self, context: BaseGaussianModel.CollectionContext, gaussian_nodes_parameters: dict[str, torch.Tensor]
    ) -> None:
        """Keep this layer's selected Gaussian rows independently of shared motion."""
        tracks_edit = unpack_optional(context.tracks_edit)
        with ScopedTimer("RigidGaussianModel/edit_mapping"):
            gaussian_cuboid_ids = gaussian_nodes_parameters["gaussian_cuboid_ids"]
            tracks_keep_mask = torch.zeros(
                (self.get_motion_source().cuboid_tracks.n_tracks,),
                dtype=torch.bool,
                device=gaussian_cuboid_ids.device,
            )
            tracks_keep_mask[tracks_edit.mapping] = True
            keep_mask = tracks_keep_mask[gaussian_cuboid_ids]
            all_indices = torch.arange(gaussian_cuboid_ids.shape[0], device=gaussian_cuboid_ids.device)
            # Boolean row filtering synchronizes CPU and GPU. Retaining all rows
            # with zero opacity could avoid it, but would change collection layout.
            indices_to_keep = all_indices[keep_mask]
            gaussian_nodes_parameters.update({k: v[indices_to_keep] for k, v in gaussian_nodes_parameters.items()})
            # Consumers such as baked appearance edits need the original row of
            # each surviving Gaussian. Body and lights have different row tables.
            context.kept_rows = indices_to_keep

    def get_tracks_calib_residuals(self) -> collect.TracksCalibResiduals:
        """Compute and cache this node's unicycle residuals on the collector backend.

        Returns:
            The named consistency, smoothness, and pose-regularization residuals.
        """
        assert type(self.tracks_calib) is ElevationAwareUnicycleTracksCalib
        tracks_calib_data = self.tracks_calib.collect_data()

        def compute() -> collect.TracksCalibResiduals:
            return self.collector.compute_tracks_calib_residuals(tracks_calib_data)

        return self.tracks_calib._get_residuals_from_collector(compute)

    def _get_interpolated_poses(
        self,
        rendering_data: RenderingData,
        frustum_cull: Optional[collect.TracksFrustumCullData],
        tracks_poses: torch.Tensor,
        tracks_timestamps: torch.Tensor,
        tracks_packinfo: torch.Tensor,
        cuboid_dims: torch.Tensor,
        max_positive_timestamp_gap_us: torch.Tensor,
        is_training_batch: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_tracks = tracks_packinfo.shape[0]

        # Static interpolation takes the first of two consecutive samples per
        # owner track. Calibration or actor edits must preserve that track order.
        if self.is_static:
            expected_pose_count = 2 * n_tracks
            if tracks_poses.shape[0] != expected_pose_count:
                raise ValueError(
                    "Static interpolation requires two pose samples per owner track: "
                    f"expected {expected_pose_count} poses for {n_tracks} tracks, got {tracks_poses.shape[0]}"
                )
            timestamps = torch.full(
                (n_tracks,),
                self.get_frame_timestamp(rendering_data),
                dtype=torch.int64,
                device=tracks_poses.device,
            )
            return tracks_poses[::2], torch.ones(n_tracks, dtype=torch.bool, device=tracks_poses.device), timestamps

        # Detect if this is a LiDAR sensor (for nearest neighbor selection).
        is_lidar = False
        if rendering_data.sensor_model_parameters is not None and len(rendering_data.sensor_model_parameters) > 0:
            sensor_model = rendering_data.sensor_model_parameters[0]
            # Check for the Parameters type (runtime data).
            is_lidar = isinstance(
                sensor_model,
                (RowOffsetStructuredSpinningLidarModel, RowOffsetStructuredSpinningLidarModelParameters),
            )

        frame_start_us, frame_end_us = BaseGaussianModel.get_frame_window(rendering_data)
        frame_exposure_us = frame_end_us - frame_start_us

        ## Raycasting based timestamp estimation
        # Get accurate timestamps for Camera and LiDAR, if rays timestamps are available
        with ScopedTimer("RigidGaussianModel/get_tracks_timestamps_estimation_data"):
            tracks_timestamps_data: collect.TracksTimestampsData
            default_timestamp = self.get_frame_timestamp(rendering_data)
            if rendering_data.rays_timestamps_us is not None and n_tracks > 0:
                # Inference-only: subsample the estimation ray bundle. Gated on `not
                # self.training` so training always uses full estimation even if a config
                # sets the stride; the knob is meant for serve/render only.
                stride = self.tracks_timestamp_estimation_ray_stride
                if stride > 1 and not self.training:
                    # Subsample the 2D sensor grid, not the flattened ray list. A flat
                    # rays[::N] aliases whenever N shares a factor with the row width (every row
                    # then samples the same columns), so a track whose footprint falls between
                    # sampled columns gets zero rays and silently loses its per-track
                    # rolling-shutter timestamp (falls back to frame mid-time) — invisible in
                    # frame-level PSNR. We decimate rows and columns each by ~sqrt(N): this
                    # minimizes the worst-case per-axis stride, which measured best for keeping
                    # small tracks (they are small in BOTH axes — few beams AND few azimuth on
                    # lidar; an aspect-weighted split that decimated one axis harder dropped
                    # ~4x more lidar tracks). Keeps ~1/N rays so kernel cost drops ~N-fold. rays
                    # is (B, H, W, 6), rays_timestamps_us is (B, H, W, 1); both carry the 2D grid
                    # for camera and lidar. frustum_cull is untouched — it indexes the
                    # full-resolution FOV-corner rays.
                    row_stride = math.isqrt(stride)
                    col_stride = -(-stride // row_stride)  # ceil: row_stride * col_stride >= stride
                    estimation_rays = rendering_data.rays[:, ::row_stride, ::col_stride, :]
                    rays = estimation_rays.reshape(-1, 6)
                    rays_timestamps = rendering_data.rays_timestamps_us[:, ::row_stride, ::col_stride, :].reshape(-1)
                else:
                    estimation_rays = rendering_data.rays
                    rays = estimation_rays.view(-1, 6)
                    rays_timestamps = rendering_data.rays_timestamps_us.view(-1)
                tracks_timestamps_data = collect.TracksTimestampsEstimationData(
                    rays=rays,
                    rays_timestamps=rays_timestamps,
                    cuboids_dimensions=cuboid_dims,
                    default_timestamp=default_timestamp,
                    rays_timestamp_layout=rendering_data.rays_timestamp_layout
                    if self.config.tracks_timestamp_interpolation_cache
                    else None,
                    rays_grid_shape=(estimation_rays.shape[0], estimation_rays.shape[1], estimation_rays.shape[2]),
                    frame_start_timestamp=frame_start_us,
                    frame_end_timestamp=frame_end_us,
                    frustum_cull=frustum_cull,
                )
            else:
                # Estimation is disabled or no ray timestamps are available: use one timestamp for all tracks.
                tracks_timestamps_data = collect.TracksTimestampsGlobalData(
                    timestamp=default_timestamp,
                )

        with ScopedTimer("RigidGaussianModel/get_tracks_poses"):
            nearest_neighbor = is_lidar and self.nearest_neighbor_track_for_lidar

            tracks_interpolation_data = collect.TracksInterpolationData(
                tracks_poses=tracks_poses,
                tracks_timestamps=tracks_timestamps,
                tracks_packinfo=tracks_packinfo,
                timestamps_data=tracks_timestamps_data,
                nearest_neighbor=nearest_neighbor,
                max_positive_timestamp_gap_us=max_positive_timestamp_gap_us,
                frame_exposure_us=frame_exposure_us,
            )

            # Timestamp estimation and pose sampling strategy:
            # - The query timestamp is either a projection-based per-track estimate (if ray timestamps
            #   are available) or the frame midpoint fallback.
            # - Ordinary interpolation returns the query timestamp. LiDAR nearest-neighbor pose lookup
            #   deliberately keeps that sensor query timestamp for DeformNet: the selected annotation is
            #   the nearest pose approximation to the queried sensor time, rather than a change to the
            #   time being rendered.
            # - A query accepted just outside a track endpoint is clamped to the endpoint pose and returns
            #   that endpoint timestamp, keeping rigid-pose and DeformNet inputs consistent. The centered
            #   DeformNet smoothness stencil can still extend beyond the supported track interval; a
            #   one-sided endpoint stencil is intentionally left to a follow-up change.
            interpolated_pose_vec, interpolated_mask, interpolated_track_timestamps = (
                self.collector.interpolate_tracks_poses(tracks_interpolation_data)
            )

            # The collector intentionally tolerates queries just outside a real
            # track's endpoints. At synthetic split boundaries that would activate
            # two independently trained sibling assets. Keep the physical shared
            # endpoint pose, but give each internal timestamp to exactly one child:
            # [start, end), with the right/incoming child owning the boundary.
            if self._split_boundary_track_indices.numel() > 0:
                boundary_inside = (default_timestamp >= self._split_internal_start_timestamps_us) & (
                    default_timestamp < self._split_internal_end_timestamps_us
                )
                # The Slang interpolation autograd function saves its returned
                # mask for backward. Do not mutate that tensor in-place.
                interpolated_mask = interpolated_mask.clone()
                interpolated_mask[self._split_boundary_track_indices] &= boundary_inside

            # Regardless of whether we are using nearest neighbor or interpolation, we filter the
            # gaussians based on the and interpolated_mask.
            #
            # For each track, interpolated_mask is true when the interpolation timestamp is
            # accepted, including endpoint clamping within cadence plus frame exposure.
            # Otherwise, the Gaussians belonging to the track are filtered out.
            return interpolated_pose_vec, interpolated_mask, interpolated_track_timestamps

    def post_process(
        self,
        offset: int,
        results: collect.CollectorResult,
        layer_data: collect.LayerDataBase,
        rows: Optional[np.ndarray] = None,
    ) -> None:
        # Apply track albedo.  Note that this could be done in the merged kernel,
        # but it's not enabled by default, so we might optimize it later.
        if self.track_albedos is None:
            return
        assert isinstance(layer_data, collect.LayerDataRigid)
        gaussian_cuboid_ids = layer_data.tracks_ids
        nb_gaussians = gaussian_cuboid_ids.shape[0]
        # This is a read-modify-write, and the affine is not idempotent: it has to run
        # over the rows the collect wrote this frame and no others. `rows` is that list;
        # None means the whole layer, which is every frame of the dense path.
        windows = ((0, nb_gaussians),) if rows is None else tuple((int(r[0]), int(r[1])) for r in rows)

        for start, count in windows:
            if count == 0:
                continue
            features = results.features[offset + start : offset + start + count]

            track_albedos = self.track_albedos[gaussian_cuboid_ids[start : start + count]]
            # Matmul in track_albedos' dtype (fp32); features may be fp16, so cast in and back.
            features_albedo = features[..., :3].to(track_albedos.dtype)
            transformed_features_albedo = (
                (
                    track_albedos
                    @ (torch.cat([features_albedo, torch.ones_like(features_albedo[..., :1])], -1).unsqueeze(-1))
                )
                .squeeze(-1)
                .to(features.dtype)
            )
            features = torch.cat([transformed_features_albedo, features[..., 3:]], -1)
            results.features[offset + start : offset + start + count] = features

    #
    # End of implementation of what is necessary for parameter collection.
    #

    def configure_optimizers(self, name_prefix: str = "") -> list[OptimizerLRSchedulerConfig]:
        """Returns a list of module-owned configured optimizers (optimizers paired with an optional LR scheduler),
        which will be stepped in the main training loop, allowing the module to interact with it's owned optimizers"""

        optimizers = super().configure_optimizers(name_prefix)

        optimizers += self.tracks_calib.configure_optimizers(name_prefix=name_prefix)

        if self.track_albedos is not None:
            optimizers += configure_optimizers(self.config.track_albedo, self.trainer_config, self, name_prefix)

        if self.track_scales is not None:
            optimizers += configure_optimizers(self.config.track_scale, self.trainer_config, self, name_prefix)

        readable = []
        for entry in optimizers:
            optimizer = entry["optimizer"]
            opt_class = optimizer.__class__.__name__
            group_names = [pg.get("name", f"group{idx}") for idx, pg in enumerate(optimizer.param_groups)]
            readable.append(f"{opt_class}[" + ", ".join(map(str, group_names)) + "]")
        log.info(f"Create optimizer for {self.__class__.__name__}: " + "; ".join(readable))
        return optimizers

    def update_step_train_batch_start(self, epoch: int, global_step: int, system, **kwargs) -> dict[str, torch.Tensor]:
        self.global_step = global_step
        return update_module_step(self.tracks_calib, epoch, global_step, system)

    def get_track_poses_and_dims(self, track_id: str) -> Optional[Tuple[lt.SE3, torch.Tensor]]:
        """
        Gets the poses and cuboid dimensions of a track id if present
        """
        if track_id not in self.cuboid_tracks.tracks_id:
            return None

        cur_track = CuboidTracks.Ops.subset_from_tracks_id(self.get_calibrated_cuboid_tracks(), [track_id])
        assert cur_track.n_tracks == 1, f"Unexpected number of tracks: {cur_track.n_tracks}"
        return cur_track.tracks_poses, cur_track.cuboids_dims[0]

    def _compute_scale_factor(
        self,
        cuboid_dims: torch.Tensor,
        dims_offset: torch.Tensor,
    ) -> float:
        """
        Computes the scale factor for an asset based on cuboid dimensions and offset.

        Args:
            cuboid_dims: The dimensions of the cuboid [x, y, z]
            dims_offset: The offset to apply to cuboid dimensions

        Returns:
            scale_factor: The computed scale factor as a scalar float
        """
        log.info(f"Applying dims_offset: {dims_offset.tolist()} to asset cuboids_dims: {cuboid_dims.tolist()}")
        return (cuboid_dims - dims_offset).max().item()

    def track_ply_override(
        self,
        track_id: str,
        asset: Asset,
        transform: torch.Tensor,
        dims_offset: torch.Tensor,
    ) -> bool:
        """
        Overrides the gaussians for a track id with gaussians from an asset.
        Note that the override is only meant to be used at inference time and should be reverted
        by calling restore_training_parameters before resuming training.
        """
        if track_id not in self.cuboid_tracks.tracks_id:
            log.warning(
                f"Failed to replace track {track_id}, not found in {self.__class__.__name__}'s cuboid_tracks, skipping"
            )
            return False
        self.validate_external_gaussian_topology_edit("track PLY replacement")

        # ``transform`` and ``scale`` mutate the Asset. A track can be represented by more than one
        # gaussian node, and the composite passes the same Asset to each node; start every replacement
        # from the caller's pristine data so later nodes do not receive an already transformed/scaled asset.
        loaded_ply = asset.clone()
        track_index = self.cuboid_tracks.tracks_id.index(track_id)
        mask = self.gaussian_cuboid_ids != track_index
        loaded_ply.transform(transform)

        if loaded_ply.cuboids_dims is not None:
            scale_factor = self._compute_scale_factor(loaded_ply.cuboids_dims, dims_offset)
        else:
            scale_factor = self._compute_scale_factor(self.cuboid_tracks.cuboids_dims[track_index], dims_offset)

        loaded_ply.scale(
            scale_factor,
            self.scale_activation,
            self.scale_activation_inv,
        )

        self.save_training_parameters()

        if self.track_albedos is not None:
            self.track_albedos[track_index] = torch.eye(4, device=self.track_albedos.device)[:3]

        if self.track_scales is not None:
            self.track_scales[track_index] = self.scale_activation_inv(torch.ones(3), device=self.track_scales.device)

        # Handle features_albedo shape mismatch for temporal Fourier features - fix needed for asset harvester ply's (lower dimension than car2sim dynamic gaussians)
        if self.features_albedo.shape[1:] != loaded_ply.features_albedo.shape[1:]:
            # Shape mismatch detected - model expects [N, fourier_features_dim, 3], but PLY has [N, 3]
            if self.fourier_features_dim > 1:
                # Expand loaded PLY features to match the expected shape
                expanded_features_albedo = torch.zeros(
                    loaded_ply.features_albedo.shape[0],
                    self.fourier_features_dim,
                    3,
                    dtype=self.features_albedo.dtype,
                    device=self.features_albedo.device,
                )
                expanded_features_albedo[:, 0, :] = loaded_ply.features_albedo
                loaded_features_albedo = expanded_features_albedo
            else:
                # This shouldn't happen, but handle it anyway
                loaded_features_albedo = loaded_ply.features_albedo
        else:
            # No shape mismatch, use features as-is
            loaded_features_albedo = loaded_ply.features_albedo

        self.positions = nn.Parameter(torch.cat([self.positions[mask], loaded_ply.positions]))
        self.scales = nn.Parameter(torch.cat([self.scales[mask], loaded_ply.scales]))
        self.rotations = nn.Parameter(torch.cat([self.rotations[mask], loaded_ply.rotations]))
        self.densities = nn.Parameter(torch.cat([self.densities[mask], loaded_ply.densities]))
        self.features_albedo = nn.Parameter(torch.cat([self.features_albedo[mask], loaded_features_albedo]))
        self.features_specular = nn.Parameter(
            torch.cat(
                [
                    self.features_specular[mask],
                    loaded_ply.features_specular
                    if loaded_ply.features_specular is not None
                    else torch.zeros(
                        loaded_ply.features_albedo.shape[0],
                        self.features_specular.shape[1],
                        dtype=self.features_specular.dtype,
                        device=self.device,
                    ),
                ]
            )
        )
        # FIXME : extra_signal is not imported from the ply file
        self.extra_signal = nn.Parameter(
            torch.cat(
                [
                    self.extra_signal[mask],
                    torch.zeros(
                        loaded_ply.positions.shape[0],
                        self.extra_signal.shape[1],
                        dtype=self.extra_signal.dtype,
                        device=self.extra_signal.device,
                    ),
                ]
            )
        )

        camera_signal = self.get_extra_signal("camera")
        if camera_signal.shape[0] > 0:
            self.camera_extra_signal = nn.Parameter(
                torch.cat(
                    [
                        camera_signal[mask],
                        torch.zeros(
                            loaded_ply.positions.shape[0],
                            camera_signal.shape[1],
                            dtype=camera_signal.dtype,
                            device=camera_signal.device,
                        ),
                    ]
                )
            )

        lidar_signal = self.get_extra_signal("lidar")
        if lidar_signal.shape[0] > 0:
            self.lidar_extra_signal = nn.Parameter(
                torch.cat(
                    [
                        lidar_signal[mask],
                        torch.zeros(
                            loaded_ply.positions.shape[0],
                            lidar_signal.shape[1],
                            dtype=lidar_signal.dtype,
                            device=lidar_signal.device,
                        ),
                    ]
                )
            )

        self.gaussian_cuboid_ids = nn.Buffer(
            torch.cat(
                [
                    self.gaussian_cuboid_ids[mask],
                    torch.full(
                        (loaded_ply.features_albedo.shape[0],),
                        track_index,
                        dtype=self.gaussian_cuboid_ids.dtype,
                        device=self.device,
                    ),
                ]
            )
        )
        log.info(f"Completed replacement of track {track_id}")
        return True

    def insert_asset(
        self,
        track_id: str,
        asset: Asset,
        cuboid_tracks: CuboidTracks,
        dims_offset: torch.Tensor,
        rgb_post_processing: RGBPostProcessing | None = None,
    ) -> bool:
        """
        Inserts a new track_id with gaussians derived from 'asset'.
        Note that the insert is only meant to be used at inference time and should be reverted
        by calling restore_training_parameters before resuming training.
        """
        prepared = self.prepare_asset_insertion(
            track_id,
            asset,
            cuboid_tracks,
            dims_offset,
            rgb_post_processing=rgb_post_processing,
        )
        self.apply_prepared_asset_insertion(prepared)
        return True

    def _prepare_asset_insertion_model_state(
        self,
        cuboid_tracks: CuboidTracks,
        inserted_track_ids: set[str],
    ) -> object | None:
        """Stage subclass-owned track tables without changing the live model."""
        return None

    def _apply_prepared_asset_insertion_model_state(self, prepared: object | None) -> None:
        """Publish subclass-owned state after all scene layers are prepared."""
        if prepared is not None:
            raise TypeError(f"Unexpected prepared asset-insertion state for {type(self).__name__}")

    def prepare_asset_insertion(
        self,
        track_id: str,
        asset: Asset,
        cuboid_tracks: CuboidTracks,
        dims_offset: torch.Tensor,
        rgb_post_processing: RGBPostProcessing | None = None,
        *,
        target_cuboid_tracks: CuboidTracks | None = None,
        inserted_track_ids: set[str] | None = None,
    ) -> PreparedRigidAssetInsertion:
        """Build the complete post-insert layer state without mutating the model.

        ``target_cuboid_tracks`` is supplied by the composite while it stages a
        scene-wide edit. Direct callers retain the historical contract: they update
        ``self.cuboid_tracks`` first and omit this argument.
        """
        self.validate_external_gaussian_topology_edit("asset insertion")
        transform_matrix = [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
        transform: torch.Tensor = torch.FloatTensor(transform_matrix).to(self.device)

        if target_cuboid_tracks is None:
            target_cuboid_tracks = self.cuboid_tracks
            tracks_calib = self.tracks_calib
            time_embed = self.time_embed
            model_specific_state = None
        else:
            if inserted_track_ids is None:
                raise ValueError("inserted_track_ids is required with target_cuboid_tracks")
            tracks_calib, time_embed = self._prepare_tracks_calib_and_time_embed(
                target_cuboid_tracks,
                inserted_track_ids,
            )
            model_specific_state = self._prepare_asset_insertion_model_state(
                target_cuboid_tracks,
                inserted_track_ids,
            )
        track_index = target_cuboid_tracks.tracks_id.index(track_id)

        # Transform a private clone so a failed preparation cannot alter the caller's
        # reusable asset before any layer has accepted the scene-wide edit.
        loaded_ply = asset.clone()
        loaded_ply.transform(transform)

        scale_factor = self._compute_scale_factor(cuboid_tracks.cuboids_dims[0], dims_offset)
        loaded_ply.scale(
            scale_factor,
            self.scale_activation,
            self.scale_activation_inv,
        )

        if self.track_albedos is not None:
            track_albedos = nn.Parameter(
                torch.cat([self.track_albedos, torch.eye(4, device=self.track_albedos.device)[:3].unsqueeze(0)])
            )
        else:
            track_albedos = None
        if self.track_scales is not None:
            track_scales = nn.Parameter(
                torch.cat(
                    [self.track_scales, self.scale_activation_inv(torch.ones(1, 3, device=self.track_scales.device))]
                )
            )
        else:
            track_scales = None

        # Handle features_albedo shape mismatch for temporal Fourier features - fix needed for asset harvester ply's (lower dimension than car2sim dynamic gaussians)
        if self.features_albedo.shape[1:] != loaded_ply.features_albedo.shape[1:]:
            # Shape mismatch detected - model expects [N, fourier_features_dim, 3], but PLY has [N, 3]
            if self.fourier_features_dim > 1:
                # Expand loaded PLY features to match the expected shape
                expanded_features_albedo = torch.zeros(
                    loaded_ply.features_albedo.shape[0],
                    self.fourier_features_dim,
                    3,
                    dtype=self.features_albedo.dtype,
                    device=self.features_albedo.device,
                )
                expanded_features_albedo[:, 0, :] = loaded_ply.features_albedo
                loaded_features_albedo = expanded_features_albedo
            else:
                # This shouldn't happen, but handle it anyway
                loaded_features_albedo = loaded_ply.features_albedo
        else:
            # No shape mismatch, use features as-is
            loaded_features_albedo = loaded_ply.features_albedo

        if rgb_post_processing is not None:
            loaded_features_albedo = loaded_features_albedo.clone()
            with torch.no_grad():
                apply_rgb_post_processing_to_features_dc(loaded_features_albedo, rgb_post_processing)

        positions = nn.Parameter(torch.cat([self.positions, loaded_ply.positions]))
        scales = nn.Parameter(torch.cat([self.scales, loaded_ply.scales]))
        rotations = nn.Parameter(torch.cat([self.rotations, loaded_ply.rotations]))
        densities = nn.Parameter(torch.cat([self.densities, loaded_ply.densities]))
        features_albedo = nn.Parameter(torch.cat([self.features_albedo, loaded_features_albedo]))
        features_specular = nn.Parameter(
            torch.cat(
                [
                    self.features_specular,
                    loaded_ply.features_specular
                    if loaded_ply.features_specular is not None
                    else torch.zeros(
                        loaded_ply.features_albedo.shape[0],
                        self.features_specular.shape[1],
                        dtype=self.features_specular.dtype,
                        device=self.device,
                    ),
                ]
            )
        )
        # FIXME : extra_signal is not imported from the ply file
        extra_signal = nn.Parameter(
            torch.cat(
                [
                    self.extra_signal,
                    torch.zeros(
                        loaded_ply.positions.shape[0],
                        self.extra_signal.shape[1],
                        dtype=self.extra_signal.dtype,
                        device=self.extra_signal.device,
                    ),
                ]
            )
        )

        camera_signal = self.get_extra_signal("camera")
        if camera_signal.shape[0] > 0:
            camera_extra_signal = nn.Parameter(
                torch.cat(
                    [
                        camera_signal,
                        torch.zeros(
                            loaded_ply.positions.shape[0],
                            camera_signal.shape[1],
                            dtype=camera_signal.dtype,
                            device=camera_signal.device,
                        ),
                    ]
                )
            )
        else:
            # Untrained optional signal kept as an empty (0, dim) buffer: materialize
            # zeros for every gaussian so its row count matches positions after insertion
            # (the collector asserts all per-gaussian buffers share the same row count).
            camera_extra_signal = nn.Parameter(
                torch.zeros(
                    positions.shape[0],
                    camera_signal.shape[1],
                    dtype=camera_signal.dtype,
                    device=camera_signal.device,
                )
            )

        lidar_signal = self.get_extra_signal("lidar")
        if lidar_signal.shape[0] > 0:
            lidar_extra_signal = nn.Parameter(
                torch.cat(
                    [
                        lidar_signal,
                        torch.zeros(
                            loaded_ply.positions.shape[0],
                            lidar_signal.shape[1],
                            dtype=lidar_signal.dtype,
                            device=lidar_signal.device,
                        ),
                    ]
                )
            )
        else:
            lidar_extra_signal = nn.Parameter(
                torch.zeros(
                    positions.shape[0],
                    lidar_signal.shape[1],
                    dtype=lidar_signal.dtype,
                    device=lidar_signal.device,
                )
            )

        gaussian_cuboid_ids = nn.Buffer(
            torch.cat(
                [
                    self.gaussian_cuboid_ids,
                    torch.full(
                        (loaded_ply.features_albedo.shape[0],),
                        track_index,
                        dtype=self.gaussian_cuboid_ids.dtype,
                        device=self.device,
                    ),
                ]
            )
        )
        return PreparedRigidAssetInsertion(
            track_id=track_id,
            cuboid_tracks=target_cuboid_tracks,
            tracks_calib=tracks_calib,
            time_embed=time_embed,
            positions=positions,
            scales=scales,
            rotations=rotations,
            densities=densities,
            features_albedo=features_albedo,
            features_specular=features_specular,
            extra_signal=extra_signal,
            camera_extra_signal=camera_extra_signal,
            lidar_extra_signal=lidar_extra_signal,
            gaussian_cuboid_ids=gaussian_cuboid_ids,
            track_albedos=track_albedos,
            track_scales=track_scales,
            model_specific_state=model_specific_state,
        )

    def apply_prepared_asset_insertion(
        self,
        prepared: PreparedRigidAssetInsertion,
        *,
        save_training_parameters: bool = True,
    ) -> None:
        """Publish an already prepared asset insertion using assignment-only updates."""
        if save_training_parameters:
            self.save_training_parameters()

        self.cuboid_tracks = prepared.cuboid_tracks
        self.tracks_calib = prepared.tracks_calib
        self.time_embed = prepared.time_embed
        self.positions = prepared.positions
        self.scales = prepared.scales
        self.rotations = prepared.rotations
        self.densities = prepared.densities
        self.features_albedo = prepared.features_albedo
        self.features_specular = prepared.features_specular
        self.extra_signal = prepared.extra_signal
        self.camera_extra_signal = prepared.camera_extra_signal
        self.lidar_extra_signal = prepared.lidar_extra_signal
        self.gaussian_cuboid_ids = prepared.gaussian_cuboid_ids
        self.track_albedos = prepared.track_albedos
        self.track_scales = prepared.track_scales
        if prepared.model_specific_state is not None:
            self._apply_prepared_asset_insertion_model_state(prepared.model_specific_state)
        log.info(f"Completed insertion of track {prepared.track_id}")

    def update_tracks_calib_and_time_embed(self, cuboid_tracks: CuboidTracks, inserted_track_ids: set[str]) -> None:
        """
        Updates tracks_calib to handle newly inserted tracks while preserving learned calibration.

        Args:
            cuboid_tracks: Combined tracks (original + inserted)
            inserted_track_ids: Set of newly inserted track IDs
        """

        self.tracks_calib, self.time_embed = self._prepare_tracks_calib_and_time_embed(
            cuboid_tracks,
            inserted_track_ids,
        )

    def _prepare_tracks_calib_and_time_embed(
        self,
        cuboid_tracks: CuboidTracks,
        inserted_track_ids: set[str],
    ) -> tuple[BaseTracksCalib, BaseInputEmbedding | None]:
        """Construct post-insert track calibration and time embedding without publishing them."""
        if isinstance(self.tracks_calib, CompositeTracksCalib):
            original_calib = self.tracks_calib.original_calib
            inserted_track_ids = self.tracks_calib.inserted_track_ids | inserted_track_ids
        else:
            original_calib = self.tracks_calib

        tracks_calib = CompositeTracksCalib(
            config=self.config.tracks_calib,
            trainer_config=self.trainer_config,
            original_calib=original_calib,
            inserted_track_ids=inserted_track_ids,
            cuboid_tracks=cuboid_tracks,
        )

        # Recreate time_embed with updated tracks if using temporal features
        time_embed = self.time_embed
        if unpack_optional(self.config.fourier_features_dim) > 1 and time_embed is not None:
            time_embed = BaseInputEmbedding.factory(
                self.config.time_embed.name,
                self.config.time_embed,
                self.trainer_config,
                torch.tensor(self.start_end_timestamp_us),
                cuboid_tracks,
            )
        return tracks_calib, time_embed

    def save_training_parameters(self):
        first_save = len(self.overriden_parameters) == 0
        super().save_training_parameters()
        if first_save:
            assert len(self.overriden_buffers) == 0, (
                "overriden_buffers should be empty if overriden_parameters was empty"
            )
            self.overriden_buffers["gaussian_cuboid_ids"] = self.gaussian_cuboid_ids
            self.overriden_state["cuboid_tracks"] = self.cuboid_tracks
            self.overriden_state["tracks_calib"] = self.tracks_calib
            self.overriden_state["time_embed"] = self.time_embed
            if self.track_albedos is not None:
                self.overriden_parameters["track_albedos"] = self.track_albedos
            if self.track_scales is not None:
                self.overriden_parameters["track_scales"] = self.track_scales
            # `camera_extra_signal`/`lidar_extra_signal` are per-gaussian tensors owned by the base
            # save/restore pair (BaseGaussianModel.save_training_parameters), reached here via super().
            # No Rigid-specific duplicate is needed.

    def restore_training_parameters(self) -> None:
        if len(self.overriden_parameters) > 0:
            self.cuboid_tracks = self.overriden_state["cuboid_tracks"]
            self.tracks_calib = self.overriden_state["tracks_calib"]
            self.time_embed = self.overriden_state["time_embed"]
            self.gaussian_cuboid_ids = self.overriden_buffers["gaussian_cuboid_ids"]
            if "track_albedos" in self.overriden_parameters:
                self.track_albedos = self.overriden_parameters["track_albedos"]
            if "track_scales" in self.overriden_parameters:
                self.track_scales = self.overriden_parameters["track_scales"]
            # camera/lidar extra signals are restored by the base restore (via super()), which also
            # tolerates an older snapshot that predates those keys.
            super().restore_training_parameters()

    def get_extra_state(self) -> dict[str, Any] | None:
        extra_state = super().get_extra_state() or {}
        extra_state["n_tracks"] = self.cuboid_tracks.n_tracks
        return extra_state

    def set_extra_state(self, state: dict[str, Any] | None) -> None:
        super().set_extra_state(state)
        if (state is not None) and "n_tracks" in state:
            assert self.cuboid_tracks.n_tracks == state["n_tracks"], (
                "State contains a different number of tracks than the model requires"
            )

    def export_ply(
        self,
        export_dir: Path,
        format: GaussianExportFormat = GaussianExportFormat._3DGS,
        percentage_gaussians: float = 100,
    ) -> None:
        """
        Export Gaussian model as PLY files in the specified format.

        For 3DGS format, it should be compatible with the original 3DGS implementation but differences
        between 3DGS/3DGUT/3DGRT rendering will cause slight differences when rendered with
        3rd-party 3DGS viewers.
        NB : extra_signal is not exported.

        Args:
            export_dir: Directory path where the PLY file will be saved as 'model.ply'.
            format: Export format for the Gaussian data. Available options:
                - GaussianExportFormat._3DGS: Original 3D Gaussian Splatting format
                - GaussianExportFormat._3DGRT: 3DGRT format for visualization
            percentage_gaussians: Percentage of Gaussians to export (0, 100]. Only used
                for _3DGRT format. Defaults to 100 (export all Gaussians).
        """
        for track_index, track_id in enumerate(self.cuboid_tracks.tracks_id):
            track_mask = self.gaussian_cuboid_ids == track_index
            if track_mask.any():
                if format == GaussianExportFormat._3DGS:
                    write_ply_3dgs(
                        export_dir / f"{track_id}.ply",
                        self.positions[track_mask],
                        self.rotations[track_mask],
                        self.scales[track_mask],
                        self.densities[track_mask],
                        self.features_albedo[track_mask],
                        self.features_specular[track_mask],
                    )
                elif format == GaussianExportFormat._3DGRT:
                    write_ply_3dgrt(
                        export_dir / f"{track_id}.ply",
                        self.get_positions(),
                        self.get_rotations(quaternion_format="xyzw"),
                        self.get_scales(),
                        self.get_densities(),
                        percentage_gaussians=percentage_gaussians,
                    )

    def get_number_of_gaussians_per_track(self) -> dict[str, int]:
        """
        Returns the number of gaussians, if tracks are available this is returned per track.
        """
        return {
            self.cuboid_tracks.tracks_id[track_idx]: count.item()
            for track_idx, count in zip(*torch.unique(self.gaussian_cuboid_ids, return_counts=True))
        }


class DeformationNetwork(nn.Module):
    config: DictConfig
    feature_volume: HashGridObjectFeatureVolume

    def __init__(
        self,
        config: DictConfig,
        trainer_config: TrainerConfig,
        precision: int,
        cuboid_tracks: CuboidTracks,
    ):
        super().__init__()
        self.config = config
        self.trainer_config = trainer_config

        self.feature_volume = HashGridObjectFeatureVolume(
            self.config.feature_volume, trainer_config, precision, cuboid_tracks
        )

    def forward(self, embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass with pre-computed concatenated embeddings.

        Args:
            embeddings: Concatenated embeddings [N, 3 + instance_dim + time_dim]

        Returns:
            Dictionary with deformation parameters (positions, rotations, scales)
        """
        out = self.feature_volume(embeddings)

        # NOTE : TCNN MLP outputs at least 16 floats
        out_dim = out.shape[-1]
        assert out_dim >= 10, f"Deformation network output dimension must be at least 10, got {out_dim}"
        positions, rotations, scales, _ = torch.split(out.float(), [3, 4, 3, out_dim - 10], dim=-1)
        ret = {"positions": positions}
        if self.config.deform_rotations:
            ret["rotations"] = rotations
        if self.config.deform_scales:
            ret["scales"] = scales

        return ret

    def configure_optimizers(self, name_prefix: str = "") -> list[OptimizerLRSchedulerConfig]:
        """Returns a list of module-owned configured optimizers (optimizers paired with an optional LR scheduler),
        which will be stepped in the main training loop, allowing the module to interact with it's owned optimizers"""

        return configure_optimizers(self.config, self.trainer_config, self, name_prefix)


class DeformableGaussianModel(RigidGaussianModel):
    deform_network: DeformationNetwork

    config: DeformableGaussiansLayerConfig  # type: ignore[assignment] # Narrow type from RigidGaussiansLayerConfig

    def __init__(
        self,
        config: DeformableGaussiansLayerConfig,
        trainer_config: TrainerConfig,
        datasource: DataSourceSummary,
        init_from_datasource: bool,
        initializer: Optional[BaseInitialization],
        precision: Optional[int] = None,
        cuboid_tracks: Optional[CuboidTracks] = None,
        start_end_timestamp_us: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__(
            config,
            trainer_config,
            datasource,
            init_from_datasource,
            initializer,
            precision,
            cuboid_tracks,
            start_end_timestamp_us,
        )

        self.config = self.config.model_copy(
            update={
                "deformnet_start_iteration": adjust_step_for_world_size(
                    trainer_config, self.config.deformnet_start_iteration
                )
            },
            deep=True,
        )
        log.info(f"DeformableGaussianModel: deformnet_start_iteration={self.config.deformnet_start_iteration}")

        aabb_extent = self.cuboid_tracks.cuboids_dims * (1.0 + self.config.cuboid_tracks_expand)
        self.aabb_extent = nn.Buffer(aabb_extent, persistent=True)
        aabb_extent = torch.max(aabb_extent, dim=1, keepdim=True).values.repeat(1, 3)
        self.aabbs = AABB3D(blb=-aabb_extent / 2.0, trf=aabb_extent / 2.0)
        self._scene_contractor = SceneContractor(None, self.aabbs, is_single=False, is_merf=False)

        self.deform_network = DeformationNetwork(
            self.config.deform_network,
            trainer_config,
            unpack_optional(precision),
            cuboid_tracks=self.cuboid_tracks,
        )

        # Frame step used to regularize the smoothness of the deformation field
        gaps = self.cuboid_tracks.tracks_timestamps_us[1:] - self.cuboid_tracks.tracks_timestamps_us[:-1]
        gaps = gaps[gaps > 0]  # To ignore negative timestamps when we switch tracks
        self.frame_step = gaps.median().item()

        self.use_deform_network = False

        # Track indices of replaced tracks. For asset replacement, we need to exclude the replaced tracks from the deformation computation
        # as the network would not be trained on some of the positions / rotations of the asset
        self._replaced_track_indices: set[int] = set()

        # Track indices of inserted tracks. The instance embedding is a fixed linspace grid the
        # deformation field was trained against, so an inserted track has no coordinate of its own
        # and renders rigid, like a replaced track.
        self._inserted_track_indices: set[int] = set()

        # Per-track tables as they were before the first insert, as (owner, attribute, tensor), so
        # `restore_training_parameters` can put them back.
        self._pre_insert_deform_tables: list[tuple[nn.Module, str, torch.Tensor]] | None = None

        # Component-only frozen tracks: a per-track (n_tracks,) bool mask,
        # True where the track is frozen. When set, the deform-network output
        # for those tracks' Gaussians is detached so no gradient reaches the
        # shared deformation hash grid from a frozen track.
        self._component_only_frozen_track_mask: Optional[torch.Tensor] = None

    def get_scene_contractor(self) -> SceneContractor:
        return self._scene_contractor

    def track_ply_override(
        self,
        track_id: str,
        asset: Asset,
        transform: torch.Tensor,
        dims_offset: torch.Tensor,
    ) -> bool:
        """
        Overrides the gaussians for a track id with gaussians from an asset.
        Replaced tracks are excluded from deformation computation.
        """
        if track_id not in self.cuboid_tracks.tracks_id:
            log.warning(
                f"Failed to replace track {track_id}, not found in {self.__class__.__name__} cuboid_tracks, skipping"
            )
            return False

        track_index = self.cuboid_tracks.tracks_id.index(track_id)

        replaced = super().track_ply_override(track_id, asset, transform, dims_offset)
        if not replaced:
            return False

        self._replaced_track_indices.add(track_index)
        return True

    def update_tracks_calib_and_time_embed(self, cuboid_tracks: CuboidTracks, inserted_track_ids: set[str]) -> None:
        """Grow the per-track tables the deform embeddings gather alongside the node's track table.

        The instance-embedding weights, the per-track timestamp ranges and this node's AABBs are
        sized to the tracks the model was constructed with. Inserting a track grows the node's track
        table but not those, and the Slang kernel gathers all three by ``instance_idx`` with no
        bounds check, so an inserted track reads past the end of each.
        """
        prepared = self._prepare_asset_insertion_model_state(cuboid_tracks, inserted_track_ids)
        super().update_tracks_calib_and_time_embed(cuboid_tracks, inserted_track_ids)
        self._apply_prepared_asset_insertion_model_state(prepared)

    def _prepare_asset_insertion_model_state(
        self,
        cuboid_tracks: CuboidTracks,
        inserted_track_ids: set[str],
    ) -> PreparedDeformableTrackInsertion:
        """Allocate post-insert DeformNet tables without publishing them."""

        from nurec.models.input_embedding import (
            IndividualTimeInputEmbedding,
            WeightedInstanceInputEmbedding,
        )

        volume = self.deform_network.feature_volume
        instance_embedding = volume.instance_input_embedding
        time_embedding = volume.time_input_embedding
        # `feature_volume`'s submodules are typed as bare Modules; these are the only shapes
        # `_compute_deform_embeddings` accepts, and it raises for anything else.
        assert isinstance(instance_embedding, WeightedInstanceInputEmbedding)
        assert isinstance(time_embedding, IndividualTimeInputEmbedding)

        inserted_track_indices = frozenset(
            index for index, track_id in enumerate(cuboid_tracks.tracks_id) if track_id in inserted_track_ids
        )

        n_rows = cuboid_tracks.n_tracks - instance_embedding.embedding.weight.shape[0]
        if n_rows <= 0:
            return PreparedDeformableTrackInsertion(
                inserted_track_indices=inserted_track_indices,
                num_embeddings=cuboid_tracks.n_tracks,
                replacements=None,
            )

        tables: list[tuple[nn.Module, str]] = [
            (instance_embedding.embedding, "weight"),
            (time_embedding, "timestamps_us_ranges"),
            (self, "aabb_extent"),
            (self.aabbs, "blb"),
            (self.aabbs, "trf"),
        ]
        # Inserted tracks render rigid, so what the field returns for their rows is discarded and
        # the rows only need to be in bounds: repeat each table's last row. Rows for pre-existing
        # tracks are untouched, and each replacement stays on its source table's device.
        replacements: list[torch.Tensor] = []
        with torch.no_grad():
            for owner, name in tables:
                old = getattr(owner, name)
                new = torch.cat([old, old[-1:].expand(n_rows, *old.shape[1:])])
                replacements.append(nn.Parameter(new, old.requires_grad) if isinstance(old, nn.Parameter) else new)

        instance_embedding_weight, timestamps_us_ranges, aabb_extent, aabb_blb, aabb_trf = replacements
        assert isinstance(instance_embedding_weight, nn.Parameter)
        return PreparedDeformableTrackInsertion(
            inserted_track_indices=inserted_track_indices,
            num_embeddings=cuboid_tracks.n_tracks,
            replacements=PreparedDeformableTableReplacements(
                instance_embedding_weight=instance_embedding_weight,
                timestamps_us_ranges=timestamps_us_ranges,
                aabb_extent=aabb_extent,
                aabb_blb=aabb_blb,
                aabb_trf=aabb_trf,
            ),
        )

    def _apply_prepared_asset_insertion_model_state(self, prepared: object | None) -> None:
        """Publish staged DeformNet tables using assignment-only updates."""
        if prepared is None:
            return
        if not isinstance(prepared, PreparedDeformableTrackInsertion):
            raise TypeError(f"Unexpected prepared DeformNet insertion state: {type(prepared).__name__}")

        # The composite inserts one track per call, so accumulate until restore.
        self._inserted_track_indices.update(prepared.inserted_track_indices)
        if prepared.replacements is None:
            return

        from nurec.models.input_embedding import WeightedInstanceInputEmbedding

        instance_embedding = self.deform_network.feature_volume.instance_input_embedding
        assert isinstance(instance_embedding, WeightedInstanceInputEmbedding)
        tables: list[tuple[nn.Module, str, torch.Tensor]] = [
            (instance_embedding.embedding, "weight", prepared.replacements.instance_embedding_weight),
            (
                self.deform_network.feature_volume.time_input_embedding,
                "timestamps_us_ranges",
                prepared.replacements.timestamps_us_ranges,
            ),
            (self, "aabb_extent", prepared.replacements.aabb_extent),
            (self.aabbs, "blb", prepared.replacements.aabb_blb),
            (self.aabbs, "trf", prepared.replacements.aabb_trf),
        ]
        # Hold the original tables by reference so restore returns the exact trained
        # objects after one or more inference-time insertions.
        if self._pre_insert_deform_tables is None:
            self._pre_insert_deform_tables = [(owner, name, getattr(owner, name)) for owner, name, _ in tables]
        for owner, name, replacement in tables:
            setattr(owner, name, replacement)
        instance_embedding.embedding.num_embeddings = prepared.num_embeddings

    def restore_training_parameters(self) -> None:
        if self._pre_insert_deform_tables is not None:
            from nurec.models.input_embedding import WeightedInstanceInputEmbedding

            for owner, name, original in self._pre_insert_deform_tables:
                setattr(owner, name, original)
            instance_embedding = self.deform_network.feature_volume.instance_input_embedding
            assert isinstance(instance_embedding, WeightedInstanceInputEmbedding)
            instance_embedding.embedding.num_embeddings = instance_embedding.embedding.weight.shape[0]
            self._pre_insert_deform_tables = None
        self._replaced_track_indices.clear()
        self._inserted_track_indices.clear()
        super().restore_training_parameters()

    def freeze_tracks(self, frozen_track_indices: Sequence[int]) -> None:
        """Freeze this deformable node's frozen tracks — the rigid params
        (``track_albedos``/``track_scales``, calibration) via the base, plus
        the shared deformation network. The hash grid is not track-sliceable,
        so instead ``get_layer_data`` detaches the deform-network output for
        frozen tracks' Gaussians (see ``_component_only_frozen_track_mask``),
        stopping any gradient from a frozen track reaching the grid.
        """
        super().freeze_tracks(frozen_track_indices)
        frozen = [int(t) for t in frozen_track_indices]
        if not frozen or self.cuboid_tracks is None:
            self._component_only_frozen_track_mask = None
            return
        n_tracks = int(self.cuboid_tracks.tracks_packinfo.shape[0])
        mask = torch.ones(n_tracks, dtype=torch.bool)
        mask[torch.tensor(frozen, dtype=torch.long)] = False
        self._component_only_frozen_track_mask = mask  # True = trainable track

    def _compute_deform_embeddings(
        self,
        xyzs: torch.Tensor,
        instance_idx: torch.Tensor,
        timestamp_us: int | torch.Tensor,
        timestamps_delta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Helper to convert positions and metadata to concatenated embeddings for deformation network.

        Args:
            xyzs: Positions [N, 3]
            instance_idx: Instance indices [N]
            timestamp_us: A single timestamp for all samples or one timestamp per
                instance/track, in microseconds. ``instance_idx`` maps each sample
                to its instance timestamp.
            timestamps_delta: Optional per-sample timestamp delta [N] in microseconds.
                            If provided, outputs 3 embeddings stacked along batch dimension.

        Returns:
            Concatenated embeddings tensor:
            - Without timestamps_delta: [N, 3 + instance_dim + time_dim]
            - With timestamps_delta: [3*N, 3 + instance_dim + time_dim] stacked as [t, t-delta, t+delta]
        """
        from nurec.models.input_embedding import (
            IndividualRemapTimeInputEmbedding,
            WeightedInstanceInputEmbedding,
        )

        instance_embedding = self.deform_network.feature_volume.instance_input_embedding
        time_embedding = self.deform_network.feature_volume.time_input_embedding

        if isinstance(instance_embedding, WeightedInstanceInputEmbedding):
            assert instance_embedding.embedding_dim == 1, (
                f"Collector only supports instance embedding dimension of 1, got {instance_embedding.embedding_dim}"
            )
            instance_embedding_weights = collect.WeightedInstanceInputEmbeddingData(
                instance_embedding_weights=instance_embedding.embedding.weight
            )
        else:
            raise NotImplementedError(
                f"Unsupported instance embedding type: {type(instance_embedding).__name__}. "
                f"Supported types: WeightedInstanceInputEmbedding"
            )

        time_embedding_config: collect.EmbeddingConfig
        if type(time_embedding) is IndividualRemapTimeInputEmbedding:
            time_embedding_config = collect.IndividualRemapTimeInputEmbeddingConfig(
                timestamps_us_ranges=time_embedding.timestamps_us_ranges,
                remap_min=time_embedding.remap_min,
                remap_max=time_embedding.remap_max,
            )
        else:
            raise NotImplementedError(
                f"Unsupported time embedding type: {type(time_embedding).__name__}. "
                f"Supported types: IndividualRemapTimeInputEmbedding"
            )

        sc = self.get_scene_contractor()
        scene_contractor = collect.SceneContractorData(
            aabb_blb=sc.aabb.blb,
            aabb_trf=sc.aabb.trf,
            degree=sc.degree,
            is_merf=sc.is_merf,
        )

        input_embedding_data = collect.InputEmbeddingData(
            xyzs=xyzs,
            instance_idx=instance_idx.to(torch.int32),
            timestamps_us=timestamp_us,
            timestamps_delta=timestamps_delta,
            scene_contractor=scene_contractor,
            instance_embedding_weights=instance_embedding_weights,
            time_embedding_config=time_embedding_config,
        )

        return self.collector.prepare_input_embeddings(input_embedding_data)

    #
    # Implementation of what is necessary for parameter collection.
    #
    def get_layer_config(self) -> collect.LayerConfigDeformable:
        layer_config = super().get_layer_config()
        assert isinstance(layer_config, collect.LayerConfigSH)
        return collect.LayerConfigDeformable(
            rotation_activation=layer_config.rotation_activation,
            scale_activation=layer_config.scale_activation,
            density_activation=layer_config.density_activation,
            fourier_features_dim=layer_config.fourier_features_dim,
            embed_config=layer_config.embed_config,
        )

    def get_layer_data(
        self, context: BaseGaussianModel.CollectionContext, gathered_parameters: dict[str, torch.Tensor]
    ) -> collect.LayerDataDeformable:
        base_layer_data = super().get_layer_data(context, gathered_parameters)

        deform_positions = None
        deform_rotations = None

        gaussian_cuboid_ids = base_layer_data.tracks_ids
        if self.use_deform_network and gaussian_cuboid_ids.numel() > 0:
            with ScopedTimer("DeformableGaussianModel/deform_network"):
                base_positions = base_layer_data.positions
                track_timestamps = context.additional_data["interpolated_track_timestamps"][0]
                if context.is_training_batch:
                    numel_deform = base_positions.shape[0]
                    delta_range = max(2, int(self.config.smoothness_frame_steps * self.frame_step))
                    # Sample via the process-wide CUDA Generator (see
                    # `nurec/utils/cuda_rng.py`) so `torch.randint` does not advance
                    # the CUDA default generator's Philox offset on the training
                    # hot path — required for `torch.cuda.graph(...)` capture.
                    timestamp_delta = torch.randint(
                        1,
                        delta_range,
                        (numel_deform,),
                        generator=get_cuda_generator(base_positions.device),
                        device=base_positions.device,
                    )

                    # Compute embeddings with time deltas in a single call
                    # Output shape: [3*N, 3 + instance_dim + time_dim]
                    # Rows ordered as: [emb(t), emb(t-delta), emb(t+delta)]
                    embeddings = self._compute_deform_embeddings(
                        xyzs=base_positions,
                        instance_idx=gaussian_cuboid_ids,
                        timestamp_us=track_timestamps,
                        timestamps_delta=timestamp_delta,
                    )
                    deform_parameters_batch = self.deform_network(embeddings)
                    if self._component_only_frozen_track_mask is not None:
                        # Component-only frozen tracks: detach the deform-network
                        # output for their Gaussians so no gradient reaches the
                        # shared hash grid. The batch is [emb(t), emb(t-delta),
                        # emb(t+delta)] — the same gaussian_cuboid_ids ordering
                        # per block — so the per-Gaussian keep mask repeats 3x.
                        keep = (
                            self._component_only_frozen_track_mask.to(gaussian_cuboid_ids.device)[gaussian_cuboid_ids]
                            .repeat(3)
                            .unsqueeze(1)
                        )
                        deform_parameters_batch = {
                            k: torch.where(keep, v, v.detach()) for k, v in deform_parameters_batch.items()
                        }
                    deform_parameters = dict[str, torch.Tensor]()
                    for k, v in deform_parameters_batch.items():
                        deform_parameters[k] = v[:numel_deform]

                    position_deform = deform_parameters["positions"]
                    position_deform_before = deform_parameters_batch["positions"][numel_deform : numel_deform * 2]
                    position_deform_after = deform_parameters_batch["positions"][numel_deform * 2 :]

                    deform_smoothness = 0.5 * (position_deform_before + position_deform_after) - position_deform

                    # Pass the mask as additional data to avoid CUDA sync from boolean indexing.
                    # The mask will be used in the loss for masked reduction.
                    interpolated_mask = context.additional_data["interpolated_mask"][0]
                    keep_mask = interpolated_mask[gaussian_cuboid_ids].float().unsqueeze(1)
                    context.additional_data["deform_smoothness"] = (deform_smoothness, True)  # [N, 3]
                    context.additional_data["deform_smoothness_mask"] = (keep_mask, True)  # [N, 1]
                else:
                    embeddings = self._compute_deform_embeddings(
                        xyzs=base_positions,
                        instance_idx=gaussian_cuboid_ids,
                        timestamp_us=track_timestamps,
                    )
                    deform_parameters = self.deform_network(embeddings)

                    # Zero out deformation for replaced and inserted tracks (rigid behavior)
                    rigid_track_indices = self._replaced_track_indices | self._inserted_track_indices
                    if rigid_track_indices:
                        replaced_ids = torch.tensor(
                            list(rigid_track_indices),
                            device=gaussian_cuboid_ids.device,
                            dtype=gaussian_cuboid_ids.dtype,
                        )
                        replaced_mask = torch.isin(gaussian_cuboid_ids, replaced_ids)
                        deform_parameters["positions"][replaced_mask] = 0.0
                        deform_parameters["rotations"][replaced_mask] = 0.0

                # We only support deformed positions and rotations, that's the only thing used in our configs.
                # We assume we always want positions and rotations, and never scales.  We could make this
                # configurable if we wanted.
                assert "positions" in deform_parameters, "Deform positions must be in deform_parameters"
                assert self.config.deform_network.deform_rotations, "Deform rotations must be enabled"
                assert "rotations" in deform_parameters, "Deform rotations must be in deform_parameters"
                assert self.config.deform_network.rotations_from_identity, "Rotations must be from identity"
                assert not self.config.deform_network.deform_scales, "Deform scales must be disabled"
                assert "scales" not in deform_parameters, "Deform scales must not be in deform_parameters"

                deform_positions = deform_parameters["positions"].to(base_positions)
                deform_rotations = deform_parameters["rotations"].to(base_layer_data.rotations)

        return collect.LayerDataDeformable(
            positions=base_layer_data.positions,
            rotations=base_layer_data.rotations,
            scales=base_layer_data.scales,
            densities=base_layer_data.densities,
            extra_signal=base_layer_data.extra_signal,
            camera_extra_signal=base_layer_data.camera_extra_signal,
            lidar_extra_signal=base_layer_data.lidar_extra_signal,
            features_albedo=base_layer_data.features_albedo,
            features_specular=base_layer_data.features_specular,
            embed_data=base_layer_data.embed_data,
            poses=base_layer_data.poses,
            keep_mask=base_layer_data.keep_mask,
            tracks_ids=base_layer_data.tracks_ids,
            deform_positions=deform_positions,
            deform_rotations=deform_rotations,
        )

    #
    # End of implementation of what is necessary for parameter collection.
    #

    def configure_optimizers(self, name_prefix: str = "") -> list[OptimizerLRSchedulerConfig]:
        """Returns a list of module-owned configured optimizers (optimizers paired with an optional LR scheduler),
        which will be stepped in the main training loop, allowing the module to interact with it's owned optimizers"""

        return super().configure_optimizers(name_prefix) + self.deform_network.configure_optimizers(name_prefix)

    def update_step_train_batch_start(self, epoch: int, global_step: int, system, **kwargs) -> dict[str, torch.Tensor]:
        if global_step >= self.config.deformnet_start_iteration:
            self.use_deform_network = True
            self.deform_network.feature_volume.encoding.update_step_train_batch_start(
                epoch, global_step - self.config.deformnet_start_iteration, system
            )

        return update_module_step(self.tracks_calib, epoch, global_step, system)

    def get_extra_state(self) -> dict[str, Any] | None:
        extra_state = super().get_extra_state()
        if extra_state is None:
            extra_state = {}
        extra_state["use_deform_network"] = self.use_deform_network
        return extra_state

    def set_extra_state(self, state: dict[str, Any] | None) -> None:
        if (state is not None) and "use_deform_network" in state:
            self.use_deform_network = state["use_deform_network"]
        super().set_extra_state(state)


class ProjectiveTransformDeformation(nn.Module):
    """
    Per-track linear deformation field.
    Each track has its own learnable transform (3x3 or 4x4) and optional per-frame translation.
    """

    def __init__(
        self,
        config: DictConfig,
        trainer_config: TrainerConfig,
        num_tracks: int,
        deform_matrix_size: int = 3,
    ) -> None:
        super().__init__()
        self.config = config
        self.trainer_config = trainer_config
        self.num_tracks = num_tracks
        self.deform_matrix_size = deform_matrix_size

        if deform_matrix_size == 3:
            self.W = torch.nn.Parameter(torch.eye(3).unsqueeze(0).repeat(num_tracks, 1, 1))
        elif deform_matrix_size == 4:
            self.W = torch.nn.Parameter(torch.eye(4).unsqueeze(0).repeat(num_tracks, 1, 1))
        else:
            raise NotImplementedError

        log.info(
            f"ProjectiveTransformDeformation: W shape: {self.W.shape}, num_tracks: {num_tracks}, deform_matrix_size: {deform_matrix_size}"
        )

    def forward(
        self,
        x: torch.Tensor,
        track_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Input coordinates (N, 3)
            track_ids: Track indices per gaussian (N,)
        Returns:
            Deformed coordinates (N, 3)
        """

        W = self.W[track_ids]

        if self.deform_matrix_size == 4:
            x_1 = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
            xw = torch.bmm(W, x_1.unsqueeze(-1)).squeeze(-1)
            denom = xw[..., [-1]].clamp_min(1e-8)
            x_deformed = (xw / denom)[..., :-1]
        elif self.deform_matrix_size == 3:
            x_deformed = torch.bmm(W, x.unsqueeze(-1)).squeeze(-1)
        else:
            raise NotImplementedError

        return x_deformed

    def configure_optimizers(self, name_prefix: str = "") -> list[OptimizerLRSchedulerConfig]:
        """Configure optimizers for the deformation field."""

        return configure_optimizers(self.config, self.trainer_config, self, name_prefix)


_VIS_FREQUENCY_PARAMETER_NAMES = frozenset(
    {
        "positions",
        "rotations",
        "densities",
        "scales",
        "features_albedo",
        "features_specular",
    }
)
_VIS_FREQUENCY_HOOK_MARKER = "_nurec_vis_frequency_drop_hook_attached"


class DeformableRigidAssetModel(RigidGaussianModel):
    """
    RigidGaussianModel with simple LinearDeformation integration.
    This prevents individual points from moving freely while maintaining spatial coherence.
    """

    config: ElasticLoadedAHLayerConfig  # type: ignore[assignment] # Narrow type from RigidGaussiansLayerConfig

    def register_mcmc_parameter_hooks(self, name: str, parameter: nn.Parameter) -> None:
        super().register_mcmc_parameter_hooks(name, parameter)
        buffers = getattr(self, "_buffers", {})
        if (
            name in _VIS_FREQUENCY_PARAMETER_NAMES
            and not isinstance(parameter, nn.UninitializedParameter)
            and parameter.requires_grad
            and getattr(
                getattr(self, "config", None),
                "restriction_policy",
                None,
            )
            == "vis_frequency_drop"
            and "gaussians_hit_counter" in buffers
            and "gaussians_hit_mask" in buffers
        ):
            self._attach_vis_frequency_hook(name, parameter)

    def __init__(
        self,
        config: ElasticLoadedAHLayerConfig,
        trainer_config: TrainerConfig,
        datasource: DataSourceSummary,
        init_from_datasource: bool,
        initializer: Optional[BaseInitialization],
        precision: Optional[int] = None,
        cuboid_tracks: Optional[CuboidTracks] = None,
        start_end_timestamp_us: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__(
            config,
            trainer_config,
            datasource,
            init_from_datasource,
            initializer,
            precision,
            cuboid_tracks,
            start_end_timestamp_us,
        )

        log.info(
            f"DeformableRigidAssetModel: initializing deformation network with num_tracks: {self.cuboid_tracks.n_tracks}, {self.cuboid_tracks.tracks_id}"
        )

        self.log_every_n_steps = self.config.log_every_n_steps or 100
        # Initialize the implicit per-track deformation field
        self.deformation_network = ProjectiveTransformDeformation(
            config=self.config.deformation_network,
            trainer_config=self.trainer_config,
            num_tracks=self.cuboid_tracks.n_tracks,
            deform_matrix_size=3 if self.config.deform_matrix_size is None else self.config.deform_matrix_size,
        )
        self.use_deformation = True
        # One-time transition logs
        self._masking_started_logged = False
        self._masking_ended_logged = False
        self._deform_started_logged = False
        self._deform_ended_logged = False
        # Resume handling
        self._resumed_from_ckpt = False
        self._first_update_done = False

        # Global compact track id mapping (raw ids -> [0..n_tracks-1]) created lazily
        self._global_track_ids_sorted: torch.Tensor | None = None  # Stored as a plain Tensor when initialized

        # Typing-only declarations to satisfy static type checkers; buffers are registered later
        # These annotations do not create runtime attributes and will not interfere with register_buffer
        self.gaussians_hit_counter: torch.Tensor  # type: ignore[assignment]
        self.gaussians_hit_mask: torch.Tensor  # type: ignore[assignment]

    def get_extra_state(self) -> dict[str, Any] | None:
        extra_state = super().get_extra_state()
        if extra_state is None:
            extra_state = {}
        extra_state["use_deformation"] = self.use_deformation
        # Persist transition flags so banners don't re-fire on resume
        extra_state["masking_started_logged"] = self._masking_started_logged
        extra_state["masking_ended_logged"] = self._masking_ended_logged
        extra_state["deform_started_logged"] = self._deform_started_logged
        extra_state["deform_ended_logged"] = self._deform_ended_logged
        return extra_state

    def set_extra_state(self, state: dict[str, Any] | None) -> None:
        if state is not None:
            if "use_deformation" in state:
                self.use_deformation = state["use_deformation"]
            # Restore flags and mark resume
            self._masking_started_logged = bool(state.get("masking_started_logged", False))
            self._masking_ended_logged = bool(state.get("masking_ended_logged", False))
            self._deform_started_logged = bool(state.get("deform_started_logged", False))
            self._deform_ended_logged = bool(state.get("deform_ended_logged", False))
            self._resumed_from_ckpt = True
        super().set_extra_state(state)

    #
    # Implementation of what is necessary for parameter collection.
    #
    def get_layer_data(
        self, context: BaseGaussianModel.CollectionContext, gathered_parameters: dict[str, torch.Tensor]
    ) -> collect.LayerDataRigid:
        layer_data = super().get_layer_data(context, gathered_parameters)
        gaussian_cuboid_ids = layer_data.tracks_ids
        if not self.use_deformation or gaussian_cuboid_ids.numel() == 0:
            # Don't apply deformation network to this layer.
            return layer_data

        with ScopedTimer("DeformableRigidAssetModel/deformation_network"):
            base_positions = layer_data.positions
            track_ids = gaussian_cuboid_ids
            with ScopedTimer("DeformableRigidAssetModel/run_network"):
                deformed_positions = self.deformation_network(
                    base_positions,
                    track_ids,
                )
            if log.isEnabledFor(logging.DEBUG):
                # Range-check track_ids. Piggy-backs on the .item() syncs
                # already paid for the debug log below, so no extra sync cost
                # when DEBUG is on and zero cost in production. Production
                # safety net is the CUDA device-side assert raised by
                # self.deformation_network's gather on OOB indices; this block
                # just surfaces a clearer message during investigation.
                if track_ids.numel() > 0:
                    max_idx = int(track_ids.max())
                    if max_idx >= self.deformation_network.num_tracks:
                        raise RuntimeError(
                            f"Deformation track_ids out of range: "
                            f"max={max_idx} >= num_tracks={self.deformation_network.num_tracks}"
                        )
                delta = (deformed_positions - base_positions).abs()
                mean_delta = delta.mean().item()
                max_delta = delta.max().item() if delta.numel() > 0 else 0.0
                n_tracks_in_batch = torch.unique(gaussian_cuboid_ids).numel()
                log.debug(
                    f"DeformableRigidAssetModel: deformation applied | mean_abs_delta={mean_delta:.6f} "
                    f"max_abs_delta={max_delta:.6f} "
                    f"tracks_in_batch={n_tracks_in_batch}"
                )

            # Adding cast to float32 to avoid type mismatch errors in the collector
            # collector expects float32 tensors
            # But torch.bmm may return float16 tensors
            layer_data.positions = deformed_positions.to(torch.float32)

        return layer_data

    #
    # End of implementation of what is necessary for parameter collection.
    #

    def configure_optimizers(self, name_prefix: str = "") -> list[OptimizerLRSchedulerConfig]:
        """Configure optimizers for both base model and deformation field."""

        log.info(f"[DeformableRigidAssetModel] Configuring optimizers (name_prefix='{name_prefix}')")

        # Get base optimizers
        base_optimizers = super().configure_optimizers(name_prefix)
        log.info(f"[DeformableRigidAssetModel] Got {len(base_optimizers)} base optimizers from parent class")

        # Use the utility function to create the proper optimizer configuration
        log.info("[DeformableRigidAssetModel] Configuring deformation network optimizers...")
        deformation_optimizers = self.deformation_network.configure_optimizers(name_prefix=name_prefix)

        log.info(f"[DeformableRigidAssetModel] Got {len(deformation_optimizers)} deformation network optimizers")

        # Combine base optimizers with deformation optimizers
        base_optimizers.extend(deformation_optimizers)
        log.info(f"[DeformableRigidAssetModel] Total optimizers: {len(base_optimizers)}")

        # Print detailed summary of all optimizers and their parameter groups
        log.info(f"\n{'=' * 80}")
        log.info("[DeformableRigidAssetModel] Optimizer Configuration Summary")
        log.info(f"{'=' * 80}")
        for idx, opt_config in enumerate(base_optimizers):
            optimizer = opt_config["optimizer"]
            has_scheduler = "lr_scheduler" in opt_config
            scheduler_name = opt_config["lr_scheduler"]["scheduler"].__class__.__name__ if has_scheduler else "None"
            log.info(f"\nOptimizer {idx}: {optimizer.__class__.__name__} (scheduler: {scheduler_name})")
            for pg_idx, param_group in enumerate(optimizer.param_groups):
                pg_name = param_group.get("name", f"group_{pg_idx}")
                lr = param_group["lr"]
                num_params = len(param_group["params"])
                log.info(f"  [{pg_name}] initial_lr={lr:.6f}, num_params={num_params}")
        log.info(f"{'=' * 80}\n")

        return base_optimizers

    def get_additional_buffers(self) -> dict[str, torch.Tensor]:
        additional_buffers = super().get_additional_buffers()
        # Persist masking buffers only for ElasticLoadedAH
        if hasattr(self, "gaussians_hit_counter") and isinstance(self.gaussians_hit_counter, torch.Tensor):
            additional_buffers["gaussians_hit_counter"] = self.gaussians_hit_counter
        if hasattr(self, "gaussians_hit_mask") and isinstance(self.gaussians_hit_mask, torch.Tensor):
            additional_buffers["gaussians_hit_mask"] = self.gaussians_hit_mask
        return additional_buffers

    def _reinstall_grad_hooks(self) -> None:
        # setup_grad_hooks reads the hit buffers through self, so the hooks it installs pick up the permuted ones.
        # It skips buffer registration when they already exist.
        if self.config.restriction_policy == "vis_frequency_drop":
            self.setup_grad_hooks()

    # Handle masking buffers on checkpoint load so they are not treated as unexpected keys
    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        policy = self.config.restriction_policy
        counter_key = prefix + "gaussians_hit_counter"
        mask_key = prefix + "gaussians_hit_mask"

        has_counter = counter_key in state_dict
        has_mask = mask_key in state_dict

        if policy == "vis_frequency_drop":
            # Require presence for resume; register buffers so assignment succeeds
            if not has_counter or not has_mask:
                # Let super report missing if strict, but give clearer error
                raise RuntimeError(
                    f"Checkpoint is missing required masking buffers: present(counter={has_counter}, mask={has_mask})."
                )
            if "gaussians_hit_counter" not in self._buffers:
                self.register_buffer(
                    "gaussians_hit_counter",
                    state_dict[counter_key].to(torch.long),
                    persistent=True,
                )
            if "gaussians_hit_mask" not in self._buffers:
                val = state_dict[mask_key]
                val_bool = val if val.dtype == torch.bool else (val > 0)
                self.register_buffer(
                    "gaussians_hit_mask",
                    val_bool.to(torch.bool),
                    persistent=True,
                )
        else:
            raise NotImplementedError(f"Restriction policy {policy} not implemented")

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

        # Re-attach autograd hooks after loading (hooks are not serialized in checkpoints)
        if policy == "vis_frequency_drop":
            try:
                self.setup_grad_hooks()
            except Exception:
                # If this fails, better to raise to avoid silent masking stalls
                raise

    def update_step_train_batch_start(self, epoch: int, global_step: int, system, **kwargs) -> dict[str, torch.Tensor]:
        # One-time resume-aware initialization for transition banners
        if not self._first_update_done:
            self._first_update_done = True
            if self._resumed_from_ckpt:
                # If resuming after masking end, suppress end banner
                if self.config.restriction_policy == "vis_frequency_drop":
                    masking_end_step = int(self.config.masking_end_step or 3000)
                    if global_step >= masking_end_step:
                        self._masking_started_logged = True
                        self._masking_ended_logged = True
                # Always keep deformation enabled across resumes
                self._deform_started_logged = True
                self._deform_ended_logged = False
                self.use_deformation = True
        # Log learning rates every 100 steps for the first 1000 steps, then every 1000 steps
        if global_step % self.log_every_n_steps == 0:
            log.debug(f"\n{'=' * 80}")
            log.debug(f"[DeformableRigidAssetModel] Learning Rates at Step {global_step}")
            log.debug(f"{'=' * 80}")
            for idx, opt_config in enumerate(self.optimizers):
                optimizer = opt_config["optimizer"]
                opt_name = f"Optimizer {idx}"
                log.debug(f"\n  {opt_name}:")
                for pg_idx, param_group in enumerate(optimizer.param_groups):
                    pg_name = param_group.get("name", f"group_{pg_idx}")
                    lr = param_group["lr"]
                    num_params = len(param_group["params"])
                    log.debug(f"    [{pg_name}] lr={lr:.6f}, num_params={num_params}")
            log.debug(f"{'=' * 80}\n")

        # Track masking start/end events (vis_frequency_drop policy)
        if self.config.restriction_policy == "vis_frequency_drop":
            masking_end_step = int(self.config.masking_end_step or 3000)
            if not self._masking_started_logged and global_step == 0:
                self._masking_started_logged = True
                log.debug(
                    f"[DeformableRigidAssetModel] ✅ Masking STARTED at step 0 (policy=vis_frequency_drop, will end at step {masking_end_step})"
                )
            if (not self._masking_ended_logged) and (global_step >= masking_end_step):
                self._masking_ended_logged = True
                # Optional quick stats snapshot
                try:
                    n = int(self.gaussians_hit_mask.numel()) if hasattr(self, "gaussians_hit_mask") else -1
                    masked = int(self.gaussians_hit_mask.sum().item()) if hasattr(self, "gaussians_hit_mask") else -1
                    log.debug(
                        f"[DeformableRigidAssetModel] 🛑 Masking ENDED at step {global_step} (n={n}, masked={masked})"
                    )
                except Exception:
                    log.debug(f"[DeformableRigidAssetModel] 🛑 Masking ENDED at step {global_step}")

        # Ensure deformation is always enabled and announce once if not already
        if not self._deform_started_logged:
            self._deform_started_logged = True
            self._deform_ended_logged = False
            self.use_deformation = True
            try:
                W = self.deformation_network.W
                log.debug(
                    f"[DeformableRigidAssetModel] ✅ Deformation ENABLED (always-on) at step {global_step} "
                    f"(W shape={tuple(W.shape)}, min={float(W.min()):.4f}, max={float(W.max()):.4f})"
                )
            except Exception:
                log.info(f"[DeformableRigidAssetModel] Deformation network failed to initialize at step {global_step}")

        if self.config.restriction_policy == "vis_frequency_drop":
            # Track current step for gated logging in hooks
            self._current_step = int(global_step)
            # Log mask stats at the first step after a resume (global_step may reset depending on restore)
            if global_step == 0:
                try:
                    n = int(self.gaussians_hit_mask.numel())
                    active = int((~self.gaussians_hit_mask).sum().item())
                    masked = int(self.gaussians_hit_mask.sum().item())
                    if hasattr(self, "gaussians_hit_counter"):
                        hit_sum = int(self.gaussians_hit_counter.sum().item())
                        hit_min = int(self.gaussians_hit_counter.min().item())
                        hit_max = int(self.gaussians_hit_counter.max().item())
                    else:
                        hit_sum, hit_min, hit_max = -1, -1, -1
                    log.info(
                        f"[MaskResumeStats][before-step] n={n}, masked={masked}, active={active}, "
                        f"hit_counter_sum={hit_sum}, hit_counter_min={hit_min}, hit_counter_max={hit_max}"
                    )
                except Exception:
                    raise RuntimeError("Error in logging mask stats at global step 0")
            # Use masking_end_step for determining when to stop updating the hit mask
            # This should match when albedo starts training (step 3000 by default)
            masking_end_step = getattr(self.config, "masking_end_step", 3000)
            if global_step > 0 and global_step < masking_end_step:
                if (global_step % self.log_every_n_steps) == 0:
                    denom = float(max(1, int(self.gaussians_hit_counter.numel())))
                    stepf = float(max(1, int(global_step)))
                    log.debug(
                        f"[DeformableRigidAssetModel] gaussian hit frequency: {self.gaussians_hit_counter.sum() / denom / stepf}"
                    )
                # Create a boolean mask tensor using explicit Tensor ops for mypy
                self.gaussians_hit_mask = torch.lt(
                    self.gaussians_hit_counter.to(torch.float32) / float(global_step), 0.2
                )

                if (global_step % self.log_every_n_steps) == 0:
                    try:
                        n = int(self.gaussians_hit_mask.numel())
                        active = int((~self.gaussians_hit_mask).sum().item())
                        masked = int(self.gaussians_hit_mask.sum().item())
                        hit_sum = int(self.gaussians_hit_counter.sum().item())
                        hit_min = int(self.gaussians_hit_counter.min().item())
                        hit_max = int(self.gaussians_hit_counter.max().item())
                        log.debug(
                            f"[MaskResumeStats][after-update] step={global_step} n={n}, masked={masked}, active={active}, "
                            f"hit_counter_sum={hit_sum}, hit_counter_min={hit_min}, hit_counter_max={hit_max}"
                        )
                    except Exception:
                        raise RuntimeError(f"Error in logging mask stats at global step {global_step}")
        else:
            raise NotImplementedError(
                f"Restriction policy {self.config.restriction_policy or 'hardcoded_masking'} not implemented"
            )

        return update_module_step(self.tracks_calib, epoch, global_step, system)

    def _attach_vis_frequency_hook(
        self,
        name: str,
        parameter: torch.nn.Parameter,
    ) -> None:
        if getattr(parameter, _VIS_FREQUENCY_HOOK_MARKER, False):
            return

        hook = self._albedo_grad_hook if name == "features_albedo" else self._freeze_gaussian_grad_hook
        parameter.register_hook(hook)
        setattr(parameter, _VIS_FREQUENCY_HOOK_MARKER, True)

    def _albedo_grad_hook(self, grad: torch.Tensor) -> torch.Tensor:
        grad_norm = torch.linalg.norm(grad[:, 0], axis=1)
        self.gaussians_hit_counter[grad_norm > 4e-7] += 1

        if hasattr(self, "_current_step") and (self._current_step % self.log_every_n_steps) == 0:
            log.debug(f"[DeformableRigidAssetModel] grad_norm quantile 0.95: {torch.quantile(grad_norm, q=0.95)}")
            log.debug(f"[DeformableRigidAssetModel] grad_norm quantile 0.9: {torch.quantile(grad_norm, q=0.9)}")
            log.debug(f"[DeformableRigidAssetModel] grad_norm quantile 0.8: {torch.quantile(grad_norm, q=0.8)}")
            log.debug(f"[DeformableRigidAssetModel] grad_norm quantile 0.5: {torch.quantile(grad_norm, q=0.5)}")
            log.debug(
                f"[DeformableRigidAssetModel] gaussians_hit_mask sum: {self.gaussians_hit_mask.sum() / len(grad_norm)}"
            )

        grad[self.gaussians_hit_mask] = 0.0
        return grad

    def _freeze_gaussian_grad_hook(self, grad: torch.Tensor) -> torch.Tensor:
        grad[self.gaussians_hit_mask] = 0.0
        return grad

    def setup_grad_hooks(self) -> None:
        policy = self.config.restriction_policy
        if policy == "vis_frequency_drop":
            positions = self.get_positions()
            num_gaussians, _ = positions.shape
            # Register persistent buffers once during initialization
            if "gaussians_hit_counter" not in self._buffers:
                self.register_buffer(
                    "gaussians_hit_counter",
                    torch.zeros((num_gaussians,), dtype=torch.long, device=self.device),
                    persistent=True,
                )
            if "gaussians_hit_mask" not in self._buffers:
                self.register_buffer(
                    "gaussians_hit_mask",
                    torch.ones((num_gaussians,), dtype=torch.bool, device=self.device),
                    persistent=True,
                )

            for name in _VIS_FREQUENCY_PARAMETER_NAMES:
                self.register_mcmc_parameter_hooks(name, getattr(self, name))
        else:
            raise NotImplementedError(f"Restriction policy {policy} not implemented")

    def initialize_gaussians_from_datasource(
        self,
        initializer: BaseInitialization,
        datasource: DataSourceSummary,
        cuboid_tracks: Optional[CuboidTracks] = None,
    ) -> None:
        super().initialize_gaussians_from_datasource(
            initializer=initializer,
            datasource=datasource,
            cuboid_tracks=cuboid_tracks,
        )

        self.setup_grad_hooks()


BaseGaussianModel.register_to_gaussians_factory("sh-gaussians", SHGaussianModel)
BaseGaussianModel.register_to_gaussians_factory("rigid-gaussians", RigidGaussianModel)
BaseGaussianModel.register_to_gaussians_factory("deformable-gaussians", DeformableGaussianModel)
BaseGaussianModel.register_to_gaussians_factory("elastic-loaded-ah", DeformableRigidAssetModel)
