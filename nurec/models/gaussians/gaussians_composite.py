# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import logging
import math
import threading

from collections import Counter, defaultdict
from dataclasses import dataclass, fields
from itertools import accumulate
from pathlib import Path
from typing import Any, Callable, ClassVar, Collection, List, Mapping, Optional, Sequence, cast

import lietorch as lt
import numpy as np
import torch

from omegaconf import DictConfig, ListConfig, OmegaConf, read_write

import nurec.models.gaussians.collect as collect

from nurec.config.component_only import (
    BACKGROUND_LAYER_ID,
    ComponentOnlyConfig,
    ComponentState,
    MaskSource,
    ScopeRegion,
    _entry_has_non_disabled_population,
)
from nurec.config.model import ActiveTileMaskStrategy, ModelConfig, SkipBackgroundConfig
from nurec.config.trainer import TrainerConfig
from nurec.datasets.summary import DataSourceSummary
from nurec.datasets.tracks import CuboidTracks
from nurec.models.background import (
    BaseBackground,
    SkyEnvMapBackground,
    SkyMlpBackground,
    below_horizon_weight,
    erode_mask,
    swap_sky_for_ground,
)
from nurec.models.base import BaseModel
from nurec.models.calib import BaseCalib, FreePoseCalib
from nurec.models.composite import CompositeModel, LayerTrackIds
from nurec.models.gaussians.edit_hook import CollectedLayerLayout, GaussianEditOverrides
from nurec.models.gaussians.fused_parameter_gather import (
    _can_prefetch_grouped_payload,
    _can_route_node_selection,
    _discard_fused_payload_gather,
    _discard_layer_count_gather,
    _finish_layer_count_gather,
    _fused_all_gather_gaussian_parameters,
    _FusedGatherGenerationPlan,
    _make_fused_gather_generation_plan,
    _PayloadIntegerRangeValidationState,
    _PendingFusedPayloadGather,
    _PendingLayerCountGather,
    _prepare_fused_gather_inputs,
    _prepare_fused_gather_inputs_from_generation_plan,
    _PreparedFusedGatherInputs,
    _resolve_fused_gather_generation_plan,
    _start_fused_payload_gather_from_generation_plan,
    _start_layer_count_gather,
)
from nurec.models.gaussians.gaussians_model import (
    BaseGaussianModel,
    GaussianExportFormat,
    PreparedRigidAssetInsertion,
    PreparedRigidMotion,
    RigidGaussianModel,
    SHGaussianModel,
    distributed_all_gather_gaussian_parameters,
)
from nurec.models.gaussians.initializations import BaseInitialization, NoPointsFoundException
from nurec.models.gaussians.renderers import BaseGaussianRenderer, GSplatRenderer
from nurec.models.gaussians.segment_plan import (
    LayerSegmentTable,
    SegmentPlan,
    assert_sparse_segments_supported,
    collect_ranges,
    rows_grouped_by_track,
    upload_segments,
    whole_layer_ranges,
)
from nurec.models.gaussians.strategies import (
    BaseGaussianStrategy,  # Note: imported from strategies/__init__.py, not base.py, so that all strategies get imported.
    ECADStrategy,
    GSplatStrategy,
    MCMCStrategy,
)
from nurec.models.gaussians.utils import SH2RGB, Asset, RGBPostProcessing
from nurec.models.nn_extensions import TypedModuleDict, TypedModuleList
from nurec.models.post_processing import BasePostProcessing, PPISPPostProcessing
from nurec.models.tracks_calib import SkipTracksCalib
from nurec.models.utils import model_config_compatibility_check, update_module_step
from nurec.utils.batch import (
    DataAndRenderingBatch,
    FrameMeta,
    RectSubsampled,
    RenderingData,
)
from nurec.utils.component_only_mask import (
    DISABLED_TRACKS_SUFFIX,
    ComponentOnlyMaskSpec,
    LayerRegionSpec,
    ProjectedTrackRegion,
    build_mask_spec,
)
from nurec.utils.geometry import se3_matrix_to_tquat
from nurec.utils.misc import all_gather_int32, unpack_optional
from nurec.utils.morton import VOXEL_SIZE
from nurec.utils.optim import (
    OptimizerLRSchedulerConfig,
    SelectiveAdam,
    classify_gaussian_optimizer_group,
)
from nurec.utils.profiling import ScopedTimer
from nurec.utils.selective_adam_visibility import prepare_async_visibility_union
from nurec.utils.tile_mask import (
    CameraToLidarTileMapping,
    camera_tile_mask_to_lidar_mask,
    compute_active_tile_mask,
    mark_inactive_lidar_tile_rays_inplace,
)
from nurec.utils.types import (
    AABB3D,
    ExtraSignal,
    GaussiansCompositeReturn,
    GaussiansRenderReturn,
    RayFlags,
    SceneContractor,
)
from nurec.visualdebugger import get_visualdebugger


log = logging.getLogger(__name__)

_MCMC_VISIBILITY_STATE_MARKER = "gaussians_strategy.invisible_steps."


def reset_incompatible_mcmc_visibility_state(
    state_dict: dict[str, Any],
    *,
    warning: Callable[..., Any] | None,
    prefix: str | None = None,
) -> None:
    """Reset legacy rank-local visibility state whose paired positions are global."""
    marker = f"{prefix}{_MCMC_VISIBILITY_STATE_MARKER}" if prefix is not None else _MCMC_VISIBILITY_STATE_MARKER
    for counter_name, counter in state_dict.items():
        if (
            (not counter_name.startswith(marker) if prefix is not None else marker not in counter_name)
            or not isinstance(counter, torch.Tensor)
            or isinstance(counter, torch.nn.UninitializedParameter)
        ):
            continue

        if prefix is None:
            state_prefix, layer_id = counter_name.split(marker, maxsplit=1)
        else:
            state_prefix, layer_id = prefix, counter_name.removeprefix(marker)
        positions = state_dict.get(f"{state_prefix}gaussians_nodes.{layer_id}.positions")
        if (
            not isinstance(positions, torch.Tensor)
            or isinstance(positions, torch.nn.UninitializedParameter)
            or (counter.ndim == 1 and counter.shape[0] == positions.shape[0])
        ):
            continue

        if warning is not None:
            warning(
                "Resetting incompatible MCMC visibility counter '%s': checkpoint has shape %s but paired "
                "positions has %d rows. Legacy multi-GPU checkpoints may contain rank-local history, which "
                "cannot be reconstructed",
                counter_name,
                tuple(counter.shape),
                positions.shape[0],
            )
        state_dict[counter_name] = torch.zeros(
            (positions.shape[0],),
            dtype=counter.dtype,
            device=counter.device,
        )


@dataclass(frozen=True, slots=True)
class _GatheredGaussianParameterSet:
    """One layer's shared payload and optional same-forward count metadata."""

    parameters: dict[str, torch.Tensor]
    local_count: int
    counts_by_rank: Optional[tuple[int, ...]] = None


GatheredGaussianParameters = dict[str, _GatheredGaussianParameterSet]


@dataclass(frozen=True, slots=True)
class _PendingGaussianParameterGather:
    """Fresh inputs for warmup or a generation payload already in flight."""

    gaussian_nodes: tuple[str, ...]
    parameters_by_node: dict[str, Mapping[str, torch.Tensor]]
    prepared_inputs: _PreparedFusedGatherInputs
    generation_plan: Optional[_FusedGatherGenerationPlan] = None
    prefetched_payload_gather: Optional[_PendingFusedPayloadGather] = None
    publish_on_success: bool = False

    def __post_init__(self) -> None:
        if (self.generation_plan is None) != (self.prefetched_payload_gather is None):
            raise ValueError("A prefetched Gaussian payload requires its generation plan")
        if self.generation_plan is None and not self.publish_on_success:
            raise ValueError("A payload-free Gaussian gather must initialize a generation")

    def discard(self) -> None:
        """Drain unused payload communication; repeated calls are harmless."""
        if self.prefetched_payload_gather is not None:
            _discard_fused_payload_gather(self.prefetched_payload_gather)


@dataclass(frozen=True, slots=True)
class _PendingGaussianGenerationPlanRefresh:
    """Post-mutation metadata packet retained until the next forward."""

    prepared_inputs: _PreparedFusedGatherInputs
    count_gather: _PendingLayerCountGather


def uses_selective_adam(config: Any) -> bool:
    """Return True if any Gaussian optimizer group is configured to use SelectiveAdam.

    Reads the three optimizer-name fields (general / density / scale) so that
    ``fused_adam``-only configurations can skip SelectiveAdam-specific renderer
    outputs and validation instead of paying for them unconditionally.
    """
    return any(
        getattr(config, name, "fused_adam") == "selective_adam"
        for name in (
            "gaussian_optimizer_name",
            "gaussian_density_optimizer_name",
            "gaussian_scale_optimizer_name",
        )
    )


def ensure_selective_adam_renderer_outputs(renderer_cfg: Any) -> None:
    """Force renderer scene outputs required for SelectiveAdam visibility masks.

    Handles every shape ``renderer_cfg.outputs`` can take across the config paths:
      * ``None`` — create the scene mapping from scratch.
      * raw ``dict`` — mutate the keys in place.
      * structured ``DictConfig`` — create/update the full path while temporarily
        relaxing read-only mode; struct and read-only flags are restored afterward.
      * typed Pydantic config — set the attributes directly.
    """
    if isinstance(renderer_cfg, DictConfig):
        with read_write(renderer_cfg):
            OmegaConf.update(
                renderer_cfg,
                "outputs.scene.enable_cumulated_weights",
                True,
                force_add=True,
            )
            OmegaConf.update(renderer_cfg, "outputs.scene.enable_visibility", True, force_add=True)
        return

    outputs = renderer_cfg.outputs
    if isinstance(outputs, DictConfig):
        with read_write(outputs):
            OmegaConf.update(outputs, "scene.enable_cumulated_weights", True, force_add=True)
            OmegaConf.update(outputs, "scene.enable_visibility", True, force_add=True)
        return

    if outputs is None:
        renderer_cfg.outputs = {"scene": {"enable_cumulated_weights": True, "enable_visibility": True}}
        return

    scene = outputs.get("scene", {}) if isinstance(outputs, dict) else getattr(outputs, "scene", {})
    if isinstance(scene, dict):
        scene["enable_cumulated_weights"] = True
        scene["enable_visibility"] = True
        if isinstance(outputs, dict):
            outputs["scene"] = scene
    else:
        scene.enable_cumulated_weights = True
        scene.enable_visibility = True


def assert_selective_adam_occlusion_supported(renderer_cfg: Any, selective_adam_occlusion_aware: bool) -> None:
    """Reject SelectiveAdam occlusion mode for renderers that cannot provide cumulated weights."""
    if not selective_adam_occlusion_aware:
        return

    renderer_name = getattr(renderer_cfg, "name", "")
    if isinstance(renderer_name, str) and renderer_name.endswith("-gsplat"):
        raise AssertionError(
            "model.selective_adam_occlusion_aware=true is not supported with GSplat renderers "
            f"(model.renderer.name={renderer_name!r}) because GSplat does not populate "
            "GaussiansRenderReturn.cumulated_weights. Set model.selective_adam_occlusion_aware=false, "
            "since no currently supported renderer populates cumulated_weights."
        )


def configure_selective_adam_renderer(config: Any) -> None:
    """Apply SelectiveAdam-only renderer validation and output configuration."""
    if not uses_selective_adam(config):
        return

    assert_selective_adam_occlusion_supported(
        config.renderer,
        config.selective_adam_occlusion_aware,
    )
    ensure_selective_adam_renderer_outputs(config.renderer)


@dataclass(frozen=True)
class _TrackPartition:
    """Resolved per-track effective states and box-variant flags for a layer.

    Returned by ``_resolve_component_only_track_partition`` so the resolver
    stays pure; the caller stores it into the composite's per-layer maps.
    """

    states: dict[str, ComponentState]  # track id -> effective state
    expanded: dict[str, bool]  # track id -> uses the expanded box variant


@dataclass(frozen=True)
class _AssetInsertionPlan:
    """One validated layer update for an external asset insertion."""

    layer_id: str
    gaussian_node: RigidGaussianModel
    layer_cuboid_tracks: CuboidTracks
    obj_track_ids: LayerTrackIds


@dataclass(frozen=True)
class _PreparedAssetInsertionPlan:
    """One layer's fully staged state for a scene-wide asset insertion."""

    plan: _AssetInsertionPlan
    insertion: PreparedRigidAssetInsertion
    collector_time_table: torch.Tensor | None


def _make_row_grad_hook(trainable_rows: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    """Gradient hook zeroing rows where ``trainable_rows`` is False.

    Used for sub-layer freezing (frozen scope regions / frozen tracks):
    a single nn.Parameter cannot carry per-row requires_grad, so the
    frozen partition's gradient rows are zeroed instead.
    """

    def hook(grad: torch.Tensor) -> torch.Tensor:
        keep = trainable_rows.to(grad.device)
        return grad * keep.view(-1, *([1] * (grad.ndim - 1))).to(grad.dtype)

    return hook


class GaussiansComposite(BaseModel, CompositeModel):
    # Keep the lazy-initialization lock out of instance state: render-context shells
    # shallow-clone that state, and deepcopy/pickle cannot copy a threading.Lock.
    # Contention is limited to the first fallback-stream creation for each composite;
    # serving passes a request-owned stream and never acquires this lock.
    _lidar_stream_init_lock: ClassVar[threading.Lock] = threading.Lock()

    # The fused kernel stores one interval per submitted track. Bound both the
    # submitted track count and the ray count so its dense outputs plus the
    # reduction intermediates stay below this conservative temporary-payload
    # budget even for production-resolution camera batches. The final
    # (N_rays,) hit/depth outputs are unavoidable and intentionally excluded.
    _COMPONENT_ONLY_INTERSECTION_TRACK_CHUNK_SIZE = 8
    _COMPONENT_ONLY_INTERSECTION_TEMP_MEMORY_BUDGET_BYTES = 64 * 1024 * 1024

    config: ModelConfig  # type: ignore[assignment] # Override type annotation from BaseModel

    calib: BaseCalib
    background: BaseBackground
    gaussians_renderer_lidar: BaseGaussianRenderer
    gaussians_renderer_camera: BaseGaussianRenderer
    gaussians_nodes: TypedModuleDict[BaseGaussianModel]
    post_processings: TypedModuleList[BasePostProcessing]

    _scene_contractor: SceneContractor
    scene_extent: float

    last_step: int = 0
    debug_viz: bool
    saturate_radiance: bool

    # One-shot latches for the active-tile-mask observability logs. Declared at class level so
    # they hold their "not yet logged" default even on instances built without ``__init__``
    # (test doubles, ``__new__``-constructed stubs); the first log shadows them per instance.
    _active_tiling_enabled_logged: bool = False
    _active_tiling_camera_fraction_logged: bool = False
    _active_tiling_lidar_fraction_logged: bool = False
    _active_tiling_no_ray_flags_warned: bool = False
    _active_tiling_no_tile_size_warned: bool = False

    # Optional render-time hook that mutates collected Gaussian parameters
    # (toggles baked asset-edit clusters on/off per frame). Installed by the
    # render CLI / gRPC server; left as None for training and standard inference.
    _edit_overrides: Optional[GaussianEditOverrides] = None

    # Whether _ensure_tracked_rows_grouped has run for the rows currently loaded.
    # Class-level, like the latches above, so instances built without __init__ have it.
    _tracked_rows_grouped: bool = False

    # Set on a per-request shell by context_composite._reset_per_context_state. A shell
    # shares its parameters with the model it was cloned from, so it must not run a
    # mutator; the one that would reach it from the render path asserts on this.
    _is_context_shell: bool = False

    # sensor_name -> layer_id -> where that layer landed in that sensor's collected
    # tensors. Written by collect_gaussian_parameters, consumed (popped) by
    # _render_collected; see the write site for why it is keyed by sensor. Declared at
    # class level, like the latches above, so instances built without ``__init__`` (test
    # doubles, ``__new__``-constructed stubs) read it safely — the write REBINDS rather
    # than mutating, so the shared class-level default always stays empty.
    _collected_edit_layouts: dict[str, dict[str, CollectedLayerLayout]] = {}

    def __init__(
        self,
        config: ModelConfig,
        trainer_config: TrainerConfig,
        datasource: DataSourceSummary,
        init_from_datasource: bool,
        component_only: Optional[ComponentOnlyConfig] = None,
    ):
        BaseModel.__init__(self, config.to_dictconfig())
        self.register_load_state_dict_pre_hook(self._invalidate_fused_gather_generation_plan_before_load)
        self.register_load_state_dict_pre_hook(self._restore_initialization_skipped_layers)
        self.register_load_state_dict_pre_hook(self._reset_incompatible_mcmc_visibility_state_on_load)
        self.register_load_state_dict_post_hook(self._refresh_mcmc_runtime_state_after_load)
        self.register_load_state_dict_post_hook(self._invalidate_derived_render_caches_after_load)
        self.register_state_dict_pre_hook(self._raise_if_mcmc_publication_failed_before_state_dict)
        self.config = config  # type: ignore[assignment] # Override with typed config
        self.trainer_config = trainer_config
        self.gaussian_process_group: Optional[torch.distributed.ProcessGroup] = None
        self._fused_gather_generation_plan: Optional[_FusedGatherGenerationPlan] = None
        self._pending_fused_gather_generation_plan_refresh: Optional[_PendingGaussianGenerationPlanRefresh] = None

        # Non-None iff component-only mode is active for this run. The
        # config itself never modifies self.config — model.* keeps
        # describing what components ARE; component_only decides what
        # exists in this run.
        self.component_only: Optional[ComponentOnlyConfig] = (
            component_only if component_only is not None and component_only.enabled else None
        )

        log.info(f"{self.__class__.__name__}/__init__")

        # Check that the selected modules are compatible before building a model
        model_config_compatibility_check(self.config)
        configure_selective_adam_renderer(self.config)

        self.scene_extent = datasource.get_aabb().get_extent().max().item()
        log.info(f"scene_extent={self.scene_extent:.3f} m (max AABB axis extent)")

        calib = unpack_optional(self.config.calib)
        if self._component_state("calib") == ComponentState.disabled:
            # Disabled = absent: swap in the pass-through no-op calib AND clear
            # both sensor enabled flags. ``skip-calib`` alone still builds a
            # trainable ``FreePoseCalib`` when trajectories exist (its enabled
            # state is ``camera.enabled or lidar.enabled``); clearing both flags
            # reaches the true disabled boundary so no pose params are optimized.
            calib = calib.model_copy(
                update={
                    "name": "skip-calib",
                    "camera": calib.camera.model_copy(update={"enabled": False}),
                    "lidar": calib.lidar.model_copy(update={"enabled": False}),
                }
            )
        self.calib = BaseCalib.factory(calib.name, calib, trainer_config, datasource)

        background = unpack_optional(self.config.background)
        if self._component_state("sky_model") == ComponentState.disabled:
            # Disabled = absent: there is no sky to render.
            background = SkipBackgroundConfig(name="skip-background")
        self.background = BaseBackground.factory(background.name, background, trainer_config)

        # initialize cuboid tracks
        if (cuboid_tracks := datasource.get_cuboid_tracks(dynamic_only=False)) is not None:
            # Clean track IDs to remove "@<source>" suffixes for consistency
            self.cuboid_tracks = CuboidTracks.Ops.clean_track_ids(cuboid_tracks, DataSourceSummary._clean_track_id_str)
        else:
            self.cuboid_tracks = CuboidTracks.Factory.empty()

        gaussians_nodes: dict[str, BaseGaussianModel] = {}
        gaussian_template_node_model: Optional[BaseGaussianModel] = None
        constructor_skipped_nodes: set[str] = set()
        self._initialization_skipped_layer_ids: set[str] = set()
        self.obj_track_ids: dict[str, LayerTrackIds] = {}
        # Per-layer cuboid track subsets for the component-only bbox
        # masks. Populated for ALL declared layers — the loss mask must
        # carve DISABLED layers' box regions out of the residual, so
        # their tracks are needed even though no Gaussian node exists.
        self._component_only_layer_tracks: dict[str, CuboidTracks] = {}
        # Per-track subsetting: effective state and box variant per
        # track id, for layers with component_only per-track overrides.
        # Unlisted tracks take the layer entry's defaults.
        self._component_only_track_states: dict[str, dict[str, ComponentState]] = {}
        self._component_only_track_expanded: dict[str, dict[str, bool]] = {}
        # Cleaned track ids rendered by at least one non-disabled layer. A
        # physical actor may satisfy more than one layer selector; a disabled
        # occurrence must not occlude the same actor rendered by another layer.
        self._component_only_globally_rendered_track_ids: frozenset[str] = frozenset()
        # Geometry partitions are config/static. Cache both their stable index
        # signatures and packed CuboidTracks chunks so the per-camera path does
        # not rebuild tensors or synchronize indices back to the host.
        self._component_only_bbox_groups_cache: dict[
            tuple[str, LayerRegionSpec, bool], list[tuple[torch.Tensor, float]]
        ] = {}
        self._component_only_bbox_group_indices_cache: dict[
            tuple[str, LayerRegionSpec, bool], tuple[tuple[int, ...], ...]
        ] = {}
        self._component_only_intersection_chunk_cache: dict[tuple[int, tuple[int, ...]], tuple[CuboidTracks, ...]] = {}

        rig_trajectories = datasource.get_rig_trajectories()

        # Viewpoint samples for background's bbox scope region (see
        # _apply_component_only_background_scope): rig positions +
        # timestamps subsampled along the trajectory. Cheap to keep
        # around; empty when no trajectories exist.
        self._component_only_viewpoints: tuple[torch.Tensor, torch.Tensor] = (
            torch.zeros((0, 3)),
            torch.zeros((0,), dtype=torch.int64),
        )
        if rig_trajectories is not None and len(rig_trajectories.rig_trajectories) > 0:
            positions = torch.cat([t.T_rig_worlds[:, :3, 3] for t in rig_trajectories.rig_trajectories])
            # On the standard NCOREDataSource path these rig origins are raw
            # NCore-world coordinates, but _background_bbox_region_rows combines
            # them with NuRec-frame cuboid tracks and Gaussian positions — convert
            # world -> NuRec first. ``world_to_nre`` is identity on the instant_nurec path, so
            # this is a no-op there (no double transform).
            positions = torch.as_tensor(
                rig_trajectories.world_to_nurec.transform_points(positions.cpu().numpy()),
                dtype=positions.dtype,
                device=positions.device,
            ).reshape(-1, 3)
            timestamps = torch.cat([t.T_rig_world_timestamps_us for t in rig_trajectories.rig_trajectories])
            max_viewpoints = 32
            stride = max(1, positions.shape[0] // max_viewpoints)
            self._component_only_viewpoints = (positions[::stride], timestamps[::stride])

        for layer_id, layer_config in self.config.layers.items():
            layer_dictconfig = layer_config.to_dictconfig()
            tracks_config = cast(Optional[DictConfig], layer_dictconfig.get("tracks"))
            obj_track_id = LayerTrackIds(config=tracks_config or DictConfig({}))

            obj_track_id.initialize_from_tracks(self.cuboid_tracks)
            self.obj_track_ids[layer_id] = obj_track_id

            cuboid_subset = obj_track_id.get_layer_tracks(self.cuboid_tracks)
            # Validate explicit component_only track-id overrides against the
            # available subset BEFORE the empty-subset early return — otherwise a
            # misspelled id is silently ignored when the layer has no matching
            # dataset tracks. The resolver raises on unknown ids; its result is
            # reused for the per-track partition below.
            partition = self._resolve_component_only_track_partition(layer_id, cuboid_subset)

            # A deformable layer's tracks share ONE deformation network, so an
            # enabled track's updates would drift a frozen track's output —
            # per-track strict-freeze is impossible with a shared network. Reject
            # a mix of enabled and frozen tracks (whole-layer freeze/enable is
            # fine; disable — not freeze — the tracks you want out).
            if partition is not None and layer_config.name == "deformable-gaussians":
                non_disabled = {s for s in partition.states.values() if s != ComponentState.disabled}
                if ComponentState.enabled in non_disabled and ComponentState.frozen in non_disabled:
                    raise ValueError(
                        f"component_only.layers.{layer_id}: mixed enabled/frozen per-track states are "
                        f"not supported for a deformable layer — its tracks share one deformation "
                        f"network, so an enabled track's updates would change a frozen track's output. "
                        f"Freeze or enable the whole layer, or disable (not freeze) the frozen tracks."
                    )

            if obj_track_id.is_empty() and tracks_config:
                if self.component_only is not None and self._component_state(layer_id) != ComponentState.disabled:
                    raise ValueError(
                        f"component_only: non-disabled track-backed layer '{layer_id}' matched no cuboid tracks. "
                        "It cannot be rendered or supervised; disable it or fix its track selector."
                    )
                continue

            # An empty track selector means "all tracks" and is therefore a
            # real track-backed layer even though the config mapping is falsy.
            if tracks_config is not None:
                self._component_only_layer_tracks[layer_id] = cuboid_subset

            # Store the per-track partition even for disabled layers — the
            # residual carve of a fully-disabled layer's boxes needs each
            # track's effective variant (bbox vs bbox_expanded).
            if partition is not None:
                self._component_only_track_states[layer_id] = partition.states
                self._component_only_track_expanded[layer_id] = partition.expanded

            if self._component_state(layer_id) == ComponentState.disabled:
                # Component-only: disabled = absent. No Gaussian node is
                # constructed; the layer's mask region stays carved out
                # of the residual via the tracks stored above.
                log.info(f"component_only: skipping disabled layer '{layer_id}'")
                continue

            # Per-track subsetting: resolve each dataset track's
            # effective state/variant and build the node from the
            # non-disabled partition only (disabled tracks = absent).
            cuboid_subset_for_node = cuboid_subset
            if partition is not None:
                supervised_ids = [
                    tid for tid in cuboid_subset.tracks_id if partition.states[tid] != ComponentState.disabled
                ]
                if not supervised_ids:
                    log.warning(
                        f"component_only: layer '{layer_id}' has per-track overrides but no "
                        f"non-disabled track — skipping the layer entirely"
                    )
                    continue
                log.info(
                    f"component_only: layer '{layer_id}' restricted to {len(supervised_ids)} / "
                    f"{len(cuboid_subset.tracks_id)} tracks"
                )
                cuboid_subset_for_node = CuboidTracks.Ops.subset_from_tracks_id(cuboid_subset, supervised_ids)

            # Check if the class specific labels are specified or if there is an intradependency on other layers
            labels_to_ignore = []
            labels_to_use = self.config.layers[layer_id].class_labels or []
            if layers_to_ignore := (self.config.layers[layer_id].ignore_classes_from_layers or []):
                labels_to_ignore.extend(
                    [
                        label
                        for layer in layers_to_ignore
                        for label in unpack_optional(self.config.layers[layer].class_labels)
                    ]
                )

            # Set start and end timestamp for SHGaussianModel with temporal appearance
            if unpack_optional(layer_config.fourier_features_dim) > 1:
                assert rig_trajectories is not None, (
                    "rig_trajectories is required for SHGaussianModel with temporal appearance"
                )
                assert len(rig_trajectories.rig_trajectories), (
                    f"{self.__class__.__name__}: expected at least a single rig_trajectory, got {len(rig_trajectories.rig_trajectories)}"
                )
                start_end_timestamp_us = (
                    int(rig_trajectories.rig_trajectories[0].T_rig_world_timestamps_us[0].item()),
                    int(rig_trajectories.rig_trajectories[-1].T_rig_world_timestamps_us[-1].item()),
                )
            else:
                start_end_timestamp_us = None

            try:
                initializer = BaseInitialization.factory(
                    layer_config.initialization.name,
                    layer_config.initialization,
                    layer_config,
                    trainer_config,
                    labels_to_ignore=labels_to_ignore,
                    labels_to_use=labels_to_use,
                )

                gaussians_nodes[layer_id] = BaseGaussianModel.factory(
                    layer_config.name,
                    layer_config,
                    trainer_config,
                    datasource,
                    init_from_datasource,
                    initializer,
                    layer_config.precision,
                    cuboid_subset_for_node,
                    start_end_timestamp_us,
                )

                if gaussian_template_node_model is None:
                    gaussian_template_node_model = gaussians_nodes[layer_id]

            except Exception as e:
                match e:
                    case NoPointsFoundException():
                        if (
                            self.component_only is not None
                            and self._component_state(layer_id) != ComponentState.disabled
                        ):
                            raise ValueError(
                                f"component_only: non-disabled layer '{layer_id}' could not be "
                                "constructed because initialization found no points. It cannot "
                                "be rendered or supervised; disable it or fix its initialization."
                            ) from e
                        log.warning("Skipping Gaussian node '%s' during construction: %s", layer_id, e)
                        constructor_skipped_nodes.add(layer_id)
                        continue
                    case _:
                        raise RuntimeError(f"Failed to initialize {layer_config.name} with error: {e}") from e

        pruned_nodes = self._prune_unavailable_motion_dependents(
            gaussians_nodes,
            unavailable_layer_ids=constructor_skipped_nodes,
        )
        if pruned_nodes:
            gaussian_template_node_model = next(iter(gaussians_nodes.values()), None)

        self._component_only_globally_rendered_track_ids = self._resolve_component_only_rendered_track_ids(
            set(gaussians_nodes)
        )

        if not gaussians_nodes:
            raise ValueError(
                "GaussiansComposite: no Gaussian layer was constructed. With component_only "
                "enabled this means every declared layer is disabled — enable at least one "
                "layer in component_only.layers."
            )

        self._bind_motion_source_layers(gaussians_nodes)

        layers_camera_extra_ray_signal_infos = [
            layer.camera_extra_ray_signal_infos for layer in gaussians_nodes.values()
        ]
        self.camera_extra_ray_signal_infos = layers_camera_extra_ray_signal_infos[0]
        for camera_extra_ray_signal_infos in layers_camera_extra_ray_signal_infos[1:]:
            assert (
                self.camera_extra_ray_signal_infos[0] == camera_extra_ray_signal_infos[0]
                and self.camera_extra_ray_signal_infos[1] == camera_extra_ray_signal_infos[1]
            ), "camera_extra_ray_signal_infos must be the same for all layers"

        layers_lidar_extra_ray_signal_infos = [layer.lidar_extra_ray_signal_infos for layer in gaussians_nodes.values()]
        self.lidar_extra_ray_signal_infos = layers_lidar_extra_ray_signal_infos[0]
        for lidar_extra_ray_signal_infos in layers_lidar_extra_ray_signal_infos[1:]:
            assert (
                self.lidar_extra_ray_signal_infos[0] == lidar_extra_ray_signal_infos[0]
                and self.lidar_extra_ray_signal_infos[1] == lidar_extra_ray_signal_infos[1]
            ), "lidar_extra_ray_signal_infos must be the same for all layers"

        post_processings: list[BasePostProcessing] = []
        post_processings_config = self.config.post_processing or {}
        if self._component_state("post_processing") == ComponentState.disabled:
            # Disabled = absent: the whole chain (no per-entry toggle).
            post_processings_config = {}
        for config_key in sorted(post_processings_config.keys()):
            post_processing = BasePostProcessing.factory(
                post_processings_config[config_key].name,
                post_processings_config[config_key],
                trainer_config,
                n_training_frames_per_camera=datasource.get_n_training_frames_per_camera().tolist(),
                n_training_frames_per_lidar=[],
            )
            post_processings.append(post_processing)

        self.gaussians_nodes = TypedModuleDict(gaussians_nodes)

        # Frozen components: constructed and rendered, never optimized.
        # Densification/pruning must not touch them either — reuse the
        # strategies' existing exclude_layer_ids mechanism.
        self._component_only_frozen_layers: frozenset[str] = frozenset(
            layer_id for layer_id in gaussians_nodes if self._component_state(layer_id) == ComponentState.frozen
        )
        strategy_excludes = set(self._component_only_frozen_layers)
        for layer_id, track_states in self._component_only_track_states.items():
            if layer_id not in gaussians_nodes:
                continue
            # Per-track subsetting: decide by the effective states of the
            # CONSTRUCTED rows. Disabled tracks are absent — their mere
            # presence in the config must not cost the layer its
            # densification. An all-enabled partition behaves like any
            # ordinary tracked layer: the strategies maintain the
            # per-Gaussian ``gaussian_cuboid_ids`` buffer through
            # add/relocate/prune (``get_additional_buffers`` +
            # ``_update_param_with_optimizer``), so the Gaussian -> track
            # mapping stays valid. Any FROZEN row, however, means the
            # whole layer must sit out: the frozen-track gradient hooks
            # capture a fixed-length row mask, and densification is not
            # row-aware.
            constructed_states = {state for state in track_states.values() if state != ComponentState.disabled}
            if ComponentState.frozen in constructed_states:
                # NOTE: this excludes the WHOLE layer, so its *enabled* tracks
                # lose densification too. Densification is not row-aware and the
                # frozen-track gradient hooks capture a fixed-length row mask;
                # row-aware per-track densification is a separate follow-up MR.
                log.warning(
                    "component_only: layer '%s' has a frozen track — excluding the whole layer from "
                    "densification/pruning; its enabled tracks lose densification too (per-track "
                    "densification is a follow-up)",
                    layer_id,
                )
                strategy_excludes.add(layer_id)
        background_scope = {r: s for r, s in self._background_scope().items() if s != ComponentState.disabled}
        background_mixed_states = len(set(background_scope.values())) > 1
        # bbox-only scope drops the out-of-frustum background rows in
        # _apply_component_only_background_scope, which now runs BEFORE the
        # strategy sizes its per-row buffers (see
        # initialize_strategy_and_maybe_gaussians), so bbox-only background can
        # stay in densification. Only mixed region states must sit out: they
        # install per-row gradient hooks a partition densify/prune can't respect.
        if background_mixed_states and BACKGROUND_LAYER_ID in gaussians_nodes:
            log.warning(
                "component_only: excluding background from densification/pruning — mixed region "
                "states install per-row gradient hooks densification cannot respect"
            )
            strategy_excludes.add(BACKGROUND_LAYER_ID)
        strategy_config = self.config.strategy
        if strategy_excludes:
            strategy_config = strategy_config.model_copy(
                update={"exclude_layer_ids": sorted(set(strategy_config.exclude_layer_ids) | strategy_excludes)}
            )
        if self.component_only is not None:
            # Exclusions are now runtime-derived (frozen layers/tracks + background
            # scope), not readable off the parsed yaml — log the final effective set.
            log.info(
                "component_only: final densification/pruning exclude_layer_ids = %s",
                sorted(strategy_config.exclude_layer_ids),
            )
        # Keep the effective but unadjusted config for deferred-empty-layer rebuilds.
        self._gaussians_strategy_config = strategy_config
        self.gaussians_strategy = BaseGaussianStrategy.factory(
            strategy_config.name,
            strategy_config,
            trainer_config,
            init_from_datasource,
            self.gaussians_nodes,
        )

        self.post_processings = TypedModuleList(post_processings)

        # Controller distillation state: computed from post-processing config if present.
        self._distillation_start_step: int = -1
        self._distillation_active: bool = False
        for pp_cfg in post_processings_config.values():
            distillation_steps: int = getattr(pp_cfg, "controller_distillation_steps", 0)
            if distillation_steps > 0 and trainer_config.max_steps > 0:
                self._distillation_start_step = trainer_config.max_steps - distillation_steps
                break

        self.debug_viz = self.config.debug_viz
        self.overriden_state: dict[str, Any] = {}

        # Create separate renderer instances for camera and lidar to enable thread-safe parallel rendering.
        # Each renderer maintains internal state (via update_model_parameters) that is not thread-safe.
        self._camera_renderer = BaseGaussianRenderer.factory(
            self.config.renderer.name, self.config.renderer, unpack_optional(gaussian_template_node_model)
        )
        self._lidar_renderer = BaseGaussianRenderer.factory(
            self.config.renderer.name, self.config.renderer, unpack_optional(gaussian_template_node_model)
        )
        # Keep backward compatibility aliases
        self.gaussians_renderer = self._camera_renderer
        self.gaussians_renderer_camera = self._camera_renderer
        self.gaussians_renderer_lidar = self._lidar_renderer

        # Provide the lidar renderer with the unique_sensor_idx -> lidar_id map so it
        # can resolve PerSensor renderer config (lidar_unstructured / cull / box_method)
        # per sensor at render time (the renderer factory does not receive the
        # datasource). Only GSplatRenderer consumes this; the map lives on the
        # concrete NCORE datasource (lidar_unique_ids: {lidar_id: (id, idx)}).
        #
        # The live datasource is absent for artifact / inference rendering
        # (DataSourceSummary.from_dict sets datasource=None, see
        # nurec/render/render.py::_load_gaussians_composite), so fall back to the
        # SERIALIZED rig trajectories, which carry lidar_calibrations keyed by unique
        # lidar id with a unique_sensor_idx on each entry -- the same map, already in
        # the checkpoint. Without this fallback a documented {lidar_id: value}
        # PerSensor renderer setting raises in resolve_per_sensor at render time,
        # because _lidar_id_from_meta() can never resolve an id (MR !3888 review).
        if isinstance(self._lidar_renderer, GSplatRenderer):
            lidar_idx_to_id = self._resolve_lidar_idx_to_id(datasource)
            if lidar_idx_to_id:
                self._lidar_renderer.set_lidar_idx_to_id(lidar_idx_to_id)

        # unique_sensor_idx -> camera name, for name-keyed camera options such as the sky background's
        # per-camera ground fill (frame metadata carries only the index for NCORE data)
        self._camera_idx_to_name = self._resolve_camera_idx_to_name(datasource)

        # Dedicated CUDA stream for lidar rendering (camera uses the current/default
        # stream). Created lazily in _resolve_lidar_stream so the canonical owner can be
        # the per-request context (see Backend._lidar_stream, which is threaded into
        # forward_gaussians); this composite-owned stream is only the fallback for
        # training / standalone use.
        self._lidar_stream: Optional[torch.cuda.Stream] = None

        # Per-camera precomputed camera→lidar tile mappings, keyed by unique_sensor_idx.
        # Lazy-init on first forward with each camera+lidar pair.
        self._cam_to_lidar_tile_mappings: dict[int, CameraToLidarTileMapping | None] = {}

        # Multi-GPU: contiguous global ranges for compacting renderer scene_data
        # to this rank (set in collect_gaussian_parameters).
        self._visibility_local_ranges: Optional[tuple[tuple[int, int], ...]] = None
        # Sparse segments: per tracked layer, the host-side window table the frame
        # selection is made from (see nurec.models.gaussians.segment_plan).
        self._segment_tables: dict[str, LayerSegmentTable] = {}

        # Per-composite weak identity/version generations for value-dependent
        # integer payload validation.  This runtime-only state never retains a
        # replaced Parameter/Buffer and is not part of checkpoints.
        self._fused_payload_integer_range_validation_state = _PayloadIntegerRangeValidationState()

        # We can have different renderers for camera and lidar frames.
        # They can be overriden from the command line.
        # Ex: to override the renderer camera
        # model.renderer.use_gsplat_for_camera_rendering=True
        if getattr(self.config.renderer, "use_gsplat_for_camera_rendering", False):
            config_gsplat = copy.deepcopy(self.config.renderer)
            config_gsplat.name = "3dgut-gsplat"
            # Instantiate the renderer used for camera frames (replaces _camera_renderer)
            self._camera_renderer = BaseGaussianRenderer.factory(
                config_gsplat.name, config_gsplat, unpack_optional(gaussian_template_node_model)
            )

        self.saturate_radiance = getattr(self.config, "saturate_radiance", True)

        # Create the gaussian parameter collector.
        layers_config = self._get_layers_config()
        # Kept because the static-collect cache asks the renderer about fold
        # eligibility before collecting, and these widths are what the renderer will
        # see. They are allocation dims, so they hold on every frame including the
        # first, unlike the collected tensors themselves.
        self._layers_config = layers_config
        self.collector = collect.CreateGaussianParameterCollector(layers_config)

        # Copy the collector to each node, so that each node can access the "global" functions.
        for node in self.gaussians_nodes.values():
            node = cast(SHGaussianModel, node)
            node.collector = self.collector

        # Component-only loss-mask spec: resolved once here (fail-loud
        # on semantic-vocabulary or dataset-capability mismatches).
        # None when component-only mode is off, or when there is no
        # live datasource (inference from json — no training, no mask).
        self._component_only_mask_spec: Optional[ComponentOnlyMaskSpec] = None
        if self.component_only is not None and datasource.datasource is not None:
            # Local import: nurec.datasets.ncore transitively imports heavy
            # modules; only needed on this path.
            from nurec.datasets.ncore import NCOREDataSource

            class_map: Optional[dict[str, int]] = None
            if isinstance(datasource.datasource, NCOREDataSource):
                class_map = datasource.datasource.get_semantic_classes_map(camera_semantics=True, lidar_semantics=False)
            self._component_only_mask_spec = build_mask_spec(
                self.component_only,
                self.config.layers,
                class_map,
                layer_track_states=self._component_only_track_states,
            )

        self._apply_component_only_freeze()
        self._log_component_inventory()

    @staticmethod
    def _resolve_camera_idx_to_name(datasource: DataSourceSummary) -> dict[int, str]:
        """``unique_sensor_idx -> camera name``, from the same two sources as ``_resolve_lidar_idx_to_id``.

        The live NCORE datasource's ``camera_unique_ids`` ({camera_id: (id, idx)}) at training time, else the
        serialized ``state.rig_trajectories.camera_calibrations``. Empty when neither is available.
        """
        underlying_datasource = getattr(datasource, "datasource", None)
        camera_unique_ids = getattr(underlying_datasource, "camera_unique_ids", None)
        if camera_unique_ids is not None:
            return {uid.idx: camera_id for camera_id, uid in camera_unique_ids.items()}

        rig_trajectories = getattr(getattr(datasource, "state", None), "rig_trajectories", None)
        camera_calibrations = getattr(rig_trajectories, "camera_calibrations", None)
        if camera_calibrations is not None:
            return {
                int(calibration.unique_sensor_idx): calibration.logical_sensor_name
                for calibration in camera_calibrations.values()
                if calibration.unique_sensor_idx >= 0
            }

        return {}

    @staticmethod
    def _resolve_lidar_idx_to_id(datasource: DataSourceSummary) -> dict[int, str]:
        """``unique_sensor_idx -> lidar_id`` for resolving PerSensor renderer config.

        Two sources, in order, because the live datasource is only present at training
        time:

        1. The live NCORE datasource's ``lidar_unique_ids`` ({lidar_id: (id, idx)}).
        2. The serialized ``state.rig_trajectories.lidar_calibrations`` -- an
           OrderedDict keyed by *unique* lidar id whose entries carry
           ``unique_sensor_idx``. This is what artifact / inference rendering has
           (``DataSourceSummary.from_dict`` sets ``datasource=None``), and it needs no
           new serialization: the mapping is already in every checkpoint.

        Returns an empty dict when neither is available, in which case only scalar
        (broadcast) PerSensor values are resolvable and a dict-valued one raises in
        ``resolve_per_sensor`` -- the pre-existing, intentional failure mode.
        """
        underlying_datasource = getattr(datasource, "datasource", None)
        lidar_unique_ids = getattr(underlying_datasource, "lidar_unique_ids", None)
        if lidar_unique_ids is not None:
            return {uid.idx: lidar_id for lidar_id, uid in lidar_unique_ids.items()}

        rig_trajectories = getattr(datasource.state, "rig_trajectories", None)
        lidar_calibrations = getattr(rig_trajectories, "lidar_calibrations", None)
        if lidar_calibrations is not None:
            return {
                int(calibration.unique_sensor_idx): calibration.logical_sensor_name
                for calibration in lidar_calibrations.values()
                if calibration.unique_sensor_idx >= 0
            }

        return {}

    def _prune_unavailable_motion_dependents(
        self,
        gaussian_nodes: dict[str, BaseGaussianModel] | TypedModuleDict[BaseGaussianModel],
        *,
        unavailable_layer_ids: set[str] | None = None,
    ) -> set[str]:
        """Remove unavailable motion owners and their recursive dependent closure."""

        known_layer_ids = set(self.config.layers)
        removed = set(unavailable_layer_ids or ())
        while True:
            dependents = {
                layer_id
                for layer_id, node in gaussian_nodes.items()
                if layer_id not in removed
                and (source_layer_id := getattr(node.config, "motion_source_layer", None)) is not None
                and source_layer_id in known_layer_ids
                and (source_layer_id in removed or source_layer_id not in gaussian_nodes)
            }
            if not dependents:
                break
            removed.update(dependents)

        for layer_id in removed:
            if layer_id in gaussian_nodes:
                del gaussian_nodes[layer_id]
            self.obj_track_ids.pop(layer_id, None)
            self.config.layers.pop(layer_id, None)
            self._component_only_layer_tracks.pop(layer_id, None)
            self._component_only_track_states.pop(layer_id, None)
            self._component_only_track_expanded.pop(layer_id, None)

        if removed and hasattr(self, "_component_only_frozen_layers"):
            self._component_only_frozen_layers = self._component_only_frozen_layers.difference(removed)
        if removed and (spec := getattr(self, "_component_only_mask_spec", None)) is not None:
            self._component_only_mask_spec = ComponentOnlyMaskSpec(
                supervised_layers={
                    layer_id: groups for layer_id, groups in spec.supervised_layers.items() if layer_id not in removed
                },
                carved_layers={
                    key: region
                    for key, region in spec.carved_layers.items()
                    if key.removesuffix(DISABLED_TRACKS_SUFFIX) not in removed
                },
                disabled_track_regions={
                    key: region
                    for key, region in spec.disabled_track_regions.items()
                    if key.removesuffix(DISABLED_TRACKS_SUFFIX) not in removed
                },
                include_residual=spec.include_residual,
                include_sky=spec.include_sky,
                sky_class_id=spec.sky_class_id,
                ego_class_id=spec.ego_class_id,
            )
        if removed:
            self._initialization_skipped_layer_ids = getattr(self, "_initialization_skipped_layer_ids", set()) | removed
        return removed

    def _bind_motion_source_layers(
        self,
        gaussian_nodes: Mapping[str, BaseGaussianModel] | TypedModuleDict[BaseGaussianModel],
    ) -> None:
        """Bind dependent rigid layers to their canonical motion owner."""

        for layer_id, gaussian_node in gaussian_nodes.items():
            source_layer_id = getattr(gaussian_node.config, "motion_source_layer", None)
            if source_layer_id is None:
                continue
            if source_layer_id not in self.config.layers:
                raise ValueError(f"Gaussian layer '{layer_id}' references unknown motion source '{source_layer_id}'")
            if source_layer_id not in gaussian_nodes:
                raise ValueError(f"Gaussian layer '{layer_id}' requires unavailable motion source '{source_layer_id}'")
            source_node = gaussian_nodes[source_layer_id]
            if not isinstance(gaussian_node, RigidGaussianModel) or not isinstance(source_node, RigidGaussianModel):
                raise TypeError(
                    f"Motion-source binding requires rigid layers, got target={type(gaussian_node).__name__}, "
                    f"source={type(source_node).__name__}"
                )
            if gaussian_node.owns_tracks:
                raise ValueError(f"Motion-dependent Gaussian layer '{layer_id}' must not own tracks")
            if not source_node.owns_tracks:
                raise ValueError(f"Gaussian layer '{layer_id}' motion source '{source_layer_id}' is not a track owner")
            if not isinstance(gaussian_node.tracks_calib, SkipTracksCalib):
                raise ValueError(f"Motion-dependent Gaussian layer '{layer_id}' must configure skip-tracks-calib")
            gaussian_node.bind_motion_source(source_node)

    def _resolve_component_only_track_partition(
        self, layer_id: str, cuboid_subset: CuboidTracks
    ) -> Optional[_TrackPartition]:
        """Resolve per-track states/variants for a layer with track overrides.

        Pure: returns a ``_TrackPartition`` (which the caller stores), or
        ``None`` when the layer has no per-track overrides. Fails loud on
        track ids that don't exist in the dataset's track subset for this
        layer (typo protection — same philosophy as unknown layer keys).
        """
        if self.component_only is None:
            return None
        entry = self.component_only.layers.get(layer_id)
        if entry is None or entry.tracks is None:
            return None

        from nurec.config.component_only import resolve_layer_masks

        available_ids = list(cuboid_subset.tracks_id)
        unknown = sorted(set(entry.tracks) - set(available_ids))
        if unknown:
            preview = available_ids[:20]
            raise ValueError(
                f"component_only.layers.{layer_id}.tracks: track id(s) {unknown} do not exist in "
                f"this dataset's '{layer_id}' tracks ({len(available_ids)} available, first "
                f"{len(preview)}: {preview})"
            )

        layer_sources, _ = resolve_layer_masks(layer_id, entry)
        layer_expanded = MaskSource.bbox_expanded in layer_sources

        states: dict[str, ComponentState] = {}
        expanded: dict[str, bool] = {}
        for tid in available_ids:
            track_entry = entry.tracks.get(tid)
            if track_entry is None:
                states[tid] = entry.state
                expanded[tid] = layer_expanded
            else:
                # An omitted per-track state (None) inherits the layer's
                # state — a mask-only override must not flip a frozen/disabled
                # layer's track to enabled.
                states[tid] = track_entry.state if track_entry.state is not None else entry.state
                track_sources, _ = resolve_layer_masks(layer_id, entry, track_entry)
                expanded[tid] = MaskSource.bbox_expanded in track_sources
        return _TrackPartition(states=states, expanded=expanded)

    def _resolve_component_only_rendered_track_ids(
        self, constructed_layer_ids: Optional[set[str]] = None
    ) -> frozenset[str]:
        """Track ids present in any rendered (enabled or frozen) layer.

        Layer selectors are independent and can overlap. Global rendered
        precedence prevents a disabled occurrence of an overlapping track from
        treating the same physical actor as absent and occluding its rendered
        occurrence at equal depth. When construction skipped a configured layer
        (for example because initialization found no points), only successfully
        constructed layers count as rendered.
        """
        rendered: set[str] = set()
        for layer_id, tracks in self._component_only_layer_tracks.items():
            if constructed_layer_ids is not None and layer_id not in constructed_layer_ids:
                continue
            states = self._component_only_track_states.get(layer_id)
            if states is None:
                if self._component_state(layer_id) != ComponentState.disabled:
                    rendered.update(tracks.tracks_id)
                continue
            rendered.update(track_id for track_id, state in states.items() if state != ComponentState.disabled)
        return frozenset(rendered)

    def _background_scope(self) -> dict[ScopeRegion, ComponentState]:
        """Background's resolved region -> state mapping.

        A background entry with ``state`` and no ``scope`` is sugar for
        ``{residual: state}``. Absent regions are disabled; an unlisted
        background is fully disabled.
        """
        if self.component_only is None:
            return {ScopeRegion.residual: ComponentState.enabled}
        entry = self.component_only.layers.get(BACKGROUND_LAYER_ID)
        if entry is None:
            return {}
        if entry.scope is not None:
            return dict(entry.scope)
        return {ScopeRegion.residual: entry.state}

    def _component_state(self, component_id: str) -> ComponentState:
        """Effective state of a component under component-only mode.

        ``enabled`` for everything when component-only mode is off (BC
        path). In component-only mode, unlisted components are
        ``disabled`` (opt-in semantics). ``background`` aggregates over
        its scope regions: the node exists iff any region is
        non-disabled; it is frozen-as-a-whole iff all non-disabled
        regions are frozen (mixed states are handled per row by the
        scope machinery).
        """
        if self.component_only is None:
            return ComponentState.enabled
        if component_id == "sky_model":
            entry = self.component_only.sky_model
        elif component_id == "calib":
            entry = self.component_only.calib
        elif component_id == "post_processing":
            entry = self.component_only.post_processing
        else:
            layer_entry = self.component_only.layers.get(component_id)
            if layer_entry is None:
                return ComponentState.disabled
            if component_id == BACKGROUND_LAYER_ID:
                states = [s for s in self._background_scope().values() if s != ComponentState.disabled]
                if not states:
                    return ComponentState.disabled
                if all(s == ComponentState.frozen for s in states):
                    return ComponentState.frozen
                return ComponentState.enabled
            if layer_entry.tracks is not None:
                # Per-track overrides can flip individual tracks against
                # the layer default (enabled tracks under a disabled
                # layer and vice versa). The layer exists iff any track
                # is non-disabled; layer-WIDE freezing applies only when
                # nothing in the population is enabled — mixed states
                # are handled per row by the frozen-track hooks.
                if not _entry_has_non_disabled_population(layer_entry):
                    return ComponentState.disabled
                if layer_entry.state == ComponentState.frozen and all(
                    track.state != ComponentState.enabled for track in layer_entry.tracks.values()
                ):
                    return ComponentState.frozen
                return ComponentState.enabled
            return layer_entry.state
        return entry.state if entry is not None else ComponentState.disabled

    def _component_only_ray_cuboid_hits(
        self,
        tracks: CuboidTracks,
        tracks_idx: torch.Tensor,
        expand: float,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        rays_timestamps_us: torch.Tensor,
        track_indices_key: Optional[tuple[int, ...]] = None,
    ) -> ProjectedTrackRegion:
        """Return flat hit mask and nearest forward depth for a track group.

        Reuses the fused ``CuboidTracks.ray_intersection`` kernel — which
        interpolates each cuboid's pose at that ray's own timestamp
        (rolling-shutter accurate) — rather than a bespoke static-pose slab
        test. ``expand`` is the group's fractional ``cuboid_tracks_expand``;
        it maps to the kernel's additive ``cuboids_dims_padding`` as
        ``dims * expand``. Tracks whose keyframes don't cover a ray's
        timestamp simply contribute no hit for that ray.

        Kernel output slots are assigned by an atomic race across tracks, not
        sorted by depth. Each bounded track chunk therefore retains all of its
        intersections and explicitly reduces ``t_near``. The kernel already
        maps an origin inside a cuboid to depth zero; clamping here documents
        and defends that forward-depth convention.
        """
        hits = torch.zeros(rays_o.shape[0], dtype=torch.bool, device=rays_o.device)
        nearest_depth = torch.full((rays_o.shape[0],), torch.inf, dtype=rays_o.dtype, device=rays_o.device)
        if tracks_idx.numel() == 0 or rays_o.shape[0] == 0:
            return ProjectedTrackRegion(mask=hits, nearest_depth=nearest_depth)

        if track_indices_key is None:
            track_indices_key = tuple(int(index) for index in tracks_idx.detach().cpu().tolist())
        chunks_key = (id(tracks), track_indices_key)
        chunks = self._component_only_intersection_chunk_cache.get(chunks_key)
        if chunks is None:
            chunk_size = self._COMPONENT_ONLY_INTERSECTION_TRACK_CHUNK_SIZE
            chunks = tuple(
                CuboidTracks.Ops.subset_from_indices(tracks, tracks_idx[start : start + chunk_size])
                for start in range(0, tracks_idx.numel(), chunk_size)
            )
            self._component_only_intersection_chunk_cache[chunks_key] = chunks
        for subset in chunks:
            padding = subset.cuboids_dims * expand if expand else None
            ray_chunk_size = self._component_only_intersection_ray_chunk_size(
                num_rays=rays_o.shape[0],
                num_tracks=subset.n_tracks,
                scalar_element_size=rays_o.element_size(),
            )
            slot_indices = torch.arange(subset.n_tracks, device=rays_o.device)
            for ray_start in range(0, rays_o.shape[0], ray_chunk_size):
                ray_slice = slice(ray_start, ray_start + ray_chunk_size)
                result = subset.ray_intersection(
                    rays_o[ray_slice].contiguous(),
                    rays_d[ray_slice].contiguous(),
                    rays_timestamps_us[ray_slice].contiguous(),
                    padding,
                    max_intersections_per_ray=subset.n_tracks,
                    with_intersections_ts=True,
                )
                assert result.intersections_ts is not None
                valid_slots = slot_indices.unsqueeze(0) < result.intersections_cnt.unsqueeze(1)
                near_depths = result.intersections_ts[..., 0].clamp_min(0)
                chunk_depth = torch.where(valid_slots, near_depths, torch.inf).amin(dim=1)
                hits[ray_slice] |= result.intersections_cnt > 0
                torch.minimum(nearest_depth[ray_slice], chunk_depth, out=nearest_depth[ray_slice])
                # Drop the dense result and reduction intermediates before the
                # next kernel invocation; otherwise evaluating the next RHS
                # can transiently keep two chunks alive at once.
                del result, valid_slots, near_depths, chunk_depth
        return ProjectedTrackRegion(mask=hits, nearest_depth=nearest_depth)

    @staticmethod
    def _component_only_intersection_temporary_bytes(
        num_rays: int,
        num_tracks: int,
        scalar_element_size: int,
    ) -> int:
        """Conservative temporary bytes for one dense intersection call.

        Accounts for potentially copied ray origins/directions/timestamps,
        native counts/track-indices/depth endpoints, validity/clamp/where/amin
        intermediates, the hit comparison, and the per-track slot-index tensor.
        """
        per_ray_bytes = (7 * scalar_element_size + 13) + num_tracks * (5 + 4 * scalar_element_size)
        return 8 * num_tracks + num_rays * per_ray_bytes

    def _component_only_intersection_ray_chunk_size(
        self,
        num_rays: int,
        num_tracks: int,
        scalar_element_size: int,
    ) -> int:
        """Largest ray chunk whose conservative temporary payload fits the budget."""
        fixed_bytes = 8 * num_tracks
        per_ray_bytes = (
            self._component_only_intersection_temporary_bytes(
                num_rays=1,
                num_tracks=num_tracks,
                scalar_element_size=scalar_element_size,
            )
            - fixed_bytes
        )
        rays_within_budget = max(
            1,
            (self._COMPONENT_ONLY_INTERSECTION_TEMP_MEMORY_BUDGET_BYTES - fixed_bytes) // per_ray_bytes,
        )
        return min(num_rays, rays_within_budget)

    def _component_only_ray_timestamps(self, rendering_cam: RenderingData, shape: tuple[int, int, int]) -> torch.Tensor:
        """Per-ray (B*H*W,) int64 timestamps for the current camera batch.

        Prefers the batch's per-ray ``rays_timestamps_us`` (per-scanline,
        so the cuboid pose is interpolated with rolling-shutter accuracy);
        falls back to each item's frame mid-timestamp broadcast to all its
        pixels when the batch carries no per-ray stamps.
        """
        batch, height, width = shape
        device = rendering_cam.rays.device
        per_ray = getattr(rendering_cam, "rays_timestamps_us", None)
        if per_ray is not None:
            return per_ray.reshape(-1).to(device=device, dtype=torch.int64)
        mid = rendering_cam.timestamps_startend_us_cpu.to(torch.int64).sum(dim=1) // 2  # (B,)
        return mid.view(batch, 1, 1).expand(batch, height, width).reshape(-1).to(device)

    def build_component_only_bbox_masks(self, rendering_cam: RenderingData) -> dict[str, ProjectedTrackRegion]:
        """Per-region projected track masks/depths for the camera batch.

        Covers every layer whose resolved mask recipe includes a bbox
        source — supervised AND carved (the residual needs the carved
        layers' regions too). Each layer's mask uses its own configured
        box variant: ``bbox_expanded`` applies the layer's
        ``cuboid_tracks_expand`` factor, raw ``bbox`` applies none.

        Cuboid poses are interpolated per ray (rolling-shutter accurate
        when the batch carries per-ray timestamps) via the shared
        ``CuboidTracks.ray_intersection`` kernel; tracks that don't cover a
        ray's timestamp contribute nothing (their box simply isn't there
        for that ray).
        """
        spec = self._component_only_mask_spec
        if spec is None:
            return {}

        # Flatten to (bbox-mask key, region group, layer id, disabled partition)
        # — supervised
        # layers now carry a tuple of per-recipe groups (each with its own
        # bbox_key), and carved pseudo-entries stay one spec per key. A dict
        # deduplicates the residual-carve and occlusion-depth uses of a disabled
        # pseudo key.
        needed: dict[str, tuple[LayerRegionSpec, str, bool]] = {}
        for layer_id, groups in spec.supervised_layers.items():
            for group in groups:
                if group.needs_bbox or group.semantic_bbox_key or group.track_depth_key:
                    needed[group.bbox_key or layer_id] = (group, layer_id, False)
        if spec.include_residual:
            for key, layer_spec in spec.carved_layers.items():
                if layer_spec.needs_bbox or layer_spec.semantic_bbox_key:
                    needed[key] = (
                        layer_spec,
                        key.removesuffix(DISABLED_TRACKS_SUFFIX),
                        key.endswith(DISABLED_TRACKS_SUFFIX),
                    )
        for key, layer_spec in spec.disabled_track_regions.items():
            needed[key] = (layer_spec, key.removesuffix(DISABLED_TRACKS_SUFFIX), True)

        rays = rendering_cam.rays  # (B, H, W, 6), NuRec space
        batch, height, width, _ = rays.shape
        rays_o = rays[..., :3].reshape(-1, 3)
        rays_d = rays[..., 3:].reshape(-1, 3)
        rays_ts = self._component_only_ray_timestamps(rendering_cam, (batch, height, width))

        masks: dict[str, ProjectedTrackRegion] = {}
        projected_geometry: dict[tuple[int, tuple[int, ...], float], ProjectedTrackRegion] = {}
        for mask_key, (layer_spec, layer_id, is_disabled_partition) in needed.items():
            hits = torch.zeros(rays_o.shape[0], dtype=torch.bool, device=rays.device)
            nearest_depth = torch.full((rays_o.shape[0],), torch.inf, dtype=rays.dtype, device=rays.device)
            tracks = self._component_only_layer_tracks.get(layer_id)
            if tracks is not None and tracks.n_tracks > 0:
                cache_key = (layer_id, layer_spec, is_disabled_partition)
                bbox_groups = self._component_only_bbox_track_groups(
                    layer_id, layer_spec, disabled_partition=is_disabled_partition
                )
                indices_keys = self._component_only_bbox_group_indices_cache[cache_key]
                for (tracks_idx, expand), indices_key in zip(bbox_groups, indices_keys, strict=True):
                    geometry_key = (id(tracks), indices_key, expand)
                    projected = projected_geometry.get(geometry_key)
                    if projected is None:
                        projected = self._component_only_ray_cuboid_hits(
                            tracks,
                            tracks_idx,
                            expand,
                            rays_o,
                            rays_d,
                            rays_ts,
                            track_indices_key=indices_key,
                        )
                        projected_geometry[geometry_key] = projected
                    hits |= projected.mask
                    nearest_depth = torch.minimum(nearest_depth, projected.nearest_depth)
            masks[mask_key] = ProjectedTrackRegion(
                mask=hits.view(batch, height, width),
                nearest_depth=nearest_depth.view(batch, height, width),
            )
        return masks

    def _component_only_bbox_track_groups(
        self, layer_id: str, layer_spec: LayerRegionSpec, disabled_partition: bool = False
    ) -> list[tuple[torch.Tensor, float]]:
        """(track indices, expand factor) groups for a region group's bbox mask.

        When the region group names explicit tracks (``bbox_track_ids`` —
        a per-recipe group from a per-track ``masks`` override), that is
        exactly one group with the recipe's box variant. Otherwise the
        layer's whole partition is used: all non-disabled tracks (or the
        disabled partition for the residual's carved pseudo entry),
        grouped by their per-track raw/expanded variant.
        """
        tracks = self._component_only_layer_tracks.get(layer_id)
        if tracks is None or tracks.n_tracks == 0:
            return []
        cache_key = (layer_id, layer_spec, disabled_partition)
        if cache_key in self._component_only_bbox_groups_cache:
            return self._component_only_bbox_groups_cache[cache_key]

        def cache(groups: list[tuple[torch.Tensor, float]]) -> list[tuple[torch.Tensor, float]]:
            self._component_only_bbox_group_indices_cache[cache_key] = tuple(
                tuple(int(index) for index in indices.detach().cpu().tolist()) for indices, _ in groups
            )
            self._component_only_bbox_groups_cache[cache_key] = groups
            return groups

        # packed_ops kernels require tracks_idx in tracks_packinfo's
        # dtype (int32) — see interpolate_tracks_poses callers.
        all_idx = torch.arange(tracks.n_tracks, dtype=tracks.tracks_packinfo.dtype, device=tracks.cuboids_dims.device)
        expand_value = float(getattr(self.config.layers[layer_id], "cuboid_tracks_expand", 0.0) or 0.0)

        if layer_spec.bbox_track_ids is not None:
            # Per-recipe group: exactly these tracks, one box variant
            # (the recipe's sources decide raw vs expanded).
            id_to_idx = {tid: idx for idx, tid in enumerate(tracks.tracks_id)}
            idxs = [id_to_idx[tid] for tid in layer_spec.bbox_track_ids if tid in id_to_idx]
            if not idxs:
                return cache([])
            group_expand = expand_value if MaskSource.bbox_expanded in layer_spec.sources else 0.0
            return cache([(all_idx.new_tensor(idxs), group_expand)])

        track_states = self._component_only_track_states.get(layer_id)
        if track_states is None:
            if disabled_partition:
                # Unlisted and fully-disabled whole layers have no per-track
                # override map; their entire stored population is the disabled
                # occluder partition.
                if self._component_state(layer_id) != ComponentState.disabled:
                    return cache([])
                layer_expand = expand_value if MaskSource.bbox_expanded in layer_spec.sources else 0.0
                disabled_ids = [
                    idx
                    for idx, track_id in enumerate(tracks.tracks_id)
                    if track_id not in self._component_only_globally_rendered_track_ids
                ]
                return cache([(all_idx.new_tensor(disabled_ids), layer_expand)] if disabled_ids else [])
            layer_expand = expand_value if MaskSource.bbox_expanded in layer_spec.sources else 0.0
            return cache([(all_idx, layer_expand)])

        track_expanded = self._component_only_track_expanded[layer_id]

        def belongs_to_partition(track_id: str) -> bool:
            if disabled_partition:
                return (
                    track_states[track_id] == ComponentState.disabled
                    and track_id not in self._component_only_globally_rendered_track_ids
                )
            return track_states[track_id] != ComponentState.disabled

        selected = [
            (idx, track_expanded[track_id])
            for idx, track_id in enumerate(tracks.tracks_id)
            if belongs_to_partition(track_id)
        ]
        groups: list[tuple[torch.Tensor, float]] = []
        raw_ids = [idx for idx, is_expanded in selected if not is_expanded]
        exp_ids = [idx for idx, is_expanded in selected if is_expanded]
        if raw_ids:
            groups.append((all_idx.new_tensor(raw_ids), 0.0))
        if exp_ids:
            groups.append((all_idx.new_tensor(exp_ids), expand_value))
        return cache(groups)

    def _log_component_inventory(self, when: str = "construction") -> None:
        """INFO log listing constructed components with trainable-parameter counts.

        Makes it possible to verify from a single training-log line
        that component-only mode wired the right components. Called at
        construction (lazy layers show ``0 (lazy)``) and again post-init
        via ``GaussiansSystem.on_train_start`` once lazy parameters have
        materialized.
        """

        def count_and_lazy(module: Optional[torch.nn.Module]) -> tuple[int, bool]:
            if module is None:
                return 0, False
            count = 0
            has_lazy = False
            for p in module.parameters():
                if isinstance(p, torch.nn.parameter.UninitializedParameter):
                    has_lazy = True
                elif p.requires_grad:
                    count += p.numel()
            return count, has_lazy

        def fmt(count: int, has_lazy: bool) -> str:
            if has_lazy and count == 0:
                return "0 (lazy)"
            if has_lazy:
                return f"{count} (+ lazy)"
            return str(count)

        def layer_tag(name: str) -> str:
            return " (frozen)" if name in self._component_only_frozen_layers else ""

        layer_entries = ", ".join(
            f"'{name}': {fmt(*count_and_lazy(node))}{layer_tag(name)}" for name, node in self.gaussians_nodes.items()
        )
        sky_count, sky_lazy = count_and_lazy(self.background)
        calib_count, calib_lazy = count_and_lazy(self.calib)
        pp_types = [type(pp).__name__ for pp in self.post_processings]
        pp_count = sum(count_and_lazy(pp)[0] for pp in self.post_processings)
        pp_lazy = any(count_and_lazy(pp)[1] for pp in self.post_processings)

        mode = "COMPONENT-ONLY" if self.component_only is not None else "FULL"
        log.info(
            "Component inventory (%s, %s): gaussian_layers={%s} sky=%s(params=%s) "
            "calib=%s(params=%s) post_processings=%s(params=%s)",
            mode,
            when,
            layer_entries,
            type(self.background).__name__,
            fmt(sky_count, sky_lazy),
            type(self.calib).__name__,
            fmt(calib_count, calib_lazy),
            pp_types,
            fmt(pp_count, pp_lazy),
        )

    @staticmethod
    def _invalidate_fused_gather_generation_plan_before_load(
        module: GaussiansComposite,
        _state_dict: dict[str, Any],
        _prefix: str,
        _local_metadata: dict[str, Any],
        _strict: bool,
        _missing_keys: list[str],
        _unexpected_keys: list[str],
        _error_msgs: list[str],
    ) -> None:
        """Drain fused-gather work before state-dict tensor replacement."""
        module._invalidate_fused_gather_generation_plan()

    @staticmethod
    def _restore_initialization_skipped_layers(
        module: GaussiansComposite,
        state_dict: dict[str, Any],
        prefix: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Restore explicitly skipped optional layers before recursive strict loading.

        Parsed configs are saved before deferred initialization. A configured
        layer may therefore be absent from the checkpoint when its datasource
        contained no points. Use the saved skip inventory, never infer absence
        from missing parameters or reread the training datasource. Incoming
        metadata replaces process-local inventory; legacy checkpoints have none.
        """
        del args, kwargs
        extra_state = state_dict.get(f"{prefix}_extra_state")
        if not isinstance(extra_state, dict) or "initialization_skipped_layer_ids" not in extra_state:
            module._initialization_skipped_layer_ids = set()
            return
        skipped = extra_state["initialization_skipped_layer_ids"]
        if not isinstance(skipped, list) or any(
            not isinstance(layer_id, str) or not layer_id or "." in layer_id for layer_id in skipped
        ):
            raise ValueError("Checkpoint initialization_skipped_layer_ids must be a list of layer names")
        for layer_id in skipped:
            node_prefix = f"{prefix}gaussians_nodes.{layer_id}."
            visibility_key = f"{prefix}gaussians_strategy.invisible_steps.{layer_id}"
            if any(key.startswith(node_prefix) or key == visibility_key for key in state_dict):
                raise ValueError(f"Skipped Gaussian layer '{layer_id}' still has checkpoint state")
            if (
                module.component_only is not None
                and layer_id in module.config.layers
                and module._component_state(layer_id) != ComponentState.disabled
            ):
                raise ValueError(
                    f"component_only: non-disabled layer '{layer_id}' was skipped during checkpoint initialization. "
                    "It cannot be rendered or supervised; disable it or use a checkpoint containing the layer."
                )
        module._initialization_skipped_layer_ids = set(skipped)
        if module._prune_initialization_skipped_layers(module._initialization_skipped_layer_ids):
            # setup() may already have initialized the old strategy. Preserve
            # runtime-only buffers (e.g. perturb noise rates) in its replacement;
            # recursive loading below restores the saved persistent counters.
            module.gaussians_strategy.maybe_initialize_buffers(module.gaussians_nodes)

    @staticmethod
    def _reset_incompatible_mcmc_visibility_state_on_load(
        module: GaussiansComposite,
        state_dict: dict[str, Any],
        prefix: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        del args, kwargs
        invisible_steps = getattr(module.gaussians_strategy, "invisible_steps", None)
        if not isinstance(invisible_steps, torch.nn.ParameterDict) or len(invisible_steps) == 0:
            marker = f"{prefix}{_MCMC_VISIBILITY_STATE_MARKER}"
            for name in tuple(state_dict):
                if name.startswith(marker):
                    del state_dict[name]
            return
        reset_incompatible_mcmc_visibility_state(state_dict, warning=log.warning, prefix=prefix)

    def load_state_dict(self, state_dict: Mapping[str, Any], *args, **kwargs):
        # The sensor models in calib aren't stored in the state_dict, let's keep on using the one that
        # we have created in the constructor.

        # TODO: ideally this code should be in the load_state_dict of the model that
        # contains the sensor models (CameraFreePoseViewGeometry or LidarFreePoseViewGeometry)
        # and somehow called from here (through BaseCalib)

        calib_sensor_models = {f"calib.{k}": v for k, v in self.calib.state_dict().items() if ".sensor_models." in k}

        updated_state_dict = dict(state_dict)
        updated_state_dict.update(calib_sensor_models)
        return super().load_state_dict(updated_state_dict, *args, **kwargs)

    def _ensure_tracked_rows_grouped(self) -> None:
        """Give each track a contiguous row range when sparse segments are on. Once per model.

        Checkpoints captured before ``checkpoint.morton_sort.sort_track_layers`` (and
        compact artifacts) store tracked layers in MCMC order, where per-track windows
        degenerate to one window per row. A permutation fixes that once and changes
        nothing in what is rendered. Nothing downstream re-orders a rigid layer's rows
        -- edits mask and append whole tracks, and a plain Morton sort skips rigid
        layers -- so this is the one place the grouping is established, and
        ``LayerSegmentTable.build`` raises on it rather than degrading per frame.

        Deliberately not run from ``load_state_dict``. It changes row identity, and an
        asset-edit manifest addresses rows by index and by a digest of
        ``gaussian_cuboid_ids``: permuting at load makes every manifest fail its own
        install gate (``EditManifest.validate_against_layers``), which ``serve.py``
        turns into a refused artifact. The install runs after the model is loaded, so
        load time cannot know an edit is coming -- but the first sparse frame can, and
        ``_sparse_segment_plan_enabled`` refuses that combination outright, so a
        manifest-carrying model never reaches this.
        """
        if self._tracked_rows_grouped:
            return
        # morton_sort_gaussians rebinds parameters, invalidates caches and refreshes the
        # strategy, none of which a shell may do to state it shares with the model it was
        # cloned from (see context_composite._reset_per_context_state).
        if self._is_context_shell:
            raise RuntimeError(
                "sparse_segments needs its tracked rows grouped before the first context shell is cloned; "
                "call _ensure_tracked_rows_grouped on the shared model"
            )

        def ungrouped() -> list[str]:
            return [
                layer_id
                for layer_id, node in self.gaussians_nodes.items()
                if isinstance(node, RigidGaussianModel)
                and not isinstance(node.gaussian_cuboid_ids, torch.nn.UninitializedBuffer)
                and node.gaussian_cuboid_ids.numel() > 0
                and not rows_grouped_by_track(node.gaussian_cuboid_ids, int(node.cuboid_tracks.n_tracks))
            ]

        fragmented = ungrouped()
        if not fragmented:
            self._tracked_rows_grouped = True
            return
        log.info("sparse_segments: grouping the rows of %s by track", fragmented)
        # morton_sort_gaussians skips a layer whose order it cannot compute, so confirm
        # the repair landed here instead of letting the first frame trip the error.
        self.morton_sort_gaussians(sort_track_layers=True, only_layers=fragmented)
        still_fragmented = ungrouped()
        if still_fragmented:
            raise ValueError(
                f"sparse_segments could not group the rows of {still_fragmented} by track; "
                "re-capture the checkpoint with checkpoint.morton_sort.sort_track_layers"
            )
        self._tracked_rows_grouped = True

    @staticmethod
    def _refresh_mcmc_runtime_state_after_load(
        module: GaussiansComposite,
        incompatible_keys: Any,
    ) -> None:
        del incompatible_keys
        # assign=True replaces Parameter identities. Reinstall runtime hooks
        # first, then bind scenes to the final live objects. Module post-load
        # hooks run for direct composite loads and recursive root-system loads.
        module.reapply_component_only_runtime_state()
        module.refresh_mcmc_scenes()

    @staticmethod
    def _invalidate_derived_render_caches_after_load(
        module: GaussiansComposite,
        incompatible_keys: Any,
    ) -> None:
        del incompatible_keys
        # The collect tables and cached values are derived from checkpoint parameters.
        # This hook runs after both direct composite loads and recursive root-system
        # loads. Clear them here so same-shaped replacements cannot reuse the previous
        # checkpoint's values or track-row topology.
        module._invalidate_collect_caches()
        module._tracked_rows_grouped = False

        # Camera-to-LiDAR mappings bake calibrated poses into their geometry,
        # while the cache-hit check can only revalidate tensor grid shapes.
        # Invalidate before the first-training-step prewarm can reuse stale
        # geometry after either a direct or recursive load.
        module._cam_to_lidar_tile_mappings.clear()

    def refresh_mcmc_scenes(self) -> None:
        self.gaussians_strategy.refresh_runtime_scenes(self.gaussians_nodes)

    def raise_if_mcmc_publication_failed(self) -> None:
        for node in self.gaussians_nodes.values():
            if isinstance(node, BaseGaussianModel):
                node.raise_if_mcmc_publication_failed()

    @staticmethod
    def _raise_if_mcmc_publication_failed_before_state_dict(
        module: GaussiansComposite,
        prefix: str,
        keep_vars: bool,
    ) -> None:
        del prefix, keep_vars
        module.raise_if_mcmc_publication_failed()

    def configure_sharded_params_and_buffers(self) -> list[str]:
        ret = []
        for name, node in self.gaussians_nodes.items():
            per_node = node.configure_sharded_params_and_buffers()
            ret += [f"gaussians_nodes.{name}.{p}" for p in per_node]

        strategy_state = self.gaussians_strategy.configure_sharded_params_and_buffers()
        ret += [f"gaussians_strategy.{name}" for name in strategy_state]
        return ret

    def _initialize_scene_contractor(self, aabb: AABB3D) -> None:
        self._scene_contractor = SceneContractor(
            None,
            aabb,
            is_merf=False,
        )

    def initialize_strategy_and_maybe_gaussians(self) -> None:
        """
        Actually initialize Gaussians Models and Strategies, potentially in a distributed environment.
        """
        self._invalidate_fused_gather_generation_plan()
        skipped_nodes: set[str] = set()
        for name, node in self.gaussians_nodes.items():
            log.info(f"GaussiansComposite/maybe_initialize_gaussians: node={name}")
            try:
                node.maybe_initialize_gaussians()
            except NoPointsFoundException as error:
                if self.component_only is not None and self._component_state(name) != ComponentState.disabled:
                    raise ValueError(
                        f"component_only: non-disabled layer '{name}' could not be initialized because "
                        "deferred initialization found no points. It cannot be rendered or supervised; "
                        "disable it or fix its initialization."
                    ) from error
                log.warning("Skipping Gaussian node '%s' after deferred initialization: %s", name, error)
                skipped_nodes.add(name)

        self._prune_initialization_skipped_layers(skipped_nodes)

        # Background scope BEFORE the strategy sizes its per-row buffers: a
        # bbox-only scope physically drops out-of-frustum rows, so doing it
        # first lets the strategy size to the final row count and keeps
        # bbox-only background in densification (mixed-state scopes install
        # per-row hooks and are still excluded — see construction).
        self._apply_component_only_background_scope()
        self.gaussians_strategy.maybe_initialize_buffers(self.gaussians_nodes)
        # Then the runtime-only handlers (gradient hooks + freeze) — lazy init
        # may have (re)created parameter tensors. These same three are reapplied
        # after a checkpoint load (see reapply_component_only_runtime_state).
        self._apply_component_only_background_hooks()
        self._apply_component_only_frozen_tracks()
        self._apply_component_only_freeze()
        self.refresh_mcmc_scenes()

    def _prune_initialization_skipped_layers(self, skipped_nodes: set[str]) -> bool:
        """Align model/strategy/collector inventories; return whether the strategy was rebuilt."""
        live_layer_ids = set(self.gaussians_nodes)
        skipped_nodes = self._prune_unavailable_motion_dependents(
            self.gaussians_nodes,
            unavailable_layer_ids=skipped_nodes,
        )

        if skipped_nodes & live_layer_ids:
            strategy_config = copy.deepcopy(self._gaussians_strategy_config)
            strategy_config.exclude_layer_ids = [
                layer_id for layer_id in strategy_config.exclude_layer_ids if layer_id not in skipped_nodes
            ]
            relocate_config = getattr(strategy_config, "relocate", None)
            if relocate_config is not None:
                relocate_config.same_track_layer_ids = [
                    layer_id for layer_id in relocate_config.same_track_layer_ids if layer_id not in skipped_nodes
                ]
                relocate_config.min_gaussians_per_track = {
                    layer_id: minimum
                    for layer_id, minimum in relocate_config.min_gaussians_per_track.items()
                    if layer_id not in skipped_nodes
                }
            add_config = getattr(strategy_config, "add", None)
            if add_config is not None:
                add_config.fixed_count_layer_ids = [
                    layer_id for layer_id in add_config.fixed_count_layer_ids if layer_id not in skipped_nodes
                ]
            perturb_config = getattr(strategy_config, "perturb", None)
            if perturb_config is not None:
                perturb_config.max_position_offset_m = {
                    layer_id: radius
                    for layer_id, radius in perturb_config.max_position_offset_m.items()
                    if layer_id not in skipped_nodes
                }
            init_from_datasource = self.gaussians_strategy.init_from_datasource
            if not self.gaussians_nodes:
                raise NoPointsFoundException("No Gaussian nodes survived deferred initialization")

            self._bind_motion_source_layers(self.gaussians_nodes)
            self.gaussians_strategy = BaseGaussianStrategy.factory(
                strategy_config.name,
                strategy_config,
                self.trainer_config,
                init_from_datasource,
                self.gaussians_nodes,
            )
            self._layers_config = self._get_layers_config()
            self.collector = collect.CreateGaussianParameterCollector(self._layers_config)
            for node in self.gaussians_nodes.values():
                cast(SHGaussianModel, node).collector = self.collector
            return True
        return False

    def _apply_component_only_frozen_tracks(
        self, layer_ids: Optional[Collection[str]] = None, freeze_track_params: bool = True
    ) -> None:
        """Zero gradients for frozen tracks — Gaussian rows AND per-track params.

        With per-track subsetting a node holds only the non-disabled
        tracks; frozen tracks render but must not train. Two families of
        parameters carry a frozen track's state:

        * Per-Gaussian params (positions, scales, SH, ...) — the composite
          owns the row -> track mapping (the node's ``gaussian_cuboid_ids``
          buffer), so it zeros the frozen tracks' Gaussian rows here.
        * Per-track / per-pose params (``track_albedos``/``track_scales``,
          ``tracks_calib`` pose deltas, the deformation network, ...) — each
          indexed by its own scheme, which only the owning module knows, so
          the freeze is delegated to ``node.freeze_tracks``.

        The layer is already excluded from densification (row alignment
        stays stable).

        ``layer_ids`` restricts the work to those layers, and None covers all
        of them.
        ``freeze_track_params`` selects both families above, or only
        the per-Gaussian rows when False. The morton row reorder passes the layers
        whose parameters it replaced and asks for rows only, because a
        permutation moves rows but never touches track-indexed parameters, and
        re-registering a hook on a tensor that was not replaced would leave it
        holding a second copy.
        """
        for layer_id, states in self._component_only_track_states.items():
            if layer_id not in self.gaussians_nodes:
                continue
            if layer_ids is not None and layer_id not in layer_ids:
                continue
            node = self.gaussians_nodes[layer_id]
            node_tracks = getattr(node, "cuboid_tracks", None)
            ids_buffer = getattr(node, "gaussian_cuboid_ids", None)
            if (
                node_tracks is None
                or ids_buffer is None
                or isinstance(ids_buffer, torch.nn.parameter.UninitializedBuffer)
            ):
                continue
            frozen_local = [
                idx for idx, tid in enumerate(node_tracks.tracks_id) if states[tid] == ComponentState.frozen
            ]
            if not frozen_local:
                continue
            frozen_rows = torch.isin(
                ids_buffer, torch.tensor(frozen_local, dtype=ids_buffer.dtype, device=ids_buffer.device)
            )
            trainable = ~frozen_rows
            log.info(
                f"component_only: layer '{layer_id}' freezes {len(frozen_local)} track(s) — "
                f"{int(frozen_rows.sum())} / {frozen_rows.shape[0]} Gaussian rows frozen via gradient hook"
            )
            # Per-Gaussian params (positions, scales, SH, ...): row -> track
            # via gaussian_cuboid_ids is the composite's to own.
            for param in node.parameters(recurse=False):
                if param.requires_grad and param.shape[:1] == trainable.shape:
                    param.register_hook(_make_row_grad_hook(trainable))
            # Per-track / per-pose params (track_albedos/scales, tracks_calib
            # pose deltas, deformation network): only the owning module knows
            # the indexing, so delegate the freeze to the node.
            # Callers that only replaced per-Gaussian rows skip this, track-indexed hooks stay valid.
            if freeze_track_params:
                node.freeze_tracks(frozen_local)

    def _background_bbox_region_rows(self, positions: torch.Tensor) -> torch.Tensor:
        """(N,) bool — which background Gaussians lie in the bbox scope region.

        A point is in the region iff, from at least one sampled rig
        viewpoint, the ray through it intersects a non-disabled
        bbox-masked sibling's box (per that sibling's configured
        variant) at the viewpoint's timestamp — i.e. the point lies in
        the frustum of pixels the siblings' masks supervise. This is
        the 3D counterpart of the pixel-level bbox masks, sampled at
        the stored rig viewpoints instead of every frame.

        NOTE: this region is computed once (from the construction-time rig
        viewpoints and track poses). If camera/track calibration is optimized
        during training the frustum shifts and the region can go slightly
        stale — accepted as an approximation for the (unshipped) bbox-scope
        path: drift is small vs. the coarse frustum test and densification
        fills any gaps. Recomputing it periodically would be a follow-up.
        """
        spec = self._component_only_mask_spec
        viewpoint_positions, viewpoint_timestamps = self._component_only_viewpoints
        n_points = positions.shape[0]
        in_region = torch.zeros(n_points, dtype=torch.bool, device=positions.device)
        if spec is None or viewpoint_positions.shape[0] == 0:
            return in_region

        # Every supervised bbox region group across all siblings spans
        # the frustum (a group is one per-recipe partition of a layer).
        bbox_groups = [
            (layer_id, group)
            for layer_id, groups in spec.supervised_layers.items()
            for group in groups
            if group.needs_bbox
        ]
        for layer_id, layer_spec in bbox_groups:
            tracks = self._component_only_layer_tracks.get(layer_id)
            if tracks is None or tracks.n_tracks == 0:
                continue
            # Per-track subsetting: only the supervised (non-disabled)
            # partition's boxes span the region — disabled tracks'
            # pixels are carved from the residual, not supervised.
            for tracks_idx, expand in self._component_only_bbox_track_groups(layer_id, layer_spec):
                for view in range(viewpoint_positions.shape[0]):
                    origin = viewpoint_positions[view].to(positions.device, positions.dtype)
                    rays_o = origin.expand(n_points, 3)
                    rays_d = positions - origin
                    rays_ts = torch.full(
                        (n_points,),
                        int(viewpoint_timestamps[view].item()),
                        dtype=torch.int64,
                        device=positions.device,
                    )
                    in_region |= self._component_only_ray_cuboid_hits(
                        tracks, tracks_idx, expand, rays_o, rays_d, rays_ts
                    ).mask
        return in_region

    def _apply_component_only_background_scope(self) -> None:
        """Realize background's scope regions on its Gaussian population.

        * ``{residual: X}`` only — no geometric action: the standard
          initialization approximates the complement region and the loss
          mask governs supervision (v1 behavior).
        * ``{bbox: X}`` only — background exists ONLY in the frustum
          region: rows outside it are dropped after initialization.
        * both regions, same state — keep all rows (frustum rows are the
          bbox population, the rest is residual), no special handling.
        * both regions, mixed states — keep all rows and install a
          gradient hook zeroing the frozen region's rows; densification
          is already disabled for background in this case (see
          construction).
        """
        scope = {r: s for r, s in self._background_scope().items() if s != ComponentState.disabled}
        if self.component_only is None or ScopeRegion.bbox not in scope:
            return
        if BACKGROUND_LAYER_ID not in self.gaussians_nodes:
            return
        node = self.gaussians_nodes[BACKGROUND_LAYER_ID]
        # During checkpoint-based val/test Lightning calls ``setup`` before
        # loading the state dict. The bbox scope was already realized before
        # the checkpoint was saved, so leave the lazy placeholder untouched;
        # checkpoint loading will materialize the scoped tensors directly.
        if isinstance(node.positions, torch.nn.parameter.UninitializedParameter):
            return
        if node.get_num_gaussians() == 0:
            return

        in_bbox_region = self._background_bbox_region_rows(node.get_positions().detach())

        if ScopeRegion.residual not in scope:
            # bbox-only scope: drop everything outside the frustums. This is
            # STRUCTURAL (changes the row count) and is baked into the
            # checkpoint — it is done once at construction and never on resume.
            keep = in_bbox_region
            log.info(
                "component_only: background scope [bbox] keeps %d / %d Gaussians (frustum region)",
                int(keep.sum()),
                keep.shape[0],
            )
            for name, param in list(node.named_parameters(recurse=False)):
                if param.shape[:1] == keep.shape:
                    setattr(node, name, torch.nn.Parameter(param.data[keep], requires_grad=param.requires_grad))
            return

    def _apply_component_only_background_hooks(self, layer_ids: Optional[Collection[str]] = None) -> None:
        """Install the mixed-region background gradient hook (runtime-only).

        For a background ``scope`` with *mixed* region states (one region
        enabled, the other frozen) the frozen region's rows are zeroed by a
        gradient hook — no structural change. Kept separate from the bbox-only
        row drop so it can be reapplied after a checkpoint load without redoing
        the (already-baked) row surgery.

        ``layer_ids`` restricts the install to those layers, and None covers all
        of them. The morton row reorder passes the layers whose parameters it replaced,
        because re-registering on a layer that kept its tensors would leave that
        layer holding a second copy of the hook.
        """
        scope = {r: s for r, s in self._background_scope().items() if s != ComponentState.disabled}
        if self.component_only is None or ScopeRegion.bbox not in scope or ScopeRegion.residual not in scope:
            return
        if len(set(scope.values())) == 1:
            return  # uniform state: freeze (if any) is handled layer-wide
        if BACKGROUND_LAYER_ID not in self.gaussians_nodes:
            return
        if layer_ids is not None and BACKGROUND_LAYER_ID not in layer_ids:
            return
        node = self.gaussians_nodes[BACKGROUND_LAYER_ID]
        if isinstance(node.positions, torch.nn.parameter.UninitializedParameter) or node.get_num_gaussians() == 0:
            return

        in_bbox_region = self._background_bbox_region_rows(node.get_positions().detach())
        frozen_rows = in_bbox_region if scope[ScopeRegion.bbox] == ComponentState.frozen else ~in_bbox_region
        trainable = ~frozen_rows
        log.info(
            "component_only: background mixed region states — %d / %d rows trainable (bbox region: %s, residual: %s)",
            int(trainable.sum()),
            trainable.shape[0],
            scope[ScopeRegion.bbox].value,
            scope[ScopeRegion.residual].value,
        )
        for param in node.parameters(recurse=False):
            if param.requires_grad and param.shape[:1] == trainable.shape:
                param.register_hook(_make_row_grad_hook(trainable))

    def reapply_component_only_runtime_state(self) -> None:
        """Reinstall the runtime-only component-only state after a state-dict load.

        The freeze (``requires_grad_(False)``) and the gradient hooks are NOT
        part of the state dict, and ``load_state_dict(assign=True)`` replaces
        the very tensor objects the hooks were bound to. On resume the initial
        setup ran while parameters were still lazy (every handler no-op'd), so
        this reapplies them once the checkpoint has materialized the tensors.
        The STRUCTURAL bbox-only row drop is deliberately NOT redone — the
        checkpoint already stores the reduced tensors.
        """
        if self.component_only is None:
            return
        self._apply_component_only_background_hooks()
        self._apply_component_only_frozen_tracks()
        self._apply_component_only_freeze()

    def reapply_component_only_row_hooks(self, layer_ids: Collection[str]) -> None:
        """Reinstall only the per-Gaussian row hooks, for layers whose rows were permuted.

        A gradient hook is bound to the tensor object. Reinstalling one where nothing was replaced leaves that
        tensor holding more than one. Only the per-Gaussian parameters of these layers
        were replaced, so the reinstall stops there. The per-track parameters are indexed by track, so a
        permutation never touches them and their hooks stay valid. The layer-wide freeze is
        requires_grad_(False), which the replacement parameters inherit.
        """
        if self.component_only is None:
            return
        self._apply_component_only_background_hooks(layer_ids)
        self._apply_component_only_frozen_tracks(layer_ids, freeze_track_params=False)

    def _apply_component_only_freeze(self) -> None:
        """Turn off gradients for all frozen components.

        Frozen = constructed + rendered + mask contribution, but no
        optimization: ``requires_grad_(False)`` on every parameter
        (optimizers skip grad-less parameters), and the layer is on the
        strategy's exclude list so densification/pruning never touches
        it. Uninitialized (lazy) parameters are skipped — this method is
        re-invoked after lazy init materializes them.
        """
        if self.component_only is None:
            return

        def _freeze(module: torch.nn.Module, what: str) -> None:
            n_frozen = 0
            for param in module.parameters():
                if isinstance(param, torch.nn.parameter.UninitializedParameter):
                    continue
                param.requires_grad_(False)
                n_frozen += param.numel()
            log.info(f"component_only: froze '{what}' ({n_frozen} parameters)")

        for layer_id in self._component_only_frozen_layers:
            _freeze(self.gaussians_nodes[layer_id], f"layer {layer_id}")
        if self._component_state("sky_model") == ComponentState.frozen:
            _freeze(self.background, "sky_model")
        if self._component_state("calib") == ComponentState.frozen:
            _freeze(self.calib, "calib")
        if self._component_state("post_processing") == ComponentState.frozen:
            for pp in self.post_processings:
                _freeze(pp, f"post_processing {type(pp).__name__}")

    def get_scene_contractor(self) -> SceneContractor:
        return self._scene_contractor

    def get_gaussians_node_ids(self, non_empty_only: bool = True) -> list[str]:
        keys = list(self.gaussians_nodes.keys())
        if non_empty_only:
            keys = [k for k in keys if not self.obj_track_ids[k].is_empty()]
        return keys

    def set_edit_overrides(self, runtime: Optional[GaussianEditOverrides]) -> None:
        """Install (or clear) the asset-edits override runtime.

        The runtime is invoked from ``_render_collected`` on every render pass
        and modifies the collected ``densities`` tensor (opacity toggles/gains)
        and ``features`` tensor (SH color warm-up fades) in place to drive the
        baked overlays. Call with ``None`` to disable overrides entirely.
        """
        self._edit_overrides = runtime

    def _collect_sensor(
        self,
        rendering_data: RenderingData,
        is_training_batch: bool,
        gaussian_nodes: Optional[list[str]],
        track_transform_ids: Optional[List[str]],
        track_transforms: Optional[torch.Tensor],
        edited_cuboid_tracks: Optional[CuboidTracks],
        sensor_name: str,
        combined_sensor_forward: bool,
        frustum_cull: Optional[collect.TracksFrustumCullData] = None,
        gathered_parameters: Optional[GatheredGaussianParameters] = None,
    ) -> tuple[dict[str, torch.Tensor], SegmentPlan | None]:
        """
        Collect gaussian parameters for a single sensor.

        This is the first phase of rendering - gathering all gaussian data
        needed for the actual render call.

        The parameter tensors use Gaussian rows on axis 0. The optional plan indexes
        the same row axis.

        Returns:
            A pair. Item 0 maps each parameter name to its tensor. Item 1 contains
            the segment plan, or ``None`` for a dense render.
        """
        with ScopedTimer(f"GaussiansComposite/collect_gaussian_parameters_{sensor_name}"):
            return self._collect_gaussian_parameters_with_segment_plan(
                rendering_data,
                is_training_batch,
                gaussian_nodes,
                track_transform_ids,
                track_transforms,
                edited_cuboid_tracks,
                frustum_cull,
                gathered_parameters,
                combined_sensor_forward=combined_sensor_forward,
                is_camera=sensor_name == "cam",
                sensor_name=sensor_name,
                build_segment_plan=self._sparse_segment_plan_enabled(
                    is_training_batch=is_training_batch,
                    sensor_name=sensor_name,
                    has_active_tile_mask=rendering_data.active_tile_mask is not None,
                    combined_sensor_forward=combined_sensor_forward,
                ),
            )

    def _sparse_segment_plan_enabled(
        self,
        *,
        is_training_batch: bool,
        combined_sensor_forward: bool = False,
        sensor_name: str = "cam",
        has_active_tile_mask: bool = False,
    ) -> bool:
        """Return whether this collect must publish a sparse draw plan.

        Invalid sparse configurations raise an error instead of using dense rendering.
        Only what varies per frame is decided here: ``GSplatRendererConfig`` has already
        rejected the unsupported settings that are fixed for the run, so reaching this
        with ``sparse_segments`` on means the renderer is 3dgut-gsplat and none of
        packed, sparse_grad, absgrad, scene visibility or unstructured LiDAR is set.
        """
        renderer_cfg = self.config.renderer
        if not renderer_cfg.sparse_segments or not renderer_cfg.cache_static_gaussians:
            return False
        if self._edit_overrides is not None:
            # An installed manifest addresses rows by index and by a digest of
            # gaussian_cuboid_ids, and _ensure_tracked_rows_grouped may permute them.
            # Re-baking does not settle it: a bake re-digests whatever order it is given,
            # so the digest moves again the next time sparse_segments is on. Re-capturing
            # with checkpoint.morton_sort.sort_track_layers is what converges both
            # settings on one row order.
            raise ValueError(
                "sparse_segments does not support an installed asset-edit manifest. Re-capture the "
                "checkpoint with checkpoint.morton_sort.sort_track_layers so both settings share one "
                "row order."
            )
        outputs = renderer_cfg.outputs.camera if sensor_name == "cam" else renderer_cfg.outputs.lidar
        layers_config = self._layers_config
        common_extra_dim = layers_config.extra_signal_dim
        sensor_extra_dim = (
            layers_config.camera_extra_signal_dim if sensor_name == "cam" else layers_config.lidar_extra_signal_dim
        )
        # The renderer consults this same list before it draws the plan this collect builds.
        assert_sparse_segments_supported(
            has_extra_signals=(outputs.enable_extended_features and common_extra_dim > 0)
            or (outputs.enable_sensor_features and sensor_extra_dim > 0),
            has_active_tile_mask=has_active_tile_mask,
            grad_enabled=self.training or is_training_batch or torch.is_grad_enabled(),
        )
        if combined_sensor_forward:
            raise ValueError("sparse_segments does not support a combined camera and LiDAR forward.")
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        if world_size != 1:
            raise ValueError("sparse_segments does not support distributed rendering.")
        # Last, so that every refusal above is decided before any row moves.
        self._ensure_tracked_rows_grouped()
        return True

    @staticmethod
    def _build_frustum_cull(rendering_data: RenderingData) -> collect.TracksFrustumCullData:
        return collect.TracksFrustumCullData(
            rays=rendering_data.rays.view(-1, 6),
            rays_height=int(rendering_data.h),
            rays_width=int(rendering_data.w),
        )

    def _resolve_visibility_counts(
        self,
        *,
        node_id: str,
        world_size: int,
        world_rank: int,
        local_count: int,
        device: torch.device,
        gathered_counts: Optional[tuple[int, ...]],
    ) -> tuple[int, ...]:
        """Reuse fused counts when available, gathering for non-fused paths."""
        all_counts = (
            tuple(
                all_gather_int32(
                    world_size,
                    local_count,
                    device=device,
                    group=self._get_gaussian_process_group(),
                )
            )
            if gathered_counts is None
            else gathered_counts
        )
        if len(all_counts) != world_size:
            raise RuntimeError(f"Expected {world_size} gathered counts for {node_id}, got {len(all_counts)}")
        if all_counts[world_rank] != local_count:
            raise RuntimeError(
                f"Gathered count mismatch for {node_id}: rank {world_rank} count "
                f"{all_counts[world_rank]} != local count {local_count}"
            )
        return all_counts

    def _get_gaussian_process_group(self) -> Optional[torch.distributed.ProcessGroup]:
        """Return the communicator borrowed from its owning DDP strategy."""
        if not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1:
            return None
        group = self.gaussian_process_group
        if group is None:
            raise RuntimeError("Distributed Gaussian setup did not install the dedicated process group")
        return group

    def drain_fused_gather_generation_plan_refresh(self) -> None:
        """Drain an outstanding refresh before validation or process-group teardown."""
        pending = getattr(self, "_pending_fused_gather_generation_plan_refresh", None)
        self._pending_fused_gather_generation_plan_refresh = None
        self._fused_gather_generation_plan = None
        if pending is not None:
            _discard_layer_count_gather(pending.count_gather)

    def _invalidate_fused_gather_generation_plan(self) -> None:
        """Retire cached topology at a rank-symmetric lifecycle boundary."""
        self.drain_fused_gather_generation_plan_refresh()

    def _supports_fused_gather_generation_plan(self) -> bool:
        """Restrict shape-schedule ownership to the two known exact strategies."""
        return type(getattr(self, "gaussians_strategy", None)) in (GSplatStrategy, MCMCStrategy, ECADStrategy)

    def _is_fused_gather_generation_refresh_step(self, global_step: int) -> bool:
        """Ask whether batch end may change per-layer row counts or layout."""
        if not self._supports_fused_gather_generation_plan():
            raise RuntimeError("Generation-plan refresh queried for an unsupported Gaussian strategy")
        return self.gaussians_strategy.may_mutate_topology_at(global_step)

    def _finish_fused_gather_generation_plan_refresh(self) -> Optional[_FusedGatherGenerationPlan]:
        """Resolve an unpublished post-mutation generation candidate."""
        pending = self._pending_fused_gather_generation_plan_refresh
        if pending is None:
            return None

        self._pending_fused_gather_generation_plan_refresh = None
        try:
            plan = _make_fused_gather_generation_plan(
                pending.prepared_inputs,
                _finish_layer_count_gather(pending.count_gather),
                process_group=self._get_gaussian_process_group(),
            )
        except BaseException:
            self._fused_gather_generation_plan = None
            raise
        return plan

    def _refresh_fused_gather_generation_plan_after_mutations(self, global_step: int) -> None:
        """Refresh rank-agreed metadata after a known structural mutation."""
        pending = self._pending_fused_gather_generation_plan_refresh
        if pending is not None:
            # A skipped forward never consumed the prior candidate.  It cannot
            # be published, and this batch may have changed structure again.
            self._pending_fused_gather_generation_plan_refresh = None
            _discard_layer_count_gather(pending.count_gather)
            self._fused_gather_generation_plan = None
            return

        plan = self._fused_gather_generation_plan
        if plan is None:
            return
        if not self._supports_fused_gather_generation_plan():
            self._invalidate_fused_gather_generation_plan()
            return
        if not torch.distributed.is_initialized():
            self._invalidate_fused_gather_generation_plan()
            return
        process_group = self._get_gaussian_process_group()
        if (
            process_group is None
            or torch.distributed.get_backend(process_group) != torch.distributed.Backend.NCCL
            or torch.distributed.get_rank(process_group) != plan.world_rank
            or torch.distributed.get_world_size(process_group) != plan.world_size
        ):
            self._invalidate_fused_gather_generation_plan()
            return
        if not self._is_fused_gather_generation_refresh_step(global_step):
            return

        # The just-completed mutation invalidates the published generation.
        # Keep the cache empty until the candidate payload is consumed.
        self._fused_gather_generation_plan = None

        gaussian_nodes = tuple(layout.node_id for layout in plan.layouts)
        if tuple(self.get_gaussians_node_ids()) != gaussian_nodes:
            self._invalidate_fused_gather_generation_plan()
            return
        parameters_by_node: dict[str, Mapping[str, torch.Tensor]] = {
            node_id: self.gaussians_nodes[node_id].get_parameters() for node_id in gaussian_nodes
        }
        prepared = _prepare_fused_gather_inputs(
            list(gaussian_nodes),
            parameters_by_node,
            process_group=process_group,
        )
        if not _can_prefetch_grouped_payload(prepared):
            self._invalidate_fused_gather_generation_plan()
            return

        with ScopedTimer("GaussiansComposite/launch_generation_plan_refresh"):
            count_gather = _start_layer_count_gather(
                prepared.world_size,
                prepared.local_counts,
                prepared.device,
                prepared.schema_hash,
                process_group=process_group,
            )
        try:
            self._pending_fused_gather_generation_plan_refresh = _PendingGaussianGenerationPlanRefresh(
                prepared_inputs=prepared,
                count_gather=count_gather,
            )
        except BaseException as wrapper_error:
            try:
                _discard_layer_count_gather(count_gather)
            except BaseException as drain_error:
                wrapper_error.add_note(f"Additionally failed to drain launched Gaussian count work: {drain_error!r}")
            raise

    def _maybe_prefetch_gaussian_parameter_gather(
        self,
        *,
        is_training_batch: bool,
        gaussian_nodes: Optional[list[str]],
        track_transform_ids: Optional[List[str]],
        track_transforms: Optional[torch.Tensor],
        edited_cuboid_tracks: Optional[CuboidTracks],
    ) -> Optional[_PendingGaussianParameterGather]:
        """Capture warmup inputs or prelaunch a published generation.

        Custom strategies, subclasses, over-capacity node layouts, and locally
        ineligible schemas return ``None`` and use the established direct fused
        gather at the collector boundary.  That keeps their metadata decision
        immediately adjacent to the payload collective and avoids speculative
        snapshot-consensus machinery.  Eligible camera-only, lidar-only,
        dual-sensor, and single-node training forwards all take this grouped
        ``_FusedGaussianParameterGather``/ReduceScatter route: modality selects
        only the downstream collector and renderer, while every rank
        communicates the same full topology.
        """
        if (
            not is_training_batch
            or not self.training
            or not torch.is_grad_enabled()
            or type(self.calib) is not FreePoseCalib
            or gaussian_nodes is not None
            or track_transform_ids is not None
            or track_transforms is not None
            or edited_cuboid_tracks is not None
            or not torch.distributed.is_initialized()
            or torch.distributed.get_world_size() <= 1
            or torch.distributed.get_backend() != torch.distributed.Backend.NCCL
        ):
            # Batch end may have launched an asynchronous count refresh.  An
            # ineligible forward cannot consume or publish it, so release the
            # captured parameter references and wait before returning.  The
            # drain clears state before waiting: an error propagates fail-closed
            # and a later invalidation remains idempotent.
            self.drain_fused_gather_generation_plan_refresh()
            return None

        if not self._supports_fused_gather_generation_plan():
            self._invalidate_fused_gather_generation_plan()
            return None

        selected_nodes = tuple(self.get_gaussians_node_ids())
        if not _can_route_node_selection(selected_nodes):
            self.drain_fused_gather_generation_plan_refresh()
            return None

        refresh_candidate = self._finish_fused_gather_generation_plan_refresh()
        publish_on_success = refresh_candidate is not None
        plan = refresh_candidate if refresh_candidate is not None else self._fused_gather_generation_plan
        if plan is not None and tuple(layout.node_id for layout in plan.layouts) != selected_nodes:
            self._invalidate_fused_gather_generation_plan()
            plan = None
            publish_on_success = True
        parameters_by_node: dict[str, Mapping[str, torch.Tensor]] = {
            node_id: self.gaussians_nodes[node_id].get_parameters() for node_id in selected_nodes
        }
        prefetched_payload_gather: Optional[_PendingFusedPayloadGather] = None
        try:
            if plan is not None:
                prepared_inputs = _prepare_fused_gather_inputs_from_generation_plan(
                    list(selected_nodes),
                    parameters_by_node,
                    plan,
                    process_group=self._get_gaussian_process_group(),
                )
                with ScopedTimer("GaussiansComposite/launch_prefetched_parameter_payload"):
                    prefetched_payload_gather = _start_fused_payload_gather_from_generation_plan(
                        prepared_inputs,
                        plan,
                        process_group=self._get_gaussian_process_group(),
                    )
            else:
                prepared_inputs = _prepare_fused_gather_inputs(
                    list(selected_nodes),
                    parameters_by_node,
                    process_group=self._get_gaussian_process_group(),
                )
                if not _can_prefetch_grouped_payload(prepared_inputs):
                    return None
                # Generation zero performs no communication before FreePose.
                # The collector boundary resolves metadata, launches the
                # payload, and publishes only after the fused gather succeeds.
                publish_on_success = True

            return _PendingGaussianParameterGather(
                gaussian_nodes=selected_nodes,
                parameters_by_node=parameters_by_node,
                prepared_inputs=prepared_inputs,
                generation_plan=plan,
                prefetched_payload_gather=prefetched_payload_gather,
                publish_on_success=publish_on_success,
            )
        except BaseException as error:
            self._fused_gather_generation_plan = None
            if prefetched_payload_gather is not None:
                try:
                    _discard_fused_payload_gather(prefetched_payload_gather)
                except BaseException as drain_error:
                    error.add_note(f"Additionally failed to drain launched Gaussian payload work: {drain_error!r}")
            if plan is not None:
                # A cached-plan mismatch is a fatal distributed invariant
                # violation. Peers may already be launching the same payload;
                # this rank must not locally retry through a different
                # collective sequence on the current process group.
                error.add_note(
                    "Cached Gaussian generation invariant failed; terminate all ranks and restart with a fresh process group"
                )
            raise

    def _gather_gaussian_parameters_once(
        self,
        gaussian_nodes: list[str],
        prefetched: Optional[_PendingGaussianParameterGather] = None,
    ) -> GatheredGaussianParameters:
        if prefetched is not None and tuple(gaussian_nodes) != prefetched.gaussian_nodes:
            prefetched.discard()
            self._fused_gather_generation_plan = None
            raise RuntimeError(
                "prefetched Gaussian node order differs from the same-forward joint gather; "
                f"prefetched {prefetched.gaussian_nodes}, gathering {tuple(gaussian_nodes)}"
            )
        if not gaussian_nodes:
            if prefetched is not None:
                prefetched.discard()
                self._fused_gather_generation_plan = None
            return {}

        def gather_per_node() -> GatheredGaussianParameters:
            gathered_parameters: GatheredGaussianParameters = {}
            for node_id in gaussian_nodes:
                with ScopedTimer(f"GaussiansComposite/gather_parameters[{node_id}]"):
                    params = self.gaussians_nodes[node_id].get_parameters()
                    local_count = next(iter(params.values())).shape[0]
                    gathered_parameters[node_id] = _GatheredGaussianParameterSet(
                        parameters=distributed_all_gather_gaussian_parameters(
                            params,
                            group=self._get_gaussian_process_group(),
                        ),
                        local_count=local_count,
                    )
            return gathered_parameters

        if not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1:
            if prefetched is not None:
                prefetched.discard()
                self._fused_gather_generation_plan = None
                raise RuntimeError("prefetched Gaussian gathering requires an initialized multi-rank process group")
            return gather_per_node()
        if len(gaussian_nodes) == 1 and prefetched is None:
            return gather_per_node()
        if len(set(gaussian_nodes)) != len(gaussian_nodes):
            if prefetched is not None:
                prefetched.discard()
                self._fused_gather_generation_plan = None
            raise RuntimeError("distributed Gaussian gathering requires distinct Gaussian nodes")

        params_by_node = (
            prefetched.parameters_by_node
            if prefetched is not None
            else {node_id: self.gaussians_nodes[node_id].get_parameters() for node_id in gaussian_nodes}
        )
        # This private fused path relies on the collector below: every selected
        # node becomes one collector layer, and CollectorFunction.backward
        # materializes a gradient for every layer on every rank. That preserves
        # the same rank-symmetric whole-layer participation required by the
        # established per-layer distributed gather backward.
        integer_range_validation_state = getattr(
            self,
            "_fused_payload_integer_range_validation_state",
            None,
        )
        if integer_range_validation_state is None:
            # Support lightweight ``__new__``-constructed composites used by
            # tests and internal tooling without weakening production reuse.
            integer_range_validation_state = _PayloadIntegerRangeValidationState()
            self._fused_payload_integer_range_validation_state = integer_range_validation_state
        generation_plan = None if prefetched is None else prefetched.generation_plan
        payload_gather = None if prefetched is None else prefetched.prefetched_payload_gather
        if prefetched is not None and generation_plan is None:
            try:
                generation_plan = _resolve_fused_gather_generation_plan(
                    prefetched.prepared_inputs,
                    process_group=self._get_gaussian_process_group(),
                )
                payload_gather = _start_fused_payload_gather_from_generation_plan(
                    prefetched.prepared_inputs,
                    generation_plan,
                    process_group=self._get_gaussian_process_group(),
                )
            except BaseException:
                self._fused_gather_generation_plan = None
                raise

        try:
            gathered = _fused_all_gather_gaussian_parameters(
                gaussian_nodes,
                params_by_node,
                process_group=self._get_gaussian_process_group(),
                integer_range_validation_state=integer_range_validation_state,
                prepared_inputs=None if prefetched is None else prefetched.prepared_inputs,
                generation_plan=generation_plan,
                prefetched_payload_gather=payload_gather,
            )
        except BaseException as gather_error:
            if payload_gather is not None:
                try:
                    _discard_fused_payload_gather(payload_gather)
                except BaseException as drain_error:
                    gather_error.add_note(f"Additionally failed to drain prefetched payload: {drain_error!r}")
            if prefetched is not None:
                self._fused_gather_generation_plan = None
            raise
        if prefetched is not None and prefetched.publish_on_success:
            assert generation_plan is not None
            self._fused_gather_generation_plan = generation_plan

        return {
            node_id: _GatheredGaussianParameterSet(
                parameters=gathered.parameters_by_node[node_id],
                local_count=next(iter(params_by_node[node_id].values())).shape[0],
                counts_by_rank=gathered.counts_by_node[node_id],
            )
            for node_id in gaussian_nodes
        }

    @staticmethod
    def _record_gathered_parameters_stream(
        gathered_parameters: Optional[GatheredGaussianParameters],
        stream: torch.cuda.Stream,
    ) -> None:
        if gathered_parameters is None:
            return

        for gathered in gathered_parameters.values():
            for value in gathered.parameters.values():
                if value.is_cuda:
                    value.record_stream(stream)

    def _render_collected(
        self,
        rendering_data: RenderingData,
        frame_meta: List[FrameMeta],
        gaussian_parameters: dict[str, torch.Tensor],
        extra_ray_signal_infos: tuple,
        renderer: BaseGaussianRenderer,
        is_training_batch: bool,
        sensor_name: str,
        segment_plan: SegmentPlan | None = None,
    ) -> GaussiansRenderReturn:
        """
        Render using pre-collected gaussian parameters.

        This is the second phase of rendering - the actual GPU render call.
        The optional plan selects ranges on axis 0 of each Gaussian parameter tensor.
        """
        # This sensor's own collect record, consumed here so a render can never reuse a
        # stale one (see the write site in collect_gaussian_parameters).
        edit_layouts = self._collected_edit_layouts.pop(sensor_name, None)
        if self._edit_overrides is not None:
            if edit_layouts is None:
                # collect_gaussian_parameters records layouts on every collect while a
                # hook is installed, so this means the render had no matching collect.
                raise RuntimeError(
                    f"asset-edit overrides are installed, but the {sensor_name!r} collect "
                    "published no layer layouts for this render."
                )
            with ScopedTimer(f"GaussiansComposite/edit_overrides_{sensor_name}"):
                try:
                    self._edit_overrides.apply(
                        gaussian_parameters,
                        frame_timestamp_us=BaseGaussianModel.get_frame_timestamp(rendering_data),
                        layouts=edit_layouts,
                    )
                except Exception:
                    # Fail the frame rather than render it. apply() mutates the collected
                    # densities/features in place, so a fault partway through leaves a torn
                    # mix of aspect states, not an unedited scene. The hook stays installed
                    # because the fault may be transient; re-raise unchanged so the caller
                    # sees the real error (an OOM stays an OOM).
                    log.exception("asset-edit overrides failed on the %s frame", sensor_name)
                    raise

        with ScopedTimer(f"GaussiansComposite/render_{sensor_name}"):
            out = renderer.render(
                rendering_data=rendering_data,
                gaussian_parameters=gaussian_parameters,
                n_active_features=self.get_n_active_features(),
                extra_ray_signal_infos=extra_ray_signal_infos,
                frame_meta=frame_meta,
                segment_plan=segment_plan,
            )

        if not is_training_batch:
            # Normalize the normal to unit length for inference
            out.normal = torch.nn.functional.normalize(out.normal, dim=-1) if out.normal is not None else None

        return out

    def _compact_scene_data_to_local(self, out: Optional[GaussiansRenderReturn]) -> None:
        """Compact every scene-level field to the local rank's contiguous layer ranges.

        In multi-GPU training, the renderer operates on all-gathered gaussians and produces
        per-gaussian scene_data of global size (total across all ranks). Each layer stores
        ranks contiguously, so the local shard is at most one contiguous range per layer.
        Concatenating those views avoids materializing and reading an O(local-gaussians)
        int64 advanced-index tensor every step.

        The loop is driven off ``GaussiansRenderReturn._SCENE_LEVEL_FIELDS`` rather than a
        hand-written list so a newly added per-gaussian field is compacted automatically:
        leaving one at global length silently desynchronizes it from ``visibility``
        downstream (e.g. the MCMC invisible-steps counters compare element counts).
        """
        if out is None or self._visibility_local_ranges is None:
            return

        visibility_local_ranges = self._visibility_local_ranges

        def compact(value: torch.Tensor) -> torch.Tensor:
            views = tuple(value.narrow(0, start, length) for start, length in visibility_local_ranges)
            return views[0] if len(views) == 1 else torch.cat(views, dim=0)

        for name in GaussiansRenderReturn._SCENE_LEVEL_FIELDS:
            value = getattr(out, name)
            if value is not None:
                setattr(out, name, compact(value))

    def _active_camera_tile_size(self, width: int, height: int) -> tuple[int, int] | None:
        """(tile_height, tile_width) the camera active tile mask must be built at so it lands
        on the camera renderer's tile grid for a width x height render, or ``None`` when the
        camera renderer does not support active tiling. Purely a delegation to the renderer's
        polymorphic hook (gsplat resolves its tile size per-resolution under RESOLUTION)."""
        return self._camera_renderer.active_tile_mask_tile_size(width, height)

    def _lidar_overlap_masking_enabled(self) -> bool:
        """Whether the configured strategy extends masking past the camera to the lidar side.

        Only ``VALID_CAMERA_LIDAR_OVERLAP`` derives a lidar mask from the camera mask (and, with
        it, builds the cam->lidar tile mapping and crops the lidar losses);
        ``CAMERA_INVALID_ONLY`` leaves lidar rendering and supervision untouched, which is what
        makes it supervision-neutral. Compared against the enum member, never its string value:
        ``self.config`` is the pydantic model, so ``strategy`` is a genuine
        ``ActiveTileMaskStrategy`` member and a string comparison would silently be False,
        disabling the lidar path everywhere.
        """
        tile_mask_config = self.config.renderer.active_tile_mask
        return (
            tile_mask_config.enabled and tile_mask_config.strategy is ActiveTileMaskStrategy.VALID_CAMERA_LIDAR_OVERLAP
        )

    def active_tile_mapping_prewarm_enabled(self) -> bool:
        """Whether training may eagerly build camera-to-LiDAR mappings."""
        return self._lidar_overlap_masking_enabled()

    def _log_active_tiling_enabled_once(self) -> None:
        """One-shot disclosure of what turning ``active_tile_mask.enabled`` on actually does.

        Every consequence is intended behavior, but none of them is obvious from the config, so
        they are stated once per process rather than left to be discovered from output. The
        lidar half depends on ``active_tile_mask.strategy``, so the strategy in force is named
        and only its own effects are claimed.
        """
        if self._active_tiling_enabled_logged:
            return
        self._active_tiling_enabled_logged = True
        lidar_effect = (
            "(2) lidar supervision is cropped to the lidar tiles that map to active camera tiles -- rays "
            "in unmapped tiles are stamped INVALID and do not contribute to the lidar losses (intended; "
            "only when the pipeline can build the cam->lidar tile mapping -- a separate warning follows "
            "if it cannot)."
            if self._lidar_overlap_masking_enabled()
            else "(2) lidar rendering and supervision are untouched: no cam->lidar tile mapping is built "
            "and no lidar ray is dropped from the losses."
        )
        log.info(
            "[tile_mask] Active tile masking is ENABLED with strategy %s. Effects: (1) camera tiles whose "
            "pixels are all INVALID are skipped by the renderer and their rendered RGB is forced to black "
            "(when the camera renderer exposes a tile grid -- a separate warning follows if it does not); "
            "%s",
            self.config.renderer.active_tile_mask.strategy.value,
            lidar_effect,
        )

    def _log_camera_mask_fraction_once(self, camera_active_tile_mask: torch.Tensor) -> None:
        """One-shot report of how much of the camera tile grid the first mask deactivates."""
        if self._active_tiling_camera_fraction_logged:
            return
        self._active_tiling_camera_fraction_logged = True
        inactive_fraction = 1.0 - float(camera_active_tile_mask.float().mean().item())
        log.info(
            "[tile_mask] First masked camera render: %.1f%% of the %d camera tiles are inactive "
            "(grid %dx%d, rendering skipped and RGB blacked there).",
            100.0 * inactive_fraction,
            camera_active_tile_mask[0].numel(),
            camera_active_tile_mask.shape[1],
            camera_active_tile_mask.shape[2],
        )

    def _log_lidar_mask_fraction_once(self, lidar_active_tile_mask: torch.Tensor) -> None:
        """One-shot report of how much lidar supervision the derived mask crops away."""
        if self._active_tiling_lidar_fraction_logged:
            return
        self._active_tiling_lidar_fraction_logged = True
        inactive_fraction = 1.0 - float(lidar_active_tile_mask.float().mean().item())
        log.info(
            "[tile_mask] First derived lidar mask: %.1f%% of the %d lidar tiles are deactivated "
            "(grid %s); rays there are stamped INVALID and skipped by the lidar losses (intended).",
            100.0 * inactive_fraction,
            lidar_active_tile_mask[0].numel(),
            "x".join(str(d) for d in lidar_active_tile_mask.shape[1:]),
        )

    def _cam_to_lidar_mapping_mismatch(
        self,
        mapping: CameraToLidarTileMapping,
        rendering_data_cam: RenderingData,
        rendering_data_lidar: RenderingData,
    ) -> str | None:
        """Describe why a camera-to-LiDAR mapping does not match the current geometry.

        The cache is keyed by ``unique_sensor_idx`` alone, but the mapping bakes in the camera
        tile grid (frame size / resolved tile size -- gsplat re-resolves its tile size per
        resolution) and the LiDAR element grid. Checking those builder invariants gives eager
        prewarm a coordinated failure and gives lazy cache hits a chance to rebuild.

        Args:
            mapping: Cached or freshly built mapping to validate.
            rendering_data_cam: Camera rendering geometry for the current frame.
            rendering_data_lidar: LiDAR rendering geometry for the current frame.

        Returns:
            A diagnostic describing the mismatch, or ``None`` when the mapping is current.
        """
        # rays: (B, H, W, 6)
        frame_height = rendering_data_cam.rays.shape[1]
        frame_width = rendering_data_cam.rays.shape[2]
        cam_tile_size = self._active_camera_tile_size(frame_width, frame_height)
        if cam_tile_size is None:
            return (
                f"the camera renderer exposes no tile geometry for {frame_width}x{frame_height}, "
                "so the mapping cannot be validated"
            )

        cam_tile_h, cam_tile_w = cam_tile_size
        cam_tiles_y = math.ceil(frame_height / cam_tile_h)
        cam_tiles_x = math.ceil(frame_width / cam_tile_w)

        mismatches: list[str] = []
        if (mapping.cam_tiles_y, mapping.cam_tiles_x) != (cam_tiles_y, cam_tiles_x):
            # Checked before the tile count so a transposed grid (identical numel) is caught.
            mismatches.append(
                f"camera tile grid cached {mapping.cam_tiles_y}x{mapping.cam_tiles_x} "
                f"vs current {cam_tiles_y}x{cam_tiles_x} (tiles_y x tiles_x)"
            )
        if mapping.n_camera_tiles != cam_tiles_y * cam_tiles_x:
            mismatches.append(
                f"camera tile count cached {mapping.n_camera_tiles} vs current {cam_tiles_y * cam_tiles_x}"
            )
        # Lidar identity, as far as it is derivable here without reaching into the renderers:
        # the per-element tile assignment covers the lidar ray grid it was built from, and the
        # tile grid shape must agree with the tile count the mapping matrix was sized for.
        lidar_rows = rendering_data_lidar.rays.shape[1]
        lidar_cols = rendering_data_lidar.rays.shape[2]
        if tuple(mapping.element_to_tile.shape) != (lidar_rows, lidar_cols):
            mismatches.append(
                f"lidar ray grid cached {tuple(mapping.element_to_tile.shape)} vs current {(lidar_rows, lidar_cols)}"
            )
        if mapping.lidar_tile_grid_shape[0] * mapping.lidar_tile_grid_shape[1] != mapping.n_lidar_tiles:
            mismatches.append(
                f"lidar tile grid {mapping.lidar_tile_grid_shape} does not cover {mapping.n_lidar_tiles} lidar tiles"
            )

        if not mismatches:
            return None

        return (
            f"{'; '.join(mismatches)} at camera resolution {frame_width}x{frame_height} "
            f"with tile size {cam_tile_h}x{cam_tile_w} (tile_h x tile_w)"
        )

    def _maybe_init_cam_to_lidar_tile_mapping(
        self,
        cam_sensor_idx: int,
        rendering_data_cam: RenderingData,
        rendering_data_lidar: RenderingData,
    ) -> CameraToLidarTileMapping | None:
        """Lazy one-time precomputation of camera→lidar tile mapping for a specific camera.

        Each camera has different intrinsics and T_cam_lidar, so a separate mapping is built
        and cached per ``cam_sensor_idx``. All backend knowledge lives behind
        ``BaseGaussianRenderer.build_camera_to_lidar_tile_mapping`` (implemented on the lidar
        renderer, which owns the lidar tile grid the mask must land on); renderers that do not
        support active tiling return ``None`` and are cached as ``None`` with a warning.
        Returns the value written to the cache so eager and lazy callers share one builder.
        """
        # Build the mapping at the camera tile size the camera renderer actually uses for this
        # frame (gsplat resolves it per-resolution), matching the camera active-tile-mask grid.
        cam_tile_size = self._active_camera_tile_size(
            rendering_data_cam.rays.shape[2],  # rays: (B, H, W, 6)
            rendering_data_cam.rays.shape[1],
        )
        if cam_tile_size is None:
            log.warning("[tile_mask] Camera→lidar mapping requires a camera renderer with tile geometry; skipping.")
            self._cam_to_lidar_tile_mappings[cam_sensor_idx] = None
            return None

        cam_tile_h, cam_tile_w = cam_tile_size
        mapping = self._lidar_renderer.build_camera_to_lidar_tile_mapping(
            rendering_data_cam,
            rendering_data_lidar,
            cam_tile_height=cam_tile_h,
            cam_tile_width=cam_tile_w,
        )
        if mapping is None:
            log.warning("[tile_mask] Camera→lidar mapping unsupported for this lidar renderer; skipping.")

        self._cam_to_lidar_tile_mappings[cam_sensor_idx] = mapping
        return mapping

    def _ensure_cam_to_lidar_tile_mapping(
        self,
        cam_sensor_idx: int,
        rendering_data_cam: RenderingData,
        rendering_data_lidar: RenderingData,
    ) -> CameraToLidarTileMapping | None:
        """Validate or build one cache entry for both eager and lazy callers.

        This cache intentionally has no internal synchronization. Supported prewarm and
        training calls run on the rank's main thread; interactive-viewer access is serialized
        with training and its camera-only render path does not use this mapping; serving
        serializes each backend or uses a distinct model instance. A future caller that shares
        this cache across host threads, including shallow context shells, must serialize the
        complete validate/evict/build sequence.
        """
        if cam_sensor_idx in self._cam_to_lidar_tile_mappings:
            mapping = self._cam_to_lidar_tile_mappings[cam_sensor_idx]
            if mapping is None:
                return None
            mismatch = self._cam_to_lidar_mapping_mismatch(
                mapping,
                rendering_data_cam,
                rendering_data_lidar,
            )
            if mismatch is None:
                return mapping
            log.warning(
                "[tile_mask] Clearing cached camera-to-LiDAR mapping for sensor %d because it does not match "
                "the current geometry (%s). Repeated clears for this sensor indicate geometry thrashing.",
                cam_sensor_idx,
                mismatch,
            )
            del self._cam_to_lidar_tile_mappings[cam_sensor_idx]

        mapping = self._maybe_init_cam_to_lidar_tile_mapping(
            cam_sensor_idx,
            rendering_data_cam,
            rendering_data_lidar,
        )
        if mapping is None:
            return None
        mismatch = self._cam_to_lidar_mapping_mismatch(
            mapping,
            rendering_data_cam,
            rendering_data_lidar,
        )
        if mismatch is not None:
            log.warning(
                "[tile_mask] Rejecting freshly built camera-to-LiDAR mapping for sensor %d because it does not "
                "match the current geometry (%s).",
                cam_sensor_idx,
                mismatch,
            )
            del self._cam_to_lidar_tile_mappings[cam_sensor_idx]
            raise RuntimeError(
                f"Fresh camera-to-LiDAR tile mapping for sensor {cam_sensor_idx} "
                "does not match the current rendering geometry"
            )
        return mapping

    def prewarm_camera_to_lidar_tile_mapping(
        self,
        cam_sensor_idx: int,
        rendering_data_cam: RenderingData,
        rendering_data_lidar: RenderingData,
    ) -> None:
        """Eagerly populate one mapping through the same builder used by live renders."""
        self._ensure_cam_to_lidar_tile_mapping(
            cam_sensor_idx,
            rendering_data_cam,
            rendering_data_lidar,
        )

    def _maybe_compute_active_tile_masks(
        self,
        rendering_data_cam: RenderingData,
        rendering_data_lidar: Optional[RenderingData],
        cam_sensor_idx: int,
    ) -> None:
        """Populate the camera (and, when a lidar frame is present, the lidar) active tile mask.

        Called from ``forward_gaussians`` before any render starts, and only when
        ``config.renderer.active_tile_mask.enabled`` is on. The camera mask must be computed
        first because lidar renders on a dedicated stream before camera, so the camera mask has
        to exist by the time the lidar mask is derived from it. The lidar half runs only under
        ``ActiveTileMaskStrategy.VALID_CAMERA_LIDAR_OVERLAP`` (see
        ``_lidar_overlap_masking_enabled``); under ``CAMERA_INVALID_ONLY`` this method stops
        after the camera mask and the cam->lidar mapping precompute never runs.

        This is the single owner of the mask: any renderer whose ``active_tile_mask_tile_size``
        returns a tile size participates (GSplatRenderer skips masked tiles at intersection via
        ``gsplat.rasterization(masks=)``), the rest opt out by returning None and the feature is
        inert for them.
        """
        rd_cam = rendering_data_cam

        self._log_active_tiling_enabled_once()

        # Build the mask at the tile size the camera renderer will actually use for this
        # render. gsplat resolves its tile size per-resolution under RESOLUTION, so the
        # mask grid must match that (not a fixed config value) or the renderer rejects it.
        cam_tile_size = self._active_camera_tile_size(rd_cam.w, rd_cam.h)

        if rd_cam.active_tile_mask is None and rd_cam.ray_flags is not None and cam_tile_size is not None:
            cam_tile_h, cam_tile_w = cam_tile_size
            rd_cam.active_tile_mask = compute_active_tile_mask(
                rd_cam.ray_flags,
                tile_width=cam_tile_w,
                tile_height=cam_tile_h,
            )

        if rd_cam.active_tile_mask is not None:
            self._log_camera_mask_fraction_once(rd_cam.active_tile_mask)
        elif cam_tile_size is None:
            # Camera renderer with no tile grid: no mask can be built at any tile size, so the
            # feature is inert. Warn once instead of silently no-opping.
            if not self._active_tiling_no_tile_size_warned:
                self._active_tiling_no_tile_size_warned = True
                log.warning(
                    "[tile_mask] active_tile_mask.enabled is on but the camera renderer exposes no tile "
                    "geometry for %dx%d renders; no active tile mask can be built and the feature has "
                    "no effect.",
                    rd_cam.w,
                    rd_cam.h,
                )
        elif rd_cam.ray_flags is None:
            # ray_flags come from labels.flags: pure inference, novel-view/trajectory rendering
            # and exports have none, so the feature does nothing at all.
            if not self._active_tiling_no_ray_flags_warned:
                self._active_tiling_no_ray_flags_warned = True
                log.warning(
                    "[tile_mask] active_tile_mask.enabled is on but this render carries no camera ray_flags "
                    "(labels.flags); no active tile mask can be built and the feature has no effect. This "
                    "is expected for inference / novel-view / export renders, which have no labels."
                )

        # Everything below is the lidar half of VALID_CAMERA_LIDAR_OVERLAP: deriving the lidar
        # mask, and with it building/validating the cam->lidar tile mapping. CAMERA_INVALID_ONLY
        # stops here, so its promise that "the cam->lidar mapping precompute never runs" holds.
        if not self._lidar_overlap_masking_enabled():
            return

        if rendering_data_lidar is None or rd_cam.active_tile_mask is None:
            return
        rd_lidar = rendering_data_lidar

        # The eager first-step prewarm and this lazy dynamic-geometry fallback share validation
        # and construction.
        mapping = self._ensure_cam_to_lidar_tile_mapping(cam_sensor_idx, rd_cam, rd_lidar)
        if mapping is None:
            return

        lidar_mask_flat = camera_tile_mask_to_lidar_mask(rd_cam.active_tile_mask, mapping)  # (B, n_lidar)
        # Reshape into the consuming renderer's lidar tile grid layout, which is elevation-major
        # (n_bins_elevation, n_bins_azimuth) == gsplat's [tile_h, tile_w].
        rd_lidar.active_tile_mask = lidar_mask_flat.reshape(lidar_mask_flat.shape[0], *mapping.lidar_tile_grid_shape)
        self._log_lidar_mask_fraction_once(rd_lidar.active_tile_mask)

    def _resolve_lidar_stream(self, lidar_stream: Optional[torch.cuda.Stream]) -> torch.cuda.Stream:
        """Pick the CUDA stream used for lidar collect+render.

        Serving threads in a per-request (``Backend``/context-owned) stream so concurrent
        lidar requests do not share a single stream. Training / standalone callers pass
        ``None`` and fall back to a composite-owned stream, created lazily here (kept off
        ``__init__`` so constructing the composite needs no CUDA context and so the stream
        lands at first use) — equivalent for those callers.

        The ``None`` fallback is safe when multiple standalone/training threads first
        use the same composite concurrently. Idempotent: returns the same object on
        repeated calls within one render.
        """
        if lidar_stream is not None:
            return lidar_stream
        if self._lidar_stream is None:
            with self._lidar_stream_init_lock:
                # A sibling may have initialized the stream while this thread waited.
                if self._lidar_stream is None:
                    self._lidar_stream = torch.cuda.Stream()
        return self._lidar_stream

    @ScopedTimer("GaussiansComposite/forward_gaussians")
    def forward_gaussians(
        self,
        rendering_data_cam: Optional[RenderingData] = None,
        rendering_data_lidar: Optional[RenderingData] = None,
        frame_meta_cam: Optional[List[FrameMeta]] = None,  # List is for batch size
        frame_meta_lidar: Optional[List[FrameMeta]] = None,  # List is for batch size
        is_training_batch: bool = False,
        track_transform_ids: Optional[List[str]] = None,
        track_transforms: Optional[torch.Tensor] = None,
        edited_cuboid_tracks: Optional[CuboidTracks] = None,
        gaussian_nodes: Optional[list[str]] = None,
        lidar_stream: Optional[torch.cuda.Stream] = None,
        prepare_selective_adam_visibility: bool = False,
        prefetched_parameter_gather: Optional[_PendingGaussianParameterGather] = None,
    ) -> GaussiansCompositeReturn:
        assert rendering_data_cam is not None or rendering_data_lidar is not None, (
            f"{self.__class__.__name__} Either camera or lidar must be provided"
        )

        # Clear cached ranges so they are recomputed with current gaussian counts.
        # Counts may change between steps due to densification/pruning strategies.
        self._visibility_local_ranges = None

        out_cam: Optional[GaussiansRenderReturn] = None
        out_lidar: Optional[GaussiansRenderReturn] = None
        gaussian_cam_parameters: Optional[dict[str, torch.Tensor]] = None
        gaussian_lidar_parameters: Optional[dict[str, torch.Tensor]] = None
        camera_segment_plan: SegmentPlan | None = None
        lidar_segment_plan: SegmentPlan | None = None
        # Resolved once (below, when has_lidar) and reused across the lidar collect/render/sync blocks so
        # every block runs on the same stream without re-taking the lazy-init lock each time.
        lidar_cuda_stream: Optional[torch.cuda.Stream] = None

        has_cam = rendering_data_cam is not None
        has_lidar = rendering_data_lidar is not None

        if has_cam:
            assert frame_meta_cam is not None and len(frame_meta_cam) == 1, "Only single-frame batch is supported"
        if has_lidar:
            assert frame_meta_lidar is not None and len(frame_meta_lidar) == 1, "Only single-frame batch is supported"

        camera_frustum_cull = None
        if rendering_data_cam is not None:
            camera_frustum_cull = self._build_frustum_cull(rendering_data_cam)

        lidar_frustum_cull = None
        gathered_parameters = None
        # Both sensors share one cached buffer, and each writes its own dynamic-layer
        # windows into it before its own render. That is only safe while the two
        # collects are separated by a render, so a forward carrying both sensors takes
        # the dense path instead. Today only the training batch does that, and training
        # already disables the cache; this keeps the buffer correct if that changes.
        combined_sensor_forward = has_cam and has_lidar
        if prefetched_parameter_gather is not None:
            # The generation route is deliberately independent of local
            # modality selection.  Consume the same full gather on every rank
            # and share it with whichever sensor(s) this batch renders.
            gathered_parameters = self._gather_gaussian_parameters_once(
                list(prefetched_parameter_gather.gaussian_nodes),
                prefetched=prefetched_parameter_gather,
            )
        elif has_cam and has_lidar:
            if gaussian_nodes is not None:
                used_gaussian_nodes = gaussian_nodes
            else:
                used_gaussian_nodes = self.get_gaussians_node_ids()
            gathered_parameters = self._gather_gaussian_parameters_once(
                used_gaussian_nodes,
                prefetched=None,
            )

        # Compute active tile masks before renders start (see
        # _maybe_compute_active_tile_masks for the full contract).
        if self.config.renderer.active_tile_mask.enabled and rendering_data_cam is not None:
            self._maybe_compute_active_tile_masks(
                cast(RenderingData, rendering_data_cam),
                rendering_data_lidar,
                cast(List[FrameMeta], frame_meta_cam)[0].unique_sensor_idx,
            )

        # Flow: collect_lidar (dedicated stream) -> collect_cam (default stream)
        #       -> render_lidar (dedicated stream) -> render_cam (default stream)
        # All operations on main thread, lidar uses dedicated CUDA stream for GPU parallelism.

        if has_lidar:
            # Per-request stream when serving (threaded in from the Backend/context),
            # else the composite-owned fallback. Same object across the lidar blocks.
            lidar_cuda_stream = self._resolve_lidar_stream(lidar_stream)
            lidar_cuda_stream.wait_stream(
                torch.cuda.current_stream()
            )  # lidar stream depends on inputs computed on the default stream
            rd_lidar = cast(RenderingData, rendering_data_lidar)
            # Collect lidar on dedicated stream
            with torch.cuda.stream(lidar_cuda_stream):
                gaussian_lidar_parameters, lidar_segment_plan = self._collect_sensor(
                    rd_lidar,
                    is_training_batch,
                    gaussian_nodes,
                    track_transform_ids,
                    track_transforms,
                    edited_cuboid_tracks,
                    "lidar",
                    combined_sensor_forward=combined_sensor_forward,
                    frustum_cull=lidar_frustum_cull,
                    gathered_parameters=gathered_parameters,
                )
                # record stream so tensors created outside of lidar stream are not deallocated before lidar stream is done
                # skip rd_lidar because it will be recorded in the _render_collected call below.
                if lidar_frustum_cull is not None:
                    lidar_frustum_cull.rays.record_stream(lidar_cuda_stream)
                # gathered_parameters are created outside the lidar stream (on the default stream), so record them here.
                self._record_gathered_parameters_stream(gathered_parameters, lidar_cuda_stream)
                # Resolve to all nodes when None, matching collect_gaussian_parameters behavior.
                used_gaussian_nodes = gaussian_nodes if gaussian_nodes is not None else self.get_gaussians_node_ids()
                for node_id in used_gaussian_nodes:
                    gaussians = self.gaussians_nodes[node_id]
                    gaussians.record_stream(lidar_cuda_stream)
                if track_transforms is not None:
                    track_transforms.record_stream(lidar_cuda_stream)

        if has_cam:
            rd_cam = cast(RenderingData, rendering_data_cam)
            # Collect cam on default stream
            gaussian_cam_parameters, camera_segment_plan = self._collect_sensor(
                rd_cam,
                is_training_batch,
                gaussian_nodes,
                track_transform_ids,
                track_transforms,
                edited_cuboid_tracks,
                "cam",
                combined_sensor_forward=combined_sensor_forward,
                frustum_cull=camera_frustum_cull,
                gathered_parameters=gathered_parameters,
            )

        if has_lidar:
            assert lidar_cuda_stream is not None  # set in the has_lidar collect block above
            rd_lidar = cast(RenderingData, rendering_data_lidar)
            # Render lidar on dedicated stream
            with torch.cuda.stream(lidar_cuda_stream):
                out_lidar = self._render_collected(
                    rd_lidar,
                    cast(List[FrameMeta], frame_meta_lidar),
                    cast(dict[str, torch.Tensor], gaussian_lidar_parameters),
                    self.lidar_extra_ray_signal_infos,
                    self._lidar_renderer,
                    is_training_batch,
                    "lidar",
                    segment_plan=lidar_segment_plan,
                )
                # record stream so tensors created outside of lidar stream are not deallocated before lidar stream is done
                if rd_lidar is not None:
                    rd_lidar.record_stream(lidar_cuda_stream)
                # skip gaussian_lidar_parameters because it is generated within the context manager of the lidar stream. So it will automatically be recorded.
                if frame_meta_lidar is not None:
                    for frame_meta in frame_meta_lidar:
                        frame_meta.record_stream(lidar_cuda_stream)
                # skip self.lidar_extra_ray_signal_infos. Assumption: it doesn't contain any tensors used by any kernel to compute out_lidar.
                # skip self._lidar_renderer. Assumption: it doesn't contain any tensors used by any kernel to compute out_lidar.

        if has_cam:
            rd_cam = cast(RenderingData, rendering_data_cam)
            # Render cam on default stream
            out_cam = self._render_collected(
                rd_cam,
                cast(List[FrameMeta], frame_meta_cam),
                cast(dict[str, torch.Tensor], gaussian_cam_parameters),
                self.camera_extra_ray_signal_infos,
                self._camera_renderer,
                is_training_batch,
                "cam",
                segment_plan=camera_segment_plan,
            )

        # Synchronize the lidar stream with the default stream to ensure lidar results are ready
        # before they are accessed by the caller (e.g., for opacity thresholding in render.py)
        if has_lidar:
            assert lidar_cuda_stream is not None  # set in the has_lidar collect block above
            torch.cuda.current_stream().wait_stream(lidar_cuda_stream)

        selective_adam_visibility = None
        if prepare_selective_adam_visibility:
            # Prepare only after the camera/LiDAR render streams are ordered onto the
            # current stream. The renderer scene fields are still global here; they
            # are compacted for losses and MCMC immediately below without mutation.
            with ScopedTimer("GaussiansComposite/selective_adam_visibility/prepare_global_mask"):
                occlusion_aware = self.config.selective_adam_occlusion_aware
                source_name = "cumulated_weights" if occlusion_aware else "visibility"
                visibility_sources: list[torch.Tensor] = []
                for rendered in (out_cam, out_lidar):
                    if rendered is not None:
                        source = getattr(rendered, source_name)
                        if source is not None:
                            visibility_sources.append(source)

                collected_parameters = (
                    gaussian_cam_parameters if gaussian_cam_parameters is not None else gaussian_lidar_parameters
                )
                assert collected_parameters is not None
                rendered_for_device = out_cam if out_cam is not None else out_lidar
                assert rendered_for_device is not None
                # Renderer scene fields follow the layer-major collected-parameter layout:
                # get_gaussians_node_ids() order, with rank-contiguous blocks inside each
                # layer. local_ranges records this rank's block from every layer in that
                # same order so resolve() reconstructs owner-local layer order.
                selective_adam_visibility = prepare_async_visibility_union(
                    visibility_sources,
                    element_count=collected_parameters["positions"].shape[0],
                    device=rendered_for_device.opacity.device,
                    local_ranges=self._visibility_local_ranges,
                    positive_only=occlusion_aware,
                )

        # Multi-GPU: slice scene_data (visibility, cumulated_weights) from global to local rank size.
        # The renderer operates on all-gathered gaussians, so these tensors have length = total gaussians
        # across all ranks. The losses expect them to match the local (sharded) parameter count.
        if self._visibility_local_ranges is not None:
            self._compact_scene_data_to_local(out_cam)
            self._compact_scene_data_to_local(out_lidar)

        # Combine deformation smoothness from both modalities
        deform_smoothness = deform_smoothness_camera = deform_smoothness_lidar = None
        deform_smoothness_mask = deform_smoothness_mask_camera = deform_smoothness_mask_lidar = None
        if gaussian_cam_parameters is not None:
            deform_smoothness_camera = gaussian_cam_parameters.get("deform_smoothness", None)
            deform_smoothness_mask_camera = gaussian_cam_parameters.get("deform_smoothness_mask", None)
        if gaussian_lidar_parameters is not None:
            deform_smoothness_lidar = gaussian_lidar_parameters.get("deform_smoothness", None)
            deform_smoothness_mask_lidar = gaussian_lidar_parameters.get("deform_smoothness_mask", None)

        if deform_smoothness_camera is not None and deform_smoothness_lidar is not None:
            deform_smoothness = torch.cat([deform_smoothness_camera, deform_smoothness_lidar], dim=0)
        elif deform_smoothness_camera is not None:
            deform_smoothness = deform_smoothness_camera
        elif deform_smoothness_lidar is not None:
            deform_smoothness = deform_smoothness_lidar

        if deform_smoothness_mask_camera is not None and deform_smoothness_mask_lidar is not None:
            deform_smoothness_mask = torch.cat([deform_smoothness_mask_camera, deform_smoothness_mask_lidar], dim=0)
        elif deform_smoothness_mask_camera is not None:
            deform_smoothness_mask = deform_smoothness_mask_camera
        elif deform_smoothness_mask_lidar is not None:
            deform_smoothness_mask = deform_smoothness_mask_lidar

        return GaussiansCompositeReturn(
            rendered_cam=out_cam,
            rendered_lidar=out_lidar,
            deform_smoothness=deform_smoothness,
            deform_smoothness_mask=deform_smoothness_mask,
            selective_adam_visibility=selective_adam_visibility,
        )

    def get_n_active_features(self) -> int:
        """Return the first nodes's number of active features"""
        for gaussians in self.gaussians_nodes.values():
            if hasattr(gaussians, "n_active_features"):
                n_active_features = getattr(gaussians, "n_active_features")
                if isinstance(n_active_features, int):
                    return n_active_features
        return 0

    @property
    def controller_distillation_active(self) -> bool:
        return self._distillation_active

    @ScopedTimer("GaussiansComposite/forward")
    def forward(
        self,
        batch: DataAndRenderingBatch,
        *,
        global_step: int = 0,
        do_render: bool = True,
        enable_calib: bool = True,
        enable_pp: bool = True,
        clamp_output: bool = True,
        is_training_batch: bool = False,
        gaussian_nodes: Optional[list[str]] = None,
        track_transform_ids: Optional[List[str]] = None,
        track_transforms: Optional[torch.Tensor] = None,
        edited_cuboid_tracks: Optional[CuboidTracks] = None,
        render_background: bool = True,
        lidar_stream: Optional[torch.cuda.Stream] = None,
        prepare_selective_adam_visibility: bool = False,
        **extra_rays_data,
    ) -> GaussiansCompositeReturn:
        """Main forward pass logic"""

        prefetched_parameter_gather: Optional[_PendingGaussianParameterGather] = None
        try:
            prefetched_parameter_gather = self._maybe_prefetch_gaussian_parameter_gather(
                is_training_batch=is_training_batch,
                gaussian_nodes=gaussian_nodes,
                track_transform_ids=track_transform_ids,
                track_transforms=track_transforms,
                edited_cuboid_tracks=edited_cuboid_tracks,
            )
            # Evaluate calibration and update batch.rendering (inplace).
            # This makes sure after this call, batch.rendering is not None.
            if batch.rendering is None:
                with ScopedTimer(f"GaussiansComposite/calib/{self.calib.__class__.__name__}"):
                    batch.rendering = self.calib(
                        batch.data, skip_calib=not enable_calib, global_step_for_prober=global_step
                    )

            # Populate ray_flags on camera RenderingData for active tile mask computation.
            # Gated on the feature toggle (both strategies derive the camera mask from these flags)
            # because ray_flags is only ever read by the active-tile-mask path, which is the sole
            # producer of the mask, and validated explicitly because this post-hoc attribute
            # assignment does not re-run RenderingData.__post_init__.
            if (
                self.config.renderer.active_tile_mask.enabled
                and batch.rendering.camera is not None
                and batch.data.camera is not None
                and batch.data.camera.labels.flags is not None
            ):
                camera_label_flags = batch.data.camera.labels.flags
                camera_rays = batch.rendering.camera.rays
                if (
                    camera_label_flags.ndim != 4
                    or camera_label_flags.shape[:3] != camera_rays.shape[:3]
                    or camera_label_flags.shape[3] != 1
                ):
                    raise ValueError(
                        "batch.data.camera.labels.flags must be a 4D tensor (B, height, width, 1) matching "
                        "batch.rendering.camera.rays (B, height, width, 6), but got flags shape "
                        f"{tuple(camera_label_flags.shape)} vs rays shape {tuple(camera_rays.shape)}"
                    )
                batch.rendering.camera.ray_flags = camera_label_flags

            # Gaussians forward
            out = self.forward_gaussians(
                rendering_data_cam=batch.rendering.camera,
                rendering_data_lidar=batch.rendering.lidar,
                frame_meta_cam=batch.data.camera.meta if batch.data.camera is not None else None,
                frame_meta_lidar=batch.data.lidar.meta if batch.data.lidar is not None else None,
                is_training_batch=is_training_batch,
                gaussian_nodes=gaussian_nodes,
                track_transform_ids=track_transform_ids,
                track_transforms=track_transforms,
                edited_cuboid_tracks=edited_cuboid_tracks,
                lidar_stream=lidar_stream,
                prepare_selective_adam_visibility=prepare_selective_adam_visibility,
                prefetched_parameter_gather=prefetched_parameter_gather,
                **extra_rays_data,
            )
        except BaseException as forward_error:
            try:
                self._invalidate_fused_gather_generation_plan()
            except BaseException as invalidation_error:
                forward_error.add_note(
                    f"Failed to invalidate cached Gaussian generation state after the primary error: "
                    f"{invalidation_error!r}"
                )
            if prefetched_parameter_gather is not None:
                try:
                    prefetched_parameter_gather.discard()
                except BaseException as discard_error:
                    forward_error.add_note(
                        f"Failed to drain prefetched Gaussian communication after the primary error: {discard_error!r}"
                    )
            raise
        else:
            if prefetched_parameter_gather is not None:
                # Consumed gathers make this a no-op; unexpected successful
                # routes still drain any launched payload.
                prefetched_parameter_gather.discard()

        # Stamp INVALID on lidar rays in inactive tiles so losses skip them. Lidar-side effect,
        # so it carries the same strategy gate as the mask derivation: under CAMERA_INVALID_ONLY
        # no lidar mask exists and no lidar ray leaves the losses.
        if (
            self._lidar_overlap_masking_enabled()
            and batch.rendering.lidar is not None
            and batch.rendering.lidar.active_tile_mask is not None
            and batch.data.lidar is not None
            and batch.data.lidar.labels.flags is not None
        ):
            cam_sensor_idx = (
                batch.data.camera.meta[0].unique_sensor_idx
                if batch.data.camera is not None and batch.data.camera.meta
                else None
            )
            tile_mapping = (
                self._cam_to_lidar_tile_mappings.get(cam_sensor_idx, None) if cam_sensor_idx is not None else None
            )
            if tile_mapping is not None:
                # Pass the labels object (not the raw flags tensor) so the cached valid-ray
                # counters used by the lidar loss normalization are invalidated with the stamp.
                mark_inactive_lidar_tile_rays_inplace(
                    batch.data.lidar.labels,
                    batch.rendering.lidar.active_tile_mask,
                    tile_mapping,
                )

        # Apply the background if selected
        rendering_camera = unpack_optional(batch.rendering).camera
        if render_background and rendering_camera is not None:
            with ScopedTimer(f"GaussiansComposite/background/{self.background.__class__.__name__}"):
                background_rays_d = rendering_camera.rays[..., 3:]
                if isinstance(self.background, SkyEnvMapBackground):
                    texture_update_mask = None
                    if is_training_batch and self.background.config.filter_texture_updates_by_sky:
                        data_camera = batch.data.camera
                        assert data_camera is not None and data_camera.labels.flags is not None, (
                            "SkyEnvMapBackground training requires camera SKY_SEMANTIC label flags "
                            "when filter_texture_updates_by_sky is enabled"
                        )
                        texture_update_mask = data_camera.labels.get_mask_flags_all(RayFlags.SKY_SEMANTIC).to(
                            device=background_rays_d.device, dtype=torch.bool
                        )
                    self.background(
                        background_rays_d.reshape(-1, 3),
                        out,
                        is_training_batch,
                        texture_update_mask=texture_update_mask,
                    )
                else:
                    self.background(background_rays_d.reshape(-1, 3), out, is_training_batch)

        # Inactive-tile contract: pixels covered by an inactive tile are deliberately forced to
        # BLACK (0.0). When background rendering is on this runs *after* self.background(...), so
        # the blacking intentionally overwrites the composited background there; running it on both
        # paths keeps inactive-tile contents identical with and without background rendering.
        # Only `rgb` is blacked: `opacity`, `distance`, `normal` and `extra_ray_signals` keep
        # whatever the skipped tiles hold -- that asymmetry is intentional, the consumers of
        # inactive pixels (RGB/SSIM losses, metrics) mask them out via labels.flags anyway.
        # GSplatRenderer honors the mask when rendering, via gsplat.rasterization(masks=) tile
        # skipping; renderers that opt out of the active-tiling hooks never receive a mask here.
        if batch.rendering.camera is not None:
            tile_mask = batch.rendering.camera.active_tile_mask
            if tile_mask is not None and out.rendered_cam is not None and out.rendered_cam.rgb is not None:
                h, w = batch.rendering.camera.h, batch.rendering.camera.w
                # Read tile dims from the renderer; deriving from ceil(h / tiles_y) mis-zeros
                # pixels when h % tile_h != 0. Expand at the same tile size the mask was built
                # at (resolved per-render for gsplat) so the pixel mask lines up with the tile
                # grid. A camera renderer without tile geometry cannot have produced this mask,
                # so there is nothing meaningful to expand it against -- warn and skip the
                # blacking rather than abort the run.
                cam_tile_size = self._active_camera_tile_size(w, h)
                if cam_tile_size is None:
                    log.warning(
                        "[tile_mask] An active tile mask is present but the camera renderer exposes no tile "
                        "geometry; skipping inactive-tile blacking."
                    )
                else:
                    tile_h, tile_w = cam_tile_size
                    # (B, tiles_y, tiles_x) -> (B, h, w) -> flat (B * h * w,) to match the flattened rgb.
                    pixel_mask = tile_mask.repeat_interleave(tile_h, dim=1).repeat_interleave(tile_w, dim=2)[:, :h, :w]
                    inactive = ~pixel_mask.reshape(-1)
                    if inactive.numel() != out.rendered_cam.rgb.shape[0]:
                        # Same warn-and-skip stance as the no-tile-geometry branch above, and for
                        # the same reason: a mask that does not cover this render's pixels cannot
                        # be applied, but it is not worth aborting a run over. A bare `assert`
                        # here would both contradict that stance and vanish under `python -O`.
                        log.warning(
                            "[tile_mask] active tile mask covers %d pixels but rgb has %d rows "
                            "(b=%d, h=%d, w=%d); skipping inactive-tile blacking.",
                            inactive.numel(),
                            out.rendered_cam.rgb.shape[0],
                            tile_mask.shape[0],
                            h,
                            w,
                        )
                    else:
                        out.rendered_cam.rgb[inactive] = 0.0

        # The ground fill (applied after post-processing) needs the image as it was before post-processing.
        ground_fill_pre_pp_rgb = (
            out.rendered_cam.rgb.detach().clone()
            if render_background
            and not is_training_batch
            and self._ground_fill_configured()
            and out.rendered_cam is not None
            and out.rendered_cam.rgb is not None
            else None
        )

        # Apply post-processing. We assume single image for post-processing. And only for camera for now.
        if (
            enable_pp
            and len(self.post_processings) > 0
            and batch.rendering.camera is not None
            and batch.data.camera is not None
        ):
            with ScopedTimer("GaussiansComposite/post_processing/subsample"):
                assert batch.rendering.camera.b == 1, "Post-processing only supports batch size 1."

                if is_training_batch:
                    training_frame_idx = batch.data.camera.meta[0].training_frame_idx
                    assert training_frame_idx >= 0, (
                        f"training_frame_idx={training_frame_idx} during training batch "
                        f"(unique_frame_idx={batch.data.camera.meta[0].unique_frame_idx}). "
                        f"This indicates a validation holdout frame leaked into the training sampler."
                    )
                subsample = batch.data.camera.meta[0].subsample
                if subsample is None:
                    subsample = RectSubsampled(
                        original_width=batch.rendering.camera.w,
                        original_height=batch.rendering.camera.h,
                        width=batch.rendering.camera.w,
                        height=batch.rendering.camera.h,
                    )
                pixel_coords = subsample.pixel_coordinates(device=batch.rendering.camera.rays.device)
                resolution = subsample.downsampled_resolution()
            for post_processing in self.post_processings:
                with ScopedTimer(
                    f"GaussiansComposite/post_processing/post_processing/{post_processing.__class__.__name__}"
                ):
                    post_processing(
                        results=out,
                        rays=batch.rendering.camera.rays.reshape(-1, 6),
                        pixel_coords=pixel_coords.reshape(-1, 2),
                        resolution=resolution,
                        training_frame_idx=batch.data.camera.meta[0].training_frame_idx,
                        unique_sensor_idx=batch.data.camera.meta[0].unique_sensor_idx,
                        global_step=global_step,
                    )

        if ground_fill_pre_pp_rgb is not None:
            self._maybe_apply_ground_fill(batch, out, ground_fill_pre_pp_rgb)

        if clamp_output:
            self._maybe_saturate_rgb(out)

        return out

    def _ground_fill_configured(self) -> bool:
        """Whether the sky background has a ground-fill colour set (``SkyMlpBackgroundConfig.ground_fill_*``)."""
        background = self.background
        if not isinstance(background, SkyMlpBackground):
            return False
        cfg = background.config
        return cfg.ground_fill_rgb is not None or bool(cfg.ground_fill_rgb_per_camera)

    def _maybe_apply_ground_fill(
        self, batch: DataAndRenderingBatch, out: GaussiansCompositeReturn, pre_pp_rgb: torch.Tensor
    ) -> None:
        """The sky background's render-time ground fill (``SkyMlpBackgroundConfig.ground_fill_*``).

        Applied to the final image, after post-processing, so post-processing (PPISP's controller predicts each frame's
        exposure and colour from the render) sees the unfilled render, and in final-image colours, the space the
        ground colours are sampled from photos in. In each filled pixel the sky's share, (1 - opacity) * sky colour,
        is swapped for (1 - opacity) * ground colour; the sky's share after post-processing is estimated with that
        pixel's post-processing gain (final / ``pre_pp_rgb``, the image before post-processing). A no-op unless a
        ground colour is set for this camera.
        """
        background = self.background
        if not isinstance(background, SkyMlpBackground):
            return
        cfg = background.config
        rendering_camera = batch.rendering.camera if batch.rendering is not None else None
        rendered = out.rendered_cam
        if rendering_camera is None or rendered is None or rendered.rgb is None or rendered.opacity is None:
            return
        data_camera = batch.data.camera
        meta = data_camera.meta[0] if data_camera is not None and data_camera.meta else None
        camera_name = (
            self._camera_idx_to_name.get(int(meta.unique_sensor_idx), meta.logical_sensor_id) if meta is not None else None
        )
        ground_rgb = (cfg.ground_fill_rgb_per_camera or {}).get(camera_name or "", cfg.ground_fill_rgb)
        if ground_rgb is None:
            return
        rgb = rendered.rgb.reshape(-1, 3)
        see_through = (1.0 - rendered.opacity.reshape(-1, 1).to(rgb.dtype)).clamp(0.0, 1.0)
        if see_through.shape[0] != rgb.shape[0]:
            return
        rays_d = rendering_camera.rays[..., 3:].reshape(-1, 3)
        weight = below_horizon_weight(
            rays_d, cfg.ground_fill_up, cfg.ground_fill_horizon_deg, cfg.ground_fill_blend_deg
        ).to(rgb.dtype)
        if cfg.ground_fill_masked_only:
            # pixels the dataset marks invalid (the ego hood), shrunk so the fill stays inside narrower metric masks
            if data_camera is None or data_camera.labels.flags is None:
                return
            invalid = data_camera.labels.get_mask_flags_any(RayFlags.INVALID)
            if invalid.numel() != rgb.shape[0]:
                return
            invalid = erode_mask(invalid.reshape(rendering_camera.h, rendering_camera.w), cfg.ground_fill_mask_erode_px)
            weight = weight * invalid.reshape(-1).to(device=weight.device, dtype=weight.dtype)
        extra = rendered.extra_ray_signals
        sky = extra.rgb_background if extra is not None else None
        if sky is not None and sky.numel() == rgb.numel() and not background.composite_in_linear_space:
            target = swap_sky_for_ground(
                rgb, pre_pp_rgb.reshape(-1, 3).to(rgb.dtype), sky.reshape(-1, 3).to(rgb.dtype), see_through, ground_rgb
            )
        else:  # no stored sky colour: move the pixel towards the ground colour by its see-through share
            ground = torch.tensor(ground_rgb, dtype=rgb.dtype, device=rgb.device)
            target = torch.lerp(rgb, ground.expand_as(rgb), see_through)
        filled = torch.lerp(rgb, target, weight[:, None])
        rendered.rgb = filled.reshape(rendered.rgb.shape)

    def _maybe_saturate_rgb(self, out: GaussiansCompositeReturn) -> None:
        if out.rendered_cam is not None and out.rendered_cam.rgb is not None and self.saturate_radiance:
            out.rendered_cam.rgb = out.rendered_cam.rgb.clamp(0.0, 1.0)

    @ScopedTimer("GaussiansComposite/update_step_train_batch_start")
    def update_step_train_batch_start(self, epoch: int, global_step: int, system, **kwargs) -> dict[str, torch.Tensor]:
        if (
            self._distillation_start_step >= 0
            and global_step >= self._distillation_start_step
            and not self._distillation_active
        ):
            # ``requires_grad`` is part of the cached gather schema.
            self._invalidate_fused_gather_generation_plan()
            self._distillation_active = True
            self._freeze_scene_for_distillation()
            log.info("PPISP controller distillation started at step %d", global_step)

        additional_parameters: dict[str, torch.Tensor] = {}
        # A frozen component must stay static, but each module's own
        # ``update_step_train_batch_start`` re-asserts ``requires_grad_(True)``
        # on its scheduled params (see ``calib.py`` / ``tracks_calib.py``),
        # which would silently unfreeze them the batch after
        # ``_apply_component_only_freeze`` ran. Skip the step for any
        # WHOLE-frozen component. (Per-track frozen layers are not
        # whole-frozen — their enabled tracks still need the step, so
        # ``_apply_component_only_frozen_tracks`` zeros the frozen tracks'
        # gradients via persistent hooks instead.)
        if not self._distillation_active:
            if self._component_state("calib") != ComponentState.frozen:
                additional_parameters |= update_module_step(self.calib, epoch, global_step, system)
            for layer_id, gaussians in self.gaussians_nodes.items():
                if layer_id in self._component_only_frozen_layers:
                    continue
                additional_parameters |= update_module_step(gaussians, epoch, global_step, system)
            if self._component_state("sky_model") != ComponentState.frozen:
                additional_parameters |= update_module_step(self.background, epoch, global_step, system)
        for post_processing in self.post_processings:
            if self._component_state("post_processing") == ComponentState.frozen:
                continue
            additional_parameters |= update_module_step(post_processing, epoch, global_step, system)

        return additional_parameters

    def _freeze_scene_for_distillation(self) -> None:
        """Freeze all scene modules for PPISP controller distillation.

        Only the PPISP controller network remains trainable.  Everything else
        is frozen to prevent stray gradient accumulation from regularisation
        losses that bypass the rendering pipeline (gaussian scale/density,
        bilateral-grid drift, etc.) and to suppress strategy mutations
        (MCMC perturbation, GSplat densification).
        """
        self.gaussians_strategy.frozen = True
        self.gaussians_nodes.requires_grad_(False)
        self.calib.requires_grad_(False)
        self.background.requires_grad_(False)
        for pp in self.post_processings:
            if isinstance(pp, PPISPPostProcessing):
                pp.freeze_for_controller_distillation()
            else:
                pp.requires_grad_(False)

    @ScopedTimer("GaussiansComposite/update_step_train_batch_end")
    def update_step_train_batch_end(
        self, epoch: int, global_step: int, batch: DataAndRenderingBatch, system, **kwargs
    ) -> None:
        if not self._distillation_active:
            # Retire on every rank before any local empty/no-donor decision or
            # tensor replacement, including independent and cuboid-local pools.
            if self.gaussians_strategy.may_mutate_topology_at(global_step):
                self._invalidate_fused_gather_generation_plan()
            for child in self.get_base_model_children():
                with ScopedTimer(f"{child.__class__.__name__}/update_step_train_batch_end"):
                    child.update_step_train_batch_end(epoch, global_step, batch, system, **kwargs)
            for name, node in self.gaussians_nodes.items():
                with ScopedTimer(f"GaussiansComposite/GaussiansNode/{name}/update_step_train_batch_end"):
                    node.update_step_train_batch_end(epoch, global_step, batch, system, **kwargs)
            with ScopedTimer(f"{self.gaussians_strategy.__class__.__name__}/update_step_train_batch_end"):
                self.gaussians_strategy.update_step_train_batch_end(
                    epoch, global_step, batch, system, self.gaussians_nodes, **kwargs
                )

        # Every Gaussian mutation is complete.  A rare metadata refresh can
        # overlap the scalar logging below and is consumed by the next forward.
        self._refresh_fused_gather_generation_plan_after_mutations(global_step)

        total_gaussians = 0
        for name, node in self.gaussians_nodes.items():
            count = node.get_num_gaussians()
            total_gaussians += count
            system.log(
                f"gaussians/{name}_num_gaussians",
                count * self.trainer_config.world_size,
                prog_bar=False,
            )

        system.log(
            "gaussians/num_gaussians",
            total_gaussians * self.trainer_config.world_size,
            prog_bar=self.config.log_to_prog_bar,
        )

    def get_extra_state(self) -> dict[str, Any] | None:
        """Persist track IDs and initialization-skipped layers for checkpoint restoration."""
        if self.obj_track_ids is None:
            assert self.obj_track_ids is None, "State requires obj_track_ids"
            return None
        state: dict[str, Any] = {
            "obj_track_ids": {key: value.track_ids for key, value in self.obj_track_ids.items() if not value.is_empty()}
        }
        if skipped := getattr(self, "_initialization_skipped_layer_ids", None):
            state["initialization_skipped_layer_ids"] = sorted(skipped)
        return state

    def set_extra_state(self, state: dict[str, Any] | None) -> None:
        if (state is None) or "obj_track_ids" not in state:
            assert self.obj_track_ids is None, "State contains no obj_track_ids which the model requires"
            return

    def configure_optimizers(self, name_prefix: str = "") -> list[OptimizerLRSchedulerConfig]:
        # collect all optimizers of model components in a flat list, if present
        submodule_optims: list[OptimizerLRSchedulerConfig] = []
        for model in (self.calib, self.background, *self.post_processings):
            if model is not None:
                submodule_optims.extend(model.configure_optimizers(name_prefix))

        per_node_optims: list[OptimizerLRSchedulerConfig] = []
        for node_id in self.get_gaussians_node_ids():
            gaussians_node = self.gaussians_nodes[node_id]
            per_node_optims.extend(gaussians_node.configure_optimizers(name_prefix=node_id))

        # Seed each SelectiveAdam's selective_enabled from config so the very first step
        # (before GaussiansSystem._sync_selective_enabled runs) already has the right flag.
        for entry in per_node_optims:
            opt = entry["optimizer"]
            if isinstance(opt, SelectiveAdam):
                group = classify_gaussian_optimizer_group(opt.param_groups)
                selective_enabled = self._initial_selective_enabled(group)
                opt.selective_enabled = selective_enabled
                for param_group in opt.param_groups:
                    param_name = str(param_group.get("name", "?"))
                    log.info(
                        "Optimizer/selective_adam/%s: selective_enabled=%s",
                        param_name,
                        selective_enabled,
                    )

        return submodule_optims + per_node_optims

    def _initial_selective_enabled(self, group: str) -> bool:
        """Return the configured initial ``selective_enabled`` for a given group kind.

        ``group`` is one of ``"general"`` (positions/rotations/features),
        ``"density"``, or ``"scale"`` — the output of
        :func:`classify_gaussian_optimizer_group`.
        """
        if group == "density":
            return self.config.gaussian_density_selective
        if group == "scale":
            return self.config.gaussian_scale_selective
        return self.config.gaussian_selective

    def _get_layers_config(self) -> collect.LayersConfig:
        layers_config = []
        for gaussians in self.gaussians_nodes.values():
            layers_config.append(gaussians.get_layer_config())

        def get_unique_value(getter: Callable[[BaseGaussianModel], int], name: str) -> int:
            values = [getter(gaussians) for gaussians in self.gaussians_nodes.values()]
            if len(set(values)) > 1:
                raise ValueError(f"All values in {name} must be the same, got {values}")
            return values[0]

        extra_signal_dim = get_unique_value(
            lambda gaussians: gaussians.config.particle.extra_signal_dim, "extra_signal_dim"
        )
        camera_extra_signal_dim = get_unique_value(
            lambda gaussians: gaussians.config.particle.camera_extra_signal_dim, "camera_extra_signal_dim"
        )
        # The lidar_extra_signal parameter is stored SH-expanded (lidar_extra_signal_dim * sph_dim),
        # so the collector must collect/validate the full SH coefficient width. The renderer
        # evaluates the SH (ShVector<Dim,Degree,NumCoeffs>), mirroring the radiance/specular path.
        lidar_extra_signal_dim = get_unique_value(
            lambda gaussians: (
                gaussians.config.particle.lidar_extra_signal_dim * gaussians.get_lidar_extra_signal_sph_dim()
            ),
            "lidar_extra_signal_dim",
        )

        albedo_sh_dim = get_unique_value(
            lambda gaussians: cast(SHGaussianModel, gaussians).get_albedo_sh_dim(), "features_albedo"
        )
        specular_sh_dim = get_unique_value(
            lambda gaussians: cast(SHGaussianModel, gaussians).get_specular_sh_dim(), "features_specular"
        )

        return collect.LayersConfig(
            layers=layers_config,
            extra_signal_dim=extra_signal_dim,
            camera_extra_signal_dim=camera_extra_signal_dim,
            lidar_extra_signal_dim=lidar_extra_signal_dim,
            albedo_dim=albedo_sh_dim,
            specular_dim=specular_sh_dim,
            sh_features_dtype=getattr(self.config.renderer, "sh_features_dtype", "fp32"),
        )

    def clear_collect_caches(self) -> None:
        """Drop every answer the collect carries across frames: static layers, segments, fold.

        Each is derived from the checkpoint tensors, so a weight update or scene mutation
        makes it stale. This is also what a freshly cloned context shell needs, which is
        why it is separate from ``_invalidate_collect_caches``: that one additionally
        reaches into the renderers, which a shell shares with the model it was cloned from.

        """
        self._segment_tables = {}
        self._static_collect_cache: Optional[collect.CollectorResult] = None
        # CUDA stream the cached tensors were allocated on, so a later use from a
        # different stream can record itself against them (see _record_cache_stream).
        self._static_collect_cache_stream: Optional[torch.cuda.Stream] = None
        # Per-layer counts at the last collect. Cached-layer sizes must match;
        # dynamic size changes may displace static destinations within the buffer.
        self._static_collect_cache_layout: Optional[tuple[int, ...]] = None
        self._fourier_fold: Optional[dict[str, Any]] = None

    def _invalidate_collect_caches(self) -> None:
        """``clear_collect_caches`` plus the fold references the collect left on the renderers.

        Callers below (train(), load_state_dict, and the scene-editing methods) use this so
        the next eval render re-collects.

        The per-frame band-0 Fourier fold is derived from the cached collect and propagated
        to the renderer(s) as ``_fourier_fold_ref`` from *below* the collect's early-returns.
        A stale fold-ref would therefore survive the invalidation above and make the first
        post-invalidation frame re-mix band-0 with an old theta, so clear it on every
        renderer here too.
        """
        self.clear_collect_caches()
        for renderer in (
            getattr(self, "_camera_renderer", None),
            getattr(self, "_lidar_renderer", None),
        ):
            if renderer is not None:
                renderer._fourier_fold_ref = None

    def _renderer_for_sensor(self, is_camera: bool):
        """The renderer that will consume this sensor's collect."""
        return getattr(self, "_camera_renderer" if is_camera else "_lidar_renderer", None)

    def _sensor_render_honors_fold(self, is_camera: bool) -> bool:
        """Whether this sensor's renderer will honor a band-0 fold this frame.

        The renderer owns the predicate; the collector only supplies the collected
        widths it is about to produce. Those come from LayersConfig rather than from
        the cached result, because the fold decision also runs on the frame that
        populates the cache.
        """
        renderer = self._renderer_for_sensor(is_camera)
        if renderer is None:
            return False
        layers_config = getattr(self, "_layers_config", None)
        if layers_config is None:
            return False
        return renderer.band0_fold_supported(
            is_camera=is_camera,
            features_dim=layers_config.albedo_dim + layers_config.specular_dim,
            extra_signal_dim=layers_config.extra_signal_dim,
            sensor_extra_signal_dim=(
                layers_config.camera_extra_signal_dim if is_camera else layers_config.lidar_extra_signal_dim
            ),
        )

    def _record_cache_stream(self, cache: collect.CollectorResult) -> None:
        """Keep the cached tensors alive across a stream switch.

        The cache outlives the forward that allocated it and both sensors read it, so
        it can be consumed on a stream other than the one it was allocated on (the
        lidar collect runs on ``_lidar_stream``). Execution order is already fenced in
        both directions by forward_gaussians, but the caching allocator also has to be
        told, or it may hand the block to another allocation while the consuming
        stream is still reading it. Reads a thread-local; no device sync.
        """
        allocating = getattr(self, "_static_collect_cache_stream", None)
        if allocating is None:
            return
        current = torch.cuda.current_stream()
        if current == allocating:
            return
        for f in fields(cache):
            tensor = getattr(cache, f.name)
            if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                tensor.record_stream(current)

    def train(self, mode: bool = True):
        # Entering training may update checkpoint tensors; drop the static-layer
        # cache so the next eval render re-collects from fresh weights.
        if mode:
            self._invalidate_collect_caches()
        return super().train(mode)

    def _collect_maybe_static_cached(
        self,
        layers_data,
        layer_indices,
        is_camera: bool = False,
        combined_sensor_forward: bool = False,
        selections: Optional[Sequence[np.ndarray]] = None,
    ) -> tuple[collect.CollectorResult, Optional[SegmentPlan], Optional[list[np.ndarray]]]:
        """Collect with static-layer caching (the cache_static_gaussians path).

        Static layers' collected parameters are pure functions of frozen checkpoint
        tensors: collect them once, then re-collect only the dynamic layers each frame,
        straight into the cached buffers at their own row positions.

        ``selections`` holds this frame's live row ranges per layer, or is ``None`` when
        no sparse plan was built. This is where the collect list and the draw list part
        company, so it is where both are marshalled (see ``upload_segments``): a
        cacheable layer is collected whole on the cold frame, not at all afterwards, and
        drawn through its ranges either way. Returns the collected rows, the draw plan
        (``None`` when every row is live), and the rows this frame wrote per layer
        (``None`` on the dense path, where that is every row).
        """
        layers = layers_data.layers
        layer_sizes = [int(ld.positions.shape[0]) for ld in layers]
        layer_starts = list(accumulate(layer_sizes, initial=0))[:-1]
        total_gaussians = sum(layer_sizes)

        def marshal(
            variant_ids: Collection[int], cold: bool
        ) -> tuple[list[collect.LayerSegments], Optional[SegmentPlan], Optional[list[np.ndarray]]]:
            """This frame's collect list and draw list, from one device copy.

            The third item is the collect list in host rows, for ``post_process``. It is
            ``None`` on the dense path, where every collected row is written every frame
            and the whole layer is the right answer.
            """
            if selections is None:
                # A cacheable layer is written whole on the cold frame; afterwards its
                # rows are already in the buffer, so a warm frame writes none of them.
                return (
                    [
                        collect.whole_layer_segments(layer_sizes[i] if cold or i in variant_ids else 0)
                        for i, ld in enumerate(layers)
                    ],
                    None,
                    None,
                )
            segments, draw_plan = upload_segments(
                layers[0].positions.device,
                selections,
                layer_sizes,
                layer_starts,
                variant_ids,
                cold,
            )
            return segments, draw_plan, collect_ranges(selections, layer_sizes, variant_ids, cold)

        # A prior frame may have installed a fold ref on this sensor's renderer.
        # Clear it up front so any dense/early return below leaves that renderer
        # without a stale fold (a silently-wrong band-0 otherwise); a live fold is
        # re-installed further down only on the cached path that stashes one. Scoped
        # to this sensor's renderer so one sensor's collect never clears the other's
        # fresh fold -- their thetas come from different frame windows.
        sensor_renderer = self._renderer_for_sensor(is_camera)
        if sensor_renderer is not None:
            sensor_renderer._fourier_fold_ref = None
        # A bare composite may carry no renderer config at all; fall through to the
        # dense collect rather than assuming one.
        renderer_cfg = getattr(getattr(self, "config", None), "renderer", None)
        cache_enabled = (
            getattr(renderer_cfg, "cache_static_gaussians", False)
            # not self.training rather than a grad check alone: the controller
            # distillation step runs under no_grad on a combined camera+lidar batch,
            # where the shared buffer would be written by both sensors.
            and not self.training
            and not torch.is_grad_enabled()
            and not combined_sensor_forward
            and layer_indices is None
        )
        if not cache_enabled:
            # Log once per process: a combined forward silently costing the cache is
            # worth knowing about, since today only training builds one.
            if (
                getattr(renderer_cfg, "cache_static_gaussians", False)
                and combined_sensor_forward
                and not getattr(GaussiansComposite, "_logged_combined_forward_cache", False)
            ):
                logging.getLogger(__name__).info(
                    "cache_static_gaussians is on but this forward carries both camera and lidar; "
                    "falling back to the dense collect for both sensors."
                )
                GaussiansComposite._logged_combined_forward_cache = True  # type: ignore[attr-defined]
            return self.collector.collect(layers_data, layer_indices=layer_indices, segments=None), None, None

        def _holistic_fourier_foldable(i, ld) -> bool:
            # Static SH layer with Fourier temporal features under a holistic time
            # embedding: geometry and specular bands are frame-invariant (cacheable),
            # and band 0 is re-mixed per frame inside the render's SH-eval kernel
            # from (fourier_albedo, theta), so the layer needs no per-frame collect.
            if isinstance(ld, (collect.LayerDataRigid, collect.LayerDataDeformable)):
                return False
            features_albedo = getattr(ld, "features_albedo", None)
            if features_albedo is None or features_albedo.dim() != 3:
                return False
            handler = self.collector.layer_handlers[i]
            embed_config = getattr(handler.layer_config, "embed_config", None)
            return isinstance(embed_config, collect.HolisticRemapTimeInputEmbeddingConfig)

        def _time_invariant(ld) -> bool:
            # Rigid/deformable layers carry per-frame poses; SH layers with a
            # 3-dim features_albedo carry Fourier temporal features (mixed with
            # the frame timestamp inside the collector) — both must be
            # re-collected every frame. Only layers whose collected output is a
            # pure function of frozen checkpoint tensors are cacheable.
            if isinstance(ld, (collect.LayerDataRigid, collect.LayerDataDeformable)):
                return False
            features_albedo = getattr(ld, "features_albedo", None)
            if features_albedo is not None and features_albedo.dim() == 3:
                return False
            return True

        # The band-0 fold is an SH-path feature: a holistic-Fourier layer is foldable
        # -- hence cacheable -- when the static cache is on and this sensor's render
        # reaches the fold-capable SH path. Its time-varying band-0 is then recomputed
        # in the render SH kernel from (albedo, theta), leaving the rest of the layer
        # frame-invariant and needing no per-frame collect. The renderer honors the
        # fold in the dense or the segments path.
        fold_ids = [i for i, ld in enumerate(layers) if _holistic_fourier_foldable(i, ld)]
        if len(fold_ids) > 1 or (fold_ids and layers_data.frame_timestamp_us is None):
            # Only a single folded layer is supported, and it needs the frame
            # timestamp; anything else re-collects those layers per frame instead.
            fold_ids = []
        if fold_ids and not self._sensor_render_honors_fold(is_camera):
            # This sensor's render would drop the fold, so folding here would render a
            # stale band-0. Leave the layer in variant_ids for a real dense per-frame
            # collect -- the fallback, not the divergence backstop in renderers.py.
            fold_ids = []
        self._fourier_fold = None
        variant_ids = [i for i, ld in enumerate(layers) if not _time_invariant(ld) and i not in fold_ids]
        if len(variant_ids) == len(layers):
            # nothing cacheable — plain per-frame collect
            segments, draw_plan, written = marshal(variant_ids, cold=True)
            return self.collector.collect(layers_data, layer_indices=None, segments=segments), draw_plan, written

        cache = getattr(self, "_static_collect_cache", None)
        displaced_ids = []
        if cache is not None:
            # Per-frame actor edits (enable_editing_actors) subset the tracks and drop
            # rows, so the dynamic layers resize between frames without passing through
            # any invalidation entry point. Cached layers' contents are frame-invariant,
            # but their offsets can move. A change in their sizes means the
            # scene itself changed and the buffer is stale.
            cached_layout = getattr(self, "_static_collect_cache_layout", None)
            cached_region_intact = (
                cached_layout is not None
                and len(cached_layout) == len(layer_sizes)
                and all(cached_layout[i] == layer_sizes[i] for i in range(len(layer_sizes)) if i not in variant_ids)
            )
            if not cached_region_intact or total_gaussians > cache.positions.shape[0]:
                # The cached region changed shape, or dynamic rows grew past the
                # buffer (only reachable when the cold frame was itself an edited
                # subset). Either way the buffer cannot be reused as-is.
                self._invalidate_collect_caches()
                cache = None
            else:
                assert cached_layout is not None
                cached_starts = list(accumulate(cached_layout, initial=0))[:-1]
                displaced_ids = [
                    i for i in range(len(layers)) if i not in variant_ids and cached_starts[i] != layer_starts[i]
                ]
        if cache is None:
            # A sparse first frame is not a complete static cache: rows outside that
            # frame's draw ranges may become visible later, but a cacheable layer is
            # never collected again on the warm path. ``cold`` makes every cacheable
            # layer whole here, and leaves the variant layers windowed.
            segments, draw_plan, written = marshal(variant_ids, cold=True)
            result = self.collector.collect(layers_data, layer_indices=None, segments=segments)
            self._static_collect_cache = result
            self._static_collect_cache_stream = torch.cuda.current_stream()
            self._static_collect_cache_layout = tuple(layer_sizes)
            return result, draw_plan, written

        # The cache outlives the forward that allocated it and is shared by both
        # sensors, so tell the allocator before reading it from another stream.
        self._record_cache_stream(cache)
        result = cache
        segments, draw_plan, written = marshal(variant_ids, cold=False)
        # A resized preceding dynamic layer moves these static destinations. Refill
        # the whole layer, including rows outside this frame's draw selection.
        for i in displaced_ids:
            segments[i] = collect.whole_layer_segments(layer_sizes[i])
            if written is not None:
                written[i] = whole_layer_ranges(layer_sizes[i])

        if fold_ids:
            i = fold_ids[0]
            ld = layers[i]
            # fold layers are concrete handlers that always define layer_config
            embed = self.collector.layer_handlers[i].layer_config.embed_config  # type: ignore[attr-defined]
            # Stash the raw fold inputs only. The gsplat-specific band-0 angle
            # (theta = 2*pi*remap(time)) is computed in the renderer -- the gsplat
            # boundary -- so this model layer imports no gsplat. frame_timestamp_us is
            # this sensor's own frame window, which is why the fold is installed on
            # this sensor's renderer alone.
            self._fourier_fold = {
                "albedo": ld.features_albedo.float().contiguous()
                if ld.features_albedo.dtype != torch.float32
                else ld.features_albedo.contiguous(),
                "frame_timestamp_us": int(layers_data.frame_timestamp_us),
                "timestamps_us_min": int(embed.timestamps_us_min),
                "timestamps_us_max": int(embed.timestamps_us_max),
                "remap_min": float(embed.remap_min),
                "remap_max": float(embed.remap_max),
                "start": layer_starts[i],
                "size": layer_sizes[i],
            }
        # Propagate the fold to this sensor's renderer only: theta is derived from
        # this sensor's frame window, and the other sensor's renderer holds (or will
        # hold) its own fold for its own window.
        if sensor_renderer is not None:
            sensor_renderer._fourier_fold_ref = self._fourier_fold
        if variant_ids or displaced_ids:
            # Hand the collector every layer and let the segments say what to write:
            # a variant layer through its ranges, a cacheable one nothing at all. Each
            # variant layer lands at its own row position in the cached buffers, so no
            # window slicing, no suffix requirement, and no copy back.
            with ScopedTimer("GaussiansComposite/collect_time_variant"):
                self.collector.collect(layers_data, layer_indices=None, out=result, segments=segments)
        self._static_collect_cache_layout = tuple(layer_sizes)

        if total_gaussians < result.positions.shape[0]:
            # Fewer active actors this frame than the buffer was sized for, so only a
            # prefix was written. Hand back just those rows: the untouched tail still
            # holds a previous frame's gaussians and would otherwise render as ghosts.
            # Slicing dim 0 of a contiguous tensor stays contiguous, so this is a view
            # with no copy and no reallocation.
            result = collect.CollectorResult(
                **{f.name: getattr(result, f.name)[:total_gaussians] for f in fields(result)}
            )

        return result, draw_plan, written

    def _collect_gaussian_parameters_with_segment_plan(
        self,
        rendering_data: RenderingData,
        is_training_batch: bool = False,
        gaussian_nodes: Optional[list[str]] = None,
        track_transform_ids: Optional[List[str]] = None,
        track_transforms: Optional[torch.Tensor] = None,
        edited_cuboid_tracks: Optional[CuboidTracks] = None,
        frustum_cull: Optional[collect.TracksFrustumCullData] = None,
        gathered_parameters: Optional[GatheredGaussianParameters] = None,
        is_camera: bool = False,
        sensor_name: str = "cam",
        combined_sensor_forward: bool = False,
        build_segment_plan: bool = False,
    ) -> tuple[dict[str, torch.Tensor], SegmentPlan | None]:
        """Collects gaussian parameters from all nodes.

        ``is_camera`` selects which sensor's renderer the static-layer cache asks
        about band-0 fold eligibility, and which renderer receives the resulting fold.
        ``sensor_name`` labels the collected-layout record this publishes for the
        matching ``_render_collected`` call (see ``_collected_edit_layouts``).

        The returned dictionary uses Gaussian rows on axis 0. The optional plan selects
        ranges on the same axis.

        ``gaussian_nodes`` lists Gaussian node names. ``track_transform_ids`` lists
        track names in the same order as axis 0 of ``track_transforms``.

        Returns:
            A pair. Item 0 maps each parameter name to its tensor. Item 1 contains
            the segment plan, or ``None`` for a dense render.
        """
        if self.debug_viz:
            visualizer = get_visualdebugger()
            visualizer.set_properties(up="z_up", front="neg_x_front")
        else:
            visualizer = None

        all_node_ids = list(self.gaussians_nodes.keys())
        if gaussian_nodes is None:
            gaussian_nodes = self.get_gaussians_node_ids()
            layer_indices = None
        else:
            existing_node_ids = set(self.get_gaussians_node_ids())
            for gaussian_node in gaussian_nodes:
                assert gaussian_node in existing_node_ids, f"{gaussian_node} not in {existing_node_ids}"
            layer_indices = [all_node_ids.index(node_id) for node_id in gaussian_nodes]

        # _sparse_segment_plan_enabled already raised on every configuration the sparse
        # path cannot serve. A node subset is the one thing it cannot see from there.
        assert not build_segment_plan or layer_indices is None, (
            "sparse_segments cannot plan a subset of the Gaussian nodes"
        )

        # Scene-data slicing is a training/optimizer contract: visibility and
        # cumulated_weights are only read by SelectiveAdam and the
        # visibility_filter losses. No entrypoint reaches this with a tracks
        # edit and world_size > 1 today, since edited_cuboid_tracks comes only
        # from nurec/render/render.py, which never initialises a process group.
        # A filtering get_layer_data would still drop rigid rows after these
        # counts were captured, so gate on training to keep the precondition
        # structural rather than assumed.
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        world_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        should_slice_scene_data = is_training_batch and world_size > 1
        visibility_local_ranges: list[tuple[int, int]] = []

        frame_start_us, frame_end_us = BaseGaussianModel.get_frame_window(rendering_data)
        frame_exposure_us = frame_end_us - frame_start_us
        layers_data = collect.LayersData(
            layers=[],
            frame_timestamp_us=(frame_start_us + frame_end_us) // 2,
        )
        offsets = []
        offset = 0
        additional_gaussian_parameters = defaultdict(list)
        # Sparse segments: which rows of each layer this frame collects and draws,
        # decided on the host from the tracks' pose time ranges before any collect
        # kernel is queued (see nurec.models.gaussians.segment_plan).
        segment_selections: list[np.ndarray] = []
        # Where each layer actually landed in THIS sensor's collected tensors, for a
        # render-time override hook whose row indices address the unfiltered layer (see
        # nurec.models.gaussians.edit_hook.CollectedLayerLayout for the index-space
        # contract). Without it, edited-actors requests misindex.
        #
        # Only recorded when a hook is installed: the per-row keep mask below gathers over
        # a per-gaussian tensor, which is not worth paying for on every training step.
        record_edit_layouts = self._edit_overrides is not None
        collected_edit_layouts: dict[str, CollectedLayerLayout] = {}
        collection_contexts: dict[str, BaseGaussianModel.CollectionContext] = {}
        # One collection receives one sensor view and one common actor-edit request.
        # Body and lights cannot request different edits of their shared owner here,
        # so the owner alone identifies both the final edit and the prepared motion.
        # Build each edit once, including transforms; layers still filter their own
        # Gaussian rows. These maps are local so the next view/edit prepares fresh motion.
        prepared_motions: dict[RigidGaussianModel, PreparedRigidMotion] = {}
        owner_edits: dict[RigidGaussianModel, LayerTrackIds.Edit | None] = {}
        # Prepare each selected layer's owner once for this sensor view, before
        # collecting any Gaussian rows. An owner need not itself be selected.
        for node_id in gaussian_nodes:
            gaussians = self.gaussians_nodes[node_id]

            tracks_edit = None
            prepared_motion = None
            if isinstance(gaussians, RigidGaussianModel):
                source = gaussians.get_motion_source()
                if source not in prepared_motions:
                    if edited_cuboid_tracks is not None:
                        tracks_edit = source.create_tracks_edit_from_cuboid_tracks(edited_cuboid_tracks)
                    if track_transform_ids is not None:
                        # Supplied edited trajectories are authoritative. Only a
                        # transform-only request starts from calibrated owner motion.
                        cuboid_tracks = (
                            tracks_edit.cuboid_tracks
                            if tracks_edit is not None
                            else source.get_calibrated_cuboid_tracks()
                        )
                        mapping = (
                            tracks_edit.mapping
                            if tracks_edit is not None
                            else torch.arange(cuboid_tracks.n_tracks, device=cuboid_tracks.device)
                        )
                        pose_deltas = se3_matrix_to_tquat(
                            torch.eye(4, device=cuboid_tracks.device)
                            .unsqueeze(0)
                            .expand(cuboid_tracks.tracks_data.tracks_poses.shape[0], -1, -1)
                        )
                        track_indices: dict[str, int] = {}
                        for index, track_id in enumerate(cuboid_tracks.tracks_id):
                            track_indices.setdefault(track_id, index)
                        for track_transform_id, track_transform in zip(
                            track_transform_ids, unpack_optional(track_transforms)
                        ):
                            track_index = track_indices.get(track_transform_id)
                            if track_index is not None:
                                pack_info = cuboid_tracks.tracks_data.tracks_packinfo[track_index]
                                pose_deltas[int(pack_info[0]) : int(pack_info[0] + pack_info[1])] = track_transform

                        adjusted_tracks = CuboidTracks.Ops.transform_with_delta_poses(
                            cuboid_tracks, lt.SE3.InitFromVec(pose_deltas), left_multiply=False
                        )
                        tracks_edit = LayerTrackIds.Edit(cuboid_tracks=adjusted_tracks, mapping=mapping)

                    # Attribute preparation to the owner even when only its lights are selected.
                    source_layer_id = node_id if source is gaussians else gaussians.config.motion_source_layer
                    with ScopedTimer(f"GaussiansComposite/prepare_motion[{source_layer_id}]"):
                        prepared_motions[source] = source.prepare_motion(
                            rendering_data, frustum_cull, is_training_batch, tracks_edit=tracks_edit
                        )
                    owner_edits[source] = tracks_edit
                tracks_edit = owner_edits[source]
                prepared_motion = prepared_motions[source]

            context = BaseGaussianModel.CollectionContext(
                rendering_data=rendering_data,
                is_training_batch=is_training_batch,
                tracks_edit=tracks_edit,
                frustum_cull=frustum_cull,
                build_segment_plan=build_segment_plan,
                prepared_motion=prepared_motion,
            )
            collection_contexts[node_id] = context

        for node_id in gaussian_nodes:
            gaussians = self.gaussians_nodes[node_id]
            context = collection_contexts[node_id]
            with ScopedTimer(f"GaussiansComposite/get_layer_data[{node_id}]"):
                gathered_counts: Optional[tuple[int, ...]] = None
                if gathered_parameters is None:
                    params = gaussians.get_parameters()
                    local_count = next(iter(params.values())).shape[0]
                    params = distributed_all_gather_gaussian_parameters(
                        params,
                        group=self._get_gaussian_process_group(),
                    )
                else:
                    gathered = gathered_parameters[node_id]
                    # Sensor-specific layer data may rebind dict entries; shared gathered tensors remain read-only.
                    params = dict(gathered.parameters)
                    local_count = gathered.local_count
                    gathered_counts = gathered.counts_by_rank
                layer_data = gaussians.get_layer_data(context, params)
                layers_data.layers.append(layer_data)

                offsets.append(offset)
                layer_total = layer_data.positions.shape[0]
                if build_segment_plan:
                    if isinstance(layer_data, collect.LayerDataRigid):
                        table = LayerSegmentTable.for_layer(
                            self._segment_tables.get(node_id),
                            layer_data.tracks_ids,
                            context.track_time_data,
                            layer_name=node_id,
                        )
                        self._segment_tables[node_id] = table
                        selection = table.select(frame_start_us, frame_end_us, frame_exposure_us)
                    else:
                        selection = whole_layer_ranges(int(layer_total))
                    segment_selections.append(selection)
                # The collector's visibility mask is PER TRACK ([num_tracks] bool, from
                # pose interpolation / track lifetime); expand it to per collected ROW
                # via each gaussian's track id so the aspect runtime can gather by row
                # and never re-show rows the collect masked out.
                if record_edit_layouts:
                    track_keep = getattr(layer_data, "keep_mask", None)
                    row_ids = getattr(layer_data, "tracks_ids", None)
                    row_keep = (
                        track_keep.to(torch.bool)[row_ids.long()]
                        if track_keep is not None and row_ids is not None
                        else None
                    )
                    collected_edit_layouts[node_id] = CollectedLayerLayout(
                        layer_offset=offset,
                        layer_size=int(layer_total),
                        layer_id=node_id,
                        kept_rows=context.kept_rows,
                        collect_keep=row_keep,
                    )
                # Multi-GPU: compute this rank's contiguous range within the all-gathered layer.
                # evenly_divisible_all_gather produces contiguous blocks [rank0 | rank1 | ...],
                # so rank R's portion starts at sum(counts[:R]) within the layer.
                # Skip if already computed (ranges are identical across camera/lidar collections).
                if should_slice_scene_data and self._visibility_local_ranges is None:
                    device = layer_data.positions.device
                    all_counts = self._resolve_visibility_counts(
                        node_id=node_id,
                        world_size=world_size,
                        world_rank=world_rank,
                        local_count=local_count,
                        device=device,
                        gathered_counts=gathered_counts,
                    )
                    gathered_total = sum(all_counts)
                    if gathered_total != layer_total:
                        raise RuntimeError(
                            f"Gathered count mismatch for {node_id}: total count "
                            f"{gathered_total} != layer total {layer_total}"
                        )
                    rank_start = sum(all_counts[:world_rank])
                    visibility_local_ranges.append((offset + rank_start, local_count))

                offset += layer_total

                for name, value in context.additional_data.items():
                    if value[1]:
                        additional_gaussian_parameters[name].append(value[0])
            offsets.append(offset)

        # Store local ranges for multi-GPU visibility/cumulated_weights compaction.
        # This is used by forward_gaussians to compact renderer scene_data back to local rank size.
        # Only set on the first call; subsequent calls (e.g., lidar after camera) reuse the existing mapping.
        if self._visibility_local_ranges is None and visibility_local_ranges:
            self._visibility_local_ranges = tuple(visibility_local_ranges)

        # Handed to the override hook by _render_collected. Keyed BY SENSOR and consumed
        # (popped) there: a combined camera+lidar forward runs both collects before either
        # render, and the two collects allocate their kept-row tensors on different CUDA
        # streams, so one shared slot would hand the lidar render the camera's record.
        # Popping also means a render can never silently reuse a previous frame's layout.
        if record_edit_layouts:
            self._collected_edit_layouts = {**self._collected_edit_layouts, sensor_name: collected_edit_layouts}

        # Sparse segments: the collect marshals this frame's ranges, because only it
        # knows which layers are cacheable and whether this is a cold frame. One host
        # decision and one device copy, before any collect kernel is queued. PyTorch's
        # pinned caching allocator owns staging-buffer reuse and transfer fencing.
        if build_segment_plan:
            assert layers_data.layers, "sparse_segments needs at least one Gaussian layer"

        with ScopedTimer("GaussiansComposite/collect"):
            collector_result, segment_plan, written_rows = self._collect_maybe_static_cached(
                layers_data,
                layer_indices,
                is_camera=is_camera,
                combined_sensor_forward=combined_sensor_forward,
                selections=segment_selections if build_segment_plan else None,
            )

        for i, node_id in enumerate(gaussian_nodes):
            gaussians = self.gaussians_nodes[node_id]

            # post_process reads back what the collect wrote, so it gets the same row
            # list. Handing it the whole layer on a windowed frame would re-apply the
            # transform to rows still holding a previous frame's transformed values.
            with ScopedTimer(f"GaussiansComposite/post_process[{node_id}]"):
                gaussians.post_process(
                    offsets[i],
                    collector_result,
                    layers_data.layers[i],
                    rows=None if written_rows is None else written_rows[i],
                )

        results = {field.name: getattr(collector_result, field.name) for field in fields(collector_result)}
        self._apply_lidar_layer_visibility(results, gaussian_nodes, offsets, sensor_name)
        additional_results = {k: torch.cat(v, 0) for k, v in additional_gaussian_parameters.items()}

        if visualizer is not None:
            for i, node_id in enumerate(gaussian_nodes):
                positions = results["positions"][offsets[i] : offsets[i + 1]]
                features = results["features"][offsets[i] : offsets[i + 1]]
                visualizer.add_point_cloud(
                    f"{node_id}_positions",
                    points=positions.detach().cpu().numpy(),
                    radius=0.0005,
                    colors_quantities={
                        f"{node_id}_color": SH2RGB(features[:, :3].detach().cpu().numpy()),
                    },
                )

            visualizer.show()

        return results | additional_results, segment_plan

    def collect_gaussian_parameters(
        self,
        rendering_data: RenderingData,
        is_training_batch: bool = False,
        gaussian_nodes: Optional[list[str]] = None,
        track_transform_ids: Optional[List[str]] = None,
        track_transforms: Optional[torch.Tensor] = None,
        edited_cuboid_tracks: Optional[CuboidTracks] = None,
        frustum_cull: Optional[collect.TracksFrustumCullData] = None,
        gathered_parameters: Optional[GatheredGaussianParameters] = None,
        is_camera: bool = False,
        sensor_name: str = "cam",
    ) -> dict[str, torch.Tensor]:
        """Dict-only view of ``_collect_gaussian_parameters_with_segment_plan``.

        Callers that only want the collected tensors keep this return type; the sparse
        path uses the private helper because it also carries the frame's row plan.
        """
        parameters, _ = self._collect_gaussian_parameters_with_segment_plan(
            rendering_data=rendering_data,
            is_training_batch=is_training_batch,
            gaussian_nodes=gaussian_nodes,
            track_transform_ids=track_transform_ids,
            track_transforms=track_transforms,
            edited_cuboid_tracks=edited_cuboid_tracks,
            frustum_cull=frustum_cull,
            gathered_parameters=gathered_parameters,
            is_camera=is_camera,
            sensor_name=sensor_name,
        )
        return parameters

    def _apply_lidar_layer_visibility(
        self,
        gaussian_parameters: dict[str, torch.Tensor],
        gaussian_nodes: list[str],
        offsets: list[int],
        sensor_name: str,
    ) -> None:
        """Zero collected opacity for layers configured as camera-only.

        Keep every row in the lidar collection so camera and lidar scene-data
        tensors retain the same layer-major layout during joint distributed
        training. Replacing the collected density tensor (rather than mutating
        the model parameter) also makes lidar gradients exactly zero for the
        hidden layer while leaving camera rendering and optimization intact.

        ``gaussian_nodes`` gives the collected layer order. ``offsets`` holds
        one half-open pair per layer: ``[start0, end0, start1, end1, ...]``.
        ``gaussian_parameters["densities"]`` is floating ``[N, 1]`` on the
        collector device. For lidar with hidden layers, replace that dictionary
        entry in place with a clone of the same shape, dtype, and device; all
        other entries and the original tensor remain unchanged. Returns None.
        Raises RuntimeError if the lidar offset list does not have two entries
        per layer. Camera and all-visible lidar collections allocate no clone.
        """
        if sensor_name != "lidar":
            return

        if len(offsets) != 2 * len(gaussian_nodes):
            raise RuntimeError(
                "Expected one collected (start, end) offset pair per Gaussian layer, "
                f"got {len(offsets)} offsets for {len(gaussian_nodes)} layers"
            )

        # Some lightweight consumers construct the composite without the full
        # typed model config.  Preserve the schema default (visible) when the
        # config or the new field is absent; only an explicit false value hides
        # a layer from lidar.
        layer_configs = getattr(getattr(self, "config", None), "layers", {})
        hidden_ranges = [
            (offsets[2 * i], offsets[2 * i + 1])
            for i, node_id in enumerate(gaussian_nodes)
            if not getattr(layer_configs.get(node_id), "render_in_lidar", True)
        ]
        if not hidden_ranges:
            return

        densities = gaussian_parameters["densities"]
        visible_densities = densities.clone()
        for start, end in hidden_ranges:
            visible_densities[start:end] = 0.0
        gaussian_parameters["densities"] = visible_densities

    def get_all_gaussian_positions(self) -> torch.Tensor | None:
        """
        Collect all Gaussian center positions from all Gaussian nodes if possible.

        Returns:
            torch.Tensor | None: Concatenated positions of all Gaussians from all nodes, shape (N, 3), or None if not available
        """
        all_positions: list[torch.Tensor] = []
        for gaussian_node in self.gaussians_nodes.values():
            positions = gaussian_node.get_positions()
            all_positions.append(positions.data)

        if not all_positions:
            return None

        return torch.cat(all_positions, dim=0)

    def track_ply_override(
        self,
        track_id: str,
        asset: Asset,
        transform: torch.Tensor,
        dims_offset: torch.Tensor,
    ) -> bool:
        """
        Overrides the gaussians for a track id with gaussians from an asset.
        Note that the override is only for validation and should be reverted by calling
        revert_track_ply_override before resuming training.
        """
        target_nodes = [
            gaussian_node
            for gaussian_node in self.gaussians_nodes.values()
            if isinstance(gaussian_node, RigidGaussianModel)
            and gaussian_node.owns_tracks
            and track_id in gaussian_node.cuboid_tracks.tracks_id
        ]
        # Validate every owner before replacing rows in any owner. This keeps a
        # rejection by a row-aligned model from leaving a partly edited scene.
        for gaussian_node in target_nodes:
            gaussian_node.validate_external_gaussian_topology_edit("track PLY replacement")

        if target_nodes:
            self._invalidate_fused_gather_generation_plan()
            self._invalidate_collect_caches()  # a scene edit invalidates every cached collect
        replaced = False
        for gaussian_node in target_nodes:
            replaced = (
                gaussian_node.track_ply_override(track_id, asset=asset, transform=transform, dims_offset=dims_offset)
                or replaced
            )

        return replaced

    def _get_collector_time_embedding_config(
        self, layer_id: str
    ) -> collect.IndividualRemapTimeInputEmbeddingConfig | collect.IndividualStepTimeInputEmbeddingConfig | None:
        """Return the live per-instance time-table config captured by the collector, if any."""
        layer_index = list(self.gaussians_nodes).index(layer_id)
        embed_config = getattr(self.collector.layers_config.layers[layer_index], "embed_config", None)
        if isinstance(
            embed_config,
            (collect.IndividualRemapTimeInputEmbeddingConfig, collect.IndividualStepTimeInputEmbeddingConfig),
        ):
            return embed_config
        return None

    def _prepare_collector_time_embedding(
        self,
        layer_id: str,
        insertion: PreparedRigidAssetInsertion,
    ) -> torch.Tensor | None:
        """Validate and stage a collector time table without changing the live config."""
        embed_config = self._get_collector_time_embedding_config(layer_id)
        if embed_config is None:
            return None
        time_embed = insertion.time_embed
        assert time_embed is not None  # a table-carrying embed_config implies a per-instance time_embed
        staged_config = copy.copy(embed_config)
        collect.swap_time_embedding_table(
            staged_config,
            cast(torch.Tensor, time_embed.timestamps_us_ranges),
            min_rows=len(insertion.cuboid_tracks.tracks_id),
        )
        return staged_config.timestamps_us_ranges

    def insert_asset(
        self,
        asset: Asset,
        inserted_cuboid_tracks: CuboidTracks,
        dims_offset: torch.Tensor,
        rgb_post_processing: RGBPostProcessing | None = None,
    ) -> bool:
        """
        Insert gaussians associated with a new cuboid track into a RigidGaussianModel.

        Append owner tracks without changing existing indices: dependent layers
        retain their original track IDs and use those indices in the owner's poses.

        Args:
            asset: Asset containing gaussian parameters
            inserted_cuboid_tracks: CuboidTracks data for inserted track, derived from sensorsim proto's EditAssetsRequest.insert
            dims_offset: Offset to apply to cuboid dimensions
        """
        (track_id,) = inserted_cuboid_tracks.tracks_data.tracks_id
        concatenated_cuboid_tracks = CuboidTracks.Ops.concatenate([self.cuboid_tracks, inserted_cuboid_tracks])
        inserted_label_class = inserted_cuboid_tracks.tracks_data.tracks_label_class[0]
        fallback_layer_id = "dynamic_rigids"
        insertion_plans: list[_AssetInsertionPlan] = []

        for layer_id, gaussian_node in self.gaussians_nodes.items():
            layer_config = self.obj_track_ids[layer_id].config
            if layer_config.get("is_dynamic", None) is not True:
                continue
            if not isinstance(gaussian_node, RigidGaussianModel) or not gaussian_node.owns_tracks:
                continue

            # Accept insert if the layer's config allows this label_class (e.g. "vehicle"), or if
            # the scene already has a track with this label (e.g. "automobile" from dataset).
            label_allowed = False
            if "label_classes" in layer_config and isinstance(layer_config.label_classes, ListConfig):
                label_allowed = inserted_label_class in list(layer_config.label_classes)
            if not label_allowed:
                label_allowed = inserted_label_class in gaussian_node.cuboid_tracks.tracks_data.tracks_label_class
            if not label_allowed:
                if layer_id != fallback_layer_id:
                    log.warning(
                        f"Insertion of {inserted_label_class} for {layer_id} is not allowed. "
                        f"Allowed label classes: {layer_config.get('label_classes', 'N/A')}"
                    )
                continue

            new_obj_track_ids = LayerTrackIds(config=self.obj_track_ids[layer_id].config)
            new_obj_track_ids.initialize_from_tracks(concatenated_cuboid_tracks)

            candidate_layer_cuboid_tracks = new_obj_track_ids.get_layer_tracks(concatenated_cuboid_tracks)

            if track_id not in candidate_layer_cuboid_tracks.tracks_id:
                log.info(f"insert_asset: skipping layer={layer_id} - track {track_id} excluded by layer filter")
                continue
            # Preserve the node-local tracks, which may have been coalesced for calibration.
            # Rebuilding them from the composite's global tracks would restore the original pose
            # count and invalidate the existing calibrator's packed pose indices.
            layer_cuboid_tracks = CuboidTracks.Ops.concatenate([gaussian_node.cuboid_tracks, inserted_cuboid_tracks])
            insertion_plans.append(
                _AssetInsertionPlan(
                    layer_id=layer_id,
                    gaussian_node=gaussian_node,
                    layer_cuboid_tracks=layer_cuboid_tracks,
                    obj_track_ids=new_obj_track_ids,
                )
            )

        if not insertion_plans and fallback_layer_id in self.gaussians_nodes:
            gaussian_node = self.gaussians_nodes[fallback_layer_id]
            if isinstance(gaussian_node, RigidGaussianModel) and gaussian_node.owns_tracks:
                # Add fallback layer for arbitrary label classes from downstream clients (NDAS-SIM).
                log.warning(
                    f"Insertion of unrecognized label class {inserted_label_class} "
                    f"will use fallback layer {fallback_layer_id}"
                )

                layer_cuboid_tracks = CuboidTracks.Ops.concatenate(
                    [gaussian_node.cuboid_tracks, inserted_cuboid_tracks]
                )
                fallback_obj_track_ids = LayerTrackIds(config=self.obj_track_ids[fallback_layer_id].config)
                fallback_obj_track_ids.initialize_from_tracks(concatenated_cuboid_tracks)
                fallback_track_ids = unpack_optional(fallback_obj_track_ids.track_ids)
                if track_id not in fallback_track_ids:
                    fallback_track_ids.append(track_id)
                    fallback_obj_track_ids.global_unique_track_idx = torch.cat(
                        [
                            unpack_optional(fallback_obj_track_ids.global_unique_track_idx),
                            torch.tensor(
                                [concatenated_cuboid_tracks.tracks_id.index(track_id)],
                                device=concatenated_cuboid_tracks.device,
                                dtype=torch.long,
                            ),
                        ]
                    )
                insertion_plans.append(
                    _AssetInsertionPlan(
                        layer_id=fallback_layer_id,
                        gaussian_node=gaussian_node,
                        layer_cuboid_tracks=layer_cuboid_tracks,
                        obj_track_ids=fallback_obj_track_ids,
                    )
                )

        # All candidate layers must accept the topology change before any
        # calibration, track table, Gaussian parameter, or cache is mutated.
        for plan in insertion_plans:
            plan.gaussian_node.validate_external_gaussian_topology_edit("asset insertion")

        # Materialize every layer and validate every collector table before publishing
        # any part of the scene edit. A construction failure in a later target layer
        # therefore leaves earlier layers and global track metadata untouched.
        prepared_plans: list[_PreparedAssetInsertionPlan] = []
        for plan in insertion_plans:
            insertion = plan.gaussian_node.prepare_asset_insertion(
                track_id,
                asset,
                inserted_cuboid_tracks,
                dims_offset,
                rgb_post_processing=rgb_post_processing,
                target_cuboid_tracks=plan.layer_cuboid_tracks,
                inserted_track_ids={track_id},
            )
            prepared_plans.append(
                _PreparedAssetInsertionPlan(
                    plan=plan,
                    insertion=insertion,
                    collector_time_table=self._prepare_collector_time_embedding(plan.layer_id, insertion),
                )
            )

        if not prepared_plans:
            return False

        self._invalidate_fused_gather_generation_plan()
        self._invalidate_collect_caches()  # a scene edit invalidates every cached collect

        # Preserve training state for every target before publishing the first one.
        # Preparation above contains all tensor/factory work; publication below is
        # assignment-only and cannot expose one layer while another is still building.
        for prepared_plan in prepared_plans:
            prepared_plan.plan.gaussian_node.save_training_parameters()

        for prepared_plan in prepared_plans:
            plan = prepared_plan.plan
            plan.gaussian_node.apply_prepared_asset_insertion(
                prepared_plan.insertion,
                save_training_parameters=False,
            )
            collector_embed_config = self._get_collector_time_embedding_config(plan.layer_id)
            if collector_embed_config is not None:
                assert prepared_plan.collector_time_table is not None
                collector_embed_config.timestamps_us_ranges = prepared_plan.collector_time_table
            self.obj_track_ids[plan.layer_id] = plan.obj_track_ids

        self.cuboid_tracks = concatenated_cuboid_tracks
        return True

    @ScopedTimer("GaussiansComposite/remove_background_gaussians")
    def remove_background_gaussians(
        self, track_ids: List[str], track_shift_m: torch.Tensor, track_padding_m: torch.Tensor
    ) -> None:
        """
        Removes background gaussians that fall inside the specified cuboid tracks. This ensure that
        objects do not overlap with the background when modifying the trajectories of these objects.
        """
        self._invalidate_fused_gather_generation_plan()
        self._invalidate_collect_caches()  # a scene edit invalidates every cached collect

        trajectories_to_remove: List[torch.Tensor] = []
        dims_to_remove: List[torch.Tensor] = []
        for track_id in track_ids:
            for gaussian_node in self.gaussians_nodes.values():
                if isinstance(gaussian_node, RigidGaussianModel) and gaussian_node.owns_tracks:
                    track_poses_and_dims = gaussian_node.get_track_poses_and_dims(track_id)
                    if track_poses_and_dims is not None:
                        trajectory = track_poses_and_dims[0].matrix()
                        trajectory[:, :3, 3] += track_shift_m

                        cuboid_dims = track_poses_and_dims[1]
                        cuboid_dims_padded = cuboid_dims + track_padding_m

                        trajectories_to_remove.append(trajectory)
                        dims_to_remove.append(cuboid_dims_padded)

        for gaussian_node in self.gaussians_nodes.values():
            if not isinstance(gaussian_node, RigidGaussianModel):
                gaussian_node.remove_gaussians_in_trajectories(trajectories_to_remove, dims_to_remove)

    @ScopedTimer("GaussiansComposite/morton_sort_gaussians")
    def morton_sort_gaussians(
        self,
        sort_track_layers: bool = False,
        voxel_size: float = VOXEL_SIZE,
        only_layers: Optional[Collection[str]] = None,
    ) -> None:
        """Reorder every layer's Gaussians into Morton order (``only_layers`` restricts which).

        Row order is otherwise whatever densification and relocation left behind. Sorting it puts
        neighbouring rows close in space.

        voxel_size is the Morton key's voxel edge in meters.

        RigidGaussianModel layers are skipped unless sort_track_layers is set. Their positions are in
        each cuboid's local frame, so the rigid override gives them a track-partitioned order rather
        than a plain Morton one.

        Every layer is validated before any row moves, so one that cannot be sorted is skipped and the rest
        still sort.
        """
        self._invalidate_fused_gather_generation_plan()

        orders: dict[str, torch.Tensor] = {}
        for layer_id, gaussian_node in self.gaussians_nodes.items():
            if only_layers is not None and layer_id not in only_layers:
                continue
            if isinstance(gaussian_node, RigidGaussianModel) and not sort_track_layers:
                continue
            try:
                order = gaussian_node.compute_morton_order(voxel_size=voxel_size)
                if order is not None:
                    gaussian_node.validate_optimizer_state(order)
            except Exception:
                log.exception("Morton sort skipped for layer '%s', its row order is unchanged", layer_id)
                continue
            if order is not None:
                orders[layer_id] = order

        if not orders:
            return

        self._invalidate_collect_caches()

        for layer_id, order in orders.items():
            self.gaussians_nodes[layer_id].apply_morton_order(order)
            log.info("Morton-sorted %d gaussians in layer '%s'", len(order), layer_id)

        # Densification needs adjustment too, its per-Gaussian accumulators live on the strategy and are row-aligned.
        self.gaussians_strategy.permute_gaussian_rows(orders)

        # Row hooks were bound to the replaced parameters so they need a reapply.
        # Scoped to the sorted layers, the rest still hold valid hooks.
        self.reapply_component_only_row_hooks(orders.keys())

        # MCMC scenes hold the parameter objects, so they need a refresh.
        self.refresh_mcmc_scenes()

    def has_saved_training_parameters(self) -> bool:
        """Whether a validation-time parameter snapshot is currently held.

        The snapshot keeps references to the live tensors, so anything that replaces them has to run before it
        is taken.
        """
        return len(self.overriden_state) > 0 or any(
            len(node.overriden_parameters) > 0 for node in self.gaussians_nodes.values()
        )

    def save_training_parameters(self) -> None:
        if len(self.overriden_state) == 0:
            self.overriden_state["cuboid_tracks"] = self.cuboid_tracks
            self.overriden_state["obj_track_ids"] = copy.copy(self.obj_track_ids)

        for gaussian_node in self.gaussians_nodes.values():
            gaussian_node.save_training_parameters()

    @ScopedTimer("GaussiansComposite/restore_training_parameters")
    def restore_training_parameters(self) -> None:
        """
        Should be called at the end of the validation phase to revert any changes made by
        track_ply_override, remove_background_gaussians, and grpc asset edits.
        """
        self._invalidate_fused_gather_generation_plan()
        self._invalidate_collect_caches()  # edit-revert mutates model state; drop the cached collects
        if len(self.overriden_state) > 0:
            self.cuboid_tracks = self.overriden_state["cuboid_tracks"]
            self.obj_track_ids = self.overriden_state["obj_track_ids"]

        for gaussian_node in self.gaussians_nodes.values():
            gaussian_node.restore_training_parameters()

        self.overriden_state = {}

    def export_plys(
        self,
        export_dir: Path,
        format: GaussianExportFormat = GaussianExportFormat._3DGS,
        percentage_gaussians: float = 100,
    ) -> None:
        """
        Exports point clouds (PLY files) in the original 3DGS format or in the 3DGRT format for each node.

        For 3DGS format, it should be compatible with the original 3DGS implementation but differences
        between 3DGS/3DGUT/3DGRT rendering will cause slight differences when rendered with
        3rd-party 3DGS viewers.

        Args:
            export_dir (Path): The directory where the exported PLY files will be saved.
            format (GaussianExportFormat, optional): The export format to use.
                - GaussianExportFormat._3DGS: Exports in the original 3D Gaussian Splatting format.
                - GaussianExportFormat._3DGRT: Exports in the 3D Gaussian Ray Tracing format.
                Default is GaussianExportFormat._3DGS.
            percentage_gaussians (float, optional): The percentage (0, 100] of gaussians to export.
                Allows for subsampling the exported gaussians. Default is 100 (all gaussians).

        Returns:
            None

        """
        for key, gaussian_node in self.gaussians_nodes.items():
            if isinstance(gaussian_node, SHGaussianModel):
                gaussian_node.export_ply(export_dir / key, format=format, percentage_gaussians=percentage_gaussians)

    @ScopedTimer("GaussiansComposite/get_updated_cuboid_tracks")
    @torch.no_grad()
    def get_updated_cuboid_tracks(self) -> CuboidTracks:
        """return the updated tracks of every primitives"""
        updated_cuboid_tracks = []
        for gaussian_node in self.gaussians_nodes.values():
            if (
                isinstance(gaussian_node, RigidGaussianModel)
                and gaussian_node.owns_tracks
                and not gaussian_node.is_static
            ):
                # The node's own tracks — NOT a recomputed layer subset:
                # under component-only per-track selection the node (and
                # its tracks_calib parameter sizing) holds only the
                # non-disabled partition, and insert_asset keeps
                # node.cuboid_tracks and the calib in sync. Feeding the
                # full layer subset here would shape-mismatch the calib
                # (and disabled tracks were never calibrated anyway).
                updated_cuboid_tracks.append(gaussian_node.get_calibrated_cuboid_tracks())

        return (
            CuboidTracks.Ops.concatenate(updated_cuboid_tracks)
            if len(updated_cuboid_tracks) > 0
            else CuboidTracks.Factory.empty()
        )

    def get_number_of_gaussians_per_track(self) -> dict[str, dict[str, int]]:
        """
        Returns the number of gaussians for each node per track.
        """
        return {
            key: gaussian_node.get_number_of_gaussians_per_track()
            for key, gaussian_node in self.gaussians_nodes.items()
        }

    def parse_extra_ray_signals(self, extra_ray_signals: torch.Tensor, lidar_extra_ray_signals: bool) -> ExtraSignal:
        """
        Parse extra ray signals from the renderer output into an ExtraSignal object.

        Args:
            extra_ray_signals: Tensor containing extra signals from renderer, shape [..., total_signal_dims]
            lidar_extra_ray_signals: whether the extra ray signals are for lidar rays

        Returns:
            ExtraSignal: Object containing parsed signals as attributes
        """
        if extra_ray_signals is None or extra_ray_signals.shape[-1] == 0:
            return ExtraSignal()
        else:
            return ExtraSignal.from_packed_tensor(
                extra_ray_signals,
                self.lidar_extra_ray_signal_infos if lidar_extra_ray_signals else self.camera_extra_ray_signal_infos,
            )

    def collect_renderer_profilings(self) -> dict[str, float]:
        """
        Returns:
            dict: A dictionary containing the profiling information from both camera and lidar renderers.
        """
        if self.gaussians_renderer_camera == self.gaussians_renderer_lidar:
            return self.gaussians_renderer_camera.collect_profilings()
        else:
            profiling_camera = self.gaussians_renderer_camera.collect_profilings()
            profiling_lidar = self.gaussians_renderer_lidar.collect_profilings()
            return dict(Counter(profiling_camera) + Counter(profiling_lidar))
