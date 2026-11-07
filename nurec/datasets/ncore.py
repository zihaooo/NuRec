# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import concurrent.futures
import ctypes
import logging
import math
import platform

from collections import OrderedDict, defaultdict
from dataclasses import dataclass, replace
from enum import Enum, IntEnum, auto, unique
from pathlib import Path
from types import EllipsisType
from typing import (
    Any,
    Collection,
    Dict,
    Generator,
    List,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    TypeVar,
    cast,
    overload,
)
from weakref import WeakValueDictionary

import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
import PIL.Image as PILImage
import simplejpeg
import torch
import torch.utils.data
import tqdm

from gsplat.lidar_context import UnstructuredLidarFrameContext
from scipy import ndimage
from upath import UPath

import ncore
import ncore.data
import ncore.impl.common.transformations as ncore_transformations
import ncore.impl.data.util as ncore_data_util
import ncore.sensors
import nurec.utils.ncore_utils as ncore_utils

from nurec.config.base_schema import PerSensor, resolve_per_sensor
from nurec.config.dataset import LidarNominalModel, NCoreDatasetConfig
from nurec.config.nurec import NuRecConfig
from nurec.config.sensor import LidarModelsConfig
from nurec.datasets.base import (
    ActiveTileMappingPrewarmEntry,
    ActiveTileMappingPrewarmPlan,
    BaseDataset,
    BaseDataSource,
    RigTrajectoriesProvider,
)
from nurec.datasets.registry import register as register_dataset
from nurec.datasets.samplers.base import (
    BaseBatchSampler,
    BaseCameraPixelSampler,
    BaseFrameSampler,
    BaseLidarPointSampler,
    BaseSensorSampler,
    SkipCameraPixelSampler,
    SkipFrameSampler,
    SkipLidarPointSampler,
    SkipSensorSampler,
)
from nurec.datasets.track_split import TrackSplitChild, TrackSplitMetadata
from nurec.datasets.tracks import CuboidTracks, TrackFlags
from nurec.datasets.utils import (
    PackedMask,
    classify_dynamic_cuboid_tracks,
    color_pc_semantics,
    compute_camera_visible_intervals_cuboid_tracks,
    compute_cameras_valid_pixels_frame_mask,
    compute_cuboid_tracks,
    compute_points_outside_tracks,
    compute_valid_lidarpoints_all,
    compute_valid_lidarpoints_trafficlight_cameravisible,
    compute_valid_pixels_ego,
    compute_valid_pixels_sceneflow,
    compute_valid_pixels_trafficlight,
    get_indices_of_points_visible_in_image,
    lidar_frame_dynamic_flag,
    visualise_point_cloud,
)
from nurec.libs.vren.interface import (  # type: ignore
    image_points_to_world_rays_shutter_pose,
    world_points_to_image_points_shutter_pose,
)
from nurec.ncore_backends import DatasetRequest, resolve_dataset
from nurec.utils.batch import (
    CameraFrameLabels,
    CameraFreePoseViewGeometry,
    ConcreteCameraModelsUnion,
    ConcreteLidarModelsUnion,
    DataAndRenderingBatch,
    DataBatch,
    FrameMeta,
    LidarFrameLabels,
    RectSubsampled,
    UnstructuredLidarTilingData,
    compute_pixel_footprint,
    generate_grid_2d_indices,
)
from nurec.utils.geometry import (
    euler_2_so3,
    se3_matrix_to_tquat,
    tquat_to_se3_matrix,
)
from nurec.utils.lidar_firing_lattice import FiringLatticeEstimator
from nurec.utils.lidar_model import LidarModelBundle, get_lidar_model_parameters_with_fallbacks
from nurec.utils.misc import (
    assert_default_device_on_local_rank,
    assert_device_on_local_rank,
    compute_process_local_rng_seed,
    map_optional,
    set_default_device,
    to_torch,
    unpack_optional,
)
from nurec.utils.ncore_types import require_concrete_camera_parameters
from nurec.utils.profiling import ScopedTimer, TimingTag
from nurec.utils.types import (
    AABB3D,
    CameraFrustum,
    FrameConversion,
    HalfClosedInterval,
    NovelViewOverrides,
    PointCloud,
    PointCloudColorType,
    RayFlags,
    RigTrajectories,
    TrackPointCloud,
)
from nurec.utils.unstructured_lidar import (
    UnstructuredFov,
    build_tiling,
    fov_from_bounds,
    unstructured_model_from_fov,
)
from nurec.visualdebugger import get_visualdebugger


logger = logging.getLogger(__name__)


class UniqueSensorId(NamedTuple):
    """Represents a unique sensor ID along with its unique index for a given sensor type"""

    id: str
    idx: int


@unique
class _CameraRGBMode(Enum):
    """Internal camera-mask policy for projecting RGB onto LiDAR points."""

    ALL_PIXELS = auto()
    EGO_VALID_PIXELS = auto()
    FRAME_VALID_PIXELS = auto()


# Virtual-ray modes that SYNTHESIZE raydrop negatives, as opposed to "off" (synthesize
# nothing) and to the independent stored-no-return classification. Kept next to the
# config Literal in nurec/config/dataset.py; adding a mode means touching both.
_SYNTHESIZED_RAYDROP_MODES = ("model_elements", "aggregate_lattice")

# Inferred metric camera-axis (z) depth methods.
#
# NOTE: "ds-gt" is deliberately excluded. It is also metric z-depth, but it is synthetic
# ground truth and ds_to_ncore writes +inf for pixels beyond max_depth_m. The camera
# point-cloud paths only mask on `depth > 0`, which does not reject inf, so admitting
# "ds-gt" here would emit points at infinity. Callers that tolerate ds-gt check for it
# explicitly instead.
_METRIC_Z_DEPTH_METHODS = frozenset(
    {
        "MetricDepthAnythingV2",
        "vggt-z-depth",
    }
)


def is_metric_z_depth_method(method: str | None) -> bool:
    """Return whether depth metadata describes inferred metric camera-axis depth.

    Does not include ``ds-gt``; see the note on :data:`_METRIC_Z_DEPTH_METHODS`.
    """
    return method in _METRIC_Z_DEPTH_METHODS


def _fov_to_lo_hi(fov: ncore_data_util.FOV) -> tuple[float, float]:
    """Convert an ncore ``FOV`` (start_rad, span_rad, direction) to ``(lo, hi)`` rad.

    ncore's ``FOV`` is direction-aware: ``start_rad`` is the FIRST element and the
    span sweeps in ``direction``. For ``cw`` the sweep goes DOWN (``start`` is the
    high end, ``lo = start - span``); for ``ccw`` it goes UP (``start`` is the low
    end, ``hi = start + span``). This mirrors ncore's own bound reconstruction in
    ``RowOffsetStructuredSpinningLidarModel`` (lidar.py: fov_*.start_rad vs
    start_rad +/- span_rad by spinning_direction). Vertical FOV is always ``cw``;
    horizontal FOV follows the sensor's spinning direction, so this must NOT assume
    ``cw`` for azimuth. Returned as an ordered ``(lo, hi)``.
    """
    start = float(fov.start_rad)
    span = float(fov.span_rad)
    other = start + span if getattr(fov, "direction", "cw") == "ccw" else start - span
    return (start, other) if start <= other else (other, start)


# Stores already instantiated NCOREDataSources to allow sharing between val/train samplers.
# We use WeakValueDictionary so that when there are no references to a value, it is evicted
# from a dictionary. This means that as long as an NCORE{Train,Sequential}Dataset exists and holds
# a reference to an NCOREDataSource, it remains discoverable for reuse via its config hash.
# Once all samplers are dropped, the data source will be evicted from this cache, avoiding
# potential memory leaks.
# This somewhat hacky caching mechanism is to be replaced with caching on LightningDataModule
# instance later.
DATASOURCE_CACHE: WeakValueDictionary[NCoreDatasetConfig, NCOREDataSource] = WeakValueDictionary()


def _get_observation_track_ids_for_dynamic_filter(
    dynamic_track_ids: list[str],
    track_split_metadata: TrackSplitMetadata | None,
) -> set[str]:
    observation_track_ids = set(dynamic_track_ids)
    if track_split_metadata is None or track_split_metadata.is_empty:
        return observation_track_ids

    dynamic_track_id_set = set(dynamic_track_ids)
    clean_dynamic_track_id_set = {track_id.partition("@")[0] for track_id in dynamic_track_ids}
    for child in track_split_metadata.children:
        if (
            child.child_track_id in dynamic_track_id_set
            or child.child_track_id.partition("@")[0] in clean_dynamic_track_id_set
        ):
            observation_track_ids.add(child.parent_track_id)
    return observation_track_ids


@register_dataset("ncore")
def ncore_dataset_factory(config: NuRecConfig, split: str = "train") -> NCOREDataset:
    """
    A factory method which creates the appropriate sampler depending on split parameter while caching
    the underlying data source.
    """
    dataset_config = NCoreDatasetConfig.model_validate(config.dataset)

    if dataset_config not in DATASOURCE_CACHE:
        source = NCOREDataSource(
            dataset_config, config.sensor.lidar_models
        )  # bind to variable to avoid drop between creation and use
        DATASOURCE_CACHE[dataset_config] = source

        # Set up the all-frame -> training-only frame index mapping for PPISP
        # and bilateral-grid per-frame parameter indexing. Derived from the
        # EFFECTIVE training frame samplers (a frame is a training frame iff
        # some sampler group can return it) — NOT from the validation holdout
        # configuration, which is free to re-render training frames (e.g.
        # overfit diagnostics) without touching training indices. For
        # unrestricted samplers (``uniform``, ``timestamp``, ...) every frame
        # is a training frame and the public accessors on NCOREDataSource
        # fall back to the identity mapping.
        source.setup_training_frame_mapping(dataset_config.samplers)
    else:
        source = DATASOURCE_CACHE[dataset_config]

    match split:
        case "train":
            return NCORETrainDataset(source, dataset_config, split)
        case "val":
            return NCORESequentialDataset(source, dataset_config, split)
        case "train-sequential":
            return NCORESequentialDataset(source, dataset_config, split)
        case _:
            raise ValueError(f"Unexpected {split=}.")


class NCOREDataset(BaseDataset):
    """Abstract base class for functionality common to train and val splits"""

    def __init__(self, datasource: NCOREDataSource, config: NCoreDatasetConfig, split: str):
        self.datasource = datasource
        self.datasource._maybe_init_worker()
        self.camera_ids = datasource.train_camera_ids if "train" in split else datasource.val_camera_ids
        self.lidar_ids = datasource.train_lidar_ids if "train" in split else datasource.val_lidar_ids
        self.load_aux_depth = split != "train" or config.load_aux_depth_in_train
        self.load_aux_normal = split != "train" or config.load_aux_normal_in_train
        # The lidar_unstructured_* / lidar_ray_mode / use_real_lidar_rays config values
        # are PerSensor[...]: a single value applies to all lidars, or a {lidar_id: value}
        # dict (with an optional "default" key). Keep the config and resolve per sensor_id
        # via resolve_per_sensor(self.config.<field>, sensor_id) at each use site.
        self.config = config

        # Pre-build the per-sensor firing lattice in the main process (before the
        # DataLoader forks) so workers inherit the cache copy-on-write instead of
        # each rebuilding it. Only needed for the aggregate_lattice raydrop mode.
        for lidar_id in self.lidar_ids:
            if resolve_per_sensor(config.lidar_ray_mode, lidar_id) != "unstructured":
                continue
            if resolve_per_sensor(config.lidar_unstructured_raydrop_virtual_rays, lidar_id) != "aggregate_lattice":
                continue

            fov = self._resolve_unstructured_lidar_fov(lidar_id)

            # Populate internal cache
            self.datasource.get_firing_lattice(
                lidar_id,
                fov,
                max_frames=resolve_per_sensor(config.lidar_unstructured_raydrop_lattice_max_frames, lidar_id),
                n_time_modes=resolve_per_sensor(config.lidar_unstructured_raydrop_lattice_time_modes, lidar_id),
                n_bins_azimuth=resolve_per_sensor(config.lidar_unstructured_raydrop_lattice_n_bins_azimuth, lidar_id),
                n_bins_elevation=resolve_per_sensor(
                    config.lidar_unstructured_raydrop_lattice_n_bins_elevation, lidar_id
                ),
            )

        # We do not create rays in the dataset, so we don't need to
        # create the rig modules unless we are using depth data or normals
        aux_loader = datasource.aux_loader
        has_depth = self.load_aux_depth and aux_loader is not None and aux_loader.has_depth()
        has_normal = self.load_aux_normal and aux_loader is not None and aux_loader.has_normal()
        if has_depth or has_normal:
            rig_trajectories = self.datasource.get_rig_trajectories()
            self.camera_rig_module = CameraFreePoseViewGeometry.from_rig_trajectories(
                rig_trajectories, interp_with_rig=config.interp_with_rig
            )

    def get_datasource(self) -> NCOREDataSource:
        self.datasource._maybe_init_worker()
        return self.datasource

    # --- per-sensor unstructured-lidar FOV resolution -----------------------
    def _resolve_unstructured_lidar_fov(self, sensor_id: str) -> UnstructuredFov:
        """Resolve the unstructured per-sensor field of view.

        Returns an :class:`UnstructuredFov` with elevation bounds, azimuth bounds,
        and the azimuth periodicity flag (radians).

        Precedence: the per-sensor config FOV (lidar_unstructured_fov_vert_deg /
        _horiz_deg) WINS over the sensor model; else the model FOV; else -- when
        neither exists -- raise (elevation FOV is required for the unstructured
        tiling). Azimuth: config span if given, else the model's horizontal FOV,
        else full-360 periodic. A near-360 span (>=350 deg) is treated as periodic
        (spinning), in which case the azimuth bounds are the full circle
        ``[-pi, pi]``; a narrower span keeps the sensor's real ``[lo, hi]`` so the
        firing lattice / tiling do not waste bins on unobserved azimuth.
        """
        fov_vert_deg = resolve_per_sensor(self.config.lidar_unstructured_fov_vert_deg, sensor_id, default=None)
        fov_horiz_deg = resolve_per_sensor(self.config.lidar_unstructured_fov_horiz_deg, sensor_id, default=None)

        model_parameters = self.datasource.lidar_model_parameters.get(sensor_id)

        el_lo = el_hi = None

        # elevation: config override wins, else model.
        if fov_vert_deg is not None:
            el_lo = math.radians(float(fov_vert_deg[0]))
            el_hi = math.radians(float(fov_vert_deg[1]))
        elif model_parameters is not None:
            el_lo, el_hi = _fov_to_lo_hi(model_parameters.get_vertical_fov())

        if el_lo is not None and el_hi is not None and el_lo > el_hi:
            el_lo, el_hi = el_hi, el_lo

        # azimuth: config override wins, else model. A near-full span is periodic
        # (full circle); a narrower span keeps the real [lo, hi] bounds.
        _az_lo = _az_hi = None
        if fov_horiz_deg is not None:
            _az_lo = math.radians(float(fov_horiz_deg[0]))
            _az_hi = math.radians(float(fov_horiz_deg[1]))
        elif model_parameters is not None:
            # Direction-aware (cw vs ccw): the horizontal FOV's start_rad is NOT
            # necessarily the high end (unlike vertical).
            _az_lo, _az_hi = _fov_to_lo_hi(model_parameters.get_horizontal_fov())

        if el_lo is None or el_hi is None:
            raise ValueError(
                f"dataset.lidar_ray_mode='unstructured' needs the vertical FOV for sensor "
                f"'{sensor_id}', but it has no spinning lidar model and no config FOV. "
                f"Set dataset.lidar_unstructured_fov_vert_deg=[lo, hi] "
                f"(or {{'{sensor_id}': [lo, hi]}} for a per-sensor value)."
            )

        if _az_lo is None or _az_hi is None:
            # Documented fallback: with neither a config span nor a model horizontal
            # FOV, assume a full-360 periodic azimuth. Elevation above still raises,
            # because the tiling cannot guess an elevation extent -- but azimuth has a
            # safe, standard default for a model-less spinning sensor.
            _az_lo, _az_hi = -math.pi, math.pi

        # Periodicity and the +/-pi seam are decided in one place, shared with serving.
        return fov_from_bounds(
            _az_lo,
            _az_hi,
            el_lo,
            el_hi,
            source=f"dataset.lidar_unstructured_fov_horiz_deg for sensor '{sensor_id}'",
        )

    def _camera_frame_meta(
        self,
        sensor_id: str,
        sensor_frame_idx: int,
        *,
        sampled_pixels: RectSubsampled | None = None,
    ) -> FrameMeta:
        datasource = self.datasource
        unique_frame_idx = (
            datasource.camera_linear_start_frame_indices[sensor_id]
            + sensor_frame_idx
            - datasource.camera_frame_ranges[sensor_id].start
        )
        return FrameMeta(
            unique_sensor_idx=datasource.camera_unique_ids[sensor_id].idx,
            unique_frame_idx=unique_frame_idx,
            training_frame_idx=datasource.get_training_frame_idx_camera(unique_frame_idx),
            subsample=sampled_pixels,
        )

    def _lidar_frame_meta(self, sensor_id: str, sensor_frame_idx: int) -> FrameMeta:
        datasource = self.datasource
        unique_frame_idx = (
            datasource.lidar_linear_start_frame_indices[sensor_id]
            + sensor_frame_idx
            - datasource.lidar_frame_ranges[sensor_id].start
        )
        return FrameMeta(
            unique_sensor_idx=datasource.lidar_unique_ids[sensor_id].idx,
            unique_frame_idx=unique_frame_idx,
            training_frame_idx=datasource.get_training_frame_idx_lidar(unique_frame_idx),
            subsample=None,
        )

    def get_camera_data_batch(
        self,
        sensor_id: str,
        sensor_frame_idx: int,
        sampled_pixels: RectSubsampled | None = None,
        *,
        load_rgb: bool = True,
        load_aux_geometry: bool = True,
    ) -> DataBatch.Camera:
        """
        This method returns a `DataBatch` based on the query criteria:

        - sensor_id: The sensor id.
        - sensor_frame_idx: The underlying "raw" index of the ncore sensor.
        - sampled_pixels: The crop/resize information.
        - load_rgb: Whether to decode the RGB image.
        - load_aux_geometry: Whether to load configured auxiliary depth and normal labels.

        Note this function is designed with the assumption there is only a single sequence & a single chunk in the dataset.
        """
        datasource = self.datasource
        datasource._maybe_init_worker()

        camera_sensor = datasource.camera_sensors[sensor_id]
        frame_timestamps_startend_us = (
            camera_sensor.get_frame_timestamp_us(sensor_frame_idx, frame_timepoint=ncore.data.FrameTimepoint.START),
            camera_sensor.get_frame_timestamp_us(sensor_frame_idx, frame_timepoint=ncore.data.FrameTimepoint.END),
        )

        frame_rgb: torch.Tensor | None = None
        if load_rgb:
            # fetch the camera labels
            with ScopedTimer("get_frame_image/NCOREDataset.get_camera_data_batch", TimingTag.DATALOADER):
                # decode image data on CPU
                frame_image_array = datasource.decode_image_cpu(
                    camera_sensor.get_frame_handle(sensor_frame_idx),
                    # perform subsampling while decoding, if required
                    subsample_factor=map_optional(sampled_pixels, lambda x: x.subsample_factor),
                )

            # Note: In the case of a non-None subsampled rect we also need to
            # - adjust the camera intrinsics (see `RectSubsampled.apply_to_model`)
            # - adjust the rolling shutter effects (see `RectSubsampled.apply_to_camera_rolling_shutter`)
            # But we won't do them here to keep the batch structure simple and clean. Instead we will carry the
            # subsample information in the camera_meta, and handle these when they are needed.
            if sampled_pixels is not None:
                # we only need to crop the decoded image array, as it was subsampled during decoding already
                frame_image_array = sampled_pixels.crop_array(frame_image_array)

            height, width, _ = frame_image_array.shape
            frame_rgb = to_torch(frame_image_array.astype(np.float32) / 255.0, device="cpu").unsqueeze(0)
        elif sampled_pixels is not None:
            # Eligibility indexing needs flags, semantics and calibrated rays, but never RGB. Derive the exact
            # sampled shape without decoding every source image during the one-time startup scan.
            height, width = sampled_pixels.height, sampled_pixels.width
        else:
            camera_width, camera_height = datasource.camera_models[sensor_id].resolution
            height, width = int(camera_height.item()), int(camera_width.item())
        # save the info to the batch
        batch = DataBatch.Camera(
            meta=[self._camera_frame_meta(sensor_id, sensor_frame_idx, sampled_pixels=sampled_pixels)],
            labels=CameraFrameLabels(rgb=frame_rgb),
        )

        # handling flags
        # TODO(ruilong): The name of "RayFlags" should be updated since we move away from the ray-based batch format.
        initial_flags = RayFlags.RGB_LABEL.value if frame_rgb is not None else 0
        flags = torch.full(
            (height, width, 1),
            initial_flags,
            dtype=torch.int32,
            device="cpu",
        )

        # store invalid flag of pixels (usually only required in validation mode)
        valid_mask = datasource.cameras_frame_valid_pixels_masks[sensor_id][sensor_frame_idx].unpacked()[
            ..., np.newaxis
        ]
        if sampled_pixels is not None:
            valid_mask = valid_mask.astype(np.float32)
            valid_mask = sampled_pixels.apply_to_array(valid_mask, interpolation=cv2.INTER_LINEAR)
            valid_mask = valid_mask > 0.5
        flags[~valid_mask] |= RayFlags.INVALID

        # handing aux data (and update flags)
        aux_loader = datasource.aux_loader
        if aux_loader is not None:
            # load semantic segmentation image
            semantics = np.asarray(aux_loader.get_semantic_segmentation(sensor_id, frame_timestamps_startend_us[1]))[
                ..., np.newaxis
            ]
            if sampled_pixels is not None:
                semantics = sampled_pixels.apply_to_array(semantics, interpolation=cv2.INTER_NEAREST)

            # classify sampled rays for sky and road
            sky_mask = semantics == datasource.sensor_sky_class_ids[sensor_id]
            road_mask = semantics == datasource.sensor_road_class_ids[sensor_id]
            vehicle_mask = np.logical_or.reduce(
                [semantics == class_id for class_id in datasource.sensor_vehicle_classes_ids[sensor_id]]
            )

            # update ray types
            flags |= RayFlags.VALID_SEMANTIC.value
            flags[sky_mask] |= RayFlags.SKY_SEMANTIC.value
            flags[road_mask] |= RayFlags.ROAD_SEMANTIC.value
            flags[vehicle_mask] |= RayFlags.VEHICLE_SEMANTIC.value

            # Note(ruilong): We probably don't need it anymore since we have set the flags.
            batch.labels.semantic = to_torch(semantics, device="cpu").unsqueeze(0)

            if load_aux_geometry and (
                (self.load_aux_depth and aux_loader.has_depth()) or (self.load_aux_normal and aux_loader.has_normal())
            ):
                # Get full resolution
                image_width, image_height = datasource.camera_model_parameters[sensor_id].resolution

                pixels = unpack_optional(sampled_pixels)
                meta = batch.meta[0]
                sensor_model = cast(ConcreteCameraModelsUnion, self.camera_rig_module.get_sensor_model(meta))

                # Only compute these once. They are need by both normals and depth
                elements, sensor_rays = self.camera_rig_module._compute_elements_and_sensor_rays(
                    sensor_model, torch.device("cpu")
                )

                if self.load_aux_depth and aux_loader.has_depth():
                    # Load dense distance supervision from aux-data
                    depth = aux_loader.get_depth(
                        sensor_id, frame_timestamps_startend_us[1], (image_width, image_height)
                    )[..., np.newaxis]
                    depth = pixels.apply_to_array(depth, interpolation=cv2.INTER_NEAREST)

                    distance = torch.abs(
                        to_torch(depth[..., 0], device="cpu") / sensor_rays[..., 2]
                    )  # convert z-axis depth into distance along rays
                    distance = torch.reshape(distance, (height, width, 1))

                    match aux_loader.get_depth_meta(sensor_id)["method"]:
                        case "DepthAnythingV2":
                            # TODO: This is a relative depth estimate, but we can use Lidar to scale it to metric space
                            batch.labels.relative_distance = (
                                datasource.world_to_nurec.target_scale * distance
                            ).unsqueeze(0)
                        case method if method == "ds-gt" or is_metric_z_depth_method(method):
                            batch.labels.metric_distance = (
                                datasource.world_to_nurec.target_scale * distance
                            ).unsqueeze(0)
                        case _:
                            raise ValueError(
                                "Depth estimation method must be one of [DepthAnythingV2, MetricDepthAnythingV2]"
                            )

                if self.load_aux_normal and aux_loader.has_normal():
                    # Load normal supervision from aux-data
                    normal = aux_loader.get_normal(
                        sensor_id, frame_timestamps_startend_us[1], (image_width, image_height)
                    )  # shape: [image_height, image_width, 3]
                    normal = np.reshape(pixels.apply_to_array(normal, interpolation=cv2.INTER_LINEAR), (-1, 3, 1))

                    pose_and_timestamps_startend_return = self.camera_rig_module.get_poses_and_timestamps_startend(meta)
                    T_sensor_world_startend = pose_and_timestamps_startend_return.T_sensor_world_startend
                    timestamps_startend_us = pose_and_timestamps_startend_return.timestamps_startend_us

                    world_rays_return = sensor_model.pixels_to_world_rays_shutter_pose(
                        elements.reshape(-1, 2),
                        T_sensor_world_startend[0],
                        T_sensor_world_startend[1],
                        start_timestamp_us=cast(int, timestamps_startend_us[0].item()),
                        end_timestamp_us=cast(int, timestamps_startend_us[1].item()),
                        camera_rays=sensor_rays.reshape(-1, 3),
                        return_timestamps=False,
                        return_T_sensor_worlds=True,
                    )

                    # T_sensor_worlds transforms from sensor to world, normals are transformed with this
                    # transformation to convert from sensor to world space
                    rotation_sensor_to_world = unpack_optional(world_rays_return.T_sensor_worlds).numpy()[:, :3, :3]
                    normal_world = np.matmul(rotation_sensor_to_world, normal).squeeze(-1)

                    # normalize the normals for valid pixels
                    norms = np.linalg.norm(normal_world, axis=1)

                    valid_normal_mask = norms > 0.1
                    norms[~valid_normal_mask] = 1
                    batch.labels.normals = to_torch(normal_world / norms[:, np.newaxis], device="cpu").reshape(
                        (1, height, width, 3)
                    )

                    flags[np.reshape(valid_normal_mask, (height, width, 1))] |= RayFlags.VALID_NORMAL.value

        batch.labels.flags = flags.unsqueeze(0)
        _ = batch.labels.n_valid_rgb  # force computation of n_valid_rgb at data batch loading time
        _ = batch.labels.n_valid  # force computation of n_valid at data batch loading time
        _ = batch.labels.n_harmonized  # force computation of n_harmonized at data batch loading time
        _ = batch.labels.n_valid_bg  # force computation of n_valid_bg at data batch loading time
        return batch

    def _synthesize_raydrop_virtual_rays(
        self,
        *,
        sensor_id: str,
        fov_model: ConcreteLidarModelsUnion | None,
        return_dirs_sensor: np.ndarray,
        return_model_elements: np.ndarray | None,
        frame_start_us: int,
        frame_end_us: int,
        fov: UnstructuredFov,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None] | None:
        """Synthesize model-free raydrop "virtual" rays for the unstructured path.

        Virtual rays are directions the sensor's FOV covers but where this frame got
        no return; they are appended to the (N, 1) return list with RayFlags.DROPPED
        so the raydrop loss + metric have negative (no-return) examples to learn from
        -- the model-free path renders only observed return rays, so without these it
        would have no raydrop supervision at all.

        Returns ``(dirs_sensor[M, 3] float32, timestamps_us[M] int64,
        model_elements[M, 2] int64 | None)`` or ``None`` when no virtual rays are
        produced. ``model_elements`` is populated only for the 'model_elements'
        mode (the modeled-at-elements ray path needs it); it is ``None`` for
        'aggregate_lattice' (which is real-ray-only, guarded in NuRecConfig).

        Modes:
          - ``model_elements``: complement of the returning sensor-grid elements;
            exact directions + exact per-element sweep timestamps from the model.
          - ``aggregate_lattice``: sample directions in unoccupied FOV tiles of the
            RETURN-ONLY occupancy; assign each a relative frame time from the
            per-sensor firing-lattice estimate (model-free). Timestamps that the
            lattice cannot resolve (NaN) are dropped -- never fabricated.
        """
        mode = resolve_per_sensor(self.config.lidar_unstructured_raydrop_virtual_rays, sensor_id)

        # Frame duration in integer microseconds (timestamps are integer us). The
        # per-ray absolute timestamp is frame_start_us + rel_t * frame_dur, cast to
        # int64 below; rel_t is float in [0, 1] so that product is float, then
        # floored on cast -- sub-microsecond phase is not representable anyway.
        frame_dur = max(frame_end_us - frame_start_us, 1)

        if mode == "model_elements":
            if fov_model is None or return_model_elements is None:
                raise ValueError(
                    "lidar_unstructured_raydrop_virtual_rays='model_elements' requires a "
                    f"spinning lidar model and per-return model elements for sensor '{sensor_id}'."
                )

            n_rows, n_cols = fov_model.n_rows, fov_model.n_columns
            grid = torch.cartesian_prod(torch.arange(n_rows), torch.arange(n_cols))  # [n_rows*n_cols, 2]
            grid_lin = grid[:, 0] * n_cols + grid[:, 1]

            ret = torch.from_numpy(return_model_elements.astype(np.int64))
            ret_lin = ret[:, 0] * n_cols + ret[:, 1]

            drop_elems = grid[~torch.isin(grid_lin, ret_lin)]  # [M, 2]

            if drop_elems.shape[0] == 0:
                return None

            v_dirs = fov_model.elements_to_sensor_rays(drop_elems).to(torch.float32).cpu().numpy()
            v_angles = fov_model.elements_to_sensor_angles(drop_elems)
            v_rel = fov_model.sensor_angles_relative_frame_times(v_angles).to(torch.float64).cpu().numpy()
            v_ts = (frame_start_us + v_rel * frame_dur).astype(np.int64)

            return v_dirs.astype(np.float32), v_ts, drop_elems.cpu().numpy().astype(np.int64)

        if mode == "aggregate_lattice":
            # Return-ONLY occupancy: build a frame context from the return rays so
            # sample_unoccupied_tile_rays proposes only genuinely empty tiles.
            ret_dirs = to_torch(return_dirs_sensor, device="cpu", dtype=torch.float32)
            ret_ts = torch.arange(ret_dirs.shape[0], dtype=torch.int64)  # occupancy only needs relative order

            # Occupancy-grid resolution. The frame-context auto (ray-count) grid is
            # FAR too coarse for raydrop: a dense return frame fills nearly every
            # tile, so virtual-ray recall of the true model_elements drop set
            # collapses to ~1%. Match the SENSOR grid (config override, else the
            # spinning model's row/col count) so empty cells are resolved at the
            # firing granularity -> ~96-98% drop-set recall (raydrop coverage exp).
            occ_n_az = resolve_per_sensor(self.config.lidar_unstructured_raydrop_occupancy_n_bins_azimuth, sensor_id)
            occ_n_el = resolve_per_sensor(self.config.lidar_unstructured_raydrop_occupancy_n_bins_elevation, sensor_id)
            if occ_n_az is None and occ_n_el is None and fov_model is not None:
                occ_n_az = int(fov_model.n_columns)
                occ_n_el = int(fov_model.n_rows)
            return_ctx = UnstructuredLidarFrameContext(
                ret_dirs,
                timestamps_us=ret_ts,
                model=unstructured_model_from_fov(fov),
                n_bins_azimuth=occ_n_az,
                n_bins_elevation=occ_n_el,
            )

            gen = torch.Generator(device="cpu").manual_seed(0)

            v_dirs_t, _ = return_ctx.sample_unoccupied_tile_rays(
                resolve_per_sensor(self.config.lidar_unstructured_raydrop_rays_per_tile, sensor_id), generator=gen
            )

            if v_dirs_t.shape[0] == 0:
                return None

            # timestamps from the per-sensor firing-lattice estimate (model-free).
            # Multi-cover sensors: a sampled direction may have been fired at K
            # distinct times per frame, so emit one virtual ray PER visit time
            # (each is a genuine firing event / raydrop negative). Single-cover
            # (n_time_modes=1) collapses to exactly one ray per sampled direction
            # -- identical to the original behaviour.
            lattice = self.datasource.get_firing_lattice(
                sensor_id,
                fov,
                max_frames=resolve_per_sensor(self.config.lidar_unstructured_raydrop_lattice_max_frames, sensor_id),
                n_time_modes=resolve_per_sensor(self.config.lidar_unstructured_raydrop_lattice_time_modes, sensor_id),
                n_bins_azimuth=resolve_per_sensor(
                    self.config.lidar_unstructured_raydrop_lattice_n_bins_azimuth, sensor_id
                ),
                n_bins_elevation=resolve_per_sensor(
                    self.config.lidar_unstructured_raydrop_lattice_n_bins_elevation, sensor_id
                ),
            )

            v_modes = lattice.estimate_rel_t_modes(v_dirs_t)  # [M, K] in [0,1], NaN-padded
            valid = ~torch.isnan(v_modes)  # [M, K]

            if not bool(valid.any()):
                return None

            ray_idx, mode_idx = torch.nonzero(valid, as_tuple=True)  # each (dir, mode) pair

            v_dirs_expanded = v_dirs_t[ray_idx]
            v_rel_modes = v_modes[ray_idx, mode_idx].to(torch.float64)
            v_ts = (frame_start_us + v_rel_modes.cpu().numpy() * frame_dur).astype(np.int64)

            return v_dirs_expanded.cpu().numpy().astype(np.float32), v_ts, None

        raise ValueError(f"Unknown lidar_unstructured_raydrop_virtual_rays mode: {mode!r}")

    def _build_unstructured_tiling(
        self,
        *,
        sensor_id: str,
        lidar_sensor,
        sensor_frame_idx: int,
        sampled_pixels: RectSubsampled | np.ndarray | EllipsisType,
        sampled_filtered: np.ndarray | None,
        stored_no_return_keep: np.ndarray | None,
        virtual_dirs_sensor: np.ndarray | None,
        sparse_timestamps: np.ndarray,
    ) -> UnstructuredLidarTilingData:
        """Build the per-frame unstructured render tiling in the worker (OFF the
        render loop).

        Builds a gsplat ``UnstructuredLidarFrameContext`` (analytic projection + occupancy
        tiling + auto-resolution) from the frame's sensor-frame ray directions + the
        per-ray timestamps, then serializes ONLY the tiling tensors + scalar metadata
        via ``to_tensors`` (the context object itself is discarded -- the renderer
        rebuilds it at zero compute via ``from_tensors``). The exact sensor FOV
        (azimuth + elevation) comes from the spinning lidar model or a per-sensor
        config override, so the FOV pre-cull is exact and azimuth periodicity is known.

        ``sparse_timestamps`` must already span the combined N return + M virtual rays
        (aligned 1:1 with the directions here) so the tiling matches the labels/flags.
        """
        # Single sensor-frame ray-direction read (unit-norm [N, 3], cheap for V4 --
        # no distance multiply), brought into the same index space as the return list.
        # The bundle is ORIGINAL length, so it has to be filtered and subsampled the
        # same way the per-ray arrays were: ignored stored no-returns first (they are
        # not part of the return list any more), then the point subsample, which is
        # expressed in filtered coordinates once anything was dropped.
        dirs_sensor = np.asarray(lidar_sensor.get_frame_ray_bundle_direction(sensor_frame_idx))
        if stored_no_return_keep is not None:
            dirs_sensor = dirs_sensor[stored_no_return_keep]
        if isinstance(sampled_pixels, np.ndarray):
            dirs_sensor = dirs_sensor[unpack_optional(sampled_filtered)]

        if virtual_dirs_sensor is not None:
            # append virtual-ray sensor-frame dirs so the render tiling spans all
            # N+M rays (aligned 1:1 with sparse_timestamps + labels + flags).
            dirs_sensor = np.concatenate([np.asarray(dirs_sensor), virtual_dirs_sensor], axis=0)

        # FOV bounds: per-sensor config FOV wins, else the spinning lidar model;
        # raises if neither provides the vertical FOV (e.g. a model-less datasource
        # with no lidar_unstructured_fov_vert_deg for this sensor).
        fov = self._resolve_unstructured_lidar_fov(sensor_id)

        return build_tiling(
            to_torch(dirs_sensor, device="cpu", dtype=torch.float32),
            to_torch(sparse_timestamps.squeeze(-1), device="cpu", dtype=torch.int64),
            fov,
        )

    def _read_return_rays(
        self,
        *,
        lidar_sensor,
        sensor_frame_idx: int,
        return_index: int,
        unstructured: bool,
        model_rays: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Read this frame's per-return intensity + distance + (optional) world rays.

        Returns ``(intensity[N,1], distance[N,1], return_valid[N], rays[N,6]
        world origin+dir, rays_timestamps_us[N],
        T_sensor_world_end[4,4] | None)``.

        Two ray-geometry modes:
          - ``model_rays`` (use_real_lidar_rays=False): FAST path -- read the ncore
            per-return distance directly (no point-cloud transform; fast for V3 and
            V4). ``rays`` is an empty stub (the modeled world rays are generated later
            in to_rendering_data from the sensor model); ``T_sensor_world_end`` is
            ``None``. Per-ray timestamps are still read for the unstructured (N, 1)
            path (needed for the rolling-shutter pose), else an empty stub.
          - real rays (use_real_lidar_rays=True): reconstruct the true per-return
            world rays by motion-compensating UNIT-distance endpoints, so
            ``rays = [xyz_s, normalize(xyz_e - xyz_s)]`` where xyz_e is the unit
            endpoint, and ``distance`` is the sensor's own measured range (scaled to
            NuRec units) rather than ``|xyz_e - xyz_s|``. Compensating unit endpoints
            instead of measured ones keeps the direction well defined for stored
            no-return emissions, whose range payload is NaN.

        Why the real-ray path pays the motion-compensation cost (and cannot use the
        cheap distances of the fast path): the render itself only needs sensor-frame
        directions + per-ray timestamps (the rolling-shutter pose is applied in-kernel
        from viewmat_start/viewmat_end). BUT the per-return world START point (xyz_s)
        is consumed downstream by the metric point-cloud reconstruction
        (systems/gaussians.py: xyz = ray_origin + ray_dir * distance), where
        ray_origin = xyz_s carries the rolling-shutter ego-motion. A single frame-end
        sensor origin would drop that and silently change chamfer / depth metrics.
        ``T_sensor_world_end`` is returned so the caller can place virtual (raydrop)
        rays at the same frame-end sensor pose.
        """
        intensity = lidar_sensor.get_frame_ray_bundle_return_intensity(sensor_frame_idx, return_index=return_index)[
            ..., np.newaxis
        ]  # [N,1]
        return_valid = lidar_sensor.get_frame_ray_bundle_return_valid_mask(sensor_frame_idx, return_index=return_index)

        if model_rays:
            distance_ncore = lidar_sensor.get_frame_ray_bundle_return_distance_m(
                frame_index=sensor_frame_idx, return_index=return_index
            )[..., np.newaxis]  # [N,1]
            # scale measured distances from ncore -> nurec units if necessary
            target_scale = self.datasource.world_to_nurec.target_scale
            distance = target_scale * distance_ncore if target_scale != 1.0 else distance_ncore

            rays = np.empty((0, 6), dtype=np.float32)  # stub (modeled rays generated later)
            rays_timestamps_us = (
                lidar_sensor.get_frame_ray_bundle_timestamp_us(sensor_frame_idx)
                if unstructured
                else np.empty((0,), dtype=np.uint64)
            )
            return intensity, distance, return_valid, rays, rays_timestamps_us, None

        # The sensor provides motion-compensated start and end points; derive rays
        # from those without applying motion compensation a second time.
        T_sensor_world_end = self.datasource.world_to_nurec.transform_poses(
            lidar_sensor.get_frames_T_sensor_target(
                "world",
                sensor_frame_idx,
                frame_timepoint=ncore.data.FrameTimepoint.END,
            )
        )
        rays_timestamps_us = lidar_sensor.get_frame_ray_bundle_timestamp_us(sensor_frame_idx)
        pc = lidar_sensor.get_frame_point_cloud(
            sensor_frame_idx,
            motion_compensation=True,
            with_start_points=True,
            return_index=return_index,
        )
        xyz_s = ncore_transformations.transform_point_cloud(unpack_optional(pc.xyz_m_start), T_sensor_world_end)
        xyz_e = ncore_transformations.transform_point_cloud(pc.xyz_m_end, T_sensor_world_end)
        xyz_v = xyz_e - xyz_s

        # T_sensor_world_end already carries world_to_nurec.target_scale, so xyz_v is in
        # NuRec units and its norm IS the measured range: no separate distance read and
        # no rescaling. Computed once and reused as the normalization divisor below.
        distance = np.linalg.norm(xyz_v, axis=1, keepdims=True)

        # A V4 bundle may keep every emitted ray, and a no-return one carries a NaN
        # range, so its endpoint (direction * range) is NaN, and with it both xyz_v and
        # the distance just computed. Distance MUST stay NaN, since there is no
        # measurement, but the direction is still known: it is stored per ray, and
        # raydrop supervision needs it. Recover the direction for exactly those rays
        # and give them a unit divisor, so `xyz_v / divisor` normalizes them without a
        # second norm over the whole frame.
        #
        # Only the endpoint goes NaN, never the origin: xyz_m_start comes from the
        # sensor pose translation, not from the point. Measured on the ET30X V4 clip
        # (437040 rays/frame, 113150 no-return): every no-return has a finite start, a
        # NaN end and a finite stored direction; every return has a finite end. The
        # check is one isfinite pass, skipped on V3 and on V4 clips that record
        # returns only, e.g. the PAI V4 sequences.
        divisor = distance
        no_return = ~np.isfinite(xyz_v).all(axis=1)
        if bool(no_return.any()):
            dirs_sensor = np.asarray(lidar_sensor.get_frame_ray_bundle_direction(sensor_frame_idx))[no_return]
            # Sensor-frame -> world: rotation only, the ray origin is xyz_s already.
            # Stored directions are unit length, so the divisor is 1 for these rays.
            xyz_v[no_return] = dirs_sensor @ np.asarray(T_sensor_world_end)[:3, :3].T
            divisor = np.where(no_return[:, np.newaxis], 1.0, distance)

        rays = np.concatenate([xyz_s, xyz_v / np.maximum(divisor, 1e-6)], axis=-1)
        return intensity, distance, return_valid, rays, rays_timestamps_us, T_sensor_world_end

    def get_lidar_data_batch(
        self,
        sensor_id: str,
        sensor_frame_idx: int,
        sampled_pixels: RectSubsampled | np.ndarray | EllipsisType = Ellipsis,
        return_index: int = 0,
    ) -> DataBatch.Lidar:
        """
        This method returns a `DataBatch` based on the query criteria:

        - sensor_id: The sensor id.
        - sensor_frame_idx: The underlying "raw" index of the ncore sensor.
        - pixel_samples: The crop/resize information. (Not supported for now)
        - return_index: The per-frame return index of the lidar sensor (for multi-return lidars).

        Note this function is designed with the assumption there is only a single sequence & a single chunk in the dataset.
        """
        if isinstance(sampled_pixels, RectSubsampled):
            raise NotImplementedError("RectSubsampled subsampling for lidar is not supported for now")

        datasource = self.datasource
        datasource._maybe_init_worker()

        lidar_sensor = datasource.lidar_sensors[sensor_id]

        # Ray mode selection (per-sensor: these are PerSensor config values).
        # model_rays = the rays come from the sensor MODEL (the fitted spinning-lidar
        # (row, col) grid), as opposed to the real per-return rays reconstructed from
        # the point cloud. "measured" would be a misnomer -- the returns are measured
        # in both cases; what differs is whether the RAY GEOMETRY is modeled or real.
        unstructured = resolve_per_sensor(self.config.lidar_ray_mode, sensor_id) == "unstructured"
        model_rays = not bool(resolve_per_sensor(self.config.use_real_lidar_rays, sensor_id))
        # Stored no-return emissions are a property of the DATASET (NCore V4 keeps the
        # rays the sensor fired but got no return for), independent of the synthesized
        # virtual-ray mode. Only the unstructured (N, 1) list needs them classified: the
        # structured range-view grid already represents non-returning beams as grid
        # cells that no return scatters into.
        use_stored_no_returns = unstructured and bool(
            resolve_per_sensor(self.config.lidar_unstructured_raydrop_use_stored_no_returns, sensor_id)
        )

        # The sensor MODEL (and its per-return "model elements", i.e. (row, col) grid
        # indices) is only needed when a grid is involved: the structured range-view,
        # or the unstructured-but-modeled variant that synthesises rays at those
        # elements. Pure model-free real-ray mode (unstructured + use_real_lidar_rays)
        # needs neither the model nor the elements. (The model_elements raydrop mode
        # is forbidden with use_real_lidar_rays=True by NuRecConfig, so it only occurs
        # in the modeled path where needs_model is already True.)
        needs_model = (not unstructured) or model_rays

        height: int | None = None
        width: int | None = None
        if needs_model:
            lidar_model = unpack_optional(
                datasource.lidar_models[sensor_id],
                msg=f"NCOREDataSource: lidar model parameters are mandatory for this datasource but not available for '{sensor_id}'",
            )
            height = lidar_model.n_rows
            width = lidar_model.n_columns

        model_elements: npt.NDArray[np.uint16] | None = None
        if needs_model:
            model_elements = unpack_optional(
                lidar_sensor.get_frame_ray_bundle_model_element(sensor_frame_idx),
                msg="Lidar sensor must provide model elements",
            )

        # Per-return intensity / distance / (real) world rays / timestamps. The
        # model_rays vs real-ray split (fast ncore distances vs motion-compensated
        # point cloud) lives in _read_return_rays. T_sensor_world_end is returned on
        # the real-ray path only (None otherwise) for placing virtual rays below.
        _intensity, _distance, _return_valid, _rays, _rays_timestamps_us, T_sensor_world_end = self._read_return_rays(
            lidar_sensor=lidar_sensor,
            sensor_frame_idx=sensor_frame_idx,
            return_index=return_index,
            unstructured=unstructured,
            model_rays=model_rays,
        )

        # Rays dropped from the per-ray arrays below (stored no-returns that this
        # sensor is configured to ignore). None means "nothing dropped", in which case
        # the filtered and original index spaces coincide.
        stored_no_return_keep: npt.NDArray[np.bool_] | None = None
        # Subsample index into the (possibly filtered) per-ray arrays. Only meaningful
        # when sampled_pixels is an explicit index array; it equals sampled_pixels
        # unless rays were dropped above, in which case it is remapped onto the
        # filtered list.
        sampled_filtered: npt.NDArray[np.int64] | None = None
        if unstructured and not use_stored_no_returns and not bool(_return_valid.all()):
            # The dataset stores no-return emissions but this sensor is configured to
            # ignore them. Gating only the raydrop / DROPPED writes would leave them in
            # the ray list carrying their NaN range and intensity, with no DROPPED bit
            # to mask them, so the distance and intensity losses would train against
            # NaN targets. Drop them from every per-ray array instead, which restores
            # exactly the pre-V4 population: returns only.
            stored_no_return_keep = _return_valid
            _intensity = _intensity[stored_no_return_keep]
            _distance = _distance[stored_no_return_keep]
            model_elements = map_optional(model_elements, lambda x: x[stored_no_return_keep])
            if not model_rays:
                _rays = _rays[stored_no_return_keep]
            if _rays_timestamps_us.shape[0] > 0:
                _rays_timestamps_us = _rays_timestamps_us[stored_no_return_keep]
            _return_valid = _return_valid[stored_no_return_keep]

        if isinstance(sampled_pixels, np.ndarray):
            # sampled_pixels indexes the ORIGINAL ray bundle and MUST keep doing so:
            # reads further down (ray-bundle directions, the frame valid mask, lidar
            # semantics) all come straight from the sensor at full length. Only the
            # per-ray arrays above were filtered, so index those with a remapped copy;
            # samples that selected a dropped ray fall out rather than shifting onto a
            # neighbour.
            subsample = sampled_pixels
            if stored_no_return_keep is not None:
                remap = np.full(stored_no_return_keep.shape[0], -1, dtype=np.int64)
                remap[np.flatnonzero(stored_no_return_keep)] = np.arange(int(stored_no_return_keep.sum()))
                subsample = remap[sampled_pixels]
                subsample = subsample[subsample >= 0]
            sampled_filtered = subsample
            _intensity = _intensity[subsample]
            _distance = _distance[subsample]
            _return_valid = _return_valid[subsample]
            model_elements = map_optional(model_elements, lambda x: x[subsample])
            if not model_rays:
                _rays = _rays[subsample]
            # Per-ray timestamps exist in every unstructured / real-ray case.
            if _rays_timestamps_us.shape[0] > 0:
                _rays_timestamps_us = _rays_timestamps_us[subsample]

        # --- Unstructured raydrop virtual rays -----------------------------------
        # Synthesize "virtual" (dropped) rays -- directions the sensor's FOV covers
        # but where this frame got no return -- and APPEND them to the return arrays
        # BEFORE n_returns is fixed, so all downstream label/flag/sparse/tiling code
        # naturally spans the combined (N + M) list. They carry NaN range/intensity
        # and are flagged RayFlags.DROPPED below (tracked via n_virtual_rays).
        n_virtual_rays = 0
        _virtual_dirs_sensor = None  # [M, 3] sensor-frame dirs of virtual rays (for the tiling)
        if (
            unstructured
            and resolve_per_sensor(self.config.lidar_unstructured_raydrop_virtual_rays, sensor_id)
            in _SYNTHESIZED_RAYDROP_MODES
        ):
            # Return-ONLY sensor-frame directions for aggregate_lattice occupancy, so
            # sample_unoccupied_tile_rays proposes negatives only where the sensor
            # genuinely measured nothing. The ray bundle spans every EMITTED ray, which
            # on a dataset with stored no-returns includes directions that are
            # themselves raydrop negatives; leaving them in would mark their tiles
            # occupied and suppress sampling exactly where negatives are known to exist.
            # Filtering here also keeps a stored no-return from being counted twice,
            # once from the stored classification and once as a synthesized ray.
            # Index spaces: the ray bundle is ORIGINAL length, while _return_valid has
            # already been filtered and subsampled with the other per-ray arrays. Drop
            # and subsample the bundle the same way instead of indexing it with a mask
            # that no longer describes it.
            _ret_dirs_sensor = np.asarray(lidar_sensor.get_frame_ray_bundle_direction(sensor_frame_idx))
            if stored_no_return_keep is not None:
                _ret_dirs_sensor = _ret_dirs_sensor[stored_no_return_keep]
            if isinstance(sampled_pixels, np.ndarray):
                _ret_dirs_sensor = _ret_dirs_sensor[unpack_optional(sampled_filtered)]
            # _return_valid is now aligned with _ret_dirs_sensor, so this keeps only the
            # rays that actually returned: occupancy must not count a no-return as a hit.
            _ret_dirs_sensor = _ret_dirs_sensor[_return_valid]

            _virtual = self._synthesize_raydrop_virtual_rays(
                sensor_id=sensor_id,
                fov_model=datasource.lidar_models[sensor_id],
                return_dirs_sensor=_ret_dirs_sensor,
                # Same rule as the directions above: only elements that genuinely
                # returned count as occupied, so a stored no-return's element stays in
                # the complement instead of being double-counted as both a stored and a
                # synthesized negative.
                # model_elements was filtered and subsampled alongside _return_valid, so
                # the mask lines up here without further remapping.
                return_model_elements=(None if model_elements is None else np.asarray(model_elements)[_return_valid]),
                frame_start_us=int(
                    lidar_sensor.get_frame_timestamp_us(
                        sensor_frame_idx, frame_timepoint=ncore.data.FrameTimepoint.START
                    )
                ),
                frame_end_us=int(
                    lidar_sensor.get_frame_timestamp_us(sensor_frame_idx, frame_timepoint=ncore.data.FrameTimepoint.END)
                ),
                fov=self._resolve_unstructured_lidar_fov(sensor_id),
            )

            if _virtual is not None:
                _v_dirs_sensor, _v_ts, _v_elems = _virtual  # [M,3] f32, [M] i64, [M,2] i64|None
                n_virtual_rays = int(_v_dirs_sensor.shape[0])
                _virtual_dirs_sensor = _v_dirs_sensor  # kept for the tiling build (N+M)

                # virtual rays have no return: NaN range + intensity
                _distance = np.concatenate(
                    [_distance, np.full((n_virtual_rays, 1), np.nan, dtype=_distance.dtype)], axis=0
                )
                _intensity = np.concatenate(
                    [_intensity, np.full((n_virtual_rays, 1), np.nan, dtype=_intensity.dtype)], axis=0
                )
                _rays_timestamps_us = np.concatenate(
                    [_rays_timestamps_us, _v_ts.astype(_rays_timestamps_us.dtype)], axis=0
                )

                if not model_rays:
                    # append world-frame rays: rotate sensor-frame dirs by the
                    # frame-END sensor->world rotation (same pose used for returns);
                    # origin at the sensor position. Only the direction matters for
                    # the range render (re-normalized + re-rotated to sensor frame).
                    # T_sensor_world_end is guaranteed non-None on the real-ray path
                    # (model_rays=False -> _read_return_rays returns the pose); assert
                    # it so a future change to that invariant fails loudly here.
                    assert T_sensor_world_end is not None, "real-ray path must provide T_sensor_world_end"
                    _R_sw = T_sensor_world_end[:3, :3]  # sensor->world rotation
                    _origin = T_sensor_world_end[:3, 3]  # sensor position in world
                    _v_dir_world = (_v_dirs_sensor.astype(np.float64) @ _R_sw.T).astype(np.float32)
                    _v_origin = np.broadcast_to(_origin.astype(np.float32), (n_virtual_rays, 3))
                    _v_rays = np.concatenate([_v_origin, _v_dir_world], axis=-1)
                    _rays = np.concatenate([_rays, _v_rays], axis=0)

                if model_elements is not None and _v_elems is not None:
                    # model_elements mode: extend the element list so the modeled-
                    # at-elements ray generation produces the virtual rays too.
                    model_elements = np.concatenate([model_elements, _v_elems], axis=0)

        # Number of measured returns for this frame (the length of every per-return
        # array). In unstructured mode this is the (N, 1) grid size; it does not
        # depend on the sensor model, so it is derived from the returns themselves.
        n_returns = len(_distance)

        # sparse_* carry the real / at-element ray list. They are populated in
        # three of the four (lidar_ray_mode x use_real_lidar_rays) combinations; only
        # (structured, use_real=False) leaves them all None (pure modeled grid):
        #   - real rays (use_real=True): sparse_rays holds the true world rays. Serves
        #     both the structured grid-OVERRIDE (patch returning cells) and the
        #     unstructured real-ray (N, 1) list.
        #   - unstructured + use_real=False: MODELED-at-elements -- sparse_rays stays
        #     None to signal synthesis in to_rendering_data.
        # Both cases carry the return elements + per-return timestamps.
        sparse_rays = None
        sparse_timestamps = None
        sparse_elements = None
        if (not model_rays) or unstructured:
            sparse_elements = model_elements
            sparse_timestamps = _rays_timestamps_us[:, np.newaxis]
            if not model_rays:
                sparse_rays = _rays

        # Labels. In UNSTRUCTURED mode the grid is the return LIST -> (N, 1, 1); the
        # dims come from the returns, not the sensor model. In STRUCTURED mode we
        # scatter into the classic (n_rows, n_columns, 1) range-view grid.
        if unstructured:
            intensity = _intensity.reshape(n_returns, 1, 1).astype(np.float32)
            distance = _distance.reshape(n_returns, 1, 1).astype(np.float32)
            # raydrop spans the FULL (N + M) list, not just the N real returns:
            # `n_returns` is fixed AFTER the virtual tail is appended, while
            # `_return_valid` only ever describes the N real returns. Sizing this
            # from `_return_valid` would leave raydrop short of distance/intensity/
            # flags and turn the virtual-tail write below into an empty slice.
            raydrop = np.zeros((n_returns, 1, 1), dtype=np.float32)
            if use_stored_no_returns:
                raydrop[: len(_return_valid), 0, 0] = (~_return_valid).astype(np.float32)
            if n_virtual_rays > 0:
                # the appended tail are virtual (dropped) rays: raydrop GT = 1.0
                raydrop[n_returns - n_virtual_rays :, :, :] = 1.0
        else:
            assert height is not None and width is not None and model_elements is not None
            intensity = np.full((height, width, 1), fill_value=np.nan, dtype=np.float32)
            intensity[rows := model_elements[:, 0], cols := model_elements[:, 1], :] = _intensity
            distance = np.full((height, width, 1), fill_value=np.nan, dtype=np.float32)
            distance[rows, cols, :] = _distance
            raydrop = np.ones((height, width, 1), dtype=np.float32)
            # Clear the prefilled DROPPED only where the sensor actually returned. A
            # stored no-return emission occupies a grid element but is NOT a return,
            # so it must keep raydrop=1 (and its DROPPED flag below); clearing it would
            # present a NaN distance to the losses as if it were a real measurement.
            # On a dataset without stored no-returns `_return_valid` is all-True and
            # this is exactly the previous behaviour.
            raydrop[rows[_return_valid], cols[_return_valid], :] = 0.0

        # --- Unstructured LiDAR: build the frame-context tiling in the worker/CPU
        # (kept OFF the render loop) -- see _build_unstructured_tiling.
        unstructured_lidar_tiling = None

        if unstructured:
            unstructured_lidar_tiling = self._build_unstructured_tiling(
                sensor_id=sensor_id,
                lidar_sensor=lidar_sensor,
                sensor_frame_idx=sensor_frame_idx,
                sampled_pixels=sampled_pixels,
                sampled_filtered=sampled_filtered,
                stored_no_return_keep=stored_no_return_keep,
                virtual_dirs_sensor=_virtual_dirs_sensor,
                sparse_timestamps=unpack_optional(sparse_timestamps),
            )

        # save the info to the batch
        batch = DataBatch.Lidar(
            meta=[self._lidar_frame_meta(sensor_id, sensor_frame_idx)],
            labels=LidarFrameLabels(
                distance=to_torch(distance, device="cpu").unsqueeze(0),
                intensity=to_torch(intensity, device="cpu").unsqueeze(0),
                raydrop=to_torch(raydrop, device="cpu").unsqueeze(0),
                sparse_rays=map_optional(sparse_rays, lambda x: to_torch(x, device="cpu").unsqueeze(0)),
                sparse_timestamps=map_optional(
                    sparse_timestamps, lambda x: to_torch(x, device="cpu", dtype=torch.int64).unsqueeze(0)
                ),
                sparse_elements=map_optional(
                    sparse_elements, lambda x: to_torch(x, device="cpu", dtype=torch.int64).unsqueeze(0)
                ),
                ray_mode="unstructured" if unstructured else "structured",
                unstructured_lidar_tiling=unstructured_lidar_tiling,
            ),
        )

        # handling flags (for all the valid elements)
        _flags = torch.zeros((n_returns,), dtype=torch.int32, device="cpu")

        # store invalid flag of rays if per-frame mask exists (usually only required in validation mode)
        valid_mask = torch.from_numpy(
            datasource.lidars_frame_valid_points_masks[sensor_id][sensor_frame_idx].unpacked()
        )
        # Original-bundle length, so drop the same rays the per-ray arrays dropped
        # before applying the (also original-space) subsample.
        if stored_no_return_keep is not None:
            valid_mask = valid_mask[torch.from_numpy(stored_no_return_keep)]
        if isinstance(sampled_pixels, np.ndarray):
            # Set together with sampled_pixels being an index array, above.
            valid_mask = valid_mask[unpack_optional(sampled_filtered)]

        if n_virtual_rays > 0:
            # The per-frame valid-points mask covers only the N RETURN rays; the M
            # appended virtual rays are valid (not INVALID) -- they are legitimate
            # raydrop negatives (marked DROPPED below), so pad the mask with True.
            valid_mask = torch.cat([valid_mask, torch.ones(n_virtual_rays, dtype=valid_mask.dtype)], dim=0)

        _flags[~valid_mask] |= RayFlags.INVALID

        # load semantic labels, if available, and update flags
        aux_loader = datasource.aux_loader
        if aux_loader is not None:
            if aux_loader.has_lidar_semantic_segmentation():
                # load semantic segmentation image
                semantics = aux_loader.get_lidar_semantic_segmentation(
                    sensor_id, lidar_sensor.get_frame_timestamp_us(sensor_frame_idx)
                )
                # Original-bundle length: drop the same rays, then subsample.
                if stored_no_return_keep is not None:
                    semantics = semantics[stored_no_return_keep]
                if isinstance(sampled_pixels, np.ndarray):
                    semantics = semantics[unpack_optional(sampled_filtered)]

                # semantics cover only the N return rays; restrict the semantic
                # flag writes to that prefix so the M virtual rays are untouched.
                _n_sem = semantics.shape[0]
                _sem_flags = _flags[:_n_sem]
                _sem_flags |= RayFlags.VALID_SEMANTIC.value
                sky_mask = semantics == datasource.sensor_sky_class_ids[sensor_id]
                _sem_flags[sky_mask] |= RayFlags.SKY_SEMANTIC.value
                road_mask = semantics == datasource.sensor_road_class_ids[sensor_id]
                _sem_flags[road_mask] |= RayFlags.ROAD_SEMANTIC.value
                vehicle_mask = np.logical_or.reduce(
                    [semantics == class_id for class_id in datasource.sensor_vehicle_classes_ids[sensor_id]]
                )
                _sem_flags[vehicle_mask] |= RayFlags.VEHICLE_SEMANTIC.value
                _flags[:_n_sem] = _sem_flags

        # DROPPED flag setup
        # - UNSTRUCTURED: (N, 1, 1):  every listed RETURN ray is a return, so there
        #   is no DROPPED prefill; flags carry only per-return INVALID/semantic bits.
        #   The M appended VIRTUAL rays (raydrop negatives) get RayFlags.DROPPED.
        # - STRUCTURED: full (n_rows, n_columns, 1) grid prefilled with DROPPED.
        #
        # This asymmetry is why raydrop_IoU / recall are NOT comparable between the
        # paths unless the virtual-ray mode is 'model_elements': the structured grid
        # contains EVERY non-returning beam as a positive, while 'aggregate_lattice'
        # contributes only a SAMPLED subset of synthesized negatives. The unstructured
        # arm therefore under-recalls by construction (high precision, low recall)
        # without the raydrop head being any worse. See the note in
        # nurec/metrics/impl/lidar_common.py and the config field description for
        # lidar_unstructured_raydrop_virtual_rays.
        if unstructured:
            if use_stored_no_returns:
                # Stored no-return emissions occupy the first N entries; the virtual
                # tail (if any) is flagged separately below. Index explicitly rather
                # than masking through a slice view, which reads ambiguously.
                _stored_drop_idx = torch.from_numpy(np.flatnonzero(~_return_valid))
                _flags[_stored_drop_idx] |= RayFlags.DROPPED.value
            if n_virtual_rays > 0:
                # tail M rays are the synthesized virtual (dropped) rays
                _flags[n_returns - n_virtual_rays :] |= RayFlags.DROPPED.value

            flags = _flags.reshape(-1, 1, 1)

            # lidar_ignore_rows is a range-view (row) concept; re-express it via the
            # returns' element row index when a sensor model is available. In pure
            # model-free real-ray mode there are no elements / rows, so it does not
            # apply (a dataset with no sensor model cannot define ignored rows).
            if self.datasource.lidar_ignore_rows and model_elements is not None:
                ignore = np.isin(model_elements[:, 0], np.asarray(self.datasource.lidar_ignore_rows))
                flags[torch.from_numpy(ignore), 0, 0] |= RayFlags.INVALID.value
        else:
            # full resolution flags, initialized with dropped rays flags.
            assert height is not None and width is not None

            flags = torch.full(
                (height, width, 1),
                RayFlags.DROPPED.value,
                dtype=torch.int32,
                device="cpu",
            )
            # Same rule as raydrop above: only genuine returns clear the prefilled
            # DROPPED bit. Stored no-return emissions keep it, so the distance /
            # intensity losses (which mask on none(INVALID) & none(DROPPED)) never
            # see their NaN payload.
            _valid_t = torch.from_numpy(_return_valid)
            flags[rows[_return_valid], cols[_return_valid], :] = _flags[_valid_t][..., None]

            # Optionally ignore some rows of the lidar points by setting them to invalid.
            if self.datasource.lidar_ignore_rows:
                flags[self.datasource.lidar_ignore_rows, :, :] |= RayFlags.INVALID.value

        batch.labels.flags = flags.unsqueeze(0)

        _ = batch.labels.n_valid_lidar  # force computation of n_valid_lidar at data batch loading time

        return batch


class NCORESequentialDataset(NCOREDataset):
    """Encapsulates the logic for sampling frames sequentially"""

    def __init__(self, datasource: NCOREDataSource, config: NCoreDatasetConfig, split: str):
        super().__init__(datasource, config, split)
        assert config.name == "ncore"

        # store relevant parameters from config
        self.n_image_subsample: int

        camera_frame_step: int | None
        camera_exclude_frame_step: int | None
        self.camera_frames: dict[str, list[int]] = {}

        lidar_frame_step: int | None
        lidar_exclude_frame_step: int | None
        self.lidar_frames: dict[str, list[int]] = {}
        self.return_lidar: bool
        self.return_camera: bool
        self.load_rgb: bool

        if split == "val":
            self.n_image_subsample = config.n_val_image_subsample
            camera_frame_start = config.val_camera_frame_start
            camera_frame_step = config.val_camera_frame_step
            camera_exclude_frame_start = config.val_camera_exclude_frame_start
            camera_exclude_frame_step = config.val_camera_exclude_frame_step
            lidar_frame_start = config.val_lidar_frame_start
            lidar_frame_step = config.val_lidar_frame_step
            lidar_exclude_frame_start = config.val_lidar_exclude_frame_start
            lidar_exclude_frame_step = config.val_lidar_exclude_frame_step
            self.return_lidar = config.val_lidar
            self.return_camera = config.val_camera
            self.load_rgb = config.val_load_rgb
        elif split == "train-sequential":
            self.n_image_subsample = config.n_train_sequential_image_subsample
            camera_frame_start = config.train_sequential_camera_frame_start
            camera_frame_step = config.train_sequential_camera_frame_step
            camera_exclude_frame_start = config.train_sequential_camera_exclude_frame_start
            camera_exclude_frame_step = config.train_sequential_camera_exclude_frame_step
            lidar_frame_start = config.train_sequential_lidar_frame_start
            lidar_frame_step = config.train_sequential_lidar_frame_step
            lidar_exclude_frame_start = config.train_sequential_lidar_exclude_frame_start
            lidar_exclude_frame_step = config.train_sequential_lidar_exclude_frame_step
            self.return_lidar = config.train_sequential_lidar
            self.return_camera = config.train_sequential_camera
            self.load_rgb = True
        else:
            raise ValueError(f"Unexpected {split=}.")

        assert (camera_frame_step or camera_exclude_frame_step) and (
            (not camera_frame_step) or (not camera_exclude_frame_step)
        ), f"{self.__class__.__name__} Exactly one of camera_frame_step or camera_exclude_frame_step must be specified"

        if camera_frame_step is not None:
            assert camera_frame_start is not None, "camera_frame_start must be specified"
            for camera_id in self.camera_ids:
                self.camera_frames[camera_id] = list(
                    datasource.camera_frame_ranges[camera_id][camera_frame_start::camera_frame_step]
                )
        else:
            assert camera_exclude_frame_start is not None, "camera_exclude_frame_start must be specified"
            assert camera_exclude_frame_step is not None, "camera_exclude_frame_step must be specified"
            for camera_id in self.camera_ids:
                excluded_frames = set(
                    datasource.camera_frame_ranges[camera_id][camera_exclude_frame_start::camera_exclude_frame_step]
                )
                self.camera_frames[camera_id] = [
                    i for i in datasource.camera_frame_ranges[camera_id] if i not in excluded_frames
                ]

        assert (lidar_frame_step or lidar_exclude_frame_step) and (
            (not lidar_frame_step) or (not lidar_exclude_frame_step)
        ), f"{self.__class__.__name__} Exactly one of lidar_frame_step or lidar_exclude_frame_step must be specified"

        if lidar_frame_step is not None:
            assert lidar_frame_start is not None, "lidar_frame_start must be specified"
            for lidar_id in self.lidar_ids:
                self.lidar_frames[lidar_id] = list(
                    datasource.lidar_frame_ranges[lidar_id][lidar_frame_start::lidar_frame_step]
                )
        else:
            assert lidar_exclude_frame_start is not None, "lidar_exclude_frame_start must be specified"
            assert lidar_exclude_frame_step is not None, "lidar_exclude_frame_step must be specified"
            for lidar_id in self.lidar_ids:
                excluded_frames = set(
                    datasource.lidar_frame_ranges[lidar_id][lidar_exclude_frame_start::lidar_exclude_frame_step]
                )
                self.lidar_frames[lidar_id] = [
                    i for i in datasource.lidar_frame_ranges[lidar_id] if i not in excluded_frames
                ]

        # initialize optional extrinsic overwrites
        self.sensor_transl_delta_m: None | npt.NDArray = None
        self.T_camera_delta_rot: None | npt.NDArray = None
        self.T_lidar_delta_rot: None | npt.NDArray = None

        sensor_transl_delta_m_array: None | npt.NDArray = None
        sensor_rot_delta_deg_array: None | npt.NDArray = None
        self.val_sensor_transl_delta_m = config.val_sensor_transl_delta_m
        self.val_sensor_rot_delta_deg = config.val_sensor_rot_delta_deg
        if split == "val":
            if self.val_sensor_transl_delta_m is not None:
                sensor_transl_delta_m_array = np.array(self.val_sensor_transl_delta_m, dtype=np.float32)
            if self.val_sensor_rot_delta_deg is not None:
                sensor_rot_delta_deg_array = np.array(self.val_sensor_rot_delta_deg, dtype=np.float32)

        self.sensor_transl_delta_m, self.T_camera_delta_rot, self.T_lidar_delta_rot = (
            self.parse_sensor_transl_rot_delta(sensor_transl_delta_m_array, sensor_rot_delta_deg_array)
        )

    def parse_sensor_transl_rot_delta(
        self, sensor_transl_delta_m: npt.NDArray | None, sensor_rot_delta_deg: npt.NDArray | None
    ) -> tuple[npt.NDArray | None, npt.NDArray | None, npt.NDArray | None]:
        parsed_sensor_transl_delta_m: npt.NDArray | None = None
        parsed_T_camera_delta_rot: npt.NDArray | None = None
        parsed_T_lidar_delta_rot: npt.NDArray | None = None
        if sensor_transl_delta_m is not None:
            assert sensor_transl_delta_m.shape == (3,), (
                f"{self.__class__.__name__}: Expecting 3d translation offset vector in 'val_sensor_transl_delta_m'"
            )
            parsed_sensor_transl_delta_m = np.array(sensor_transl_delta_m, dtype=np.float32)
        if sensor_rot_delta_deg is not None:
            sensor_rot_delta_deg_array = np.array(sensor_rot_delta_deg, dtype=np.float32)
            assert sensor_rot_delta_deg_array.shape == (3,), (
                f"{self.__class__.__name__}: Expecting three rotation Euler angles vector in 'val_sensor_rot_delta_deg'"
            )
            # precompute validation extrinsic delta rotation once (these are specified relative to the
            # rig frame so that they are always consistent for differently posed cameras)
            parsed_T_camera_delta_rot = (
                self.datasource.T_CAMERA_RIG.transpose()
                @ (T_sensor_delta_rot := euler_2_so3(sensor_rot_delta_deg_array, degrees=True, seq="xyz"))
                @ self.datasource.T_CAMERA_RIG
            )
            parsed_T_lidar_delta_rot = T_sensor_delta_rot
        return parsed_sensor_transl_delta_m, parsed_T_camera_delta_rot, parsed_T_lidar_delta_rot

    def get_max_num_rays_per_train_sample(self) -> int:
        raise ValueError(
            f"{self.__class__.__name__}: get_max_num_rays_per_train_sample should only be called during training"
        )

    def __len__(self) -> int:
        """Returns the total number of samples provided by the dataset (depending on parametrization)"""

        self.datasource._maybe_init_worker()

        # sum of all per-camera / per-lidar frame-ranges
        size = 0
        if self.return_camera:
            size += sum(len(self.camera_frames[camera_id]) for camera_id in self.camera_ids)
        if self.return_lidar:
            size += sum(len(self.lidar_frames[lidar_id]) for lidar_id in self.lidar_ids)

        return size

    def get_item_novel_view_overrides(
        self, tuple_idx: tuple[int, Optional[NovelViewOverrides]]
    ) -> DataAndRenderingBatch:
        """
        Returns a specific sample of the dataset (depending on split type and parametrization)

        Args:
            tuple_idx: A tuple of (batch_idx, novel_view_overrides).

        Note that we use a single argument as this is supposed to be used with a pytorch DataLoader/Sampler.
        """

        self.datasource._maybe_init_worker()
        sequence_id = self.datasource.sequence_id

        batch_idx, novel_view_overrides = tuple_idx

        # parse optional extrinsic overwrites
        if novel_view_overrides is not None:
            transl_delta_m, T_camera_delta_rot, T_lidar_delta_rot = self.parse_sensor_transl_rot_delta(
                novel_view_overrides.transl_delta_m, novel_view_overrides.rot_delta_deg
            )
        else:
            transl_delta_m = self.sensor_transl_delta_m
            T_camera_delta_rot = self.T_camera_delta_rot
            T_lidar_delta_rot = self.T_lidar_delta_rot

        run_frames = 0
        if self.return_camera:
            for camera_id in self.camera_ids:
                n_camera_frames = len(self.camera_frames[camera_id])
                if batch_idx >= run_frames + n_camera_frames:
                    # current camera depleted, check next one
                    run_frames += n_camera_frames
                    continue

                # determine frame of current camera
                camera_frame_index = self.camera_frames[camera_id][batch_idx - run_frames]

                camera_model = self.datasource.camera_models[camera_id]
                camera_width, camera_height = (
                    int(camera_model.resolution[0].item()),
                    int(camera_model.resolution[1].item()),
                )

                T_offset = np.eye(4, dtype=np.float32)
                # apply optional rotation deltas (specified relative to rig frame)
                if T_camera_delta_rot is not None:
                    T_offset[:3, :3] = T_camera_delta_rot

                # translation deltas relative to the rig are simple linear offsets
                if transl_delta_m is not None:
                    T_offset[:3, 3] = transl_delta_m

                T_rig_nurec = self.datasource.world_to_nurec.transform_poses(
                    self.datasource.camera_sensors[camera_id].get_frames_T_source_target(
                        source_node="rig",
                        target_node="world",
                        frame_indices=camera_frame_index,
                        frame_timepoint=None,  # evaluates both start/end timepoints
                    )  # 2x4x4
                )

                T_offset_nurec_startend = np.stack(
                    [
                        T_rig_nurec[0] @ T_offset @ np.linalg.inv(T_rig_nurec[0]),
                        T_rig_nurec[1] @ T_offset @ np.linalg.inv(T_rig_nurec[1]),
                    ]
                )
                del T_rig_nurec, T_offset

                data_camera_batch = self.get_camera_data_batch(
                    camera_id,
                    camera_frame_index,
                    RectSubsampled(
                        i=0,
                        j=0,
                        height=camera_height // self.n_image_subsample,
                        width=camera_width // self.n_image_subsample,
                        subsample_factor=float(self.n_image_subsample),
                        original_width=camera_width,
                        original_height=camera_height,
                    ),
                    load_rgb=self.load_rgb,
                )

                if T_camera_delta_rot is not None or transl_delta_m is not None:
                    # Pose will be modified, so set all labels to INVALID flag
                    if data_camera_batch.labels.flags is not None:
                        data_camera_batch.labels.flags |= RayFlags.INVALID.value

                # Add the extrinsic override information to the frame meta
                # [TODO]: Move the computation of the extrinsic overrides to torch once v1 is retired
                data_camera_batch.meta[0].T_offset_nurec_startend = to_torch(
                    T_offset_nurec_startend, device="cpu", dtype=torch.float32
                )

                data_batch = DataBatch(
                    idx=batch_idx,
                    worker_id=None,
                    sequence_id=[sequence_id],
                    lidar=None,
                    camera=data_camera_batch,
                )

                # Only includes the DataBatch as rendering batch will be initialized in the calibration module on GPU
                return DataAndRenderingBatch(data=data_batch, rendering=None)

        if self.return_lidar != "off":
            for lidar_id in self.lidar_ids:
                n_lidar_frames = len(self.lidar_frames[lidar_id])
                if batch_idx >= run_frames + n_lidar_frames:
                    # current lidar depleted, check next one
                    run_frames += n_lidar_frames
                    continue

                # determine frame of current camera
                lidar_frame_index = self.lidar_frames[lidar_id][batch_idx - run_frames]

                data_lidar_batch = self.get_lidar_data_batch(
                    sensor_id=lidar_id,
                    sensor_frame_idx=lidar_frame_index,
                )

                if T_lidar_delta_rot is not None or transl_delta_m is not None:
                    # Pose will be modified, so set all labels to INVALID flag
                    if data_lidar_batch.labels.flags is not None:
                        data_lidar_batch.labels.flags |= RayFlags.INVALID.value

                T_offset = np.eye(4, dtype=np.float32)
                # apply optional rotation deltas (specified relative to rig frame)
                if T_lidar_delta_rot is not None:
                    T_offset[:3, :3] = T_lidar_delta_rot

                # translation deltas relative to the rig are simple linear offsets
                if transl_delta_m is not None:
                    T_offset[:3, 3] = transl_delta_m

                T_rig_nurec = self.datasource.world_to_nurec.transform_poses(
                    self.datasource.lidar_sensors[lidar_id].get_frames_T_source_target(
                        source_node="rig",
                        target_node="world",
                        frame_indices=lidar_frame_index,
                        frame_timepoint=None,  # evaluates both start/end timepoints
                    )  # 2x4x4
                )

                T_offset_nurec_startend = np.stack(
                    [
                        T_rig_nurec[0] @ T_offset @ np.linalg.inv(T_rig_nurec[0]),
                        T_rig_nurec[1] @ T_offset @ np.linalg.inv(T_rig_nurec[1]),
                    ]
                )
                del T_rig_nurec, T_offset

                data_batch = DataBatch(
                    idx=batch_idx,
                    worker_id=None,
                    sequence_id=[sequence_id],
                    lidar=data_lidar_batch,
                    camera=None,
                )

                data_lidar_batch.meta[0].T_offset_nurec_startend = to_torch(
                    T_offset_nurec_startend, device="cpu", dtype=torch.float32
                )

                return DataAndRenderingBatch(data=data_batch, rendering=None)

        raise IndexError(f"Out of range sequential sample {batch_idx}")

    @overload
    def __getitem__(self, idx: int) -> DataAndRenderingBatch: ...

    @overload
    def __getitem__(self, idx: tuple[int, Optional[NovelViewOverrides]]) -> DataAndRenderingBatch: ...

    def __getitem__(self, idx: tuple[int, Optional[NovelViewOverrides]] | int) -> DataAndRenderingBatch:
        """
        Returns a specific sample of the dataset (depending on split type and parametrization)

        Args:
            idx: In the normal case this is the batch_idx [int].
                 Also supports novel view sampling when used in combination with a sampler that supports it.
                 Then this is a tuple of (batch_idx, novel_view_overrides) [tuple[int, NovelViewOverrides]].
        """
        if isinstance(idx, int):
            batch = self.get_item_novel_view_overrides((idx, None))
            batch_idx = batch.data.idx if isinstance(batch, DataAndRenderingBatch) else batch.idx
            assert batch_idx == idx
        else:
            batch = self.get_item_novel_view_overrides(idx)
            batch_idx = batch.data.idx if isinstance(batch, DataAndRenderingBatch) else batch.idx
            assert batch_idx == idx[0]

        return batch


def compute_effective_samples_per_epoch(
    *,
    configured_n_samples_per_epoch: int,
    eligible_pairs: int,
    total_pairs: int,
    scale_samples_per_epoch_by_frame_selection_fraction: bool,
) -> int:
    """Return the runtime epoch length for an eligible-frame coverage restriction.

    Scaling preserves the configured samples-per-reachable-frame budget, rounded up so a non-empty
    eligible set never loses a fractional final step. Disabling scaling, or retaining every reachable
    pair, preserves the configured epoch length exactly.
    """
    if total_pairs <= 0:
        raise ValueError(f"total_pairs must be positive, got {total_pairs}.")
    if eligible_pairs <= 0 or eligible_pairs > total_pairs:
        raise ValueError(f"eligible_pairs must be in [1, total_pairs], got {eligible_pairs}/{total_pairs}.")
    if not scale_samples_per_epoch_by_frame_selection_fraction or eligible_pairs == total_pairs:
        return configured_n_samples_per_epoch
    return math.ceil(configured_n_samples_per_epoch * eligible_pairs / total_pairs)


class NCORETrainDataset(NCOREDataset):
    """Encapsulates the logic for sampling frames during training"""

    def __init__(
        self,
        datasource: NCOREDataSource,
        config: NCoreDatasetConfig,
        split: str,
    ):
        super().__init__(datasource, config, split)

        # store relevant parameters from config
        self.configured_n_samples_per_epoch: int = config.n_samples_per_epoch
        self.effective_n_samples_per_epoch: int = self.configured_n_samples_per_epoch
        self.n_train_sample_camera_rays: int = config.n_train_sample_camera_rays
        self.n_train_sample_lidar_rays: int = config.n_train_sample_lidar_rays

        self._worker_id: Optional[int] = None
        self._rng: Optional[np.random.Generator] = None

        self.batch_samplers: list[BaseBatchSampler] = []

        for batch_sampler_config in config.samplers.values():
            self.batch_samplers.append(
                BaseBatchSampler.batch_sampler_factory(batch_sampler_config.name, batch_sampler_config, dataset=self)
            )

    @property
    def rng(self) -> np.random.Generator:
        self._maybe_reinit_rng()
        return unpack_optional(self._rng)

    def _maybe_reinit_rng(self) -> None:
        """
        Sets the rng according to which worker is executing.
        """
        # Determine current worker / process and whether re-initialization is necessary
        match torch.utils.data.get_worker_info():
            case None:
                # main process case
                if self._rng and self._worker_id is None:
                    # skip re-initialization
                    return
                # in case we move back from a worker to the main process we should also re-init
                self._worker_id = None
                seed = 0  # deterministic sampling

            case torch.utils.data._utils.worker.WorkerInfo(id=worker_id, seed=worker_seed):  # type:ignore
                # worker process case
                if self._rng and self._worker_id is worker_id:
                    # skip re-initialization
                    return
                self._worker_id = worker_id
                seed = worker_seed  # non-deterministic sampling

        seed = compute_process_local_rng_seed(seed)

        self._rng = np.random.default_rng(seed=seed)

    @ScopedTimer("NCORETrainDataset.__getitem__", TimingTag.DATALOADER)
    def __getitem__(self, batch_idx: int) -> DataAndRenderingBatch:
        # make sure worker is initialized
        self.datasource._maybe_init_worker()
        with ScopedTimer("NCORETrainDataset.__getitem__/get_batch", TimingTag.DATALOADER):
            batches = [batch_sampler.get_batch(batch_idx, self) for batch_sampler in self.batch_samplers]
        assert len(batches) == 1, "NCOREDataset: multiple batches per sample is not supported for now"
        return batches[0]

    def __len__(self) -> int:
        """Returns the total number of samples provided by the dataset (depending on split type and parametrization)"""

        return self.effective_n_samples_per_epoch

    @property
    def n_samples_per_epoch(self) -> int:
        """Backward-compatible alias for the effective runtime epoch length."""
        return self.effective_n_samples_per_epoch

    def apply_samples_per_epoch_scaling(
        self,
        *,
        eligible_pairs: int,
        total_pairs: int,
        scale_samples_per_epoch_by_frame_selection_fraction: bool,
    ) -> int:
        """Set and return the runtime epoch length for an eligible-frame coverage restriction."""
        self.effective_n_samples_per_epoch = compute_effective_samples_per_epoch(
            configured_n_samples_per_epoch=self.configured_n_samples_per_epoch,
            eligible_pairs=eligible_pairs,
            total_pairs=total_pairs,
            scale_samples_per_epoch_by_frame_selection_fraction=scale_samples_per_epoch_by_frame_selection_fraction,
        )
        return self.effective_n_samples_per_epoch

    def get_active_tile_mapping_prewarm_plan(self) -> ActiveTileMappingPrewarmPlan:
        """Describe deterministic inputs for the one-time camera-to-LiDAR mapping prewarm.

        This method does not construct a tile mapping. It builds one metadata-only camera/LiDAR
        pair per training camera. The prewarmer calibrates each pair and passes the resulting
        geometry through the same mapping builder and cache used by the on-demand (lazy) render
        path. The prewarmer changes the construction time, chooses these deterministic frames, and
        isolates a deterministic mapping seed; it does not introduce a second mapping algorithm.

        The plan is built as follows:

        1. Admit only the production geometry that can be reproduced without sampling a real
           training batch: one modeled LiDAR, one configured batch sampler, both camera and LiDAR
           training enabled, and at least one training camera. Unsupported cases return a stable
           ``skip_reason`` and retain lazy initialization.
        2. Enumerate the configured LiDAR's frames, remove holdout/non-training frames using the
           datasource's all-frame-to-training-frame mapping, and select the middle eligible
           *position*. This is a deterministic representative anchor, not a median-timestamp
           calculation. Centrality itself does not reduce ego motion; the per-camera match in step
           4 does that.
        3. Visit cameras in unique-sensor order so every rank constructs and inserts cache entries
           in the same order.
        4. For each camera, remove holdout/non-training frames and select the eligible frame whose
           START timestamp minimizes the absolute difference from the anchor LiDAR START timestamp.
           The mapping builder projects the LiDAR sweep's world-space rays through the camera START
           pose. Comparing unrelated frames would therefore fold the rig motion between their
           capture times into a cache keyed only by camera sensor. Nearest-START matching minimizes
           temporal separation and therefore the expected inter-frame ego motion. The generated
           world-space LiDAR rays already encode motion during the anchor sweep; finite bounded
           pose-jitter samples approximate the remaining relative-pose variation rather than
           guaranteeing complete coverage.
        5. Ask the configured camera pixel sampler for its metadata-only fixed geometry. ``None``
           means the sampler cannot promise the same crop/subsample as normal training, so the
           entire plan falls back to lazy initialization instead of prewarming different geometry.
        6. Build a one-camera/one-LiDAR ``DataBatch`` from ``FrameMeta`` plus empty label containers.
           No target tensors are loaded: camera poses/rays and modeled LiDAR poses/rays come from
           frame metadata, rig trajectories, and sensor models; RGB, validity, distance, and
           intensity targets are not consumed by calibration or by the mapping builder. The default
           structured ``LidarFrameLabels.ray_mode`` selects modeled grid-ray generation. Real LiDAR
           rays are rejected in step 1 because those rays would have to come from labels.
        7. Return the entries in a tuple-backed outer plan with fixed construction order. Every
           entry shares the same LiDAR anchor, while its camera frame and fixed sampling geometry
           are camera-specific.
        """
        # Step 1: restrict prewarm to a shape that this metadata-only plan can reproduce exactly.
        # A different or ambiguous shape keeps the cache empty so the first applicable render uses
        # the existing lazy construction path.
        if len(self.lidar_ids) != 1:
            return ActiveTileMappingPrewarmPlan(skip_reason="requires_one_training_lidar")
        lidar_id = self.lidar_ids[0]
        # Measured LiDAR rays live in LidarFrameLabels. An empty-label plan can reconstruct only the
        # structured rays generated from the configured LiDAR sensor model.
        if bool(resolve_per_sensor(self.config.use_real_lidar_rays, lidar_id)):
            return ActiveTileMappingPrewarmPlan(skip_reason="requires_modeled_lidar_rays")
        # Multiple batch samplers may describe different camera crops or modality mixtures; there
        # would be no single geometry for a precomputed per-camera cache entry to represent.
        if len(self.batch_samplers) != 1:
            return ActiveTileMappingPrewarmPlan(skip_reason="requires_one_batch_sampler")

        batch_sampler = self.batch_samplers[0]
        # The camera-to-LiDAR overlap mapping is useful only when the normal training batch contains
        # both modalities. Avoid manufacturing a modality that training itself does not request.
        if batch_sampler.n_train_sample_camera_rays <= 0 or batch_sampler.n_train_sample_lidar_rays <= 0:
            return ActiveTileMappingPrewarmPlan(skip_reason="requires_camera_and_lidar_training")

        if not self.camera_ids:
            return ActiveTileMappingPrewarmPlan(skip_reason="requires_training_cameras")

        # Step 2: choose one deterministic, eligible LiDAR frame as the common temporal anchor.
        camera_pixel_sampler = batch_sampler.camera_pixel_sampler
        datasource = self.datasource
        lidar_frame_indices = datasource.get_training_lidar_sensor_frame_indices(lidar_id)
        if lidar_frame_indices.size == 0:
            return ActiveTileMappingPrewarmPlan(skip_reason="requires_training_lidar_frames")

        # Select the frame at the middle position of the eligible datasource-ordered
        # sequence. This provides a deterministic, non-edge anchor; timestamps are
        # used only afterward to match each camera to this LiDAR frame.
        lidar_frame_idx = int(lidar_frame_indices[lidar_frame_indices.size // 2])
        lidar_start_timestamp_us = int(
            datasource.lidar_sensors[lidar_id].frames_timestamps_us[lidar_frame_idx, ncore.data.FrameTimepoint.START]
        )

        # Step 3: use a rank-invariant order for per-camera cache construction. Sensor index matches
        # the cache's identity, and textual id supplies a deterministic secondary sort key.
        entries: list[ActiveTileMappingPrewarmEntry] = []
        sorted_camera_ids = sorted(
            self.camera_ids,
            key=lambda camera_id: (datasource.camera_unique_ids[camera_id].idx, camera_id),
        )
        for camera_id in sorted_camera_ids:
            # Step 4a: ask the datasource for this camera's eligible training frames.
            camera_frame_indices = datasource.get_training_camera_sensor_frame_indices(camera_id)
            if camera_frame_indices.size == 0:
                return ActiveTileMappingPrewarmPlan(skip_reason=f"camera_without_training_frames:{camera_id}")

            # Step 4b: match the camera START closest to the LiDAR START. The renderer's mapping
            # builder uses the camera START pose, so this minimizes temporal separation and therefore
            # the expected unrelated inter-frame rig motion embedded in the cached relationship.
            # np.argmin deterministically selects the first eligible frame if two are equally close.
            camera_start_timestamps_us = np.asarray(
                datasource.camera_sensors[camera_id].frames_timestamps_us[
                    camera_frame_indices, ncore.data.FrameTimepoint.START
                ],
                dtype=np.int64,
            )
            nearest_training_position = int(np.abs(camera_start_timestamps_us - lidar_start_timestamp_us).argmin())
            camera_frame_idx = int(camera_frame_indices[nearest_training_position])

            # Step 5: reproduce the normal sampler's fixed full-image crop/subsample from metadata
            # alone. A dynamic or unsupported sampler returns None and leaves construction lazy.
            original_width, original_height = datasource.camera_model_parameters[camera_id].resolution
            original_width, original_height = int(original_width), int(original_height)
            camera_subsample = camera_pixel_sampler.fixed_sampled_pixels(
                original_width=original_width,
                original_height=original_height,
            )
            if camera_subsample is None:
                return ActiveTileMappingPrewarmPlan(skip_reason="requires_fixed_full_image_sampling")

            # Step 6: FrameMeta identifies the sensor/frame pose and timestamp and carries the exact
            # sampled image geometry. Empty label containers load no target tensors; their default
            # structured LiDAR mode still selects modeled grid-ray generation during calibration.
            camera_meta = self._camera_frame_meta(
                camera_id,
                camera_frame_idx,
                sampled_pixels=camera_subsample,
            )

            data_batch = DataBatch(
                idx=None,
                worker_id=None,
                sequence_id=[datasource.sequence_id],
                camera=DataBatch.Camera(
                    meta=[camera_meta],
                    labels=CameraFrameLabels(),
                ),
                lidar=DataBatch.Lidar(
                    meta=[self._lidar_frame_meta(lidar_id, lidar_frame_idx)],
                    labels=LidarFrameLabels(),
                ),
            )
            entries.append(
                ActiveTileMappingPrewarmEntry(
                    camera_sensor_idx=camera_meta.unique_sensor_idx,
                    data_batch=data_batch,
                )
            )

        # Step 7: preserve the deterministic, one-entry-per-camera construction order in a tuple.
        return ActiveTileMappingPrewarmPlan(entries=tuple(entries))

    def update_epoch(self, epoch: int, system, **kwargs) -> None:
        for batch_sampler in self.batch_samplers:
            batch_sampler.update_epoch(epoch, system, **kwargs)

    def get_max_num_rays_per_train_sample(self):
        return sum([batch_sampler.get_max_rays_num() for batch_sampler in self.batch_samplers])

    @ScopedTimer("NCORETrainDataset.get_train_batch", TimingTag.DATALOADER)
    def get_train_batch(
        self,
        batch_idx: int,
        n_train_sample_camera_rays: int,
        camera_sensor_sampler: BaseSensorSampler,
        camera_frame_sampler: BaseFrameSampler,
        camera_pixel_sampler: BaseCameraPixelSampler,
        n_train_sample_lidar_rays: int,
        lidar_sensor_sampler: BaseSensorSampler,
        lidar_frame_sampler: BaseFrameSampler,
        lidar_point_sampler: BaseLidarPointSampler,
    ) -> DataAndRenderingBatch:
        """
        This method returns a batch from the training split, applying random sampling internally.
        Lidar / camera samplers need to be provided if requesting sensor-specific rays.
        """

        # make sure worker is initialized
        self.datasource._maybe_init_worker()

        # randomly sample chunk according to per-chunk probability ~ induces underlying sequence to sample from
        sequence_id = self.datasource.sequence_id

        data_camera_batch: DataBatch.Camera | None = None
        if n_train_sample_camera_rays:
            assert not isinstance(camera_sensor_sampler, SkipSensorSampler), (
                f"{self.__class__.__name__}: require camera sensor sampler if requesting camera rays in training batch"
            )
            assert not isinstance(camera_frame_sampler, SkipFrameSampler), (
                f"{self.__class__.__name__}: require camera frame sampler if requesting camera rays in training batch"
            )
            assert not isinstance(camera_pixel_sampler, SkipCameraPixelSampler), (
                f"{self.__class__.__name__}: require camera pixel sampler if requesting camera rays in training batch"
            )

            # sample a camera sensor using the SensorSampler
            sampled_camera_ids: list[str] = camera_sensor_sampler.sample_sensor(
                self.rng, batch_idx, self.camera_ids
            ).sampled_sensor_ids
            assert len(sampled_camera_ids) == 1, (
                "NCOREDataset: multiple camera sensors per batch is not supported for now"
            )
            camera_id = sampled_camera_ids[0]
            camera_frame_range = self.datasource.camera_frame_ranges[camera_id]

            # sample camera frame using FrameSampler
            camera_frame_idx = camera_frame_sampler.sample_frame(
                self.rng,
                batch_idx,
                frame_range=camera_frame_range,
                unique_sensor_id=(unique_camera_id := self.datasource.camera_unique_ids[camera_id].id),
            ).sampled_frame_idx

            # determine valid pixels of frame
            frame_valid_pixels_mask = self.datasource.cameras_frame_valid_pixels_masks[camera_id][
                camera_frame_idx
            ].unpacked()

            # sample pixels of frame using CameraPixelSampler
            pixel_samples = camera_pixel_sampler.sample_camera_pixels(
                rng=self.rng,
                batch_idx=batch_idx,
                frame_all_pixels=self.datasource.cameras_all_pixels[camera_id],
                frame_valid_pixels_mask=frame_valid_pixels_mask,
                n_frame_pixel_samples=n_train_sample_camera_rays,
                unique_camera_id=unique_camera_id,
                camera_frame_idx=camera_frame_idx,
                frame_range=camera_frame_range,
            )

            assert isinstance(pixel_samples.sampled_pixels, RectSubsampled)
            data_camera_batch = self.get_camera_data_batch(
                sensor_id=camera_id,
                sensor_frame_idx=camera_frame_idx,
                sampled_pixels=pixel_samples.sampled_pixels,
            )

        data_lidar_batch: DataBatch.Lidar | None = None
        if n_train_sample_lidar_rays:
            assert not isinstance(lidar_sensor_sampler, SkipSensorSampler), (
                f"{self.__class__.__name__}: require lidar sensor sampler if requesting lidar rays in training batch"
            )
            assert not isinstance(lidar_frame_sampler, SkipFrameSampler), (
                f"{self.__class__.__name__}: require lidar frame sampler if requesting lidar rays in training batch"
            )
            assert not isinstance(lidar_frame_sampler, SkipLidarPointSampler), (
                f"{self.__class__.__name__}: require lidar point sampler if requesting lidar rays in training batch"
            )

            # sample a lidar sensor using the SensorSampler
            sampled_lidar_ids = lidar_sensor_sampler.sample_sensor(
                self.rng, batch_idx, self.lidar_ids
            ).sampled_sensor_ids
            assert len(sampled_lidar_ids) == 1, (
                "NCOREDataset: multiple lidar sensors per batch is not supported for now"
            )
            lidar_id = sampled_lidar_ids[0]
            lidar_frame_range = self.datasource.lidar_frame_ranges[lidar_id]

            # sample a lidar frame using FrameSampler
            lidar_frame_idx = lidar_frame_sampler.sample_frame(
                self.rng,
                batch_idx,
                lidar_frame_range,
                unique_sensor_id=(unique_lidar_id := self.datasource.lidar_unique_ids[lidar_id].id),
            ).sampled_frame_idx

            # determine valid points of frame
            valid_points_mask = self.datasource.lidars_frame_valid_points_masks[lidar_id][lidar_frame_idx].unpacked()
            # sample points of frame using LidarPointSampler
            point_samples = lidar_point_sampler.sample_lidar_points(
                self.rng,
                batch_idx,
                lidar_frame_range,
                n_train_sample_lidar_rays,
                valid_points_mask,
                unique_lidar_id,
                lidar_frame_idx,
            )

            data_lidar_batch = self.get_lidar_data_batch(
                sensor_id=lidar_id,
                sensor_frame_idx=lidar_frame_idx,
                sampled_pixels=point_samples.sampled_point_idxs,
            )

        data_batch = DataBatch(
            idx=batch_idx,
            worker_id=[self._worker_id] if self._worker_id is not None else None,
            sequence_id=[sequence_id],
            lidar=data_lidar_batch,
            camera=data_camera_batch,
        )

        return DataAndRenderingBatch(data=data_batch, rendering=None)


class NCOREDataSource(BaseDataSource, RigTrajectoriesProvider):
    UNCONDITIONALLY_DYNAMIC_CLASSES: set[str] = set(
        [
            "pedestrian",
            "stroller",
            "person",
            "person_group",
            "rider",
            "bicycle_with_rider",
            "bicycle",
            "CYCLIST",
            "motorcycle",
            "motorcycle_with_rider",
            "cycle",
        ]
    )

    # Frame convention conversion from camera frame convention (x right, y down, z forward)
    # to rig frame convention (x forward, y left, z up)
    T_CAMERA_RIG = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float32)

    @unique
    class ValidMeasurementsMethod(IntEnum):
        """Different method variants to determine valid pixels/lidarpoints"""

        EGO = auto()  # ego camera masks only
        EGO_CUBOIDTRACKS = (
            auto()
        )  # ego camera masks and frame-projected dynamic objects from cuboid tracks; dynamic_flag lidar points
        EGO_SCENEFLOW = auto()  # ego camera masks and scene flow masks
        EGO_CUBOIDTRACKS_SCENEFLOW = auto()  # ego camera masks, frame-projected dynamic objects from cuboid tracks, and scene flow masks; dynamic_flag lidar points
        EGO_CUBOIDTRACKS_TRAFFICLIGHT = auto()  # ego camera masks, frame-projected dynamic objects from cuboid tracks, and traffic light masks; dynamic_flag, non-traffic light, camera-visible lidar points
        EGO_FRAMEMASKS = auto()  # ego camera masks and per-frame camera masks

    def __init__(self, config: NCoreDatasetConfig, lidar_model_config: Optional[LidarModelsConfig]) -> None:
        """This method loads **metadata** and defers loading actual data to _load_data"""
        super().__init__()

        # Init parameters from config
        self.camera_mask_sources: List[Literal["dataset", "aux"]] = config.camera_mask_sources
        self.n_camera_mask_dilation_iterations: int = config.n_camera_mask_dilation_iterations
        self.camera_mask_overrides: Dict[str, str] = config.camera_mask_overrides
        self.camera_mask_border: Optional[Dict[str, list[int]]] = config.camera_mask_border

        self.camera_max_fov_deg: float = config.camera_max_fov_deg
        self.lidar_model_parameter_cwccw_fallback: bool = config.lidar_model_parameter_cwccw_fallback
        self.lidar_model_parameter_nominal_values: PerSensor[LidarNominalModel] = (
            config.lidar_model_parameter_nominal_values
        )
        self.cuboid_tracks_params = config.cuboid_tracks_params
        self.lidar_dynamic_points_params = config.lidar_dynamic_points

        # load valid measurements (pixels/lidarpoints) method and method-dependent parameters
        self.valid_measurements_method = NCOREDataSource.ValidMeasurementsMethod[config.valid_measurements_method]
        self.valid_pixels_cuboid_tracks_params = config.valid_pixels_cuboid_track_params
        self.valid_pixels_pixels_scene_flow_params = config.valid_pixels_scene_flow_params
        self.valid_pixels_traffic_light_params = config.valid_pixels_traffic_light_params
        self.valid_pixels_frame_mask_params = config.valid_pixels_frame_mask_params
        self.valid_points_cuboid_track_params = config.valid_lidarpoints_cuboid_track_params

        # V3 sequence loader parameters
        self.cuboid_loading_max_workers: Optional[int] = config.cuboid_loading_max_workers

        # V4 sequence loader parameters
        self.poses_component_group: str = config.poses_component_group
        self.intrinsics_component_group: str = config.intrinsics_component_group
        self.masks_component_group: str = config.masks_component_group
        self.cuboids_component_group: str = config.cuboids_component_group

        self.camera_ids: list[str] = config.camera_ids
        self.lidar_ids: list[str] = config.lidar_ids
        self.train_camera_ids: list[str] = config.train_camera_ids
        self.train_lidar_ids: list[str] = config.train_lidar_ids
        self.lidar_ignore_rows: list[int] = config.lidar_ignore_rows
        self.val_camera_ids: list[str] = (
            config.val_camera_ids
        )  # validation cameras are the ones not in train_camera_ids
        self.val_lidar_ids: list[str] = config.val_lidar_ids

        self.all_camera_ids = sorted(set(self.camera_ids + self.train_camera_ids))
        self.all_lidar_ids = sorted(set(self.lidar_ids + self.train_lidar_ids))
        self.camera_point_cloud_ignore_classes = set(config.camera_point_cloud_ignore_classes)
        self.camera_point_cloud_dynamic_classes = set(config.camera_point_cloud_dynamic_classes)

        self.lidar_model_config: Optional[LidarModelsConfig] = lidar_model_config

        # Maximum distance away from the ego pose that will still be covered by the bounding box and the feature volume
        self.max_dist_m: float = config.max_dist_m

        self.open_consolidated: bool = config.open_consolidated

        self.jpeg_backend_cpu: Literal["PIL", "simplejpeg"] = config.jpeg_backend_cpu
        self.simplejpeg_fastdct: bool = config.simplejpeg_fastdct
        self.simplejpeg_fastupsample: bool = config.simplejpeg_fastupsample

        self.frame_generic_data_pose_overwrite: bool = config.frame_generic_data_pose_overwrite

        self.aux_data: bool = config.aux_data

        self.tqdm_disabled = not config.show_progress_bars

        self.path = UPath(config.path)
        self.resolved_dataset = resolve_dataset(
            DatasetRequest(
                metadata_path=self.path,
                open_consolidated=self.open_consolidated,
                loader_max_workers=self.cuboid_loading_max_workers,
                poses_component_group=self.poses_component_group,
                intrinsics_component_group=self.intrinsics_component_group,
                masks_component_group=self.masks_component_group,
                cuboids_component_group=self.cuboids_component_group,
            )
        )
        self.sequence_id = self.resolved_dataset.metadata.sequence_id
        self.time_range_us = HalfClosedInterval(
            self.resolved_dataset.metadata.time_range_start_us,
            self.resolved_dataset.metadata.time_range_end_us,
        )
        self.dataset_paths = list(self.resolved_dataset.metadata.dataset_paths)

        def update_time_range_bounds(time_range_us: HalfClosedInterval) -> HalfClosedInterval:
            """Restricts time range according to config seek_offset_sec and duration_sec"""

            SEC_TO_USEC = 1_000_000
            if seek_offset_sec := config.seek_offset_sec:
                time_range_us.start += int(seek_offset_sec * SEC_TO_USEC)
                if time_range_us.end <= time_range_us.start:
                    raise AssertionError(f"{config.seek_offset_sec=} leads to an empty interval.")

            if duration_sec := config.duration_sec:
                time_range_us.end = min(time_range_us.start + int(duration_sec * SEC_TO_USEC), time_range_us.end)
                if time_range_us.end <= time_range_us.start:
                    raise AssertionError(f"{config.duration_sec=} leads to an empty interval.")

            return time_range_us

        self.time_range_us = update_time_range_bounds(self.time_range_us)

        # sensor-id to range of sensor frame indices
        self.camera_frame_ranges: dict[str, range] = {}
        self.lidar_frame_ranges: dict[str, range] = {}
        # sensor-id to global linear start frame index (across all cameras/lidars)
        self.camera_linear_start_frame_indices: dict[str, int] = {}
        self.lidar_linear_start_frame_indices: dict[str, int] = {}

        # initialize shards once fully
        self.sequence_loader: Optional[ncore.data.SequenceLoaderProtocol] = None
        self.worker_id: Optional[int] = None

        # initialize params to generate traffic light cuboid tracks
        self.generate_static_rigid = config.generate_static_rigid_cuboid_tracks.enabled
        self.visibility_check_camera_ids = config.generate_static_rigid_cuboid_tracks.visibility_check_camera_ids
        self.max_visible_distance = config.generate_static_rigid_cuboid_tracks.max_distance_m
        self.generated_static_rigid_cuboidtracks: Optional[CuboidTracks] = None
        self.static_rigid_classes = config.generate_static_rigid_cuboid_tracks.rigid_classes
        self.static_rigid_point_cloud_path = config.generate_static_rigid_cuboid_tracks.point_cloud_path
        # make sure worker is initialized
        self._maybe_init_worker()

    def get_path(self) -> UPath:
        """
        The path to the .json file defining this datasource.
        """
        return self.path

    def get_offset(self) -> npt.NDArray[np.float32]:
        nurec_to_world = self.world_to_nurec.inverse()
        return nurec_to_world.target_origin

    def _maybe_init_worker(self) -> None:
        """
        Detects if worker ID has changed (indicating multiprocessing fork) and either
            * returns early if no changes
            * performs full data initialization if this is the first use of this instance
            * recreates file descriptors to avoid accidental sharing if fork has happened
        """

        # Determine current worker / process and whether re-initialization is necessary.
        # Terminate if not necessary.
        match torch.utils.data.get_worker_info():
            case None:
                # main process case
                if self.sequence_loader is not None and self.worker_id is None:
                    # skip re-initialization
                    return
                # in case we move back from a worker to the main process we should also re-init
                self.worker_id = None
            case torch.utils.data._utils.worker.WorkerInfo(id=worker_id):  # type:ignore
                # worker process case
                if self.sequence_loader is not None and self.worker_id is worker_id:
                    # skip re-initialization
                    return
                self.worker_id = worker_id

        # Reload case: check if reloading is sufficient for initialization, terminate if yes
        if self.sequence_loader is not None:
            # only reload data loaders
            self.sequence_loader.reload_resources()
            if self.aux_loader is not None:
                self.aux_loader.reload_store_resources()
            return

        # No termination means we need a full initialization
        self._load_data()

    @staticmethod
    def _check_if_present_sensor_ids(
        sensor_ids_requested: List[str], sensor_ids_shard: List[str], sensor_type: str
    ) -> None:
        """Check if requested sensor data in the shard."""
        if sensor_ids_not_in_shard := [
            sensor_id for sensor_id in sensor_ids_requested if sensor_id not in sensor_ids_shard
        ]:
            raise RuntimeError(f"Requested {sensor_type} not present in the data: {', '.join(sensor_ids_not_in_shard)}")

    def _get_dataset_camera_mask(
        self,
        camera_id: str,
        camera_sensor: ncore.data.CameraSensorProtocol,
    ) -> np.ndarray | None:
        mask_image, resolution = self.resolved_dataset.backend.get_camera_mask_and_resolution(camera_sensor)
        return ncore_utils.get_mask_image(
            mask_image,
            tuple(resolution),
            self.camera_mask_overrides.get(camera_id),
        )

    def _load_data(self) -> None:
        """Initializes the dataset from scratch (start-up case)"""
        # Full initial load case
        self.camera_sensors: dict[str, ncore.data.CameraSensorProtocol] = {}
        self.camera_models: dict[str, ncore.sensors.CameraModel] = {}
        self.camera_model_parameters: dict[str, ncore.data.ConcreteCameraModelParametersUnion] = {}
        self.camera_unique_ids: dict[str, UniqueSensorId] = {}
        self.cameras_all_pixels: dict[str, np.ndarray] = {}
        self.cameras_all_rays: dict[str, np.ndarray] = {}
        self.cameras_all_rays_halfpixoffset: dict[str, np.ndarray] = {}
        self.cameras_all_footprints: dict[str, np.ndarray] = {}
        self.cameras_all_footprints_halfpixoffset: dict[str, np.ndarray] = {}
        self.cameras_valid_pixels_ego_masks: dict[str, PackedMask] = {}
        self.cameras_frame_valid_pixels_masks: dict[str, dict[int, PackedMask]] = defaultdict(dict)
        self.cameras_frame_track_idxs: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
        self.lidars_frame_valid_points_masks: dict[str, dict[int, PackedMask]] = defaultdict(dict)
        self.lidars_frame_non_dynamic_points_masks: dict[str, dict[int, PackedMask]] = defaultdict(dict)
        self.lidar_sensors: dict[str, ncore.data.LidarSensorProtocol] = {}
        self.lidar_models: dict[str, ConcreteLidarModelsUnion | None] = {}
        self.lidar_model_parameters: dict[str, Optional[ncore.data.ConcreteLidarModelParametersUnion]] = {}
        self.lidar_unique_ids: dict[str, UniqueSensorId] = {}

        # Per-sensor model-free firing-lattice estimators (built lazily on first
        # use for the 'aggregate_lattice' unstructured raydrop mode). The firing
        # lattice is a PER-SENSOR invariant (validated: per-direction rel_t std
        # across frames ~8us), so caching it once per sensor is correct -- unlike
        # the per-frame tiling/pose which must never be cached per sensor.
        self._firing_lattices: dict[tuple[str, int], FiringLatticeEstimator] = {}

        self.cuboidtracks_all: CuboidTracks | None = None
        self.cuboidtracks_dynamic: CuboidTracks | None = None
        self.cuboids_df: pd.DataFrame | None = None
        self.track_split_metadata: TrackSplitMetadata | None = None

        self.aux_loader: Optional[ncore_utils.AuxShardDataLoader] = None
        self.sensor_sky_class_ids: dict[str, int] = {}
        self.sensor_road_class_ids: dict[str, int] = {}
        self.sensor_vehicle_classes_ids: dict[str, list[int]] = {}
        self.sensor_trafficlight_class_ids: dict[str, int] = {}
        self.sensor_dinov2_windows: dict[str, np.ndarray] = {}

        # init sequence loader
        sequence_loader = self.sequence_loader = self.resolved_dataset.create_sequence_loader()

        # init aux loader if enabled
        if self.aux_data:
            self.aux_loader = ncore_utils.AuxShardDataLoader(
                sequence_id=sequence_loader.sequence_id,
                dataset_paths=self.dataset_paths,
                open_consolidated=self.open_consolidated,
            )

        ## Load sensor data and

        # conditionally applied per-sensor overwrites
        S = TypeVar("S", bound=ncore.data.SensorProtocol)

        def patch_sensor(sensor: S) -> S:
            """Patch sensor behaviour conditionally based on config options"""

            if self.frame_generic_data_pose_overwrite:
                # Create a new pose graph associated with this sensor
                # instance that makes use of overwritten T_sensor_world poses
                # instead of the regular rig-consistent T_rig_world poses

                # Collect all per-frame start/end T_sensor_world poses from generic data
                T_sensor_worlds_list = []

                for frame_idx in range(sensor.frames_count):
                    if sensor.has_frame_generic_data(frame_idx, "T_sensor_worlds"):
                        T_sensor_worlds_list.append(
                            sensor.get_frame_generic_data(frame_idx, "T_sensor_worlds")
                        )  # (2,4,4)
                    else:
                        raise RuntimeError(
                            f"Sensor {sensor.sensor_id} frame {frame_idx} missing generic data field 'T_sensor_worlds' required for pose overwrite"
                        )
                T_sensor_worlds = np.concatenate(T_sensor_worlds_list, axis=0)  # (2*N_frames,4,4)
                T_sensor_worlds_timestamps_us = sensor.frames_timestamps_us.flatten()

                # Filter out unique timestamps / poses
                # (for instance, global shutter sensors / virtual sensors tend to have identical start/end poses)
                T_sensor_worlds_timestamps_us, unique_indices = np.unique(
                    T_sensor_worlds_timestamps_us, return_index=True
                )
                T_sensor_worlds = T_sensor_worlds[unique_indices]

                # Grab the existing pose-graph edges except for the rig-world edge, and add our new sensor-world edge
                edges = [
                    edge
                    for nodes, edge in sensor.pose_graph.normalized_edge_map.items()
                    if nodes != ncore_transformations.PoseGraphInterpolator.normalize_node_pair("rig", "world")
                ] + [
                    ncore_transformations.PoseGraphInterpolator.Edge(
                        sensor.sensor_id,
                        "world",
                        T_sensor_worlds,
                        T_sensor_worlds_timestamps_us,
                    )
                ]

                # Instantiate and assign overwritten pose graph to sensor
                sensor.set_pose_graph(ncore_transformations.PoseGraphInterpolator(edges))

            return sensor

        # load camera sensors
        self._check_if_present_sensor_ids(self.all_camera_ids, sequence_loader.camera_ids, "cameras")
        self.camera_sensors = {
            camera_id: patch_sensor(sequence_loader.get_camera_sensor(camera_id)) for camera_id in self.all_camera_ids
        }

        # load camera model parameters and models
        self.camera_model_parameters = {
            camera_id: require_concrete_camera_parameters(camera_sensor.model_parameters)
            for camera_id, camera_sensor in self.camera_sensors.items()
        }

        # restrict effective FOV of omnidirectional cameras for cuboid projections to prevent issues with *invalid* points projected
        # back into the valid image domain FOV - this way they get properly classified as invalid
        for camera_model_parameter in self.camera_model_parameters.values():
            if not isinstance(
                camera_model_parameter,
                (ncore.data.FThetaCameraModelParameters, ncore.data.OpenCVFisheyeCameraModelParameters),
            ):
                continue
            camera_model_parameter.max_angle = min(
                np.deg2rad(self.camera_max_fov_deg) / 2.0, camera_model_parameter.max_angle
            )

        camera_models = self.camera_models = {
            camera_id: ncore.sensors.camera_model_from_parameters(
                camera_model_parameter, device="cpu", dtype=torch.float32
            )
            for camera_id, camera_model_parameter in self.camera_model_parameters.items()
        }

        # construct unique camera instance ids (<camera_id>@<sequence_id>) and instance indices
        self.camera_unique_ids = {
            camera_id: UniqueSensorId("@".join((camera_id, self.sequence_id)), camera_instance_idx)
            for camera_instance_idx, camera_id in enumerate(self.all_camera_ids)
        }

        # determine all pixels and rays per camera
        for camera_id in self.all_camera_ids:
            camera_sensor = self.camera_sensors[camera_id]
            camera_model = camera_models[camera_id]

            # sample pixel ranges
            w = int(camera_model.resolution[0].item())
            h = int(camera_model.resolution[1].item())

            # all pixels
            self.cameras_all_pixels |= {camera_id: generate_grid_2d_indices((w, h)).numpy()}

            # dataset associated mask or mask overwrite (if present)
            dataset_camera_mask_array = self._get_dataset_camera_mask(camera_id, camera_sensor)

            # aux camera mask (if enabled and present)
            aux_camera_mask_array: Optional[np.ndarray] = None
            if self.aux_loader is not None:
                if self.aux_loader.has_egomask(camera_id):
                    aux_camera_mask_array = self.aux_loader.get_egomask(camera_id, 0)  # get the aggregated ego-mask

            # determine final mask to use based on priority of sources defined in config
            camera_mask_array: Optional[np.ndarray] = None
            for camera_mask_source in self.camera_mask_sources:
                if camera_mask_source == "dataset":
                    camera_mask_array = dataset_camera_mask_array
                elif camera_mask_source == "aux":
                    camera_mask_array = aux_camera_mask_array
                else:
                    raise ValueError(f"Invalid camera mask source {camera_mask_source} in config")

                if camera_mask_array is not None:
                    break  # found a valid mask from this source

            if camera_mask_array is not None:
                # Dilate mask boundary
                camera_mask_array = ndimage.binary_dilation(
                    camera_mask_array, iterations=self.n_camera_mask_dilation_iterations
                )

                # Subsample valid pixels relative to mask (True for parts that we want to keep)
                camera_valid_pixels_ego_mask = np.logical_not(cast(np.ndarray, camera_mask_array))
            else:
                # No mask / consider all pixels as valid
                camera_valid_pixels_ego_mask = np.ones(
                    (int(camera_model.resolution[1].item()), int(camera_model.resolution[0].item())), dtype=bool
                )

            if self.camera_mask_border is not None and camera_id in self.camera_mask_border:
                # top, right, bottom, left
                camera_valid_pixels_ego_mask[: self.camera_mask_border[camera_id][0]] = False
                camera_valid_pixels_ego_mask[:, : self.camera_mask_border[camera_id][1]] = False
                if self.camera_mask_border[camera_id][2] != 0:
                    camera_valid_pixels_ego_mask[-self.camera_mask_border[camera_id][2] :] = False

                if self.camera_mask_border[camera_id][3] != 0:
                    camera_valid_pixels_ego_mask[:, -self.camera_mask_border[camera_id][3] :] = False

            self.cameras_valid_pixels_ego_masks |= {camera_id: PackedMask(camera_valid_pixels_ego_mask)}

            # precompute all rays at pixel centers and at half-pixel offsets
            self.cameras_all_rays |= {
                camera_id: camera_model.pixels_to_camera_rays(self.cameras_all_pixels[camera_id])
                .reshape(h, w, 3)
                .numpy()
            }
            self.cameras_all_rays_halfpixoffset |= {
                camera_id: camera_model.image_points_to_camera_rays(
                    camera_model.pixels_to_image_points(self.cameras_all_pixels[camera_id]) + 0.5
                )
                .reshape(h, w, 3)
                .numpy()
            }

            # Compute the footprint of the pixels at unit length
            self.cameras_all_footprints |= {camera_id: compute_pixel_footprint(self.cameras_all_rays[camera_id])}
            self.cameras_all_footprints_halfpixoffset |= {
                camera_id: compute_pixel_footprint(self.cameras_all_rays_halfpixoffset[camera_id])
            }

        # load lidar sensors and lidar model parameters
        self._check_if_present_sensor_ids(self.all_lidar_ids, sequence_loader.lidar_ids, "lidars")
        self.lidar_sensors = {
            lidar_id: patch_sensor(sequence_loader.get_lidar_sensor(lidar_id)) for lidar_id in self.all_lidar_ids
        }

        def get_lidar_model_parameters(
            lidar_id: str,
            lidar_sensor: ncore.data.LidarSensorProtocol,
            nominal_values: PerSensor[LidarNominalModel],
        ) -> Optional[ncore.data.ConcreteLidarModelParametersUnion]:
            """Wrapper to load lidar model parameters from the dataset, applying optional fallbacks if enabled"""
            # Pydantic has already narrowed the values to LidarNominalModel. Cast
            # through str because the mutable dict arm makes PerSensor invariant.
            nominal_values_str = cast(PerSensor[str], nominal_values)
            nominal_model = resolve_per_sensor(nominal_values_str, lidar_id, default=None)
            if nominal_model is not None:
                lidar_models: LidarModelsConfig = unpack_optional(self.lidar_model_config)
                if nominal_model == "HESAI-AT128":
                    return LidarModelBundle.load_from_config(lidar_models.HESAI_AT128).lidar_parameters
                elif nominal_model == "HESAI-Pandar128":
                    return LidarModelBundle.load_from_config(lidar_models.HESAI_Pandar128).lidar_parameters
                else:
                    raise ValueError(f"{self.__class__.__name__} lidar model type {nominal_model} not supported")
            lidar_model_parameters = get_lidar_model_parameters_with_fallbacks(
                lidar_sensor,
                backend=self.resolved_dataset.backend,
                cwccw_fallback=self.lidar_model_parameter_cwccw_fallback,
            )
            return lidar_model_parameters

        self.lidar_model_parameters = {
            lidar_id: get_lidar_model_parameters(lidar_id, lidar_sensor, self.lidar_model_parameter_nominal_values)
            for lidar_id, lidar_sensor in self.lidar_sensors.items()
        }

        self.lidar_models = {
            lidar_id: cast(
                ConcreteLidarModelsUnion | None,
                ncore.sensors.maybe_lidar_model_from_parameters(
                    lidar_model_parameters, device="cpu", dtype=torch.float32
                ),
            )
            for lidar_id, lidar_model_parameters in self.lidar_model_parameters.items()
        }

        # construct unique lidar instance ids (<lidar_id>@<sequence_id>) and instance indices
        self.lidar_unique_ids = {
            lidar_id: UniqueSensorId("@".join((lidar_id, self.sequence_id)), lidar_instance_idx)
            for lidar_instance_idx, lidar_id in enumerate(self.all_lidar_ids)
        }

        # load aux data shards if enabled
        if self.aux_loader is not None:
            # get index of 'sky' class for each sensor
            self.sensor_sky_class_ids = {
                camera_id: self.aux_loader.get_semantic_segmentation_meta(camera_id)["stuff_classes"].index("sky")
                for camera_id in self.all_camera_ids
            }
            if self.aux_loader.has_lidar_semantic_segmentation():
                self.sensor_sky_class_ids.update(
                    {
                        lidar_id: self.aux_loader.get_lidar_semantic_segmentation_meta(lidar_id)["stuff_classes"].index(
                            "sky"
                        )
                        for lidar_id in self.all_lidar_ids
                    }
                )
            # get index of 'road' class for each sensor
            self.sensor_road_class_ids = {
                camera_id: self.aux_loader.get_semantic_segmentation_meta(camera_id)["stuff_classes"].index("road")
                for camera_id in self.all_camera_ids
            }
            if self.aux_loader.has_lidar_semantic_segmentation():
                self.sensor_road_class_ids.update(
                    {
                        lidar_id: self.aux_loader.get_lidar_semantic_segmentation_meta(lidar_id)["stuff_classes"].index(
                            "road"
                        )
                        for lidar_id in self.all_lidar_ids
                    }
                )
            # get index of vehicle classes (including "car", "truck", "bus", "train") for each sensor
            VEHICLE_CLASSES = ["car", "truck", "bus", "train"]
            self.sensor_vehicle_classes_ids = {
                camera_id: [
                    stuff_classes.index(vehicle_class)
                    for vehicle_class in VEHICLE_CLASSES
                    if vehicle_class
                    in (stuff_classes := self.aux_loader.get_semantic_segmentation_meta(camera_id)["stuff_classes"])
                ]
                for camera_id in self.all_camera_ids
            }

            if self.aux_loader.has_lidar_semantic_segmentation():
                self.sensor_vehicle_classes_ids.update(
                    {
                        lidar_id: [
                            self.aux_loader.get_lidar_semantic_segmentation_meta(lidar_id)["stuff_classes"].index(
                                vehicle_class
                            )
                            for vehicle_class in ["car", "truck", "bus", "train"]
                        ]
                        for lidar_id in self.all_lidar_ids
                    }
                )

            # get DINOv2 related metadata
            if self.aux_loader.has_dinov2():
                self.sensor_dinov2_windows = {}
                for camera_id in self.all_camera_ids:
                    dinov2_meta_info, _, _ = self.aux_loader.get_dinov2_meta(camera_id)
                    self.sensor_dinov2_windows[camera_id] = np.asarray(
                        [
                            dinov2_meta_info["window_left"],
                            dinov2_meta_info["window_top"],
                            dinov2_meta_info["window_width"],
                            dinov2_meta_info["window_height"],
                        ]
                    )

            match self.valid_measurements_method:
                case NCOREDataSource.ValidMeasurementsMethod.EGO_CUBOIDTRACKS_TRAFFICLIGHT:
                    self.sensor_trafficlight_class_ids = {
                        camera_id: self.aux_loader.get_semantic_segmentation_meta(camera_id)["stuff_classes"].index(
                            "traffic light"
                        )
                        for camera_id in self.all_camera_ids
                    }
                    self.sensor_trafficlight_class_ids.update(
                        {
                            lidar_id: self.aux_loader.get_lidar_semantic_segmentation_meta(lidar_id)[
                                "stuff_classes"
                            ].index("traffic light")
                            for lidar_id in self.all_lidar_ids
                        }
                    )

        cumulative_camera_start_frame_index: int = 0
        cumulative_lidar_start_frame_index: int = 0

        # load all rig->world and world->world-global poses

        # TODO: frame-pose only data might fail here as there are no rig poses and might require refined logic
        rig_world_edge: ncore_transformations.PoseGraphInterpolator.Edge = unpack_optional(
            sequence_loader.pose_graph.get_edge("rig", "world"),
            msg="Rig-to-world poses are currently required to determine scene extend",
        )

        T_rig_world = rig_world_edge.T_source_target
        T_rig_world_timestamps_us: np.ndarray = unpack_optional(
            rig_world_edge.timestamps_us, msg="Rig-to-world pose requires to be dynamic"
        )

        # NCore V4: world→world_global is optional (global CRS, e.g. ECEF). Absent edge ⇒ identity float64, matching
        # world_world_global_edge.T_source_target dtype when the edge exists and RigTrajectories.T_world_base.
        world_world_global_edge = sequence_loader.pose_graph.get_edge("world", "world_global")
        self.T_world_world_global = (
            world_world_global_edge.T_source_target
            if world_world_global_edge is not None
            else np.eye(4, dtype=np.float64)
        )

        # restrict the time_range_us to ensure all rig-world poses are known in the full interval
        # (crucial when using seek_offset_sec/duration_sec, because the next available rig-world
        #  pose may be sampled after the time range starts, rendering the very beginning un-posed)
        self.time_range_us = self.time_range_us.restricted(T_rig_world_timestamps_us)

        # determine linear per-sensor-frame index ranges depending on dataset time restrictions,
        # making sure *both* frame start and end-times are fully covered
        def get_sensor_frame_range(frames_timestamps_us: np.ndarray) -> range:
            # make sure end-of-frame times are are covered by the time range
            cover_range = self.time_range_us.cover_range(frames_timestamps_us[:, ncore.data.FrameTimepoint.END])
            # make sure that the first frame's start-of-frame time is also covered - skip frames as required
            # (could be more than a single frame if frame ranges are not exclusively partitioning the time range)
            while len(cover_range) and (
                int(frames_timestamps_us[cover_range.start, ncore.data.FrameTimepoint.START]) not in self.time_range_us
            ):
                cover_range = cover_range[1:]

            return cover_range

        self.camera_frame_ranges = {
            camera_id: get_sensor_frame_range(self.camera_sensors[camera_id].frames_timestamps_us)
            for camera_id in self.all_camera_ids
        }
        self.lidar_frame_ranges = {
            lidar_id: get_sensor_frame_range(self.lidar_sensors[lidar_id].frames_timestamps_us)
            for lidar_id in self.all_lidar_ids
        }

        # determine per-sensor starting index to return unique linear frame indices for each sample
        for camera_id, camera_frame_range in self.camera_frame_ranges.items():
            self.camera_linear_start_frame_indices[camera_id] = cumulative_camera_start_frame_index
            cumulative_camera_start_frame_index += len(camera_frame_range)

        for lidar_id, lidar_frame_range in self.lidar_frame_ranges.items():
            self.lidar_linear_start_frame_indices[lidar_id] = cumulative_lidar_start_frame_index
            cumulative_lidar_start_frame_index += len(lidar_frame_range)

        # compute average position and extent (largest axis of the scene's AABB relative to the world frame)
        # to put the scene's center at the origin
        rig_world_positions = T_rig_world[self.time_range_us.cover_range(T_rig_world_timestamps_us)][:, :3, 3]
        mean_rig_world_position_m = rig_world_positions.mean(axis=0).astype(np.float32)

        # Center the scenes at the mean rig position
        self.world_to_nurec = FrameConversion.from_origin_scale_axis(
            target_origin=mean_rig_world_position_m,  # put the scene's center at the origin
            target_scale=1.0,
            target_axis=[0, 1, 2],
        )

        # Compute per-frame valid measurements (pixels/lidarpoints), e.g., excluding dynamic objects, ego-car, ...
        match self.valid_measurements_method:
            case NCOREDataSource.ValidMeasurementsMethod.EGO:
                # camera pixel masks
                self.cameras_frame_valid_pixels_masks = compute_valid_pixels_ego(
                    self.camera_frame_ranges, self.cameras_valid_pixels_ego_masks
                )
                # lidar point masks (all lidar points are considered valid in this measurement method)
                self.lidars_frame_valid_points_masks = compute_valid_lidarpoints_all(
                    self.lidar_frame_ranges,
                    self.lidar_sensors,
                )
            case NCOREDataSource.ValidMeasurementsMethod.EGO_CUBOIDTRACKS:
                # camera pixel masks
                self.cameras_frame_valid_pixels_masks = compute_valid_pixels_ego(
                    self.camera_frame_ranges, self.cameras_valid_pixels_ego_masks
                )
                self._compute_valid_pixels_cuboidtracks()
                # lidar point masks
                self._compute_valid_lidarpoints_dynamic()
            case NCOREDataSource.ValidMeasurementsMethod.EGO_SCENEFLOW:
                # camera pixel masks
                self.cameras_frame_valid_pixels_masks = compute_valid_pixels_ego(
                    self.camera_frame_ranges, self.cameras_valid_pixels_ego_masks
                )
                assert self.aux_loader is not None
                self.cameras_frame_valid_pixels_masks = compute_valid_pixels_sceneflow(
                    aux_loader=self.aux_loader,
                    camera_sensors=self.camera_sensors,
                    camera_models=self.camera_models,
                    camera_frame_ranges=self.camera_frame_ranges,
                    valid_pixels_scene_flow_config=self.valid_pixels_pixels_scene_flow_params,
                    cameras_frame_valid_pixels_masks=self.cameras_frame_valid_pixels_masks,
                )
                # lidar point masks (all lidar points are considered valid in this measurement method)
                self.lidars_frame_valid_points_masks = compute_valid_lidarpoints_all(
                    self.lidar_frame_ranges,
                    self.lidar_sensors,
                )
            case NCOREDataSource.ValidMeasurementsMethod.EGO_CUBOIDTRACKS_SCENEFLOW:
                # camera pixel masks
                self.cameras_frame_valid_pixels_masks = compute_valid_pixels_ego(
                    self.camera_frame_ranges, self.cameras_valid_pixels_ego_masks
                )
                self._compute_valid_pixels_cuboidtracks()
                assert self.aux_loader is not None
                self.cameras_frame_valid_pixels_masks = compute_valid_pixels_sceneflow(
                    aux_loader=self.aux_loader,
                    camera_sensors=self.camera_sensors,
                    camera_models=self.camera_models,
                    camera_frame_ranges=self.camera_frame_ranges,
                    valid_pixels_scene_flow_config=self.valid_pixels_pixels_scene_flow_params,
                    cameras_frame_valid_pixels_masks=self.cameras_frame_valid_pixels_masks,
                )
                # lidar point masks
                self._compute_valid_lidarpoints_dynamic()
            case NCOREDataSource.ValidMeasurementsMethod.EGO_CUBOIDTRACKS_TRAFFICLIGHT:
                # camera pixel masks
                self.cameras_frame_valid_pixels_masks = compute_valid_pixels_ego(
                    self.camera_frame_ranges, self.cameras_valid_pixels_ego_masks
                )
                self._compute_valid_pixels_cuboidtracks()

                assert self.aux_loader is not None
                self.cameras_frame_valid_pixels_masks = compute_valid_pixels_trafficlight(
                    aux_loader=self.aux_loader,
                    camera_sensors=self.camera_sensors,
                    camera_frame_ranges=self.camera_frame_ranges,
                    valid_pixels_traffic_light_params=self.valid_pixels_traffic_light_params,
                    sensor_trafficlight_class_ids=self.sensor_trafficlight_class_ids,
                    cameras_frame_valid_pixels_masks=self.cameras_frame_valid_pixels_masks,
                    tqdm_disabled=self.tqdm_disabled,
                )
                # lidar point masks
                self.lidars_frame_valid_points_masks = compute_valid_lidarpoints_trafficlight_cameravisible(
                    aux_loader=self.aux_loader,
                    lidar_sensors=self.lidar_sensors,
                    all_camera_ids=self.all_camera_ids,
                    time_range_us=self.time_range_us,
                    sensor_trafficlight_class_ids=self.sensor_trafficlight_class_ids,
                    lidars_frame_valid_points_masks=self.lidars_frame_valid_points_masks,
                    tqdm_disabled=self.tqdm_disabled,
                )
                self._compute_valid_lidarpoints_dynamic()
            case NCOREDataSource.ValidMeasurementsMethod.EGO_FRAMEMASKS:
                # camera pixel masks
                self.cameras_frame_valid_pixels_masks = compute_valid_pixels_ego(
                    self.camera_frame_ranges, self.cameras_valid_pixels_ego_masks
                )
                self.cameras_frame_valid_pixels_masks = compute_cameras_valid_pixels_frame_mask(
                    camera_sensors=self.camera_sensors,
                    camera_frame_ranges=self.camera_frame_ranges,
                    valid_pixels_frame_mask_params=self.valid_pixels_frame_mask_params,
                    cameras_frame_valid_pixels_masks=self.cameras_frame_valid_pixels_masks,
                    tqdm_disabled=self.tqdm_disabled,
                )
                # lidar point masks (all lidar points are considered valid in this measurement method)
                self.lidars_frame_valid_points_masks = compute_valid_lidarpoints_all(
                    self.lidar_frame_ranges,
                    self.lidar_sensors,
                )
            case _:
                raise ValueError(
                    f"{self.__class__.__name__}: unsupported valid measurements method {self.valid_measurements_method}"
                )

        # Compute scene extent from point-clouds, if available, otherwise fall back to trajectory-inferred extent
        extent = torch.tensor(0.0, dtype=torch.float32)
        if len(
            point_clouds := list(
                self.get_point_clouds(
                    device=torch.device("cuda"), step_frame=10, non_dynamic_points_only=True, force=False
                )
            )
        ):
            pc = PointCloud.collate_fn(point_clouds, device=torch.device("cuda"))
            if pc.n_points > 0:
                extent = (pc.xyz_end.max(0).values - pc.xyz_end.min(0).values).max()

        if extent == 0:
            # make sure that the max distance at the boundary is included when scaling the scene extent to the target AABB scale
            world_max_extent_m = np.max(np.max(rig_world_positions, axis=0) - np.min(rig_world_positions, axis=0))
            extent = torch.tensor(world_max_extent_m + 2 * self.max_dist_m, dtype=torch.float32)

        # TODO: Cube for now, change to cuboid
        extent = torch.tensor([extent] * 3, dtype=torch.float32)

        self.aabb = AABB3D.from_center_extent(center=torch.zeros(3, dtype=torch.float32), extent=extent)

    def _compute_cuboid_tracks(self) -> None:
        """Computes cuboid tracks for all dynamic objects in the dataset"""
        # Only compute tracks if not initialized before
        if (self.cuboidtracks_all is None) or (self.cuboidtracks_dynamic is None) or (self.cuboids_df is None):
            cuboidtracks_all, self.cuboids_df = compute_cuboid_tracks(
                sequence_loader=unpack_optional(self.sequence_loader),
                time_range_us=self.time_range_us,
                cuboid_tracks_params=self.cuboid_tracks_params,
                tqdm_disabled=self.tqdm_disabled,
                serialize_observation=False,
            )

            # Optional: compute per-track camera-visible time intervals to restrict dynamic classification to those windows
            cuboidtracks_visible_intervals = None
            if self.cuboid_tracks_params.camera_visibility:
                cuboidtracks_visible_intervals = compute_camera_visible_intervals_cuboid_tracks(
                    cuboidtracks=cuboidtracks_all,
                    camera_sensors=self.camera_sensors,
                    camera_models=self.camera_models,
                    camera_frame_ranges=self.camera_frame_ranges,
                    cameras_all_pixels=self.cameras_all_pixels,
                    cameras_all_rays=self.cameras_all_rays,
                    subsample=4,
                    tqdm_disabled=self.tqdm_disabled,
                )

            self.cuboidtracks_all, self.cuboidtracks_dynamic = classify_dynamic_cuboid_tracks(
                cuboidtracks=cuboidtracks_all,
                cuboid_tracks_params=self.cuboid_tracks_params,
                unconditionally_dynamic_classes=self.UNCONDITIONALLY_DYNAMIC_CLASSES,
                tqdm_disabled=self.tqdm_disabled,
                cuboidtracks_visible_intervals=cuboidtracks_visible_intervals,
            )
            self.track_split_metadata = None

            visible_track_split_config = self.cuboid_tracks_params.visible_track_split
            if visible_track_split_config.enabled:
                assert self.cuboidtracks_all is not None
                target_label_classes = visible_track_split_config.target_label_classes
                if not target_label_classes:
                    raise ValueError(
                        "dataset.cuboid_tracks_params.visible_track_split requires target_label_classes when enabled"
                    )

                split_camera_ids = visible_track_split_config.camera_ids or self.train_camera_ids
                if not split_camera_ids:
                    raise ValueError(
                        "dataset.cuboid_tracks_params.visible_track_split.enabled requires at least one camera id"
                    )
                unknown_camera_ids = sorted(set(split_camera_ids) - set(self.camera_sensors))
                if unknown_camera_ids:
                    raise ValueError(
                        "dataset.cuboid_tracks_params.visible_track_split.camera_ids contains unknown camera ids: "
                        f"{unknown_camera_ids}"
                    )

                split_track_mask = CuboidTracks.Ops.mask_by_label_classes(self.cuboidtracks_all, target_label_classes)
                if visible_track_split_config.min_track_distance_m is not None:
                    split_track_mask &= CuboidTracks.Ops.mask_by_min_distance(
                        self.cuboidtracks_all,
                        visible_track_split_config.min_track_distance_m,
                    )
                if not split_track_mask.any():
                    return

                split_visible_intervals = compute_camera_visible_intervals_cuboid_tracks(
                    cuboidtracks=self.cuboidtracks_all,
                    camera_sensors={camera_id: self.camera_sensors[camera_id] for camera_id in split_camera_ids},
                    camera_models={camera_id: self.camera_models[camera_id] for camera_id in split_camera_ids},
                    camera_frame_ranges={
                        camera_id: self.camera_frame_ranges[camera_id] for camera_id in split_camera_ids
                    },
                    cameras_all_pixels={
                        camera_id: self.cameras_all_pixels[camera_id] for camera_id in split_camera_ids
                    },
                    cameras_all_rays={camera_id: self.cameras_all_rays[camera_id] for camera_id in split_camera_ids},
                    subsample=visible_track_split_config.visibility_subsample,
                    tqdm_disabled=self.tqdm_disabled,
                )

                split_children: list[TrackSplitChild] = []
                self.cuboidtracks_all = CuboidTracks.Ops.split_by_visible_intervals(
                    cuboid_tracks=self.cuboidtracks_all,
                    visible_intervals_us=split_visible_intervals,
                    max_visible_duration_us=int(visible_track_split_config.max_visible_duration_s * 1_000_000),
                    split_mask=split_track_mask,
                    merge_gap_us=int(visible_track_split_config.merge_gap_s * 1_000_000),
                    min_child_visible_duration_us=int(
                        visible_track_split_config.min_child_visible_duration_s * 1_000_000
                    ),
                    split_children_out=split_children,
                )
                if split_children:
                    self.track_split_metadata = TrackSplitMetadata(children=split_children)
                dynamic_track_mask = self.cuboidtracks_all.get_mask_flags_all(TrackFlags.DYNAMIC)
                self.cuboidtracks_dynamic = CuboidTracks.Ops.subset_from_mask(
                    self.cuboidtracks_all,
                    dynamic_track_mask,
                )

    def _compute_valid_pixels_cuboidtracks(self) -> None:
        """Computes per-frame projections of cuboid track dynamic objects as valid-pixels masks"""

        # tracks are required
        self._compute_cuboid_tracks()

        assert self.cuboidtracks_dynamic is not None
        compute_camera_visible_intervals_cuboid_tracks(
            cuboidtracks=self.cuboidtracks_dynamic,
            camera_sensors=self.camera_sensors,
            camera_models=self.camera_models,
            camera_frame_ranges=self.camera_frame_ranges,
            cameras_all_pixels=self.cameras_all_pixels,
            cameras_all_rays=self.cameras_all_rays,
            cameras_frame_valid_pixels_masks=self.cameras_frame_valid_pixels_masks,
            cameras_frame_track_idxs=self.cameras_frame_track_idxs
            if self.valid_pixels_cuboid_tracks_params.frame_track_idxs
            else None,
            valid_pixels_cuboid_tracks_params=self.valid_pixels_cuboid_tracks_params,
            max_intersections_per_ray=1,
            tqdm_disabled=self.tqdm_disabled,
        )

    def _compute_lidarpoints_dynamic(self, visualize=False) -> None:
        """
        Computes per-frame lidar dynamic point masks, either directly loading dynamic flag that comes
        with the data, or computing the flag by overlap with dynamic cuboid tracks.
        """

        def visualize_dynamic_points(points_all: np.ndarray, segmentation_mask: np.ndarray, tracks: dict) -> None:
            col = np.repeat(np.array([[0, 0, 255]], dtype=np.float32), points_all.shape[0], axis=0) / 255
            col[segmentation_mask, :] = np.array([[255, 0, 0]], dtype=np.float32)

            visualdebugger = get_visualdebugger()
            visualdebugger.clear()
            visualdebugger.add_point_cloud(
                "Lidar points", points_all, colors_quantities={"All points": col}, enabled=True, radius=0.0001
            )

            for track_id, (pose, half_dim) in tracks.items():
                cuboid_v = np.array(
                    [
                        [-0.5, -0.5, -0.5],
                        [0.5, -0.5, -0.5],
                        [0.5, 0.5, -0.5],
                        [-0.5, 0.5, -0.5],
                        [-0.5, -0.5, 0.5],
                        [0.5, -0.5, 0.5],
                        [0.5, 0.5, 0.5],
                        [-0.5, 0.5, 0.5],
                    ]
                )
                cuboid_v *= half_dim * 2
                cuboid_v = np.dot(cuboid_v, pose[:3, :3].T) + pose[:3, 3]

                cuboid_f = np.array(
                    [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4]]
                )

                cuboid_color = (235 / 255.0, 240 / 255.0, 11 / 255.0)

                visualdebugger.add_surface_mesh(
                    f"t-{track_id}",
                    cuboid_v,
                    cuboid_f,
                    color=cuboid_color,
                    edge_width=2,
                    edge_color=(0.0, 0.0, 0.0),
                    transparency=0.3,
                )
            visualdebugger.show()

        if len(self.lidars_frame_non_dynamic_points_masks):
            # masks already computed / exit early
            return

        if self.lidar_dynamic_points_params.method == "dynamic_tracks":
            # tracks are required for "dynamic_tracks" method
            self._compute_cuboid_tracks()

        for lidar_id in self.lidar_sensors.keys():
            lidar_frame_valid_points_masks = self.lidars_frame_valid_points_masks[lidar_id]
            lidar_frame_non_dynamic_points_masks = self.lidars_frame_non_dynamic_points_masks[lidar_id]

            lidar_sensor = self.lidar_sensors[lidar_id]

            for lidar_frame_idx in tqdm.tqdm(
                self.lidar_frame_ranges[lidar_id],
                desc=f"Dynamic Lidar Points Masks [lidars->frames] via '{self.lidar_dynamic_points_params.method}'",
                disable=self.tqdm_disabled,
            ):
                xyz_m_end: np.ndarray | None = None
                frame_dyn_tracks = {}  # only filled by for "dynamic_tracks" method / used for visualization
                non_dynamic_point_mask: np.ndarray
                match self.lidar_dynamic_points_params.method:
                    case "dynamic_flag":
                        # load dynamic flag from the source data, mark non-dynamic points as *valid* only
                        if (
                            dynamic_flag := lidar_frame_dynamic_flag(
                                lidar_sensor,
                                lidar_frame_idx,
                                backend=self.resolved_dataset.backend,
                            )
                        ) is None:
                            # dynamic flag data is not available - error out
                            raise ValueError(
                                f"{self.__class__.__name__}: dynamic_flag data missing for lidar {lidar_id} at frame {lidar_frame_idx}"
                            )

                        non_dynamic_point_mask = (
                            dynamic_flag != 1
                        )  # 1 ~ Point is classified to be dynamic [deprecated value of DynamicFlagState.DYNAMIC]

                    case "dynamic_tracks":
                        assert self.cuboidtracks_dynamic is not None and self.cuboids_df is not None

                        # load dynamic track observations in current frame time
                        frame_time_start = lidar_sensor.get_frame_timestamp_us(
                            lidar_frame_idx, frame_timepoint=ncore.data.FrameTimepoint.START
                        )
                        frame_time_end = lidar_sensor.get_frame_timestamp_us(
                            lidar_frame_idx, frame_timepoint=ncore.data.FrameTimepoint.END
                        )
                        frame_cuboids_df = self.cuboids_df[
                            (self.cuboids_df["timestamp_us"] >= frame_time_start)
                            & (self.cuboids_df["timestamp_us"] <= frame_time_end)
                        ]

                        observation_track_ids = _get_observation_track_ids_for_dynamic_filter(
                            self.cuboidtracks_dynamic.tracks_id,
                            self.track_split_metadata,
                        )
                        dynamic_frame_cuboids_df = frame_cuboids_df[
                            frame_cuboids_df["track_id"].isin(observation_track_ids)
                        ]

                        dynamic_cuboid_observations = []
                        for row in dynamic_frame_cuboids_df.itertuples(index=False):
                            row_data: dict[str, Any] = cast(Any, row)._asdict()
                            dynamic_cuboid_observations.append(
                                ncore.data.CuboidTrackObservation.from_dict(row_data)
                                if isinstance(row.bbox3, dict)
                                else ncore.data.CuboidTrackObservation(**row_data)
                            )

                        # Default annotation - all points are non-dynamic
                        non_dynamic_point_mask = np.full(lidar_sensor.get_frame_ray_bundle_count(lidar_frame_idx), True)

                        # If there are any dynamic tracks in this frame use them to labels the point cloud
                        if len(dynamic_cuboid_observations):
                            # Grab the current point-cloud to annotate it (motion-compensated to end-of-spin frame)
                            pc = lidar_sensor.get_frame_point_cloud(
                                lidar_frame_idx,
                                motion_compensation=True,
                                with_start_points=False,
                                return_index=0,  # closest returns only
                            )

                            non_dynamic_point_mask_tensor, cuboid_transforms, track_dim = compute_points_outside_tracks(
                                pc.xyz_m_end,
                                # make sure observations are relative to end-of-spin frame of the lidar
                                ncore_utils.transform_cuboid_track_observations(
                                    observations=dynamic_cuboid_observations,
                                    pose_graph=lidar_sensor.pose_graph,
                                    target_frame_id=lidar_sensor.sensor_id,
                                    target_frame_timestamp_us=lidar_sensor.get_frame_timestamp_us(
                                        lidar_frame_idx,
                                        ncore.data.FrameTimepoint.END,
                                    ),
                                    tqdm_disabled=True,  # don't log individual transformations
                                ),
                                self.valid_points_cuboid_track_params.track_padding_m,
                            )
                            non_dynamic_point_mask = non_dynamic_point_mask_tensor.cpu().numpy()

                            # Visualize the dynamic tracks
                            if visualize:
                                for cuboid_observation_idx, cuboid_observation in enumerate(
                                    dynamic_cuboid_observations
                                ):
                                    frame_dyn_tracks[cuboid_observation.track_id] = (
                                        cuboid_transforms[cuboid_observation_idx].cpu().numpy(),
                                        track_dim[cuboid_observation_idx].cpu().numpy() / 2,
                                    )
                                    cuboid_observation_idx += 1

                    case _:
                        raise ValueError(
                            f"{self.__class__.__name__} unknown lidar_dynamic_point_method {self.lidar_dynamic_points_params.method}"
                        )

                # store per-point static / non-dynamic point flag
                lidar_frame_non_dynamic_points_masks[lidar_frame_idx] = PackedMask(non_dynamic_point_mask)

                if visualize and (xyz_m_end is not None):
                    visualize_dynamic_points(
                        xyz_m_end, lidar_frame_valid_points_masks[lidar_frame_idx].unpacked(), frame_dyn_tracks
                    )

    def _compute_valid_lidarpoints_dynamic(self, visualize=False) -> None:
        """
        Initializes or extends valid lidar point masks based on per-frame lidar dynamic point masks
        (see '_compute_lidarpoints_dynamic()'). Points classified as dynamic will be marked as invalid.
        """

        if not len(self.lidars_frame_non_dynamic_points_masks):
            # compute dynamic point masks explicitly if not present yet
            self._compute_lidarpoints_dynamic()

        for lidar_id in self.lidar_sensors.keys():
            lidar_frame_valid_points_masks = self.lidars_frame_valid_points_masks[lidar_id]
            lidar_frame_non_dynamic_points_masks = self.lidars_frame_non_dynamic_points_masks[lidar_id]

            for lidar_frame_idx in tqdm.tqdm(
                self.lidar_frame_ranges[lidar_id],
                desc=f"Dynamic Lidar Points Masks [lidars->frames] via '{self.lidar_dynamic_points_params.method}'",
                disable=self.tqdm_disabled,
            ):
                # load per-point static / non-dynamic point flag
                non_dynamic_point_mask = unpack_optional(
                    lidar_frame_non_dynamic_points_masks[lidar_frame_idx]
                ).unpacked()

                if (frame_valid_points_mask := lidar_frame_valid_points_masks.get(lidar_frame_idx)) is not None:
                    # extend existing mask
                    lidar_frame_valid_points_masks[lidar_frame_idx] = PackedMask(
                        frame_valid_points_mask.unpacked() & non_dynamic_point_mask
                    )
                else:
                    # initialize new mask
                    lidar_frame_valid_points_masks[lidar_frame_idx] = PackedMask(non_dynamic_point_mask)

    @staticmethod
    def _get_sorted_unique_sensor_ids_list(sensor_unique_ids: dict[str, UniqueSensorId]) -> list[str]:
        """Returns a list of unique sensor ids sorted by their unique index"""
        # sort unique sensor IDs by their unique index
        unique_ids_idx_sorted = sorted(sensor_unique_ids.values(), key=lambda u: u.idx)

        # drop indices, only return string IDs
        return [unique_id.id for unique_id in unique_ids_idx_sorted]

    def get_camera_sensor_ids(self, unique_sensors: bool = True) -> list[str]:
        """Returns the unique (unique_sensors=True, sorted by unique camera sensor idx) or logical (unique_sensors=False) camera sensor ids"""
        if unique_sensors:
            return self._get_sorted_unique_sensor_ids_list(self.camera_unique_ids)
        else:
            return self.all_camera_ids

    def get_lidar_sensor_ids(self, unique_sensors: bool = True) -> list[str]:
        """Returns the unique (unique_sensors=True, sorted by unique lidar sensor idx) or logical (unique_sensors=False) lidar sensor ids"""
        if unique_sensors:
            return self._get_sorted_unique_sensor_ids_list(self.lidar_unique_ids)
        else:
            return self.all_lidar_ids

    # ---- All-frame -> training-only frame index mapping (used by PPISP / post-processing) ----

    # Populated by setup_training_frame_mapping() when the training sampler excludes
    # frames from the all-frame set (e.g. HoldOutFrameSampler). None means training
    # spans the full all-frame set, and the public accessors return identity mappings.
    _n_training_frames_per_camera: npt.NDArray[np.int32] | None = None
    # shape (n_camera_frames_in_source,) over all cameras in the source dataset
    # (training + held-out frames); -1 marks non-training frames in the dataset.
    _global_to_training_camera_frame_idx: npt.NDArray[np.int32] | None = None
    # shape (n_lidar_frames_in_source,) over all lidars in the source dataset
    # (training + held-out frames); -1 marks non-training frames in the dataset.
    _global_to_training_lidar_frame_idx: npt.NDArray[np.int32] | None = None

    def setup_training_frame_mapping(self, sampler_groups: Mapping[str, Any] | Collection[Any]) -> None:
        """Compute the all-frame -> training-only frame index mapping and the
        per-camera training frame counts from the EFFECTIVE training samplers.

        Call this once after the datasource is initialised, from
        ``ncore_dataset_factory``. A frame is a training frame iff at least
        one batch-sampler group's frame sampler can return it (union across
        groups, per each sampler class's ``training_frame_mask``) — the
        mapping is deliberately NOT inferred from the validation holdout
        configuration, so validation is free to render training frames (e.g.
        overfit diagnostics) without invalidating their training indices.
        The mapping converts ``unique_frame_idx`` values (as produced by
        ``get_camera_data_batch`` / ``get_lidar_data_batch``) to contiguous
        0-based training-frame indices, with ``-1`` for frames no training
        sampler can produce.

        When every frame of every sensor is reachable (unrestricted samplers
        such as ``uniform`` / ``timestamp``) — or when no sampler groups are
        configured at all (e.g. pure-validation runs) — the mapping stays
        unset and the public accessors (``get_training_frame_idx_camera``,
        ``get_global_to_training_camera_frame_mapping``,
        ``get_n_training_frames_per_camera``) fall back to the identity
        mapping, sizing PPISP / bilateral-grid parameters to all frames.
        """
        self._maybe_init_worker()

        groups = list(sampler_groups.values()) if isinstance(sampler_groups, Mapping) else list(sampler_groups)

        def union_mask(sensor_kind: str, n_frames: int) -> npt.NDArray[np.bool_]:
            mask = np.zeros(n_frames, dtype=np.bool_)
            for group in groups:
                mask |= BaseFrameSampler.group_training_frame_mask(group, sensor_kind, n_frames)
            return mask

        camera_masks = {
            camera_id: union_mask("camera", len(frame_range))
            for camera_id, frame_range in self.camera_frame_ranges.items()
        }
        lidar_masks = {
            lidar_id: union_mask("lidar", len(frame_range)) for lidar_id, frame_range in self.lidar_frame_ranges.items()
        }

        every_frame_reachable = all(bool(m.all()) for m in camera_masks.values()) and all(
            bool(m.all()) for m in lidar_masks.values()
        )
        no_frame_reachable = not any(bool(m.any()) for m in camera_masks.values()) and not any(
            bool(m.any()) for m in lidar_masks.values()
        )
        if not groups or every_frame_reachable or no_frame_reachable:
            # Identity mapping. The all-empty case (no groups, or every
            # ratio zero) carries no usable information — it only occurs
            # for non-training runs, where the legacy identity sizing is
            # what checkpoints were built with.
            self._n_training_frames_per_camera = None
            self._global_to_training_camera_frame_idx = None
            self._global_to_training_lidar_frame_idx = None
            return

        # --- cameras ---
        total_camera_frames = sum(len(r) for r in self.camera_frame_ranges.values())
        global_to_training_camera = np.full(total_camera_frames, -1, dtype=np.int32)
        n_training_per_camera = np.zeros(len(self.all_camera_ids), dtype=np.int32)
        training_counter = 0

        for camera_id in self.all_camera_ids:
            frame_range = self.camera_frame_ranges[camera_id]
            n_frames = len(frame_range)
            camera_unique_idx = self.camera_unique_ids[camera_id].idx
            linear_start = self.camera_linear_start_frame_indices[camera_id]

            mask = camera_masks[camera_id]
            for local_idx in range(n_frames):
                global_idx = linear_start + local_idx
                if mask[local_idx]:
                    global_to_training_camera[global_idx] = training_counter
                    training_counter += 1
            n_training_per_camera[camera_unique_idx] += int(mask.sum())

        self._n_training_frames_per_camera = n_training_per_camera
        self._global_to_training_camera_frame_idx = global_to_training_camera

        # --- lidars ---
        total_lidar_frames = sum(len(r) for r in self.lidar_frame_ranges.values())
        global_to_training_lidar = np.full(total_lidar_frames, -1, dtype=np.int32)
        training_counter = 0

        for lidar_id in self.all_lidar_ids:
            frame_range = self.lidar_frame_ranges[lidar_id]
            n_frames = len(frame_range)
            linear_start = self.lidar_linear_start_frame_indices[lidar_id]

            mask = lidar_masks[lidar_id]
            for local_idx in range(n_frames):
                global_idx = linear_start + local_idx
                if mask[local_idx]:
                    global_to_training_lidar[global_idx] = training_counter
                    training_counter += 1

        self._global_to_training_lidar_frame_idx = global_to_training_lidar

    def get_n_training_frames_per_camera(self) -> npt.NDArray[np.int32]:
        """Return per-camera training frame counts (unique sensors).

        Falls back to ``get_n_frames_per_camera()`` (all-frame counts) when no
        training-frame mapping has been set up (i.e. every frame is a training
        frame).
        """
        if self._n_training_frames_per_camera is not None:
            return self._n_training_frames_per_camera
        return self.get_n_frames_per_camera(unique_sensors=True)

    def get_training_frame_idx_camera(self, unique_frame_idx: int) -> int:
        """Map an all-frame camera ``unique_frame_idx`` to a contiguous training-frame index.

        Returns ``unique_frame_idx`` unchanged (identity) when no training-frame
        mapping has been set up (every frame is a training frame).
        Returns ``-1`` for excluded frames (e.g. holdout validation frames) when
        a training-frame mapping is active.
        """
        if self._global_to_training_camera_frame_idx is None:
            return unique_frame_idx  # no excluded frames → identity mapping
        return int(self._global_to_training_camera_frame_idx[unique_frame_idx])

    def get_firing_lattice(
        self,
        sensor_id: str,
        fov: UnstructuredFov,
        max_frames: int = 64,
        n_time_modes: int = 1,
        n_bins_azimuth: Optional[int] = None,
        n_bins_elevation: Optional[int] = None,
    ) -> FiringLatticeEstimator:
        """Model-free per-sensor firing-lattice estimator (built + cached lazily).

        Reconstructs the sensor's fixed firing lattice by aggregating
        ``(azimuth, elevation) -> relative frame time`` over the sensor's frames
        (see :class:`FiringLatticeEstimator`). Used by the 'aggregate_lattice'
        unstructured raydrop mode to assign physically-consistent timestamps to
        virtual (dropped) rays without a sensor model.

        The lattice is a PER-SENSOR invariant (the firing pattern repeats every
        revolution), so it is built once and cached. ``max_frames`` frames are
        sampled uniformly across the sequence for the aggregation (more than
        enough to fill the lattice; the union over frames covers single-frame
        dropouts); it is set from
        ``dataset.lidar_unstructured_raydrop_lattice_max_frames`` (default 64) at
        both call sites, so the cache is consistent even though the cache key keys
        only on ``(sensor_id, n_time_modes)``. ``n_time_modes>1`` enables
        multi-cover support (distinct visit times per direction); ``1`` (default)
        is the single-cover mean lattice.
        The cache key includes ``n_time_modes`` so a later multi-mode request does
        not return a previously-cached single-mode lattice. ``n_bins_azimuth`` /
        ``n_bins_elevation`` (from
        ``dataset.lidar_unstructured_raydrop_lattice_n_bins_{azimuth,elevation}``,
        both None -> mode-appropriate built-in defaults) set the lookup-grid
        resolution; like ``max_frames`` they are passed consistently at both call
        sites and are not part of the cache key. The grid BOUNDS come from the
        caller's already-resolved ``fov`` (:class:`UnstructuredFov`, from the
        per-sensor config override or the sensor model) -- REQUIRED, so the grid
        always covers exactly the real sensor span with NO silent full-sphere /
        full-circle fallback.

        BUILD SITE: NCOREDataset.__init__ pre-builds this in the MAIN PROCESS
        (before the DataLoader forks) for the aggregate_lattice mode, so forked
        workers inherit the cached result read-only (copy-on-write) instead of each
        rebuilding it (~5s shard-I/O per worker) and stalling the first step. The
        lazy path here is the fallback (e.g. a code path that requests a lattice
        the main process did not pre-build).

        TODO(perf/aux-precompute, low priority): the build is ~5s (measured on the
        A/B P128 clip, 199 frames, max_frames=64; ~90% shard I/O, ~0.5s compute;
        coverage saturates by 64 frames). Now paid ONCE in the main process. A
        further optimization is an NuRec aux precompute step that serializes the small
        rel_t_grid / count_grid (+ meta) so it need not be rebuilt every run.
        """
        self._maybe_init_worker()

        # Check for cached instance
        cache_key = (sensor_id, n_time_modes)
        if (cached := self._firing_lattices.get(cache_key)) is not None:
            return cached

        lidar_sensor = self.lidar_sensors[sensor_id]

        # Grid BOUNDS come straight from the caller's resolved FOV (config override
        # or sensor model). No re-resolution and NO full-sphere/full-circle fallback
        # here: the FOV is resolved exactly once (NCOREDataset._resolve_unstructured_
        # lidar_fov) and threaded through so the lattice grid always matches the real
        # sensor span. A periodic (spinning) sensor's resolved az bounds are already
        # the full circle [-pi, pi]; a limited-FOV sensor's are its real [lo, hi].
        lattice = FiringLatticeEstimator.from_sensor(
            lidar_sensor,
            max_frames=max_frames,
            n_time_modes=n_time_modes,
            n_bins_azimuth=n_bins_azimuth,
            n_bins_elevation=n_bins_elevation,
            az_lo=fov.az_lo,
            az_hi=fov.az_hi,
            el_lo=fov.el_lo,
            el_hi=fov.el_hi,
        )

        # Remember the lattice for this sensor and mode so we don't rebuild it on subsequent calls
        self._firing_lattices[cache_key] = lattice

        return lattice

    def get_training_frame_idx_lidar(self, unique_frame_idx: int) -> int:
        """Map an all-frame lidar ``unique_frame_idx`` to a contiguous training-frame index.

        Returns ``unique_frame_idx`` unchanged (identity) when no training-frame
        mapping has been set up (every frame is a training frame).
        Returns ``-1`` for excluded frames (e.g. holdout validation frames) when
        a training-frame mapping is active.
        """
        if self._global_to_training_lidar_frame_idx is None:
            return unique_frame_idx  # no excluded frames → identity mapping
        return int(self._global_to_training_lidar_frame_idx[unique_frame_idx])

    def get_global_to_training_camera_frame_mapping(self) -> npt.NDArray[np.int32]:
        """Return the all-frame -> training-only camera frame index mapping as a concrete array.

        When a training-frame mapping is active, returns the array where excluded
        frames (e.g. holdout validation frames) map to ``-1``. When no training-
        frame mapping is configured (every frame is a training frame), returns
        an identity array ``[0, 1, ..., N-1]``.

        Unlike ``get_training_frame_idx_camera`` (which returns a scalar for a single
        lookup), this method returns the full array — suitable for batch consumers such
        as USD export that need to iterate over all frames at once.
        """
        if self._global_to_training_camera_frame_idx is not None:
            return self._global_to_training_camera_frame_idx
        total = sum(len(r) for r in self.camera_frame_ranges.values())
        return np.arange(total, dtype=np.int32)

    def get_global_to_training_lidar_frame_mapping(self) -> npt.NDArray[np.int32]:
        """Return the all-frame -> training-only lidar frame index mapping as a concrete array.

        When a training-frame mapping is active, returns the array where excluded
        frames (e.g. holdout validation frames) map to ``-1``. When no training-
        frame mapping is configured (every frame is a training frame), returns
        an identity array ``[0, 1, ..., N-1]``.
        """
        if self._global_to_training_lidar_frame_idx is not None:
            return self._global_to_training_lidar_frame_idx
        total = sum(len(r) for r in self.lidar_frame_ranges.values())
        return np.arange(total, dtype=np.int32)

    @staticmethod
    def _training_sensor_frame_indices(
        *,
        frame_range: range,
        linear_start_frame_idx: int,
        global_to_training_frame_mapping: npt.NDArray[np.int32],
    ) -> npt.NDArray[np.int64]:
        """Return frame-range values whose global entries are reachable during training."""
        sensor_frame_indices = np.fromiter(frame_range, dtype=np.int64)
        unique_frame_indices = linear_start_frame_idx + sensor_frame_indices - frame_range.start
        return sensor_frame_indices[global_to_training_frame_mapping[unique_frame_indices] >= 0]

    def get_training_camera_sensor_frame_indices(self, camera_id: str) -> npt.NDArray[np.int64]:
        """Return this camera's eligible raw frame indices in frame-range order.

        These are sensor-frame indices accepted by camera data-loading APIs, not datasource-global
        ``unique_frame_idx`` values or compact ``training_frame_idx`` values. The result may be empty.
        """
        return self._training_sensor_frame_indices(
            frame_range=self.camera_frame_ranges[camera_id],
            linear_start_frame_idx=self.camera_linear_start_frame_indices[camera_id],
            global_to_training_frame_mapping=self.get_global_to_training_camera_frame_mapping(),
        )

    def get_training_lidar_sensor_frame_indices(self, lidar_id: str) -> npt.NDArray[np.int64]:
        """Return this LiDAR's eligible raw frame indices in frame-range order.

        These are sensor-frame indices accepted by LiDAR data-loading APIs, not datasource-global
        ``unique_frame_idx`` values or compact ``training_frame_idx`` values. The result may be empty.
        """
        return self._training_sensor_frame_indices(
            frame_range=self.lidar_frame_ranges[lidar_id],
            linear_start_frame_idx=self.lidar_linear_start_frame_indices[lidar_id],
            global_to_training_frame_mapping=self.get_global_to_training_lidar_frame_mapping(),
        )

    def get_n_frames_per_camera(self, unique_sensors: bool = True) -> npt.NDArray[np.int32]:
        """Returns an array of total frame numbers per unique (unique_sensors=True) or logical (unique_sensors=False) camera sensor instance"""

        # make sure worker is initialized
        self._maybe_init_worker()

        if unique_sensors:
            n_frames_per_camera = np.zeros((len(self.all_camera_ids),), np.int32)

            for camera_id in self.all_camera_ids:
                camera_unique_idx = self.camera_unique_ids[camera_id].idx
                n_frames_per_camera[camera_unique_idx] += len(self.camera_frame_ranges[camera_id])

            return n_frames_per_camera
        else:
            return np.array(
                [len(self.camera_frame_ranges[camera_id]) for camera_id in self.all_camera_ids],
                dtype=np.int32,
            )

    def get_n_frames_per_lidar(self, unique_sensors: bool = True) -> npt.NDArray[np.int32]:
        """Returns an array of total frame numbers per unique (unique_sensors=True) or logical (unique_sensors=False) lidar sensor instance"""

        # make sure worker is initialized
        self._maybe_init_worker()

        if unique_sensors:
            n_frames_per_lidar = np.zeros((len(self.all_lidar_ids),), np.int32)

            for lidar_id in self.all_lidar_ids:
                lidar_unique_idx = self.lidar_unique_ids[lidar_id].idx
                n_frames_per_lidar[lidar_unique_idx] += len(self.lidar_frame_ranges[lidar_id])

            return n_frames_per_lidar
        else:
            return np.array(
                [len(self.lidar_frame_ranges[lidar_id]) for lidar_id in self.all_lidar_ids],
                dtype=np.int32,
            )

    def _sensor_ids_to_unique_ids(self, input_sensor_ids: list[str], sensor_type: str) -> Generator[str, None, None]:
        """Converts logical or unique sensor ids to the corresponding set of unique ids for a certain sensor type.
        Valid unique ids are provided as is, logical sensor ids are expanded to the corresponding unique ids.
        Raises KeyError if input sensor was not found"""

        match sensor_type:
            case "camera":
                sensor_unique_ids = self.camera_unique_ids
            case "lidar":
                sensor_unique_ids = self.lidar_unique_ids
            case _:
                raise ValueError(f"{self.__class__.__name__} unknown sensor type {sensor_type}")

        for input_sensor_id in input_sensor_ids:
            sensor_found = False
            if input_sensor_id in sensor_unique_ids:
                # input sensor id is a logical sensor name
                yield sensor_unique_ids[input_sensor_id].id
                sensor_found = True
            else:
                # input sensor id might be a unique name - look for it and return if found
                for unique_sensor_id in sensor_unique_ids.values():
                    if unique_sensor_id.id == input_sensor_id:
                        yield input_sensor_id
                        sensor_found = True
            if not sensor_found:
                raise KeyError(f"{self.__class__.__name__} unknown sensor id {input_sensor_id}")

    def get_dynamic_semantic_class_ids(
        self, sensor_type: list[str], dynamic_class_names: list[str]
    ) -> Optional[list[int]]:
        """Returns semantic class IDs for given dynamic class names and sensor type.

        Args:
            sensor_type: List of sensor types (e.g., ["camera"], ["lidar"], ["camera", "lidar"])
            dynamic_class_names: List of semantic class names to consider dynamic
                                 (e.g., ["person", "car", "bicycle"])

        Returns:
            List of semantic class IDs for dynamic classes, or None if semantic data unavailable
        """
        self._maybe_init_worker()

        semantic_classes_map = self.get_semantic_classes_map(
            camera_semantics="camera" in sensor_type,
            lidar_semantics="lidar" in sensor_type,
        )

        if semantic_classes_map is None:
            return None

        dynamic_class_ids = []
        for class_name in dynamic_class_names:
            if class_name in semantic_classes_map:
                dynamic_class_ids.append(semantic_classes_map[class_name])

        return dynamic_class_ids if dynamic_class_ids else None

    def get_track_split_metadata(self) -> TrackSplitMetadata | None:
        self._maybe_init_worker()
        self._compute_cuboid_tracks()
        return self.track_split_metadata

    def get_cuboid_tracks(
        self, dynamic_only: bool = False, world_frame: bool = False, include_generated: bool = False
    ) -> CuboidTracks:
        """Returns the cuboid tracks with optional filtering and transformation to world frame.

        Args:
            dynamic_only (bool, optional): if set return only tracks associated with dynamic objects. Defaults to False.
            world_frame (bool, optional): if set return tracks in world frame instead of NuRec frame. Defaults to False.
            include_generated (bool, optional): if set return generated tracks. Defaults to False.

        Returns:
            CuboidTracks: cuboid tracks after optional filtering and transformation
        """

        # make sure we are initialized
        self._maybe_init_worker()

        # tracks are required
        self._compute_cuboid_tracks()

        # select all or dynamic only subset to return
        cuboidtracks = self.cuboidtracks_dynamic if dynamic_only else self.cuboidtracks_all
        assert cuboidtracks is not None, "NCoreDataSource: CuboidTracks not initialized"

        cuboidtracks = CuboidTracks.Ops.transform_with_frame_conversion(
            cuboidtracks,
            self.world_to_nurec if not world_frame else None,
            np.identity(4, dtype=np.float32),
        )

        if include_generated:
            if self.generate_static_rigid:
                if self.generated_static_rigid_cuboidtracks is None:
                    self.generated_static_rigid_cuboidtracks = self._get_static_rigid_cuboid_tracks(
                        visibility_check_camera_ids=self.visibility_check_camera_ids,
                        max_visible_distance=self.max_visible_distance,
                        rigid_cls=self.static_rigid_classes,
                        point_cloud_path=self.static_rigid_point_cloud_path,
                    )
                # The static rigid cuboid tracks are generated in NuRec frame
                static_rigid_cuboidtracks = CuboidTracks.Ops.transform_with_frame_conversion(
                    self.generated_static_rigid_cuboidtracks,
                    self.world_to_nurec.inverse() if world_frame else None,
                    np.identity(4, dtype=np.float32),
                )
                cuboidtracks = CuboidTracks.Ops.concatenate(
                    [cuboidtracks, static_rigid_cuboidtracks],
                )

        return cuboidtracks

    def _get_static_rigid_cuboid_tracks(
        self,
        visibility_check_camera_ids: list[str],
        max_visible_distance: float,
        rigid_cls: list[str],
        point_cloud_path: Optional[str] = None,
    ) -> CuboidTracks:
        """Generate cuboid tracks for static rigid objects. The tracks will be in the NuRec frame.

        Supports two modes:
        - Lidar mode: uses lidar point clouds with semantic labels (original behavior)
        - Lidar-free mode: uses accumulated point cloud PLY with segment_id when lidar
          semantic segmentation is unavailable. Requires point_cloud_path to be set.
        """
        # make sure we are initialized
        self._maybe_init_worker()

        assert self.aux_loader is not None, "aux data was not loaded"
        use_lidar = self.aux_loader.has_lidar_semantic_segmentation()

        if use_lidar:
            stuff_classes = unpack_optional(self.get_semantic_classes_map(False, True))
        else:
            assert point_cloud_path is not None, (
                "point_cloud_path must be set in generate_static_rigid_cuboid_tracks config for lidar-free pipelines"
            )
            stuff_classes = unpack_optional(self.get_semantic_classes_map(True, False))
        assert stuff_classes is not None, "Semantic class mapping not available"
        label_idx_list = [stuff_classes[cls] for cls in rigid_cls]
        device = torch.device("cuda")

        # Get the camera frusta
        camera_frusta = [
            (camera_frustum.to(torch.device("cuda")), timestamp_us)
            for camera_id in visibility_check_camera_ids
            for camera_frustum, timestamp_us in unpack_optional(
                self.get_camera_frusta(camera_id=camera_id, far_plane_depth=max_visible_distance)
            )
        ]

        # For lidar-free: load the accumulated point cloud once and transform to NuRec frame
        ply_xyz_nurec = None
        ply_segment_id = None
        if not use_lidar:
            import point_cloud_utils as pcu

            ply_data = pcu.load_triangle_mesh(point_cloud_path).vertex_data
            ply_xyz_world = ply_data.positions
            ply_segment_id = ply_data.custom_attributes.get("segment_id")
            assert ply_segment_id is not None, f"PLY file {point_cloud_path} does not contain segment_id attribute"
            assert ply_segment_id.min() >= 0 and ply_segment_id.max() <= 255, (
                f"ply_segment_id values out of uint8 range: min={ply_segment_id.min()}, max={ply_segment_id.max()}"
            )
            ply_segment_id = ply_segment_id.astype(np.uint8)
            ply_xyz_nurec = self.world_to_nurec.transform_points(ply_xyz_world)

        all_tracks_ids = []
        all_tracks_poses = []
        all_tracks_timestamps_us = []
        all_tracks_label_class = []
        all_tracks_flags = []
        all_cuboid_dims = []

        for idx, label_idx in enumerate(label_idx_list):
            rigid_cls_name = rigid_cls[idx]

            if use_lidar:
                # Original lidar path
                pcs = [
                    pc
                    for pc in unpack_optional(
                        self.get_lidar_point_clouds(
                            lidar_ids=self.all_lidar_ids,
                            valid_points_only=True,
                            non_dynamic_points_only=True,
                            color_type=None,
                            step_frame=1,
                            device=device,
                            visualize=False,
                            semantic_colormap=None,
                            lidar_point_cloud_selected_classes=[label_idx],
                        )
                    )
                ]
                if len(pcs) == 0:
                    continue
                point_cloud_xyz = PointCloud.collate_fn(pcs, device=device).xyz_end
            else:
                # Lidar-free path: filter accumulated point cloud by segment_id
                assert ply_xyz_nurec is not None and ply_segment_id is not None
                class_mask = ply_segment_id == label_idx
                if not np.any(class_mask):
                    continue
                point_cloud_xyz = torch.from_numpy(ply_xyz_nurec[class_mask].astype(np.float32)).to(device)

            if point_cloud_xyz.numel() == 0:
                continue

            # Use DBSCAN to cluster the points
            pc_pos = point_cloud_xyz.cpu().numpy()

            # cuml and the RAPIDS libraries under it find the CUDA maths libraries through an
            # RPATH into site-packages/nvidia, which only the x86_64 nvidia-*-cu12 wheels provide.
            # On aarch64 torch bundles them itself, so load those. RTLD_GLOBAL covers the chain.
            if platform.machine() == "aarch64":
                torch_lib = Path(torch.__file__).parent / "lib"
                for lib in sorted(torch_lib.glob("libcu*.so.*")) + sorted(torch_lib.glob("libnv*.so.*")):
                    ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)

            # Use cuML DBSCAN (gpu acceleration)
            from cuml.cluster import DBSCAN

            dbscan = DBSCAN(eps=0.5, min_samples=50, metric="euclidean")
            dtype = "int64" if pc_pos.shape[0] >= 46340 else "int32"
            cluster_labels = dbscan.fit_predict(pc_pos, out_dtype=dtype)

            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

            assert_default_device_on_local_rank()

            for cluster_idx in range(n_clusters):
                track_id = f"generated_{rigid_cls_name}_{cluster_idx}"
                cluster_mask = cluster_labels == cluster_idx
                cluster_points = point_cloud_xyz[cluster_mask]
                # Check if the point is visible from any of the cameras
                timestamp_list_us = []
                for camera_frustum, timestamp_us in camera_frusta:
                    # if 20% of the points are in the frustum, then consider it visible
                    if camera_frustum.points_in_frustum(cluster_points).sum() > 0.2 * cluster_points.shape[0]:
                        timestamp_list_us.append(timestamp_us)

                if len(timestamp_list_us) == 0:
                    continue
                # Determine the AABB track cuboid in NuRec frame and start and end timestamp to the traffic light
                start_timestamp_us = min(timestamp_list_us)
                end_timestamp_us = max(timestamp_list_us)
                track_centroid = cluster_points.mean(dim=0).cpu().numpy()
                track_pose = np.identity(4)
                track_pose[:3, 3] = track_centroid
                track_dims = cluster_points.max(dim=0).values - cluster_points.min(dim=0).values
                all_tracks_ids.append(track_id)
                all_tracks_poses.append(np.array([track_pose, track_pose], dtype=np.float32))
                all_tracks_timestamps_us.append(np.array([start_timestamp_us, end_timestamp_us], dtype=np.int64))
                all_tracks_label_class.append(f"generated {rigid_cls_name}")
                all_tracks_flags.append(TrackFlags.NONE | TrackFlags.DYNAMIC)
                all_cuboid_dims.append(track_dims.cpu().numpy())

        return CuboidTracks.Factory.from_numpy(
            all_tracks_ids,
            all_tracks_poses,
            all_tracks_timestamps_us,
            all_tracks_label_class,
            all_tracks_flags,
            cuboids_dims=all_cuboid_dims,
        )

    def get_track_point_clouds(
        self,
        cuboid_tracks: CuboidTracks,
        cuboid_dim_scale_factor: float = 1.0,
        lidar_ids: Optional[list[str]] = None,
        camera_ids: Optional[list[str]] = None,
        return_color: bool = False,
        step_frame: int = 1,
        keep_all_track_poses: bool = False,
        device: torch.device = torch.device("cuda"),
    ) -> Generator[TrackPointCloud, None, None]:
        """Returns a generator for all object point-clouds available for point-cloud sensor (lidar / camera), transformed into NuRec frame.
        The returned value is a TrackPointCloud.

        Point-cloud sensor are specified by either logical or unique sensor IDs.

        The provided cuboid tracks should be in the original world coordinates.

        Defaults to first logical data-set specific point-cloud sensor if no dedicated sensors are specified
        (raises error if unsupported sensors are specified).

        Default point-cloud sensor: *first* logical lidar
        """

        if camera_ids is not None:
            lidar_ids = [] if lidar_ids is None else list(lidar_ids)
        elif lidar_ids is not None:
            camera_ids = []
        else:
            # default to available lidar sensor if not provided explicitly, or to the first available camera sensor
            # if lidar is not available
            if self.all_lidar_ids:
                lidar_ids = [self.all_lidar_ids[0]]
                camera_ids = []
            else:
                lidar_ids = []
                camera_ids = [self.all_camera_ids[0]]

        # we can produce point clouds for cameras
        for pc in self.get_camera_track_point_clouds(
            camera_ids,
            cuboid_tracks,
            cuboid_dim_scale_factor,
            return_color,
            step_frame,
            keep_all_track_poses,
            device,
        ):
            yield pc

        # we can produce point clouds for lidars
        for pc in self.get_lidar_track_point_clouds(
            lidar_ids,
            cuboid_tracks,
            cuboid_dim_scale_factor,
            return_color,
            step_frame,
            keep_all_track_poses,
            device,
        ):
            yield pc

    def get_lidar_track_point_clouds(
        self,
        lidar_ids: list[str],
        cuboid_tracks: CuboidTracks,
        cuboid_dim_scale_factor: float = 1.0,
        return_color: bool = False,
        step_frame: int = 1,
        keep_all_track_poses: bool = False,
        device: torch.device = torch.device("cuda"),
    ) -> List[TrackPointCloud]:
        """Returns a generator for all object point-clouds available for lidar sensor, transformed into NuRec frame.
        The returned value is a TrackPointCloud.

        Point-cloud sensor are specified by either logical or unique sensor IDs.

        The provided cuboid tracks should be in the original world coordinates.

        Defaults to first logical data-set specific point-cloud sensor if no dedicated sensors are specified
        (raises error if unsupported sensors are specified).

        Default point-cloud sensor: *first* logical lidar
        """

        # make sure we are initialized
        self._maybe_init_worker()

        if str(device) != "cpu":
            assert self.worker_id is None, "Loading the lidar point clouds on the GPU must be done on the main process"
            # verify that the current GPU is correct
            assert_device_on_local_rank(device)

        track_point_cloud_list: list[TrackPointCloud] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            for unique_lidar_id in self._sensor_ids_to_unique_ids(lidar_ids, "lidar"):
                for lidar_id in self.lidar_sensors.keys():
                    if self.lidar_unique_ids[lidar_id].id != unique_lidar_id:
                        continue

                    lidar_frame_range = self.lidar_frame_ranges[lidar_id][::step_frame]
                    track_point_cloud_results: list[None | List[TrackPointCloud]] = [None] * len(lidar_frame_range)

                    def load_lidar_track_point_cloud(result_index: int, lidar_frame_index: int, lidar_id: str) -> None:
                        set_default_device()

                        lidar_sensor = self.lidar_sensors[lidar_id]
                        return_valid = to_torch(
                            lidar_sensor.get_frame_ray_bundle_return_valid_mask(lidar_frame_index, return_index=0),
                            device=device,
                            non_blocking=True,
                        )

                        # First check if any of the cuboids exist in the current frame's timestamp range
                        xyz_time_start = lidar_sensor.get_frame_timestamp_us(
                            lidar_frame_index, frame_timepoint=ncore.data.FrameTimepoint.START
                        )
                        xyz_time_end = lidar_sensor.get_frame_timestamp_us(
                            lidar_frame_index, frame_timepoint=ncore.data.FrameTimepoint.END
                        )

                        if not keep_all_track_poses:
                            current_cuboid_idxs = (
                                torch.logical_and(
                                    cuboid_tracks.tracks_timestamps_us >= xyz_time_start,
                                    cuboid_tracks.tracks_timestamps_us <= xyz_time_end,
                                )
                                .nonzero()
                                .squeeze()
                            )
                        else:
                            current_cuboid_idxs = (cuboid_tracks.tracks_timestamps_us >= 0).nonzero().squeeze()

                        if current_cuboid_idxs.dim() == 0:
                            current_cuboid_idxs = current_cuboid_idxs.unsqueeze(0).unsqueeze(1)

                        if current_cuboid_idxs.numel() == 0:
                            return

                        # Get existed cuboid poses and dimensions in the current frame
                        current_cuboid_jidx = (
                            torch.searchsorted(cuboid_tracks.tracks_packinfo[:, 0], current_cuboid_idxs, right=True) - 1
                        )
                        cuboid_poses = cuboid_tracks.tracks_poses[current_cuboid_idxs].data
                        cuboid_poses = tquat_to_se3_matrix(cuboid_poses, unbatch=False)
                        cuboid_dim = cuboid_tracks.cuboids_dims[current_cuboid_jidx] * cuboid_dim_scale_factor

                        frame_T_sensor_world = lidar_sensor.get_frames_T_sensor_target("world", lidar_frame_index)

                        # load motion-compensated point clouds (represented in sensor-frame)
                        pc = lidar_sensor.get_frame_point_cloud(
                            frame_index=lidar_frame_index,
                            motion_compensation=True,
                            with_start_points=True,
                            return_index=0,  # closest returns only
                        )
                        xyz_s = to_torch(unpack_optional(pc.xyz_m_start), device=device, non_blocking=True)[
                            return_valid
                        ]
                        xyz_e = to_torch(pc.xyz_m_end, device=device, non_blocking=True)[return_valid]

                        T_sensor_nurec = to_torch(
                            self.world_to_nurec.transform_poses(frame_T_sensor_world), device=device, non_blocking=True
                        )

                        if device.type == "cuda":
                            torch.cuda.synchronize()

                        # color the point clouds
                        if return_color:
                            color, color_mask, scale = self._project_lidar_camera_rgb(
                                xyz_sensor=xyz_e,
                                frame_T_sensor_world=frame_T_sensor_world,
                                lidar_frame_timestamp=xyz_time_end,
                                device=device,
                                mode=_CameraRGBMode.EGO_VALID_PIXELS,
                            )

                            color = color[color_mask]
                            scale = scale[color_mask]
                            xyz_s = xyz_s[color_mask]
                            xyz_e = xyz_e[color_mask]

                        xyz_s = (
                            (self.world_to_nurec.target_scale * T_sensor_nurec[:3, :3]) @ xyz_s.T
                            + T_sensor_nurec[:3, 3:4]
                        ).T
                        xyz_e = (
                            (self.world_to_nurec.target_scale * T_sensor_nurec[:3, :3]) @ xyz_e.T
                            + T_sensor_nurec[:3, 3:4]
                        ).T

                        # Spawn the points within each cuboid (in local coordinates)
                        ret = []
                        for cuboid_idx in range(cuboid_poses.shape[0]):
                            cuboid_T = cuboid_poses[cuboid_idx, :, :].to(device=device)
                            cuboid_extent = cuboid_dim[cuboid_idx, :3].to(device=device)

                            xyz_e_local = (cuboid_T[:3, :3].T @ (xyz_e.T - cuboid_T[:3, 3:4])).T
                            xyz_s_local = (cuboid_T[:3, :3].T @ (xyz_s.T - cuboid_T[:3, 3:4])).T
                            xyz_mask = torch.all(torch.abs(xyz_e_local) < cuboid_extent / 2, dim=1)

                            ret.append(
                                TrackPointCloud(
                                    track_id=cuboid_tracks.tracks_id[current_cuboid_jidx[cuboid_idx]],
                                    point_cloud=PointCloud(
                                        xyz_start=xyz_s_local[xyz_mask],
                                        xyz_end=xyz_e_local[xyz_mask],
                                        flags=None,
                                        color=color[xyz_mask] if return_color else None,
                                        semantic_class_id=None,
                                        camera_footprint_scale=scale[xyz_mask] if return_color else None,
                                    ),
                                )
                            )

                        track_point_cloud_results[result_index] = ret

                    for future in tqdm.tqdm(
                        concurrent.futures.as_completed(
                            [
                                executor.submit(load_lidar_track_point_cloud, index, lidar_frame_range[index], lidar_id)
                                for index in range(len(lidar_frame_range))
                            ]
                        ),
                        desc="Get Lidar Track Point Clouds",
                        total=len(lidar_frame_range),
                        disable=self.tqdm_disabled,
                    ):
                        try:
                            future.result()
                        except BaseException as exc:
                            raise exc  # Exceptions in threads need to be re-raised to be visible.

                    for track_point_cloud in track_point_cloud_results:
                        if track_point_cloud is not None and len(track_point_cloud) > 0:
                            track_point_cloud_list.extend(track_point_cloud)  # type: ignore

        return track_point_cloud_list

    def get_camera_track_point_clouds(
        self,
        camera_ids: list[str],
        cuboid_tracks: CuboidTracks,
        cuboid_dim_scale_factor: float = 1.0,
        return_color: bool = True,
        step_frame: int = 1,
        keep_all_track_poses: bool = False,
        device: torch.device = torch.device("cuda"),
    ) -> Generator[TrackPointCloud, None, None]:
        """Returns a generator for all object point-clouds available for camera sensor, transformed into NuRec frame.
        The returned value is a TrackPointCloud.

        Use camera depth information to replace lidar to capture point clouds in each cuboid

        Point-cloud sensor are specified by either logical or unique sensor IDs.

        The provided cuboid tracks should be in the original world coordinates.

        Defaults to first logical data-set specific point-cloud sensor if no dedicated sensors are specified
        (raises error if unsupported sensors are specified).

        Default point-cloud sensor: *first* logical camera

        Args:
            cuboid_tracks: Cuboid tracks in world coordinates
            cuboid_dim_scale_factor: Cuboid dimension scale factor
            camera_ids: List of camera IDs to use
            step_frame: Frame sampling step
            device: Computation device

        Returns:
            Generator for TrackPointCloud, containing point clouds in cuboid local coordinates
        """
        # make sure we are initialized
        self._maybe_init_worker()

        # provided samples are ordered by sensors, then chunks
        for unique_camera_id in self._sensor_ids_to_unique_ids(camera_ids, "camera"):
            aux_loader = self.aux_loader
            assert aux_loader is not None, f"{self.__class__.__name__}: aux data was not loaded"

            for camera_id in self.camera_sensors.keys():
                if self.camera_unique_ids[camera_id].id != unique_camera_id:
                    continue

                if not aux_loader.has_depth(camera_id):
                    logger.warning(
                        f"{self.__class__.__name__}: No depth data available for camera '{camera_id}', "
                        "skipping it for track point cloud initialization. RGB supervision for this camera is unaffected."
                    )
                    continue

                camera_sensor = self.camera_sensors[camera_id]
                camera_params = camera_sensor.model_parameters
                frame_width, frame_height = camera_params.resolution

                all_pixels = to_torch(self.cameras_all_pixels[camera_id], device=device)
                all_camera_rays = to_torch(
                    self.cameras_all_rays[camera_id].reshape((-1, 3)),
                    device=device,
                )
                all_camera_footprints = to_torch(
                    self.cameras_all_footprints[camera_id].reshape(-1),
                    device=device,
                )

                for camera_frame_index in self.camera_frame_ranges[camera_id][::step_frame]:
                    # First check if any of the cuboids exist in the current frame's timestamp range
                    camera_frame_timestamp_start, camera_frame_timestamp_end = camera_sensor.frames_timestamps_us[
                        camera_frame_index
                    ]

                    if not keep_all_track_poses:
                        current_cuboid_idxs = (
                            torch.logical_and(
                                cuboid_tracks.tracks_timestamps_us >= camera_frame_timestamp_start,
                                cuboid_tracks.tracks_timestamps_us <= camera_frame_timestamp_end,
                            )
                            .nonzero()
                            .squeeze()
                        )
                    else:
                        current_cuboid_idxs = (cuboid_tracks.tracks_timestamps_us >= 0).nonzero().squeeze()

                    if current_cuboid_idxs.dim() == 0:
                        current_cuboid_idxs = current_cuboid_idxs.unsqueeze(0).unsqueeze(1)

                    if current_cuboid_idxs.numel() == 0:
                        continue

                    # Get existed cuboid poses and dimensions in the current frame
                    current_cuboid_jidx = (
                        torch.searchsorted(cuboid_tracks.tracks_packinfo[:, 0], current_cuboid_idxs, right=True) - 1
                    )
                    cuboid_poses = cuboid_tracks.tracks_poses[current_cuboid_idxs].data
                    cuboid_poses = tquat_to_se3_matrix(cuboid_poses, unbatch=False).to(device)
                    cuboid_dim = cuboid_tracks.cuboids_dims[current_cuboid_jidx] * cuboid_dim_scale_factor

                    depth = to_torch(
                        aux_loader.get_depth(
                            camera_id, camera_frame_timestamp_end, (frame_width, frame_height)
                        ).reshape(-1)
                        * self.world_to_nurec.target_scale,
                        device=device,
                    )

                    valid_depth_mask = depth > 0
                    if not valid_depth_mask.any():
                        continue

                    depth = depth[valid_depth_mask]
                    pixels = all_pixels[valid_depth_mask]
                    camera_rays = all_camera_rays[valid_depth_mask]
                    footprints = all_camera_footprints[valid_depth_mask]

                    if return_color:
                        color = to_torch(
                            camera_sensor.get_frame_image_array(camera_frame_index).reshape((-1, 3)), device=device
                        )
                        color = color[valid_depth_mask]

                    assert is_metric_z_depth_method(aux_loader.get_depth_meta(camera_id)["method"]), (
                        f"{self.__class__.__name__}: assuming z-axis metric depth"
                    )

                    distance = torch.abs(depth / camera_rays[:, 2])
                    scale = footprints * distance

                    T_sensor_nurec_startend = self.world_to_nurec.transform_poses(
                        camera_sensor.get_frames_T_sensor_target(
                            target_node="world",
                            frame_indices=camera_frame_index,
                            frame_timepoint=None,  # evaluates both start/end timepoints
                        )  # 2x4x4
                    )

                    if str(device) == "cpu":
                        camera_model = self.camera_models[camera_id]
                        world_rays = camera_model.pixels_to_world_rays_shutter_pose(
                            pixels,
                            T_sensor_nurec_startend[0],
                            T_sensor_nurec_startend[1],
                        ).world_rays
                    else:
                        camera_model = ncore.sensors.camera_model_from_parameters(camera_params, device=str(device))
                        world_rays = image_points_to_world_rays_shutter_pose(
                            camera_params,
                            camera_model.pixels_to_image_points(pixels),
                            se3_matrix_to_tquat(T_sensor_nurec_startend),
                            torch.LongTensor([camera_frame_timestamp_start, camera_frame_timestamp_end]),
                        ).world_rays

                    # NuRec space
                    xyz_s = world_rays[:, :3]
                    xyz_e = world_rays[:, :3] + world_rays[:, 3:] * distance.unsqueeze(-1)

                    # Spawn the points within each cuboid (in local coordinates)
                    for cuboid_idx in range(cuboid_poses.shape[0]):
                        cuboid_T = cuboid_poses[cuboid_idx, :, :]
                        cuboid_extent = cuboid_dim[cuboid_idx, :3]

                        xyz_e_local = (cuboid_T[:3, :3].T @ (xyz_e.T - cuboid_T[:3, 3:4])).T
                        xyz_s_local = (cuboid_T[:3, :3].T @ (xyz_s.T - cuboid_T[:3, 3:4])).T
                        xyz_mask = torch.all(torch.abs(xyz_e_local) < cuboid_extent / 2, dim=1)

                        yield TrackPointCloud(
                            track_id=cuboid_tracks.tracks_id[current_cuboid_jidx[cuboid_idx]],
                            point_cloud=PointCloud(
                                xyz_start=xyz_s_local[xyz_mask],
                                xyz_end=xyz_e_local[xyz_mask],
                                flags=None,
                                color=color[xyz_mask] if return_color else None,
                                semantic_class_id=None,
                                camera_footprint_scale=scale[xyz_mask],
                            ),
                        )

    def decode_image_cpu(
        self, image_handle: ncore.data.EncodedImageHandle, subsample_factor: float | None = None
    ) -> np.ndarray:
        """Decode encoded image data on CPU using enabled decoding backend and optionally subsample
        the image by the given factor (if possible while decoding)"""

        # load the encoded image data
        image_data = image_handle.get_data()

        # use simplejpeg to decode the image data if possible (wraps libjpeg-turbo)
        if self.jpeg_backend_cpu == "simplejpeg" and image_data.get_encoded_image_format().lower() in (
            "jpeg",
            "jpg",
        ):
            # decode jpeg header to obtain source resolution
            height, width, _, _ = simplejpeg.decode_jpeg_header(encoded_image := image_data.get_encoded_image_data())

            if subsample_factor is not None:
                # resize if requested and required
                height, width = int(height), int(width)
                assert height % subsample_factor == 0 and width % subsample_factor == 0, (
                    f"Image dimensions ({height}x{width}) must be divisible by subsample_factor ({subsample_factor})"
                )
                height = height // subsample_factor
                width = width // subsample_factor

            image = simplejpeg.decode_jpeg(
                encoded_image,
                fastdct=self.simplejpeg_fastdct,
                fastupsample=self.simplejpeg_fastupsample,
                min_width=width,
                min_height=height,
            )
        else:
            # fall back to ncore-internal (PIL-based) decoding and optional subsampling
            pil_image = image_data.get_decoded_image()
            if subsample_factor is not None and subsample_factor != 1.0:
                # resize if requested and required
                new_size = (
                    int(pil_image.width // subsample_factor),
                    int(pil_image.height // subsample_factor),
                )
                pil_image = pil_image.resize(new_size, resample=PILImage.Resampling.LANCZOS)
            image = np.asarray(pil_image)

        return image

    def _project_lidar_camera_rgb(
        self,
        xyz_sensor: torch.Tensor,
        frame_T_sensor_world: npt.NDArray,
        lidar_frame_timestamp: int,
        device: torch.device = torch.device("cuda"),
        mode: _CameraRGBMode = _CameraRGBMode.ALL_PIXELS,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project camera RGB onto LiDAR points.

        ``ALL_PIXELS`` preserves the default behavior: every geometrically
        valid projection is eligible and a later camera overwrites an earlier
        one. ``FRAME_VALID_PIXELS`` applies each camera's per-frame mask, while
        ``EGO_VALID_PIXELS`` applies only its camera-housing mask. The masked
        modes select the valid observation with the smallest physical pixel
        footprint when cameras overlap. RGB and footprint scale always come
        from the same observation.
        """

        # Transform the point cloud to the world coordinate frame
        xyz_world = ncore_transformations.transform_point_cloud(
            xyz_sensor.unsqueeze(0), to_torch(frame_T_sensor_world, device=device).unsqueeze(0)
        ).squeeze(0)
        assert len(xyz_world.shape) == 2 and xyz_world.shape[1] == 3

        color = torch.zeros_like(xyz_world, dtype=torch.uint8)
        scale = torch.zeros_like(xyz_world[:, 0])
        color_mask = torch.zeros_like(xyz_world[:, 0], dtype=torch.bool)

        if not self.camera_sensors:
            raise ValueError(
                "[NCOREDataSource] At least a single camera needs to be available for point-cloud coloring"
            )

        for camera_id, camera_sensor in self.camera_sensors.items():
            # Find the closest camera frame
            cam_frame_index = camera_sensor.get_closest_frame_index(lidar_frame_timestamp)
            if mode is _CameraRGBMode.FRAME_VALID_PIXELS and cam_frame_index not in self.camera_frame_ranges[camera_id]:
                # Frame-valid masks exist only for the configured camera range.
                continue

            # Decode image
            img_frame = to_torch(self.decode_image_cpu(camera_sensor.get_frame_handle(cam_frame_index)), device="cpu")

            T_world_sensor = camera_sensor.get_frames_T_source_sensor(
                source_node="world",
                frame_indices=cam_frame_index,
                frame_timepoint=None,  # evaluates both start/end timepoints
            )  # 2x4x4

            T_sensor_world_end = ncore_transformations.se3_inverse(T_world_sensor[1])

            # Project pc to image points
            if str(device) == "cpu":
                camera_model = self.camera_models[camera_id]
                world_point_projections = camera_model.world_points_to_image_points_shutter_pose(
                    xyz_world,
                    T_world_sensor[0],
                    T_world_sensor[1],
                    return_valid_indices=True,
                )
            else:
                camera_params = self.camera_model_parameters[camera_id]
                world_point_projections = world_points_to_image_points_shutter_pose(
                    camera_params,
                    xyz_world,
                    se3_matrix_to_tquat(T_world_sensor),
                    torch.LongTensor(camera_sensor.frames_timestamps_us[cam_frame_index, :]),
                )

            pixels = world_point_projections.image_points.floor().int().cpu()
            valid_idx = unpack_optional(world_point_projections.valid_indices).to(device)  # torch

            if mode in (_CameraRGBMode.EGO_VALID_PIXELS, _CameraRGBMode.FRAME_VALID_PIXELS):
                if mode is _CameraRGBMode.EGO_VALID_PIXELS:
                    valid_pixels_mask = self.cameras_valid_pixels_ego_masks[camera_id].unpacked()
                else:
                    valid_pixels_mask = self.cameras_frame_valid_pixels_masks[camera_id][cam_frame_index].unpacked()
                valid_projection_mask = to_torch(
                    valid_pixels_mask[pixels[:, 1].numpy(), pixels[:, 0].numpy()],
                    device=device,
                    dtype=torch.bool,
                )
                pixels = pixels[valid_projection_mask.cpu()]
                valid_idx = valid_idx[valid_projection_mask]
                if valid_idx.numel() == 0:
                    continue

            point_colors = img_frame[pixels[:, 1], pixels[:, 0]]

            # We could use the T_world_sensors that camera_model.world_points_to_image_points_shutter_pose to compensate
            # for rolling shutter instead of using T_sensor_world_end but it will be slower and that level of accuracy is not important here.
            footprints = to_torch(self.cameras_all_footprints[camera_id], device="cpu")
            offset = (xyz_world[valid_idx] - torch.FloatTensor(T_sensor_world_end[:3, 3]).to(device)).norm(dim=-1)
            candidate_scale = footprints[pixels[:, 1], pixels[:, 0]].to(device) * offset
            if mode in (_CameraRGBMode.EGO_VALID_PIXELS, _CameraRGBMode.FRAME_VALID_PIXELS):
                # A smaller footprint is a finer spatial observation. Strict
                # comparison keeps datasource order as the deterministic tie-break.
                use_observation = torch.logical_or(~color_mask[valid_idx], candidate_scale < scale[valid_idx])
                selected_idx = valid_idx[use_observation]
                color[selected_idx] = point_colors.to(device)[use_observation]
                scale[selected_idx] = candidate_scale[use_observation]
                color_mask[selected_idx] = True
            else:
                # Preserve the established camera-rgb overlap policy.
                color[valid_idx] = point_colors.to(device)
                scale[valid_idx] = candidate_scale
                color_mask[valid_idx] = True

        return color, color_mask, scale

    def _color_lidar_point_cloud_from_cameras(
        self,
        point_cloud: PointCloud,
        frame_T_sensor_world: npt.NDArray,
        lidar_frame_timestamp: int,
        device: torch.device,
        mode: _CameraRGBMode,
    ) -> PointCloud:
        """Apply a camera-RGB policy while preserving per-point correspondence.

        The compatibility mode returns only points with a camera observation.
        The frame-valid mode retains every LiDAR point and communicates RGB
        validity through ``RayFlags.RGB_LABEL``. XYZ, color, semantics,
        intensity, scale, and flags retain the same point ordering.
        """

        color, color_valid_mask, scale = self._project_lidar_camera_rgb(
            point_cloud.xyz_end,
            frame_T_sensor_world,
            lidar_frame_timestamp,
            device,
            mode,
        )
        flags = unpack_optional(point_cloud.flags).clone()
        if mode is _CameraRGBMode.FRAME_VALID_PIXELS:
            flags[color_valid_mask] |= RayFlags.RGB_LABEL.value
        colored_point_cloud = replace(
            point_cloud,
            color=color,
            flags=flags,
            camera_footprint_scale=scale,
        )
        return colored_point_cloud[color_valid_mask] if mode is _CameraRGBMode.ALL_PIXELS else colored_point_cloud

    def _color_lidar_point_cloud(
        self,
        point_cloud: PointCloud,
        color_type: PointCloudColorType,
        lidar_sensor: ncore.data.LidarSensorProtocol,
        lidar_frame_index: int,
        point_filter: slice | torch.Tensor,
        frame_T_sensor_world: npt.NDArray,
        lidar_timestamp: int,
        semantic_colormap: Optional[np.ndarray],
        ignore_label_class: Optional[int],
        device: torch.device,
    ) -> PointCloud:
        """Apply the color strategy while preserving each point field's ordering."""

        match color_type:
            case "camera-rgb":
                return self._color_lidar_point_cloud_from_cameras(
                    point_cloud,
                    frame_T_sensor_world,
                    lidar_timestamp,
                    device,
                    _CameraRGBMode.ALL_PIXELS,
                )
            case "camera-rgb-frame-valid-pixels":
                return self._color_lidar_point_cloud_from_cameras(
                    point_cloud,
                    frame_T_sensor_world,
                    lidar_timestamp,
                    device,
                    _CameraRGBMode.FRAME_VALID_PIXELS,
                )
            case "generic-data":
                return replace(
                    point_cloud,
                    color=to_torch(
                        lidar_sensor.get_frame_generic_data(lidar_frame_index, "rgb"),
                        device=device,
                    )[point_filter],
                )
            case "semantics":
                return replace(
                    point_cloud,
                    color=color_pc_semantics(
                        unpack_optional(point_cloud.semantic_class_id),
                        to_torch(unpack_optional(semantic_colormap), device=device),
                        unpack_optional(ignore_label_class),
                    ),
                )
            case None:
                return point_cloud

    def _transform_lidar_point_cloud_to_nurec(
        self,
        point_cloud: PointCloud,
        T_sensor_nurec: torch.Tensor,
    ) -> PointCloud:
        """Transform both motion-compensated positions into the NuRec frame."""

        rotation = self.world_to_nurec.target_scale * T_sensor_nurec[:3, :3]
        translation = T_sensor_nurec[:3, 3:4]
        return replace(
            point_cloud,
            xyz_start=(rotation @ point_cloud.xyz_start.T + translation).T,
            xyz_end=(rotation @ point_cloud.xyz_end.T + translation).T,
        )

    def _apply_lidar_semantics(
        self,
        point_cloud: PointCloud,
        lidar_id: str,
        valid_points_only: bool,
        selected_classes: Optional[list[int]],
        ignore_label_class: int,
        device: torch.device,
    ) -> PointCloud:
        """Apply semantics while preserving correspondence among all per-point fields."""

        semantic_class_id = unpack_optional(point_cloud.semantic_class_id)
        if valid_points_only:
            point_cloud = point_cloud[semantic_class_id != ignore_label_class]
        else:
            flags = unpack_optional(point_cloud.flags).clone()
            flags[semantic_class_id == ignore_label_class] |= RayFlags.INVALID
            point_cloud = replace(point_cloud, flags=flags)

        if selected_classes is not None:
            semantic_class_id = unpack_optional(point_cloud.semantic_class_id)
            point_cloud = point_cloud[torch.isin(semantic_class_id, torch.tensor(selected_classes, device=device))]

        semantic_class_id = unpack_optional(point_cloud.semantic_class_id)
        flags = unpack_optional(point_cloud.flags).clone()
        flags |= RayFlags.VALID_SEMANTIC.value
        flags[semantic_class_id == self.sensor_road_class_ids[lidar_id]] |= RayFlags.ROAD_SEMANTIC
        flags[semantic_class_id == self.sensor_sky_class_ids[lidar_id]] |= RayFlags.SKY_SEMANTIC
        for vehicle_class_id in self.sensor_vehicle_classes_ids[lidar_id]:
            flags[semantic_class_id == vehicle_class_id] |= RayFlags.VEHICLE_SEMANTIC
        return replace(point_cloud, flags=flags)

    def get_point_clouds(
        self,
        device: torch.device,
        lidar_ids: Optional[list[str]] = None,
        camera_ids: Optional[list[str]] = None,
        valid_points_only: bool = True,
        non_dynamic_points_only: bool = True,
        color_type: PointCloudColorType = None,
        step_frame: int = 1,
        visualize: bool = False,
        force: bool = True,
    ) -> Generator[PointCloud, None, None]:
        """Returns a generator for all point-clouds available for point-cloud sensor (lidar / camera with aux-depth), transformed into NuRec frame.

        Point-cloud sensor are specified by either logical or unique sensor IDs.

        Defaults to first logical data-set specific point-cloud sensor if no dedicated sensors are specified
        (raises error if unsupported sensors are specified).

        Can be parameterized to only return valid (default), non-dynamic (default),
        and colored points colorized by one of the following strategies:
        "camera-rgb" (RGB scene colors), "camera-rgb-frame-valid-pixels" (RGB
        from frame-valid camera pixels while retaining LiDAR geometry), or
        "semantics" (semantic colors obtained from shard metadata).

        If forced, at least a single point-cloud providing sensor needs to be available (otherwise result can be empty).

        Default point-cloud sensor: *first* logical lidar
        """

        # make sure we are initialized
        self._maybe_init_worker()

        if camera_ids is not None:
            lidar_ids = [] if lidar_ids is None else lidar_ids
        elif lidar_ids is not None:
            camera_ids = []
        else:
            # default to available lidar sensor if not provided explicitly, or to the first available camera sensor
            # if lidar is not available
            if self.all_lidar_ids:
                lidar_ids = [self.all_lidar_ids[0]]
                camera_ids = []
            else:
                lidar_ids = []
                camera_ids = [self.all_camera_ids[0]]

        if force and (not len(lidar_ids) and not len(camera_ids)):
            # require at least a single point-cloud providing sensor in forced mode
            raise ValueError(
                f"{self.__class__.__name__}: At least a single sensor required for forced point cloud generation"
            )

        if len(camera_ids) and color_type == "camera-rgb-frame-valid-pixels":
            raise ValueError('"camera-rgb-frame-valid-pixels" is only supported for LiDAR point clouds')

        if non_dynamic_points_only and len(lidar_ids):
            # compute dynamic point masks required to filter for dynamic points
            self._compute_lidarpoints_dynamic()

        semantic_colormap = (
            self.get_semantic_colormap(
                camera_semantics=len(camera_ids) > 0,
                lidar_semantics=len(lidar_ids) > 0,
            )
            if color_type == "semantics"
            else None
        )

        # aux depth-data is required for camera-based point-clouds - skip if not available
        if len(camera_ids):
            if self.aux_loader and self.aux_loader.has_depth():
                # we can produce point clouds for cameras
                for pc in self.get_camera_point_clouds(
                    camera_ids,
                    valid_points_only,
                    non_dynamic_points_only,
                    color_type,
                    step_frame,
                    device,
                    visualize,
                    semantic_colormap,
                ):
                    yield pc
            elif force:
                # insufficient aux-data for camera-based point clouds - error out if forced
                raise ValueError(f"{self.__class__.__name__}: aux depth data is required for camera-based point clouds")

        for pc in self.get_lidar_point_clouds(
            lidar_ids=lidar_ids,
            valid_points_only=valid_points_only,
            non_dynamic_points_only=non_dynamic_points_only,
            color_type=color_type,
            step_frame=step_frame,
            device=device,
            visualize=visualize,
            semantic_colormap=semantic_colormap,
        ):
            yield pc

    def get_camera_point_clouds(
        self,
        camera_ids: list[str],
        valid_points_only: bool,
        non_dynamic_points_only: bool,
        color_type: PointCloudColorType,
        step_frame: int,
        device: torch.device,
        visualize: bool,
        semantic_colormap: Optional[np.ndarray],
        filter_previous_predictions: bool = True,
    ) -> Generator[PointCloud, None, None]:
        """Returns a generator for all point-clouds available for the specified camera ids, transformed into NuRec frame.

        To avoid redundancy and since camera depth predictions are imperfect (especially in far-away regions), we filter out
        points that are visible in the next camera frame by default.

        Point-cloud sensor are specified by either logical or unique sensor IDs.

        Can be parameterized to only return valid (default), non-dynamic (default),
        and colored points colorized by one of the following strategies: "camera-rgb" (rgb scene colors),
        "semantics" (semantic colors obtained from shard meta data).
        Args:
            camera_ids: camera ids to generate point clouds for [list[str]]
            valid_points_only: whether to filter out invalid points (typically points in the camera mask or that are semantically labeled as egovehicle) [bool]
            non_dynamic_points_only: whether to filter out dynamic points (as determined by their semantic label) [bool]
            color_type: color type to use for the point cloud (rgb or semantic) [PointCloudColorType]
            step_frame: frame step to for generating point clouds across frames [int]
            device: device to run the operation on [torch.device]
            visualize: whether to visualize the point cloud [bool]
            semantic_colormap: semantic colormap to use for the point cloud [np.ndarray]
            filter_previous_predictions: whether to filter out points that are visible in the previous camera frame [bool]
        """
        # make sure we are initialized
        self._maybe_init_worker()

        # TODO: Add multithreading as in get_lidar_point_clouds
        for unique_camera_id in self._sensor_ids_to_unique_ids(camera_ids, "camera"):
            aux_loader = self.aux_loader
            assert aux_loader is not None, f"{self.__class__.__name__}: aux data was not loaded"

            if non_dynamic_points_only or color_type == "semantics":
                assert aux_loader.has_semantic_segmentation(), (
                    f"{self.__class__.__name__}: Camera semantics must be available"
                )

            for camera_id in self.camera_sensors.keys():
                if self.camera_unique_ids[camera_id].id != unique_camera_id:
                    continue

                if not aux_loader.has_depth(camera_id):
                    logger.warning(
                        f"{self.__class__.__name__}: No depth data available for camera '{camera_id}', "
                        "skipping it for point cloud initialization. RGB supervision for this camera is unaffected."
                    )
                    continue

                assert is_metric_z_depth_method(aux_loader.get_depth_meta(camera_id)["method"]), (
                    f"{self.__class__.__name__}: Metric depth must be available when projecting camera point clouds"
                )

                camera_sensor = self.camera_sensors[camera_id]
                exclude_classes = []
                for class_index, class_name in enumerate(
                    aux_loader.get_semantic_segmentation_meta(camera_id)["stuff_classes"]
                ):
                    if (valid_points_only and class_name in self.camera_point_cloud_ignore_classes) or (
                        non_dynamic_points_only and class_name in self.camera_point_cloud_dynamic_classes
                    ):
                        exclude_classes.append(class_index)

                all_pixels = to_torch(self.cameras_all_pixels[camera_id], device=device)
                all_camera_rays = to_torch(self.cameras_all_rays[camera_id].reshape((-1, 3)), device=device)
                all_camera_footprints = to_torch(self.cameras_all_footprints[camera_id].reshape(-1), device=device)
                camera_params = self.camera_model_parameters[camera_id]
                frame_width, frame_height = camera_params.resolution

                assert step_frame > 0, "step_frame must be positive"
                filter_step_frame = step_frame

                if filter_previous_predictions:
                    # check if the camera is facing in the direction of motion
                    # to do so we check if the position of the first frame is visible in the second frame

                    range_idxs = self.camera_frame_ranges[camera_id]
                    first_camera_frame_index = range_idxs[0]
                    first_frame_position = camera_sensor.get_frames_T_sensor_target(
                        "world", first_camera_frame_index, ncore.data.FrameTimepoint.START
                    )[:3, 3]

                    second_camera_frame_index = range_idxs[1]
                    second_frame_T_sensor_startend = self.world_to_nurec.transform_poses(
                        np.stack(
                            [
                                camera_sensor.get_frames_T_sensor_target(
                                    "world", second_camera_frame_index, ncore.data.FrameTimepoint.START
                                ),
                                camera_sensor.get_frames_T_sensor_target(
                                    "world", second_camera_frame_index, ncore.data.FrameTimepoint.END
                                ),
                            ]
                        )
                    )
                    second_frame_nurec_to_sensor_startend = ncore_transformations.se3_inverse(
                        second_frame_T_sensor_startend
                    )
                    second_frame_camera_timestamp_startend = camera_sensor.frames_timestamps_us[
                        second_camera_frame_index
                    ]

                    valid_indices = get_indices_of_points_visible_in_image(
                        first_frame_position[np.newaxis, :],
                        second_frame_nurec_to_sensor_startend,
                        camera_params,
                        device,
                        second_frame_camera_timestamp_startend[0],
                        second_frame_camera_timestamp_startend[1],
                    )

                    facing_in_direction_of_motion = True if len(valid_indices) > 0 else False
                    if not facing_in_direction_of_motion:
                        filter_step_frame = -filter_step_frame

                previous_frame_prediction: Optional[PointCloud] = None
                for camera_frame_index in self.camera_frame_ranges[camera_id][::filter_step_frame]:
                    camera_frame_timestamp_start = camera_sensor.get_frame_timestamp_us(
                        camera_frame_index, ncore.data.FrameTimepoint.START
                    )
                    camera_frame_timestamp_end = camera_sensor.get_frame_timestamp_us(
                        camera_frame_index, ncore.data.FrameTimepoint.END
                    )
                    if aux_loader.has_semantic_segmentation():
                        semantic_class_id = np.asarray(
                            aux_loader.get_semantic_segmentation(camera_id, camera_frame_timestamp_end)
                        ).reshape(-1)
                    else:
                        semantic_class_id = None

                    color: Optional[np.ndarray]
                    match color_type:
                        case "camera-rgb":
                            color = camera_sensor.get_frame_image_array(camera_frame_index).reshape((-1, 3))
                        case "semantics":
                            color = color_pc_semantics(
                                to_torch(unpack_optional(semantic_class_id), device="cpu"),
                                to_torch(unpack_optional(semantic_colormap), device="cpu"),
                                -1,
                            ).numpy()
                        case None:
                            color = None
                        case _:
                            raise ValueError(f"Unexpected color type {color_type}.")

                    pixels = all_pixels
                    camera_rays = all_camera_rays
                    depth = (
                        aux_loader.get_depth(
                            camera_id, camera_frame_timestamp_end, (frame_width, frame_height)
                        ).reshape(-1)
                        * self.world_to_nurec.target_scale
                    )

                    footprints = all_camera_footprints

                    valid_pixels_mask = self.cameras_frame_valid_pixels_masks[camera_id][camera_frame_index].unpacked()
                    if valid_points_only:
                        point_filter = valid_pixels_mask.reshape(-1)
                    else:
                        point_filter = np.ones(len(depth), dtype=bool)

                    for exclude_class in exclude_classes:
                        point_filter = np.logical_and(point_filter, semantic_class_id != exclude_class)

                    point_filter = np.logical_and(depth > 0, point_filter)
                    if not np.any(point_filter):
                        continue

                    depth = depth[point_filter]
                    pixels = pixels[point_filter]
                    camera_rays = camera_rays[point_filter]
                    footprints = footprints[point_filter]
                    if color is not None:
                        color = color[point_filter]
                    if semantic_class_id is not None:
                        semantic_class_id = semantic_class_id[point_filter]

                    distance = to_torch(depth, device=device) / camera_rays[:, 2]
                    scale = footprints * distance

                    T_sensor_startend = self.world_to_nurec.transform_poses(
                        np.stack(
                            [
                                camera_sensor.get_frames_T_sensor_target(
                                    "world", camera_frame_index, ncore.data.FrameTimepoint.START
                                ),
                                camera_sensor.get_frames_T_sensor_target(
                                    "world", camera_frame_index, ncore.data.FrameTimepoint.END
                                ),
                            ]
                        )
                    )

                    # Project pc to image points
                    if str(device) == "cpu":
                        camera_model = self.camera_models[camera_id]
                        world_rays = camera_model.pixels_to_world_rays_shutter_pose(
                            pixels,
                            T_sensor_startend[0],
                            T_sensor_startend[1],
                        ).world_rays
                    else:
                        camera_model = ncore.sensors.camera_model_from_parameters(camera_params, device=str(device))
                        world_rays = image_points_to_world_rays_shutter_pose(
                            camera_params,
                            camera_model.pixels_to_image_points(pixels),
                            se3_matrix_to_tquat(T_sensor_startend),
                            torch.LongTensor([camera_frame_timestamp_start, camera_frame_timestamp_end]),
                        ).world_rays

                    xyz = world_rays[:, :3] + world_rays[:, 3:] * distance.unsqueeze(-1)
                    flags = torch.zeros((len(xyz),), dtype=torch.int32, device=device)

                    # Load semantic labels, if available, and update flags
                    if semantic_class_id is not None:
                        flags |= RayFlags.VALID_SEMANTIC.value
                        flags[semantic_class_id == self.sensor_road_class_ids[camera_id]] |= RayFlags.ROAD_SEMANTIC
                        flags[semantic_class_id == self.sensor_sky_class_ids[camera_id]] |= RayFlags.SKY_SEMANTIC
                        for vehicle_class_id in self.sensor_vehicle_classes_ids[camera_id]:
                            flags[semantic_class_id == vehicle_class_id] |= RayFlags.VEHICLE_SEMANTIC

                    if color is not None and visualize:
                        visualise_point_cloud(xyz.cpu().numpy(), color / 255)

                    point_cloud = PointCloud(
                        xyz_start=xyz,
                        xyz_end=xyz,
                        flags=flags,
                        color=to_torch(color, device=device) if color is not None else None,
                        camera_footprint_scale=scale,
                        semantic_class_id=(
                            to_torch(semantic_class_id, device=device) if semantic_class_id is not None else None
                        ),
                        sensor_type=["camera"],
                    )

                    if filter_previous_predictions:
                        if previous_frame_prediction is not None:
                            nurec_to_sensor_startend = ncore_transformations.se3_inverse(T_sensor_startend)
                            keep_mask = torch.ones(
                                previous_frame_prediction.xyz_start.shape[0], dtype=torch.bool, device=device
                            )
                            valid_indices = get_indices_of_points_visible_in_image(
                                previous_frame_prediction.xyz_end,
                                nurec_to_sensor_startend,
                                camera_params,
                                device,
                                camera_frame_timestamp_start,
                                camera_frame_timestamp_end,
                                to_torch(valid_pixels_mask, device=device),
                            )
                            keep_mask[valid_indices] = False
                            yield previous_frame_prediction[keep_mask]

                        previous_frame_prediction = point_cloud
                    else:
                        yield point_cloud

                if previous_frame_prediction is not None:
                    yield previous_frame_prediction

    def get_lidar_point_clouds(
        self,
        lidar_ids: list[str],
        valid_points_only: bool,
        non_dynamic_points_only: bool,
        color_type: PointCloudColorType,
        step_frame: int,
        device: torch.device,
        visualize: bool,
        semantic_colormap: Optional[np.ndarray],
        lidar_point_cloud_selected_classes: Optional[list[int]] = None,
    ) -> list[PointCloud]:
        """Returns a list of all point-clouds available for the specified lidar ids, transformed into NuRec frame.

        Point-cloud sensor are specified by either logical or unique sensor IDs.

        Can be parameterized to only return valid (default), non-dynamic (default),
        and colored points colorized by one of the following strategies:
        "camera-rgb" (RGB scene colors), "camera-rgb-frame-valid-pixels" (RGB
        from frame-valid camera pixels while retaining LiDAR geometry), or
        "semantics" (semantic colors obtained from shard metadata).
        """
        # make sure we are initialized
        self._maybe_init_worker()

        if str(device) != "cpu":
            assert self.worker_id is None, "Loading the lidar point clouds on the GPU must be done on the main process"
            # verify that the current GPU is correct
            assert_device_on_local_rank(device)

        point_cloud_list: list[PointCloud] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            for unique_lidar_id in self._sensor_ids_to_unique_ids(lidar_ids, "lidar"):
                lidar_aux_loader = self.aux_loader

                for lidar_id in self.lidar_sensors.keys():
                    if self.lidar_unique_ids[lidar_id].id != unique_lidar_id:
                        continue

                    lidar_sensor = self.lidar_sensors[lidar_id]

                    if has_lidar_semantics := (
                        lidar_aux_loader is not None and lidar_aux_loader.has_lidar_semantic_segmentation()
                    ):
                        ignore_label_class = unpack_optional(lidar_aux_loader).get_lidar_semantic_segmentation_meta(
                            lidar_id
                        )["ignore_label"]
                    else:
                        ignore_label_class = None

                    if not has_lidar_semantics and color_type == "semantics":
                        raise ValueError(
                            f'"semantics" coloring selected, but no semantic aux data available for lidar "{lidar_id}".'
                        )

                    lidar_frame_range = self.lidar_frame_ranges[lidar_id][::step_frame]
                    point_cloud_results: list[None | PointCloud] = [None] * len(lidar_frame_range)

                    def load_lidar_point_cloud(result_index: int, lidar_frame_index: int):
                        set_default_device()

                        point_filter: slice | torch.Tensor = slice(None)
                        if non_dynamic_points_only and valid_points_only:
                            point_filter = to_torch(
                                np.logical_and(
                                    self.lidars_frame_valid_points_masks[lidar_id][lidar_frame_index].unpacked(),
                                    self.lidars_frame_non_dynamic_points_masks[lidar_id][lidar_frame_index].unpacked(),
                                ),
                                device=device,
                            )
                        elif non_dynamic_points_only:
                            point_filter = to_torch(
                                self.lidars_frame_non_dynamic_points_masks[lidar_id][lidar_frame_index].unpacked(),
                                device=device,
                            )
                        elif valid_points_only:
                            point_filter = to_torch(
                                self.lidars_frame_valid_points_masks[lidar_id][lidar_frame_index].unpacked(),
                                device=device,
                            )

                        # A ray-bundle can retain no-return emissions (for raydrop
                        # supervision) with NaN return payloads. They are rays, not
                        # points, and must never enter point-cloud initialization,
                        # coloring, scene-extent, or track geometry.
                        return_valid_filter = to_torch(
                            lidar_sensor.get_frame_ray_bundle_return_valid_mask(lidar_frame_index, return_index=0),
                            device=device,
                            non_blocking=True,
                        )
                        if isinstance(point_filter, slice):
                            point_filter = return_valid_filter
                        else:
                            point_filter &= return_valid_filter

                        # Load one point cloud whose per-point fields share the same ordering.
                        raw_point_cloud = lidar_sensor.get_frame_point_cloud(
                            lidar_frame_index,
                            motion_compensation=True,
                            with_start_points=True,
                            return_index=0,  # closest returns only
                        )
                        xyz_start = to_torch(
                            unpack_optional(raw_point_cloud.xyz_m_start),
                            device=device,
                            non_blocking=True,
                        )[point_filter]
                        xyz_end = to_torch(
                            raw_point_cloud.xyz_m_end,
                            device=device,
                            non_blocking=True,
                        )[point_filter]
                        intensity = to_torch(
                            lidar_sensor.get_frame_ray_bundle_return_intensity(lidar_frame_index),
                            device=device,
                            non_blocking=True,
                        )[point_filter]

                        lidar_timestamp = lidar_sensor.get_frame_timestamp_us(lidar_frame_index)
                        semantic_class_id = (
                            to_torch(
                                unpack_optional(lidar_aux_loader).get_lidar_semantic_segmentation(
                                    lidar_id, lidar_timestamp
                                ),
                                device=device,
                            )[point_filter]
                            if has_lidar_semantics
                            else None
                        )
                        point_cloud = PointCloud(
                            xyz_start=xyz_start,
                            xyz_end=xyz_end,
                            flags=torch.zeros((xyz_end.shape[0],), dtype=torch.int32, device=device),
                            semantic_class_id=semantic_class_id,
                            sensor_type=["lidar"],
                            intensity=intensity,
                        )

                        frame_T_sensor_world = lidar_sensor.get_frames_T_sensor_target("world", lidar_frame_index)
                        T_sensor_nurec = to_torch(
                            self.world_to_nurec.transform_poses(frame_T_sensor_world),
                            device=device,
                            non_blocking=True,
                        )

                        if device.type == "cuda":
                            torch.cuda.synchronize()

                        point_cloud = self._color_lidar_point_cloud(
                            point_cloud=point_cloud,
                            color_type=color_type,
                            lidar_sensor=lidar_sensor,
                            lidar_frame_index=lidar_frame_index,
                            point_filter=point_filter,
                            frame_T_sensor_world=frame_T_sensor_world,
                            lidar_timestamp=lidar_timestamp,
                            semantic_colormap=semantic_colormap,
                            ignore_label_class=ignore_label_class,
                            device=device,
                        )
                        point_cloud = self._transform_lidar_point_cloud_to_nurec(point_cloud, T_sensor_nurec)

                        # Load semantic labels, if available, and update flags
                        if has_lidar_semantics:
                            assert ignore_label_class is not None
                            point_cloud = self._apply_lidar_semantics(
                                point_cloud,
                                lidar_id,
                                valid_points_only,
                                lidar_point_cloud_selected_classes,
                                ignore_label_class,
                                device,
                            )

                        if point_cloud.color is not None and visualize:
                            visualise_point_cloud(
                                point_cloud.xyz_end.cpu().numpy(), point_cloud.color.cpu().numpy() / 255
                            )

                        point_cloud_results[result_index] = point_cloud

                    for future in tqdm.tqdm(
                        concurrent.futures.as_completed(
                            [
                                executor.submit(load_lidar_point_cloud, index, lidar_frame_range[index])
                                for index in range(len(lidar_frame_range))
                            ]
                        ),
                        desc="Get Lidar Point Clouds",
                        total=len(lidar_frame_range),
                        disable=self.tqdm_disabled,
                    ):
                        try:
                            future.result()
                        except BaseException as exc:
                            raise exc  # Exceptions in threads need to be re-raised to be visible

                    assert None not in point_cloud_results
                    point_cloud_list.extend(point_cloud_results)  # type: ignore

        return point_cloud_list

    def get_camera_frusta(
        self,
        camera_id: Optional[str] = None,
        near_plane_depth: float = 0.1,
        far_plane_depth: float = 150.0,
        step_frame: int = 1,
    ) -> Generator[tuple[CameraFrustum, int], None, None]:
        """Returns a generator for all camera frusta for a given camera sensor, transformed into NuRec frame.

        Camera sensor are specified by either logical or unique sensor IDs.

        A single camera sensor needs to be specified - defaults to first camera sensor if not specified."""

        assert near_plane_depth < far_plane_depth, (
            "[NCOREDataSource] Near plane depth of camera frustum is larger than far plane depth"
        )

        # make sure we are initialized
        self._maybe_init_worker()

        # default to first camera if not provided explicitly
        assert len(self.all_camera_ids), "[NCOREDataSource] no camera sensors loaded"
        camera_id = self.all_camera_ids[0] if camera_id is None else camera_id

        # provided samples are ordered by unique sensors
        for _ in self._sensor_ids_to_unique_ids([camera_id], "camera"):
            camera_sensor = self.camera_sensors[camera_id]
            camera_model = self.camera_models[camera_id]

            w = int(camera_model.resolution[0].item())
            h = int(camera_model.resolution[1].item())

            # Initialize the corner uv values
            corner_pixels = torch.tensor([[0, h], [0, 0], [w, 0], [w, h]], dtype=torch.int32)

            # Extract the rays in the camera coordinate system and compute the distance along the ray
            camera_rays = camera_model.pixels_to_camera_rays(corner_pixels)

            # Camera rays are already normalized
            near_plane_dists = near_plane_depth / camera_rays[:, 2:3]
            far_plane_dists = far_plane_depth / camera_rays[:, 2:3]

            for camera_frame_index in self.camera_frame_ranges[camera_id][::step_frame]:
                T_sensor_startend_nurec = self.world_to_nurec.transform_poses(
                    camera_sensor.get_frames_T_sensor_target(
                        target_node="world",
                        frame_indices=camera_frame_index,
                        frame_timepoint=None,  # evaluates both start/end timepoints
                    )  # 2x4x4
                )
                rays = camera_model.pixels_to_world_rays_mean_pose(
                    corner_pixels,
                    T_sensor_startend_nurec[0],
                    T_sensor_startend_nurec[1],
                    camera_rays=camera_rays,
                ).world_rays

                # For origin we take the mean value of the ray origins
                corners = torch.cat(
                    [
                        rays[:, :3] + rays[:, 3:6] * near_plane_dists * self.world_to_nurec.target_scale,
                        rays[:, :3] + rays[:, 3:6] * far_plane_dists * self.world_to_nurec.target_scale,
                    ],
                    dim=0,
                )

                yield (
                    CameraFrustum(corners=corners),
                    camera_sensor.get_frame_timestamp_us(camera_frame_index),
                )

    def get_semantic_classes_frame_masks(
        self, class_names: list[str], camera_semantics: bool, lidar_semantics: bool
    ) -> dict[str, dict[int, PackedMask]]:  # indexed as [unique-sensor-id][sensor-frame-index] -> PackedMask
        """Returns the boolean (packed) 1d masks for each frame of each camera / lidar, where the flag is True if
        the pixel/point belongs to one of the classes selected in class_names and false otherwise"""

        # make sure we are initialized
        self._maybe_init_worker()

        # aux data is required for semantic classes
        aux_loader = self.aux_loader
        assert aux_loader is not None, f"{self.__class__.__name__}: aux data was not loaded"

        ret: dict[str, dict[int, PackedMask]] = defaultdict(dict)

        # check for availability of camera semantics
        semantic_class_id_map = self.get_semantic_classes_map(camera_semantics=True, lidar_semantics=False)
        assert semantic_class_id_map, f"{self.__class__.__name__}: No aux semantic classes map available"

        # iterate over each frame, collecting masks
        if camera_semantics:
            assert aux_loader.has_semantic_segmentation(), (
                f"{self.__class__.__name__}: No camera semantic segmentation auxiliary data available"
            )

            for camera_id, camera_frame_range in tqdm.tqdm(
                self.camera_frame_ranges.items(), desc="Semantic Classes Masks [cameras]", disable=self.tqdm_disabled
            ):
                camera_sensor = self.camera_sensors[camera_id]
                unique_camera_id = self.camera_unique_ids[camera_id].id
                for camera_frame_idx in tqdm.tqdm(
                    camera_frame_range, desc="Semantic Classes Masks [cameras->frames]", disable=self.tqdm_disabled
                ):
                    # load semantic annotation for current frame
                    semantic_mask = np.asarray(
                        aux_loader.get_semantic_segmentation(
                            camera_id,
                            int(
                                camera_sensor.get_frame_timestamp_us(
                                    camera_frame_idx, frame_timepoint=ncore.data.FrameTimepoint.END
                                )
                            ),
                        )
                    ).flatten()

                    # construct mask value
                    valid_semantic_mask = np.zeros_like(semantic_mask, dtype=bool)
                    for class_name in class_names:
                        valid_semantic_mask |= semantic_mask == semantic_class_id_map[class_name]

                    # store as memory efficient packed-mask
                    ret[unique_camera_id] |= {camera_frame_idx: PackedMask(valid_semantic_mask)}

        if lidar_semantics:
            assert aux_loader.has_lidar_semantic_segmentation(), (
                f"{self.__class__.__name__}: No lidar semantic segmentation auxiliary data available"
            )

            for lidar_id, lidar_frame_range in tqdm.tqdm(
                self.lidar_frame_ranges.items(), desc="Semantic Classes Masks [lidars]", disable=self.tqdm_disabled
            ):
                lidar_sensor = self.lidar_sensors[lidar_id]
                unique_lidar_id = self.lidar_unique_ids[lidar_id].id
                for lidar_frame_idx in tqdm.tqdm(
                    lidar_frame_range, desc="Semantic Classes Masks [lidars->frames]", disable=self.tqdm_disabled
                ):
                    # load semantic annotation for current frame
                    semantic_mask = aux_loader.get_lidar_semantic_segmentation(
                        lidar_id,
                        int(
                            lidar_sensor.get_frame_timestamp_us(
                                lidar_frame_idx, frame_timepoint=ncore.data.FrameTimepoint.END
                            )
                        ),
                    )

                    # construct mask value
                    valid_semantic_mask = np.zeros_like(semantic_mask, dtype=bool)
                    for class_name in class_names:
                        valid_semantic_mask |= semantic_mask == semantic_class_id_map[class_name]

                    # store as memory efficient packed-mask
                    ret[unique_lidar_id] |= {lidar_frame_idx: PackedMask(valid_semantic_mask)}

        return ret

    # Data export functionality
    @dataclass(slots=True)
    class CameraFrameExport:
        """Represents one exported camera-associated frame"""

        sequence_id: str
        frame_idx: int
        timestamp_start_us: int
        timestamp_end_us: int
        T_sensor_to_world_start: npt.NDArray[np.float64]
        T_sensor_to_world_end: npt.NDArray[np.float64]
        camera_model_parameters: ncore.data.ConcreteCameraModelParametersUnion
        image_data: ncore.data.EncodedImageData  # TODO(janickm): should be exported publicly in NCore API
        valid_pixels_mask: npt.NDArray[np.bool_]

        frame_track_idxs: Optional[npt.NDArray[np.int32]]

        # Aux data - exported if aux is enabled
        sem_seg_meta: Optional[dict]
        sem_seg_image: Optional[PILImage.Image]

    def export_camera_frames(self, camera_id: str) -> Generator[NCOREDataSource.CameraFrameExport, None, None]:
        """Returns a generator for all camera frame data of a given sensor"""

        # make sure we are initialized
        self._maybe_init_worker()

        camera_sensor = self.camera_sensors[camera_id]
        camera_model_parameters = self.camera_model_parameters[camera_id]

        sem_seg_meta: dict | None = None
        if (aux_loader := self.aux_loader) is not None:
            sem_seg_meta = aux_loader.get_semantic_segmentation_meta(camera_id)

        for camera_frame_idx in self.camera_frame_ranges[camera_id]:
            sem_seg_image: PILImage.Image | None = None
            if aux_loader is not None:
                sem_seg_image = aux_loader.get_semantic_segmentation(
                    camera_id, int(camera_sensor.get_frame_timestamp_us(camera_frame_idx))
                )

            frame_track_idxs = self.cameras_frame_track_idxs[camera_id].get(camera_frame_idx)

            yield NCOREDataSource.CameraFrameExport(
                sequence_id=self.sequence_id,
                frame_idx=camera_frame_idx,
                timestamp_start_us=int(
                    camera_sensor.get_frame_timestamp_us(
                        camera_frame_idx, frame_timepoint=ncore.data.FrameTimepoint.START
                    )
                ),
                timestamp_end_us=int(
                    camera_sensor.get_frame_timestamp_us(
                        camera_frame_idx, frame_timepoint=ncore.data.FrameTimepoint.END
                    )
                ),
                T_sensor_to_world_start=camera_sensor.get_frames_T_sensor_target(
                    "world", camera_frame_idx, frame_timepoint=ncore.data.FrameTimepoint.START
                ),
                T_sensor_to_world_end=camera_sensor.get_frames_T_sensor_target(
                    "world", camera_frame_idx, frame_timepoint=ncore.data.FrameTimepoint.END
                ),
                camera_model_parameters=camera_model_parameters,
                image_data=camera_sensor.get_frame_data(camera_frame_idx),
                valid_pixels_mask=self.cameras_frame_valid_pixels_masks[camera_id][camera_frame_idx].unpacked(),
                frame_track_idxs=frame_track_idxs,
                sem_seg_meta=sem_seg_meta,
                sem_seg_image=sem_seg_image,
            )

    @dataclass(slots=True)
    class LidarFrameExport:
        """Represents one exported lidar-associated frame"""

        sequence_id: str
        frame_idx: int
        timestamp_start_us: int
        timestamp_end_us: int
        T_sensor_to_world_start: npt.NDArray[np.float64]
        T_sensor_to_world_end: npt.NDArray[np.float64]
        xyz_s: npt.NDArray[np.float32]  # in sensor frame at end time
        xyz_e: npt.NDArray[np.float32]  # in sensor frame at end time
        intensity: npt.NDArray[np.float32]
        ray_drop: npt.NDArray[np.float32]
        dynamic_flag: npt.NDArray[np.int8] | None

    def export_lidar_frames(
        self, lidar_id: str, return_index: int = 0
    ) -> Generator[NCOREDataSource.LidarFrameExport, None, None]:
        """Returns a generator for all lidar frame data of a given sensor"""

        # make sure we are initialized
        self._maybe_init_worker()

        lidar_sensor = self.lidar_sensors[lidar_id]

        for lidar_frame_idx in self.lidar_frame_ranges[lidar_id]:
            pc = lidar_sensor.get_frame_point_cloud(
                frame_index=lidar_frame_idx,
                motion_compensation=True,
                with_start_points=True,
                return_index=0,  # closest returns only
            )
            xyz_s: np.ndarray = unpack_optional(pc.xyz_m_start)
            xyz_e = pc.xyz_m_end
            intensity = lidar_sensor.get_frame_ray_bundle_return_intensity(lidar_frame_idx, return_index=return_index)
            raydrop = np.zeros_like(intensity)
            dynamic_flag = lidar_frame_dynamic_flag(
                lidar_sensor,
                lidar_frame_idx,
                backend=self.resolved_dataset.backend,
            )

            yield NCOREDataSource.LidarFrameExport(
                sequence_id=self.sequence_id,
                frame_idx=lidar_frame_idx,
                timestamp_start_us=int(
                    lidar_sensor.get_frame_timestamp_us(
                        lidar_frame_idx, frame_timepoint=ncore.data.FrameTimepoint.START
                    )
                ),
                timestamp_end_us=int(
                    lidar_sensor.get_frame_timestamp_us(lidar_frame_idx, frame_timepoint=ncore.data.FrameTimepoint.END)
                ),
                T_sensor_to_world_start=lidar_sensor.get_frames_T_sensor_target(
                    "world", lidar_frame_idx, frame_timepoint=ncore.data.FrameTimepoint.START
                ),
                T_sensor_to_world_end=lidar_sensor.get_frames_T_sensor_target(
                    "world", lidar_frame_idx, frame_timepoint=ncore.data.FrameTimepoint.END
                ),
                xyz_s=xyz_s,
                xyz_e=xyz_e,
                intensity=intensity,
                ray_drop=raydrop,
                dynamic_flag=dynamic_flag,
            )

    def get_rig_trajectories(self) -> RigTrajectories:
        # implements the protocol: RigTrajectoriesProvider

        # make sure we are initialized
        self._maybe_init_worker()

        def rig_trajectories_generator() -> Generator[RigTrajectories.RigTrajectory, None, None]:
            """Produces individual rig trajectories"""

            ## collect rig poses
            loader = self.sequence_loader
            assert loader is not None, "NCOREDataSource: sequence_loader is not initialized"

            # TODO: frame-pose only data might fail here as there are no rig poses and might require refined logic
            rig_world_edge: ncore_transformations.PoseGraphInterpolator.Edge = unpack_optional(
                loader.pose_graph.get_edge("rig", "world"),
                msg="Rig-to-world poses required for rig-trajectories",
            )

            # all rig poses
            T_rig_world: np.ndarray = rig_world_edge.T_source_target
            T_rig_world_timestamps_us: np.ndarray = unpack_optional(
                rig_world_edge.timestamps_us, msg="Rig-to-world pose requires to be dynamic"
            )

            # rig poses restricted to the time range of interest
            T_rig_world = T_rig_world[poses_range := self.time_range_us.cover_range(T_rig_world_timestamps_us)]
            T_rig_world_timestamps_us = T_rig_world_timestamps_us[poses_range]

            # TODO: this pose resolution will differ for the current free-pose hacks that use different per-frame poses
            #       relative to the rig in V3 data - consider a to replicate a special case / config here as a fallback
            cameras_frame_T_rig_worlds = {
                camera_id: to_torch(
                    self.camera_sensors[camera_id].get_frames_T_source_target(
                        source_node="rig",
                        target_node="world",
                        frame_indices=np.array(camera_frame_range),
                        frame_timepoint=None,  # evaluates both start/end timepoints
                    ),  # Nx2x4x4
                    device="cpu",
                    dtype=torch.float64,
                )
                for camera_id, camera_frame_range in self.camera_frame_ranges.items()
            }

            def get_sensor_frame_timestamps_us(sensor: ncore.data.SensorProtocol, frame_range: range) -> torch.Tensor:
                return torch.tensor(
                    sensor.frames_timestamps_us[frame_range],
                    # cast np.uint64 -> torch.int64 (torch doesn't support unsigned integers - mind potential overflows in the future)
                    dtype=torch.int64,
                    device="cpu",
                ).reshape(
                    len(frame_range), 2
                )  # only a sanity check to be sure that sensor.frames_timestamps_us is Nx2 givin per-frame start/end timestamps

            def get_rig_bbox(
                loader: ncore.data.SequenceLoaderProtocol,
            ) -> Optional[ncore.data.BBox3]:
                """Attempts to parse a dataset-specific ego bounding box centered around the rig frame, if available

                Note: this is currently a non-specified property of NCore and might be abscent in most datasets
                """

                # try to load ego bbox from NV rig meta-data (available, e.g., for clipgt data)
                if nv_rig := loader.generic_meta_data.get("nv-rig", None):
                    # parse a NV rig file for it's body-associated bbox
                    return ncore.data.BBox3.from_array(
                        ncore_utils.get_nv_rig_vehicle_bbox(cast(ncore_utils.NVRigMetadata, nv_rig))
                    )
                elif vehicle_bbox := cast(dict | None, loader.generic_meta_data.get("vehicle-bbox", None)):
                    # load bbox from vehicle-bbox field in meta data (available for some datasets, e.g., PAI)
                    return ncore.data.BBox3(
                        centroid=tuple(vehicle_bbox["centroid"]),
                        dim=tuple(vehicle_bbox["dim"]),
                        rot=tuple(vehicle_bbox["rot"]),
                    )

                # no further ego bbox candidates available
                return None

            yield RigTrajectories.RigTrajectory(
                sequence_id=self.sequence_id,
                rig_bbox=get_rig_bbox(loader),
                T_rig_worlds=to_torch(
                    T_rig_world,
                    device="cpu",
                    # forcing to float64 here to maintain data-structure contract,
                    # altough world poses are really only float32 precision (global poses have higher precision)
                    dtype=torch.float64,
                ),
                T_rig_world_timestamps_us=to_torch(
                    T_rig_world_timestamps_us
                    # cast np.uint64 -> torch.int64 (torch doesn't support unsigned integers - mind potential overflows in the future)
                    .astype(np.int64),
                    device="cpu",
                ),
                cameras_frame_timestamps_us={
                    self.camera_unique_ids[camera_id].id: get_sensor_frame_timestamps_us(
                        self.camera_sensors[camera_id],
                        self.camera_frame_ranges[camera_id],
                    )
                    for camera_id in self.all_camera_ids
                },
                lidars_frame_timestamps_us={
                    self.lidar_unique_ids[lidar_id].id: get_sensor_frame_timestamps_us(
                        self.lidar_sensors[lidar_id], self.lidar_frame_ranges[lidar_id]
                    )
                    for lidar_id in self.all_lidar_ids
                },
                cameras_linear_start_frame_indices={
                    self.camera_unique_ids[camera_id].id: self.camera_linear_start_frame_indices[camera_id]
                    for camera_id in self.all_camera_ids
                },
                lidars_linear_start_frame_indices={
                    self.lidar_unique_ids[lidar_id].id: self.lidar_linear_start_frame_indices[lidar_id]
                    for lidar_id in self.all_lidar_ids
                },
                cameras_frame_T_rig_worlds={
                    self.camera_unique_ids[camera_id].id: cameras_frame_T_rig_worlds[camera_id]
                    for camera_id in self.all_camera_ids
                },
            )

        def camera_calibrations_generator() -> Generator[tuple[str, RigTrajectories.CameraCalibration], None, None]:
            """Produces individual camera calibrations"""

            for camera_id, camera_sensor in self.camera_sensors.items():
                unique_id = self.camera_unique_ids[camera_id]
                yield (
                    unique_id.id,  # unique sensor id
                    RigTrajectories.CameraCalibration(
                        sequence_id=self.sequence_id,
                        logical_sensor_name=camera_id,  # logical sensor name
                        unique_sensor_idx=unique_id.idx,  # unique sensor index
                        T_sensor_rig=to_torch(
                            unpack_optional(
                                camera_sensor.T_sensor_rig, msg=f"Missing T_sensor_rig for camera {camera_id}"
                            ),
                            device="cpu",
                        ),
                        camera_model_parameters=self.camera_model_parameters[camera_id],
                    ),
                )

        def lidar_calibrations_generator() -> Generator[tuple[str, RigTrajectories.LidarCalibration], None, None]:
            """Produces individual lidar calibrations"""

            for lidar_id, lidar_sensor in self.lidar_sensors.items():
                unique_id = self.lidar_unique_ids[lidar_id]
                yield (
                    unique_id.id,  # unique sensor id
                    RigTrajectories.LidarCalibration(
                        sequence_id=self.sequence_id,
                        logical_sensor_name=lidar_id,  # logical sensor name
                        unique_sensor_idx=unique_id.idx,  # unique sensor index
                        T_sensor_rig=to_torch(
                            unpack_optional(
                                lidar_sensor.T_sensor_rig, msg=f"Missing T_sensor_rig for lidar {lidar_id}"
                            ),
                            device="cpu",
                        ),
                        lidar_model_parameters=self.lidar_model_parameters[lidar_id],
                    ),
                )

        return RigTrajectories(
            T_world_base=to_torch(self.T_world_world_global, device="cpu"),
            world_to_nurec=self.world_to_nurec,
            rig_trajectories=list(rig_trajectories_generator()),
            camera_calibrations=OrderedDict(list(camera_calibrations_generator())),
            lidar_calibrations=OrderedDict(list(lidar_calibrations_generator())),
        )

    def get_semantic_colormap(self, camera_semantics: bool, lidar_semantics: bool) -> Optional[npt.NDArray[np.uint8]]:
        """Returns the semantic colormap for requested sensor types (camera and/or lidar), if available

        Raise an error if the mapping is not consistent across all sequences and enabled sensors
        """

        # make sure worker is initialized
        self._maybe_init_worker()

        semantic_colormap: Optional[npt.NDArray[np.uint8]] = None

        def collect_semantic_colormap(
            semantic_colormap: Optional[npt.NDArray[np.uint8]], semantic_colormap_current: npt.NDArray[np.uint8]
        ) -> npt.NDArray:
            if semantic_colormap is None:
                semantic_colormap = semantic_colormap_current
            elif (semantic_colormap != semantic_colormap_current).any():
                raise ValueError(
                    f"{self.__class__.__name__} semantic color map is not consistent across all sequences and "
                    f"requested sensors (camera={camera_semantics}, lidar={lidar_semantics})"
                )
            return semantic_colormap

        aux_loader = self.aux_loader
        assert aux_loader is not None, f"{self.__class__.__name__}: aux data was not loaded"

        if camera_semantics and aux_loader.has_semantic_segmentation():
            for camera_id in self.all_camera_ids:
                semantic_colormap_current = np.array(
                    aux_loader.get_semantic_segmentation_meta(camera_id)["stuff_colors"], dtype=np.uint8
                )

                semantic_colormap = collect_semantic_colormap(semantic_colormap, semantic_colormap_current)

        if lidar_semantics and aux_loader.has_lidar_semantic_segmentation():
            for lidar_id in self.all_lidar_ids:
                semantic_colormap_current = np.array(
                    aux_loader.get_lidar_semantic_segmentation_meta(lidar_id)["stuff_colors"],
                    dtype=np.uint8,
                )

                semantic_colormap = collect_semantic_colormap(semantic_colormap, semantic_colormap_current)

        return semantic_colormap

    def get_semantic_classes_map(self, camera_semantics: bool, lidar_semantics: bool) -> Optional[dict[str, int]]:
        """
        Returns a dictionary mapping semantic class names to their corresponding class indices
        for requested sensor types (camera and/or lidar).

        Raise an error if the mapping is not consistent across all sequences and enabled sensors
        """

        # make sure worker is initialized
        self._maybe_init_worker()

        semantic_mapping: Optional[dict[str, int]] = None

        def collect_semantic_mapping(
            semantic_mapping: Optional[dict[str, int]], semantic_mapping_current: dict[str, int]
        ) -> dict[str, int]:
            if semantic_mapping is None:
                semantic_mapping = semantic_mapping_current
            elif semantic_mapping != semantic_mapping_current:
                raise ValueError(
                    f"{self.__class__.__name__} semantic class mapping is not consistent across all sequences and "
                    f"requested sensors (camera={camera_semantics}, lidar={lidar_semantics})"
                )
            return semantic_mapping

        aux_loader = self.aux_loader
        assert aux_loader is not None, f"{self.__class__.__name__}: aux data was not loaded"

        if camera_semantics and aux_loader.has_semantic_segmentation():
            for camera_id in self.all_camera_ids:
                semantic_mapping_current = {
                    class_name: class_idx
                    for class_idx, class_name in enumerate(
                        aux_loader.get_semantic_segmentation_meta(camera_id)["stuff_classes"]
                    )
                }

                semantic_mapping = collect_semantic_mapping(semantic_mapping, semantic_mapping_current)

        if lidar_semantics and aux_loader.has_lidar_semantic_segmentation():
            for lidar_id in self.all_lidar_ids:
                semantic_mapping_current = {
                    class_name: class_idx
                    for class_idx, class_name in enumerate(
                        aux_loader.get_lidar_semantic_segmentation_meta(lidar_id)["stuff_classes"]
                    )
                }

                semantic_mapping = collect_semantic_mapping(semantic_mapping, semantic_mapping_current)

        return semantic_mapping
