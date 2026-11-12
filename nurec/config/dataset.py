# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from ncore.data import LabelSource as NCoreLabelSource
from nurec.config.base_schema import BaseConfigSchema, Field, PerSensor


LidarNominalModel = Literal["HESAI-Pandar128", "HESAI-AT128"]


class _BaseDatasetSOConfig(BaseConfigSchema):
    n_samples_per_epoch: int
    n_train_sample_camera_rays: int
    n_train_sample_lidar_rays: int = Field(default=2048)
    n_val_image_subsample: int
    n_train_sequential_image_subsample: int
    aux_data: bool


class VisibleTrackSplitConfig(BaseConfigSchema):
    def model_post_init(self, __context) -> None:
        if self.enabled and not self.target_label_classes:
            raise ValueError(
                "dataset.cuboid_tracks_params.visible_track_split requires target_label_classes when enabled"
            )

    enabled: bool = Field(
        default=False,
        description=(
            "If enabled, split selected pedestrian/person cuboid tracks whose camera-visible interval exceeds "
            "max_visible_duration_s. This is not intended for generated traffic-light/static-rigid tracks."
        ),
    )
    max_visible_duration_s: float = Field(
        default=10.0,
        gt=0.0,
        description="Maximum camera-visible duration per split child track in seconds.",
    )
    merge_gap_s: float = Field(
        default=0.5,
        ge=0.0,
        description="Merge selected visible intervals separated by at most this many seconds before splitting.",
    )
    min_child_visible_duration_s: float = Field(
        default=3.0,
        ge=0.0,
        description=(
            "Merge a short final visible child segment into the previous visible child when possible, and merge "
            "short leading/trailing invisible segments into the nearest visible child."
        ),
    )
    target_label_classes: Optional[list[str]] = Field(
        default=None,
        description="Pedestrian/person track label classes selected for visible-interval splitting.",
    )
    min_track_distance_m: Optional[float] = Field(
        default=5.0,
        ge=0.0,
        description="If specified, only split selected tracks whose accumulated translation distance exceeds this.",
    )
    camera_ids: Optional[list[str]] = Field(
        default=None,
        description="If specified, compute visibility using only these camera ids. Null means training cameras.",
    )
    visibility_subsample: int = Field(
        default=4,
        ge=1,
        description="Pixel subsampling factor for camera-frustum visibility checks.",
    )


class CuboidTracksConfig(BaseConfigSchema):
    excluded_class_names: list[str] = Field(
        default_factory=list,
        description="Classes handled by a separate static layer; exclude before track assembly and motion classification.",
    )

    def model_post_init(self, __context) -> None:
        assert self.track_label_sources is None or (
            all(
                (
                    # Label sources are optionally suffixed by a version string after the first '@'
                    # e.g. AUTOLABEL@v1.0 - check _only_ the source type prefix here as the version
                    # suffix is an arbitrary non-empty string
                    s.split("@", 1)[0] in NCoreLabelSource.__members__
                    for s in self.track_label_sources
                )
            )
            and
            # Check that the version suffixes (if specified) are non-empty
            all(len(s.split("@", 1)[1]) > 0 for s in self.track_label_sources if len(s.split("@")) > 1)
        ), (
            f"[NCOREDataSource]: Label sources {self.track_label_sources} need to be "
            f"prefixed strings from {list(NCoreLabelSource.__members__.keys())} with non-empty version suffixes (if specified)"
        )

    track_min_distance_m: float = Field(
        description="Distance (length of path taken by the object) threshold to classify objects to be dynamic (m)",
        default=0.1,
        ge=0.0,
    )
    track_min_displacement_m: float = Field(
        description="Displacement (difference between start and end positions) threshold to classify objects to be dynamic (m)",
        default=0.1,
        ge=0.0,
    )
    track_speed_reduction_op: Literal["median", "max"] = Field(
        description="Reduction operation [median, max] on all pose-inferred speeds to classify for dynamic/static objects",
        default="median",
    )
    track_min_speed_ms: float = Field(
        description="Speed threshold to classify objects to be dynamic (m/s)",
        default=1.5,
        ge=0.0,
    )
    track_min_centroid_rig_dist_m: float = Field(
        description="Distance threshold for cubic tracks to be considered self-classifications to skip (m)",
        default=3.0,
        ge=0.0,
    )
    track_extrapolate: bool = Field(
        description="Extrapolate label pose of dynamic tracks once at the start / end (to improve interpolation coverage)",
        default=True,
    )
    track_label_sources: Optional[list[str]] = Field(
        description="If specified, a list of allowed label sources from [AUTOLABEL, EXTERNAL, GT_SYNTHETIC, GT_ANNOTATION] with optional '@<source-version>' suffixes"
        "Required if dataset contains labels from more than a single source",
        default=None,
    )
    use_displacement_and_distance: bool = Field(
        description="Use displacement and distance thresholds to classify objects to be dynamic",
        default=True,
    )
    camera_visibility: bool = Field(
        description="If enabled, will only use camera-visible time-intervals for track dynamic classification",
        default=False,
    )
    visible_track_split: VisibleTrackSplitConfig = Field(
        default_factory=VisibleTrackSplitConfig,
        description="Optionally split long-visible pedestrian/person cuboid tracks into shorter child tracks.",
    )


class ValidLidarPointsCuboidTrackConfig(BaseConfigSchema):
    track_padding_m: list[float] = Field(
        min_length=3,
        max_length=3,
    )


class ValidPixelsCuboidTrackConfig(BaseConfigSchema):
    track_padding_m: list[float] = Field(
        description="Cuboid bbox padding in local box x/y/z directions (m)",
        min_length=3,
        max_length=3,
    )
    frame_track_idxs: bool = Field(
        description="If enabled, store per-frame track image buffers with per-pixel track indices",
    )


class ValidPixelsSceneFlowConfig(BaseConfigSchema):
    flow_min_speed_ms: float = Field(
        description="Speed threshold to classify objects to be dynamic  [m/s]",
        ge=0.0,
    )
    flow_dilate_radius: int = Field(
        description="Dilation radius for the dynamic mask from scene flow (dilation is performed after downsampling the mask to a low-res)",
        ge=0,
    )
    flow_downsample_scale: int = Field(
        description="Downsample scale for dynamic mask from scene flow (only for accelerating the dilation operation)",
        ge=1,
    )


class ValidPixelsTrafficLightConfig(BaseConfigSchema):
    seg_dilate_radius: int = Field(
        default=21,
        ge=0,
    )


class ValidPixelsFrameMaskConfig(BaseConfigSchema):
    n_dilation_iterations: int = Field(
        description="Number of dilation iterations to apply to per-frame masks",
        default=10,
    )
    classes: Optional[List[str]] = Field(
        description="If specified, keys to query for per-frame masks. We construct per-frame masks by loading each key via `get_frame_generic_data` and include pixels marked as true as true in any of the per-key masks",
        default=None,
    )


class LidarDynamicPointsConfig(BaseConfigSchema):
    method: Literal["dynamic_tracks", "dynamic_flag"] = Field(
        default="dynamic_tracks",
        description="'dynamic_tracks': Dynamic point classification computed at data loading time from tracks classified to be dynamic\n"
        "'dynamic_flag': NCore-internal per-point 'dynamic_flag' is used [precomputed on potentially different dynamic object classification!]",
    )


class GenerateStaticRigidCuboidTracksConfig(BaseConfigSchema):
    enabled: bool = Field(
        default=False,
        description="If enabled, will generate cuboid tracks for traffic lights",
    )
    max_distance_m: float = Field(
        default=100.0,
        description="Maximum distance to generate static rigid cuboid tracks",
    )
    visibility_check_camera_ids: list[str] = Field(
        default=[],
        description="Camera ids to check for visibility of static rigid objects",
    )
    rigid_classes: list[str] = Field(
        default=["traffic light"],
        description="Classes to generate static rigid cuboid tracks for",
    )
    point_cloud_path: Optional[str] = Field(
        default=None,
        description="Path to accumulated point cloud PLY with segment_id. "
        "Required for lidar-free pipelines where lidar semantic segmentation is unavailable.",
    )


class NCoreDatasetConfig(_BaseDatasetSOConfig):
    name: Literal["ncore"]

    load_aux_depth_in_train: bool = Field(
        default=True,
        description=(
            "Load and convert auxiliary depth labels for random training batches. "
            "Disable when no active training loss consumes camera depth; validation and "
            "train-sequential batches continue to load depth."
        ),
    )
    load_aux_normal_in_train: bool = Field(
        default=True,
        description=(
            "Load and convert auxiliary normal labels for random training batches. "
            "Disable when no active training loss consumes camera normals; validation and "
            "train-sequential batches continue to load normals."
        ),
    )

    samplers: dict[str, Any] = Field(
        default={},
    )

    path: str = Field(
        description="Data-source specification by a 'path' to an NCore meta-file.\n"
        "The sequence time range can be restricted with additional 'seek_offset_sec' / 'duration_sec' parameters.\n",
    )
    seek_offset_sec: float | None
    duration_sec: float | None

    # Sensors and sensor models
    camera_ids: list[str]
    lidar_ids: list[str]

    # V3 sequence loader parameters
    cuboid_loading_max_workers: Optional[int] = Field(
        description="If specified, set the max_workers for thread parallel cuboid loading in the V3 sequence loader"
        " (automatically inferred number for 'null' - serial execution with '0')",
        default=0,
    )

    # V4 sequence loader parameters
    poses_component_group: str = Field(
        description="Name of the V4 component group for poses",
        default="default",
    )
    intrinsics_component_group: str = Field(
        description="Name of the V4 component group for intrinsics",
        default="default",
    )
    masks_component_group: str = Field(
        description="Name of the V4 component group for masks",
        default="default",
    )
    cuboids_component_group: str = Field(
        description="Name of the V4 component group for cuboids",
        default="default",
    )

    train_camera_ids: list[str]
    train_lidar_ids: list[str]

    val_camera_ids: list[str]
    val_lidar_ids: list[str]

    lidar_ignore_rows: list[int] = Field(
        description="If specified, will ignore the specified rows of the lidar points",
        default=[],
    )

    show_progress_bars: bool = Field(
        description="If enabled (default), will show progress bars during dataset initialization.",
        default=True,
    )

    # Scale
    max_dist_m: float
    val_camera: bool = Field(
        description="If enabled, will return camera rays in validation/test time.",
        default=True,
    )
    val_lidar: bool = Field(
        description="If enabled, will return lidar rays in validation/test time.",
        default=False,
    )
    # Translation (meters, relative to rig frame) / rotation (deg, XYZ / roll-pitch-yaw Euler angles relative
    # to rig frame) extrinsic deltas to apply to all sensor extrinsics for validation / test data generation (e.g., for novel-view rendering)
    # If extrinsic deltas are applied, rays will be inconsistent with label data / most metrics can't be evaluated
    val_sensor_transl_delta_m: Optional[list[float]]
    val_sensor_rot_delta_deg: Optional[list[float]]
    val_camera_frame_start: Optional[int] = Field(
        description="Camera frame start index for validation / test mode. Used together with val_camera_frame_step.",
        default=0,
    )
    val_camera_frame_step: Optional[int] = Field(
        description="Camera frame step count for validation / test mode. Mutually exclusive with val_camera_exclude_frame_step, but one must be specified.",
        default=1,
    )
    val_load_rgb: bool = Field(
        default=True,
        description="Decode each validation frame's camera image (the ground truth). False for render-only runs: the "
        "photo is then never loaded and validation computes no image metrics, only renders and saves outputs "
        "(mcity multiclip colour-only renders, 2026-11-06).",
    )
    val_camera_frames_file: Optional[str] = Field(
        description="JSON file: camera id -> local camera frame indices to validate / render (as the coverage sampler's "
        "include_frames_file). Overrides val_camera_frame_start/step for cameras it lists; for cells whose frames are "
        "not contiguous, e.g. place cells over a revisited drive (pipeline/multiclip/ncell, 2026-11-06).",
        default=None,
    )
    val_camera_exclude_frame_start: Optional[int] = Field(
        description="Start camera frame index to exclude in validation / test mode. "
        "Used together with val_camera_exclude_frame_step.",
        default=None,
    )
    val_camera_exclude_frame_step: Optional[int] = Field(
        description="Excludes camera frames every nth step for validation / test mode. Mutually exclusive with val_camera_frame_step, but one must be specified.",
        default=None,
    )
    val_lidar_frame_start: Optional[int] = Field(
        description="Lidar frame start index for validation / test mode. Used together with val_lidar_frame_step.",
        default=0,
    )
    val_lidar_frame_step: Optional[int] = Field(
        description="Lidar frame step count for validation / test mode. Mutually exclusive with val_lidar_exclude_frame_step, but one must be specified.",
        default=1,
    )
    val_lidar_exclude_frame_start: Optional[int] = Field(
        description="Start lidar frame index to exclude in validation / test mode. "
        "Used together with val_lidar_exclude_frame_step.",
        default=None,
    )
    val_lidar_exclude_frame_step: Optional[int] = Field(
        description="Excludes lidar frames every nth step for validation / test mode. Mutually exclusive with val_lidar_frame_step, but one must be specified.",
        default=None,
    )

    camera_mask_sources: List[Literal["dataset", "aux"]] = Field(
        description="Priority order of data sources to load camera masks from (possible values: [dataset, aux])",
        default=["dataset", "aux"],
    )
    n_camera_mask_dilation_iterations: int = Field(
        description="Number of dilation iterations to apply to camera masks",
        default=30,
    )
    camera_mask_overrides: Dict[str, str] = Field(
        description="Override camera masks for specified sensors",
        default={},
    )
    camera_mask_border: Optional[Dict[str, list[int]]] = Field(
        description="If specified, excludes pixels within the specified border (top, right, bottom, left) for each listed camera",
        default=None,
    )

    open_consolidated: bool
    jpeg_backend_cpu: Literal["PIL", "simplejpeg"] = Field(
        description="Backend to use for CPU-based JPEG decoding [PIL, simplejpeg]",
        default="simplejpeg",
    )
    simplejpeg_fastdct: bool = Field(
        description="If enabled, use fastest DCT method for JPEG decoding (via simplejpeg) - speeds up decoding by 4-5% for a minor loss in quality",
        default=False,
    )
    simplejpeg_fastupsample: bool = Field(
        description="If enabled, use fastest color upsampling method for JPEG decoding (via simplejpeg) - speeds up decoding by 4-5% for a minor loss in quality",
        default=False,
    )
    frame_generic_data_pose_overwrite: bool = Field(
        description="If enabled, will overwrite per-frame sensor-world poses from generic data fields (used in some, e.g., separately optimized datasets) - it's an error if these are not provided by the data if enabled",
        default=False,
    )
    camera_max_fov_deg: float = Field(
        description="Limit effective FOV of omnidirectional camera models (like FTheta / OpenCVFisheye) to prevent erroneous flipped re-projections",
        default=190.0,
    )
    lidar_model_parameter_cwccw_fallback: bool = Field(
        description="If enabled, try to apply a backwards-compatibility fallback for NCore lidar model parameters, which were stored with wrong spin direction conventions",
        default=False,
    )
    lidar_model_parameter_nominal_values: PerSensor[LidarNominalModel] = Field(
        description=(
            "Nominal LiDAR model override for all sensors, or overrides keyed by lidar_id for heterogeneous sensors."
        ),
        default_factory=dict,
    )

    # The following parameters are used for 'train_sequential' dataset splits, which exhaustively provide datasource samples in linear order (e.g., used for modules such as the uncertainty estimators that need to be fitted against training rays at the end of training)
    train_sequential_camera: bool = Field(
        description="Used for 'train_sequential' dataset splits, which exhaustively provide datasource samples in linear order "
        "(e.g., used for modules such as the uncertainty estimators that need to be fitted against training rays at the end of training)",
        default=True,
    )
    train_sequential_lidar: bool = Field(
        description="Used for 'train_sequential' dataset splits, which exhaustively provide datasource samples in linear order "
        "(e.g., used for modules such as the uncertainty estimators that need to be fitted against training rays at the end of training)",
        default=False,
    )
    train_sequential_camera_frame_start: Optional[int] = Field(
        description="Camera frame start index for 'train_sequential' mode. "
        "Used together with train_sequential_camera_frame_step.",
        default=0,
    )
    train_sequential_camera_frame_step: Optional[int] = Field(
        description="Used for 'train_sequential' dataset splits. "
        "Mutually exclusive with `train_sequential_camera_exclude_frame_step`, but one must be specified.",
        default=1,
    )
    train_sequential_camera_exclude_frame_start: Optional[int] = Field(
        description="Start camera frame index to exclude in 'train_sequential' mode. "
        "Used together with train_sequential_camera_exclude_frame_step.",
        default=None,
    )
    train_sequential_camera_exclude_frame_step: Optional[int] = Field(
        description="Used for 'train_sequential' dataset splits. "
        "Mutually exclusive with `train_sequential_camera_frame_step`, but one must be specified.",
        default=None,
    )
    train_sequential_lidar_frame_start: Optional[int] = Field(
        description="Lidar frame start index for 'train_sequential' mode. "
        "Used together with train_sequential_lidar_frame_step.",
        default=0,
    )
    train_sequential_lidar_frame_step: Optional[int] = Field(
        description="Used for 'train_sequential' dataset splits. "
        "Mutually exclusive with `train_sequential_lidar_exclude_frame_step`, but one must be specified.",
        default=1,
    )
    train_sequential_lidar_exclude_frame_start: Optional[int] = Field(
        description="Start lidar frame index to exclude in 'train_sequential' mode. "
        "Used together with train_sequential_lidar_exclude_frame_step.",
        default=None,
    )
    train_sequential_lidar_exclude_frame_step: Optional[int] = Field(
        description="Used for 'train_sequential' dataset splits. "
        "Mutually exclusive with `train_sequential_lidar_frame_step`, but one must be specified.",
        default=None,
    )

    cuboid_tracks_params: CuboidTracksConfig
    valid_measurements_method: Literal[
        "EGO",
        "EGO_CUBOIDTRACKS",
        "EGO_SCENEFLOW",
        "EGO_CUBOIDTRACKS_SCENEFLOW",
        "EGO_CUBOIDTRACKS_TRAFFICLIGHT",
        "EGO_FRAMEMASKS",
    ] = Field(
        description="Valid pixels/lidarpoints method to invoke. Options:\n"
        " - EGO: ego camera masks only\n"
        " - EGO_CUBOIDTRACKS: ego camera masks and frame-projected dynamic objects from cuboid tracks; dynamic_flag lidar points\n"
        " - EGO_SCENEFLOW: ego camera masks and scene flow masks\n"
        " - EGO_CUBOIDTRACKS_SCENEFLOW: ego camera masks, frame-projected dynamic objects from cuboid tracks, and scene flow masks;"
        " dynamic_flag lidar points\n"
        " - EGO_CUBOIDTRACKS_TRAFFICLIGHT: ego camera masks, frame-projected dynamic objects from cuboid tracks, and traffic light"
        " masks; dynamic_flag, non-traffic light, camera-visible lidar points\n"
        " - EGO_FRAMEMASKS: ego camera masks and per-frame masks\n",
    )

    valid_lidarpoints_cuboid_track_params: ValidLidarPointsCuboidTrackConfig
    valid_pixels_cuboid_track_params: ValidPixelsCuboidTrackConfig
    valid_pixels_scene_flow_params: ValidPixelsSceneFlowConfig
    valid_pixels_traffic_light_params: ValidPixelsTrafficLightConfig
    valid_pixels_frame_mask_params: ValidPixelsFrameMaskConfig
    lidar_dynamic_points: LidarDynamicPointsConfig
    camera_point_cloud_ignore_classes: list[str]
    camera_point_cloud_dynamic_classes: list[str]
    generate_static_rigid_cuboid_tracks: GenerateStaticRigidCuboidTracksConfig = Field(
        default_factory=GenerateStaticRigidCuboidTracksConfig
    )

    interp_with_rig: bool = Field(
        description="If enabled, will interpolate the camera poses with the rig trajectory.",
        default=True,
    )

    lidar_ray_mode: PerSensor[Literal["structured", "unstructured"]] = Field(
        default="structured",
        description=(
            "LiDAR ray delivery mode. 'structured' (default): the dataset emits a "
            "sensor-model (n_rows, n_columns) range-view grid of rays + labels "
            "(the classic path). 'unstructured': the dataset emits a "
            "model-INDEPENDENT (N, 1) list of the N returning rays + per-ray labels "
            "-- the grid dimensions come from the return list, not the sensor model, "
            "so nothing downstream depends on the RowOffsetStructuredSpinningLidarModel "
            "resolution. 'unstructured' must be paired with "
            "model.renderer.lidar_unstructured=True (enforced by NuRecConfig). The ray "
            "GEOMETRY within a mode is chosen by use_real_lidar_rays: True = true "
            "measured returns (point-cloud geometry, highest fidelity); False = "
            "modeled rays evaluated at the return elements (requires a sensor model; "
            "used for a fair renderer-only A/B against the structured path). Rays are "
            "ALWAYS in NuRec world frame regardless of mode/use_real_lidar_rays. "
            "PER-SENSOR: a single value applies to all active lidars, or pass a dict "
            "{lidar_id: mode} (every active lidar_id must be listed) to differ per sensor. "
            "For 'unstructured' the per-sensor FOV comes from the spinning lidar model "
            "OR the lidar_unstructured_fov_vert_deg/_horiz_deg overrides (required when "
            "the sensor has no model). "
            "See nurec/datasets/README.md#model-free-unstructured-lidar for the full model-free LiDAR path."
        ),
    )

    use_real_lidar_rays: PerSensor[bool] = Field(
        description=(
            "If enabled, use the measured lidar rays instead of model-provided ones. "
            "PER-SENSOR: single value for all lidars, or {lidar_id: bool} listing "
            "every active lidar_id."
        ),
        default=False,
    )

    lidar_unstructured_raydrop_use_stored_no_returns: PerSensor[bool] = Field(
        default=False,
        description=(
            "Classify no-return emissions already stored in the NCore ray bundle "
            "(NCore V4 and later) as unstructured LiDAR raydrop negatives. This is a "
            "property of the DATASET -- whether the recording kept the rays the sensor "
            "fired but got no return for -- and is therefore ORTHOGONAL to "
            "lidar_unstructured_raydrop_virtual_rays, which selects how to SYNTHESIZE "
            "negatives when the dataset has none. Both may be enabled together: stored "
            "no-returns are real measurements and are always preferable, while the "
            "synthesized modes fill the directions the recording does not cover. "
            "Has no effect on a dataset without stored no-returns (every emission is a "
            "return, so the classification is empty) and no effect on the structured "
            "path, whose range-view grid already represents non-returning beams. "
            "Both cases are SILENT: the config validator does not reject or warn about "
            "setting this on a structured sensor, so on a mixed rig it can be enabled "
            "and simply ignored. "
            "PER-SENSOR: single value for all lidars, or {lidar_id: bool} (every active lidar_id must be listed)."
        ),
    )
    lidar_unstructured_raydrop_virtual_rays: PerSensor[Literal["off", "model_elements", "aggregate_lattice"]] = Field(
        default="off",
        description=(
            "How to SYNTHESIZE unstructured LiDAR raydrop negatives when the dataset "
            "does not store no-return emissions (for those, see "
            "lidar_unstructured_raydrop_use_stored_no_returns, which is independent of "
            "this field). "
            "'off' synthesizes nothing. 'model_elements' uses "
            "non-returning sensor grid elements and requires a spinning LiDAR model. "
            "'aggregate_lattice' "
            "samples unoccupied FOV tiles from a firing lattice estimated across frames "
            "and does not require a sensor model. "
            "COMPARING raydrop METRICS ACROSS MODES: raydrop_IoU / recall are only "
            "comparable to the STRUCTURED path under 'model_elements', which supplies "
            "the exact drop set (the complement of returned grid elements) over the "
            "same ray population the structured grid scores. Under "
            "'aggregate_lattice' the negatives are SAMPLED synthesized rays, so the "
            "evaluated population is smaller than the true drop set and recall (hence "
            "IoU) is structurally lower even when the head is equally good -- expect "
            "high precision with low recall. Use 'model_elements' for like-for-like "
            "raydrop A/B; geometry metrics (depth_*, chamfer, psnr) are comparable "
            "under either mode. "
            "See nurec/datasets/README.md#model-free-unstructured-lidar for the full model-free LiDAR path. "
            "PER-SENSOR: single value for all lidars, or {lidar_id: mode} (every active lidar_id must be listed)."
        ),
    )
    lidar_unstructured_raydrop_rays_per_tile: PerSensor[int] = Field(
        default=1,
        description=(
            "Number of virtual rays sampled per in-FOV unoccupied tile when "
            "lidar_unstructured_raydrop_virtual_rays='aggregate_lattice'. Ignored "
            "by the other modes. PER-SENSOR: single value or {lidar_id: int}."
        ),
    )
    lidar_unstructured_raydrop_lattice_time_modes: PerSensor[int] = Field(
        default=1,
        description=(
            "Number of distinct per-direction visit times (K) the aggregate firing "
            "lattice keeps per (az, el) cell. '1' (default) is the single-cover "
            "behaviour: one mean relative frame time per direction (correct for "
            "spinning sensors like P128). '>1' enables multi-time-mode support for "
            "MULTI-COVER sensors (e.g. a self-crossing MEMS Lissajous), where one "
            "direction is fired at several distinct times per frame; the lattice "
            "then recovers up to K visit times by clustering each cell's cross-frame "
            "time histogram. Single-cover cells still collapse to one mode, so this "
            "degrades cleanly. Only used by the 'aggregate_lattice' raydrop mode. "
            "PER-SENSOR: single value or {lidar_id: K} (e.g. keep K=1 for a spinning "
            "sensor and K>1 for a MEMS unit in the same rig)."
        ),
    )
    lidar_unstructured_raydrop_lattice_max_frames: PerSensor[int] = Field(
        default=64,
        description=(
            "Number of frames sampled uniformly across the sequence to build the "
            "aggregate firing lattice (the model-free (az, el) -> relative frame "
            "time reconstruction). The union over frames fills the lattice by "
            "covering per-frame dropouts; coverage saturates quickly (measured: no "
            "gain past ~64 frames on a 199-frame P128 clip) and the build is ~90% "
            "shard I/O, so higher values mainly add startup I/O. Only used by the "
            "'aggregate_lattice' raydrop mode. PER-SENSOR: single value or {lidar_id: int}."
        ),
    )
    lidar_unstructured_raydrop_lattice_n_bins_azimuth: PerSensor[Optional[int]] = Field(
        default=None,
        description=(
            "Azimuth resolution of the aggregate firing-lattice lookup grid. None "
            "(default) uses a mode-appropriate built-in default (7200 for the "
            "single-mode lattice, which resolves a P128's 3600 columns; 1024 for "
            "multi-time-mode). Override for sensors whose firing density differs "
            "from a P128: too coarse averages distinct-time directions together "
            "(blurs rel_t); too fine leaves cells unobserved (NaN holes). Must be "
            "set together with lidar_unstructured_raydrop_lattice_n_bins_elevation. "
            "Only used by the 'aggregate_lattice' raydrop mode. PER-SENSOR: single "
            "value or {lidar_id: int|null}."
        ),
    )
    lidar_unstructured_raydrop_lattice_n_bins_elevation: PerSensor[Optional[int]] = Field(
        default=None,
        description=(
            "Elevation resolution of the aggregate firing-lattice lookup grid. None "
            "(default) uses a mode-appropriate built-in default (512 single-mode, "
            "128 multi-time-mode). Must be set together with "
            "lidar_unstructured_raydrop_lattice_n_bins_azimuth. See that field for "
            "the resolution tradeoff. Only used by the 'aggregate_lattice' raydrop "
            "mode. PER-SENSOR: single value or {lidar_id: int|null}."
        ),
    )
    lidar_unstructured_raydrop_occupancy_n_bins_azimuth: PerSensor[Optional[int]] = Field(
        default=None,
        description=(
            "Azimuth resolution of the RETURN-OCCUPANCY grid used to detect empty "
            "(no-return) tiles for aggregate_lattice virtual rays. None (default) "
            "uses the sensor model's column count when available (e.g. 3600 for a "
            "P128), else the frame-context auto-resolution. CRITICAL for raydrop "
            "coverage: the auto (ray-count) resolution is far too coarse (~180-360 "
            "az bins) -- a dense return frame then fills nearly every tile, leaving "
            "almost no empty tiles, so virtual-ray recall of the true (model_elements) "
            "drop set collapses to ~1%. Matching the sensor grid (~3600x128) recovers "
            "~96-98% recall (FINDINGS: raydrop coverage experiment). Must be set with "
            "lidar_unstructured_raydrop_occupancy_n_bins_elevation. PER-SENSOR."
        ),
    )
    lidar_unstructured_raydrop_occupancy_n_bins_elevation: PerSensor[Optional[int]] = Field(
        default=None,
        description=(
            "Elevation resolution of the return-occupancy grid (see "
            "lidar_unstructured_raydrop_occupancy_n_bins_azimuth). None (default) "
            "uses the sensor model's row count when available (e.g. 128 for a P128). "
            "Must be set with lidar_unstructured_raydrop_occupancy_n_bins_azimuth. "
            "PER-SENSOR: single value or {lidar_id: int|null}."
        ),
    )
    lidar_unstructured_fov_vert_deg: PerSensor[Optional[List[float]]] = Field(
        default=None,
        description=(
            "Unstructured LiDAR vertical FOV as [elevation_lo_deg, elevation_hi_deg] "
            "(degrees). When set it OVERRIDES the spinning-lidar model's elevation FOV "
            "for the render tiling + firing-lattice bounds, and is REQUIRED when the "
            "sensor has no model (else the unstructured render errors). None (default) "
            "= use the model's FOV. PER-SENSOR: a single [lo, hi] applies to all "
            "lidars, or pass {lidar_id: [lo, hi]} (every active lidar_id listed)."
        ),
    )
    lidar_unstructured_fov_horiz_deg: PerSensor[Optional[List[float]]] = Field(
        default=None,
        description=(
            "Unstructured LiDAR horizontal FOV as [azimuth_lo_deg, azimuth_hi_deg] "
            "(degrees). When set it OVERRIDES the model's azimuth FOV; a span of ~360 "
            "deg is treated as a periodic (spinning) sensor, narrower spans as "
            "limited-FOV. Optional even for model-less sensors (azimuth defaults to "
            "full 360 / periodic). None (default) = use the model. PER-SENSOR: single "
            "[lo, hi] or {lidar_id: [lo, hi]}."
        ),
    )
