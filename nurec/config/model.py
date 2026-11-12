# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from enum import Enum
from typing import Annotated, Any, Literal, Optional, Self

from omegaconf import DictConfig, open_dict
from pydantic import model_validator

from nurec.config.base_schema import BaseConfigSchema, Field, PerSensor
from nurec.config.optim import OptimizerConfig, SchedulerConfig


class RendererBackend(Enum):
    """Renderer backend selection for serving and rendering.

    DEFAULT: Use the artifact's trained renderer as-is (PyTorch forward pass).
    GSPLAT:  Force GSplatRenderer override regardless of artifact configuration.
    """

    DEFAULT = "default"
    GSPLAT = "gsplat"


class SensorCalibConfig(BaseConfigSchema):
    """Sensor-specific calibration configuration."""

    enabled: bool


class RigTrajectoryCalibConfig(BaseConfigSchema):
    """Shared rig-trajectory component of composite rig calibration."""

    enabled: bool = True
    spline_degree: int = Field(default=3, ge=1, le=5)
    knot_spacing_seconds: float = Field(default=0.10, gt=0.0, le=60.0)
    max_translation_m: float = Field(default=0.25, gt=0.0, le=100.0)
    max_rotation_degrees: float = Field(default=3.0, gt=0.0, le=180.0)
    curvature_l1_weight: float = Field(default=0.0001, ge=0.0)
    curvature_translation_normalization_m_s2: float = Field(default=2.0, gt=0.0)
    curvature_rotation_normalization_degrees_s2: float = Field(default=20.0, gt=0.0)
    # mcity multiclip (2026-11-08): a run that trains only some of its dataset's frames (a whole-drive place cell via
    # the camera frame sampler's include_frames_file) keeps the dataset's global spline, one time axis shared by every
    # cell, but optimises only the knots its frames use. JSON {camera_id: [frame index, ...]} (or a list of frame
    # indices for every camera) of the frames whose poses the run uses, training and held out. Knots outside their
    # B-spline support get no gradient and are left out of the curvature mean, so the curvature weight acts as in a
    # dataset of just those frames. None: every knot (NuRec's behaviour).
    active_frames_file: Optional[str] = None
    # Anchor the active knots to their values when training starts (e.g. the coarse model's refined trajectory loaded
    # by a weights-only resume), so a cell makes small local corrections without drifting to its own global offset:
    # group L1 of the deviation, translation / anchor_translation_m + rotation / anchor_rotation_degrees, mean over the
    # active knots, times anchor_weight. 0: off.
    anchor_weight: float = Field(default=0.0, ge=0.0)
    anchor_translation_m: float = Field(default=0.02, gt=0.0)
    anchor_rotation_degrees: float = Field(default=0.05, gt=0.0)


class CameraExtrinsicsCalibConfig(BaseConfigSchema):
    """Static per-camera extrinsic component of composite rig calibration."""

    enabled: bool = True
    gauge: Literal["mean-zero-per-trajectory"] = "mean-zero-per-trajectory"
    max_translation_m: float = Field(default=0.10, gt=0.0, le=100.0)
    max_rotation_degrees: float = Field(default=3.0, gt=0.0, le=180.0)
    # mcity multiclip (2026-11-08): keep the loaded residuals (no gradient), e.g. every whole-drive cell keeps the
    # coarse model's camera extrinsics, so the cells share one calibration.
    freeze: bool = False


class ImagePoseCalibConfig(BaseConfigSchema):
    """mcity multiclip (2026-11-11): a small per-image camera rotation on top of the rig spline and the camera mounting
    (rig_calib only), for per-camera, per-frame motion the rig model cannot represent (e.g. the mcity front camera's
    pitch jitter). One rotation vector (radians, OpenCV camera frame) per camera frame, zero at start, applied last:
    T_rig_world(t) . Sp(t) . S_c . E_c . R(w_image). Trained only on the frames a step trains (row-selective Adam), with
    an L2 prior to zero and a penalty on each camera's mean correction (a constant per-camera offset belongs to E_c).
    Frames never trained (held out) keep a zero correction."""

    enabled: bool = False
    max_rotation_degrees: float = Field(default=0.1, gt=0.0, le=10.0)  # radial bound per image
    lr: float = Field(default=3.0e-5, gt=0.0)  # Adam step, radians
    start_global_step: int = Field(default=1000, ge=0)
    prior_weight: float = Field(default=1.0e-4, ge=0.0)  # x mean over trained rows of |w / prior_rotation|^2
    prior_rotation_degrees: float = Field(default=0.02, gt=0.0)
    mean_weight: float = Field(default=1.0e-2, ge=0.0)  # x sum over cameras of |mean_c w / prior_rotation|^2


class CalibConfig(BaseConfigSchema):
    """Calibration component configuration."""

    name: str

    # Common fields
    enabled: bool = False
    start_global_step: int = 250
    skip_first_pose_delta: bool = True
    enable_torch_compile: bool = True

    # Sensor-specific configs
    lidar: SensorCalibConfig
    camera: SensorCalibConfig

    # Composite rig-calibration components. Other calibration modes leave these unset.
    rig_trajectory: RigTrajectoryCalibConfig | None = None
    camera_extrinsics: CameraExtrinsicsCalibConfig | None = None
    image_pose: ImagePoseCalibConfig | None = None  # mcity multiclip: per-image camera rotation (off when None)

    # Optimizer and scheduler configs
    optimizer: OptimizerConfig | None = None
    scheduler: SchedulerConfig | None = None

    @model_validator(mode="after")
    def _validate_rig_calib_components_and_sensors(self) -> "CalibConfig":
        if self.name != "rig-calib":
            return self

        if self.rig_trajectory is None or self.camera_extrinsics is None:
            raise ValueError("rig-calib requires rig_trajectory and camera_extrinsics sections")

        rig_enabled = self.rig_trajectory.enabled
        extrinsics_enabled = self.camera_extrinsics.enabled
        if not (rig_enabled or extrinsics_enabled):
            raise ValueError("rig-calib requires at least one enabled component")
        if extrinsics_enabled and not self.camera.enabled:
            raise ValueError("camera extrinsics require camera calibration to be enabled")
        if rig_enabled and not (self.camera.enabled or self.lidar.enabled):
            raise ValueError("rig trajectory calibration requires at least one enabled sensor type")
        if self.camera.enabled and not (rig_enabled or extrinsics_enabled):
            raise ValueError("camera calibration is enabled but no camera-applicable component is enabled")
        if self.lidar.enabled and not rig_enabled:
            raise ValueError("lidar calibration requires the rig trajectory component")
        return self


class BaseBackgroundConfig(BaseConfigSchema):
    """Base background configuration with common fields."""

    name: str

    # Common settings across all backgrounds
    composite_in_linear_space: bool = False


class SkyEnvMapBackgroundConfig(BaseBackgroundConfig):
    """Sky environment map background configuration."""

    name: Literal["sky-env-map"]

    # Required for env map
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    envmap_type: Literal["cubemap", "equirectangular"]
    filter_texture_updates_by_sky: bool = False
    should_inpaint: bool
    inpaint_threshold: float
    inpaint_kernel_size: int = Field(ge=1)
    min_grad_updates: int = Field(ge=0)
    # Track texture gradients every N training forwards (1 = every call, gsplat's default).
    grad_track_interval: int = Field(default=1, ge=1)
    saturate_radiance: bool = True


class SkyMlpBackgroundConfig(BaseBackgroundConfig):
    """Sky MLP background configuration."""

    name: Literal["sky-mlp"]

    # Required for MLP
    dir_encoding_config: Any
    mlp_network_config: Any
    enable_jit_if_supported: bool
    seed: int
    saturate_radiance: bool = True

    # Ground fill, a render-time option that training never uses: rays pointing below the horizon see ground, not
    # sky, so the MLP's colour there is a guess that shows wherever the Gaussians are see-through (holes under the
    # ego hood). With a colour set, evaluation and validation renders move those pixels towards it by their
    # see-through share, on the final image after post-processing (GaussiansComposite._maybe_apply_ground_fill).
    ground_fill_rgb: list[float] | None = None  # RGB in [0, 1] in final-image colours, e.g. sampled from photos
    ground_fill_rgb_per_camera: dict[str, list[float]] | None = None  # by camera (logical sensor) name; overrides
    ground_fill_up: list[float] = [0.0, 0.0, 1.0]  # world up axis
    ground_fill_horizon_deg: float = 0.0  # rays at or above this elevation keep their colour ...
    ground_fill_blend_deg: float = Field(default=1.0, gt=0.0)  # ... below it the ground colour fades in over this
    # Fill only pixels the dataset marks invalid (the ego hood and other masked pixels, which metrics skip). The sky
    # MLP also draws part of see-through road, lane markings above all, so filling every ground ray changes scored
    # pixels: -0.2 dB held-out on a camera-only mcity run. Without a ray mask nothing is filled.
    ground_fill_masked_only: bool = True
    # Shrink that mask by this many pixels first, so the fill stays inside a narrower mask an external scorer uses
    # (mcity: NuRec's ego mask is ~15 px wider than the scorer's hood mask at 960x600; use 16).
    ground_fill_mask_erode_px: int = Field(default=0, ge=0)
    # Detach the ray directions the sky sees, so its colour gradients never reach the camera poses (rig_calib). A sky
    # frozen from another run is a fixed, detailed function of direction: its gradients pull the camera rotations
    # towards wherever it fits best (mcity 40 s: a frozen sky trained without Gaussians moved the mountings 1.6-3 deg).
    detach_ray_directions: bool = False
    # Time-dependent sky (pipeline/multiclip/sky_stage1.py, 2026-11-05): time_n_knots latent codes of time_dim numbers,
    # one every time_knot_s seconds from time_origin_us, linearly interpolated at each frame's mid timestamp and
    # appended to the direction features before the MLP (whose input grows by time_dim). 0 = direction only.
    time_knot_s: float = 0.0
    time_dim: int = 8
    time_n_knots: int = 0
    time_origin_us: int = 0
    # softplus on the MLP output (with mlp_network_config.output_activation: None): the stage-1 time sky's output; an
    # output ReLU can die for good
    output_softplus: bool = False


class ExtraSignalSkyMlpBackgroundConfig(SkyMlpBackgroundConfig):
    """Sky MLP with extra signal support."""

    name: Literal["extra-signal-sky-mlp"]  # type: ignore[assignment]

    extra_signal: Any


class BackgroundColorConfig(BaseBackgroundConfig):
    """Solid color background configuration."""

    name: Literal["background-color"]

    color: Literal["white", "black", "random"]
    random_eval_color: list[float] | None = None


class SkipBackgroundConfig(BaseBackgroundConfig):
    """Skip/no-op background configuration."""

    name: Literal["skip-background"]


# Discriminated union of all background types
BackgroundConfigType = (
    SkyEnvMapBackgroundConfig
    | SkyMlpBackgroundConfig
    | ExtraSignalSkyMlpBackgroundConfig
    | BackgroundColorConfig
    | SkipBackgroundConfig
)


class ProjectionConfig(BaseConfigSchema):
    """Projection settings for Gaussian renderers."""

    n_rolling_shutter_iterations: int = 5
    ut_dim: int = 3
    ut_alpha: float = 1.0
    ut_beta: float = 2.0
    ut_kappa: float = 0.0
    ut_require_all_sigma_points: bool = False
    image_margin_factor: float = 0.1
    min_projected_ray_radius: float = 0.5477225575051661


class CullingConfig(BaseConfigSchema):
    """Culling settings for Gaussian renderers."""

    near_clip_distance: float = 0.01
    far_clip_distance: float | None = Field(
        default=None,
        deprecated="Use `far_clip_distance_camera` and `far_clip_distance_lidar` instead. "
        "If set, overwrites both sensor-specific fields.",
    )
    far_clip_distance_camera: float = 1e10
    far_clip_distance_lidar: float = 1e10
    near_far_z_culling: bool = False
    rect_bounding: bool = False
    tight_opacity_bounding: bool = False
    tile_based: bool = False
    enable_ray_based_culling: bool = False
    require_positive_hit_t: bool = False

    # gsplat-specific exact dead-pair pre-cull for the unstructured LiDAR render
    # (drops (gaussian, tile) pairs that provably contribute nothing -- no ray in
    # the tile reaches the Gaussian's alpha cutoff -- before the eval3d blend
    # kernel loads them; exact/bit-identical fwd+bwd).
    cull_dead_pairs_lidar_unstructured: PerSensor[bool] = Field(
        default=True,
        description=(
            "Apply the gsplat exact dead-pair pre-cull to the unstructured LiDAR "
            "render path (model.renderer.lidar_unstructured=True). Exact / "
            "bit-identical: drops (gaussian, tile) pairs where no actual ray in "
            "the tile comes near the Gaussian under eval3d's own metric. ON by "
            "default because it is REQUIRED for dense / high-Gaussian-count "
            "scenes: the unstructured model is omnidirectional while the rays are "
            "directional and sparse, so under densification the (gaussian, tile) "
            "pair count reaches ~1B and overflows the eval3d rasterizer's int32 "
            "fwd_batch_state buffer, crashing training. The cull's effectiveness "
            "grows with density (~58% drop at 2.8M Gaussians, ~99.7% / 1.48B->2.6M "
            "in the heavy regime), so the ~26% net-loss seen on a sparse benchmark "
            "does not apply once densification kicks in. PER-SENSOR: a single value "
            "applies to all lidars, or pass {lidar_id: bool} listing every active lidar_id."
        ),
    )

    # gsplat per-Gaussian angular-bounding-box method for the unstructured LiDAR
    # render (the acceleration-structure candidate test). "ut" = the 7-sigma-point
    # unscented-transform AABB (default); "pbf" = the closed-form silhouette
    # (Particle Bounding Frustum) box -- a provably conservative superset that is
    # wider (~+8% candidate pairs, ~+11% render time with the opacity-aware
    # footprint) with no render-quality win on real frames (UT's misses are all
    # sub-threshold). "pbf" is offered as an exact-bound baseline / wide-FoV safety
    # net; keep "ut" unless a measured under-cover motivates the switch.
    box_method_lidar_unstructured: PerSensor[Literal["ut", "pbf", "hybrid"]] = Field(
        default="ut",
        description=(
            "Angular bounding-box method for the unstructured LiDAR render: "
            "'ut' (7-sigma-point AABB, default), 'pbf' (exact closed-form silhouette "
            "box), or 'hybrid' (exact PBF for NEAR gaussians within "
            "pbf_near_range_lidar_unstructured, cheap UT beyond). The UT AABB "
            "UNDER-COVERS the true angular extent of near / large-footprint gaussians "
            "(proven vs a surface oracle -- FINDINGS Phase 38); 'hybrid' fixes that at "
            "~UT cost, 'pbf' fixes it everywhere at ~1.4x cost. Only the gsplat "
            "renderer honours this. PER-SENSOR: single value or "
            "{lidar_id: 'ut'|'pbf'|'hybrid'} listing every active lidar_id."
        ),
    )
    pbf_near_range_lidar_unstructured: PerSensor[float] = Field(
        default=8.0,
        description=(
            "Hybrid-box range threshold in NuRec units (meters): gaussians whose center "
            "is closer than this use the exact PBF silhouette box, farther ones use "
            "the cheap UT AABB. Only used when box_method_lidar_unstructured='hybrid'. "
            "PER-SENSOR: single value or {lidar_id: float} listing every active lidar_id."
        ),
    )
    rs_single_cover_lidar_unstructured: PerSensor[bool] = Field(
        default=False,
        description=(
            "Rolling-shutter single-cover UT box for the unstructured LiDAR render. "
            "When True, the UT (sigma-point) box is computed by projecting each of the "
            "7 sigma points at its OWN converged frame-time (a per-sigma-point "
            "rolling-shutter fixed-point iteration seeded from the tiling's measured "
            "per-cell mean time) instead of unioning the whole frame-start/frame-end "
            "ego-motion arc. This narrows the azimuth box and slices the segment "
            "time-window, cutting candidate (gaussian, tile) pairs by ~28% on a "
            "P128-like grid and making the lidar FORWARD render ~24% faster in "
            "isolation, whether or not cull_dead_pairs is on (FINDINGS Phase 42/43). "
            "NOTE: it is a render-LATENCY optimization, not a training-throughput one "
            "-- on a full car2sim 30k A/B the per-step speedup was ~noise (the lidar "
            "forward render is not the training-step bottleneck; backward + camera + "
            "optimizer dominate) and quality was neutral (chamfer/depth flat, psnr "
            "+0.18). Hence default OFF; enable for render / eval-bound or "
            "int32-pair-ceiling workloads. Like plain UT it can slightly "
            "UNDER-COVER near / large-footprint gaussians (~0.3% dropped hits measured), "
            "so it MUST be paired with box_method='hybrid' (near gaussians keep the "
            "exact PBF box; only the far UT branch uses the RS box). Guarded to "
            "single-cover frames only -- multi-visit (time-tiled / MEMS) frames reject "
            "it, because the per-cell mean-time premise (all rays in a cell share one "
            "firing time) does not hold there. PER-SENSOR: single value or "
            "{lidar_id: bool} listing every active lidar_id."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _migrate_far_clip_distance(cls, data: dict[str, Any] | DictConfig) -> dict[str, Any] | DictConfig:
        """Backward compat: promote legacy ``far_clip_distance`` to the sensor-specific fields."""
        legacy = data.get("far_clip_distance")
        if legacy is not None:
            ctx = open_dict(data) if isinstance(data, DictConfig) else nullcontext()
            with ctx:
                if "far_clip_distance_camera" not in data:
                    data["far_clip_distance_camera"] = legacy
                if "far_clip_distance_lidar" not in data:
                    data["far_clip_distance_lidar"] = legacy
        return data


class CameraTilingConfig(BaseConfigSchema):
    """Camera tiling settings."""

    tile_width: int = 16
    tile_height: int = 16


class LidarTilingConfig(BaseConfigSchema):
    """LiDAR tiling settings."""

    tile_size_elevation: int = 16
    tile_size_azimuth: int = 16
    n_bins_elevation: int = 16
    resolution_elevation: int = 1600
    densification_factor_azimuth: int = 8


class TilingConfig(BaseConfigSchema):
    """Tiling settings with LiDAR support."""

    lidar: LidarTilingConfig = Field(default_factory=LidarTilingConfig)


class TileSizeStrategy(Enum):
    """Tile size selection strategy for the gsplat 3DGUT rasterizer.

    RESOLUTION: Defer to gsplat's resolution-based fallback. gsplat picks
                tile=8 (compact CTA, CTA=32, PPT=2) when min(W,H) < 1080
                and tile=16 (one thread per pixel, CTA=256) at 1080p+.
                Lidar grids are wide-but-shallow and always pick 8. The
                renderer ignores tile_size_* fields in CameraTilingConfig /
                LidarTilingConfig in this mode and pins lidar's
                max_pts_per_tile to 8*8=64.
    CUSTOM:     Pass the configured tile_width / tile_size_* fields through
                to gsplat verbatim. Use for profiling or pinning a specific
                tile size; the caller is responsible for matching kernel
                constraints (3DGUT supports tile_size in {8, 16}).
    """

    RESOLUTION = "resolution"
    CUSTOM = "custom"


_GSPLAT_SUPPORTED_TILE_SIZES = (8, 16)
_EXPLICIT_TILE_FIELDS = (
    ("camera", "tile_width"),
    ("camera", "tile_height"),
    ("lidar", "tile_size_elevation"),
    ("lidar", "tile_size_azimuth"),
)


class CameraLidarTilingConfig(TilingConfig):
    """Tiling settings with both camera and LiDAR support (used by GSplat)."""

    camera: CameraTilingConfig = Field(default_factory=CameraTilingConfig)
    # RESOLUTION: gsplat picks tile_size based on resolution (recommended).
    # The camera/lidar tile_size_* fields above are ignored. Older artifacts
    # that predate this field load with RESOLUTION by default and any stale
    # explicit tile_width / tile_size_* values are silently bypassed.
    # CUSTOM: pass the configured tile sizes through to gsplat verbatim.
    strategy: TileSizeStrategy = TileSizeStrategy.RESOLUTION

    @model_validator(mode="before")
    @classmethod
    def _normalize_and_validate_strategy(cls, data: dict[str, Any] | DictConfig) -> dict[str, Any] | DictConfig:
        """Accept OmegaConf's ``.name`` form on the strategy field, then in CUSTOM mode
        reject any explicit tile size outside the gsplat-supported set.

        OmegaConf serializes Enum members as their ``.name`` ("RESOLUTION"), but pydantic's
        default enum validator only accepts the ``.value`` ("resolution"). Normalize the
        input here so both forms reload from a saved parsed.yaml.
        """
        strategy = data.get("strategy")
        normalized = strategy.value if isinstance(strategy, TileSizeStrategy) else strategy
        if isinstance(normalized, str) and normalized in TileSizeStrategy.__members__:
            normalized = TileSizeStrategy[normalized].value
            data["strategy"] = normalized
        if normalized != TileSizeStrategy.CUSTOM.value:
            return data

        invalid = [
            f"{section}.{field}"
            for section, field in _EXPLICIT_TILE_FIELDS
            for value in [(data.get(section) or {}).get(field)]
            if value is not None and value not in _GSPLAT_SUPPORTED_TILE_SIZES
        ]
        if invalid:
            raise ValueError(
                f"Unsupported gsplat tile size in {', '.join(invalid)}; expected one of {_GSPLAT_SUPPORTED_TILE_SIZES}."
            )
        return data


class RenderModeConfig(BaseConfigSchema):
    """Render mode settings."""

    mode: str = ""
    k_buffer_size: int = 0
    enable_warp_atomic_optim: bool = True


class SensorOutputConfig(BaseConfigSchema):
    """Per-sensor rendering output channel configuration."""

    enable_features: bool = True
    enable_normals: bool = True
    # When False (with features enabled), the rasterizer renders color only and skips the
    # depth/distance channel (CDIM 4 -> 3). Inference loads disable it for the camera path;
    # depth-only configs (no features) keep depth regardless.
    enable_distance: bool = True
    enable_extended_features: bool = True
    enable_sensor_features: bool = True
    enable_ray_gradients: bool = False


class SceneOutputConfig(BaseConfigSchema):
    """Scene-level rendering output configuration."""

    enable_cumulated_weights: bool = False
    enable_visibility: bool = False


class OutputsConfig(BaseConfigSchema):
    """Rendering output channels configuration for all sensor types."""

    camera: SensorOutputConfig = Field(default_factory=SensorOutputConfig)
    lidar: SensorOutputConfig = Field(default_factory=SensorOutputConfig)
    scene: SceneOutputConfig = Field(default_factory=SceneOutputConfig)


class ProfilingRendererConfig(BaseConfigSchema):
    """Profiling configuration for renderers."""

    frequency: float = 0.0


class AntialiasingConfig(BaseConfigSchema):
    """Antialiasing settings for renderers."""

    lidar_divergence: float = 0.002


class CameraToLidarTileMappingConfig(BaseConfigSchema):
    """Cam->lidar tile-mapping precompute knobs.

    Named after its runtime product, ``nurec.utils.tile_mask.CameraToLidarTileMapping``: the
    mapping is built once per camera by projecting subsampled lidar rays into the camera at
    several depth shells, repeated over jittered copies of the camera pose so the cached
    mapping stays valid across frames despite ego motion. Defaults reproduce the historical
    hard-coded values.
    """

    n_pose_samples: int = Field(
        default=64,
        ge=1,
        description="Number of jittered camera poses OR-ed into the cam->lidar tile mapping. "
        "Sample 0 is always the unjittered pose; higher values widen the ego-motion envelope at "
        "linear precompute cost.",
    )
    depth_shells: tuple[float, ...] = Field(
        default=(5.0, 15.0, 30.0, 60.0, 100.0, 200.0),
        min_length=1,
        description="Depths (metres) at which each lidar ray is projected into the camera when "
        "building the cam->lidar tile mapping.",
    )
    ray_subsample: int = Field(
        default=2,
        ge=1,
        description="Stride applied to both lidar ray axes before projection (a value of N keeps "
        "1/N^2 of the rays). Larger strides speed up the precompute but can miss sparsely "
        "populated lidar tiles.",
    )
    pose_jitter_translation_m: float = Field(
        default=0.3,
        ge=0.0,
        description="Half-width (metres) of the uniform per-axis translation jitter applied to the "
        "sampled camera poses, in the camera-local frame.",
    )
    pose_jitter_rotation_rad: float = Field(
        default=0.05,
        ge=0.0,
        description="Half-width (radians) of the uniform per-axis axis-angle rotation jitter applied "
        "to the sampled camera poses, in the camera-local frame.",
    )


class ActiveTileMaskStrategy(str, Enum):
    """How active tiling is used — what the mask covers and what it culls.

    The ``str`` mixin is a serialization contract, not style: this enum lands on
    ``BaseRendererConfig``, so it is part of every renderer config that gets dumped to
    ``parsed.yaml`` and reloaded from an artifact. A python-mode ``model_dump`` keeps enum
    fields as live members; with the mixin such a member *is* a ``str``, so plain-text
    serializers round-trip it as its ``.value`` and string comparisons against the value
    hold. See ``_normalize_strategy`` below for the reload direction.

    CAMERA_INVALID_ONLY:
        Mask camera tiles whose pixels are all RayFlags.INVALID and skip them at
        render time; a skipped tile's rgb pixels are intentionally left black (0.0)
        rather than filled with background. Lidar rendering and supervision are
        untouched; the cam->lidar mapping precompute never runs. Changes no
        supervision terms — no pixel or lidar ray leaves the losses — though it still
        narrows the rendered-visibility signal that gates SelectiveAdam and the
        visibility regularizers.
    VALID_CAMERA_LIDAR_OVERLAP:
        Everything CAMERA_INVALID_ONLY does, plus: cull lidar tiles that do not
        overlap any valid (active) camera tile — skipped at render time, their rays
        stamped INVALID and dropped from the lidar losses. Deliberately crops lidar
        supervision to the camera field of view; the overlap is resolved via the
        precomputed cam->lidar tile mapping (built from ``mapping_params``). Pair with
        a compensated lidar loss weight: the crop changes the 1/n_valid lidar loss
        normalization.
    """

    CAMERA_INVALID_ONLY = "camera_invalid_only"
    VALID_CAMERA_LIDAR_OVERLAP = "valid_camera_lidar_overlap"


class ActiveTileMaskConfig(BaseConfigSchema):
    """Active tile masking: mark tiles inactive and skip rendering them.

    The mask is derived from RayFlags.INVALID pixels, so per-ray flags must be available.
    A *camera* mask supplied on ``RenderingData.active_tile_mask`` short-circuits the
    camera-side derivation and is used as-is — there is no AND/OR against a derived mask.
    The lidar-side mask is always derived from the camera mask and overwrites whatever was
    supplied on the lidar ``RenderingData``. ``GaussiansComposite`` is the sole in-tree
    producer.

    Masking changes what is supervised and optimized, not only what is rendered: gaussians
    whose tiles are all masked out are deliberately excluded from the visibility-gated
    regularizers as well as from the optimizer step, since they receive no data gradient
    and must not be shaped by an unopposed scale/density penalty (the union in
    ``nurec/libs/losses/models/utils.py`` is the one-line lever to gate those regularizers on
    pre-mask ``frustum_visibility`` instead). See ``ActiveTileMaskStrategy`` for the
    per-strategy effects on rendering and on lidar supervision.
    """

    enabled: bool = False
    # Which mode runs when enabled. Separate from `enabled` so a config body can toggle the
    # feature off without losing its strategy choice (note: switching the Hydra option group
    # to `disabled` replaces the whole selection, strategy included).
    strategy: ActiveTileMaskStrategy = ActiveTileMaskStrategy.CAMERA_INVALID_ONLY
    # Parameters for *building* the cam->lidar tile mapping — not the mapping itself,
    # which is derived at runtime and cached per camera. Consumed only under
    # VALID_CAMERA_LIDAR_OVERLAP; defaults reproduce the historical hard-coded values.
    mapping_params: CameraToLidarTileMappingConfig = Field(default_factory=CameraToLidarTileMappingConfig)

    @model_validator(mode="before")
    @classmethod
    def _normalize_strategy(cls, data: Any) -> Any:
        """Accept OmegaConf's ``.name`` form on the strategy field.

        OmegaConf serializes Enum members as their ``.name`` ("CAMERA_INVALID_ONLY"), but
        pydantic's default enum validator only accepts the ``.value`` ("camera_invalid_only").
        Normalize the input here so both forms reload from a saved parsed.yaml (cf. the same
        normalization on ``CameraLidarTilingConfig.strategy``).

        Non-mapping input falls through untouched so pydantic still reports it as a normal
        validation error. The rewrite goes into a shallow copy: the caller's mapping is never
        mutated, and a read-only ``DictConfig`` (a frozen resume config -- exactly the case
        this normalizer exists for) would otherwise raise ``ReadonlyConfigError``.
        """
        if not isinstance(data, (dict, DictConfig)):
            return data
        strategy = data.get("strategy")
        if isinstance(strategy, str) and strategy in ActiveTileMaskStrategy.__members__:
            data = dict(data)
            data["strategy"] = ActiveTileMaskStrategy[strategy].value
        return data


class NRendPipelineConfig(BaseConfigSchema):
    """Pipeline configuration retained for checkpoint backward-compatibility (D18 upgrade guard).

    NRend has been removed (Phases 2 and 6). This class is kept only so that
    legacy artifact configs can be deserialized and rejected gracefully.
    """

    type: str = "reference"
    fwd_type: str | None = None
    bwd_type: str | None = None
    k_buffer_size: int = 16


class NRendPrimitivesConfig(BaseConfigSchema):
    """Primitives configuration retained for checkpoint backward-compatibility (D18 upgrade guard).

    NRend has been removed (Phases 2 and 6). This class is kept only so that
    legacy artifact configs can be deserialized and rejected gracefully.
    """

    type: str = "transformed_aabb"
    density_scale: bool = True


class BaseRendererConfig(BaseConfigSchema):
    """Base renderer configuration with fields shared by all Gaussian renderers."""

    name: str
    log_level: int = 3
    prepare_before_render: bool = False

    # Shared sub-configs
    background: Any = None
    active_tile_mask: ActiveTileMaskConfig = Field(default_factory=ActiveTileMaskConfig)
    projection: ProjectionConfig = Field(default_factory=ProjectionConfig)
    culling: CullingConfig = Field(default_factory=CullingConfig)
    outputs: OutputsConfig = Field(default_factory=OutputsConfig)
    tiling: TilingConfig = Field(default_factory=TilingConfig)
    render: RenderModeConfig = Field(default_factory=RenderModeConfig)
    profiling: ProfilingRendererConfig = Field(default_factory=ProfilingRendererConfig)
    antialiasing: AntialiasingConfig = Field(default_factory=AntialiasingConfig)


class GSplatRendererConfig(BaseRendererConfig):
    """GSplat renderer configuration."""

    name: Literal["3dgs-gsplat", "3dgut-gsplat"]

    tiling: CameraLidarTilingConfig = Field(default_factory=CameraLidarTilingConfig)  # type: ignore[assignment]
    max_camera_tensor_cache_entries: int = Field(
        default=4096,
        ge=1,
        description=(
            "Maximum gsplat camera-tensor cache entries. Entries are keyed by values, device, and "
            "inference mode. For a conservative capacity, multiply the number of devices and modes "
            "by (3 * distinct pinhole calibrations + 2 * distinct fisheye calibrations + distinct "
            "FTheta calibrations + transformed-intrinsics K variants). Count normal and inference "
            "as separate modes when both are used."
        ),
    )
    max_lidar_geometry_cache_entries: int = Field(
        default=1024,
        ge=1,
        description=(
            "Maximum gsplat LiDAR-geometry cache entries. Entries are keyed by values, device, and "
            "inference mode. For a conservative capacity, multiply 3 by the number of distinct "
            "LiDAR geometries, devices, and modes. Count normal and inference as separate modes "
            "when both are used."
        ),
    )
    max_lidar_frame_context_cache_entries: int = Field(
        default=1024,
        ge=1,
        description=(
            "Maximum gsplat LiDAR frame-context cache entries. Entries are keyed by geometry, device, "
            "inference mode, and tiling settings. Size this to at least the number of distinct "
            "geometry/device/mode/tiling combinations used in one renderer."
        ),
    )

    rasterize_mode: Literal["classic", "antialiased"] = "classic"

    log_level: int | str = 3  # type: ignore[assignment]
    sparse_grad: bool = False
    absgrad: bool = False
    packed: bool = False
    cache_static_gaussians: bool = Field(
        default=True,
        description=(
            "Inference-only: collect frame-invariant layers once and re-collect only the dynamic tail each frame."
        ),
    )
    sparse_segments: bool = Field(
        default=False,
        description="Render through gsplat's gaussian-segments draw list instead of the dense path.",
    )
    radius_clip: float = 0.0
    use_rays: bool = False
    lidar_unstructured: PerSensor[bool] = Field(
        default=False,
        description=(
            "Route the lidar branch through gsplat's model-free "
            "render_unstructured (true measured ray geometry + per-ray "
            "timestamps + start/end poses) instead of the structured gridded "
            "rasterization. Camera path is unaffected. Uses opacity-aware "
            "footprint to match structured culling; arc_sampling auto-selects "
            "adaptive-arc for multi-visit (MEMS) frames, else endpoints (spinning). "
            "Disabled by default; enable only with dataset.lidar_ray_mode='unstructured'. "
            "PER-SENSOR: a single value applies to all lidars, or pass "
            "{lidar_id: bool} (every active lidar_id listed) and it must agree per sensor "
            "with dataset.lidar_ray_mode (enforced by NuRecConfig)."
        ),
    )
    sh_features_dtype: Literal["fp32", "fp16"] = Field(
        default="fp32",
        description="Output dtype of features_albedo / features_specular from the Slang collector.",
    )
    renderer_config: Literal["mixed_batch", "parallel_batch"] = "mixed_batch"

    @model_validator(mode="before")
    @classmethod
    def validate_use_rays(cls, config: dict[str, Any] | DictConfig):
        """Default ``use_rays=True`` for 3dgut-gsplat when the field is omitted.

        The rewrite goes into a shallow copy: a read-only ``DictConfig`` (a frozen resume
        config) would otherwise raise ``ReadonlyConfigError`` under ``open_dict``, which only
        lifts struct mode, not read-only mode (cf. the same shallow-copy approach on
        ``ActiveTileMaskConfig._normalize_strategy``).
        """
        if config.get("name") == "3dgut-gsplat" and "use_rays" not in config:
            config = dict(config)
            config["use_rays"] = True  # Should be True by default for 3dgut-gsplat
        return config

    @model_validator(mode="after")
    def validate_3dgut_rasterize_mode(self) -> "GSplatRendererConfig":
        if self.name == "3dgut-gsplat" and self.rasterize_mode == "antialiased":
            raise ValueError(
                "3dgut-gsplat does not support rasterize_mode='antialiased': "
                "3DGUT should use rasterize_mode='classic' to match 3DGUT opacity behavior "
                "when min_projected_ray_radius dilation is enabled."
            )
        return self

    @model_validator(mode="after")
    def validate_renderer_config_compat(self) -> "GSplatRendererConfig":
        if self.name == "3dgs-gsplat" and self.renderer_config == "parallel_batch":
            raise ValueError(
                "3dgs-gsplat does not support renderer_config='parallel_batch': "
                "parallel_batch requires with_eval3d=True (3DGUT path). "
                "Use renderer_config='mixed_batch' for 3dgs-gsplat."
            )
        return self

    @model_validator(mode="after")
    def validate_sparse_segments_support(self) -> "GSplatRendererConfig":
        """Reject the settings the sparse-segments draw list cannot render.

        Every setting named here is fixed for the run, so a rejected combination can
        only fail at the first frame otherwise. The three remaining restrictions --
        gradients, extra signals and an active tile mask -- depend on the frame or on
        the layers config, and ``assert_sparse_segments_supported`` raises on those
        where the frame is known.

        ``cache_static_gaussians=False`` is deliberately not rejected: leaving
        ``sparse_segments`` on with the static cache off asks for the dense path.
        """
        if not self.sparse_segments:
            return self
        if self.name != "3dgut-gsplat":
            raise ValueError(f"sparse_segments requires the 3dgut-gsplat renderer, not {self.name!r}.")
        lidar_unstructured = self.lidar_unstructured
        unsupported = [
            ("packed rendering", "packed", self.packed),
            ("sparse gradients", "sparse_grad", self.sparse_grad),
            ("absolute gradients", "absgrad", self.absgrad),
            ("scene visibility output", "outputs.scene.enable_visibility", self.outputs.scene.enable_visibility),
            (
                "unstructured LiDAR",
                "lidar_unstructured",
                any(lidar_unstructured.values()) if isinstance(lidar_unstructured, dict) else lidar_unstructured,
            ),
        ]
        rejected = [f"{feature} ({field}=true)" for feature, field, enabled in unsupported if enabled]
        if rejected:
            raise ValueError(
                f"sparse_segments does not support {', '.join(rejected)}. "
                "Turn the listed settings off, or set sparse_segments=false."
            )
        return self


class NRendRendererConfig(BaseRendererConfig):
    """Legacy renderer config schema retained for checkpoint backward-compatibility (D18 upgrade guard).

    No renderer name ending in ``-nrend`` reaches config parsing: the D18 config upgrade
    (``upgrade_config_2607_9_to_2607_10_migrate_nrend_renderer``) either rewrites it to the
    equivalent GSplat name or, for the ray-traced renderers that have no equivalent, raises
    a legible ValueError.  This class exists only so that legacy artifact configs remain
    describable.  Not included in RendererConfigType.
    """

    name: Literal["3dgrt-optix-nrend", "3dgrt-rejection-optix-nrend", "3dgut-nrend", "3dgs-nrend", "3dgrut-nrend"]

    global_z_order: bool = False
    per_ray_features: bool = False
    use_gsplat_for_camera_rendering: bool = False
    checkpoint_friendly_backward: bool = Field(
        default=False,
        description="If true, use a non-reentrant checkpointing-friendly backward pass implementation (but slower).",
    )

    # Legacy nested configs (OptiX-specific — only 3dgrt renderers used them).
    # Default to None so that non-OptiX renderer names (3dgut, 3dgs) don't inject
    # unexpected pipeline/primitive settings into the config dict.
    pipeline: NRendPipelineConfig | None = None
    primitives: NRendPrimitivesConfig | None = None
    backward: dict[str, Any] | None = None


# Discriminated union of all active renderer types.
# NRendRendererConfig is intentionally excluded; use GSplatRendererConfig for all configs.
# The D18 config upgrade migrates legacy -nrend renderer names to their GSplat equivalent,
# and raises a ValueError for the ray-traced ones that have none.
RendererConfigType = GSplatRendererConfig


class BaseStrategyConfig(BaseConfigSchema):
    """Base strategy configuration with common fields."""

    name: str
    enforce_empty_cache_after_parameter_update: bool = Field(
        default=True,
        description="Immediately empty the CUDA allocator cache after Gaussian parameter updates.",
    )

    # Common settings across all strategies
    print_stats: bool = False
    exclude_layer_ids: list[str] = Field(default_factory=list)


class DensifyConfig(BaseConfigSchema):
    """Gsplat densification sub-config."""

    frequency: int = Field(
        gt=0,
        description="Frequency of densification, in global iterations. `densification_interval` in 3DGS.",
    )
    start_iteration: int = Field(
        ge=0,
        description="Start densifying Gaussians from this global iteration. `densify_from_iter` in 3DGS.",
    )
    end_iteration: int = Field(
        ge=0,
        description="Stop densifying Gaussians at this global iteration. `densify_until_iter` in 3DGS.",
    )
    clone_grad_threshold: float = Field(
        ge=0.0,
        default=0.0002,
        description="Positional gradient threshold above which Gaussians are cloned. `densify_grad_threshold` in 3DGS.",
    )
    split_grad_threshold: float = Field(
        ge=0.0,
        default=0.0002,
        description="Positional gradient threshold above which Gaussians are split. `densify_grad_threshold` in 3DGS.",
    )
    relative_size_threshold: float = Field(
        ge=0.0,
        default=0.01,
        description=(
            "Gaussians larger than `relative_size_threshold * scene_extent` are split rather than cloned. "
            "`percent_dense` in 3DGS."
        ),
    )
    split_n_gaussians: int = Field(
        gt=0,
        default=2,
        description="Number of Gaussians produced per split. Hardcoded to 2 in 3DGS.",
    )


class PruneConfig(BaseConfigSchema):
    """Gsplat pruning sub-config."""

    frequency: int = Field(
        gt=0,
        description="Frequency of pruning, in global iterations.",
    )
    start_iteration: int = Field(
        ge=0,
        description="Start pruning Gaussians from this global iteration.",
    )
    end_iteration: int = Field(
        ge=0,
        description="Stop pruning Gaussians at this global iteration.",
    )
    density_threshold: float = Field(
        ge=0.0,
        le=1.0,
        default=0.01,
        description="Gaussians with density below this threshold are pruned.",
    )


class ResetDensityConfig(BaseConfigSchema):
    """Gsplat density reset sub-config."""

    frequency: int = Field(
        gt=0,
        description="Frequency of density reset, in global iterations. `opacity_reset_interval` in 3DGS.",
    )
    start_iteration: int = Field(
        ge=0,
        description="Start resetting densities from this global iteration.",
    )
    end_iteration: int = Field(
        ge=0,
        description="Stop resetting densities at this global iteration.",
    )
    new_max_density: float = Field(
        ge=0.0,
        le=1.0,
        default=0.01,
        description="Cap Gaussian densities at this value on reset. Hardcoded to 0.01 in 3DGS.",
    )


class GsplatStrategyConfig(BaseStrategyConfig):
    """Gsplat densification/pruning strategy configuration."""

    name: Literal["gsplat"]

    densify: DensifyConfig | None = None
    prune: PruneConfig | None = None
    reset_density: ResetDensityConfig | None = None


class MCMCStrategyConfig(BaseStrategyConfig):
    """MCMC sampling strategy configuration."""

    class MCMCPerturbationConfig(BaseConfigSchema):
        """Configuration settings of MCMC perturbations"""

        class MCMCNoiseLearningRateConfig(BaseConfigSchema):
            """Configuration settings of MCMC noise learning rate"""

            default: float = Field(
                default=5000.0,
                description="Default noise learning rate for MCMC strategy.",
            )
            layers: dict[str, float] = Field(
                default={
                    "dynamic_rigids": 5000.0,
                    "dynamic_deformables": 5000.0,
                    "road": 5000.0,
                },
                description="Noise learning rate overrides for specific layers.",
            )

        start_iteration: int = Field(
            ge=0,
            default=1,
            description="Start perturbing Gaussians from this global iteration.",
        )

        end_iteration: int = Field(
            ge=0,
            default=10_000_000,  # Default to a high number to make sure it runs until the end of training if not set
            description="Stop perturbing Gaussians from this global iteration.",
        )

        frequency: int = Field(
            gt=0,
            default=1,
            description="Frequency of applying Gaussian perturbation, in global iterations.",
        )

        noise_lr: MCMCNoiseLearningRateConfig = Field(
            default_factory=MCMCNoiseLearningRateConfig,
            description="Noise learning rate settings for MCMC strategy.",
        )

        move_outside_of_cuboid: bool = Field(
            default=False,
            description="Move the gaussians outside of the cuboid tracks if they are inside.",
        )

        max_position_offset_m: dict[str, Annotated[float, Field(gt=0.0)]] = Field(
            default_factory=dict,
            description="Per-layer position trust-region radius around each Gaussian's MCMC anchor, in meters.",
        )

    class MCMCRelocationConfig(BaseConfigSchema):
        """Configuration settings of MCMC relocation"""

        start_iteration: int = Field(
            ge=0,
            default=500,
            description="Start relocating Gaussians from this global iteration.",
        )
        end_iteration: int = Field(
            ge=0,
            default=25_000,
            description="Stop relocating Gaussians from this global iteration.",
        )
        frequency: int = Field(
            gt=0,
            default=100,
            description="Frequency of Gaussians relocation, in global iterations.",
        )
        max_invisible_steps: Optional[int] = Field(
            default=None,
            description="Maximum number of steps a Gaussian can be invisible before being marked for relocation.",
        )
        same_track_layer_ids: list[str] = Field(
            default_factory=list,
            description="Rigid layers whose dead Gaussians may only use donors from the same cuboid track.",
        )
        min_gaussians_per_track: dict[str, Annotated[int, Field(ge=1)]] = Field(
            default_factory=dict,
            description=(
                "Global minimum number of protected donor Gaussians per track for track-local relocation. "
                "The target is distributed across DDP ranks, but each rank containing rows for a track "
                "keeps at least one; the aggregate may therefore exceed the target when world_size > target."
            ),
        )

        @model_validator(mode="after")
        def validate_unique_same_track_layer_ids(self) -> Self:
            duplicates = sorted(
                layer_id for layer_id in set(self.same_track_layer_ids) if self.same_track_layer_ids.count(layer_id) > 1
            )
            if duplicates:
                raise ValueError(f"same_track_layer_ids must not contain duplicates: {duplicates}")
            return self

    class MCMCAdditionConfig(BaseConfigSchema):
        """Configuration settings of MCMC relocation"""

        start_iteration: int = Field(
            ge=0,
            default=500,
            description="Start adding new Gaussians from this global iteration.",
        )
        end_iteration: int = Field(
            ge=0,
            default=25_000,
            description="Stop adding new Gaussians from this global iteration.",
        )
        frequency: int = Field(
            gt=0,
            default=100,
            description="Frequency of adding new Gaussians, in global iterations.",
        )
        max_n_gaussians: int = Field(
            gt=0,
            default=2_000_000,
            description="Maximum number of Gaussians. New Gaussians will only be added until this number is reached.",
        )
        fixed_count_layer_ids: list[str] = Field(
            default_factory=list,
            description="Layers excluded from MCMC addition while remaining eligible for relocation and perturbation.",
        )

    class MCMCMaxScaleGuardConfig(BaseConfigSchema):
        """Post-optimizer hard limit for pathological Gaussian scale growth."""

        enabled: bool = Field(
            default=False,
            description="Enable the post-optimizer Gaussian scale guard.",
        )
        start_iteration: int = Field(ge=0, default=0)
        end_iteration: int = Field(ge=0, default=10_000_000)
        frequency: int = Field(gt=0, default=1)
        global_limit_layer_ids: Optional[list[str]] = Field(
            default_factory=lambda: ["background"],
            min_length=1,
            description=(
                "Layers subject to relative_max_scale_to_scene_extent and absolute_max_scale_m. "
                "None applies these global limits to every layer. This does not scope the rigid cuboid limit."
            ),
        )
        relative_max_scale_to_scene_extent: Optional[float] = Field(
            default=None,
            gt=0.0,
            description="Maximum activated scale as a fraction of scene extent. None disables this limit.",
        )
        absolute_max_scale_m: Optional[float] = Field(
            default=3.0,
            gt=0.0,
            description="Absolute maximum activated scale in metres. None disables this limit.",
        )
        relative_max_scale_to_rigid_cuboid: Optional[float] = Field(
            default=1.0,
            gt=0.0,
            description=(
                "For rigid Gaussian layers, limit every activated scale axis to this "
                "multiple of the owning track's largest full cuboid dimension. This "
                "limit applies to every rigid layer independently of global_limit_layer_ids. "
                "None disables it."
            ),
        )

        @model_validator(mode="after")
        def validate_enabled_limits(self) -> "MCMCStrategyConfig.MCMCMaxScaleGuardConfig":
            global_limits = (self.relative_max_scale_to_scene_extent, self.absolute_max_scale_m)
            if self.enabled and all(
                limit is None for limit in (*global_limits, self.relative_max_scale_to_rigid_cuboid)
            ):
                raise ValueError("enabled MCMC max_scale_guard requires at least one configured limit")
            if self.global_limit_layer_ids is not None and all(limit is None for limit in global_limits):
                raise ValueError(
                    "global_limit_layer_ids requires relative_max_scale_to_scene_extent or absolute_max_scale_m"
                )
            return self

    name: Literal["mcmc"]

    binom_n_max: int = Field(
        default=51,
        description=(
            "Maximum number of binomial coefficients to precompute for the MCMC strategy. "
            "Default value from MCMC paper (https://github.com/ubc-vision/3dgs-mcmc/issues/8)"
        ),
    )

    opacity_threshold: float = Field(
        default=0.005,
        description="Minimum opacity for a Gaussian to be marked alive",
    )

    relocate: MCMCRelocationConfig = Field(
        default_factory=MCMCRelocationConfig,
        description="Gaussian relocation settings for MCMC strategy.",
    )

    perturb: MCMCPerturbationConfig = Field(
        default_factory=MCMCPerturbationConfig,
        description="Gaussian perturbation settings for MCMC strategy.",
    )

    add: MCMCAdditionConfig = Field(
        default_factory=MCMCAdditionConfig,
        description="Gaussian addition settings for MCMC strategy.",
    )

    max_scale_guard: MCMCMaxScaleGuardConfig = Field(
        default_factory=MCMCMaxScaleGuardConfig,
        description="Hard scale postcondition applied after all MCMC and optimizer updates.",
    )

    independent_layer_max_n_gaussians: dict[str, Annotated[int, Field(gt=0)]] = Field(
        default_factory=dict,
        description=(
            "Separate MCMC pools for the named layers, with global per-layer growth caps. "
            "They neither use the shared add.max_n_gaussians budget nor exchange relocation donors "
            "with other layers. Empty by default, preserving the shared-pool behavior."
        ),
    )

    cuboid_grouped_layer_ids: list[str] = Field(
        default_factory=list,
        description="Single-GPU static barrier layers with same-cuboid MCMC donors and equal growth-cap shares.",
    )


class ECADStrategyConfig(MCMCStrategyConfig):
    """Efficient Confidence-Aware Densification (Hofer et al., arXiv 2610.13541) on top of the MCMC strategy.

    Keeps MCMC's fixed budget, 5% growth per `add.frequency` and opacity/scale split correction, but draws source
    Gaussians from image error instead of opacity and relocates the Gaussians that contribute least to the images.
    Relocation runs every `relocate.frequency` steps (400 in the paper) between `relocate.start_iteration` and
    `relocate.end_iteration`; the opacity-threshold relocation of MCMC and its SGLD position noise are not used.
    Per-Gaussian confidence lives in channel 0 of `camera_extra_signal` and is trained by the `ecad_confidence` loss.
    """

    class ECADConfig(BaseConfigSchema):
        layer_ids: list[str] = Field(
            default_factory=lambda: ["background"],
            description="Layers that ECAD densifies; the population pool may hold no other non-empty layer.",
        )
        confidence_warmup_steps: int = Field(
            default=5000, ge=0, description="eta = min(t / confidence_warmup_steps, 1) blends in the confidence term."
        )
        gradient_share: float = Field(
            default=0.8, ge=0.0, le=1.0, description="Share of sources drawn from the gradient x confidence term."
        )
        relocation_share: float = Field(
            default=0.1, ge=0.0, le=1.0, description="Share of Gaussians relocated per relocation step at full rate."
        )
        relocation_dead_only_until: int = Field(
            default=5000, ge=0, description="Before this step only Gaussians with zero contribution are relocated."
        )
        relocation_decay_start: int = Field(
            default=10000, ge=0, description="From this step the share decays as exp(-((t - start) / sigma)^2 / 2)."
        )
        relocation_decay_sigma: float = Field(default=5000.0, gt=0.0)
        sample_positions: bool = Field(
            default=True, description="Place each new Gaussian at a draw from N(mu, Sigma') of its source."
        )
        keep_mcmc_noise: bool = Field(default=False, description="Keep MCMC's SGLD position noise (off in the paper).")
        init_confidence: float = Field(
            default=0.0, description="Initial log-confidence gamma written to camera_extra_signal[:, 0] on a fresh start."
        )
        log_every: int = Field(default=1000, ge=0, description="Print ECAD statistics every this many steps (0 = off).")

    name: Literal["ecad"]  # type: ignore[assignment]

    ecad: ECADConfig = Field(default_factory=ECADConfig, description="ECAD sampling and relocation settings.")


# Discriminated union of all strategy types
StrategyConfigType = GsplatStrategyConfig | MCMCStrategyConfig | ECADStrategyConfig


class BasePostProcessingConfig(BaseConfigSchema):
    """Base post-processing module configuration."""

    name: str

    # Common optional fields
    optimizer: OptimizerConfig | None = None
    scheduler: SchedulerConfig | None = None
    start_global_step: int | None = None


class BilateralGridPostProcessingConfig(BasePostProcessingConfig):
    """Bilateral grid post-processing configuration."""

    name: Literal["bilateral-grid-per-camera", "bilateral-grid-per-frame"]

    # Required grid dimensions
    width: int
    height: int
    depth: int


class PPISPPostProcessingConfig(BasePostProcessingConfig):
    """PPISP configuration wrapping the new ppisp library."""

    name: Literal["ppisp-post-processing"]

    # Controller settings
    use_controller: bool = False
    controller_distillation: bool = True
    controller_distillation_steps: int = 0  # 0 = disabled; extends training by this many steps
    controller_activation_ratio: float = 1.0  # set by system init; ratio < 1.0 activates controller

    # Regularization weights (forwarded to ppisp.PPISPConfig)
    exposure_mean: float = 1.0
    vig_center: float = 0.02
    vig_channel: float = 0.1
    vig_non_pos: float = 0.01
    color_mean: float = 1.0
    crf_channel: float = 0.1

    # PPISP main params optimizer (forwarded to ppisp.PPISPConfig)
    ppisp_lr: float = 0.002
    ppisp_eps: float = 1e-15
    ppisp_betas: tuple[float, float] = (0.9, 0.999)
    # mcity multiclip (2026-11-09): update the per-frame PPISP rows (exposure, colour) only on the steps that train
    # their image, each row with its own Adam step count (per-camera vignetting / CRF stay dense Adam). The library's
    # dense Adam keeps moving a row on momentum after each of its rare samples (~5 per image in an 80 s stage 2),
    # which left isolated over-exposed frames (+0.2 to +0.6 exposure). False: the library's dense Adam.
    ppisp_row_selective: bool = False
    # With ppisp_row_selective: the per-frame rows' learning rate and the dense (per-camera) parameters' learning rate
    # as multiples of the scheduled ppisp_lr (2026-11-09; a per-frame refit against a frozen model uses e.g. rows x10,
    # dense x0 = vignetting / CRF kept).
    ppisp_row_lr_scale: float = 1.0
    ppisp_dense_lr_scale: float = 1.0

    # Controller optimizer (forwarded to ppisp.PPISPConfig)
    controller_lr: float = 0.001
    controller_eps: float = 1e-15
    controller_betas: tuple[float, float] = (0.9, 0.999)

    # Scheduler settings for PPISP main params (forwarded to ppisp.PPISPConfig)
    # Controller uses fixed LR with no scheduler.
    scheduler_type: str = "linear_exp"
    scheduler_base_lr: float = 0.002
    scheduler_warmup_steps: int = 500
    scheduler_start_factor: float = 0.01
    scheduler_decay_max_steps: int = 30000
    scheduler_final_factor: float = 0.01

    # Render-time per-frame parameters (mcity multiclip, 2026-11-06; default = the library's behaviour):
    # "controller": the controller predicts exposure/colour for every frame (training frames too).
    # "interp": a training frame uses its own fitted exposure/colour; a held-out frame the values interpolated
    #   (by frame index) from the nearest training frames of the same camera before and after it. A diagnosis of
    #   the controller (and a "re-render the recorded drive" protocol); training is unaffected.
    heldout_params: Literal["controller", "interp"] = "controller"
    # CSV to append, per rendered frame: the controller's exposure/colour against the fitted (training frame) or
    # interpolated (held-out frame) values. Render-time only; None = off.
    diag_csv: Optional[str] = None


class PPISPLegacyPostProcessingConfig(BasePostProcessingConfig):
    """Legacy PPISP post-processing (render-only, for upgraded checkpoints).

    Training with legacy PPISP is not supported; this exists solely so that
    upgraded checkpoints whose post-processing was renamed by the 26.5.58 -> 26.5.59
    config upgrade can pass typed-config validation.
    """

    name: Literal["ppisp-post-processing-legacy"]


class DINOv2PEPostProcessingConfig(BasePostProcessingConfig):
    """DINOv2 positional encoding post-processing configuration."""

    name: Literal["dinov2-pe-post-processing"]

    test_no_pe_map: bool
    pe_height: int
    pe_width: int
    n_dims: int


# Discriminated union of all post-processing types.
# Discriminator is attached to the inner union so it composes correctly when this
# type is used as the value type of a dict[str, ...] field.
PostProcessingConfigType = Annotated[
    BilateralGridPostProcessingConfig
    | PPISPPostProcessingConfig
    | PPISPLegacyPostProcessingConfig
    | DINOv2PEPostProcessingConfig,
    Field(discriminator="name"),
]


class BaseLayerConfig(BaseConfigSchema):
    """Base layer configuration."""

    name: str

    render_in_lidar: bool = Field(
        default=True,
        description="Whether this Gaussian layer contributes opacity and signals to lidar renders. "
        "Camera rendering is unaffected. Disable for camera-supervised overlay layers that duplicate "
        "geometry already represented by another layer.",
    )

    nearest_neighbor_track_for_lidar: bool = False

    # Rationale and the CLI/serve wiring live in RenderableModel.tracks_timestamp_ray_stride_overrides.
    tracks_timestamp_estimation_ray_stride: int = Field(
        default=1,
        ge=1,
        description="Inference-time ~N-fold subsampling of the per-track timestamp-estimation ray grid "
        "(1 = no subsampling, backward-compatible). Applied as a 2D grid decimation (~sqrt(N) per "
        "sensor-grid axis) in RigidGaussianModel so coverage stays spatially uniform. "
        "Layers with motion_source_layer set use the owner's value, including serve-time overrides.",
    )

    # May appear in any layer but not required by all
    initialization: Any = None
    class_labels: list[str] | None = None
    ignore_classes_from_layers: list[str] | None = None
    use_slang: bool = False


class SHGaussiansLayerConfig(BaseLayerConfig):
    """SH Gaussians layer configuration (static/background)."""

    name: Literal["sh-gaussians"]

    density_activation: str
    scale_activation: str
    rotation_activation: str
    progressive_training: Any
    particle: Any

    # May not be in legacy configs
    fourier_features_dim: int | None = None
    time_embed: Any = None

    # When >0 and fourier_features_dim>1, the gradient on features_albedo[..., 1:, :]
    # is zeroed until global_step reaches this value, letting the DC band (static SH)
    # converge before the temporal Fourier coefficients start learning. Default 0 = disabled.
    fourier_freeze_until_step: int = 0

    # SH-specific with defaults
    precision: int | None = None
    transmittance_threshold: float | None = None
    scale_pos_lr_by_scene_extent: bool = False
    # mcity multiclip (2026-11-08): the extent (m) the position learning rate is scaled by, instead of the dataset's
    # (its longest trajectory axis + 2 x max_dist_m): a cell trained on part of the whole drive passes its own frames'
    # extent; the dataset AABB itself (out-of-bound loss, initialisation) is unchanged. None: the dataset's.
    pos_lr_scene_extent: Optional[float] = None
    debug_viz: bool = False

    # Optional complex configs
    optimizer: Any = None
    optimizers: Any = None
    extra_signal: Any = None
    tracks: Any = None


class ConstructionBarriersGaussiansLayerConfig(SHGaussiansLayerConfig):
    """Static SH Gaussians with source-cuboid identity and local noise rejection."""

    name: Literal["construction-barriers-gaussians"]  # type: ignore[assignment]
    cuboid_perturbation_scale: float = Field(
        default=1.3,
        ge=1.0,
        allow_inf_nan=False,
        description="Source-cuboid dimension multiplier for rigid-style MCMC noise rejection, not an optimizer clamp.",
    )


class RigidGaussiansLayerConfig(SHGaussiansLayerConfig):
    """Rigid Gaussians layer configuration (dynamic with tracks)."""

    name: Literal["rigid-gaussians"]  # type: ignore[assignment]

    # Track-related
    tracks: Any
    optimize_track_albedo: bool
    optimize_track_scale: bool
    track_albedo: Any
    track_scale: Any
    tracks_calib: Any = None
    track_timestamp_est_precision: Literal["int32", "int64"] | None = None
    tracks_timestamp_interpolation_cache: bool = Field(
        default=True,
        description="Reuse track poses for shared camera shutter timestamps when ray-layout metadata is "
        "available. Currently supplied by camera batches used for training; standalone rendering and "
        "export paths leave it unset and use uncached estimation. Unknown layouts and short-track "
        "tensors also use uncached estimation. Preparation and allocation overhead can outweigh "
        "savings for sparse active tracks; set false to disable the cache.",
    )
    is_static: bool = False

    # Owner of raw motion tracks and calibration. Track IDs/order, is_static and
    # nearest_neighbor_track_for_lidar must match when binding. Shared motion uses
    # the owner's tracks_timestamp_estimation_ray_stride, including serve-time
    # overrides; the dependent's stride may differ. Owners may append tracks without
    # changing existing indices. Local timestamps and packinfo select appearance frames.
    motion_source_layer: str | None = None

    # May not be in legacy configs
    time_embed: Any = None


class VehicleLightRigidGaussiansLayerConfig(RigidGaussiansLayerConfig):
    """Rigid vehicle-light Gaussians with an explicit per-track-frame DC table."""

    name: Literal["vehicle-light-rigid-gaussians"]  # type: ignore[assignment]
    features_albedo_lr: float = Field(
        default=0.0125,
        gt=0.0,
        description="Learning rate for the vehicle-light temporal albedo table.",
    )


class DeformableGaussiansLayerConfig(RigidGaussiansLayerConfig):
    """Deformable Gaussians layer configuration."""

    name: Literal["deformable-gaussians"]  # type: ignore[assignment]

    # Required for deformable
    deform_network: Any
    deformnet_start_iteration: int
    cuboid_tracks_expand: float
    smoothness_frame_steps: int


class InterpolatedDeformationTableConfig(BaseConfigSchema):
    """Configuration for per-Gaussian temporal deformation control points."""

    frame_period_us: int = Field(
        default=100_000,
        gt=0,
        description="Spacing between deformation control points in microseconds.",
    )
    interpolation: Literal["nearest", "linear_slerp"] = "linear_slerp"
    storage: Literal["frame_major"] = "frame_major"
    position_smoothness: bool = Field(
        default=True,
        description="Emit position second differences for the deform_smoothness loss.",
    )
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig | None = None


class InterpolatedTableDeformableGaussiansLayerConfig(RigidGaussiansLayerConfig):
    """Rigid actor Gaussians with independently learned temporal deformation tables."""

    name: Literal["interpolated-table-deformable-gaussians"]  # type: ignore[assignment]
    deformation_start_iteration: int = Field(default=1000, ge=0)
    # Component-only rendering uses this factor when a deformable layer asks for
    # expanded cuboid masks. Keep the same contract as the Permuto deformable model.
    cuboid_tracks_expand: float = Field(default=0.05, ge=0.0)
    table: InterpolatedDeformationTableConfig


class ElasticLoadedAHLayerConfig(RigidGaussiansLayerConfig):
    """Asset Harvester elastic loaded layer configuration."""

    name: Literal["elastic-loaded-ah"]  # type: ignore[assignment]

    # Required for asset harvester
    fourier_features_dim: int
    time_embed: Any
    particle: Any
    optimizers: Any
    deformation_network: Any
    restriction_policy: str

    # Optional with default fallbacks in code
    deform_matrix_size: int | None = None
    log_every_n_steps: int | None = None
    masking_end_step: int | None = None


# Discriminated union of all layer types
LayerConfigType = (
    SHGaussiansLayerConfig
    | ConstructionBarriersGaussiansLayerConfig
    | RigidGaussiansLayerConfig
    | VehicleLightRigidGaussiansLayerConfig
    | DeformableGaussiansLayerConfig
    | InterpolatedTableDeformableGaussiansLayerConfig
    | ElasticLoadedAHLayerConfig
)


class ModelConfig(BaseConfigSchema):
    """
    Model configuration for Gaussians-based systems.
    """

    name: str
    debug_viz: bool = False
    log_to_prog_bar: bool = True
    saturate_radiance: bool = True
    train_num_rays: int

    layers: dict[str, LayerConfigType] = Field(description="Layer-specific configurations")

    gaussian_optimizer_name: str = Field(
        default="fused_adam",
        description="Optimizer type for Gaussian layer parameters ('fused_adam' or 'selective_adam')",
    )
    gaussian_density_optimizer_name: str = Field(
        default="fused_adam",
        description=(
            "Optimizer type for the Gaussian density parameter group ('fused_adam' or 'selective_adam'). "
            "Consumed via Hydra interpolation (${model.gaussian_density_optimizer_name}) in layer optimizer configs."
        ),
    )
    gaussian_scale_optimizer_name: str = Field(
        default="fused_adam",
        description=(
            "Optimizer type for the Gaussian scale parameter group ('fused_adam' or 'selective_adam'). "
            "Consumed via Hydra interpolation (${model.gaussian_scale_optimizer_name}) in layer optimizer configs."
        ),
    )

    # Per-group runtime flags that toggle SelectiveAdam's visibility masking on/off.
    # Optimizers are always SelectiveAdam instances; these flags are synced onto them
    # each training batch so the behaviour can be flipped live. Defaults match the
    # original MR intent: positions/rotations/features are masked by visibility; density
    # and scale update densely so regularizers (out_of_bound, node_semantic, etc.) still
    # reach invisible Gaussians.
    gaussian_selective: bool = Field(
        default=True,
        description="If true, position/rotation/feature optimizers mask updates by visibility.",
    )
    gaussian_density_selective: bool = Field(
        default=False,
        description="If true, the density optimizer masks updates by visibility.",
    )
    gaussian_scale_selective: bool = Field(
        default=False,
        description="If true, the scale optimizer masks updates by visibility.",
    )
    selective_adam_occlusion_aware: bool = Field(
        default=False,
        description=(
            "If true, SelectiveAdam visibility masks use occlusion-aware cumulated_weights "
            "instead of plain visibility (requires renderer.outputs.scene.enable_cumulated_weights). "
            "Available camera and LiDAR sources are unioned across ranks; if no sources are available "
            "or any provided source has an unexpected Gaussian count, the optimizer safely falls back "
            "to dense updates."
        ),
    )

    calib: CalibConfig | None = Field(default=None, description="Calibration component configuration")

    background: BackgroundConfigType | None = Field(
        default=None, discriminator="name", description="Background component configuration"
    )

    renderer: RendererConfigType = Field(discriminator="name", description="Renderer configuration")

    strategy: StrategyConfigType = Field(
        discriminator="name", description="Gaussian densification/pruning strategy configuration"
    )

    post_processing: dict[str, PostProcessingConfigType] | None = Field(
        default=None,
        description="Post-processing modules (bilateral grids, PPISP, DINOv2 PE, etc.)",
    )
