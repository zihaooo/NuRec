# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal, Optional, Tuple

from nurec.config.base_schema import BaseConfigSchema, Field


class PrecomputeExtraSignalsConfig(BaseConfigSchema):
    enabled: bool = False
    sample_every_n_frames: int = 0
    n_rays_per_frame: int = 0


class SystemTestConfig(BaseConfigSchema):
    save_results: bool = False
    png_compress_level: Optional[int] = Field(
        default=None,
        description="zlib level (0-9) of the result PNGs; None = Pillow's default (6). PNG is lossless, so the pixels are "
        "the same at any level; 1 is several times faster to write (writing dominated colour-only renders: mcity, 2026-11-06).",
    )
    image_writer_threads: int = Field(
        default=0,
        description="Background threads that write the result PNGs, so encoding overlaps the next image's rendering; "
        "0 = write in the main thread. All writes finish before the validation epoch ends.",
    )
    result_types: Optional[list[str]] = Field(
        default=None,
        description="With save_results, the result images to write (e.g. [pred_rgb]); None = all of pred_rgb, "
        "pred_distance, pred_normal, pred_opacity. A colour-only render skips the rest (mcity multiclip, 2026-11-06).",
    )
    save_inputs: bool = False
    save_videos: bool = False
    save_stats: bool = False
    save_metrics_parquet: bool = False
    save_extra_signals: bool = False
    separate_val_dir_per_step: bool = Field(
        default=False,
        description="When true, validation outputs are saved to a separate subdirectory per global step "
        "(e.g. val/000200/), preventing overwrites across epochs during trainval.",
    )
    use_camera_name_dirs: bool = False
    frame_naming: Literal["batch-index", "frame-end-timestamp"] = "batch-index"
    precompute_extra_signals: PrecomputeExtraSignalsConfig = Field(default_factory=PrecomputeExtraSignalsConfig)
    video_fps: int
    track_ply: Any
    track_orbit: Any
    track_rotation: Any
    background_removal: Any
    val_render_selected_nodes: Any
    metrics: Any


class SaveRendersConfig(BaseConfigSchema):
    enabled: bool = False
    root: Tuple[float, float, float] = Field(default=(0, 0, 0))
    look_dir: Tuple[float, float, float] = Field(default=(1, 0, 0))
    up_dir: Tuple[float, float, float] = Field(default=(0, 0, 1))
    radius: float = Field(default=0.04)


class FloatingPointFilterConfig(BaseConfigSchema):
    enabled: bool = False
    threshold: float = Field(default=0.02)
    # Source of the ray neighbourhood for the distance-gradient filter:
    #   "rangeview"  — 2D range-view image (structured lidar only; the original path).
    #   "angle"      — sensor-agnostic: bin sensor-frame ELEVATION into rows, sort azimuth within a
    #                  row, gradient along azimuth. Fits discrete-ring lidars and model-free
    #                  unstructured lidar; needs elevation rows separable in elevation.
    neighbor_source: Literal["rangeview", "angle"] = "rangeview"
    # Angle-path only.
    #   elevation_resolution_deg: elevation row (scanline) spacing. None => rings are auto-labelled
    #     from the elevation gaps; RAISES if the rings are not separable in elevation (interleaved-
    #     elevation sensors like ET30X) -- set it explicitly (~FOV_el / n_beams) there.
    #   azimuth_max_gap_deg: max azimuth gap for two rays to count as neighbours (skips scan holes).
    #     None => auto (3x the native azimuth spacing).
    elevation_resolution_deg: Optional[float] = None
    azimuth_max_gap_deg: Optional[float] = None


class LidarEvaluatorTestConfig(BaseConfigSchema):
    raydrop_threshold: float
    opacity_threshold: float = Field(
        default=0.0,
        description="Drop predicted lidar returns whose accumulated opacity is <= this value, "
        "applied before metric computation and visualization export (mirrors the gRPC "
        "SensorSim rendering path, see nurec/render/render.py). Range [0, 1]; 0 disables it.",
    )
    ROI: Any
    save_renders: SaveRendersConfig = Field(default_factory=SaveRendersConfig)
    floating_point_filter: FloatingPointFilterConfig = Field(default_factory=FloatingPointFilterConfig)


class LidarSystemTestConfig(SystemTestConfig):
    lidar: LidarEvaluatorTestConfig


class RecordTimingsConfig(BaseConfigSchema):
    train_interval: int
    val_interval: int


class RecordQualityMetricsConfig(BaseConfigSchema):
    psnr_interval: int


class BaseSystemConfig(BaseConfigSchema):
    """
    Collect config options common to all systems.

    NOTE: this may interfere with how docs for subclasses are rendered (we'd like them to list
    all properties each time) - in that case we will want to get rid of this class.
    """

    max_split_size_mb: Optional[int] = Field(
        default=None,
        description="Read docs before using: https://pytorch.org/docs/stable/notes/cuda.html#environment-variables",
    )

    save_logger: bool

    warmup_steps: int

    test: SystemTestConfig

    optimizer: Optional[Any] = None
    optimizers: Optional[list[Any]] = None


class InitPartsConfig(BaseConfigSchema):
    """Copy selected parameters from another checkpoint before training and optionally freeze them -- e.g. a coarse
    global model's sky and PPISP per-camera parameters, shared by every cell of a multi-cell reconstruction."""

    checkpoint: str = Field(description="NuRec checkpoint to copy from")
    load: list[str] = Field(
        default_factory=list, description="state_dict key prefixes to copy (every key must exist with the same shape)"
    )
    freeze: list[str] = Field(default_factory=list, description="parameter name prefixes set to requires_grad=False")


class BaseGaussiansSystemConfig(BaseSystemConfig):
    init_parts: Optional[InitPartsConfig] = Field(
        default=None,
        description="Copy (and optionally freeze) parameters from another checkpoint at setup, before any resume.",
    )
    scheduler: Optional[Any] = Field(
        default=None,
        description="Examples: SequentialLR, ChainedScheduler",
    )
    record_timings: RecordTimingsConfig = Field(description="Controls the frequency of log timing.")
    record_quality_metrics: RecordQualityMetricsConfig = Field(
        description="Controls the frequency of quality metrics recording."
    )

    collect_garbage_mem_usage: Optional[float] = Field(
        default=None,
        description="Collect garbage and empty cache when GPU memory usage exceeds this threshold (0.0-1.0, e.g., 0.8 for 80%)",
    )
    collect_garbage_check_interval: int = Field(
        default=10000,
        description="Check GPU memory usage every N global steps (to reduce overhead)",
    )


class GaussiansSystemConfig(BaseGaussiansSystemConfig):
    name: Literal["gaussians-system"]
    test: LidarSystemTestConfig
