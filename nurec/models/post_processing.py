# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

from abc import abstractmethod
from typing import Any, List, Type, TypeVar

import ppisp as ppisp_lib  # type: ignore[import-not-found]
import torch
import torch.nn as nn

from ppisp import PPISP as PPISPModule  # type: ignore[import-not-found]
from ppisp import PPISPConfig  # type: ignore[import-not-found]

from nurec.config.model import (
    BasePostProcessingConfig,
    BilateralGridPostProcessingConfig,
    DINOv2PEPostProcessingConfig,
    PPISPPostProcessingConfig,
    VastAppearancePostProcessingConfig,
)
from nurec.config.trainer import TrainerConfig
from nurec.models.base import BaseModel
from nurec.models.nn_extensions import module_call_type
from nurec.models.post_processings.bilateral_grid.bilateral_grid import BilateralGrid
from nurec.models.post_processings.bilateral_grid.bilateral_grid_cuda import BilateralGridCUDA
from nurec.models.post_processings.ppisp_legacy import PPISPLegacy
from nurec.utils.batch import DataAndRenderingBatch, RectSubsampled
from nurec.utils.misc import unpack_optional
from nurec.utils.optim import OptimizerLRSchedulerConfig, configure_optimizers
from nurec.utils.prober import get_global_prober
from nurec.utils.types import ExtraSignal, GaussiansCompositeReturn


logger = logging.getLogger(__name__)


# Use the global prober instance
prober = get_global_prober()


# ---------------------------------------------------------------------------
# PPISP version detection
# ---------------------------------------------------------------------------

PPISP_VERSION_LEGACY = "legacy"
"""Sentinel value indicating the old in-tree PPISP implementation (no version stored in checkpoint)."""


def detect_ppisp_version(
    ppisp_post_processing: "PPISPLegacyPostProcessing | PPISPPostProcessing",
) -> str:
    """Return the PPISP version string for a loaded PPISP post-processing module.

    Returns ``PPISP_VERSION_LEGACY`` for old checkpoints that have no version
    metadata, or the library version string (e.g. ``"1.0.1"``) for new ones.
    """
    return ppisp_post_processing._ppisp_version


class BasePostProcessing(BaseModel):
    variants: dict[str, Type[BasePostProcessing]] = {}
    config: BasePostProcessingConfig  # type: ignore[assignment] # Override type annotation from BaseModel

    def __init__(
        self,
        config: BasePostProcessingConfig,
        trainer_config: TrainerConfig,
        n_training_frames_per_camera: List[int],
        n_training_frames_per_lidar: List[int],
    ):
        super().__init__(config.to_dictconfig())
        self.config = config  # type: ignore[assignment] # Override with typed config
        self.trainer_config = trainer_config
        logger.info(f"{self.__call__.__name__}/n_training_frames_per_camera: {n_training_frames_per_camera}")
        logger.info(f"{self.__call__.__name__}/n_training_frames_per_lidar: {n_training_frames_per_lidar}")

    @classmethod
    def factory(
        cls,
        name: str,
        config: BasePostProcessingConfig,
        trainer_config: TrainerConfig,
        n_training_frames_per_camera: List[int],
        n_training_frames_per_lidar: List[int],
    ) -> BasePostProcessing:
        if name not in cls.variants:
            raise ValueError(f"Unknown variant: {name}")
        variant_class = cls.variants[name]
        return variant_class(
            config=config,
            trainer_config=trainer_config,
            n_training_frames_per_camera=n_training_frames_per_camera,
            n_training_frames_per_lidar=n_training_frames_per_lidar,
        )

    @classmethod
    def register_variant(cls, name: str, variant: Type[BasePostProcessing]) -> None:
        cls.variants[name] = variant

    @abstractmethod
    def forward(
        self,
        results: GaussiansCompositeReturn,
        rays: torch.Tensor | None = None,  # [N, 6]
        pixel_coords: torch.Tensor | None = None,  # [N, 2] pixel center coordinates in original sensor space
        resolution: tuple[int, int] | None = None,  # (width, height) of the original sensor
        training_frame_idx: int | None = None,
        unique_sensor_idx: int | None = None,
        global_step: int = 0,
    ) -> None: ...

    __call__ = module_call_type(forward)


class DinoV2PEPostProcessing(BasePostProcessing):
    """
    Positional embedding post-processing for DINOv2.
    """

    config: DINOv2PEPostProcessingConfig

    def __init__(
        self,
        config: DINOv2PEPostProcessingConfig,
        trainer_config: TrainerConfig,
        n_training_frames_per_camera: List[int],
        n_training_frames_per_lidar: List[int],
    ):
        super().__init__(config, trainer_config, n_training_frames_per_camera, n_training_frames_per_lidar)

        # Globally-shared post-processing for all cameras.
        self.n_dims = int(config.n_dims)
        self.learnable_pe_map = nn.Parameter(
            0.05
            * torch.randn(
                1,
                self.n_dims // 2,
                self.config.pe_height,
                self.config.pe_width,
            ),
            requires_grad=True,
        )
        # positional embedding head to decode the PE map to create some correalation.
        self.pe_head = nn.Sequential(
            nn.Conv2d(self.n_dims // 2, self.n_dims, 1),
        )

    def forward(
        self,
        results: GaussiansCompositeReturn,
        rays: torch.Tensor | None = None,  # [N, 6]
        pixel_coords: torch.Tensor | None = None,  # [N, 2]
        resolution: tuple[int, int] | None = None,  # (width, height)
        training_frame_idx: int | None = None,
        unique_sensor_idx: int | None = None,
        global_step: int = 0,
    ) -> None:
        if not self.training and self.config.test_no_pe_map:
            return

        rendered_cam = results.rendered_cam
        if rendered_cam is None or rendered_cam.extra_ray_signals is None:
            assert not self.training, (
                f"[{self.__class__.__name__}] extra_ray_signals is None during training; "
                f"expected populated by the rasterizer."
            )
            return
        extras = rendered_cam.extra_ray_signals
        dinov2_feats = extras.dinov2_feats
        if dinov2_feats is None:
            # Training: the rasterizer must produce dinov2_feats; a None means a
            # real misconfiguration and we want to fail loudly. Inference: the
            # override defaults can legitimately disable the producing gate.
            assert not self.training, (
                f"[{self.__class__.__name__}] extra_ray_signals.dinov2_feats is None "
                f"during training; expected populated by the rasterizer."
            )
            return

        if rays is not None:
            assert pixel_coords is not None, (
                f"[{self.__class__.__name__}] No camera meta information found for camera rays!"
            )
            assert resolution is not None, f"{self.__class__.__name__} requires resolution"

            # Normalize pixel_coords to [0, 1] range, then to [-1, 1] for grid_sample
            w, h = resolution
            coords_normalized = pixel_coords / torch.tensor(
                [w, h], device=pixel_coords.device, dtype=pixel_coords.dtype
            )

            pe_map = self.pe_head(self.learnable_pe_map)
            pe_ray = (
                (
                    torch.nn.functional.grid_sample(
                        pe_map,
                        coords_normalized.reshape(1, 1, -1, 2) * 2 - 1,
                        align_corners=False,
                        mode="bilinear",
                        padding_mode="border",
                    )
                )
                .squeeze(2)
                .squeeze(0)
                .permute(1, 0)
            )

            extras.dinov2_feats = dinov2_feats + pe_ray


BasePostProcessing.register_variant("dinov2-pe-post-processing", DinoV2PEPostProcessing)


def create_bilateral_grid(config: BilateralGridPostProcessingConfig, n_frames: int):
    logger.info(f"Creating bilateral grid with {n_frames} frames")
    use_cuda = getattr(config, "use_cuda", True)
    CurrentBilateralGrid = BilateralGridCUDA if use_cuda else BilateralGrid
    return CurrentBilateralGrid(
        n_frames,
        config.width,
        config.height,
        config.depth,
        getattr(config, "enable_gridsize1_optimization", True),
    )


class BilateralGridPerFrame(BasePostProcessing):
    """
    Implementation of "Bilateral Guided Radiance Field Processing (Bilarf)".
    This approach follows the published work and learns one bilateral grid per frame.
    """

    config: BilateralGridPostProcessingConfig

    def __init__(
        self,
        config: BilateralGridPostProcessingConfig,
        trainer_config: TrainerConfig,
        n_training_frames_per_camera: List[int],
        n_training_frames_per_lidar: List[int],
    ):
        super().__init__(config, trainer_config, n_training_frames_per_camera, n_training_frames_per_lidar)

        self.bilateral_grid = create_bilateral_grid(self.config, sum(n_training_frames_per_camera))

        # This is used so that we don't apply temporal variation loss across frames belonging to different cameras
        self.same_camera_as_next_frame = nn.Buffer(
            torch.ones(sum(n_training_frames_per_camera) - 1, dtype=torch.bool), persistent=False
        )
        offset = 0
        for camera_n_frames in n_training_frames_per_camera[:-1]:
            self.same_camera_as_next_frame[offset + camera_n_frames - 1] = 0
            offset += camera_n_frames

    def forward(
        self,
        results: GaussiansCompositeReturn,
        rays: torch.Tensor | None = None,  # [N, 6]
        pixel_coords: torch.Tensor | None = None,  # [N, 2]
        resolution: tuple[int, int] | None = None,  # (width, height)
        training_frame_idx: int | None = None,
        unique_sensor_idx: int | None = None,
        global_step: int = 0,
    ) -> None:
        if rays is None:
            return

        assert results.rendered_cam is not None, "rendered_cam should not be None if rays is not None"
        if results.rendered_cam.extra_ray_signals is None:
            results.rendered_cam.extra_ray_signals = ExtraSignal()

        if results.rendered_cam.rgb is None:
            return
        rgb = unpack_optional(results.rendered_cam.rgb)

        # We check if rgb_before_post_processing is None since we potentially apply bilateral
        # grids both at on per-frame or per-camera level, and we want to store the rgb values
        # before any sort of post-processing takes place
        if results.rendered_cam.extra_ray_signals.rgb_before_post_processing is None:
            results.rendered_cam.extra_ray_signals.rgb_before_post_processing = rgb.clone()

        assert pixel_coords is not None, f"{self.__class__.__name__} requires pixel_coords"
        assert resolution is not None, f"{self.__class__.__name__} requires resolution"
        assert training_frame_idx is not None, f"{self.__class__.__name__} requires training_frame_idx"
        assert training_frame_idx == -1 or 0 <= training_frame_idx < self.bilateral_grid.num_grids, (
            f"training_frame_idx={training_frame_idx} out of valid range "
            f"[-1, 0..{self.bilateral_grid.num_grids - 1}] (num_frames={self.bilateral_grid.num_grids})"
        )

        # Bilateral grids need [0, 1] normalized coordinates (align_corners=True convention).
        # pixel_coords are pixel centers (k + 0.5), so subtract 0.5 and divide by (size - 1)
        # to map the first pixel center to 0 and the last to 1.
        w, h = resolution
        coords_xy = (pixel_coords - 0.5) / torch.tensor(
            [w - 1, h - 1], device=pixel_coords.device, dtype=pixel_coords.dtype
        )

        frame_idcs = torch.full((rgb.shape[0],), training_frame_idx, dtype=torch.int32, device=rgb.device)
        results.rendered_cam.rgb = self.bilateral_grid(rgb, coords_xy, frame_idcs)

        if (
            prober_result := prober(
                global_step,
                "bilateral_grid_per_frame",
                rgb=rgb,
                coords_xy=coords_xy,
                grid_idcs=frame_idcs,
                bilateral_grid=self.bilateral_grid.grid,
                output_rgb_grad=results.rendered_cam.rgb,
            )
        ) is not None:
            # Connect the gradient probing to the rest of the computation graph
            (results.rendered_cam.rgb,) = prober_result

    def configure_optimizers(self, name_prefix: str = "") -> List[OptimizerLRSchedulerConfig]:
        return configure_optimizers(self.config.to_dictconfig(), self.trainer_config, self, name_prefix)


BasePostProcessing.register_variant("bilateral-grid-per-frame", BilateralGridPerFrame)


class BilateralGridPerCamera(BasePostProcessing):
    """
    Learn one bilateral grid per camera.
    """

    config: BilateralGridPostProcessingConfig

    def __init__(
        self,
        config: BilateralGridPostProcessingConfig,
        trainer_config: TrainerConfig,
        n_training_frames_per_camera: List[int],
        n_training_frames_per_lidar: List[int],
    ):
        super().__init__(config, trainer_config, n_training_frames_per_camera, n_training_frames_per_lidar)

        self.num_cameras = len(n_training_frames_per_camera)
        if self.num_cameras == 1:
            logger.warning(
                "Single camera training with per-camera bilateral grids may reduce quality! It's recommended to disable per-camera bilateral grids in this configuration."
            )

        self.bilateral_grid = create_bilateral_grid(self.config, self.num_cameras)

    def forward(
        self,
        results: GaussiansCompositeReturn,
        rays: torch.Tensor | None = None,  # [N, 6]
        pixel_coords: torch.Tensor | None = None,  # [N, 2]
        resolution: tuple[int, int] | None = None,  # (width, height)
        training_frame_idx: int | None = None,
        unique_sensor_idx: int | None = None,
        global_step: int = 0,
    ) -> None:
        if rays is None:
            return

        assert results.rendered_cam is not None, "rendered_cam should not be None if rays_cam is not None"
        if results.rendered_cam.extra_ray_signals is None:
            results.rendered_cam.extra_ray_signals = ExtraSignal()

        if results.rendered_cam.rgb is None:
            return
        rgb = unpack_optional(results.rendered_cam.rgb)

        # We check if rgb_before_post_processing is None since we potentially apply bilateral
        # grids both at on per-frame or per-camera level, and we want to store the rgb values
        # before any sort of post-processing takes place
        if results.rendered_cam.extra_ray_signals.rgb_before_post_processing is None:
            results.rendered_cam.extra_ray_signals.rgb_before_post_processing = rgb.clone()

        assert unique_sensor_idx is not None, f"{self.__class__.__name__} requires unique_sensor_idx"
        assert unique_sensor_idx < self.num_cameras, "Camera indices out of bounds"
        assert pixel_coords is not None, f"{self.__class__.__name__} requires pixel_coords"
        assert resolution is not None, f"{self.__class__.__name__} requires resolution"

        # Bilateral grids need [0, 1] normalized coordinates (align_corners=True convention).
        # pixel_coords are pixel centers (k + 0.5), so subtract 0.5 and divide by (size - 1)
        # to map the first pixel center to 0 and the last to 1.
        w, h = resolution
        coords_xy = (pixel_coords - 0.5) / torch.tensor(
            [w - 1, h - 1], device=pixel_coords.device, dtype=pixel_coords.dtype
        )

        sensor_idcs = torch.full((rgb.shape[0],), unique_sensor_idx, dtype=torch.int32, device=rgb.device)
        transformed_rgb = self.bilateral_grid(rgb, coords_xy, sensor_idcs)
        results.rendered_cam.rgb = transformed_rgb

        if (
            prober_result := prober(
                global_step,
                "bilateral_grid_per_camera",
                rgb=rgb,
                coords_xy=coords_xy,
                grid_idcs=sensor_idcs,
                bilateral_grid=self.bilateral_grid.grid,
                output_rgb_grad=results.rendered_cam.rgb,
            )
        ) is not None:
            # Connect the gradient probing to the rest of the computation graph
            (results.rendered_cam.rgb,) = prober_result

    def configure_optimizers(self, name_prefix: str = "") -> List[OptimizerLRSchedulerConfig]:
        return configure_optimizers(self.config.to_dictconfig(), self.trainer_config, self, name_prefix)


BasePostProcessing.register_variant("bilateral-grid-per-camera", BilateralGridPerCamera)

BilateralGridT = TypeVar("BilateralGridT", BilateralGridPerFrame, BilateralGridPerCamera)


class VastAppearancePostProcessing(BasePostProcessing):
    """VastGaussian's decoupled appearance modeling (Lin et al., CVPR 2024, Sec. 4.2).

    During training only, the rendered image I_r (after any earlier post-processing) is downsampled by
    ``downsample``, concatenated per pixel with a learnable per-image embedding, and turned by a CNN (a 3x3 conv, four
    pixel-shuffle x2 upsampling blocks with 3x3 conv + ReLU, a bilinear resize to the full size, then two 3x3 convs)
    into a per-pixel, per-channel map M; the adjusted image is I_a = I_r * M. The L1 loss sees I_a, the SSIM loss sees
    I_r (``ExtraSignal.rgb_before_appearance``). Outside training (validation, novel views, export) it does nothing,
    so it is discarded after optimization. The last conv starts at zero, so M starts as 1.
    The CNN input is detached: the map compensates appearance and is not a path for geometry gradients.
    """

    config: VastAppearancePostProcessingConfig

    def __init__(
        self,
        config: VastAppearancePostProcessingConfig,
        trainer_config: TrainerConfig,
        n_training_frames_per_camera: List[int],
        n_training_frames_per_lidar: List[int],
    ):
        super().__init__(config, trainer_config, n_training_frames_per_camera, n_training_frames_per_lidar)
        n_frames = max(1, sum(n_training_frames_per_camera))
        dim = self.config.embedding_dim
        self.embeddings = nn.Parameter(torch.randn(n_frames, dim) * 0.01)
        channels = self.config.hidden_channels
        layers: list[nn.Module] = [nn.Conv2d(3 + dim, channels, 3, padding=1)]
        for _ in range(4):
            layers += [nn.PixelShuffle(2), nn.Conv2d(channels // 4, channels // 2, 3, padding=1), nn.ReLU(inplace=True)]
            channels //= 2
        self.upsample = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(channels, 3, 3, padding=1)
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        results: GaussiansCompositeReturn,
        rays: torch.Tensor | None = None,  # [N, 6]
        pixel_coords: torch.Tensor | None = None,  # [N, 2]
        resolution: tuple[int, int] | None = None,  # (width, height)
        training_frame_idx: int | None = None,
        unique_sensor_idx: int | None = None,
        global_step: int = 0,
    ) -> None:
        if not self.training or rays is None or resolution is None or training_frame_idx is None or training_frame_idx < 0:
            return
        cam = results.rendered_cam
        if cam is None or cam.rgb is None:
            return
        width, height = resolution
        rgb = cam.rgb
        if rgb.shape[0] != width * height:
            return  # only full-image batches can be treated as images
        if cam.extra_ray_signals is None:
            cam.extra_ray_signals = ExtraSignal()
        cam.extra_ray_signals.rgb_before_appearance = rgb
        image = rgb.reshape(height, width, 3).permute(2, 0, 1).unsqueeze(0)
        factor = self.config.downsample
        small = torch.nn.functional.interpolate(
            image.detach(), size=(max(1, height // factor), max(1, width // factor)), mode="area"
        )
        embedding = self.embeddings[training_frame_idx].view(1, -1, 1, 1).expand(-1, -1, *small.shape[2:])
        features = self.upsample(torch.cat([small, embedding], 1))
        features = torch.nn.functional.interpolate(features, size=(height, width), mode="bilinear", align_corners=False)
        transform = 1.0 + self.head(features)
        cam.rgb = (image * transform).squeeze(0).permute(1, 2, 0).reshape(-1, 3)

    def configure_optimizers(self, name_prefix: str = "") -> List[OptimizerLRSchedulerConfig]:
        return configure_optimizers(self.config.to_dictconfig(), self.trainer_config, self, name_prefix)


BasePostProcessing.register_variant("vast-appearance", VastAppearancePostProcessing)


# ppisp_legacy is intentionally retained: it is the target of the
# 26.5.58->26.5.59 checkpoint upgrade (upgrade_config_2605_58_to_2605_59_rename_ppisp_to_ppisp_legacy)
# and has live consumers in nurec/models. Removing NRend does not unblock deleting it.
class PPISPLegacyPostProcessing(BasePostProcessing):
    """
    Legacy PPISP post-processing module (render-only).

    Uses the old in-tree Slang-kernel PPISP implementation for rendering
    backwards compatibility with old checkpoints. Training is not supported.
    """

    def __init__(
        self,
        config: BasePostProcessingConfig,
        trainer_config: TrainerConfig,
        n_training_frames_per_camera: list[int],
        n_training_frames_per_lidar: list[int],
    ):
        super().__init__(config, trainer_config, n_training_frames_per_camera, n_training_frames_per_lidar)

        self.num_cameras = len(n_training_frames_per_camera)
        self.num_frames = sum(n_training_frames_per_camera)

        self.ppisp = PPISPLegacy(self.device, n_training_frames_per_camera, self.num_cameras, self.num_frames)

        # Version tracking
        self._ppisp_version: str = PPISP_VERSION_LEGACY

    def forward(
        self,
        results: GaussiansCompositeReturn,
        rays: torch.Tensor | None = None,  # [N, 6]
        pixel_coords: torch.Tensor | None = None,  # [N, 2]
        resolution: tuple[int, int] | None = None,  # (width, height)
        training_frame_idx: int | None = None,
        unique_sensor_idx: int | None = None,
        global_step: int = 0,
    ) -> None:
        if rays is None:
            return

        assert results.rendered_cam is not None, "rendered_cam should not be None if rays_cam is not None"
        if results.rendered_cam.extra_ray_signals is None:
            results.rendered_cam.extra_ray_signals = ExtraSignal()

        if results.rendered_cam.rgb is None:
            return
        rgb = unpack_optional(results.rendered_cam.rgb)

        if results.rendered_cam.extra_ray_signals.rgb_before_post_processing is None:
            results.rendered_cam.extra_ray_signals.rgb_before_post_processing = rgb.clone()

        assert pixel_coords is not None, f"{self.__class__.__name__} requires pixel_coords"
        assert resolution is not None, f"{self.__class__.__name__} requires resolution"

        # Legacy PPISP Slang kernel expects [0, 1] normalized coords (divide by size-1).
        # This matches the old convention trained checkpoints were optimized with.
        w, h = resolution
        coords_xy = pixel_coords / torch.tensor([w - 1, h - 1], device=pixel_coords.device, dtype=pixel_coords.dtype)

        assert unique_sensor_idx is not None, f"{self.__class__.__name__} requires unique_sensor_idx"
        assert unique_sensor_idx == -1 or 0 <= unique_sensor_idx < self.num_cameras, (
            f"unique_sensor_idx={unique_sensor_idx} out of valid range "
            f"[-1, 0..{self.num_cameras - 1}] (num_cameras={self.num_cameras})"
        )
        unique_camera_idcs = torch.full((rgb.shape[0],), unique_sensor_idx, dtype=torch.int16, device=rgb.device)

        assert training_frame_idx is not None, f"{self.__class__.__name__} requires training_frame_idx"
        assert training_frame_idx == -1 or 0 <= training_frame_idx < self.num_frames, (
            f"training_frame_idx={training_frame_idx} out of valid range "
            f"[-1, 0..{self.num_frames - 1}] (num_frames={self.num_frames})"
        )
        frame_idcs = torch.full((rgb.shape[0],), training_frame_idx, dtype=torch.int32, device=rgb.device)

        transformed_rgb = self.ppisp(rgb, coords_xy, unique_camera_idcs, frame_idcs)
        results.rendered_cam.rgb = transformed_rgb

    def get_extra_state(self) -> dict[str, Any]:
        return {"ppisp_version": self._ppisp_version}

    def set_extra_state(self, state: dict[str, Any] | None) -> None:
        if state is not None and "ppisp_version" in state:
            self._ppisp_version = state["ppisp_version"]
        else:
            self._ppisp_version = PPISP_VERSION_LEGACY

    def configure_optimizers(self, name_prefix: str = "") -> List[OptimizerLRSchedulerConfig]:
        raise RuntimeError(
            "Training with legacy PPISP is not supported. Legacy PPISP is render-only "
            "for backwards compatibility with old checkpoints. Use the new PPISP library "
            "for training."
        )


BasePostProcessing.register_variant("ppisp-post-processing-legacy", PPISPLegacyPostProcessing)


class PPISPPostProcessing(BasePostProcessing):
    """Post-processing module wrapping the new PPISP library.

    Supports both training and rendering.  Delegates to ``ppisp.PPISP`` for the
    ISP pipeline, optimizer/scheduler creation, and regularization loss.
    """

    config: PPISPPostProcessingConfig

    def __init__(
        self,
        config: PPISPPostProcessingConfig,
        trainer_config: TrainerConfig,
        n_training_frames_per_camera: list[int],
        n_training_frames_per_lidar: list[int],
    ):
        super().__init__(config, trainer_config, n_training_frames_per_camera, n_training_frames_per_lidar)

        self.num_cameras = len(n_training_frames_per_camera)
        self.num_frames = sum(n_training_frames_per_camera)

        self.ppisp = PPISPModule(
            num_cameras=self.num_cameras,
            num_frames=self.num_frames,
            config=self._to_ppisp_config(),
        )

        # Version tracking: store the library version
        self._ppisp_version: str = ppisp_lib.__version__

    def _to_ppisp_config(self) -> PPISPConfig:
        """Convert NuRec's pydantic config to the library's PPISPConfig."""
        cfg = self.config
        return PPISPConfig(
            use_controller=cfg.use_controller,
            controller_distillation=cfg.controller_distillation,
            controller_activation_ratio=cfg.controller_activation_ratio,
            exposure_mean=cfg.exposure_mean,
            vig_center=cfg.vig_center,
            vig_channel=cfg.vig_channel,
            vig_non_pos=cfg.vig_non_pos,
            color_mean=cfg.color_mean,
            crf_channel=cfg.crf_channel,
            ppisp_lr=cfg.ppisp_lr,
            ppisp_eps=cfg.ppisp_eps,
            ppisp_betas=cfg.ppisp_betas,
            controller_lr=cfg.controller_lr,
            controller_eps=cfg.controller_eps,
            controller_betas=cfg.controller_betas,
            scheduler_type=cfg.scheduler_type,
            scheduler_base_lr=cfg.scheduler_base_lr,
            scheduler_warmup_steps=cfg.scheduler_warmup_steps,
            scheduler_start_factor=cfg.scheduler_start_factor,
            scheduler_decay_max_steps=cfg.scheduler_decay_max_steps,
            scheduler_final_factor=cfg.scheduler_final_factor,
        )

    def forward(
        self,
        results: GaussiansCompositeReturn,
        rays: torch.Tensor | None = None,  # [N, 6]
        pixel_coords: torch.Tensor | None = None,  # [N, 2]
        resolution: tuple[int, int] | None = None,  # (width, height)
        training_frame_idx: int | None = None,
        unique_sensor_idx: int | None = None,
        global_step: int = 0,
    ) -> None:
        if rays is None:
            return

        assert results.rendered_cam is not None, "rendered_cam should not be None if rays_cam is not None"
        if results.rendered_cam.extra_ray_signals is None:
            results.rendered_cam.extra_ray_signals = ExtraSignal()

        if results.rendered_cam.rgb is None:
            return
        rgb = unpack_optional(results.rendered_cam.rgb)

        if results.rendered_cam.extra_ray_signals.rgb_before_post_processing is None:
            results.rendered_cam.extra_ray_signals.rgb_before_post_processing = rgb.clone()

        assert pixel_coords is not None, f"{self.__class__.__name__} requires pixel_coords"
        assert resolution is not None, f"{self.__class__.__name__} requires resolution"
        assert unique_sensor_idx is not None, f"{self.__class__.__name__} requires unique_sensor_idx"
        assert 0 <= unique_sensor_idx < self.num_cameras, (
            f"unique_sensor_idx={unique_sensor_idx} out of valid range "
            f"[0..{self.num_cameras - 1}] (num_cameras={self.num_cameras})"
        )

        assert training_frame_idx is not None, f"{self.__class__.__name__} requires training_frame_idx"
        assert training_frame_idx == -1 or (0 <= training_frame_idx < self.num_frames), (
            f"training_frame_idx={training_frame_idx} out of valid range "
            f"[-1, 0..{self.num_frames - 1}] (num_frames={self.num_frames})"
        )

        # The new library expects pixel coordinates and (W, H) resolution
        transformed_rgb = self.ppisp(
            rgb,
            pixel_coords,
            resolution=resolution,
            camera_idx=unique_sensor_idx,
            frame_idx=training_frame_idx,
        )

        results.rendered_cam.rgb = transformed_rgb

    def forward_rendered_batch(
        self,
        results: GaussiansCompositeReturn,
        batch: DataAndRenderingBatch,
        *,
        global_step: int = 0,
        is_training_batch: bool = False,
    ) -> None:
        """Apply PPISP to a rendered camera batch.

        This is used by controller distillation, where the scene is rendered
        without autograd first and only PPISP/controller remains in the
        gradient path.
        """
        assert batch.rendering is not None, (
            "PPISPPostProcessing.forward_rendered_batch requires batch.rendering. "
            "Controller distillation must run after view-geometry preprocessing."
        )
        assert batch.rendering.camera is not None, (
            "PPISPPostProcessing.forward_rendered_batch requires batch.rendering.camera. "
            "Controller distillation only supports camera batches."
        )
        assert batch.data.camera is not None, (
            "PPISPPostProcessing.forward_rendered_batch requires batch.data.camera. "
            "Controller distillation only supports camera batches."
        )

        assert batch.rendering.camera.b == 1, "PPISP post-processing only supports batch size 1."

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
        self(
            results=results,
            rays=batch.rendering.camera.rays.reshape(-1, 6),
            pixel_coords=pixel_coords.reshape(-1, 2),
            resolution=subsample.downsampled_resolution(),
            training_frame_idx=batch.data.camera.meta[0].training_frame_idx,
            unique_sensor_idx=batch.data.camera.meta[0].unique_sensor_idx,
            global_step=global_step,
        )

    def freeze_for_controller_distillation(self) -> None:
        """Freeze PPISP static params and leave only controller params trainable."""
        self.requires_grad_(False)
        controllers = getattr(self.ppisp, "controllers", None)
        if controllers is None:
            raise RuntimeError(
                "PPISP controller distillation is enabled, but the PPISP module has no controllers. "
                "Enable PPISP controllers or disable controller distillation."
            )
        controllers.requires_grad_(True)

    def get_extra_state(self) -> dict[str, Any]:
        return {"ppisp_version": self._ppisp_version}

    def set_extra_state(self, state: dict[str, Any] | None) -> None:
        if state is not None and "ppisp_version" in state:
            self._ppisp_version = state["ppisp_version"]
        else:
            self._ppisp_version = PPISP_VERSION_LEGACY

    def configure_optimizers(self, name_prefix: str = "") -> List[OptimizerLRSchedulerConfig]:
        max_steps = self.trainer_config.max_steps
        assert max_steps > 0, "trainer_config.max_steps must be set before configuring PPISP optimizers"

        ppisp_optimizers = self.ppisp.create_optimizers()
        ppisp_schedulers = self.ppisp.create_schedulers(ppisp_optimizers, max_steps)

        result: List[OptimizerLRSchedulerConfig] = []

        # First optimizer: PPISP main params with scheduler
        result.append(
            {
                "optimizer": ppisp_optimizers[0],
                "lr_scheduler": {
                    "scheduler": ppisp_schedulers[0],
                    "interval": "step",
                },
            }
        )

        # Second optimizer: controller (if present) — fixed LR, no scheduler
        if self.config.use_controller:
            result.append(
                {
                    "optimizer": ppisp_optimizers[1],
                }
            )
        return result


BasePostProcessing.register_variant("ppisp-post-processing", PPISPPostProcessing)


def get_ppisp_from_model(model: Any) -> PPISPModule | None:
    """Extract the PPISP library module from the model.

    Returns the ``ppisp.PPISP`` module if the model has a new-library
    ``PPISPPostProcessing``, or ``None`` if no PPISP is present.

    Raises ``RuntimeError`` for legacy PPISP (not supported for export).
    Logs an informational note if the controller is active; the export
    command authors the controller graph by default in ``spg-runtime``
    mode unless the caller forces the static path.
    """
    if not hasattr(model, "post_processings"):
        return None

    for pp in model.post_processings:
        if isinstance(pp, PPISPLegacyPostProcessing):
            raise RuntimeError(
                "SPG export is not supported for legacy PPISP checkpoints. "
                "Retrain with the new PPISP library to enable SPG export."
            )
        if isinstance(pp, PPISPPostProcessing):
            if pp.config.use_controller:
                logger.info(
                    "PPISP controller is enabled in the model. spg-runtime export "
                    "will use the controller graph unless explicitly disabled."
                )
            return pp.ppisp

    return None
