# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional, Tuple, Type

import cv2
import numpy as np
import tinycudann as tcnn
import torch

from gsplat.scene.components.env_map_background import (
    EnvMapBackground as GsplatEnvMapBackground,
)
from python.runfiles import runfiles
from simple_lama_inpainting import SimpleLama
from torch import nn

from nurec.config.base_schema import config_to_primitive
from nurec.config.model import (
    BackgroundColorConfig,
    BaseBackgroundConfig,
    ExtraSignalSkyMlpBackgroundConfig,
    SkyEnvMapBackgroundConfig,
    SkyMlpBackgroundConfig,
)
from nurec.config.trainer import TrainerConfig
from nurec.models.background_sampling import cubemap_coverage_seamless, sample_cubemap_seamless
from nurec.models.base import BaseModel
from nurec.models.nn_extensions import module_call_type
from nurec.models.utils import eval_tcnn_network, get_activation
from nurec.utils.cuda_rng import get_cuda_generator
from nurec.utils.geometry import cartesian_to_spherical
from nurec.utils.misc import linear_to_srgb, srgb_to_linear, unpack_optional
from nurec.utils.tcnn_allocator import use_torch_allocator_for_tcnn
from nurec.utils.trainer import adjust_step_for_world_size
from nurec.utils.types import ExtraSignal, GaussiansCompositeReturn


RUNFILES = runfiles.Create()

use_torch_allocator_for_tcnn(tcnn)


class BaseBackground(BaseModel, ABC):
    composite_in_linear_space: bool
    config: BaseBackgroundConfig  # type: ignore[assignment] # Override type annotation from BaseModel

    @staticmethod
    def factory(name: str, config: BaseBackgroundConfig, trainer_config: TrainerConfig) -> BaseBackground:
        variants: dict[str, Type[BaseBackground]] = {
            "sky-mlp": SkyMlpBackground,
            "sky-env-map": SkyEnvMapBackground,
            "extra-signal-sky-mlp": ExtraSignalSkyMlpBackground,
            "background-color": BackgroundColor,
            "skip-background": SkipBackground,
        }
        return variants[name](config, trainer_config)

    def __init__(self, config: BaseBackgroundConfig, trainer_config: TrainerConfig) -> None:
        super().__init__(config.to_dictconfig())
        self.config = config  # type: ignore[assignment] # Override with typed config
        self.trainer_config = trainer_config
        self.composite_in_linear_space = self.config.composite_in_linear_space

    def combine_rgb_colors(self, results: GaussiansCompositeReturn, background_rgb: torch.Tensor) -> None:
        """Compute the linear combination of RGB colors within the volume and background"""
        assert results.rendered_cam is not None, "rendered_cam should not be None"
        assert results.rendered_cam.rgb is not None, "rendered_cam.rgb should not be None"
        if results.rendered_cam.extra_ray_signals is None:
            results.rendered_cam.extra_ray_signals = ExtraSignal()

        if self.composite_in_linear_space:
            rgb_linear = srgb_to_linear(results.rendered_cam.rgb)
            background_rgb_linear = srgb_to_linear(background_rgb)
            background_rgb_linear_scaled = background_rgb_linear * (1.0 - results.rendered_cam.opacity[..., None])
            results.rendered_cam.rgb = linear_to_srgb(rgb_linear + background_rgb_linear_scaled)
            results.rendered_cam.extra_ray_signals.rgb_background = background_rgb_linear
        else:
            background_color = background_rgb * (1.0 - results.rendered_cam.opacity[..., None])
            results.rendered_cam.rgb = results.rendered_cam.rgb + background_color
            results.rendered_cam.extra_ray_signals.rgb_background = background_rgb

    @abstractmethod
    def forward(
        self,
        rays_d: torch.Tensor,
        results: GaussiansCompositeReturn,
        is_training: bool,
    ) -> None: ...

    __call__ = module_call_type(forward)


class EnvMapType(Enum):
    EQUIRECTANGULAR = "equirectangular"
    CUBEMAP = "cubemap"


class SkyEnvMapBackground(BaseBackground, GsplatEnvMapBackground):  # type: ignore[misc] # nn.Module.parameters() wins via MRO (see docstring), overriding gsplat's list[Parameter]
    """
    Models the background with an environment map which can either be modeled as a cubemap or with equirectangular projection.
    We also support inpainting unobserved regions of the scene (which we track via gradient updates) through Lama.

    The direction-dependent radiance uses gsplat's environment-map component and
    NuRec's shared seamless cubemap sampler. NuRec-side pieces are preserved: it stays
    an ``nn.Module``/``BaseBackground`` (so Lightning owns the ``textures``
    Parameter and buffers), keeps the ``to_opengl`` basis change, LAMA inpainting,
    USD export helpers and the equirectangular convention (``_sample_equirect`` is
    overridden to match NuRec's historical texture layout so existing checkpoints
    keep rendering identically).

    Because ``BaseBackground`` (an ``nn.Module``) precedes ``GsplatEnvMapBackground``
    in the MRO, ``parameters()``/``state_dict()`` resolve to ``nn.Module`` (needed by
    Lightning). Sampling builds on gsplat's environment-map component, while
    gradient tracking is implemented locally by ``_track_texture_grads``.
    """

    config: SkyEnvMapBackgroundConfig  # type: ignore[assignment] # narrow from BaseBackgroundConfig
    # gsplat's EnvMapBackground infers ``envmap_type`` as its own EnvMapType; re-annotate
    # with NuRec's enum so the ``match`` statements below stay exhaustive (same override
    # pattern as ``config`` above / in BaseBackground).
    envmap_type: EnvMapType  # type: ignore[assignment]

    def __init__(self, config: SkyEnvMapBackgroundConfig, trainer_config: TrainerConfig):
        # Initialise the nn.Module / BaseBackground surface. gsplat's
        # EnvMapBackground.__init__ is intentionally NOT invoked: it would overwrite
        # self.config with its own dataclass and register `textures` as a plain
        # attribute. Instead we replicate its (gsplat-identical) texture setup here so
        # Lightning owns the Parameter and buffers. The texture shapes and the neutral
        # 0.5 initialisation match gsplat's EnvMapBackground exactly.
        BaseBackground.__init__(self, config, trainer_config)

        self.activation = (
            get_activation("saturate") if getattr(config, "saturate_radiance", True) else get_activation("relu")
        )
        # Attributes read by the inherited gsplat samplers and local gradient tracking.
        self.envmap_type = EnvMapType(config.envmap_type)  # NuRec enum (used by USD export + TV loss)
        self.saturate_radiance = bool(getattr(config, "saturate_radiance", True))
        self.width = int(config.width)
        self.height = int(config.height)

        self.min_grad_updates = adjust_step_for_world_size(trainer_config, self.config.min_grad_updates)
        logging.getLogger(__name__).info(f"SkyEnvMapBackground: min_grad_updates={self.min_grad_updates}")

        self.grad_track_interval = int(self.config.grad_track_interval)

        match self.envmap_type:
            case EnvMapType.EQUIRECTANGULAR:
                self.textures = torch.nn.Parameter(
                    torch.full((1, self.config.height, self.config.width, 3), 0.5), requires_grad=True
                )
            case EnvMapType.CUBEMAP:
                assert self.config.height == self.config.width, (
                    f"[{self.__class__.__name__}] Height and width must be identical when using cubemap"
                )
                self.textures = torch.nn.Parameter(
                    torch.full((1, 6, self.config.height, self.config.width, 3), 0.5), requires_grad=True
                )

        # Running max of per-texel coverage (see _track_texture_grads). Persistent buffer
        # (in state_dict, moved by .to()); only ever updated in place, so the buffer
        # registration stays intact.
        self.texture_grads = torch.nn.Buffer(torch.zeros_like(self.textures.squeeze(0)[..., 0]))

        # Tracks how many times the environment map has been updated. We generally don't want to inpaint the environment map
        # until the scene has converged somewhat
        self.n_grad_updates = 1  # Keep as a plain int. Using nn.Buffer(torch.tensor(...)) causes a GPU to CPU sync during conditional checks (e.g., if/assert).

        # NuRec: x forward, y left, z up
        # OpenGL: x right, y up, z back
        # gsplat's samplers use the OpenGL cube-map convention, so we rotate rays into
        # OpenGL before sampling. This is not required for volume rendering correctness
        # but makes it easier to reason about the cube-map faces and inpainting.
        self.to_opengl = nn.Buffer(torch.FloatTensor([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]), persistent=False)

        self.original_textures = (
            nn.Buffer(torch.zeros_like(self.textures), persistent=False) if self.config.should_inpaint else None
        )

    def get_extra_state(self) -> dict[str, Any]:
        return {"n_grad_updates": self.n_grad_updates}

    def set_extra_state(self, state_dict: dict[str, Any]) -> None:
        if "n_grad_updates" in state_dict:
            self.n_grad_updates = state_dict["n_grad_updates"]
        else:
            logging.getLogger(__name__).warning(
                f"[{self.__class__.__name__}] Key 'n_grad_updates' not found in state_dict. "
                "This may occur when loading checkpoints from older versions. Using default value of 1."
            )
            self.n_grad_updates = 1

    def forward(
        self,
        rays_d: torch.Tensor,
        results: GaussiansCompositeReturn,
        is_training: bool,
        texture_update_mask: Optional[torch.Tensor] = None,
    ) -> None:
        assert results.rendered_cam is not None, "rendered_cam should not be None"
        # Rotate world rays into OpenGL before sampling (gsplat's samplers use the
        # OpenGL cube-map convention).
        rays_d = rays_d @ self.to_opengl.T

        raw_rgb = self._sample_raw(rays_d)
        if texture_update_mask is not None:
            texture_update_mask = texture_update_mask.reshape(-1, 1)
            assert texture_update_mask.shape[0] == raw_rgb.shape[0]
            # Preserve the sampled values for compositing, but remove the autograd
            # connection to the texture for all non-sky rays.
            raw_rgb = torch.where(texture_update_mask, raw_rgb, raw_rgb.detach())
        background_rgb = self.activation(raw_rgb).to(rays_d.dtype)

        self._track_texture_grads(raw_rgb, rays_d, results.rendered_cam.opacity, texture_update_mask, is_training)

        self.combine_rgb_colors(results, background_rgb)

    def _track_texture_grads(
        self,
        raw_rgb: torch.Tensor,
        rays_d: torch.Tensor,
        opacity: torch.Tensor,
        texture_update_mask: Optional[torch.Tensor],
        is_training: bool,
    ) -> None:
        """Accumulate per-texel coverage into ``texture_grads`` to detect unobserved texels.

        Coverage is the texture gradient of the sampled radiance with ``1 - opacity`` as
        the cotangent: how strongly sky rays observe each texel. ``texture_grads`` keeps
        its running max.

        Same gate, cadence and DDP averaging as gsplat's ``_track_gradients``; only how
        per-texel magnitude is computed differs. Cubemaps call the sampler's VJP directly
        instead of ``autograd.grad`` to avoid the autograd worker hand-off. The gate reads
        only counters that advance identically on every rank, so all ranks enter the
        ``all_reduce`` on the same steps.
        """
        # ``raw_rgb`` can still require gradients through trainable rays when the texture is
        # frozen, so check the texture explicitly before either tracking path.
        if not (is_training and self.textures.requires_grad and raw_rgb.requires_grad):
            return
        warmed_up = self.n_grad_updates >= self.min_grad_updates
        due = self.n_grad_updates % self.grad_track_interval == 0
        if warmed_up and due:
            weights = 1 - opacity.detach().reshape(-1)
            if texture_update_mask is not None:
                # Non-sky rays were detached in ``forward``, which autograd honours but the
                # direct VJP cannot see. Zero their weights so both paths agree.
                weights = torch.where(texture_update_mask.reshape(-1), weights, 0.0)
            match self.envmap_type:
                case EnvMapType.CUBEMAP:
                    # Use the same ray dtype as _sample_raw so coverage lands on the sampled texels.
                    coverage = cubemap_coverage_seamless(self.height, rays_d.to(self.textures.dtype), weights)
                case EnvMapType.EQUIRECTANGULAR:
                    grad_outputs = weights[:, None].expand_as(raw_rgb)
                    texture_grad = torch.autograd.grad(raw_rgb, self.textures, grad_outputs, retain_graph=True)[0]
                    coverage = texture_grad.squeeze(0).mean(dim=-1)
            if (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
                and torch.distributed.get_world_size() > 1
            ):
                coverage.div_(torch.distributed.get_world_size())
                torch.distributed.all_reduce(coverage, op=torch.distributed.ReduceOp.SUM)
            torch.maximum(coverage, self.texture_grads, out=self.texture_grads)
        self.n_grad_updates += 1

    def sample(self, rays_d: torch.Tensor) -> torch.Tensor:
        """Return background RGB ``[N, 3]`` for OpenGL-frame ray directions ``[N, 3]``.

        Cubemap sampling uses NuRec's seamless OpenGL-convention sampler; equirectangular
        sampling is overridden below to preserve NuRec's historical texture layout.
        NuRec's ``activation`` (saturate/relu) is applied to match prior behaviour
        exactly.

        The fused cubemap kernel is float32-only, but under autocast (mixed precision
        serving/training) ``rays_d`` arrives as float16. Sample in the texture dtype
        (float32) and restore the caller's dtype to keep downstream compositing
        unchanged.
        """
        return self.activation(self._sample_raw(rays_d)).to(rays_d.dtype)

    def _sample_raw(self, rays_d: torch.Tensor) -> torch.Tensor:
        """Sample pre-activation radiance in the texture dtype."""
        rays_d = rays_d.to(self.textures.dtype)
        match self.envmap_type:
            case EnvMapType.EQUIRECTANGULAR:
                return self._sample_equirect(rays_d)
            case EnvMapType.CUBEMAP:
                return self._sample_cubemap(rays_d)

    def _sample_cubemap(self, rays_d: torch.Tensor) -> torch.Tensor:
        """Sample the cubemap with seamless cross-face filtering."""
        return sample_cubemap_seamless(self.textures.squeeze(0), rays_d)

    def _sample_equirect(self, rays_d: torch.Tensor) -> torch.Tensor:
        """Equirectangular sampling in NuRec's historical convention.

        NuRec maps the polar angle ``acos(z)`` to the horizontal (width) axis and the
        azimuth ``atan2(x, y)`` to the vertical (height) axis. The texture is
        circularly padded by one texel on both axes for bilinear filtering across the
        seams, and the grid is offset to align with ``align_corners=False``
        half-texel centres.
        """
        phi, theta, _ = cartesian_to_spherical(rays_d)  # phi = acos(z) polar, theta = atan2(x, y) azimuth
        u = phi / torch.pi * 2 - 1  # horizontal (width): polar, [-1, 1]
        v = theta / torch.pi  # vertical (height): azimuth, [-1, 1]

        H, W = self.height, self.width
        # Convert normalized coordinates to the periodic [0, 1) interval.
        fu = u - torch.floor(u)  # -> [0, 1)
        fv = v - torch.floor(v)  # -> [0, 1)
        # Circular pad by 1 on each side of both axes: [1..W] real, col 0 == col W-1,
        # col W+1 == col 0 (and likewise for rows).
        tex = self.textures.permute(0, 3, 1, 2)  # [1, 3, H, W]
        tex = torch.cat([tex[..., -1:], tex, tex[..., :1]], dim=-1)  # wrap width -> [1, 3, H, W+2]
        tex = torch.cat([tex[..., -1:, :], tex, tex[..., :1, :]], dim=-2)  # wrap height -> [1, 3, H+2, W+2]
        # Map fractional coords into the padded interior at half-texel centres.
        gu = 2.0 * (fu * W + 1.0) / (W + 2.0) - 1.0
        gv = 2.0 * (fv * H + 1.0) / (H + 2.0) - 1.0
        grid = torch.stack([gu, gv], dim=-1)[None, None].to(tex.dtype)  # [1, 1, N, 2]
        rgb = torch.nn.functional.grid_sample(
            tex,
            grid,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        )
        return rgb.squeeze(0).squeeze(1).T  # [N, 3]

    def maybe_inpaint(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Potentially inpaints the textures and returns the inpainted textures and mask for logging.
        The inpainted textures will then be used until restore_original_textures is called (typically when
        resuming training).
        """

        if not self._should_inpaint:
            return None

        unpack_optional(self.original_textures).copy_(self.textures)
        match self.envmap_type:
            case EnvMapType.EQUIRECTANGULAR:
                texture_grads = self.texture_grads.data
            case EnvMapType.CUBEMAP:
                texture_grads = self._flatten_cube(self.texture_grads.data)

        inpaint_mask = texture_grads < self.config.inpaint_threshold

        kernel = np.ones((self.config.inpaint_kernel_size, self.config.inpaint_kernel_size), np.uint8)
        inpaint_mask = torch.BoolTensor(cv2.dilate(inpaint_mask.byte().cpu().numpy(), kernel, iterations=1))

        textures = self.flattened_textures().clamp(0, 1)
        # Use the big-lama weights fetched at build time rather than letting
        # simple_lama_inpainting download them from GitHub at runtime. The LAMA_MODEL
        # env var makes SimpleLama load this file directly and skip the download.
        lama_model_path = RUNFILES.Rlocation("lama_model/file/big-lama.pt")
        if not lama_model_path or not os.path.exists(lama_model_path):
            raise FileNotFoundError(f"big-lama weights not found in runfiles: {lama_model_path}")
        os.environ["LAMA_MODEL"] = lama_model_path
        simple_lama = SimpleLama(textures.device)
        inpainted_textures = (
            simple_lama.model(
                textures.permute(2, 0, 1).unsqueeze(0),
                inpaint_mask.to(textures.device).unsqueeze(0).unsqueeze(0).float(),
            )
            .detach()
            .squeeze(0)
            .permute(1, 2, 0)
        )

        match self.envmap_type:
            case EnvMapType.EQUIRECTANGULAR:
                self.textures.copy_(inpainted_textures.unsqueeze(0))
            case EnvMapType.CUBEMAP:
                inpainted_texture_list = []
                for row in range(2):
                    for col in range(3):
                        inpainted_texture_list.append(
                            inpainted_textures[
                                row * self.textures.shape[2] : (row + 1) * self.textures.shape[2],
                                col * self.textures.shape[3] : (col + 1) * self.textures.shape[3],
                            ]
                        )
                self.textures.copy_(torch.stack(inpainted_texture_list).unsqueeze(0))

        return inpainted_textures, inpaint_mask

    def restore_original_textures(self) -> None:
        if self._should_inpaint:
            self.textures.copy_(unpack_optional(self.original_textures))

    def flattened_textures(self) -> torch.Tensor:
        textures = self.textures.squeeze(0)
        match self.envmap_type:
            case EnvMapType.EQUIRECTANGULAR:
                return textures
            case EnvMapType.CUBEMAP:
                return self._flatten_cube(textures)

    @property
    def _should_inpaint(self) -> bool:
        return self.config.should_inpaint and self.n_grad_updates > self.min_grad_updates

    def _flatten_cube(self, tensor: torch.Tensor) -> torch.Tensor:
        """Flattens a (6, H, W, ...) cube map into a (2 * H, 3 * W, ...) tensor"""
        return torch.cat([torch.cat([x for x in tensor[:3]], dim=1), torch.cat([x for x in tensor[3:]], dim=1)], dim=0)

    __call__ = module_call_type(forward)


def ground_fill(
    rays_d: torch.Tensor,
    background_rgb: torch.Tensor,
    ground_rgb: list[float],
    up: list[float],
    horizon_deg: float,
    blend_deg: float,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Blend ``rgb`` ``[N, 3]`` (named ``background_rgb``) towards ``ground_rgb`` for world rays ``[N, 3]`` below the horizon.

    Rays at or above elevation ``horizon_deg`` (relative to ``up``'s horizontal plane) keep their colour. Below it
    the ground weight rises linearly and reaches 1 at ``horizon_deg - blend_deg``, so no hard line shows where sky
    meets ground fill and nothing above the horizon changes. ``mask`` ``[N]`` (boolean, or a weight in [0, 1])
    multiplies the weight, e.g. to fill only masked pixels, and only the see-through part of each.
    """
    up_t = torch.tensor(up, dtype=rays_d.dtype, device=rays_d.device)
    up_t = up_t / up_t.norm()
    sin_elevation = (rays_d @ up_t) / rays_d.norm(dim=-1).clamp_min(1e-12)
    elevation_deg = torch.rad2deg(torch.asin(sin_elevation.clamp(-1.0, 1.0)))
    weight = ((horizon_deg - elevation_deg) / blend_deg).clamp(0.0, 1.0)[:, None]
    if mask is not None:
        weight = weight * mask.reshape(-1, 1).to(device=weight.device, dtype=weight.dtype)
    ground = torch.tensor(ground_rgb, dtype=background_rgb.dtype, device=background_rgb.device).expand_as(background_rgb)
    return torch.lerp(background_rgb, ground, weight.to(background_rgb.dtype))


def erode_mask(mask: torch.Tensor, radius_px: int) -> torch.Tensor:
    """Shrink a boolean ``[H, W]`` mask: a pixel stays set only if every pixel within ``radius_px`` (square) is set.

    Pixels beyond the image border count as set, so a mask touching the border (an ego hood) is not eaten from there.
    """
    if radius_px <= 0:
        return mask
    unset = (~mask).to(torch.float32)[None, None]
    grown = torch.nn.functional.max_pool2d(unset, kernel_size=2 * radius_px + 1, stride=1, padding=radius_px)
    return ~(grown[0, 0] > 0)


class SkyMlpBackground(BaseBackground):
    config: SkyMlpBackgroundConfig  # narrow from BaseBackgroundConfig

    def __init__(self, config: SkyMlpBackgroundConfig, trainer_config: TrainerConfig):
        super().__init__(config, trainer_config)

        self.dir_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config=config_to_primitive(self.config.dir_encoding_config),
            dtype=torch.float,
            seed=self.config.seed,
        )
        self.dir_encoding.jit_fusion = self.config.enable_jit_if_supported and tcnn.supports_jit_fusion()

        self.sky_mlp = tcnn.Network(
            n_input_dims=self.dir_encoding.n_output_dims,
            n_output_dims=3,
            network_config=config_to_primitive(self.config.mlp_network_config),
            seed=self.config.seed,
        )
        self.sky_mlp.jit_fusion = self.config.enable_jit_if_supported and tcnn.supports_jit_fusion()

    def forward(
        self,
        rays_d: torch.Tensor,
        results: GaussiansCompositeReturn,
        is_training: bool,
    ) -> None:
        dir_encoding = self.dir_encoding((rays_d.reshape(-1, 3) + 1) / 2)  # rays_d assumed to be normalized
        background_rgb = eval_tcnn_network(self.sky_mlp, dir_encoding).float()

        self.combine_rgb_colors(results, background_rgb)

    __call__ = module_call_type(forward)


class ExtraSignalSkyMlpBackground(SkyMlpBackground):
    config: ExtraSignalSkyMlpBackgroundConfig  # narrow from SkyMlpBackgroundConfig

    _SUPPORTED_SIGNAL_TYPES = ("semantic_logits", "dinov2_feats")

    def __init__(self, config: ExtraSignalSkyMlpBackgroundConfig, trainer_config: TrainerConfig):
        super().__init__(config, trainer_config)

        self.signal_type = self.config.extra_signal.type
        if self.signal_type not in self._SUPPORTED_SIGNAL_TYPES:
            raise ValueError(
                f"[{self.__class__.__name__}] unknown signal type {self.signal_type!r}; "
                f"expected one of {self._SUPPORTED_SIGNAL_TYPES}."
            )

        self.extra_signal_sky_mlp = tcnn.Network(
            n_input_dims=self.dir_encoding.n_output_dims,
            n_output_dims=self.config.extra_signal.n_output_dims,
            network_config=config_to_primitive(self.config.extra_signal.mlp_network_config),
            seed=self.config.seed,
        )
        self.extra_signal_sky_mlp.jit_fusion = self.config.enable_jit_if_supported and tcnn.supports_jit_fusion()

    def forward(
        self,
        rays_d: torch.Tensor,
        results: GaussiansCompositeReturn,
        is_training: bool,
    ) -> None:
        super().forward(rays_d, results, is_training)  # writes rgb independently of extras below

        rendered_cam = results.rendered_cam
        if rendered_cam is None or rendered_cam.extra_ray_signals is None:
            # Training: the rasterizer must populate this slot; a None means a real
            # misconfiguration and we want to fail loudly. Inference: the override
            # defaults can legitimately disable the producing gate — rgb is already
            # written by super().forward(), so early-return cleanly.
            assert not is_training, (
                f"[{self.__class__.__name__}] extra_ray_signals is None during training; "
                f"expected populated by the rasterizer for signal_type={self.signal_type!r}."
            )
            return
        extras = rendered_cam.extra_ray_signals

        # signal_type is validated in __init__; the match is exhaustive.
        current: torch.Tensor | None
        match self.signal_type:
            case "semantic_logits":
                current = extras.semantic_logits
            case "dinov2_feats":
                current = extras.dinov2_feats
            case _:
                raise ValueError(f"[{self.__class__.__name__}] unknown signal type {self.signal_type}")

        if current is None:
            assert not is_training, (
                f"[{self.__class__.__name__}] extra_ray_signals.{self.signal_type} is None "
                f"during training; expected populated by the rasterizer."
            )
            return

        # Input order: [v, h]
        dir_encoding = self.dir_encoding((rays_d.reshape(-1, 3) + 1) / 2)  # rays_d assumed to be normalized
        background_extra_signal = eval_tcnn_network(self.extra_signal_sky_mlp, dir_encoding).float()

        match self.signal_type:
            case "semantic_logits":
                extras.semantic_logits = current + background_extra_signal * (1.0 - rendered_cam.opacity[..., None])
            case "dinov2_feats":
                extras.dinov2_feats = current + background_extra_signal * (
                    1.0 - rendered_cam.opacity[..., None].detach()
                )
                # Background is actually the mask.
                extras.dinov2_mask = rendered_cam.opacity > 0.8

    __call__ = module_call_type(forward)


class BackgroundColor(BaseBackground):
    config: BackgroundColorConfig  # narrow from BaseBackgroundConfig

    def __init__(self, config: BackgroundColorConfig, trainer_config: TrainerConfig):
        super().__init__(config, trainer_config)

        self.background_color_type = self.config.color

        assert self.background_color_type in [
            "white",
            "black",
            "random",
        ], f"[{self.__class__.__name__}] Background color must be one of 'white', 'black', 'random'"

        self.color: Optional[nn.Buffer]
        if self.background_color_type == "white":
            self.color = nn.Buffer(torch.ones((3,), dtype=torch.float32, device=self.device), persistent=False)
        elif self.background_color_type == "black":
            self.color = nn.Buffer(torch.zeros((3,), dtype=torch.float32, device=self.device), persistent=False)
        else:
            self.color = None

        self.random_eval_color: Optional[nn.Buffer]
        if self.background_color_type == "random":
            self.random_eval_color = nn.Buffer(torch.FloatTensor(self.config.random_eval_color), persistent=False)
            assert self.random_eval_color.min() >= 0 and self.random_eval_color.max() <= 1, (
                "random_eval_color values should be between 0 and 1"
            )
        else:
            self.random_eval_color = None

    def forward(
        self,
        rays_d: torch.Tensor,
        results: GaussiansCompositeReturn,
        is_training: bool,
    ) -> None:
        if self.background_color_type == "random":
            if is_training:
                # Sample via the process-wide CUDA Generator (see
                # `nurec/utils/cuda_rng.py`) so this hot-path call does not
                # advance the CUDA default generator's Philox offset --
                # required for `torch.cuda.graph(...)` capture.
                color = torch.rand(
                    (3,),
                    generator=get_cuda_generator(self.device) if self.device.type == "cuda" else None,
                    dtype=torch.float32,
                    device=self.device,
                )
            else:
                color = unpack_optional(self.random_eval_color)
        else:
            color = unpack_optional(self.color)

        self.combine_rgb_colors(results, color)

    __call__ = module_call_type(forward)


class SkipBackground(BaseBackground):
    def forward(
        self,
        rays_d: torch.Tensor,
        results: GaussiansCompositeReturn,
        is_training: bool,
    ) -> None:
        pass

    __call__ = module_call_type(forward)
