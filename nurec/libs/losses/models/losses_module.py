# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Layer 2: Losses model layer - Neural network modules for fused CUDA losses."""

import logging
import math

from typing import Any, ClassVar

import omegaconf
import torch

from nurec.datasets.tracks import CuboidTracks
from nurec.libs.losses.functional.cuda_losses_function import (
    GRID_NUM_CHANNELS,
    SCALAR_LOSS_OUTPUT_INDEX,
    CudaLossesFunction,
)
from nurec.libs.losses.models.reduce_functions import SumReduceFn
from nurec.libs.losses.models.utils import (
    _get_bilateral_grids,
    _get_out_of_bound_gaussian_nodes,
    _maybe_update_mcmc_visibility_counters,
    fused_ssim_loss_map,
    get_rendered_visibility_mask,
)
from nurec.libs.losses.orchestration.config import LossReturn
from nurec.models.background import SkyEnvMapBackground
from nurec.models.base import BaseModel
from nurec.models.gaussians.gaussians_composite import GaussiansComposite
from nurec.models.gaussians.gaussians_model import RigidGaussianModel
from nurec.models.post_processing import (
    BilateralGridPerCamera,
    BilateralGridPerFrame,
)
from nurec.utils.batch import DataAndRenderingBatch
from nurec.utils.misc import unpack_optional
from nurec.utils.profiling import ScopedTimer
from nurec.utils.types import GaussiansCompositeReturn


log = logging.getLogger(__name__)

# Node-semantic predicates are (class_ids, select_matches), evaluated in the
# kernel as `selected = (class_matches == select_matches)`. The polarity is
# easy to invert, so never write the tuples inline:
#   empty IDs + True  -> matches nothing: no point is penalized
#   empty IDs + False -> "use nothing":   EVERY point is penalized
#     (the meaning of an explicitly empty layer_labels_to_use list).
_NODE_SEMANTIC_SELECT_NONE: tuple[tuple[int, ...], bool] = ((), True)


class ModuleLosses(torch.nn.Module):
    """
    Stateful class for all fused CUDA losses model layer

    Wraps fused losses for NRE training; see nurec/libs/losses/README.md.
    """

    MODULE_SIDE_LOSSES: ClassVar[set[str]] = {"ssim_ssim_mean"}
    # Class attribute so CPU-only tests can read the CUDA-eligible loss names
    # without instantiating (the constructor allocates CUDA dummy tensors).
    # Mean reduction is converted to pre-multiplication and sum reduction in
    # the CUDA kernels.
    available: ClassVar[list[str]] = [
        "rgb_l1_mean",
        "ssim_ssim_mean",
        "lidar_l1_mean",
        "background_mse_mean",
        "background_lidar_mse_mean",
        "bilateral_grid_drift_identity_distance_mean",
        "bilateral_grid_per_camera_tv_total_variation_spatial_mean",
        "bilateral_grid_per_frame_spatial_tv_total_variation_spatial_mean",
        "gaussian_scale_abs_mean",
        "sky_env_map_background_total_variation_spatial_mean",
        "gaussian_density_abs_mean",
        "out_of_bound_l1_mean",
        "gaussian_z_scale_abs_mean",
        "intensity_mse_mean",
        "raydrop_mse_mean",
        "background_in_track_gaussian_bce_with_logits_mean",
        "node_semantic_gaussians_bce_with_logits_mean",
        "semantic_cross_entropy_mean",
        "deform_smoothness_abs_mean",
    ]
    cuda_loss_names: list[str]
    losses: list[Any]
    sum_reduce_fn: SumReduceFn
    dummy_flags: torch.Tensor
    dummy_rgb: torch.Tensor
    dummy_lidar: torch.Tensor
    dummy_out: torch.Tensor
    dummy_bg: torch.Tensor
    dummy_gc: torch.Tensor
    dummy_gf: torch.Tensor
    dummy_tex: torch.Tensor
    fused_ssim_losses: torch.nn.ModuleDict

    def __init__(self):
        super().__init__()
        # CudaLossesFunction's per-element outputs follow this order; grouped scalar
        # losses are returned separately and handled through SCALAR_LOSS_OUTPUT_INDEX.
        self.cuda_loss_names = [name for name in self.available if name not in self.MODULE_SIDE_LOSSES]
        self.losses = []
        self.sum_reduce_fn = SumReduceFn(omegaconf.DictConfig({}))
        self.fused_ssim_losses = torch.nn.ModuleDict()
        dummy_device = torch.device("cuda")
        self.dummy_flags = torch.empty((1, 1, 1, 1), dtype=torch.int32, device=dummy_device)
        self.dummy_rgb = torch.empty((1, 1, 1, 3), dtype=torch.float32, device=dummy_device)
        self.dummy_lidar = torch.empty((1, 1, 1, 1), dtype=torch.float32, device=dummy_device)
        self.dummy_out = torch.empty((1, 1, 1), dtype=torch.float32, device=dummy_device)
        self.dummy_bg = torch.empty((1,), dtype=torch.float32, device=dummy_device)
        self.dummy_gc = torch.empty((1 * GRID_NUM_CHANNELS, 1, 1, 1), dtype=torch.float32, device=dummy_device)
        self.dummy_gf = torch.empty((1 * GRID_NUM_CHANNELS, 1, 1, 1), dtype=torch.float32, device=dummy_device)
        self.dummy_scales = torch.empty((1, 3), dtype=torch.float32, device=dummy_device)
        self.dummy_tex = torch.empty((1, 1, 1, 1), dtype=torch.float32, device=dummy_device)
        self.dummy_densities = torch.empty((1,), dtype=torch.float32, device=dummy_device)
        self.dummy_gaussian_positions = torch.empty((1, 3), dtype=torch.float32, device=torch.device("cuda"))
        self.dummy_gaussian_cuboid_dims = torch.empty((1, 3), dtype=torch.float32, device=torch.device("cuda"))
        self.dummy_z_scales = torch.empty((1, 3), dtype=torch.float32, device=dummy_device)
        self.dummy_intensity = torch.empty((1, 1, 1, 1), dtype=torch.float32, device=dummy_device)
        self.dummy_raydrop = torch.empty((1, 1, 1, 1), dtype=torch.float32, device=dummy_device)
        self.dummy_visibility = torch.ones((1,), dtype=torch.float32, device=dummy_device)
        self.dummy_bg_track_semantic_logits = torch.empty((0, 1), dtype=torch.float32, device=dummy_device)
        self.dummy_bg_track_timestamps_startend_us = torch.zeros((1, 2), dtype=torch.int64, device=dummy_device)
        self.dummy_bg_track_tracks_packinfo = torch.empty((0, 2), dtype=torch.int32, device=dummy_device)
        self.dummy_bg_track_tracks_poses = torch.empty((0, 7), dtype=torch.float32, device=dummy_device)
        self.dummy_bg_track_tracks_timestamps_us = torch.empty((0,), dtype=torch.int64, device=dummy_device)
        self.dummy_bg_track_cuboids_dims = torch.empty((0, 3), dtype=torch.float32, device=dummy_device)
        self.dummy_semantic_logits = torch.empty((0, 1), dtype=torch.float32, device=dummy_device)
        self.dummy_semantic_targets = torch.empty((0,), dtype=torch.uint8, device=dummy_device)
        self.dummy_deformation = torch.empty((0, 3), dtype=torch.float32, device=dummy_device)

    def _get_fused_ssim_loss(
        self,
        window_size: int,
        mask_mode: str,
        constant_mask_value: float,
    ) -> torch.nn.Module:
        key = f"{window_size}:{mask_mode}:{constant_mask_value!r}".replace(".", "_")
        if key not in self.fused_ssim_losses:
            from gsplat.losses import FusedSSIMLosses

            self.fused_ssim_losses[key] = FusedSSIMLosses(
                window_size=window_size,
                mask_mode=mask_mode,
                constant_mask_value=constant_mask_value,
            )
        return self.fused_ssim_losses[key]

    def is_enabled(
        self,
        loss_name: str,
        model: BaseModel,
        camera_grids: list[BilateralGridPerCamera],
        frame_grids: list[BilateralGridPerFrame],
        target: DataAndRenderingBatch,
        out_of_bound_gaussian_nodes: list[RigidGaussianModel],
    ) -> bool:
        if loss_name == "rgb_l1_mean":
            if target.data.camera is not None:
                assert target.data.camera.labels.flags is not None and target.data.camera.labels.rgb is not None, (
                    "Slang losses requires camera labels flags and rgb!"
                )
                return True
        elif loss_name == "ssim_ssim_mean":
            if target.data.camera is not None:
                assert target.data.camera.labels.flags is not None and target.data.camera.labels.rgb is not None, (
                    "SSIM loss requires camera labels flags and rgb."
                )
                return True
        elif loss_name == "lidar_l1_mean":
            if target.data.lidar is not None:
                assert target.data.lidar.labels.flags is not None and target.data.lidar.labels.distance is not None, (
                    "Slang losses requires lidar labels flags and distance!"
                )
                return True
        elif loss_name == "background_mse_mean":
            if target.data.camera is not None:
                assert target.data.camera.labels.flags is not None, "Slang losses requires camera labels flags!"
                return True
        elif loss_name == "background_lidar_mse_mean":
            if target.data.lidar is not None:
                assert target.data.lidar.labels.flags is not None, "Slang losses requires lidar labels flags!"
                return True
        elif loss_name == "bilateral_grid_drift_identity_distance_mean":
            if len(camera_grids) > 0 or len(frame_grids) > 0:
                return True
        elif loss_name == "bilateral_grid_per_camera_tv_total_variation_spatial_mean":
            if len(camera_grids) > 0:
                return True
        elif loss_name == "bilateral_grid_per_frame_spatial_tv_total_variation_spatial_mean":
            if len(frame_grids) > 0:
                return True
        elif loss_name == "gaussian_scale_abs_mean":
            return True
        elif loss_name == "sky_env_map_background_total_variation_spatial_mean":
            # The sky model may be disabled via a component-only overlay
            # (a SkipBackground stand-in replaces it) — there are no
            # env-map textures to regularize then: no-op this loss.
            return isinstance(getattr(model, "background", None), SkyEnvMapBackground)
        elif loss_name == "gaussian_density_abs_mean":
            return True
        elif loss_name == "out_of_bound_l1_mean":
            assert isinstance(model, GaussiansComposite), (
                "Slang losses requires GaussiansComposite model for out_of_bound!"
            )

            return len(out_of_bound_gaussian_nodes) > 0
        elif loss_name == "gaussian_z_scale_abs_mean":
            return True
        elif loss_name == "intensity_mse_mean":
            if target.data.lidar is not None:
                return True
        elif loss_name == "raydrop_mse_mean":
            if target.data.lidar is not None:
                return True
        elif loss_name == "background_in_track_gaussian_bce_with_logits_mean":
            assert isinstance(model, GaussiansComposite), (
                "background_in_track_gaussian loss requires model to be GaussiansComposite"
            )
            if target.rendering is None:
                return False
            if target.rendering.camera is None:
                return False
            loss = next((item for item in self.losses if item.name == loss_name), None)
            if loss is None:
                return False
            has_tracks = any(
                isinstance(model.gaussians_nodes[node_id], RigidGaussianModel)
                for node_id in model.get_gaussians_node_ids()
            )
            has_background_points = any(
                node_id in loss.layer_names and not isinstance(model.gaussians_nodes[node_id], RigidGaussianModel)
                for node_id in model.get_gaussians_node_ids()
            )
            return has_tracks and has_background_points
        elif loss_name == "node_semantic_gaussians_bce_with_logits_mean":
            assert isinstance(model, GaussiansComposite), (
                "node_semantic_gaussians loss requires model to be GaussiansComposite"
            )
            return any(item.name == loss_name for item in self.losses)
        elif loss_name == "semantic_cross_entropy_mean":
            if target.data.camera is None:
                return False
            labels = target.data.camera.labels
            if labels.flags is None or labels.semantic is None:
                raise ValueError("semantic loss requires camera label flags and semantic labels")
            return True
        elif loss_name == "deform_smoothness_abs_mean":
            return True
        return False

    def forward(
        self,
        step: int,
        model: BaseModel,  # noqa: ARG002, pylint: disable=unused-argument
        results: GaussiansCompositeReturn,
        target: DataAndRenderingBatch,
    ) -> dict[str, LossReturn]:
        assert results is not None and target is not None, "Slang losses requires results and target!"

        camera_grids = _get_bilateral_grids(model, BilateralGridPerCamera)
        frame_grids = _get_bilateral_grids(model, BilateralGridPerFrame)
        out_of_bound_gaussian_nodes = _get_out_of_bound_gaussian_nodes(model)

        run_losses = []
        for loss in self.losses:
            if loss.should_run_fn(step) and self.is_enabled(
                loss.name, model, camera_grids, frame_grids, target, out_of_bound_gaussian_nodes
            ):
                run_losses.append(loss)

        # Check which loss types are being run
        run_rgb = any("rgb" == loss.name.split("_")[0] for loss in run_losses)
        run_ssim = any(loss.name == "ssim_ssim_mean" for loss in run_losses)
        run_lidar = any("lidar" == loss.name.split("_")[0] for loss in run_losses)
        run_bg = any(loss.name == "background_mse_mean" for loss in run_losses)
        run_bg_lidar = any(loss.name == "background_lidar_mse_mean" for loss in run_losses)
        run_bilateral_grid_drift = any("bilateral_grid_drift" in loss.name for loss in run_losses)
        run_per_camera_spatial_tv = any("per_camera_tv" in loss.name for loss in run_losses)
        run_per_frame_spatial_tv = any("per_frame_spatial_tv" in loss.name for loss in run_losses)
        run_gaussian_scale = any("gaussian_scale" in loss.name for loss in run_losses)
        run_sky_env_map = any(loss.name == "sky_env_map_background_total_variation_spatial_mean" for loss in run_losses)
        run_gaussian_density = any("gaussian_density" in loss.name for loss in run_losses)
        run_out_of_bound = any(loss.name == "out_of_bound_l1_mean" for loss in run_losses)
        run_gaussian_z_scale = next((loss for loss in run_losses if "gaussian_z_scale" in loss.name), None)
        run_intensity = any("intensity" == loss.name.split("_")[0] for loss in run_losses)
        run_raydrop = any("raydrop" == loss.name.split("_")[0] for loss in run_losses)
        run_background_in_track = next(
            (loss for loss in run_losses if loss.name == "background_in_track_gaussian_bce_with_logits_mean"), None
        )
        run_node_semantic = next(
            (loss for loss in run_losses if loss.name == "node_semantic_gaussians_bce_with_logits_mean"), None
        )
        run_semantic = next((loss for loss in run_losses if loss.name == "semantic_cross_entropy_mean"), None)
        run_deform = next((loss for loss in run_losses if loss.name == "deform_smoothness_abs_mean"), None)

        if (run_rgb or run_ssim or run_bg or run_semantic is not None) and results.rendered_cam is None:
            raise ValueError("Fused camera losses require results.rendered_cam")

        # Load RGB data or use dummy tensors
        rgb_flags: torch.Tensor = self.dummy_flags
        rgb_gt: torch.Tensor = self.dummy_rgb
        rgb_pred: torch.Tensor = self.dummy_rgb
        rgb_factor: float = -1.0  # use -1 to indicate dummy tensors (like null tensors)
        ssim_factor: float = -1.0

        if (run_rgb or run_ssim) and results.rendered_cam is not None:
            assert target.data.camera is not None, "Slang losses requires camera data for RGB/SSIM loss!"
            rgb_flags = unpack_optional(target.data.camera.labels.flags)  # [B_rgb,H_rgb,W_rgb,1]
            rgb_gt = unpack_optional(target.data.camera.labels.rgb)  # [B_rgb,H_rgb,W_rgb,3]
            rgb_pred = unpack_optional(results.rendered_cam.rgb)  # [n_rays_rgb, 3]
            rgb_pred = rgb_pred.reshape_as(rgb_gt)  # [B_rgb,H_rgb,W_rgb,3]
            if run_rgb:
                n_valid_rgb = target.data.camera.labels.n_valid_rgb
                rgb_factor = 1.0 / n_valid_rgb if n_valid_rgb > 0 else 0.0
            if run_ssim:
                # SSIM reads validity straight from `rgb_flags` (RayFlags bitmask) inside the CUDA
                # kernel — no intermediary bool mask tensor is materialized here. SSIM averages
                # channels before masking, so the denominator is valid pixels (`n_valid`), not
                # RGB scalar elements (`n_valid_rgb`).
                n_valid = target.data.camera.labels.n_valid
                ssim_factor = 1.0 / n_valid if n_valid > 0 else 0.0

        # Load Lidar data or use dummy tensors
        lidar_flags: torch.Tensor = self.dummy_flags
        lidar_gt: torch.Tensor = self.dummy_lidar
        lidar_pred: torch.Tensor = self.dummy_lidar
        lidar_factor: float = -1.0  # use -1 to indicate dummy tensors (like null tensors)

        if run_lidar:
            assert target.data.lidar is not None, "Slang losses requires lidar data for Lidar loss!"
            lidar_flags = unpack_optional(target.data.lidar.labels.flags)  # [B_lidar,H_lidar,W_lidar,1]
            lidar_gt = unpack_optional(target.data.lidar.labels.distance)  # [B_lidar,H_lidar,W_lidar,1]
            lidar_pred = unpack_optional(unpack_optional(results.rendered_lidar).distance)  # [n_rays_lidar, 1]
            assert lidar_gt is not None, "lidar_gt should not be None"
            lidar_pred = lidar_pred.reshape_as(lidar_gt)  # [B_lidar,H_lidar,W_lidar,1]
            n_valid_lidar = target.data.lidar.labels.n_valid_lidar
            lidar_factor = 1.0 / n_valid_lidar if n_valid_lidar > 0 else 0.0

        # Load Intensity data or use dummy tensors
        intensity_pred: torch.Tensor = self.dummy_intensity
        intensity_gt: torch.Tensor = self.dummy_intensity
        intensity_factor: float = -1.0

        if run_intensity:
            assert target.data.lidar is not None, "Slang losses requires lidar data for Intensity loss!"
            if not run_lidar:
                lidar_flags = unpack_optional(target.data.lidar.labels.flags)
            intensity_gt = unpack_optional(target.data.lidar.labels.intensity)  # [B,H,W,1]
            intensity_pred = unpack_optional(
                unpack_optional(unpack_optional(results.rendered_lidar).extra_ray_signals).intensity
            )
            intensity_pred = intensity_pred.reshape_as(intensity_gt)  # [B,H,W,1]
            n_valid = target.data.lidar.labels.n_valid_lidar
            intensity_factor = 1.0 / n_valid if n_valid > 0 else 0.0

        # Load Raydrop data or use dummy tensors
        raydrop_pred: torch.Tensor = self.dummy_raydrop
        raydrop_gt: torch.Tensor = self.dummy_raydrop
        raydrop_factor: float = -1.0

        if run_raydrop:
            assert target.data.lidar is not None, "Slang losses requires lidar data for Raydrop loss!"
            if not run_lidar and not run_intensity:
                lidar_flags = unpack_optional(target.data.lidar.labels.flags)
            raydrop_gt = unpack_optional(target.data.lidar.labels.raydrop)  # [B,H,W,1]
            raydrop_pred = unpack_optional(
                unpack_optional(unpack_optional(results.rendered_lidar).extra_ray_signals).raydrop
            )
            raydrop_pred = raydrop_pred.reshape_as(raydrop_gt)  # [B,H,W,1]
            # Raydrop kernel checks INVALID only (no DROPPED), so use INVALID-only count for normalization.
            # n_valid_raydrop is cached on the labels at collate time so no CUDA sync happens here.
            n_valid_raydrop = target.data.lidar.labels.n_valid_raydrop
            raydrop_factor = 1.0 / n_valid_raydrop if n_valid_raydrop > 0 else 0.0

        # Load Background data or use dummy tensors
        bg_pred: torch.Tensor = self.dummy_bg
        bg_factor: float = -1.0  # use -1 to indicate dummy tensors (like null tensors)

        if run_bg and results.rendered_cam is not None:
            assert target.data.camera is not None, "Slang losses requires camera data for Background loss!"
            if not run_rgb:  # rgb_flags was not loaded and it is needed for background
                rgb_flags = unpack_optional(target.data.camera.labels.flags)  # [B_rgb,H_rgb,W_rgb,1]
            bg_pred = unpack_optional(results.rendered_cam.opacity)  # [n_rays]
            n_valid_bg = target.data.camera.labels.n_valid_bg
            bg_factor = 1.0 / n_valid_bg if n_valid_bg > 0 else 0.0

        # Camera semantic cross-entropy shares Dispatch 1's flags but keeps its
        # packed, row-strided logits view so the renderer output is not copied.
        semantic_logits = self.dummy_semantic_logits
        semantic_targets = self.dummy_semantic_targets
        semantic_lambda = -1.0
        if run_semantic is not None:
            camera = unpack_optional(target.data.camera)
            camera_flags = unpack_optional(camera.labels.flags)
            camera_semantic_targets = unpack_optional(camera.labels.semantic).flatten()
            camera_semantic_logits = unpack_optional(
                unpack_optional(unpack_optional(results.rendered_cam).extra_ray_signals).semantic_logits
            )
            semantic_lambda = run_semantic.lambda_
            rgb_flags = camera_flags.to(torch.int32)
            semantic_logits = camera_semantic_logits.float()
            semantic_targets = camera_semantic_targets

        # Load Background Lidar data or use dummy tensors
        bg_lidar_pred: torch.Tensor = self.dummy_bg
        bg_lidar_factor: float = -1.0  # use -1 to indicate dummy tensors (like null tensors)

        if run_bg_lidar:
            assert target.data.lidar is not None, "Slang losses requires lidar data for Background Lidar loss!"
            if not run_lidar:  # lidar_flags was not loaded and it is needed for background_lidar
                lidar_flags = unpack_optional(target.data.lidar.labels.flags)  # [B_lidar,H_lidar,W_lidar,1]
            bg_lidar_pred = unpack_optional(unpack_optional(results.rendered_lidar).opacity)  # [n_rays_lidar]
            n_valid_lidar = target.data.lidar.labels.n_valid_lidar
            bg_lidar_factor = 1.0 / n_valid_lidar if n_valid_lidar > 0 else 0.0

        # Load Bilateral Grid data or use dummy tensors
        grids_per_camera: torch.Tensor = self.dummy_gc
        grids_per_frame: torch.Tensor = self.dummy_gf

        numel_grids_per_camera = 0
        numel_grids_per_frame = 0
        if len(camera_grids) > 0:
            assert len(camera_grids) == 1, "Slang losses requires exactly one BilateralGridPerCamera"
            grids_per_camera = camera_grids[0].bilateral_grid.grid
            B_gc, C_gc, D_gc, H_gc, W_gc = grids_per_camera.shape
            assert C_gc == GRID_NUM_CHANNELS, f"BilateralGridPerCamera must have {GRID_NUM_CHANNELS} channels"
            grids_per_camera = grids_per_camera.view(B_gc * C_gc, D_gc, H_gc, W_gc)
            numel_grids_per_camera = B_gc * D_gc * H_gc * W_gc
        if len(frame_grids) > 0:
            assert len(frame_grids) == 1, "Slang losses requires exactly one BilateralGridPerFrame"
            grids_per_frame = frame_grids[0].bilateral_grid.grid
            B_gf, C_gf, D_gf, H_gf, W_gf = grids_per_frame.shape
            assert C_gf == GRID_NUM_CHANNELS, f"BilateralGridPerFrame must have {GRID_NUM_CHANNELS} channels"
            grids_per_frame = grids_per_frame.view(B_gf * C_gf, D_gf, H_gf, W_gf)
            numel_grids_per_frame = B_gf * D_gf * H_gf * W_gf

        grid_drift_per_camera_factor: float = -1.0  # use -1 to indicate dummy tensors (like null tensors)
        grid_drift_per_frame_factor: float = -1.0  # use -1 to indicate dummy tensors (like null tensors)
        if run_bilateral_grid_drift:
            if numel_grids_per_camera > 0 or numel_grids_per_frame > 0:
                grid_drift_per_camera_factor = (
                    1.0 / (numel_grids_per_camera + numel_grids_per_frame)
                    if numel_grids_per_camera + numel_grids_per_frame > 0
                    else 0.0
                )
                grid_drift_per_frame_factor = (
                    1.0 / (numel_grids_per_camera + numel_grids_per_frame)
                    if numel_grids_per_camera + numel_grids_per_frame > 0
                    else 0.0
                )

        per_camera_spatial_tv_factor: float = -1.0  # use -1 to indicate dummy tensors (like null tensors)
        per_frame_spatial_tv_factor: float = -1.0  # use -1 to indicate dummy tensors (like null tensors)
        if run_per_camera_spatial_tv:
            if numel_grids_per_camera > 0:
                per_camera_spatial_tv_factor = 1.0 / numel_grids_per_camera if numel_grids_per_camera > 0 else 0.0
        if run_per_frame_spatial_tv:
            if numel_grids_per_frame > 0:
                per_frame_spatial_tv_factor = 1.0 / numel_grids_per_frame if numel_grids_per_frame > 0 else 0.0

        # Load Gaussian scale data or use dummy tensors
        gaussian_scales: torch.Tensor = self.dummy_scales
        scale_factor: float = -1.0  # use -1 to indicate dummy tensors (like null tensors)

        if run_gaussian_scale:
            # Type narrow: gaussian_scale loss requires GaussiansComposite
            assert isinstance(model, GaussiansComposite), "gaussian_scale loss requires model to be GaussiansComposite"
            # The loss is active even when there are currently no eligible
            # Gaussians. A zero factor makes the fused kernel write a
            # deterministic zero instead of returning an unwritten dummy
            # output.
            scale_factor = 0.0

            # Find the gaussian_scale loss config to get layer_lambdas
            gaussian_scale_loss = next((loss for loss in self.losses if "gaussian_scale" in loss.name), None)

            if gaussian_scale_loss is not None:
                # Get layer_lambdas from config (might not exist)
                layer_lambdas_dict = getattr(gaussian_scale_loss, "layer_lambdas", {})

                # Collect pre-activation scales with layer-specific weights applied in log-space
                # This allows exp() to be fused in the Slang kernel
                scales_list = []
                for node_id in model.get_gaussians_node_ids():
                    layer_lambda = layer_lambdas_dict.get(node_id, 1.0)

                    # Get pre-activation (log-space) scales to move exp() to Slang kernel
                    node_scales_preact = model.gaussians_nodes[node_id].get_scales(preactivation=True)

                    # Handle λ <= 0 specially to avoid math.log(0) ValueError
                    # When λ=0, the Python impl produces zeros: get_scales() * 0 = 0
                    # We match this by passing -inf in log-space: exp(-inf) = 0
                    if layer_lambda <= 0.0:
                        weighted_scales_preact = torch.full_like(node_scales_preact, float("-inf"))
                    else:
                        # Apply layer weighting in log-space: exp(x + log(λ)) = exp(x) * λ
                        # When λ=1.0, log(1.0)=0.0 and adding 0.0 is a no-op
                        weighted_scales_preact = node_scales_preact + math.log(layer_lambda)

                    scales_list.append(weighted_scales_preact)

                if scales_list:  # Only concatenate if we have scales
                    gaussian_scales = torch.cat(scales_list, dim=0)
                    if gaussian_scales.numel() > 0:
                        scale_factor = 1.0 / gaussian_scales.numel()

        # Load Sky-Env-Map textures or use dummy tensors
        bg_tex: torch.Tensor = self.dummy_tex
        bg_tex_factor: float = -1.0  # use -1 to indicate dummy tensors (like null tensors)

        if run_sky_env_map:
            assert isinstance(model, GaussiansComposite), "SkyEnvMap loss requires GaussiansComposite model"
            assert isinstance(model.background, SkyEnvMapBackground), "SkyEnvMap loss requires SkyEnvMapBackground"
            bg_tex = model.background.textures
            assert bg_tex is not None, "SkyEnvMap loss requires background textures"
            # SkyEnvMap factor is -1 (deactivated), 0 (active but invalid texture) or 1 (active and valid texture)
            # It has 3 pre-multiplication factors (for D, H, W) that will be computed in the Slang kernel
            bg_tex_factor = 1.0 if bg_tex.numel() > 0 else 0.0

        # Load Gaussian density data or use dummy tensors
        gaussian_densities: torch.Tensor = self.dummy_densities
        density_factor: float = -1.0  # use -1 to indicate dummy tensors (like null tensors)

        if run_gaussian_density:
            # Type narrow: gaussian_density loss requires GaussiansComposite
            assert isinstance(model, GaussiansComposite), (
                "gaussian_density loss requires model to be GaussiansComposite"
            )
            # Keep an active-but-empty loss distinct from a disabled loss.
            # The zero factor makes the kernel initialize the returned slot.
            density_factor = 0.0

            # Find the gaussian_density loss config to get layer_lambdas
            gaussian_density_loss = next((loss for loss in self.losses if "gaussian_density" in loss.name), None)

            if gaussian_density_loss is not None:
                # Get layer_lambdas from config (might not exist)
                layer_lambdas_dict = getattr(gaussian_density_loss, "layer_lambdas", {})

                # Collect densities with layer-specific weights
                densities_list = []
                for node_id in model.get_gaussians_node_ids():
                    node_densities = model.gaussians_nodes[node_id].get_densities()
                    layer_lambda = layer_lambdas_dict.get(node_id, 1.0)
                    weighted_densities = node_densities * layer_lambda
                    # Flatten to 1D tensor (densities might be [N, 1] but we need [N])
                    densities_list.append(weighted_densities.flatten())

                if densities_list:  # Only concatenate if we have densities
                    gaussian_densities = torch.cat(densities_list, dim=0)
                    if gaussian_densities.numel() > 0:
                        density_factor = 1.0 / gaussian_densities.numel()

        # Visibility mask for gaussian_scale / gaussian_density (same logic as PyTorch path)
        visibility_gaussian: torch.Tensor | None = None
        if run_gaussian_scale or run_gaussian_density:
            scale_loss_cfg = next((loss for loss in self.losses if "gaussian_scale" in loss.name), None)
            density_loss_cfg = next((loss for loss in self.losses if "gaussian_density" in loss.name), None)
            use_visibility_scale = run_gaussian_scale and getattr(scale_loss_cfg, "visibility_filter", False)
            use_visibility_density = run_gaussian_density and getattr(density_loss_cfg, "visibility_filter", False)
            active_visibility_scale = use_visibility_scale and scale_factor > 0
            active_visibility_density = use_visibility_density and density_factor > 0

            if use_visibility_scale != use_visibility_density and run_gaussian_scale and run_gaussian_density:
                raise ValueError(
                    "gaussian_scale and gaussian_density must have matching visibility_filter when both are active. "
                    f"Got scale={use_visibility_scale}, density={use_visibility_density}. "
                    "The Slang kernel applies a single shared visibility mask to both."
                )

            if active_visibility_scale or active_visibility_density:
                element_count = max(
                    gaussian_scales.shape[0] if active_visibility_scale else 0,
                    gaussian_densities.shape[0] if active_visibility_density else 0,
                )
                ref_tensor = gaussian_scales if active_visibility_scale else gaussian_densities
                loss_cfg = scale_loss_cfg if scale_loss_cfg else density_loss_cfg
                occlusion_aware = bool(getattr(loss_cfg, "occlusion_aware", False))
                visibility_gaussian = get_rendered_visibility_mask(
                    results=results,
                    element_count=element_count,
                    device=ref_tensor.device,
                    dtype=ref_tensor.dtype,
                    visibility_filter=True,
                    occlusion_aware=occlusion_aware,
                )
                _maybe_update_mcmc_visibility_counters(model=model, results=results, visibility=visibility_gaussian)

        n_gaussian = max(gaussian_scales.shape[0], gaussian_densities.shape[0])
        ref_tensor_vis = gaussian_scales if (run_gaussian_scale and scale_factor >= 0) else gaussian_densities
        gaussian_visibility = (
            visibility_gaussian
            if visibility_gaussian is not None
            else torch.ones(n_gaussian, device=ref_tensor_vis.device, dtype=ref_tensor_vis.dtype)
        )

        # Load out_of_bound data or use dummy tensors
        out_of_bound_positions: torch.Tensor = self.dummy_gaussian_positions
        out_of_bound_cuboid_dims: torch.Tensor = self.dummy_gaussian_cuboid_dims
        out_of_bound_factor: float = -1.0  # use -1 to indicate dummy tensors (like null tensors)

        if run_out_of_bound:
            positions_list: list[torch.Tensor] = []
            dims_list: list[torch.Tensor] = []
            for node in out_of_bound_gaussian_nodes:
                positions = node.get_positions()
                cuboid_dims = node.cuboid_tracks.cuboids_dims
                gaussian_cuboid_ids = node.gaussian_cuboid_ids
                dims = cuboid_dims[gaussian_cuboid_ids]
                positions_list.append(positions)
                dims_list.append(dims)
            if positions_list:
                out_of_bound_positions = torch.cat(positions_list, dim=0)
                out_of_bound_cuboid_dims = torch.cat(dims_list, dim=0)
                num_elements = out_of_bound_positions.shape[0] * out_of_bound_positions.shape[1]
                out_of_bound_factor = 1.0 / num_elements if num_elements > 0 else 0.0

        # Load Gaussian Z-scale data or use dummy tensors
        # Note: gaussian_z_scales is loaded separately from gaussian_scales because:
        # 1. gaussian_scales = ALL layers concatenated, with layer_lambda weighting applied
        # 2. gaussian_z_scales = ONE specific layer (e.g. "road"), WITHOUT weighting
        # Even if the same layer appears in both, the data differs due to weighting.
        gaussian_z_scales: torch.Tensor = self.dummy_z_scales
        z_scale_threshold: float = -1.0  # use -1 to indicate dummy/disabled
        z_scale_factor: float = -1.0

        if run_gaussian_z_scale:
            # Type narrow: gaussian_z_scale loss requires GaussiansComposite
            assert isinstance(model, GaussiansComposite), (
                "gaussian_z_scale loss requires model to be GaussiansComposite"
            )

            # Get layer_name and threshold from config
            layer_name = run_gaussian_z_scale.layer_name

            # A component-only overlay may omit the configured layer. Match
            # the non-fused loss path by omitting the loss entry entirely;
            # a negative factor is only safe when its unwritten output is not
            # returned.
            if layer_name not in model.gaussians_nodes:
                run_losses.remove(run_gaussian_z_scale)
            else:
                z_scale_threshold = run_gaussian_z_scale.road_z_scale
                # An existing layer can temporarily contain zero Gaussians.
                # Keep its loss entry, but make the kernel write zero.
                z_scale_factor = 0.0

                # Get pre-activation Z-scales (full [N, 3] tensor, kernel will extract z-component)
                layer_scales = model.gaussians_nodes[layer_name].get_scales(preactivation=True)
                if layer_scales.shape[0] > 0:
                    gaussian_z_scales = layer_scales  # Pass the full [N, 3] tensor; the kernel extracts [:, 2]
                    z_scale_factor = 1.0 / layer_scales.shape[0]  # N elements (not N*3)

        # Deform smoothness consumes the camera/LiDAR-concatenated tensors from
        # the render result. Its point order and length are independent of the
        # Gaussian parameter tensors already carried by Dispatch 3.
        deformation = self.dummy_deformation
        deform_mask: torch.Tensor | None = None
        deform_mask_source: torch.Tensor | None = None
        deform_lambda = -1.0
        if run_deform is not None:
            if results.deform_smoothness is None:
                run_losses.remove(run_deform)
                run_deform = None
            else:
                deform_lambda = run_deform.lambda_
                deformation = results.deform_smoothness.float()
                if results.deform_smoothness_mask is not None:
                    deform_mask_source = results.deform_smoothness_mask.float()
                    deformation, deform_mask = torch.broadcast_tensors(deformation, deform_mask_source)

        # Load background-in-track data or use shape-valid dummy tensors. Semantic-filtered
        # layers are packed first. Their host-known shapes and configured labels become
        # immutable by-value launch metadata; unfiltered and missing-semantic layers form
        # an include-all suffix.
        bg_track_positions: torch.Tensor = self.dummy_gaussian_positions
        bg_track_density_logits: torch.Tensor = self.dummy_densities
        bg_track_semantic_logits: torch.Tensor = self.dummy_bg_track_semantic_logits
        bg_track_n_semantic_points: int | None = None
        bg_track_semantic_segments: tuple[tuple[int, tuple[int, ...], torch.Tensor], ...] = ()
        bg_track_camera_timestamps_startend_us: torch.Tensor = self.dummy_bg_track_timestamps_startend_us
        bg_track_tracks_packinfo: torch.Tensor = self.dummy_bg_track_tracks_packinfo
        bg_track_tracks_poses: torch.Tensor = self.dummy_bg_track_tracks_poses
        bg_track_tracks_timestamps_us: torch.Tensor = self.dummy_bg_track_tracks_timestamps_us
        bg_track_cuboids_dims: torch.Tensor = self.dummy_bg_track_cuboids_dims
        bg_track_density_logits_min: float = -1.0
        bg_track_lambda: float = -1.0
        filtered_entries: list[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        unfiltered_entries: list[tuple[torch.Tensor, torch.Tensor]] = []
        tracks_list: list[CuboidTracks] = []

        if run_background_in_track is not None:
            assert isinstance(model, GaussiansComposite), (
                "background_in_track_gaussian loss requires model to be GaussiansComposite"
            )

            layer_labels_to_use = run_background_in_track.layer_labels_to_use

            for node_id in model.get_gaussians_node_ids():
                bg_node = model.gaussians_nodes[node_id]
                if isinstance(bg_node, RigidGaussianModel):
                    if bg_node.owns_tracks:
                        tracks_list.append(bg_node.cuboid_tracks)
                    continue
                if node_id not in run_background_in_track.layer_names:
                    continue

                bg_positions = bg_node.get_positions().float()
                density_logits = bg_node.get_densities(preactivation=True).flatten().float()
                node_semantic_logits = None
                if node_id in layer_labels_to_use:
                    try:
                        node_semantic_logits = bg_node.get_extra_signal_by_key("semantic_logits")
                    except ValueError:
                        node_semantic_logits = None
                    if node_semantic_logits is None:
                        log.warning(
                            "Node '%s' is in layer_labels_to_use but missing 'semantic_logits', "
                            "collecting all points without semantic filtering",
                            node_id,
                        )

                if node_semantic_logits is not None:
                    assert node_semantic_logits.ndim == 2 and node_semantic_logits.shape[0] == bg_positions.shape[0], (
                        f"semantic_logits for node '{node_id}' must have shape [N,C] matching positions"
                    )
                    filtered_entries.append((node_id, bg_positions, density_logits, node_semantic_logits))
                else:
                    unfiltered_entries.append((bg_positions, density_logits))

            if tracks_list and (filtered_entries or unfiltered_entries):
                bg_track_n_semantic_points = 0
                ordered_positions = [entry[1] for entry in filtered_entries] + [
                    entry[0] for entry in unfiltered_entries
                ]
                ordered_density_logits = [entry[2] for entry in filtered_entries] + [
                    entry[1] for entry in unfiltered_entries
                ]
                bg_track_positions = (
                    ordered_positions[0] if len(ordered_positions) == 1 else torch.cat(ordered_positions, dim=0)
                )
                bg_track_density_logits = (
                    ordered_density_logits[0]
                    if len(ordered_density_logits) == 1
                    else torch.cat(ordered_density_logits, dim=0)
                )

                if filtered_entries:
                    bg_track_semantic_logits = filtered_entries[0][3]
                    segment_end = 0
                    semantic_segments: list[tuple[int, tuple[int, ...], torch.Tensor]] = []
                    semantic_classes_map = run_background_in_track.semantic_classes_map
                    for node_id, _, _, node_semantic_logits in filtered_entries:
                        node_point_count = int(node_semantic_logits.shape[0])
                        segment_end += node_point_count
                        # Empty layers have no points to classify, so omitting their
                        # zero-width segment preserves behavior and keeps endpoints ordered.
                        if node_point_count == 0:
                            continue
                        allowed_ids = tuple(
                            sorted(
                                {
                                    semantic_classes_map[label]
                                    for label in layer_labels_to_use[node_id]
                                    if semantic_classes_map[label] >= 0
                                }
                            )
                        )
                        semantic_segments.append((segment_end, allowed_ids, node_semantic_logits))
                    bg_track_n_semantic_points = segment_end
                    bg_track_semantic_segments = tuple(semantic_segments)
                else:
                    # This shape-valid empty view carries no metadata allocation.
                    bg_track_semantic_logits = bg_track_positions[:0, :1]

                tracks = tracks_list[0] if len(tracks_list) == 1 else CuboidTracks.Ops.concatenate(tracks_list)
                bg_track_tracks_packinfo = tracks.tracks_packinfo
                bg_track_tracks_poses = tracks.tracks_poses.data
                bg_track_tracks_timestamps_us = tracks.tracks_timestamps_us
                bg_track_cuboids_dims = tracks.cuboids_dims
                bg_track_camera_timestamps_startend_us = unpack_optional(
                    unpack_optional(target.rendering).camera
                ).timestamps_startend_us
                bg_track_density_logits_min = run_background_in_track.density_logits_min
                bg_track_lambda = run_background_in_track.lambda_

        cuda_scalar_loss_returns: dict[str, LossReturn] = {}
        other_node_density_input = self.dummy_densities
        joint_metadata: (
            tuple[
                tuple[int, ...],
                tuple[tuple[int, ...], bool] | None,
                tuple[tuple[int, tuple[int, ...], bool], ...],
                tuple[torch.Tensor, ...],
                float,
            ]
            | None
        ) = None

        if run_node_semantic is not None:
            assert isinstance(model, GaussiansComposite)
            node_metadata: dict[str, tuple[tuple[int, ...], bool]] = {}
            configured_node_ids = (
                run_node_semantic.layer_labels_to_use.keys() | run_node_semantic.layer_labels_to_exclude.keys()
            )
            for node_id in configured_node_ids:
                if node_id in run_node_semantic.layer_labels_to_use:
                    labels = run_node_semantic.layer_labels_to_use[node_id]
                    select_matches = False
                else:
                    labels = run_node_semantic.layer_labels_to_exclude[node_id]
                    select_matches = True
                class_ids = tuple(
                    sorted(
                        {
                            run_node_semantic.semantic_classes_map[label]
                            for label in labels
                            if run_node_semantic.semantic_classes_map[label] >= 0
                        }
                    )
                )
                node_metadata[node_id] = (class_ids, select_matches)

            node_entries: list[tuple[str, torch.Tensor, torch.Tensor, tuple[int, ...], bool]] = []
            for node_id in model.get_gaussians_node_ids():
                if node_id not in node_metadata:
                    continue
                semantic_node = model.gaussians_nodes[node_id]
                try:
                    semantic_input = semantic_node.get_extra_signal_by_key("semantic_logits")
                except ValueError:
                    semantic_input = None
                if semantic_input is None:
                    log.warning(
                        "Node '%s' is configured for node-semantic but missing 'semantic_logits', skipping it",
                        node_id,
                    )
                    continue
                density_input = semantic_node.get_densities(preactivation=True).flatten().float()
                semantic_input = semantic_input.float()
                class_ids, select_matches = node_metadata[node_id]
                node_entries.append((node_id, density_input, semantic_input, class_ids, select_matches))

            if not node_entries:
                run_losses.remove(run_node_semantic)
                run_node_semantic = None
            else:
                background_allowed_class_ids: tuple[int, ...] = ()
                primary_domain: tuple[str, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
                primary_predicate: tuple[tuple[int, ...], bool] | None = None
                other_entries = node_entries
                if run_background_in_track is not None and bg_track_lambda >= 0:
                    if len(filtered_entries) == 1 and not unfiltered_entries:
                        primary_domain = filtered_entries[0]
                        primary_node_id = primary_domain[0]
                        primary_entry = next(
                            (entry for entry in node_entries if entry[0] == primary_node_id),
                            None,
                        )
                        if primary_entry is not None:
                            primary_predicate = (primary_entry[3], primary_entry[4])
                            other_entries = [entry for entry in node_entries if entry[0] != primary_node_id]
                        else:
                            # bg-track's layer is not node-semantic-configured:
                            # exclude the whole primary domain from the
                            # node-semantic loss, explicitly.
                            primary_predicate = _NODE_SEMANTIC_SELECT_NONE
                        semantic_classes_map = run_background_in_track.semantic_classes_map
                        background_allowed_class_ids = tuple(
                            sorted(
                                {
                                    semantic_classes_map[label]
                                    for label in run_background_in_track.layer_labels_to_use[primary_node_id]
                                    if semantic_classes_map[label] >= 0
                                }
                            )
                        )
                else:
                    non_road_entries = [entry for entry in node_entries if entry[0] != "road"]
                    if len(node_entries) == 2 and len(non_road_entries) == 1:
                        primary_entry = non_road_entries[0]
                    else:
                        primary_entry = node_entries[0]
                    primary_node_id, primary_density, primary_semantic, primary_class_ids, primary_select_matches = (
                        primary_entry
                    )
                    primary_positions = model.gaussians_nodes[primary_node_id].get_positions().float()
                    primary_domain = (primary_node_id, primary_positions, primary_density, primary_semantic)
                    primary_predicate = (primary_class_ids, primary_select_matches)
                    other_entries = [entry for entry in node_entries if entry is not primary_entry]

                other_density_tensors = [entry[1] for entry in other_entries]
                if other_density_tensors:
                    other_node_density_input = (
                        other_density_tensors[0]
                        if len(other_density_tensors) == 1
                        else torch.cat(other_density_tensors)
                    )
                else:
                    other_node_density_input = node_entries[0][1][:0]

                segment_end = 0
                other_segments: list[tuple[int, tuple[int, ...], bool]] = []
                for _, density_input, semantic_input, class_ids, select_matches in other_entries:
                    segment_end += density_input.numel()
                    other_segments.append((segment_end, class_ids, select_matches))

                joint_metadata = (
                    background_allowed_class_ids,
                    primary_predicate,
                    tuple(other_segments),
                    tuple(entry[2] for entry in other_entries),
                    run_node_semantic.lambda_,
                )
                if primary_domain is not None:
                    _, primary_positions, primary_density, primary_semantic = primary_domain
                    bg_track_positions = primary_positions
                    bg_track_density_logits = primary_density
                    bg_track_semantic_logits = primary_semantic
                    bg_track_n_semantic_points = None

        losses = CudaLossesFunction.forward(
            rgb_flags,
            rgb_pred,
            rgb_gt,
            rgb_factor,
            lidar_flags,
            lidar_pred,
            lidar_gt,
            lidar_factor,
            intensity_pred,
            intensity_gt,
            intensity_factor,
            raydrop_pred,
            raydrop_gt,
            raydrop_factor,
            bg_pred,
            bg_factor,
            bg_lidar_pred,
            bg_lidar_factor,
            grids_per_camera,
            grids_per_frame,
            grid_drift_per_camera_factor,
            grid_drift_per_frame_factor,
            per_camera_spatial_tv_factor,
            per_frame_spatial_tv_factor,
            gaussian_scales,
            scale_factor,
            bg_tex,
            bg_tex_factor,
            gaussian_densities,
            density_factor,
            gaussian_visibility,
            out_of_bound_positions,
            out_of_bound_cuboid_dims,
            out_of_bound_factor,
            gaussian_z_scales,
            z_scale_threshold,
            z_scale_factor,
            bg_track_positions,
            bg_track_density_logits,
            bg_track_semantic_logits,
            bg_track_n_semantic_points,
            bg_track_semantic_segments,
            bg_track_camera_timestamps_startend_us,
            bg_track_tracks_packinfo,
            bg_track_tracks_poses,
            bg_track_tracks_timestamps_us,
            bg_track_cuboids_dims,
            bg_track_density_logits_min,
            bg_track_lambda,
            other_node_density_input,
            joint_metadata,
            semantic_logits,
            semantic_targets,
            semantic_lambda,
            deformation,
            deform_mask,
            deform_mask_source,
            deform_lambda,
        )

        if run_background_in_track is not None:
            bg_track_raw, bg_track_weighted = SCALAR_LOSS_OUTPUT_INDEX["bg_track"]
            cuda_scalar_loss_returns[run_background_in_track.name] = LossReturn(
                name=run_background_in_track.name,
                lambda_=run_background_in_track.lambda_,
                value=losses[bg_track_raw],
                reduce_fn=run_background_in_track.reduce_fn,
                _reduced_value=losses[bg_track_weighted],
            )

        if joint_metadata is not None:
            assert run_node_semantic is not None
            node_semantic_raw, node_semantic_weighted = SCALAR_LOSS_OUTPUT_INDEX["node_semantic"]
            cuda_scalar_loss_returns[run_node_semantic.name] = LossReturn(
                name=run_node_semantic.name,
                lambda_=run_node_semantic.lambda_,
                value=losses[node_semantic_raw],
                reduce_fn=run_node_semantic.reduce_fn,
                _reduced_value=losses[node_semantic_weighted],
            )

        if semantic_lambda >= 0:
            assert run_semantic is not None
            semantic_raw, semantic_weighted = SCALAR_LOSS_OUTPUT_INDEX["semantic"]
            cuda_scalar_loss_returns[run_semantic.name] = LossReturn(
                name=run_semantic.name,
                lambda_=run_semantic.lambda_,
                value=losses[semantic_raw],
                reduce_fn=run_semantic.reduce_fn,
                _reduced_value=losses[semantic_weighted],
            )

        if deform_lambda >= 0:
            assert run_deform is not None
            deform_raw, deform_weighted = SCALAR_LOSS_OUTPUT_INDEX["deform"]
            cuda_scalar_loss_returns[run_deform.name] = LossReturn(
                name=run_deform.name,
                lambda_=run_deform.lambda_,
                value=losses[deform_raw],
                reduce_fn=run_deform.reduce_fn,
                _reduced_value=losses[deform_weighted],
            )

        ret: dict[str, LossReturn] = {}
        for loss in run_losses:
            if loss.name == "ssim_ssim_mean":
                window_size = loss.window_size if getattr(loss, "window_size", None) is not None else 11
                channel = loss.channel if getattr(loss, "channel", None) is not None else 3
                mask_cfg = getattr(loss, "mask", None)
                if mask_cfg is not None:
                    mask_mode = mask_cfg.mode if hasattr(mask_cfg, "mode") else mask_cfg["mode"]
                    constant_mask_value = float(mask_cfg.value if hasattr(mask_cfg, "value") else mask_cfg["value"])
                else:
                    mask_mode = "GT"
                    constant_mask_value = 0.0
                fused_ssim_loss = self._get_fused_ssim_loss(
                    window_size,
                    "gt" if mask_mode.lower() != "constant" else "constant",
                    constant_mask_value,
                )
                with ScopedTimer("LossAggregator/ssim_ssim_mean"):
                    value = fused_ssim_loss_map(
                        rgb_pred,
                        rgb_gt,
                        unpack_optional(rgb_flags),
                        ssim_factor,
                        channel=channel,
                        fused_ssim_loss=fused_ssim_loss,
                    )
            elif loss.name in cuda_scalar_loss_returns:
                ret[loss.name] = cuda_scalar_loss_returns[loss.name]
                continue
            else:
                loss_idx = self.cuda_loss_names.index(loss.name)
                value = losses[loss_idx]

            ret[loss.name] = LossReturn(
                name=loss.name,
                lambda_=loss.lambda_,
                value=value,
                reduce_fn=self.sum_reduce_fn,
            )

        return ret
