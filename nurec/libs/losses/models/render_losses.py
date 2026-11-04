# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import logging

from typing import List, Optional, cast

import torch

from gsplat.losses import FusedGroundGaussiansLosses

from nurec.config.trainer import TrainerConfig
from nurec.datasets.base import BaseDataset
from nurec.datasets.ncore import NCOREDataset
from nurec.datasets.tracks import CuboidTracks
from nurec.libs.losses.models.base_losses import BaseLossWithSemanticWeights, BaseRenderLoss
from nurec.libs.losses.models.camera_loss_masks import CameraLossKind, camera_loss_valid_mask
from nurec.libs.losses.models.reduce_functions import MeanReduceFn, SumReduceFn
from nurec.libs.losses.models.registry import register_loss
from nurec.libs.losses.models.utils import (
    _get_bilateral_grids,
    _maybe_update_mcmc_visibility_counters,
    create_window,
    get_rendered_visibility_mask,
)
from nurec.libs.losses.orchestration.config import LossItemConfig, LossReturn
from nurec.models.background import EnvMapType, SkyEnvMapBackground
from nurec.models.base import BaseModel
from nurec.models.gaussians.gaussians_composite import GaussiansComposite
from nurec.models.gaussians.gaussians_model import RigidGaussianModel
from nurec.models.post_processing import (
    BilateralGridPerCamera,
    BilateralGridPerFrame,
    PPISPPostProcessing,
)
from nurec.models.tracks_calib import ElevationAwareUnicycleTracksCalib
from nurec.models.tracks_calib_data import TracksCalibResiduals
from nurec.utils.batch import DataAndRenderingBatch
from nurec.utils.cuda_rng import get_cuda_generator
from nurec.utils.geometry import (
    quat_to_euler,
    quat_to_so3_matrix,
    se3_matrix_inverse,
    so3_matrix_to_quat,
    tquat_to_se3_matrix,
)
from nurec.utils.lpips_network import LPIPSNetwork
from nurec.utils.misc import unpack_optional
from nurec.utils.types import (
    GaussiansCompositeReturn,
    RayFlags,
)


log = logging.getLogger(__name__)


@register_loss("rgb")
class RGBLoss(BaseRenderLoss, BaseLossWithSemanticWeights):
    camera_loss_kind = CameraLossKind.RGB

    def __init__(self, config: LossItemConfig, trainer_config: TrainerConfig, **kwargs) -> None:
        super().__init__("rgb", config, trainer_config, **kwargs)

    def forward(
        self,
        results: GaussiansCompositeReturn,
        target: DataAndRenderingBatch,
        model: BaseModel,
    ) -> Optional[LossReturn]:
        if (camera := target.data.camera) is None:
            return None

        if results.rendered_cam is None:
            return None

        labels = camera.labels  # [batch, height, width, D]
        rgb_loss_mask = camera_loss_valid_mask(self.camera_loss_kind, unpack_optional(labels.flags))
        rgb_loss_mask = rgb_loss_mask.squeeze(-1)  # [batch, height, width]

        assert labels.rgb is not None, "RGB labels are required"
        gt_rgb = labels.rgb  # [batch, height, width, 3]
        pred_rgb = unpack_optional(results.rendered_cam.rgb)  # [n_rays, 3]
        pred_rgb = pred_rgb.reshape_as(gt_rgb)  # [n_rays, 3]

        return self.apply_loss_fn(
            pred_rgb[rgb_loss_mask],
            gt_rgb[rgb_loss_mask],
            frame_labels=labels,
            frame_labels_mask=rgb_loss_mask,
        )


@register_loss("velocity")
class VelocityLoss(BaseRenderLoss, BaseLossWithSemanticWeights):
    camera_loss_kind = CameraLossKind.VELOCITY

    def __init__(self, config: LossItemConfig, trainer_config: TrainerConfig, **kwargs) -> None:
        super().__init__("velocity", config, trainer_config, **kwargs)

    def forward(
        self,
        results: GaussiansCompositeReturn,
        target: DataAndRenderingBatch,
        model: BaseModel,
    ) -> Optional[LossReturn]:
        if (camera := target.data.camera) is None:
            return None

        if (pred_extra_signals := unpack_optional(results.rendered_cam).extra_ray_signals) is None:
            return None

        if (gt_velocity := camera.labels.velocity) is None:  # [batch, height, width, 3]
            return None

        labels = camera.labels  # [batch, height, width, D]
        velocity_loss_mask = camera_loss_valid_mask(self.camera_loss_kind, unpack_optional(labels.flags))
        velocity_loss_mask = velocity_loss_mask.squeeze(-1)  # [batch, height, width]

        pred_velocity = unpack_optional(pred_extra_signals.velocity)  # [n_rays, 3]
        pred_velocity = pred_velocity.reshape_as(gt_velocity)  # [n_rays, 3]

        return self.apply_loss_fn(
            pred_velocity[velocity_loss_mask],
            gt_velocity[velocity_loss_mask],
            frame_labels=labels,
            frame_labels_mask=velocity_loss_mask,
        )


@register_loss("lidar")
class LidarLoss(BaseRenderLoss):
    def __init__(self, config: LossItemConfig, trainer_config: TrainerConfig, **kwargs) -> None:
        super().__init__("lidar", config, trainer_config, **kwargs)

    def forward(
        self,
        results: GaussiansCompositeReturn,
        target: DataAndRenderingBatch,
        model: BaseModel,
    ) -> Optional[LossReturn]:
        if (lidar := target.data.lidar) is None:
            return None

        pred_dis = unpack_optional(unpack_optional(results.rendered_lidar).distance)  # [n_rays, 1]
        # [batch, height, width, 1] -> [n_rays]
        gt_dis = unpack_optional(lidar.labels.distance).reshape(-1)

        valid_lidar_rays_mask = lidar.labels.get_mask_flags_none(RayFlags.INVALID) & lidar.labels.get_mask_flags_none(
            RayFlags.DROPPED
        )  # [batch, height, width, 1]
        valid_lidar_rays_mask = valid_lidar_rays_mask.flatten()  # [n_rays]

        return self.apply_loss_fn(
            pred_dis[valid_lidar_rays_mask],
            gt_dis[valid_lidar_rays_mask],
        )


@register_loss("intensity")
class IntensityLoss(BaseRenderLoss):
    def __init__(self, config: LossItemConfig, trainer_config: TrainerConfig, **kwargs) -> None:
        super().__init__("intensity", config, trainer_config, **kwargs)

    def forward(
        self,
        results: GaussiansCompositeReturn,
        target: DataAndRenderingBatch,
        model: BaseModel,
    ) -> Optional[LossReturn]:
        if (lidar := target.data.lidar) is None:
            return None

        valid_lidar_rays_mask = lidar.labels.get_mask_flags_none(RayFlags.INVALID) & lidar.labels.get_mask_flags_none(
            RayFlags.DROPPED
        )  # [batch, height, width, 1]
        gt_intensity = unpack_optional(lidar.labels.intensity)  # [batch, height, width, 1]

        return self.apply_loss_fn(
            unpack_optional(
                unpack_optional(unpack_optional(results.rendered_lidar).extra_ray_signals).intensity
            ).reshape_as(gt_intensity)[valid_lidar_rays_mask],
            gt_intensity[valid_lidar_rays_mask],
        )


@register_loss("raydrop")
class RaydropLoss(BaseRenderLoss):
    def __init__(self, config: LossItemConfig, trainer_config: TrainerConfig, **kwargs) -> None:
        super().__init__("raydrop", config, trainer_config, **kwargs)

    def forward(
        self,
        results: GaussiansCompositeReturn,
        target: DataAndRenderingBatch,
        model: BaseModel,
    ) -> Optional[LossReturn]:
        if (lidar := target.data.lidar) is None:
            return None

        valid_lidar_rays_mask = lidar.labels.get_mask_flags_none(RayFlags.INVALID)  # [batch, height, width, 1]

        return self.apply_loss_fn(
            unpack_optional(
                unpack_optional(unpack_optional(results.rendered_lidar).extra_ray_signals).raydrop
            ).squeeze()[valid_lidar_rays_mask.flatten()],
            unpack_optional(lidar.labels.raydrop)[valid_lidar_rays_mask],
        )


@register_loss("background")
class BackgroundLoss(BaseRenderLoss):
    camera_loss_kind = CameraLossKind.BACKGROUND

    def __init__(self, config: LossItemConfig, **kwargs) -> None:
        super().__init__("background", config, **kwargs)

        if config.fn in ["bce_clipped", "bce_truncated"]:
            clip_opacity = float(unpack_optional(config.clip_opacity))
            assert 0 <= clip_opacity < 0.5, f"[{self.__class__.__name__}] `clip_opacity` should be in range [0,0.5)]"
            self.loss_fn = functools.partial(self.loss_fn, eps=clip_opacity)  # type: ignore[assignment]

    def forward(
        self,
        results: GaussiansCompositeReturn,
        target: DataAndRenderingBatch,
        model: BaseModel,
    ) -> Optional[LossReturn]:
        if (camera := target.data.camera) is None:
            return None

        if results.rendered_cam is None:
            return None

        background_ray_mask = camera.labels.get_mask_flags_all(RayFlags.SKY_SEMANTIC).reshape(-1)  # [n_rays]

        foreground_mask = torch.ones_like(results.rendered_cam.opacity)  # [n_rays, 1]
        foreground_mask[background_ray_mask] = 0.0

        # Valid mask - also exclude HARMONIZED and SYNTHETIC rays to avoid CUDA sync from n_harmonized check
        valid_mask = camera_loss_valid_mask(self.camera_loss_kind, unpack_optional(camera.labels.flags)).reshape(-1)

        return self.apply_loss_fn(
            torch.clamp(results.rendered_cam.opacity, 0, 1)[valid_mask],
            foreground_mask[valid_mask],
        )


@register_loss("background_lidar")
class BackgroundLidarLoss(BaseRenderLoss):
    def __init__(self, config: LossItemConfig, **kwargs) -> None:
        super().__init__("background_lidar", config, **kwargs)

        if config.fn in ["bce_clipped", "bce_truncated"]:
            clip_opacity = float(unpack_optional(config.clip_opacity))
            assert 0 <= clip_opacity < 0.5, f"[{self.__class__.__name__}] `clip_opacity` should be in range [0,0.5)]"
            self.loss_fn = functools.partial(self.loss_fn, eps=clip_opacity)  # type: ignore[assignment]

    def forward(
        self,
        results: GaussiansCompositeReturn,
        target: DataAndRenderingBatch,
        model: BaseModel,
    ) -> Optional[LossReturn]:
        if (lidar := target.data.lidar) is None:
            return None

        # [batch, height, width, 1] -> [n_rays]
        background_ray_mask = lidar.labels.get_mask_flags_all(RayFlags.SKY_SEMANTIC).reshape(-1)

        foreground_mask = torch.ones_like(unpack_optional(results.rendered_lidar).opacity)  # [n_rays, 1]
        foreground_mask[background_ray_mask] = 0.0

        # Valid mask
        valid_mask = lidar.labels.get_mask_flags_none(RayFlags.INVALID) & lidar.labels.get_mask_flags_none(
            RayFlags.DROPPED
        )  # [batch, height, width, 1]
        valid_mask = valid_mask.flatten()  # [n_rays]

        return self.apply_loss_fn(
            torch.clamp(unpack_optional(results.rendered_lidar).opacity, 0, 1)[valid_mask],
            foreground_mask[valid_mask],
        )


@register_loss("distance")
class DistanceLoss(BaseRenderLoss, BaseLossWithSemanticWeights):
    camera_loss_kind = CameraLossKind.DISTANCE

    def __init__(self, config: LossItemConfig, trainer_config: TrainerConfig, **kwargs) -> None:
        loss_name = "distance"
        super().__init__(loss_name, config, trainer_config, **kwargs)
        self.min_distance = unpack_optional(config.min_distance)
        self.max_distance = unpack_optional(config.max_distance)
        self.normalize_by_opacity: bool = unpack_optional(config.normalize_by_opacity)
        self.allow_missing_supervision: bool = config.allow_missing_supervision or False

    def forward(
        self,
        results: GaussiansCompositeReturn,
        target: DataAndRenderingBatch,
        model: BaseModel,
    ) -> Optional[LossReturn]:
        if (camera := target.data.camera) is None:
            return None

        # [batch, height, width, 1]
        if (target_distance := camera.labels.metric_distance) is None:
            if self.allow_missing_supervision:
                return None
            raise ValueError(
                f"[{self.__class__.__name__}] target labels should contain distance labels, but got {target_distance}."
            )

        if results.rendered_cam is None:
            return None

        # Get prediction:
        pred_distance = unpack_optional(results.rendered_cam.distance).reshape_as(target_distance)
        if self.normalize_by_opacity:
            pred_opacity = results.rendered_cam.opacity.reshape_as(target_distance)
            pred_distance = pred_distance / pred_opacity.clamp(min=1.0e-6)

        # keep only pixels within distance range - also exclude HARMONIZED rays to avoid CUDA sync
        mask_valid = camera_loss_valid_mask(
            self.camera_loss_kind,
            unpack_optional(camera.labels.flags),
            metric_distance=target_distance,
            distance_bounds=(self.min_distance, self.max_distance),
        )  # [batch, height, width, 1]

        # Use reduce_mask approach to avoid boolean indexing which causes CUDA sync via nonzero()
        # Also handles empty mask case properly (returns 0 instead of NaN)
        # Note: Don't pass frame_labels_mask since we compute on full tensors - semantic weights
        # will be applied to full tensor and then masked by reduce_mask
        return self.apply_loss_fn(
            pred_distance,
            target_distance,
            eps=self.min_distance,
            frame_labels=camera.labels,
            reduce_mask=mask_valid.float(),
        )


@register_loss("normal")
class NormalLoss(BaseRenderLoss, BaseLossWithSemanticWeights):
    camera_loss_kind = CameraLossKind.NORMAL

    def __init__(self, config: LossItemConfig, trainer_config: TrainerConfig, **kwargs) -> None:
        super().__init__("normal", config, trainer_config, **kwargs)
        self.allow_missing_supervision: bool = config.allow_missing_supervision or False

    def forward(
        self,
        results: GaussiansCompositeReturn,
        target: DataAndRenderingBatch,
        model: BaseModel,
    ) -> Optional[LossReturn]:
        if (camera := target.data.camera) is None:
            return None

        pred_normal = unpack_optional(unpack_optional(results.rendered_cam).normal)

        if (gt_normal := camera.labels.normals) is None:
            if self.allow_missing_supervision:
                return None
            raise ValueError(
                f"[{self.__class__.__name__}] target labels should contain normal labels, but got {gt_normal}."
            )

        # [batch, height, width, 1] -> [n_rays]
        # Exclude HARMONIZED rays in the mask instead of early-return to avoid CUDA sync from n_harmonized check
        normal_loss_mask = camera_loss_valid_mask(self.camera_loss_kind, unpack_optional(camera.labels.flags)).reshape(
            -1
        )

        # Use reduce_mask approach to avoid boolean indexing which causes CUDA sync via nonzero()
        # Also handles empty mask case properly (returns 0 instead of NaN)
        # Unsqueeze mask to [n_rays, 1] for proper broadcasting with [n_rays, 3] values
        # Note: Don't pass frame_labels_mask since we compute on full tensors
        return self.apply_loss_fn(
            pred_normal.reshape(-1, 3),
            gt_normal.reshape(-1, 3),
            frame_labels=camera.labels,
            reduce_mask=normal_loss_mask.float().unsqueeze(-1),
        )


@register_loss("background_in_track_gaussian")
class BackgroundInTrackGaussianLoss(BaseRenderLoss):
    """
    Used in GaussiansComposite model.
    Suppress the gaussians in the background layer (specified by the layer name) to prevent them from appearing in the cuboid tracks.
    """

    supported_model_types = (GaussiansComposite,)

    def __init__(self, config: LossItemConfig, trainer_config: TrainerConfig, **kwargs) -> None:
        super().__init__("background_in_track_gaussian", config, trainer_config, **kwargs)

        assert config.reduce.name == "mean", "Invalid reduce name for BackgroundInTrackGaussianLoss: expected 'mean'."

        self.layer_names = config.layer_names or []
        assert len(self.layer_names) > 0, "layer_names must be provided"

        self.layer_labels_to_use = config.layer_labels_to_use or {}
        self.density_logits_min = config.density_logits_min

    def initialize(self, train_dataset: BaseDataset) -> None:
        if self.lambda_ == 0 or len(self.layer_labels_to_use) == 0:
            return

        super().initialize(train_dataset)

        assert isinstance(train_dataset, NCOREDataset), "BackgroundInTrackGaussianLoss only supports NCOREDataset"
        self.semantic_classes_map = unpack_optional(
            train_dataset.get_datasource().get_semantic_classes_map(
                camera_semantics=True,
                lidar_semantics=False,
            )
        )
        assert self.semantic_classes_map is not None, (
            f"{self.__class__.__name__} failed to load semantic classes "
            "from NCore dataset (is aux-data enabled / is semantic aux-data available?)"
        )

        # Validate all configured labels exist in semantic_classes_map
        invalid_labels = []
        for node_id, labels in self.layer_labels_to_use.items():
            for label in labels:
                if label not in self.semantic_classes_map:
                    invalid_labels.append(f"{node_id}: {label}")

        if invalid_labels:
            raise ValueError(
                f"{self.__class__.__name__}: The following labels in 'layer_labels_to_use' "
                f"are not found in semantic_classes_map: {invalid_labels}. "
                f"Available labels: {list(self.semantic_classes_map.keys())}"
            )

    def forward(
        self,
        results: GaussiansCompositeReturn,
        target: DataAndRenderingBatch,
        model: BaseModel,
    ) -> Optional[LossReturn]:
        assert isinstance(model, self.supported_model_types), (
            f"{self.__class__.__name__} is only defined for {[cls.__name__ for cls in self.supported_model_types]}"
        )

        tracks_list = []
        background_points_list: list[torch.Tensor] = []
        background_density_logits_list: list[torch.Tensor] = []
        semantic_mask_list: list[torch.Tensor] = []
        for node_id in model.get_gaussians_node_ids():
            node = model.gaussians_nodes[node_id]
            if isinstance(node, RigidGaussianModel):
                if node.owns_tracks:
                    tracks_list.append(node.cuboid_tracks)
            elif node_id in self.layer_names:
                # Determine if semantic filtering should be used
                use_semantic_filter = node_id in self.layer_labels_to_use
                semantic_logits = None
                try:
                    semantic_logits = node.get_extra_signal_by_key("semantic_logits")
                except ValueError:
                    use_semantic_filter = False

                if use_semantic_filter and semantic_logits is not None:
                    semantics = semantic_logits.argmax(dim=1)
                    density_logits = node.get_densities(preactivation=True)  # no activation as input for cross_entropy
                    positions = node.get_positions()
                    mask = torch.zeros_like(density_logits, dtype=torch.bool)
                    for label_to_use in self.layer_labels_to_use[node_id]:
                        mask[semantics == self.semantic_classes_map[label_to_use]] = True
                    mask = mask.squeeze(-1)

                    background_points_list.append(positions)
                    background_density_logits_list.append(density_logits)
                    semantic_mask_list.append(mask)
                else:
                    # Log warning if semantic filtering was expected but unavailable
                    if node_id in self.layer_labels_to_use:
                        log.warning(
                            f"Node '{node_id}' is in layer_labels_to_use but missing 'semantic_logits', "
                            f"falling back to collecting all points without semantic filtering"
                        )
                    # Collect all points without filtering
                    node_positions = node.get_positions()
                    background_points_list.append(node_positions)
                    background_density_logits_list.append(
                        node.get_densities(preactivation=True)
                    )  # no activation as input for cross_entropy
                    semantic_mask_list.append(
                        torch.ones(node_positions.shape[0], dtype=torch.bool, device=node_positions.device)
                    )

        if len(background_points_list) == 0 or len(tracks_list) == 0:
            return None

        background_points = torch.cat([t for t in background_points_list])
        background_density_logits = torch.cat([t for t in background_density_logits_list])
        semantic_mask = torch.cat([t for t in semantic_mask_list])
        tracks = CuboidTracks.Ops.concatenate(tracks_list)

        # approximate the timestamp as the median timestamp of the batch.
        # Use the GPU-resident `timestamps_startend_us` directly (no host↔device
        # copy). Keep the result on-device and pass a shape-[1] view as the
        # kernel's broadcast arg — `point_cuboidtracks_intersection_cu` accepts
        # `timestamps_us.size(0) == 1` and the size-1 branch reads `[0]`
        # directly, so we avoid the N-element buffer that `.expand(N).contiguous()`
        # would materialize every forward.
        timestamps_startend_us = unpack_optional(unpack_optional(target.rendering).camera).timestamps_startend_us
        # timestamps_startend_us[0] is shape [2] (start, end). torch.mean(int64,
        # dtype=float64) casts elements to float64 element-wise before summing,
        # so the precision bound is per-value: microsecond Unix-epoch timestamps
        # round-trip through float64 exactly until ~year 2255 (when an element
        # crosses 2^53).
        mean_timestamp_us = torch.mean(timestamps_startend_us[0], dtype=torch.float64).to(torch.int64)
        timestamp_us_mean = mean_timestamp_us.reshape(1)

        point_cuboidtracks_intersection_mask = tracks.point_intersection(
            background_points, timestamps_us=timestamp_us_mean, return_dense_mask=False
        )

        point_cuboidtracks_intersection_mask = point_cuboidtracks_intersection_mask & semantic_mask.view_as(
            point_cuboidtracks_intersection_mask
        )

        # Clamp logits to avoid numerical instability.
        point_cuboidtracks_intersection_mask = point_cuboidtracks_intersection_mask & (
            background_density_logits > self.density_logits_min
        )
        # Only in-track gaussians should contribute: use reduce_mask so out-of-track positions
        # are zeroed in the reduction. BCE(0, 0) would be ~0.693, so multiplying logits by
        # the mask would not zero the loss; we must use reduce_mask on the per-element loss.
        # Guard the loss input with the mask before evaluating it: a non-finite
        # logit fails the `> density_logits_min` comparison above (NaN compares
        # false) and is masked out, but it would still flow through
        # bce_with_logits and poison the masked-mean reduction
        # (f(NaN) * 0 == NaN). The boolean mask and scalar false branch avoid
        # any auxiliary mask or zero-tensor allocations; where's guarded
        # output is the only new tensor.
        mask_as_logits = point_cuboidtracks_intersection_mask.view(-1, *([1] * (background_density_logits.dim() - 1)))
        guarded_logits = torch.where(mask_as_logits, background_density_logits, 0.0)
        return self.apply_loss_fn(
            guarded_logits,
            torch.zeros_like(background_density_logits),
            reduce_mask=point_cuboidtracks_intersection_mask,
        )


@register_loss("semantic")
class SemanticLoss(BaseRenderLoss):
    camera_loss_kind = CameraLossKind.SEMANTIC

    def __init__(self, config: LossItemConfig, trainer_config: TrainerConfig, **kwargs) -> None:
        super().__init__("semantic", config, trainer_config, **kwargs)

    def forward(
        self,
        results: GaussiansCompositeReturn,
        target: DataAndRenderingBatch,
        model: BaseModel,
    ) -> Optional[LossReturn]:
        if (camera := target.data.camera) is None:
            return None

        # Skip HARMONIZED rays in the mask instead of checking n_harmonized to avoid CUDA sync
        rgb_loss_mask = camera_loss_valid_mask(self.camera_loss_kind, unpack_optional(camera.labels.flags))
        rgb_loss_mask = rgb_loss_mask.squeeze(-1)  # [batch, height, width]

        # Get full tensors without boolean indexing to avoid CUDA sync from nonzero()
        semantic_logits = unpack_optional(
            unpack_optional(unpack_optional(results.rendered_cam).extra_ray_signals).semantic_logits
        )  # [n_rays, num_classes]
        semantic_target = unpack_optional(camera.labels.semantic)  # [batch, height, width]

        assert semantic_logits.shape[0] == semantic_target.numel(), (
            f"Semantic logits and targets shape mismatch: "
            f"logits has {semantic_logits.shape[0]} rays but target has {semantic_target.numel()} pixels "
            f"(target shape: {semantic_target.shape})"
        )

        # Flatten for cross_entropy: logits [n_pixels, num_classes], target [n_pixels]
        semantic_logits_flat = semantic_logits  # already [n_rays, num_classes]
        semantic_target_flat = semantic_target.flatten()  # [n_pixels]
        mask_flat = rgb_loss_mask.flatten().float()  # [n_pixels]

        # Replace invalid target indices with 0 to avoid index-out-of-bounds in cross_entropy
        # The loss for these pixels will be zeroed out by the mask anyway
        semantic_target_safe = torch.where(
            rgb_loss_mask.flatten(), semantic_target_flat, torch.zeros_like(semantic_target_flat)
        )

        # Compute loss on full tensor, then mask. This avoids boolean indexing which causes sync.
        return self.apply_loss_fn(
            semantic_logits_flat,
            semantic_target_safe,
            reduce_mask=mask_flat,
        )


@register_loss("bilateral_grid_per_frame_spatial_tv")
class BilateralGridPerFrameSpatialTVLoss(BaseRenderLoss):
    def __init__(self, config: LossItemConfig, trainer_config: TrainerConfig, **kwargs) -> None:
        super().__init__("bilateral_grid_per_frame_spatial_tv", config, trainer_config, **kwargs)
        # Different grids might have different shapes, so we apply the loss function
        # on each before reducing
        inner_loss_fn = self.loss_fn
        self.loss_fn = lambda x: torch.cat([inner_loss_fn(grid).view(-1) for grid in x])

    def forward(
        self,
        results: GaussiansCompositeReturn,
        target: DataAndRenderingBatch,
        model: BaseModel,
    ) -> Optional[LossReturn]:
        """
        Compute the total variation loss for the per-frame bilateral grid.

        Returns:
            tv_loss (torch.Tensor): Total variation loss, a scalar tensor.
        """
        grids = _get_bilateral_grids(model, BilateralGridPerFrame)

        if not grids:
            return None

        return self.apply_loss_fn([x.bilateral_grid.grid for x in grids])


@register_loss("bilateral_grid_per_frame_temporal_tv")
class BilateralGridPerFrameTemporalTVLoss(BaseRenderLoss):
    def __init__(self, config: LossItemConfig, trainer_config: TrainerConfig, **kwargs) -> None:
        super().__init__("bilateral_grid_per_frame_temporal_tv", config, trainer_config, **kwargs)
        # Different grids might have different shapes, so we apply the loss function
        # on each before reducing
        inner_loss_fn = self.loss_fn
        self.loss_fn = lambda x: torch.cat([inner_loss_fn(grid, mask).view(-1) for grid, mask in x])

    def forward(
        self,
        results: GaussiansCompositeReturn,
        target: DataAndRenderingBatch,
        model: BaseModel,
    ) -> Optional[LossReturn]:
        """
        Compute the total variation loss for the per-frame bilateral grid.

        Returns:
            tv_loss (torch.Tensor): Total variation loss, a scalar tensor.
        """
        grids = _get_bilateral_grids(model, BilateralGridPerFrame)

        if not grids:
            return None

        return self.apply_loss_fn([(x.bilateral_grid.grid, x.same_camera_as_next_frame) for x in grids])


@register_loss("bilateral_grid_per_camera_tv")
class BilateralGridPerCameraTVLoss(BaseRenderLoss):
    def __init__(self, config: LossItemConfig, trainer_config: TrainerConfig, **kwargs) -> None:
        super().__init__("bilateral_grid_per_camera_tv", config, trainer_config, **kwargs)
        # Different grids might have different shapes, so we apply the loss function
        # on each before reducing
        inner_loss_fn = self.loss_fn
        self.loss_fn = lambda x: torch.cat([inner_loss_fn(grid).view(-1) for grid in x])

    def forward(
        self,
        results: GaussiansCompositeReturn,
        target: DataAndRenderingBatch,
        model: BaseModel,
    ) -> Optional[LossReturn]:
        """
        Compute the total variation loss for the per-camera bilateral grid.

        Returns:
            tv_loss (torch.Tensor): Total variation loss, a scalar tensor.
        """
        grids = _get_bilateral_grids(model, BilateralGridPerCamera)

        if not grids:
            return None

        return self.apply_loss_fn([x.bilateral_grid.grid for x in grids])


@register_loss("bilateral_grid_drift")
class BilateralGridDriftLoss(BaseRenderLoss):
    def __init__(self, config: LossItemConfig, trainer_config: TrainerConfig, **kwargs) -> None:
        super().__init__("bilateral_grid_drift", config, trainer_config, **kwargs)
        # Different grids might have different shapes, so we apply the loss function on each
        # before reducing
        inner_loss_fn = self.loss_fn
        self.loss_fn = lambda x: torch.cat([inner_loss_fn(grid).view(-1) for grid in x])

    def forward(
        self,
        results: GaussiansCompositeReturn,
        target: DataAndRenderingBatch,
        model: BaseModel,
    ) -> Optional[LossReturn]:
        """
        Penalizes deviation from the identity transform across the whole grid.
        """
        grids = _get_bilateral_grids(model, BilateralGridPerCamera) + _get_bilateral_grids(model, BilateralGridPerFrame)

        if not grids:
            return None

        return self.apply_loss_fn([x.bilateral_grid.grid for x in grids])


@register_loss("ppisp")
class PPISPLoss(BaseRenderLoss):
    """
    Compute regularization loss for PPISP using the library's built-in regularization.

    This loss wraps the new PPISP library's `get_regularization_loss()` method, which
    computes weighted regularization terms for exposure, vignetting, color correction,
    and CRF parameters. The regularization weights are configured on the PPISP module
    itself, not through the loss config.

    For legacy PPISP (render-only), this loss will not find a PPISP module and returns None.
    """

    def __init__(self, config: LossItemConfig, trainer_config: TrainerConfig, **kwargs) -> None:
        super().__init__("ppisp", config, trainer_config, **kwargs)

    def forward(
        self,
        results: GaussiansCompositeReturn,
        target: DataAndRenderingBatch,
        model: BaseModel,
    ) -> Optional[LossReturn]:
        # Find the PPISP post-processing module (new library version).
        match model:
            case GaussiansComposite():
                for pp in model.post_processings:
                    if isinstance(pp, PPISPPostProcessing):
                        return self.apply_loss_fn(pp.ppisp)
            case _:
                raise RuntimeError(f"{self.__class__.__name__} got unsupported model type {type(model)}")

        return None


@register_loss("ssim")
class SSIMLoss(BaseRenderLoss, BaseLossWithSemanticWeights):
    """PyTorch SSIM loss used when ``ssim_ssim_mean`` is not routed through ``ModuleLosses``.

    Default configs (e.g. ``configs/loss/gaussians.yaml``: ``window_size: 11``, ``reduce: mean``,
    ``mask.mode: GT``, no semantic lambdas) take the gsplat SSIM path instead:
    ``LossAggregator`` → ``ModuleLosses.forward`` → ``nurec.libs.losses.models.utils.fused_ssim_loss_map``
    → ``gsplat.losses.FusedSSIMLosses``. Gsplat uses fused CUDA for supported 11x11 cases
    and falls back internally otherwise; targets requiring gradients are rejected.

    This class remains for ``force_disable_cuda``, semantic SSIM weights, or any setup where
    ``ssim_ssim_mean`` is not listed in ``ModuleLosses.available``.
    """

    camera_loss_kind = CameraLossKind.SSIM

    def __init__(self, config: LossItemConfig, trainer_config: TrainerConfig, **kwargs) -> None:
        loss_name = "ssim"
        super().__init__(loss_name, config, trainer_config, **kwargs)

        assert config.reduce.name == "mean", "Invalid reduce name for SSIM: expected 'mean'."
        # The implementation uses a sum reduction and pre-multiplied mask to avoid CUDA synchronization,
        # the final behavior is equivalent to a mean reduction.
        # If we choose other reduction, we need to update the ssim function in loss_fns.py accordinly.
        self.reduce_fn = SumReduceFn(config.reduce)

        self.window_size = unpack_optional(config.window_size)
        self.channel = unpack_optional(config.channel)

        self.window = create_window(window_size=self.window_size, channel=self.channel)

        self.mask_mode = unpack_optional(config.mask).mode
        self.mask_value = unpack_optional(config.mask).value

    def forward(
        self,
        results: GaussiansCompositeReturn,
        target: DataAndRenderingBatch,
        model: BaseModel,
    ) -> Optional[LossReturn]:
        if (camera := target.data.camera) is None:
            return None

        target_rgb = unpack_optional(camera.labels.rgb)  # [batch, height, width, 3]
        pred_rgb = unpack_optional(unpack_optional(results.rendered_cam).rgb).reshape_as(target_rgb)
        extra = unpack_optional(results.rendered_cam).extra_ray_signals
        if extra is not None and extra.rgb_before_appearance is not None:
            pred_rgb = extra.rgb_before_appearance.reshape_as(target_rgb)  # VastGaussian: SSIM sees the raw render
        rays_valid_mask = camera_loss_valid_mask(
            self.camera_loss_kind, unpack_optional(camera.labels.flags)
        )  # [batch, height, width, 1]

        # To [B, C, H, W]
        target_rgb = target_rgb.permute(0, 3, 1, 2)
        pred_rgb = pred_rgb.permute(0, 3, 1, 2)
        valid_mask = rays_valid_mask.permute(0, 3, 1, 2)
        window = self.window.type_as(pred_rgb)

        if self.mask_mode == "constant":
            mask_value = torch.full_like(target_rgb, self.mask_value)
        else:
            mask_value = target_rgb

        return self.apply_loss_fn(
            img1=pred_rgb,
            img2=target_rgb,
            mask=valid_mask,
            mask_value=mask_value,
            window=window,
            window_size=self.window_size,
            channel=self.channel,
            frame_labels=camera.labels,
            # The return value of ssim is still the same as input, so no need to pass in mask
            frame_labels_mask=None,
        )


def _ssim_map(img1: torch.Tensor, img2: torch.Tensor, window: torch.Tensor, window_size: int) -> torch.Tensor:
    """Per-pixel SSIM of two [H, W, C] images, averaged over channels -> [H, W]."""
    x = img1.permute(2, 0, 1).unsqueeze(0)
    y = img2.permute(2, 0, 1).unsqueeze(0)
    channels = x.shape[1]
    pad = window_size // 2

    def blur(t: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.conv2d(t, window, padding=pad, groups=channels)

    mu_x, mu_y = blur(x), blur(y)
    var_x = blur(x * x) - mu_x * mu_x
    var_y = blur(y * y) - mu_y * mu_y
    cov = blur(x * y) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2))
    return ssim.mean(dim=1)[0]


@register_loss("ecad_confidence")
class ECADConfidenceLoss(BaseRenderLoss):
    """Confidence loss of ECAD (Hofer et al., arXiv 2610.13541, Eq. 3) plus the probes the ECAD strategy reads.

    L_conf(p) = L_rgb(p) * C(p) - beta * log C(p), with C(p) the alpha-blended per-Gaussian confidence clamped to
    [c_min, c_max] and L_rgb = l1 * |pred - gt| + (1 - l1) * (1 - SSIM) per pixel. Only the confidences learn from it:
    L_rgb is detached and C reaches them through a zero-valued twin channel (see `_ECADRenderState`), so colours and
    geometry receive no gradient from this term. The same surrogate routes the per-pixel SSIM error and a constant 1
    into the two probe channels, whose gradients the strategy reads. Its forward value is exactly zero.

    `lambdas` may set beta (0.075), l1 (0.8), c_min (1e-3) and c_max (5).
    """

    camera_loss_kind = CameraLossKind.RGB

    def __init__(self, config: LossItemConfig, trainer_config: TrainerConfig, **kwargs) -> None:
        super().__init__("ecad_confidence", config, trainer_config, **kwargs)
        lambdas = config.lambdas or {}
        self.beta = float(lambdas.get("beta", 0.075))
        self.l1_weight = float(lambdas.get("l1", 0.8))
        self.c_min = float(lambdas.get("c_min", 1e-3))
        self.c_max = float(lambdas.get("c_max", 5.0))
        self.window_size = 11
        self.window = create_window(window_size=self.window_size, channel=3)

    def forward(
        self,
        results: GaussiansCompositeReturn,
        target: DataAndRenderingBatch,
        model: BaseModel,
    ) -> Optional[LossReturn]:
        from nurec.models.gaussians.renderers import ECAD_RENDER_STATE

        rendered = ECAD_RENDER_STATE.rendered
        ECAD_RENDER_STATE.rendered = None
        if rendered is None or (camera := target.data.camera) is None or results.rendered_cam is None:
            return None

        labels = camera.labels
        gt = unpack_optional(labels.rgb)  # [1, H, W, 3]
        assert gt.shape[0] == 1, "ecad_confidence expects one image per batch"
        if rendered.shape[:2] != gt.shape[1:3]:
            raise ValueError(f"ECAD channels {tuple(rendered.shape)} do not match the image {tuple(gt.shape)}")
        valid = camera_loss_valid_mask(self.camera_loss_kind, unpack_optional(labels.flags))[0, ..., 0]  # [H, W]
        gt = gt[0]
        with torch.no_grad():
            pred = unpack_optional(results.rendered_cam.rgb).reshape(gt.shape).detach()
            window = self.window.to(device=gt.device, dtype=gt.dtype)
            ssim = _ssim_map(torch.where(valid[..., None], pred, gt), gt, window, self.window_size)
            ssim_error = (1.0 - ssim).clamp_min(0.0)
            l_rgb = self.l1_weight * (pred - gt).abs().mean(dim=-1) + (1.0 - self.l1_weight) * ssim_error
            c_raw = rendered[..., 0].detach()
            inside = (c_raw > self.c_min) & (c_raw < self.c_max)
            c = c_raw.clamp(self.c_min, self.c_max)
            n_valid = valid.sum().clamp_min(1)
            g_conf = torch.where(valid & inside, (l_rgb - self.beta / c) / n_valid, 0.0)
            e_ssim = torch.where(valid, ssim_error, 0.0)
            w_pix = valid.to(rendered.dtype)

        surrogate = (
            (g_conf * rendered[..., 1]).sum() + (e_ssim * rendered[..., 2]).sum() + (w_pix * rendered[..., 3]).sum()
        )
        return LossReturn(
            name=self.name,
            lambda_=self.lambda_,
            value=surrogate.reshape(1),
            reduce_fn=self.reduce_fn,
        )


@register_loss("sky_env_map_background")
class SkyEnvMapBackgroundLoss(BaseRenderLoss):
    supported_model_types = (GaussiansComposite,)
    supported_background_types = (SkyEnvMapBackground,)

    def __init__(self, config: LossItemConfig, trainer_config: TrainerConfig, **kwargs) -> None:
        super().__init__("sky_env_map_background", config, trainer_config, **kwargs)

    def forward(
        self,
        results: GaussiansCompositeReturn,
        target: DataAndRenderingBatch,
        model: BaseModel,
    ) -> Optional[LossReturn]:
        assert isinstance(model, self.supported_model_types), (
            f"{self.__class__.__name__} is only defined for {[cls.__name__ for cls in self.supported_model_types]}"
        )
        assert isinstance(model.background, self.supported_background_types), (
            f"{self.__class__.__name__} is only defined for {[cls.__name__ for cls in self.supported_background_types]}"
        )

        wrap_around = False
        match model.background.envmap_type:
            case EnvMapType.EQUIRECTANGULAR:
                textures = model.background.textures.permute(0, 3, 1, 2).unsqueeze(2)
                wrap_around = True
            case EnvMapType.CUBEMAP:
                # total_variation_spatial assumes [B, C, D, H, W]. Treat cubemap faces as independent batches.
                textures = model.background.textures.permute(1, 4, 0, 2, 3)
            case _:
                raise NotImplementedError(f"Unsupported envmap_type: {model.background.envmap_type}")

        return self.apply_loss_fn(textures, wrap_around=wrap_around)


@register_loss("deform_smoothness")
class DeformSmoothnessLoss(BaseRenderLoss):
    def __init__(self, config: LossItemConfig, trainer_config: TrainerConfig, **kwargs) -> None:
        super().__init__("deform_smoothness", config, trainer_config, **kwargs)

    def forward(
        self,
        results: GaussiansCompositeReturn,
        target: DataAndRenderingBatch,
        model: BaseModel,
    ) -> Optional[LossReturn]:
        # This can happen if an image doesn't contain any deformable objects

        return (
            self.apply_loss_fn(
                results.deform_smoothness,
                reduce_mask=results.deform_smoothness_mask,
            )
            if results.deform_smoothness is not None
            else None
        )


@register_loss("unicycle_pose")
class UnicyclePoseLoss(BaseRenderLoss):
    """Regularize elevation-aware unicycle track calibration.

    The calibrator provides three residual families:
    - consistency: pose changes versus learned motion
    - smoothness: acceleration changes
    - pose: learned versus initialized positions

    Each family is mean-reduced, weighted by its ``lambdas`` entry, and summed.
    """

    def __init__(self, config: LossItemConfig, trainer_config: TrainerConfig, **kwargs) -> None:
        super().__init__("unicycle_pose", config, trainer_config, **kwargs)

        assert config.reduce.name == "mean", "Invalid reduce name for UnicyclePoseLoss: expected 'mean'"

        lambdas = config.lambdas or {}
        self.consistency_weight = lambdas.get("consistency", 0.5)
        self.smoothness_weight = lambdas.get("smoothness", 0.01)
        self.pose_weight = lambdas.get("pose", 0.1)

    @staticmethod
    def _collect_unicycle_tracks_calib_owners(
        model: BaseModel,
    ) -> list[RigidGaussianModel]:
        candidates: list[BaseModel] = [model]
        if isinstance(model, GaussiansComposite):
            candidates.extend(model.gaussians_nodes.values())

        return [
            candidate
            for candidate in candidates
            if isinstance(candidate, RigidGaussianModel)
            and isinstance(candidate.tracks_calib, ElevationAwareUnicycleTracksCalib)
        ]

    def forward(
        self,
        results: GaussiansCompositeReturn,
        target: DataAndRenderingBatch,
        model: BaseModel,
    ) -> Optional[LossReturn]:
        loss: torch.Tensor | None = None

        for owner in self._collect_unicycle_tracks_calib_owners(model):
            tracks_calib = cast(ElevationAwareUnicycleTracksCalib, owner.tracks_calib)
            uses_cuda = any(parameter.is_cuda for parameter in tracks_calib.parameters())
            if uses_cuda:
                residual_families = owner.get_tracks_calib_residuals()
            else:
                residual_families = TracksCalibResiduals(
                    consistency=tracks_calib.get_consistency(),
                    smoothness=tracks_calib.get_smoothness(),
                    pose_regularization=tracks_calib.get_pose_regularization(),
                )

            for residuals, weight in (
                (residual_families.consistency, self.consistency_weight),
                (residual_families.smoothness, self.smoothness_weight),
                (residual_families.pose_regularization, self.pose_weight),
            ):
                if residuals.numel() > 0:
                    term = weight * self.loss_fn(residuals).mean()
                    loss = term if loss is None else loss + term

        if loss is None:
            return None

        return LossReturn(
            name=self.name,
            lambda_=self.lambda_,
            value=loss,
            reduce_fn=self.reduce_fn,
        )


@register_loss("out_of_bound")
class OutOfBoundLoss(BaseRenderLoss):
    supported_model_types = (GaussiansComposite,)

    def __init__(self, config: LossItemConfig, trainer_config: TrainerConfig, **kwargs) -> None:
        super().__init__("out_of_bound", config, trainer_config, **kwargs)

    def forward(
        self,
        results: GaussiansCompositeReturn,
        target: DataAndRenderingBatch,
        model: BaseModel,
    ) -> Optional[LossReturn]:
        assert isinstance(model, self.supported_model_types), (
            f"{self.__class__.__name__} is only defined for {[cls.__name__ for cls in self.supported_model_types]}"
        )

        losses = []
        for node_id in model.get_gaussians_node_ids():
            node = model.gaussians_nodes[node_id]
            if not isinstance(node, RigidGaussianModel):
                continue

            positions = node.get_positions()
            cuboid_dims = node.cuboid_tracks.cuboids_dims
            gaussian_cuboid_ids = node.gaussian_cuboid_ids
            gaussian_cuboid_dims = cuboid_dims[gaussian_cuboid_ids]

            loss = torch.relu(positions.abs() - gaussian_cuboid_dims / 2)
            losses.append(loss)

        if len(losses) == 0:
            return None

        loss = torch.cat(losses)

        return self.apply_loss_fn(
            loss,
            torch.zeros_like(loss),
        )


@register_loss("road_gaussians")
class RoadGaussiansLoss(BaseRenderLoss):
    """
    Road distortion loss as proposed in:
        HUGSIM: A Real-Time, Photo-Realistic and Closed-Loop Simulator for Autonomous Driving
        Constrains the height variance of gaussians in the specified layer.
        We also constraint the rotation variance of the gaussians, whereas the paper just hardcodes them to the identity transformation
        for the road layer.
    """

    supported_model_types = (GaussiansComposite,)

    def __init__(self, config: LossItemConfig, trainer_config: TrainerConfig, **kwargs) -> None:
        loss_name = "road_gaussians"
        super().__init__(loss_name, config, trainer_config, **kwargs)
        self.layer_name = unpack_optional(config.layer_name)
        self.n_samples = unpack_optional(config.n_samples)
        self.grid_len = unpack_optional(config.grid_len)
        self.min = unpack_optional(config.min)
        self.range = unpack_optional(config.range)
        self.rotation_lambda = unpack_optional(config.rotation_lambda)
        self.use_cuda = kwargs.get("use_cuda", True)
        self.cuda_module = (
            FusedGroundGaussiansLosses(
                min_bias=self.min,
                range_bias=self.range,
                grid_len=self.grid_len,
                rotation_lambda=self.rotation_lambda,
                rotations_are_raw_wxyz=True,
            )
            if self.use_cuda
            else None
        )
        # The fused loss returns one non-negative scalar (a weighted mean over
        # bins of standard deviations), so with `fn: abs` / `reduce: mean` the
        # generic epilogue's abs, mean and lambda launches are exact identities.
        # Tiny rotation weights can underflow that scalar to zero while its
        # native derivative stays nonzero; there abs must still zero the
        # gradient, so only skip the epilogue for a safe weight range. The bound
        # leans on the kernel's std gradient cutoff: gsplat's std_dev_grad
        # returns 0 once (count - 1) * std < eps, so every term that still has a
        # gradient is >= eps^2 / (count * n_bins) and cannot flush to zero (the
        # kernels build with fast-math FTZ) unless scaled by a weight below eps.
        # The bounds are float32 by design: gsplat's launch functions take
        # rotation_lambda as a C float, so the weight is narrowed to float32
        # before it reaches the kernel whatever the tensor dtype.
        rotation_weight_is_safe = self.rotation_lambda == 0 or (
            torch.finfo(torch.float32).eps <= self.rotation_lambda <= torch.finfo(torch.float32).max
        )
        self._scalar_epilogue_is_identity = (
            self.use_cuda
            and self.loss_fn is torch.abs
            and isinstance(self.reduce_fn, MeanReduceFn)
            and rotation_weight_is_safe
        )

    def _make_random_values(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if device.type == "cuda":
            return torch.rand(
                self.n_samples,
                generator=get_cuda_generator(device),
                device=device,
                dtype=dtype,
            )
        return torch.rand(self.n_samples, device=device, dtype=dtype)

    def forward_direct_cam(
        self,
        positions_cam: torch.Tensor,
        rotations_cam: torch.Tensor,
        random_values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Direct forward pass that takes explicit arguments instead of extracting from model/target.
        This method is useful for testing and when you have the data directly available.

        This is the second substep of the forward pass, split in two functions to allow easier testing and debugging.

        Args:
            positions_cam: Camera space positions [N, 3]
            rotations_cam: Camera space rotations as quaternions [N, 4] in xyzw format
            random_values: Random values for the bias [N]

        Returns:
            Loss tensor (scalar)
        """

        if random_values is None:
            # On CUDA, route through the shared generator so this fallback is
            # safe inside a torch.cuda.graph(...) captured region.
            random_values = self._make_random_values(positions_cam.device, positions_cam.dtype)

        biases = self.min + self.range * random_values

        distort_3d_loss = torch.zeros(
            1, dtype=positions_cam.dtype, device=positions_cam.device, requires_grad=True
        ).sum()
        euler_cam = quat_to_euler(rotations_cam)

        for bias_idx, bias in enumerate(biases):
            mask = (bias <= positions_cam[:, 2]) & (positions_cam[:, 2] < (bias + self.grid_len))
            if torch.sum(mask) <= 1:  # std on 1 element would return nan
                continue

            ys = positions_cam[mask, 1]
            pitch = euler_cam[mask, 1]
            roll = euler_cam[mask, 0]

            distort_3d_loss += torch.std(ys)
            distort_3d_loss += self.rotation_lambda * (torch.std(pitch) + torch.std(roll))

        distort_3d_loss = distort_3d_loss / self.n_samples

        return distort_3d_loss

    def positions_and_rotations_cam_from_world(
        self, positions_world: torch.Tensor, rotations_world: torch.Tensor, tquat_cam_world: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Transform world space positions and rotations to camera space.

        This is the first substep of the forward pass, split in two functions to allow easier testing and debugging.

        Args:
            positions_world: World space positions [N, 3]
            rotations_world: World space rotations as quaternions [N, 4] in xyzw format
            tquat_cam_world: Camera pose as translation-quaternion [7] in format [tx, ty, tz, qx, qy, qz, qw]

        Returns:
            tuple: (positions_cam, rotations_cam) where:
                - positions_cam: Camera space positions [N, 3]
                - rotations_cam: Camera space rotations as quaternions [N, 4] in xyzw format
        """
        T_cam_world = tquat_to_se3_matrix(tquat_cam_world)
        T_world_cam = se3_matrix_inverse(T_cam_world)
        positions_cam = (T_world_cam[:3, :3] @ positions_world.T).T + T_world_cam[:3, 3]
        rotations_cam = so3_matrix_to_quat(
            T_cam_world[:3, :3].T @ quat_to_so3_matrix(rotations_world) @ T_cam_world[:3, :3]
        )

        return positions_cam, rotations_cam

    def forward_direct(
        self,
        positions_world: torch.Tensor,
        rotations_world: torch.Tensor,
        tquat_cam_world: torch.Tensor,
        random_values: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Direct forward pass that takes explicit arguments instead of extracting from model/target.
        This method is useful for testing and when you have the data directly available.

        Args:
            positions_world: World space positions [N, 3]
            rotations_world: World space rotations as quaternions [N, 4] in wxyz format
                (the model's stored parameter layout); normalization is not required
            tquat_cam_world: Camera pose as translation-quaternion [7] in format [tx, ty, tz, qx, qy, qz, qw]
            random_values: Optional random values for bias sampling [n_samples]

        Returns:
            Loss tensor (scalar)
        """
        if random_values is None:
            random_values = self._make_random_values(positions_world.device, positions_world.dtype)
        if self.use_cuda and self.cuda_module is not None:
            return self.cuda_module(
                positions_world,
                rotations_world,
                tquat_cam_world,
                random_values,
            )
        else:
            # wxyz -> xyzw; quat_to_so3_matrix normalizes.
            rotations_world = torch.cat((rotations_world[:, 1:], rotations_world[:, :1]), dim=-1)
            # Transform world coordinates to camera coordinates
            positions_cam, rotations_cam = self.positions_and_rotations_cam_from_world(
                positions_world, rotations_world, tquat_cam_world
            )

            return self.forward_direct_cam(positions_cam, rotations_cam, random_values)

    def forward(
        self,
        results: GaussiansCompositeReturn,
        target: DataAndRenderingBatch,
        model: BaseModel,
    ) -> Optional[LossReturn]:
        """
        Forward pass that takes model and target as arguments.

        Args:
            results: Results from the model
            target: Target data
            model: Model

        Returns:
            Loss return object
        """
        if (rendering := target.rendering) is None:
            return None
        if (camera := rendering.camera) is None:
            return None
        assert camera.b == 1, f"Expected camera batch size to be 1, got {camera.b}"

        assert isinstance(model, self.supported_model_types), (
            f"{self.__class__.__name__} is only defined for {[cls.__name__ for cls in self.supported_model_types]}"
        )

        # The layer may be disabled via a component-only overlay —
        # disabled layers are not built into model.gaussians_nodes, so a
        # missing entry means "layer is disabled": no-op this loss.
        if self.layer_name not in model.gaussians_nodes:
            return None

        node = model.gaussians_nodes[self.layer_name]
        positions_world = node.get_positions()
        # Raw stored parameters: the loss normalizes and reorders them itself,
        # which saves the separate F.normalize + cat launches every step.
        rotations_world = node.get_rotations(quaternion_format="wxyz", preactivation=True)
        # camera.poses_tquat_startend has shape (batch_size == 1, 2, 7)
        tquat_cam_world = camera.poses_tquat_startend.squeeze(0)[0]

        # Use the direct forward method
        distort_3d_loss = self.forward_direct(
            positions_world,
            rotations_world,
            tquat_cam_world,
        )

        if self._scalar_epilogue_is_identity and distort_3d_loss.is_cuda and distort_3d_loss.numel() == 1:
            # Weight the native scalar directly (no launch when lambda_ is 1.0).
            # Read lambda_ per call: a LambdaScheduler rewrites it at train-batch
            # start, which is also when reduce() would have read it.
            lambda_ = self.lambda_
            return LossReturn(
                name=self.name,
                lambda_=lambda_,
                value=distort_3d_loss,
                reduce_fn=self.reduce_fn,
                _reduced_value=distort_3d_loss if lambda_ == 1.0 else lambda_ * distort_3d_loss,
            )
        return self.apply_loss_fn(distort_3d_loss)


@register_loss("gaussian_scale")
class GaussianScaleLoss(BaseRenderLoss):
    supported_model_types = (GaussiansComposite,)
    """
    Penalizes high scale values - intended to be used with MCMC
    """

    def __init__(self, config: LossItemConfig, trainer_config: TrainerConfig, **kwargs) -> None:
        loss_name = "gaussian_scale"
        super().__init__(loss_name, config, trainer_config, **kwargs)
        self.layer_lambdas = config.layer_lambdas or {}

    def forward(
        self,
        results: GaussiansCompositeReturn,
        target: DataAndRenderingBatch,
        model: BaseModel,
    ) -> Optional[LossReturn]:
        assert isinstance(model, self.supported_model_types), (
            f"{self.__class__.__name__} is only defined for {[cls.__name__ for cls in self.supported_model_types]}"
        )

        scales = []
        for node_id in model.get_gaussians_node_ids():
            scales.append(model.gaussians_nodes[node_id].get_scales() * self.layer_lambdas.get(node_id, 1.0))

        all_scales = torch.cat(scales)
        visibility = get_rendered_visibility_mask(
            results=results,
            element_count=all_scales.shape[0],
            device=all_scales.device,
            dtype=all_scales.dtype,
            visibility_filter=self.visibility_filter,
            occlusion_aware=self.occlusion_aware,
        ).view(-1, *[1] * (all_scales.ndim - 1))
        if self.visibility_filter:
            _maybe_update_mcmc_visibility_counters(
                model=model,
                results=results,
                visibility=visibility,
            )

        return self.apply_loss_fn(all_scales * visibility)


@register_loss("gaussian_density")
class GaussianDensityLoss(BaseRenderLoss):
    supported_model_types = (GaussiansComposite,)
    """
    Penalizes high density values - intended to be used with MCMC
    """

    def __init__(self, config: LossItemConfig, trainer_config: TrainerConfig, **kwargs) -> None:
        loss_name = "gaussian_density"
        super().__init__(loss_name, config, trainer_config, **kwargs)
        self.layer_lambdas = config.layer_lambdas or {}

    def forward(
        self,
        results: GaussiansCompositeReturn,
        target: DataAndRenderingBatch,
        model: BaseModel,
    ) -> Optional[LossReturn]:
        assert isinstance(model, self.supported_model_types), (
            f"{self.__class__.__name__} is only defined for {[cls.__name__ for cls in self.supported_model_types]}"
        )

        densities: List[torch.Tensor] = []
        for node_id in model.get_gaussians_node_ids():
            densities.append(model.gaussians_nodes[node_id].get_densities() * self.layer_lambdas.get(node_id, 1.0))

        all_densities = torch.cat(densities)
        visibility = get_rendered_visibility_mask(
            results=results,
            element_count=all_densities.shape[0],
            device=all_densities.device,
            dtype=all_densities.dtype,
            visibility_filter=self.visibility_filter,
            occlusion_aware=self.occlusion_aware,
        ).view(-1, *[1] * (all_densities.ndim - 1))
        if self.visibility_filter:
            _maybe_update_mcmc_visibility_counters(
                model=model,
                results=results,
                visibility=visibility,
            )

        return self.apply_loss_fn(all_densities * visibility)


@register_loss("gaussian_z_scale")
class GaussianZScaleLoss(BaseRenderLoss):
    supported_model_types = (GaussiansComposite,)
    """
    Penalizes z-scale values above a given threshold for the specified layer
    """

    def __init__(self, config: LossItemConfig, trainer_config: TrainerConfig, **kwargs) -> None:
        loss_name = "gaussian_z_scale"
        super().__init__(loss_name, config, trainer_config, **kwargs)
        self.layer_name = unpack_optional(config.layer_name)
        self.road_z_scale = unpack_optional(config.road_z_scale)

    def forward(
        self,
        results: GaussiansCompositeReturn,
        target: DataAndRenderingBatch,
        model: BaseModel,
    ) -> Optional[LossReturn]:
        assert isinstance(model, self.supported_model_types), (
            f"{self.__class__.__name__} is only defined for {[cls.__name__ for cls in self.supported_model_types]}"
        )

        # The layer may be disabled via a component-only overlay —
        # disabled layers are not built into model.gaussians_nodes, so a
        # missing entry means "layer is disabled": no-op this loss.
        if self.layer_name not in model.gaussians_nodes:
            return None

        return self.apply_loss_fn(
            torch.relu(model.gaussians_nodes[self.layer_name].get_scales()[:, 2] - self.road_z_scale)
        )


@register_loss("node_semantic_gaussians")
class NodeSemanticGaussiansLoss(BaseRenderLoss):
    supported_model_types = (GaussiansComposite,)
    """
    A loss that penalizes the density of Gaussian nodes based on their semantic labels.
    For each node, it either:
    1. For layer_labels_to_use: Penalizes densities of points that have semantic labels other than the specified ones,
       effectively restricting this layer to only contain the specified semantic classes
    2. For layer_labels_to_exclude: Penalizes densities of points that have the specified semantic labels,
       effectively preventing this layer from containing these semantic classes
    This helps enforce semantic consistency by controlling which semantic classes can appear in each layer.
    """

    def __init__(self, config: LossItemConfig, trainer_config: TrainerConfig, **kwargs) -> None:
        super().__init__("node_semantic_gaussians", config, trainer_config, **kwargs)
        self.layer_labels_to_use = config.layer_labels_to_use or {}
        self.layer_labels_to_exclude = config.layer_labels_to_exclude or {}

    def initialize(self, train_dataset: BaseDataset) -> None:
        if self.lambda_ == 0:
            return

        super().initialize(train_dataset)

        assert isinstance(train_dataset, NCOREDataset), "NodeSemanticGaussiansLoss only supports NCOREDataset"
        self.semantic_classes_map = unpack_optional(
            train_dataset.get_datasource().get_semantic_classes_map(
                camera_semantics=True,
                lidar_semantics=False,
            )
        )
        assert self.semantic_classes_map is not None, (
            f"{self.__class__.__name__} failed to load semantic classes "
            "from NCore dataset (is aux-data enabled / is semantic aux-data available?)"
        )

        # Validate all configured labels exist in semantic_classes_map
        invalid_labels = []
        for node_id, labels in self.layer_labels_to_use.items():
            for label in labels:
                if label not in self.semantic_classes_map:
                    invalid_labels.append(f"{node_id}: {label}")
        for node_id, labels in self.layer_labels_to_exclude.items():
            for label in labels:
                if label not in self.semantic_classes_map:
                    invalid_labels.append(f"{node_id}: {label}")

        if invalid_labels:
            raise ValueError(
                f"{self.__class__.__name__}: The following labels in 'layer_labels_to_use' or 'layer_labels_to_exclude' "
                f"are not found in semantic_classes_map: {invalid_labels}. "
                f"Available labels: {list(self.semantic_classes_map.keys())}"
            )

    def forward(
        self,
        results: GaussiansCompositeReturn,
        target: DataAndRenderingBatch,
        model: BaseModel,
    ) -> Optional[LossReturn]:
        assert isinstance(model, self.supported_model_types), (
            f"{self.__class__.__name__} is only defined for {[cls.__name__ for cls in self.supported_model_types]}"
        )

        # Collect full densities and penalty masks to avoid boolean indexing which causes CUDA sync
        densities_list: list[torch.Tensor] = []
        penalty_mask_list: list[torch.Tensor] = []

        for node_id in model.get_gaussians_node_ids():
            node = model.gaussians_nodes[node_id]
            try:
                semantic_logits = node.get_extra_signal_by_key("semantic_logits")
            except ValueError:
                semantic_logits = None
            if semantic_logits is None:
                # Skip this node if no semantic logits are available
                if node_id in self.layer_labels_to_use or node_id in self.layer_labels_to_exclude:
                    log.warning(
                        "Node '%s' is configured for node-semantic but missing 'semantic_logits', skipping it",
                        node_id,
                    )
                continue
            semantics = semantic_logits.argmax(dim=1)
            densities = node.get_densities(preactivation=True)  # no activation as input for cross_entropy

            if node_id in self.layer_labels_to_use:
                # Penalize densities where semantics DON'T match the labels to use
                keep_mask = torch.zeros_like(densities).bool()
                for label_to_use in self.layer_labels_to_use[node_id]:
                    keep_mask[semantics == self.semantic_classes_map[label_to_use]] = True
                penalty_mask = ~keep_mask  # penalize those NOT in keep_mask
                densities_list.append(densities)
                penalty_mask_list.append(penalty_mask)
            elif node_id in self.layer_labels_to_exclude:
                # Penalize densities where semantics match the excluded labels
                penalty_mask = torch.zeros_like(densities).bool()
                for label_to_exclude in self.layer_labels_to_exclude[node_id]:
                    penalty_mask[semantics == self.semantic_classes_map[label_to_exclude]] = True
                densities_list.append(densities)
                penalty_mask_list.append(penalty_mask)
            else:
                continue

        if len(densities_list) == 0:
            return None

        # Concatenate all densities and masks - no boolean indexing needed
        all_densities = torch.cat(densities_list)
        all_penalty_mask = torch.cat(penalty_mask_list)

        # Compute loss on all densities, use reduce_mask to only include
        # penalized ones. Guard the loss input with the mask before evaluating
        # it: the value is reduced as f(x) * mask, so a non-finite density on a
        # masked-out point would otherwise poison the reduction
        # (NaN * 0 == NaN). Masked lanes never see the raw logit; finite inputs
        # are unaffected (f(0) * 0 == f(x) * 0 == 0). The mask stays boolean
        # end-to-end and the false branch is a scalar (the masked-mean reduce
        # promotes bool in `value * mask` and counts it in `mask.sum()`), so
        # the guard adds no auxiliary mask or zero-tensor allocations on this
        # sync-free hot path -- only where's guarded output itself.
        guarded_densities = torch.where(all_penalty_mask, all_densities, 0.0)
        return self.apply_loss_fn(
            guarded_densities,
            torch.zeros_like(all_densities),
            reduce_mask=all_penalty_mask,
        )


@register_loss("gaussian_flatten")
class GaussianFlattenLoss(BaseRenderLoss):
    supported_model_types = (GaussiansComposite,)
    """
    Gaussian flatten regularization loss as proposed in:
        MTGS: Multi-Traversal Gaussian Splatting
        Penalizes non-flat gaussians - intended to be used with NormalLoss
        Choose axes_type = "fixed" if the gaussian model defines the normal as the direction of z-axis, which is the case for our gaussians.
        The MTGS paper uses axes_type = "free" as the paper defines the normal as the direction of the shortest axis.
    """

    def __init__(self, config: LossItemConfig, trainer_config: TrainerConfig, **kwargs) -> None:
        loss_name = "gaussian_flatten"
        super().__init__(loss_name, config, trainer_config, **kwargs)
        self.max_to_median_ratio_threshold = unpack_optional(config.max_to_median_ratio_threshold)
        self.axes_type = unpack_optional(config.axes_type)

    def forward(
        self,
        results: GaussiansCompositeReturn,
        target: DataAndRenderingBatch,
        model: BaseModel,
    ) -> Optional[LossReturn]:
        assert isinstance(model, self.supported_model_types), (
            f"{self.__class__.__name__} is only defined for {[cls.__name__ for cls in self.supported_model_types]}"
        )

        scales_list = []
        for node_id in model.get_gaussians_node_ids():
            scales_list.append(model.gaussians_nodes[node_id].get_scales())
        if not scales_list:
            raise ValueError(f"{self.__class__.__name__} needs at least one Gaussian node")
        scales = torch.cat(scales_list)

        if self.axes_type == "fixed":
            max_to_median_ratio = torch.max(
                scales[:, 0] / (scales[:, 1] + 1.0e-6), scales[:, 1] / (scales[:, 0] + 1.0e-6)
            )
            flatten_loss = torch.relu(max_to_median_ratio - self.max_to_median_ratio_threshold) + scales[:, 2]
        elif self.axes_type == "free":
            scales_sorted = scales.sort(dim=1, descending=False).values
            max_to_median_ratio = scales_sorted[:, 2] / (scales_sorted[:, 1] + 1.0e-6)
            flatten_loss = torch.relu(max_to_median_ratio - self.max_to_median_ratio_threshold) + scales_sorted[:, 0]
        else:
            raise ValueError(f"Invalid axes_type: {self.axes_type}")

        return self.apply_loss_fn(flatten_loss)


@register_loss("lpips")
class LPIPSLoss(BaseRenderLoss, BaseLossWithSemanticWeights):
    def __init__(self, config: LossItemConfig, trainer_config: TrainerConfig, **kwargs) -> None:
        super().__init__("lpips", config, trainer_config, **kwargs)
        # Lazy init network when operating devices can be determined
        self.network: LPIPSNetwork | None = None
        assert config.fn == "lpips", "LPIPSLoss is only compatible with lpips loss function"
        self.per_image_loss = True
        self.limit_max_resolution: int = config.limit_max_resolution or -1

    def _get_network(self, device: torch.device) -> LPIPSNetwork:
        if self.network is None:
            self.network = LPIPSNetwork().eval().to(device)
        return self.network

    def forward(
        self,
        results: GaussiansCompositeReturn,
        target: DataAndRenderingBatch,
        model: BaseModel,
    ) -> LossReturn | None:
        if (camera := target.data.camera) is None:
            return None

        if results.rendered_cam is None:
            return None

        labels = camera.labels  # [batch, height, width, D]
        rgb_loss_mask = camera_loss_valid_mask(
            CameraLossKind.RGB, unpack_optional(labels.flags)
        )  # [batch, height, width, 1]

        assert labels.rgb is not None, "RGB labels are required"
        gt_rgb = labels.rgb  # [batch, height, width, 3]
        pred_rgb = unpack_optional(results.rendered_cam.rgb)  # [n_rays, 3]
        pred_rgb = pred_rgb.reshape_as(gt_rgb)  # [batch, height, width, 3]

        # Copy over mask values
        pred_rgb = pred_rgb * rgb_loss_mask + gt_rgb * (~rgb_loss_mask)

        # Reshuffle to [B, C, H, W], and normalize to [-1, 1]
        pred_rgb = pred_rgb.moveaxis(-1, 1) * 2.0 - 1.0
        gt_rgb = gt_rgb.moveaxis(-1, 1) * 2.0 - 1.0

        # Per experience it seems that use higher resolution for LPIPS input does not help much
        # We can then limit the resolution to a smaller value to reduce the memory usage and improve speed
        if self.limit_max_resolution > 0:
            height, width = pred_rgb.shape[2], pred_rgb.shape[3]
            if height > self.limit_max_resolution or width > self.limit_max_resolution:
                scale_factor = min(self.limit_max_resolution / height, self.limit_max_resolution / width)
                new_height, new_width = int(height * scale_factor), int(width * scale_factor)
                pred_rgb = torch.nn.functional.interpolate(
                    pred_rgb, size=(new_height, new_width), mode="bilinear", align_corners=False
                )
                gt_rgb = torch.nn.functional.interpolate(
                    gt_rgb, size=(new_height, new_width), mode="bilinear", align_corners=False
                )

        return self.apply_loss_fn(
            pred_rgb, gt_rgb, self._get_network(pred_rgb.device), frame_labels=labels, frame_labels_mask=None
        )
