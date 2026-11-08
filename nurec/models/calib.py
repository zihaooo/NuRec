# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
import logging
import math

from abc import ABC, abstractmethod
from typing import List, Optional

import torch

from torch import nn

from nurec.config.model import CalibConfig
from nurec.config.trainer import TrainerConfig
from nurec.datasets.summary import DataSourceSummary
from nurec.models.base import BaseModel
from nurec.models.nn_extensions import module_call_type
from nurec.models.rig_spline import CameraExtrinsicResiduals, PackedRigSpline
from nurec.utils.batch import (
    CameraFreePoseViewGeometry,
    DataBatch,
    FrameMeta,
    LidarFreePoseViewGeometry,
    PoseCorrection,
    RenderingBatch,
    RenderingData,
)
from nurec.utils.optim import OptimizerLRSchedulerConfig, configure_optimizers
from nurec.utils.trainer import adjust_step_for_world_size
from nurec.utils.types import RigTrajectories


log = logging.getLogger(__name__)


class BaseCalib(BaseModel, ABC):
    """Base class for all calibration models"""

    config: CalibConfig  # type: ignore[assignment] # Override type annotation from BaseModel

    def __init__(self, config: CalibConfig, trainer_config: TrainerConfig):
        super().__init__(config.to_dictconfig())
        self.config = config  # type: ignore[assignment]  # Override with typed config
        self.trainer_config = trainer_config

        self.enabled: bool = config.lidar.enabled or config.camera.enabled
        if self.enabled:
            self.start_global_step: int = adjust_step_for_world_size(trainer_config, config.start_global_step)
            self.skip_first_pose_delta: bool = config.skip_first_pose_delta

            log.info(f"BaseCalib: start_global_step={self.start_global_step}")

    @staticmethod
    def factory(
        name: str, config: CalibConfig, trainer_config: TrainerConfig, datasource: DataSourceSummary
    ) -> BaseCalib:
        if name != config.name:
            raise ValueError(f"Calibration factory name {name!r} does not match config name {config.name!r}")
        if name == "rig-calib":
            if (rig_trajectories := datasource.get_rig_trajectories()) is None:
                raise RuntimeError("rig trajectories required for rig-calib calibration")
            return RigCalib.from_rig_trajectories(rig_trajectories, config, trainer_config)
        if name == "free-pose-calib" or name == "skip-calib":
            if (rig_trajectories := datasource.get_rig_trajectories()) is not None:
                return FreePoseCalib.from_rig_trajectories(rig_trajectories, config, trainer_config)
            if name == "skip-calib":
                # skip-calib is a no-op pass-through. Allow construction when the caller
                # has no rig_trajectories available (e.g. the PLY viewer, which feeds
                # rendering_data directly and never invokes calib.forward()).
                return FreePoseCalib(config=config, trainer_config=trainer_config)
            raise RuntimeError(f"Rig_trajectories required for {name} calibration")
        else:
            raise TypeError(f"Unknown calib {name=}.")

    @property
    @abstractmethod
    def params(self) -> List[nn.Parameter]: ...

    @abstractmethod
    def forward(
        self, data_batch: DataBatch, skip_calib: bool = False, global_step_for_prober: Optional[int] = None
    ) -> RenderingBatch: ...

    def configure_optimizers(self, name_prefix: str = "") -> list[OptimizerLRSchedulerConfig]:
        return configure_optimizers(self.config.to_dictconfig(), self.trainer_config, self, name_prefix)

    def update_step_train_batch_start(self, epoch: int, global_step: int, system, **kwargs) -> dict[str, torch.Tensor]:
        if self.enabled and global_step >= self.start_global_step:
            # unfreeze parameters if enabled
            for param in self.params:
                param.requires_grad_(True)
        else:
            self.requires_grad_(False)  # freeze all unconditionally, as start step not reached yet
        return {}  # no new / additional parameters

    def regularization_loss(self) -> tuple[str, torch.Tensor, float] | None:
        return None

    def apply_to_rig_trajectories(self, rig_trajectories: RigTrajectories) -> RigTrajectories:
        """Return the rig trajectories to serialize into an artifact."""
        return rig_trajectories

    __call__ = module_call_type(forward)


class FreePoseCalib(BaseCalib):
    """Free pose calibration"""

    def __init__(
        self,
        config: CalibConfig,
        trainer_config: TrainerConfig,
        lidar_view_geometry: Optional[LidarFreePoseViewGeometry] = None,
        camera_view_geometry: Optional[CameraFreePoseViewGeometry] = None,
    ):
        super().__init__(config, trainer_config)
        self.lidar_view_geometry = lidar_view_geometry
        self.camera_view_geometry = camera_view_geometry
        self.enable_torch_compile: bool = config.enable_torch_compile

        calib_enabled_camera = camera_view_geometry is not None and camera_view_geometry.enable_calib
        calib_enabled_lidar = lidar_view_geometry is not None and lidar_view_geometry.enable_calib
        log.info(f"FreePoseCalib: calib {'ON' if calib_enabled_camera else 'OFF'} for cameras")
        log.info(f"FreePoseCalib: calib {'ON' if calib_enabled_lidar else 'OFF'} for lidars")
        log.info(f"FreePoseCalib: enable_torch_compile={self.enable_torch_compile}")

        # forward() requires at least one view geometry; the skip-calib no-op path
        # (no rig_trajectories) constructs without either and never calls forward().
        if self.enabled:
            assert lidar_view_geometry is not None or camera_view_geometry is not None, (
                "At least one view geometry must be provided when calibration is enabled"
            )

    @property
    def params(self) -> List[nn.Parameter]:
        parameters: list[nn.Parameter] = []
        if self.camera_view_geometry is not None and self.camera_view_geometry.enable_calib:
            assert self.camera_view_geometry.embeds is not None
            parameters.extend(list(self.camera_view_geometry.embeds.parameters()))
        if self.lidar_view_geometry is not None and self.lidar_view_geometry.enable_calib:
            assert self.lidar_view_geometry.embeds is not None
            parameters.extend(list(self.lidar_view_geometry.embeds.parameters()))
        return parameters

    def forward(
        self, data_batch: DataBatch, skip_calib: bool = False, global_step_for_prober: Optional[int] = None
    ) -> RenderingBatch:
        rendering_data_camera: Optional[RenderingData] = None
        if data_batch.camera is not None:
            assert self.camera_view_geometry is not None
            skip_camera_calib = skip_calib or (
                self.config.skip_first_pose_delta and data_batch.camera.meta[0].unique_frame_idx == 0
            )
            rendering_data_camera = self.camera_view_geometry.to_rendering_data(
                data_batch.camera,
                cache_sensor_params=True,
                skip_calib=skip_camera_calib,
                global_step_for_prober=global_step_for_prober,
                enable_torch_compile=self.enable_torch_compile,
            )

        rendering_data_lidar: Optional[RenderingData] = None
        if data_batch.lidar is not None:
            assert self.lidar_view_geometry is not None
            skip_lidar_calib = skip_calib or (
                self.config.skip_first_pose_delta and data_batch.lidar.meta[0].unique_frame_idx == 0
            )
            rendering_data_lidar = self.lidar_view_geometry.to_rendering_data(
                data_batch.lidar,
                cache_sensor_params=True,
                skip_calib=skip_lidar_calib,
                global_step_for_prober=global_step_for_prober,
                enable_torch_compile=self.enable_torch_compile,
            )

        return RenderingBatch(camera=rendering_data_camera, lidar=rendering_data_lidar)

    @staticmethod
    def from_rig_trajectories(
        rig_trajectories: RigTrajectories,
        config: CalibConfig,
        trainer_config: TrainerConfig,
    ) -> FreePoseCalib:
        """Constructs a FreePoseCalib from rig-trajectories"""
        return FreePoseCalib(
            config=config,
            trainer_config=trainer_config,
            lidar_view_geometry=LidarFreePoseViewGeometry.from_rig_trajectories(
                rig_trajectories, enable_calib=config.lidar.enabled
            ),
            camera_view_geometry=CameraFreePoseViewGeometry.from_rig_trajectories(
                rig_trajectories, enable_calib=config.camera.enabled
            ),
        )


class RigCalib(BaseCalib):
    """Composite rig-trajectory spline and static per-camera extrinsic calibration."""

    def __init__(
        self,
        config: CalibConfig,
        trainer_config: TrainerConfig,
        rig_trajectories: RigTrajectories,
        camera_view_geometry: CameraFreePoseViewGeometry,
        lidar_view_geometry: LidarFreePoseViewGeometry,
    ) -> None:
        super().__init__(config, trainer_config)
        if config.rig_trajectory is None or config.camera_extrinsics is None:
            raise ValueError("rig-calib requires rig_trajectory and camera_extrinsics sections")

        self.camera_view_geometry = camera_view_geometry
        self.lidar_view_geometry = lidar_view_geometry
        self.enable_torch_compile = config.enable_torch_compile
        self.nurec_scale = rig_trajectories.world_to_nurec.target_scale

        if config.camera.enabled and not rig_trajectories.camera_calibrations:
            raise ValueError("Camera calibration is enabled but the dataset has no cameras")
        if config.lidar.enabled and not rig_trajectories.lidar_calibrations:
            raise ValueError("Lidar calibration is enabled but the dataset has no lidars")

        (
            camera_trajectory_indices,
            camera_sensor_to_rig,
            self._camera_sensor_id_to_index,
        ) = self._make_sensor_lookups(rig_trajectories, sensor_type="camera")
        (
            lidar_trajectory_indices,
            lidar_sensor_to_rig,
            self._lidar_sensor_id_to_index,
        ) = self._make_sensor_lookups(rig_trajectories, sensor_type="lidar")
        self.camera_trajectory_indices = nn.Buffer(camera_trajectory_indices, persistent=False)
        self.camera_sensor_to_rig = nn.Buffer(camera_sensor_to_rig, persistent=False)
        self.lidar_trajectory_indices = nn.Buffer(lidar_trajectory_indices, persistent=False)
        self.lidar_sensor_to_rig = nn.Buffer(lidar_sensor_to_rig, persistent=False)

        self.rig_spline: PackedRigSpline | None
        if config.rig_trajectory.enabled:
            trajectory_starts, trajectory_ends = self._trajectory_timestamp_bounds(rig_trajectories, config)
            self.rig_spline = PackedRigSpline(
                trajectory_starts,
                trajectory_ends,
                degree=config.rig_trajectory.spline_degree,
                knot_spacing_seconds=config.rig_trajectory.knot_spacing_seconds,
                skip_first_control=config.skip_first_pose_delta,
                max_translation_m=config.rig_trajectory.max_translation_m,
                max_rotation_rad=math.radians(config.rig_trajectory.max_rotation_degrees),
            )
            if config.rig_trajectory.active_frames_file:
                mask = self._active_control_mask(rig_trajectories, config.rig_trajectory.active_frames_file)
                self.rig_spline.set_active_controls(mask)
                log.info(
                    "rig spline: %d of %d controls active (frames in %s); the rest stay as loaded",
                    int(mask.sum().item()),
                    mask.numel(),
                    config.rig_trajectory.active_frames_file,
                )
        else:
            self.rig_spline = None

        self.camera_extrinsics: CameraExtrinsicResiduals | None
        self._camera_ids_by_extrinsic_index: list[str] = []
        camera_extrinsic_indices = torch.full_like(camera_trajectory_indices, -1)
        if config.camera_extrinsics.enabled:
            extrinsic_trajectory_indices: list[int] = []
            for residual_idx, sensor_id in enumerate(rig_trajectories.camera_calibrations):
                lookup_index = self._camera_sensor_id_to_index[sensor_id]
                camera_extrinsic_indices[lookup_index] = residual_idx
                extrinsic_trajectory_indices.append(int(camera_trajectory_indices[lookup_index].item()))
                self._camera_ids_by_extrinsic_index.append(sensor_id)
            self.camera_extrinsics = CameraExtrinsicResiduals(
                torch.tensor(extrinsic_trajectory_indices, dtype=torch.int64),
                num_trajectories=len(rig_trajectories.rig_trajectories),
                max_translation_m=config.camera_extrinsics.max_translation_m,
                max_rotation_rad=math.radians(config.camera_extrinsics.max_rotation_degrees),
            )
            if config.camera_extrinsics.freeze:  # keep the loaded residuals (e.g. the coarse model's in every cell)
                self.camera_extrinsics.residuals.register_hook(torch.zeros_like)
                log.info("camera extrinsic residuals frozen at their loaded values")
        else:
            self.camera_extrinsics = None
        self.camera_extrinsic_indices = nn.Buffer(camera_extrinsic_indices, persistent=False)

        log.info(
            "RigCalib: %d trajectories, rig spline %s, camera extrinsics %s",
            len(rig_trajectories.rig_trajectories),
            "ON" if self.rig_spline is not None else "OFF",
            "ON" if self.camera_extrinsics is not None else "OFF",
        )

    @staticmethod
    def _make_sensor_lookups(
        rig_trajectories: RigTrajectories, *, sensor_type: str
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
        calibrations = (
            rig_trajectories.camera_calibrations if sensor_type == "camera" else rig_trajectories.lidar_calibrations
        )
        max_index = max(
            (
                calibration.unique_sensor_idx if calibration.unique_sensor_idx >= 0 else 0
                for calibration in calibrations.values()
            ),
            default=-1,
        )
        trajectory_indices = torch.full((max_index + 1,), -1, dtype=torch.int64)
        extrinsics = torch.eye(4, dtype=torch.float32).repeat(max_index + 1, 1, 1)
        sensor_id_to_index: dict[str, int] = {}
        for sensor_id, calibration in calibrations.items():
            lookup_index = calibration.unique_sensor_idx if calibration.unique_sensor_idx >= 0 else 0
            timestamps_attr = "cameras_frame_timestamps_us" if sensor_type == "camera" else "lidars_frame_timestamps_us"
            candidates = [
                trajectory_idx
                for trajectory_idx, trajectory in enumerate(rig_trajectories.rig_trajectories)
                if sensor_id in getattr(trajectory, timestamps_attr)
            ]
            if not candidates:
                candidates = [
                    trajectory_idx
                    for trajectory_idx, trajectory in enumerate(rig_trajectories.rig_trajectories)
                    if trajectory.sequence_id == calibration.sequence_id
                ]
            if len(candidates) != 1:
                raise ValueError(f"Expected exactly one trajectory for {sensor_type} {sensor_id}")
            trajectory_indices[lookup_index] = candidates[0]
            extrinsics[lookup_index] = calibration.T_sensor_rig
            sensor_id_to_index[sensor_id] = lookup_index
        return trajectory_indices, extrinsics, sensor_id_to_index

    def _active_control_mask(self, rig_trajectories: RigTrajectories, path: str) -> torch.Tensor:
        """Controls supporting the listed frames: JSON {camera_id: [frame index, ...]} or a list for every camera."""
        assert self.rig_spline is not None
        with open(path) as f:
            listed = json.load(f)
        mask = torch.zeros(self.rig_spline.controls.shape[0], dtype=torch.bool)
        matched = 0
        for trajectory_idx, trajectory in enumerate(rig_trajectories.rig_trajectories):
            for camera_id, timestamps in trajectory.cameras_frame_timestamps_us.items():
                frames = listed if isinstance(listed, list) else listed.get(camera_id, [])
                if not frames:
                    continue
                index = torch.as_tensor(sorted(set(int(g) for g in frames)), dtype=torch.int64)
                index = index[(index >= 0) & (index < timestamps.shape[0])]
                matched += int(index.numel())
                mask |= self.rig_spline.control_mask_for(timestamps[index], trajectory_idx).cpu()
        if matched == 0:
            raise ValueError(f"active_frames_file {path}: no listed frame matches a camera of the dataset")
        return mask

    @staticmethod
    def _trajectory_timestamp_bounds(
        rig_trajectories: RigTrajectories, config: CalibConfig
    ) -> tuple[torch.Tensor, torch.Tensor]:
        trajectory_starts: list[int] = []
        trajectory_ends: list[int] = []
        for trajectory in rig_trajectories.rig_trajectories:
            frame_timestamps: list[torch.Tensor] = []
            if config.camera.enabled:
                frame_timestamps.extend(trajectory.cameras_frame_timestamps_us.values())
            if config.lidar.enabled:
                frame_timestamps.extend(trajectory.lidars_frame_timestamps_us.values())
            if frame_timestamps:
                trajectory_starts.append(min(int(timestamps.min().item()) for timestamps in frame_timestamps))
                trajectory_ends.append(max(int(timestamps.max().item()) for timestamps in frame_timestamps))
            else:
                trajectory_starts.append(int(trajectory.T_rig_world_timestamps_us.min().item()))
                trajectory_ends.append(int(trajectory.T_rig_world_timestamps_us.max().item()))
        return torch.tensor(trajectory_starts, dtype=torch.int64), torch.tensor(trajectory_ends, dtype=torch.int64)

    @property
    def params(self) -> List[nn.Parameter]:
        parameters: list[nn.Parameter] = []
        if self.rig_spline is not None:
            parameters.append(self.rig_spline.controls)
        if self.camera_extrinsics is not None:
            parameters.append(self.camera_extrinsics.residuals)
        return parameters

    def _zero_parameter_dependency(self) -> torch.Tensor:
        zero = self.params[0].sum() * 0.0
        for parameter in self.params[1:]:
            zero = zero + parameter.sum() * 0.0
        return zero

    def _correct_sensor_poses(
        self,
        base_poses: torch.Tensor,
        timestamps_us: torch.Tensor,
        lookup_index: int,
        *,
        sensor_type: str,
        apply_rig_trajectory: bool,
        apply_camera_extrinsics: bool,
    ) -> torch.Tensor:
        trajectory_indices = (
            self.camera_trajectory_indices if sensor_type == "camera" else self.lidar_trajectory_indices
        )
        sensor_to_rig_all = self.camera_sensor_to_rig if sensor_type == "camera" else self.lidar_sensor_to_rig
        if lookup_index >= trajectory_indices.numel() or int(trajectory_indices[lookup_index].item()) < 0:
            raise IndexError(f"No {sensor_type} trajectory mapping for sensor index {lookup_index}")
        trajectory_idx = int(trajectory_indices[lookup_index].item())
        sensor_to_rig = sensor_to_rig_all[lookup_index]

        correction = torch.eye(4, dtype=base_poses.dtype, device=base_poses.device)
        if apply_rig_trajectory:
            if self.rig_spline is None:
                raise RuntimeError("Rig trajectory correction requested while the component is disabled")
            correction = (
                torch.linalg.inv(sensor_to_rig) @ self.rig_spline(timestamps_us, trajectory_idx) @ sensor_to_rig
            )
        if apply_camera_extrinsics:
            if sensor_type != "camera":
                raise ValueError("Camera extrinsic corrections cannot be applied to lidar")
            if self.camera_extrinsics is None:
                raise RuntimeError("Camera extrinsic correction requested while the component is disabled")
            if lookup_index >= self.camera_extrinsic_indices.numel():
                raise IndexError(f"No camera extrinsic mapping for sensor index {lookup_index}")
            residual_idx = int(self.camera_extrinsic_indices[lookup_index].item())
            if residual_idx < 0:
                raise IndexError(f"No camera extrinsic mapping for sensor index {lookup_index}")
            correction = correction @ self.camera_extrinsics(residual_idx)

        correction_nurec = correction.clone()
        correction_nurec[..., :3, 3] = correction[..., :3, 3] * self.nurec_scale
        return base_poses @ correction_nurec

    def _pose_correction(self, sensor_type: str, skip_calib: bool) -> PoseCorrection:
        sensor_enabled = getattr(self.config, sensor_type).enabled
        rig_trajectory = self.config.rig_trajectory
        camera_extrinsics = self.config.camera_extrinsics
        assert rig_trajectory is not None and camera_extrinsics is not None

        def apply(base_poses: torch.Tensor, timestamps_us: torch.Tensor, meta: FrameMeta) -> torch.Tensor:
            if skip_calib:
                return base_poses + self._zero_parameter_dependency()
            if not sensor_enabled:
                return base_poses
            lookup_index = meta.unique_sensor_idx if meta.unique_sensor_idx >= 0 else 0
            return self._correct_sensor_poses(
                base_poses,
                timestamps_us,
                lookup_index,
                sensor_type=sensor_type,
                apply_rig_trajectory=rig_trajectory.enabled,
                apply_camera_extrinsics=sensor_type == "camera" and camera_extrinsics.enabled,
            )

        # Both early returns above leave the poses numerically unchanged, so only the final branch
        # moves the sensor. Consumers holding rays precomputed from the uncalibrated poses (e.g.
        # `use_real_lidar_rays`) rely on this flag to reject the combination they cannot honour.
        return PoseCorrection(apply=apply, modifies_poses=sensor_enabled and not skip_calib)

    def forward(
        self, data_batch: DataBatch, skip_calib: bool = False, global_step_for_prober: Optional[int] = None
    ) -> RenderingBatch:
        rendering_data_camera: Optional[RenderingData] = None
        if data_batch.camera is not None:
            rendering_data_camera = self.camera_view_geometry.to_rendering_data(
                data_batch.camera,
                cache_sensor_params=True,
                skip_calib=True,
                global_step_for_prober=global_step_for_prober,
                enable_torch_compile=self.enable_torch_compile,
                pose_correction=self._pose_correction("camera", skip_calib),
            )

        rendering_data_lidar: Optional[RenderingData] = None
        if data_batch.lidar is not None:
            rendering_data_lidar = self.lidar_view_geometry.to_rendering_data(
                data_batch.lidar,
                cache_sensor_params=True,
                skip_calib=True,
                global_step_for_prober=global_step_for_prober,
                enable_torch_compile=self.enable_torch_compile,
                pose_correction=self._pose_correction("lidar", skip_calib),
            )
        return RenderingBatch(camera=rendering_data_camera, lidar=rendering_data_lidar)

    def regularization_loss(self) -> tuple[str, torch.Tensor, float] | None:
        rig_trajectory = self.config.rig_trajectory
        assert rig_trajectory is not None
        if self.rig_spline is None:
            return None
        if rig_trajectory.anchor_weight > 0.0:  # mcity multiclip: curvature + anchor to the loaded trajectory
            value = rig_trajectory.anchor_weight * self.rig_spline.anchor_loss(
                translation_normalization_m=rig_trajectory.anchor_translation_m,
                rotation_normalization_rad=math.radians(rig_trajectory.anchor_rotation_degrees),
            )
            if rig_trajectory.curvature_l1_weight > 0.0:
                value = value + rig_trajectory.curvature_l1_weight * self.rig_spline.group_l1_curvature_loss(
                    translation_normalization_m_s2=rig_trajectory.curvature_translation_normalization_m_s2,
                    rotation_normalization_rad_s2=math.radians(
                        rig_trajectory.curvature_rotation_normalization_degrees_s2
                    ),
                )
            return "rig_spline_curvature_anchor", value, 1.0
        if rig_trajectory.curvature_l1_weight <= 0.0:
            return None
        value = self.rig_spline.group_l1_curvature_loss(
            translation_normalization_m_s2=rig_trajectory.curvature_translation_normalization_m_s2,
            rotation_normalization_rad_s2=math.radians(rig_trajectory.curvature_rotation_normalization_degrees_s2),
        )
        return "rig_spline_curvature", value, rig_trajectory.curvature_l1_weight

    def update_step_train_batch_start(self, epoch: int, global_step: int, system, **kwargs) -> dict[str, torch.Tensor]:
        ret = super().update_step_train_batch_start(epoch, global_step, system, **kwargs)
        if self.rig_spline is not None:
            for name, value in self.rig_spline.statistics().items():
                system.log(f"calib/rig_spline_{name}", value, prog_bar=False)
        if self.camera_extrinsics is not None:
            for name, value in self.camera_extrinsics.statistics().items():
                if name.startswith("camera_"):
                    camera_idx_text, suffix = name.removeprefix("camera_").split("_", maxsplit=1)
                    sensor_id = self._camera_ids_by_extrinsic_index[int(camera_idx_text)]
                    name = f"{sensor_id.replace('@', '_')}_{suffix}"
                system.log(f"calib/camera_extrinsics_{name}", value, prog_bar=False)
        return ret

    @torch.no_grad()
    def apply_to_rig_trajectories(self, rig_trajectories: RigTrajectories) -> RigTrajectories:
        """Bake the learned local rig and camera corrections into a detached CPU copy."""
        calibrated = copy.deepcopy(rig_trajectories)
        calibrated.T_world_base = rig_trajectories.T_world_base.detach().cpu().clone()

        for trajectory_idx, (source_trajectory, calibrated_trajectory) in enumerate(
            zip(rig_trajectories.rig_trajectories, calibrated.rig_trajectories)
        ):
            calibrated_trajectory.T_rig_world_timestamps_us = (
                source_trajectory.T_rig_world_timestamps_us.detach().cpu().clone()
            )
            calibrated_trajectory.cameras_frame_timestamps_us = {
                sensor_id: timestamps.detach().cpu().clone()
                for sensor_id, timestamps in source_trajectory.cameras_frame_timestamps_us.items()
            }
            calibrated_trajectory.lidars_frame_timestamps_us = {
                sensor_id: timestamps.detach().cpu().clone()
                for sensor_id, timestamps in source_trajectory.lidars_frame_timestamps_us.items()
            }

            coarse_poses = source_trajectory.T_rig_worlds
            if self.rig_spline is not None:
                coarse_correction = self.rig_spline(
                    source_trajectory.T_rig_world_timestamps_us.to(self.rig_spline.controls.device),
                    trajectory_idx,
                ).to(device=coarse_poses.device, dtype=coarse_poses.dtype)
                coarse_poses = coarse_poses @ coarse_correction
            calibrated_trajectory.T_rig_worlds = coarse_poses.detach().cpu().clone()

            if source_trajectory.cameras_frame_T_rig_worlds is not None:
                calibrated_trajectory.cameras_frame_T_rig_worlds = {}
                for sensor_id, frame_poses in source_trajectory.cameras_frame_T_rig_worlds.items():
                    corrected_frame_poses = frame_poses
                    if self.rig_spline is not None:
                        frame_correction = self.rig_spline(
                            source_trajectory.cameras_frame_timestamps_us[sensor_id].to(
                                self.rig_spline.controls.device
                            ),
                            trajectory_idx,
                        ).to(device=frame_poses.device, dtype=frame_poses.dtype)
                        corrected_frame_poses = frame_poses @ frame_correction
                    calibrated_trajectory.cameras_frame_T_rig_worlds[sensor_id] = (
                        corrected_frame_poses.detach().cpu().clone()
                    )

        for sensor_id, source_camera_calibration in rig_trajectories.camera_calibrations.items():
            corrected_extrinsic = source_camera_calibration.T_sensor_rig
            if self.camera_extrinsics is not None:
                lookup_index = self._camera_sensor_id_to_index[sensor_id]
                residual_idx = int(self.camera_extrinsic_indices[lookup_index].item())
                residual = self.camera_extrinsics(residual_idx).to(
                    device=corrected_extrinsic.device, dtype=corrected_extrinsic.dtype
                )
                corrected_extrinsic = corrected_extrinsic @ residual
            calibrated.camera_calibrations[sensor_id].T_sensor_rig = corrected_extrinsic.detach().cpu().clone()

        for sensor_id, source_lidar_calibration in rig_trajectories.lidar_calibrations.items():
            calibrated.lidar_calibrations[sensor_id].T_sensor_rig = (
                source_lidar_calibration.T_sensor_rig.detach().cpu().clone()
            )
        return calibrated

    @staticmethod
    def from_rig_trajectories(
        rig_trajectories: RigTrajectories,
        config: CalibConfig,
        trainer_config: TrainerConfig,
    ) -> "RigCalib":
        return RigCalib(
            config=config,
            trainer_config=trainer_config,
            rig_trajectories=rig_trajectories,
            camera_view_geometry=CameraFreePoseViewGeometry.from_rig_trajectories(rig_trajectories, enable_calib=False),
            lidar_view_geometry=LidarFreePoseViewGeometry.from_rig_trajectories(rig_trajectories, enable_calib=False),
        )
