# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Type

import numpy as np
import point_cloud_utils as pcu
import torch

from numpy.typing import NDArray
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch import nn

import ncore.data

from nurec.config.model import SHGaussiansLayerConfig
from nurec.config.trainer import TrainerConfig
from nurec.datasets.ncore import NCOREDataSource
from nurec.datasets.summary import DataSourceSummary
from nurec.datasets.tracks import CuboidTracks
from nurec.datasets.utils import compute_point_cloud_inside_tracks_mask
from nurec.models.base import BaseModel
from nurec.models.gaussians.construction_barriers import BarrierCuboids
from nurec.models.gaussians.utils import (
    RGB2SH,
    PLYGaussianLoader,
    sh_degree_to_specular_dim,
    subsample_points,
    track_random_initialization,
    uniform_sample_sphere,
)
from nurec.models.utils import get_activation
from nurec.utils.geometry import (
    se3_matrix_inverse,
    so3_matrix_to_quat,
    vector_align_rotation,
)
from nurec.utils.io.mesh import Mesh
from nurec.utils.knn import knn_points
from nurec.utils.lidar_object_init_points import build_lidar_cone_init_points
from nurec.utils.misc import sync_objects_or_raise, to_numpy, to_torch, unpack_optional
from nurec.utils.profiling import ScopedTimer
from nurec.utils.types import FrameConversion, PointCloud
from nurec.visualdebugger import get_visualdebugger


if TYPE_CHECKING:
    from apps.av.road_mesh.ground_mesh import GroundMeshReconstructionSettings


log = logging.getLogger(__name__)


def _resolve_num_workers(value: int) -> int:
    """Resolve the ground-mesh road-segmentation worker count.

    ``-1`` (default): pick a sensible value automatically — uses up to
    ``os.cpu_count()`` workers, capped at 32 to avoid OS pressure and IPC
    contention on large nodes.

    ``0`` or ``1``: keep the legacy serial path (single-threaded).

    ``N > 1``: use exactly N workers.
    """
    if value == -1:
        cpu_count = os.cpu_count() or 1
        return min(cpu_count, 32)
    if value <= 1:
        return 1
    return value


class BaseInitialization(BaseModel, ABC):
    # no gaussians here to avoid circle import
    GAUSSIANS_INIT_VARIANTS: dict[str, Type[BaseInitialization]] = {}

    @staticmethod
    def register_to_gaussian_initialization_factory(name: str, cls: Type[BaseInitialization]) -> None:
        if name in BaseInitialization.GAUSSIANS_INIT_VARIANTS:
            raise KeyError(f"{name=} already in GAUSSIANS_INIT_VARIANTS.")
        BaseInitialization.GAUSSIANS_INIT_VARIANTS[name] = cls

    @staticmethod
    def factory(
        name: str,
        config: DictConfig,
        gaussian_config: SHGaussiansLayerConfig,
        trainer_config: TrainerConfig,
        labels_to_ignore: list[str],
        labels_to_use: list[str],
    ) -> BaseInitialization:
        return BaseInitialization.GAUSSIANS_INIT_VARIANTS[name](
            config, gaussian_config, trainer_config, labels_to_ignore, labels_to_use
        )

    def __init__(
        self,
        config: DictConfig,
        gaussian_config: SHGaussiansLayerConfig,
        trainer_config: TrainerConfig,
        labels_to_ignore: list[str],
        labels_to_use: list[str],
    ) -> None:
        super().__init__(config=config)

        self.gaussian_config = gaussian_config
        self.trainer_config = trainer_config
        self.labels_to_ignore = labels_to_ignore
        self.labels_to_use = labels_to_use
        self.gaussian_cuboid_ids: List[torch.Tensor] = []
        self.ground_mesh_artifact: Mesh | None = None
        self.ground_mesh_artifact_settings: GroundMeshReconstructionSettings | None = None

        assert not (self.labels_to_ignore and self.labels_to_use), (
            f"[{self.__class__.__name__}] Only one of labels_to_ignore and labels_to_use should be specified"
        )

        self.debug_viz = self.gaussian_config.debug_viz
        if self.debug_viz:
            visualizer = get_visualdebugger()
            visualizer.set_properties(up="z_up", front="neg_x_front", navigation_style="free")

        self.debug_dir: Optional[str] = None
        if self.config.local_debug_dir is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.debug_dir = os.path.expanduser(os.path.join(self.config.local_debug_dir, timestamp))
            # Create all parent directories as needed
            os.makedirs(self.debug_dir, exist_ok=True)

    def initialize_from_datasource(self, datasource: DataSourceSummary, **kwargs) -> None:
        raise NotImplementedError

    def init_from_positions(
        self,
        positions: torch.Tensor,
        normals: Optional[torch.Tensor] = None,
        observation_scale: Optional[torch.Tensor] = None,
    ) -> None:
        N = positions.shape[0]
        dtype = torch.float32
        density_activation_inv = get_activation(self.gaussian_config.density_activation, inverse=True)
        scale_activation_inv = get_activation(self.gaussian_config.scale_activation, inverse=True)

        # identity rotations
        rots = torch.zeros((N, 4), dtype=dtype, device=self.device)
        rots[:, 0] = 1.0  # they're quaternions stored in wxyz format

        if observation_scale is None:
            # Initialize isotropic scales if self.config.default_scale is a float, otherwise it should be a three element list
            assert isinstance(self.config.default_scale, float) or (
                isinstance(self.config.default_scale, ListConfig) and len(self.config.default_scale) == 3
            ), f"[{self.__class__.__name__}] default_scale must be a float or a 3-element list"

            observation_scale = (
                torch.full((N, 3), self.config.default_scale)
                if isinstance(self.config.default_scale, float)
                else torch.tensor(self.config.default_scale, device=self.device, dtype=torch.float32)
                .unsqueeze(0)
                .expand(positions.shape[0], -1)
            )

        scales = scale_activation_inv(observation_scale * self.config.scale_multiplier)

        # Repeat the scale in case only an isotropic scale was provided
        if scales.shape[1] == 1:
            scales = scales.repeat(1, 3)

        if normals is not None:
            mat = vector_align_rotation(
                torch.tensor([0, 0, 1], dtype=normals.dtype, device=self.device).repeat(normals.shape[0], 1),
                normals,
            )
            rots = so3_matrix_to_quat(mat)  # xyzw format
            rots = rots[:, [3, 0, 1, 2]]  # wxyz format
            if observation_scale.shape[1] == 1 or isinstance(self.config.default_scale, float):
                scales[..., 2] = scale_activation_inv(observation_scale * self.config.scale_multiplier / 10.0).squeeze(
                    -1
                )[..., 2]  # compress z-scale to bias Gaussians toward flatter geometry when using normals
            else:
                scales[..., 2] = scale_activation_inv(observation_scale * self.config.scale_multiplier).squeeze(-1)[
                    ..., 2
                ]  # use default scale if z-scale is provided

        # set density as a constant
        opacities = density_activation_inv(
            torch.full((N, 1), fill_value=self.config.default_density, dtype=dtype, device=self.device)
        )

        self.positions = nn.Parameter(positions.to(dtype=dtype, device=self.device))
        self.rotations = nn.Parameter(rots.to(dtype=dtype, device=self.device))
        self.scales = nn.Parameter(scales.to(dtype=dtype, device=self.device))
        self.densities = nn.Parameter(opacities.to(dtype=dtype, device=self.device))

    def init_additional_buffers(self, point_cloud: PointCloud) -> None:
        if self.gaussian_config.name in [
            "sh-gaussians",
            "rigid-gaussians",
            "deformable-gaussians",
            "interpolated-table-deformable-gaussians",
        ]:
            self.init_additional_buffers_SH(point_cloud)
        if self.debug_dir is not None:
            pcu.save_mesh_vnc(
                os.path.join(self.debug_dir, f"{self.gaussian_config.name}.ply"),
                to_numpy(point_cloud.xyz_end),
                to_numpy(point_cloud.normal) if point_cloud.normal is not None else None,
                to_numpy(point_cloud.color) if point_cloud.color is not None else None,
            )

    def init_additional_buffers_SH(self, point_cloud: PointCloud) -> None:
        dtype = torch.float32
        N = self.get_num_gaussians()

        # Initialize colors, constant if they weren't given
        if point_cloud.color is None:
            features_albedo = torch.rand((N, 3), dtype=dtype, device=self.device) / 255.0
        else:
            num_random_points = N - point_cloud.color.shape[0]

            features_albedo = torch.cat(
                [
                    to_torch(RGB2SH(to_numpy(point_cloud.color.float() / 255.0)), dtype=dtype, device=self.device),
                    # Check this part for random init
                    torch.rand((num_random_points, 3), dtype=dtype, device=self.device) / 255.0,
                ]
            )

        num_specular_dims = sh_degree_to_specular_dim(self.gaussian_config.progressive_training.max_n_features)
        features_specular = torch.zeros(N, num_specular_dims, device=self.device, dtype=dtype)
        self.features_albedo = nn.Parameter(features_albedo)
        self.features_specular = nn.Parameter(features_specular)

    def get_num_gaussians(self) -> int:
        return self.positions.shape[0]

    def filter_point_cloud_by_labels(self, point_cloud: PointCloud, datasource: DataSourceSummary) -> PointCloud:
        """
        Filter point cloud based on labels_to_use or labels_to_ignore.
        If neither is provided, return the point cloud as is.
        If both are provided, raise an error as this class requires only one of them to be provided.
        """

        if not (self.labels_to_use or self.labels_to_ignore):
            # No labels to filter by, return the point cloud as is
            return point_cloud
        else:
            assert bool(self.labels_to_ignore) ^ bool(self.labels_to_use), (
                f"[{self.__class__.__name__}] Only one of labels_to_ignore and labels_to_use should be specified"
            )

            assert point_cloud.semantic_class_id is not None, (
                f"{self.__class__.__name__}: Semantic class id is not present in the point cloud"
            )

            assert isinstance(datasource.datasource, NCOREDataSource), (
                f"[{self.__class__.__name__}] Only NCOREDataSource is supported"
            )

            sensor_type = unpack_optional(point_cloud.sensor_type)
            stuff_classes = unpack_optional(
                datasource.datasource.get_semantic_classes_map(
                    camera_semantics="camera" in sensor_type, lidar_semantics="lidar" in sensor_type
                )
            )

            if self.labels_to_use:
                keep_idx = torch.zeros_like(point_cloud.semantic_class_id, dtype=torch.bool)
                for label in self.labels_to_use:
                    label_idx = stuff_classes[label]
                    keep_idx = torch.logical_or(keep_idx, point_cloud.semantic_class_id == label_idx)
            else:
                keep_idx = torch.ones_like(point_cloud.semantic_class_id, dtype=torch.bool)
                for label in self.labels_to_ignore:
                    label_idx = stuff_classes[label]
                    keep_idx = torch.logical_and(keep_idx, point_cloud.semantic_class_id != label_idx)

            return point_cloud[keep_idx]


class RandomInitialization(BaseInitialization):
    def initialize_from_datasource(self, datasource: DataSourceSummary, **kwargs) -> None:
        """
        Initialize points randomly within a bounding box
        """

        # We create random points inside the bounds of the AABB
        aabb = datasource.get_aabb().to(self.device)

        positions = (
            torch.rand((self.config.num_points, 3), dtype=torch.float32, device=self.device) * aabb.get_extent()
            + aabb.blb
        )

        # Just initialize the points here, random colors will be initialzied in init_additional_buffers
        point_cloud = PointCloud(xyz_start=positions, xyz_end=positions)
        dist = (
            torch.sqrt(knn_points(positions.unsqueeze(0), positions.unsqueeze(0), K=2).dists[0, :, 1]).clamp_min(1e-3)
        ).unsqueeze(1)

        self.init_from_positions(positions=positions, observation_scale=dist)
        self.init_additional_buffers(point_cloud)


class SFMPointCloudInitialization(BaseInitialization):
    def initialize_from_datasource(self, datasource: DataSourceSummary, **kwargs) -> None:
        """
        Initialize points from SFM point cloud
        """
        pcs = [pc for pc in unpack_optional(datasource.get_point_clouds(device=self.device, color_type="generic-data"))]
        if not pcs:
            raise RuntimeError("sfm-point-cloud initialization received no points (color_type='generic-data').")
        point_cloud = PointCloud.collate_fn(pcs, device=self.device)
        # TODO(tylerz): shall we support the multiple camera sensors?
        #     (Qi): we should support multiple camera sensors in the future.
        assert isinstance(datasource.datasource, NCOREDataSource), (
            "Only NCOREDataSource is supported for sfm-point-cloud initialization"
        )
        assert len(datasource.datasource.camera_sensors) == 1, (
            "Only one camera sensor is currently supported for sfm-point-cloud initialization"
        )
        camera_sensor = list(datasource.datasource.camera_sensors.values())[0]
        camera_centers: list[np.ndarray] = []
        for idx in range(camera_sensor.frames_count):
            T_sensor_world_end = camera_sensor.get_frames_T_sensor_target(
                target_node="world",
                frame_indices=idx,
                frame_timepoint=ncore.data.FrameTimepoint.END,
            )  # 4x4
            camera_centers.append(T_sensor_world_end[:3, 3])
        camera_positions = torch.tensor(np.stack(camera_centers), device=self.device)
        dist = (
            torch.sqrt(knn_points(point_cloud.xyz_end.unsqueeze(0), camera_positions.unsqueeze(0)).dists)
            .clamp_min(1e-7)
            .squeeze(0)
        )
        scale = dist * self.config.scale_factor
        self.init_from_positions(positions=point_cloud.xyz_end, observation_scale=scale)
        self.init_additional_buffers(point_cloud)


class LidarRigTrajInitializationBase(BaseInitialization):
    @ScopedTimer("LidarRigTrajInitializationBase/get_point_clouds_and_ego_poses")
    def get_point_clouds_and_ego_poses(self, datasource: DataSourceSummary) -> tuple[PointCloud, torch.Tensor, float]:
        """
        Returns points and ego poses from the datasource, optionally filtering by class id.
        Subsamples the point cloud to the maximum size defined by `num_point_cloud_points` and returns a sparsity compensation factor
        that can be used to increase the scale of the remaining points.
        """

        assert isinstance(datasource.datasource, NCOREDataSource), (
            f"[{self.__class__.__name__}] Only NCOREDataSource is supported for from_lidar_rigtraj initialization"
        )

        # Which sensors to seed from. Only an unset (null) list is filled in, an explicitly empty
        # one means "seed nothing from this modality" and is kept as is.
        # Use train_lidar_ids to avoid a validation-only sensor being supervised here.
        # In case of a lidar-free rig, seeds from cameras instead.
        lidar_ids = self.config.lidar_ids
        if lidar_ids is None:
            lidar_ids = list(datasource.datasource.train_lidar_ids) or None
        camera_ids = self.config.camera_ids
        if lidar_ids is None and camera_ids is None:
            camera_ids = list(datasource.datasource.train_camera_ids)
            log.info(f"[{self.__class__.__name__}] No lidar available, seeding from cameras: {camera_ids}")

        # Function to read point clouds from the datasource
        def maybe_read_point_cloud():
            pcs = [
                pc
                for pc in unpack_optional(
                    datasource.get_point_clouds(
                        device=self.device,
                        lidar_ids=lidar_ids,
                        camera_ids=camera_ids,
                        non_dynamic_points_only=self.config.non_dynamic_points_only,
                        color_type="camera-rgb",
                        step_frame=self.config.step_frame,
                    )
                )
            ]

            assert len(pcs) > 0, "No point clouds were found in the dataset"
            point_cloud = PointCloud.collate_fn(pcs, device=self.device)

            # Filter the point cloud based on labels_to_use or labels_to_ignore
            if bool(self.labels_to_ignore) or bool(self.labels_to_use):
                point_cloud = self.filter_point_cloud_by_labels(point_cloud, datasource)

            original_n_points = point_cloud.n_points

            # subsample point cloud to desired size
            if point_cloud.n_points > self.config.num_point_cloud_points:
                random_idxs = torch.from_numpy(
                    np.random.choice(point_cloud.n_points, self.config.num_point_cloud_points, replace=False)
                ).to(self.device)
                point_cloud = point_cloud[random_idxs]

            return point_cloud, original_n_points

        # Actually read the point cloud
        point_cloud, original_n_points = maybe_read_point_cloud()

        # We need to use the original number of points (before subsampling) to compute the sparsity compensation factor
        sparsity_compensation = max(1, original_n_points / self.config.num_point_cloud_points)
        log.info(
            f"[{self.__class__.__name__}] Original number of points: {original_n_points}, number of points: {point_cloud.n_points}, sparsity compensation: {sparsity_compensation}"
        )

        # Get ego positions in NuRec coordinates
        rig_trajectories = unpack_optional(datasource.get_rig_trajectories())
        T_world_to_nurec_np, S_world_to_nurec_np = rig_trajectories.world_to_nurec.get_transformation_matrices()
        T_world_to_nurec = torch.tensor(T_world_to_nurec_np, device=self.device, dtype=torch.float32)
        S_world_to_nurec = torch.tensor(S_world_to_nurec_np, device=self.device, dtype=torch.float32)
        # Ego poses that map from camera_to_nurec
        ego_c2nurec = torch.cat(
            [
                (T_world_to_nurec @ rig_traj.T_rig_worlds.float().to(self.device) @ S_world_to_nurec)
                for rig_traj in rig_trajectories.rig_trajectories
            ]
        )

        return point_cloud, ego_c2nurec, sparsity_compensation


class InitializationMixin(BaseInitialization):
    @abstractmethod
    def get_point_clouds_and_ego_poses(
        self, datasource: DataSourceSummary
    ) -> tuple[PointCloud, torch.Tensor, float]: ...

    def _initialize_from_datasource(self, datasource: DataSourceSummary, **kwargs) -> None:
        """
        Initialize from a combination of colored point clouds and random points.

        Random points are sampled from two strategies: near and far.
        Near points are sampled from a set of spheres of radius `near_radius` around each of the ego positions.
        Far points are sampled from a sphere of the size of the scene extent (centered around the ego positions in the middle of the sequence).

        Camera frusta are used to filter out invisible points.
        """
        assert isinstance(self, (LidarRigTrajInitialization, AccumulatedPointCloudInitialization)), (
            f"[{self.__class__.__name__}] This initialization is only supported for LidarRigTrajInitialization and AccumulatedPointCloudInitialization"
        )

        point_cloud, ego_c2nurec, sparsity_compensation = self.get_point_clouds_and_ego_poses(datasource)
        ego_positions = ego_c2nurec[:, :3, 3]

        scene_extent = datasource.get_aabb().get_extent().max().item()
        if isinstance(self, AccumulatedPointCloudInitialization):
            scene_extent, far_plane_depth = self._resolve_scene_extent_and_far_plane(scene_extent)
        else:
            far_plane_depth = self.config.far_radius_factor * scene_extent / 2

        assert isinstance(datasource.datasource, NCOREDataSource), (
            f"[{self.__class__.__name__}] Only NCOREDataSource is supported for this initialization"
        )

        # Get all camera frusta
        def maybe_read_camera_frusta():
            return [
                camera_frustum.to(self.device)
                for camera_id in datasource.datasource.camera_ids
                for camera_frustum, _ in unpack_optional(
                    datasource.get_camera_frusta(camera_id=camera_id, far_plane_depth=far_plane_depth)
                )
            ]

        camera_frusta = [camera_frustum.to(self.device) for camera_frustum in maybe_read_camera_frusta()]

        # sample random points
        random_pts_list: list[torch.Tensor] = []
        if self.debug_viz:
            visualizer = get_visualdebugger()
        else:
            visualizer = None

        # sample random points
        num_near_pts = self.config.num_near_points
        if num_near_pts > 0:  # distances uniformly from (0, 0.5 * scene_extent)
            near_pts = uniform_sample_sphere(num_near_pts, ego_positions.device) * scene_extent * 0.5
            if visualizer is not None:
                visualizer.add_point_cloud("random_near", near_pts.cpu().numpy(), radius=1e-3, point_render_mode="quad")
            random_pts_list.append(near_pts)

        num_far_pts = self.config.num_far_points
        if num_far_pts > 0:  # distances uniformly from (0, 10 * scene_extent)
            far_pts = (
                uniform_sample_sphere(
                    num_far_pts, ego_positions.device, inverse=True, radius=self.config.far_radius_factor
                )
                * scene_extent
                * 0.5
            )
            if visualizer is not None:
                visualizer.add_point_cloud("random_far", far_pts.cpu().numpy(), radius=1e-3, point_render_mode="quad")
            random_pts_list.append(far_pts)

        if num_near_pts > 0 or num_far_pts > 0:
            random_pts = torch.cat(random_pts_list, dim=0)

            # check visibility using camera frusta, one frustum at a time: stacking every frustum's mask needs
            # (frusta x points) memory, e.g. 91 GB for 30k camera frames
            visibility_mask = torch.zeros(random_pts.shape[0], dtype=torch.bool, device=random_pts.device)
            for camera_frustum in camera_frusta:
                visibility_mask |= camera_frustum.points_in_frustum(random_pts).reshape(-1) > 0

            random_pts = random_pts[visibility_mask]
            if visualizer is not None:
                visualizer.add_point_cloud(
                    "random_points", random_pts.cpu().numpy(), radius=1e-3, point_render_mode="quad"
                )
                visualizer.add_point_cloud(
                    "ego_positions", ego_positions.cpu().numpy(), radius=1e-3, point_render_mode="quad"
                )
                visualizer.add_point_cloud(
                    "point_cloud",
                    point_cloud.xyz_end.cpu().numpy(),
                    radius=1e-3,
                    point_render_mode="quad",
                    colors_quantities={"colored": unpack_optional(point_cloud.color).cpu().numpy()},
                    enabled=True,
                )
                visualizer.show()

            # positions
            positions = torch.cat([point_cloud.xyz_end.to(self.device), random_pts])

            # normals
            normals = None
            if point_cloud.normal is not None:
                normals = torch.cat(
                    [
                        point_cloud.normal.to(self.device),
                        torch.tensor([0, 0, 1], dtype=point_cloud.normal.dtype, device=self.device).repeat(
                            random_pts.shape[0], 1
                        ),
                    ]
                )

            # get scales based on distance to ego poses
            dist_to_ego_positions = (
                torch.sqrt(knn_points(random_pts.unsqueeze(0), ego_positions.to(positions).unsqueeze(0)).dists)
                .clamp_min(1e-7)
                .squeeze(0)
            )
            observation_scale = torch.cat(
                [
                    unpack_optional(point_cloud.camera_footprint_scale).to(self.device).unsqueeze(-1)
                    * np.sqrt(sparsity_compensation),
                    dist_to_ego_positions * self.config.observation_scale_factor,
                ]
            )
        else:
            positions = point_cloud.xyz_end.to(self.device)
            normals = point_cloud.normal.to(self.device) if point_cloud.normal is not None else None
            observation_scale = unpack_optional(point_cloud.camera_footprint_scale).to(self.device).unsqueeze(
                -1
            ) * np.sqrt(sparsity_compensation)

        if self.config.get("init_with_normals", False):
            self.init_from_positions(positions=positions, normals=normals, observation_scale=observation_scale)
        else:
            self.init_from_positions(positions=positions, observation_scale=observation_scale)
        self.init_additional_buffers(point_cloud)


class LidarRigTrajInitialization(LidarRigTrajInitializationBase, InitializationMixin):
    def initialize_from_datasource(self, datasource: DataSourceSummary, **kwargs) -> None:
        """
        Initialize from a combination of colored point clouds and random points.

        Random points are sampled from two strategies: near and far.
        Near points are sampled from a set of spheres of radius `near_radius` around each of the ego positions.
        Far points are sampled from a sphere of the size of the scene extent (centered around the ego positions in the middle of the sequence).

        Camera frusta are used to filter out invisible points.
        """
        self._initialize_from_datasource(datasource, **kwargs)


class RoadInitializationMixin(BaseInitialization):
    @abstractmethod
    def get_point_clouds_and_ego_poses(
        self, datasource: DataSourceSummary
    ) -> tuple[PointCloud, torch.Tensor, float]: ...

    def _project_to_ego_z_plane(
        self,
        points: torch.Tensor,
        ego_c2nurec: torch.Tensor,
        ego_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project points onto the local ego z-plane to encourage flat road geometry.

        For each point, finds the nearest ego pose, transforms the point into that ego's
        local coordinate frame, sets z to ``config.z_offset``, then transforms back to
        NuRec world space.

        Args:
            points: World-space positions to project. Shape ``[N, 3]``.
            ego_c2nurec: Ego-to-NuRec SE(3) transforms. Shape ``[M, 4, 4]``.
            ego_positions: Ego translation vectors (``ego_c2nurec[:, :3, 3]``). Shape ``[M, 3]``.

        Returns:
            A tuple of:
                - projected_points: Projected world-space positions. Shape ``[N, 3]``.
                - nearest_ego_indices: Index of the nearest ego pose per point. Shape ``[N]``.
        """
        nearest_ego_indices = (points[..., None, :] - ego_positions).norm(dim=-1).argmin(dim=-1)
        ego_c2nurec_per_point = ego_c2nurec[nearest_ego_indices]
        ego_nurec2c_per_point = se3_matrix_inverse(ego_c2nurec_per_point)

        local = (ego_nurec2c_per_point[:, :3, :3] @ points[..., :, None] + ego_nurec2c_per_point[:, :3, 3:4]).squeeze(
            -1
        )
        local[..., 2] = self.config.get("z_offset", 0)

        projected_points = (
            ego_c2nurec_per_point[:, :3, :3] @ local[..., :, None] + ego_c2nurec_per_point[:, :3, 3:4]
        ).squeeze(-1)
        return projected_points, nearest_ego_indices

    def _initialize_from_datasource(self, datasource: DataSourceSummary, **kwargs) -> None:
        """
        Initialize from colored point clouds and with HUGSIM's strategy which encourages a flatter road initialization
        by setting the height of each point to that of the closest ego position. This initialization is intended to be used
        in conjunction with a default_scale configuration that initializes the points to be flat disks
        (via a parameter such as [0.1, 0.1, 0.001])
        """

        assert isinstance(self, (LidarRigTrajRoadInitialization, AccumulatedPointCloudRoadInitialization)), (
            f"[{self.__class__.__name__}] This initialization is only supported for LidarRigTrajRoadInitialization and AccumulatedPointCloudRoadInitialization"
        )

        point_cloud, ego_c2nurec, sparsity_compensation = self.get_point_clouds_and_ego_poses(datasource)
        ego_positions = ego_c2nurec[:, :3, 3]  # [M, 3]
        positions = point_cloud.xyz_end.to(self.device)  # [N, 3]

        base_scale = (
            self.config.default_scale if isinstance(self.config.default_scale, float) else self.config.default_scale[0]
        )

        # Set the height of each point to that of the closest ego position (as in HUGSIM)
        # This encourages a flatter road initialization
        if self.config.project_to_z_offset:
            positions, pc_nearest_ego_indices = self._project_to_ego_z_plane(positions, ego_c2nurec, ego_positions)
        else:
            pc_nearest_ego_indices = (positions[..., None, :] - ego_positions).norm(dim=-1).argmin(dim=-1)

        if point_cloud.camera_footprint_scale is not None:
            pc_scale = unpack_optional(point_cloud.camera_footprint_scale).to(self.device).unsqueeze(-1).clamp_min(
                base_scale
            ) * np.sqrt(sparsity_compensation)
        else:
            pc_scale = torch.full(
                (positions.shape[0], 1), base_scale, device=self.device, dtype=torch.float32
            ) * np.sqrt(sparsity_compensation)

        pc_normals = None
        if self.config.init_with_normals:
            if point_cloud.normal is not None:
                pc_normals = point_cloud.normal.to(self.device)
            else:
                normals_local = torch.tensor([0, 0, 1], dtype=positions.dtype, device=self.device).repeat(
                    positions.shape[0], 1
                )
                pc_normals = (ego_c2nurec[pc_nearest_ego_indices][:, :3, :3] @ normals_local[..., :, None]).squeeze(-1)

        # initialize random points along the ego trajectory
        random_pts = None
        rand_scale = None
        rand_normals = None

        if self.config.num_random_points > 0:
            scene_extent = datasource.get_aabb().get_extent().max().item()
            if isinstance(self, AccumulatedPointCloudRoadInitialization):
                _, far_plane_depth = self._resolve_scene_extent_and_far_plane(scene_extent)
            else:
                far_plane_depth = self.config.far_radius_factor * scene_extent / 2
            num_random_pts = self.config.num_random_points

            # Get camera frusta for visibility filtering
            camera_frusta = [
                camera_frustum.to(self.device)
                for camera_id in getattr(datasource.datasource, "camera_ids", [])
                for camera_frustum, _ in unpack_optional(
                    datasource.get_camera_frusta(camera_id=camera_id, far_plane_depth=far_plane_depth)
                )
            ]

            # Sample base positions uniformly from the ego trajectory
            # ego_positions shape is [M, 3]
            idx = torch.randint(0, ego_positions.shape[0], (num_random_pts,), device=self.device)
            base_pts = ego_positions[idx]

            road_width_radius = self.config.get("random_road_width_radius", 15.0)
            r = torch.sqrt(torch.rand(num_random_pts, 1, device=self.device)) * road_width_radius
            theta = torch.rand(num_random_pts, 1, device=self.device) * 2 * np.pi

            offset_x = r * torch.cos(theta)
            offset_y = r * torch.sin(theta)
            offset_z = torch.zeros_like(offset_x)

            random_pts = base_pts + torch.cat([offset_x, offset_y, offset_z], dim=1)

            visibility_mask = (
                torch.vstack([frustum.points_in_frustum(random_pts) for frustum in camera_frusta]).sum(0) > 0
            )
            random_pts = random_pts[visibility_mask]

            random_pts, rand_nearest_ego_indices = self._project_to_ego_z_plane(random_pts, ego_c2nurec, ego_positions)

            dist_to_ego = (
                torch.sqrt(knn_points(random_pts.unsqueeze(0), ego_positions.unsqueeze(0)).dists)
                .clamp_min(1e-7)
                .squeeze(0)
            )
            rand_scale = (dist_to_ego * self.config.get("observation_scale_factor", 0.01)).clamp_min(base_scale)

            if self.config.init_with_normals:
                normals_local = torch.tensor([0, 0, 1], dtype=random_pts.dtype, device=self.device).repeat(
                    random_pts.shape[0], 1
                )
                rand_normals = (ego_c2nurec[rand_nearest_ego_indices][:, :3, :3] @ normals_local[..., :, None]).squeeze(
                    -1
                )

        if random_pts is not None:
            assert rand_scale is not None
            all_positions = torch.cat([positions, random_pts], dim=0)
            observation_scale = torch.cat([pc_scale, rand_scale], dim=0)
            normals = (
                torch.cat([pc_normals, rand_normals], dim=0)
                if (pc_normals is not None and rand_normals is not None)
                else None
            )
        else:
            all_positions = positions
            observation_scale = pc_scale
            normals = pc_normals

        if not isinstance(self.config.default_scale, float):
            z_scale = torch.full(
                (all_positions.shape[0], 1), self.config.default_scale[2], device=self.device, dtype=torch.float32
            )
            observation_scale = torch.cat([observation_scale, observation_scale, z_scale], dim=1)

        if self.debug_viz:
            visualizer = get_visualdebugger()
            visualizer.add_point_cloud(
                "road_point_cloud",
                positions.cpu().numpy(),
                radius=1e-3,
                point_render_mode="quad",
                colors_quantities={"colored": unpack_optional(point_cloud.color).cpu().numpy()},
                enabled=True,
            )
            if random_pts is not None:
                visualizer.add_point_cloud(
                    "road_random_points",
                    random_pts.cpu().numpy(),
                    radius=1e-3,
                    point_render_mode="quad",
                    enabled=True,
                )
            visualizer.show()

        if normals is not None:
            self.init_from_positions(positions=all_positions, normals=normals, observation_scale=observation_scale)
        else:
            self.init_from_positions(positions=all_positions, observation_scale=observation_scale)

        self.init_additional_buffers(point_cloud)


class LidarRigTrajRoadInitialization(LidarRigTrajInitializationBase, RoadInitializationMixin):
    def initialize_from_datasource(self, datasource: DataSourceSummary, **kwargs) -> None:
        """
        Initialize from colored point clouds and with HUGSIM's strategy which encourages a flatter road initialization
        by setting the height of each point to that of the closest ego position. This initialization is intended to be used
        in conjunction with a default_scale configuration that initializes the points to be flat disks
        (via a parameter such as [0.1, 0.1, 0.001])
        """
        self._initialize_from_datasource(datasource, **kwargs)


class NoPointsFoundException(Exception):
    pass


class ConstructionBarriersInitialization(BaseInitialization):
    """Random static SH seeds from upstream construction detections only."""

    cuboids: BarrierCuboids | None = None

    def initialize_from_datasource(self, datasource: DataSourceSummary, **kwargs) -> None:
        """Sample neutral-color SH seeds from static NCore cuboids, skipping empty inputs."""
        source = datasource.datasource
        if not isinstance(source, NCOREDataSource):
            raise ValueError("Construction-barrier initialization requires an NCOREDataSource")
        if not {"cone", "pole", "barrier"}.issubset(source.cuboid_tracks_params.excluded_class_names):
            raise ValueError("NCore barriers must be excluded from the ordinary cuboid-track path")
        self.cuboids = BarrierCuboids.from_ncore(
            source.sequence_loader, source.time_range_us, source.world_to_nurec.matrix
        )
        if not len(self.cuboids.centers):
            raise NoPointsFoundException("No construction-barrier detections in the upstream export")
        positions = self.cuboids.sample(self.config.points_per_cuboid, self.config.seed, self.device)
        self.init_from_positions(positions)
        color = torch.full_like(positions, 128, dtype=torch.uint8)
        self.init_additional_buffers_SH(PointCloud(xyz_start=positions, xyz_end=positions, color=color))
        log.info(
            "Construction barriers: %d cuboids x %d random points = %d Gaussians",
            len(self.cuboids.centers),
            self.config.points_per_cuboid,
            len(positions),
        )


class LidarConstructionInitialization(BaseInitialization):
    """Initialize static SH Gaussians from measured cone initialization points."""

    def initialize_from_datasource(self, datasource: DataSourceSummary, **kwargs) -> None:
        """Preserve measured positions and RGB with a fixed initial scale in world metres."""
        source = datasource.datasource
        if not isinstance(source, NCOREDataSource):
            raise ValueError("LiDAR construction initialization requires an NCOREDataSource")
        if "cone" not in source.cuboid_tracks_params.excluded_class_names:
            raise ValueError("LiDAR cones must be excluded from the ordinary cuboid-track path")
        aux = source.aux_loader
        if aux is None:
            raise NoPointsFoundException("SAM3 object masks are unavailable; skipping the optional object layer")
        camera_ids = [self.config.camera_id] if self.config.camera_id is not None else source.train_camera_ids
        cone_camera_ids = []
        for camera_id in camera_ids:
            if not aux.has_object_instance_segmentation(camera_id):
                continue
            meta = aux.get_object_instance_segmentation_meta(camera_id)
            if "method" not in meta:
                raise ValueError("Instance-segmentation metadata is missing 'method'")
            if meta["method"] == "sam3-objects":
                cone_camera_ids.append(camera_id)
        if not cone_camera_ids:
            raise NoPointsFoundException("No SAM3 cone masks; skipping the cone layer")
        if len(cone_camera_ids) != 1:
            raise ValueError("Multiple cameras have SAM3 cone masks; set the cone initializer's camera_id")
        lidar_ids = [self.config.lidar_id] if self.config.lidar_id is not None else source.train_lidar_ids
        if not lidar_ids:
            raise NoPointsFoundException("No training LiDAR; skipping the cone layer")
        if len(lidar_ids) != 1:
            raise ValueError("Multiple training LiDAR sensors; set the cone initializer's lidar_id")
        init_points = build_lidar_cone_init_points(source, cone_camera_ids[0], lidar_ids[0], self.device)
        xyz, rgb = init_points.xyz_world, init_points.rgb
        if (
            xyz.ndim != 2
            or xyz.shape[1] != 3
            or xyz.dtype.kind not in "fiu"
            or not np.isfinite(xyz).all()
            or rgb.shape != xyz.shape
            or rgb.dtype != np.uint8
        ):
            raise ValueError("LiDAR cone initialization points require finite XYZ and row-aligned uint8 RGB")
        if not len(xyz):
            raise NoPointsFoundException("No cone initialization points available for this section")
        scale_m = float(self.config.default_scale)
        if not np.isfinite(scale_m) or scale_m <= 0 or not 0 < float(self.config.default_density) < 1:
            raise ValueError("LiDAR construction scale must be positive and density must be in (0, 1)")
        transform = np.asarray(source.world_to_nurec.get_transformation_matrices()[0], dtype=np.float64)
        scale = float(source.world_to_nurec.target_scale)
        if transform.shape != (4, 4) or not np.isfinite(scale) or scale <= 0:
            raise ValueError("LiDAR construction initialization requires a uniformly scaled rigid world_to_nurec")
        rotation = transform[:3, :3] / scale
        if (
            not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1])
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1)
        ):
            raise ValueError("LiDAR construction initialization requires a uniformly scaled rigid world_to_nurec")
        positions = torch.tensor(
            source.world_to_nurec.transform_points(xyz).reshape(-1, 3), dtype=torch.float32, device=self.device
        )
        self.init_from_positions(positions, observation_scale=torch.full_like(positions, scale_m * scale))
        color = torch.tensor(rgb, dtype=torch.uint8, device=self.device)
        self.init_additional_buffers_SH(PointCloud(xyz_start=positions, xyz_end=positions, color=color))
        log.info("LiDAR cones: %d measured initialization points", len(xyz))


class DynamicTracksInitialization(BaseInitialization):
    def initialize_from_datasource(self, datasource: DataSourceSummary, **kwargs) -> None:
        cuboid_tracks = kwargs.get("cuboid_tracks", None)
        assert isinstance(cuboid_tracks, CuboidTracks), (
            f"[{self.__class__.__name__}] cuboid_tracks is required and must be CuboidTracks"
        )

        assert isinstance(datasource.datasource, NCOREDataSource), (
            f"[{self.__class__.__name__}] Only NCOREDataSource is supported for dynamic-tracks initialization"
        )

        # As in LidarRigTrajInitializationBase, if no sensor_ids are specified, use train_sensor_ids.
        # Only an unset (null) list is filled in. An explicitly empty one means "seed nothing from
        # this modality".
        if self.config.name == "lidar-dynamic-tracks":
            camera_ids = None
            lidar_ids = self.config.get("lidar_ids", None)
            if lidar_ids is None:
                lidar_ids = list(datasource.datasource.train_lidar_ids)
                assert lidar_ids, "lidar-dynamic-tracks init needs either a lidar_id specified or in train_lidar_ids"
        else:
            lidar_ids = None
            camera_ids = self.config.get("camera_ids", None)
            if camera_ids is None:
                camera_ids = list(datasource.datasource.train_camera_ids)
                assert camera_ids, (
                    "camera-dynamic-tracks init needs either a camera_id specified or in train_camera_ids"
                )

        # TODO filter out cuboid tracks ahead of time
        point_clouds: List[PointCloud] = []
        gaussian_cuboid_ids: List[torch.Tensor] = []
        gaussian_scales: List[torch.Tensor] = []
        observed_counts: List[int] = []

        track_point_clouds = list(
            unpack_optional(
                datasource.get_track_point_clouds(
                    cuboid_tracks=cuboid_tracks,
                    cuboid_dim_scale_factor=1.0,
                    lidar_ids=lidar_ids,
                    camera_ids=camera_ids,
                    return_color=True,
                    step_frame=self.config.get("step_frame", 1),
                    keep_all_track_poses=self.config.keep_all_track_poses,
                    device=self.device,
                )
            )
        )

        track_id_counts: dict[str, int] = defaultdict(int)
        track_pcs: dict[str, list[PointCloud]] = defaultdict(list)
        track_gids: dict[str, list[torch.Tensor]] = defaultdict(list)
        track_scales: dict[str, list[torch.Tensor]] = defaultdict(list)
        for track_point_cloud in track_point_clouds:
            point_cloud = track_point_cloud.point_cloud

            if self.config.symmetric_axis is not None:
                symmetric_point_cloud = point_cloud.flipped(axis=self.config.symmetric_axis)
                point_cloud = PointCloud.collate_fn([point_cloud, symmetric_point_cloud], device=self.device)

            track_id_counts[track_point_cloud.track_id] += point_cloud.n_points

            track_pcs[track_point_cloud.track_id].append(point_cloud.to(self.device))
            track_gids[track_point_cloud.track_id].append(
                torch.full(
                    (point_cloud.n_points, 1),
                    fill_value=cuboid_tracks.tracks_id.index(track_point_cloud.track_id),
                    dtype=torch.int32,
                    device=self.device,
                )
            )
            track_scales[track_point_cloud.track_id].append(
                unpack_optional(point_cloud.camera_footprint_scale).unsqueeze(-1)
            )

        for track_id in cuboid_tracks.tracks_id:
            # Add tracks with no point clouds yielded to the list. This can happen to a track if:
            # 1. all its cuboids are skipped by step_frame
            # 2. all its cuboids are skipped by timestamp checks when keep_all_track_poses is false
            if track_id not in track_id_counts:
                track_id_counts[track_id] = 0

        # subsample the point clouds to desired size
        for track_id, count in track_id_counts.items():
            if count == 0:
                # Add random initialization for tracks with no detected points
                track_index = cuboid_tracks.tracks_id.index(track_id)
                cuboid_extent = cuboid_tracks.cuboids_dims[track_index].to(self.device)

                point_cloud, gaussian_cuboid_id, gaussian_scale = track_random_initialization(
                    self.config.num_point_cloud_points_per_track,
                    track_index,
                    cuboid_extent,
                    self.config.default_scale,
                    self.device,
                )

                point_clouds.append(point_cloud)
                gaussian_cuboid_ids.append(gaussian_cuboid_id)
                gaussian_scales.append(gaussian_scale)
                observed_counts.append(
                    0
                )  # These are random points, so no need to track the observed count for sparsity compensation
            elif count > self.config.num_point_cloud_points_per_track:
                # uniformly sample points from each point cloud
                original_observed_counts = [pc.n_points for pc in track_pcs[track_id]]
                subsampled_track_pcs, subsampled_track_gids, subsampled_track_scales, subsampled_observed_counts = (
                    subsample_points(
                        self.config.num_point_cloud_points_per_track,
                        track_pcs[track_id],
                        track_gids[track_id],
                        track_scales[track_id],
                        original_observed_counts,
                    )
                )

                point_clouds.extend(subsampled_track_pcs)
                gaussian_cuboid_ids.extend(subsampled_track_gids)
                gaussian_scales.extend(subsampled_track_scales)
                observed_counts.extend(subsampled_observed_counts)
            else:
                point_clouds.extend(track_pcs[track_id])
                gaussian_cuboid_ids.extend(track_gids[track_id])
                gaussian_scales.extend(track_scales[track_id])
                observed_counts.extend([pc.n_points for pc in track_pcs[track_id]])

                if self.config.get("fill_with_random_points", False):
                    num_random_points = self.config.num_point_cloud_points_per_track - count
                    # Add random initialization for tracks
                    track_index = cuboid_tracks.tracks_id.index(track_id)
                    cuboid_extent = cuboid_tracks.cuboids_dims[track_index].to(self.device)

                    point_cloud, gaussian_cuboid_id, gaussian_scale = track_random_initialization(
                        num_random_points,
                        track_index,
                        cuboid_extent,
                        self.config.default_scale,
                        self.device,
                    )

                    point_clouds.append(point_cloud)
                    gaussian_cuboid_ids.append(gaussian_cuboid_id)
                    gaussian_scales.append(gaussian_scale)
                    observed_counts.append(
                        0
                    )  # These are random points, so no need to track the observed count for sparsity compensation

        if len(point_clouds) == 0:
            raise NoPointsFoundException("No point clouds were found in the dataset")

        if self.config.num_point_cloud_points_in_layer is not None:
            # Further subsample the point clouds to the desired number of points in the layer
            point_clouds, gaussian_cuboid_ids, gaussian_scales, observed_counts = subsample_points(
                self.config.num_point_cloud_points_in_layer,
                point_clouds,
                gaussian_cuboid_ids,
                gaussian_scales,
                observed_counts,
            )

        point_cloud = PointCloud.collate_fn(point_clouds, device=self.device)

        # Sparsity compensation
        observation_scales = []
        for scale, observed_count in zip(gaussian_scales, observed_counts):
            sparsity_compensation = max(1, np.sqrt(observed_count / scale.shape[0])) if observed_count > 0 else 1
            observation_scales.append(scale * sparsity_compensation)

        self.init_from_positions(
            point_cloud.xyz_end,
            observation_scale=torch.cat(observation_scales),
        )
        self.init_additional_buffers(point_cloud)
        self.gaussian_cuboid_ids = gaussian_cuboid_ids


class AccumulatedPointCloudInitializationBase(BaseInitialization):
    _fused_scene_extent: float | None = None

    def _resolve_scene_extent_and_far_plane(self, scene_extent: float) -> tuple[float, float]:
        far_plane_depth = self.config.far_radius_factor * scene_extent / 2
        if np.isfinite(far_plane_depth) and far_plane_depth > 0.1:
            return scene_extent, far_plane_depth

        if self._fused_scene_extent is None:
            raise RuntimeError("Cannot compute scene extent from an empty accumulated point cloud")
        fused_scene_extent = self._fused_scene_extent
        fused_far_plane_depth = self.config.far_radius_factor * fused_scene_extent / 2
        if not np.isfinite(fused_far_plane_depth) or fused_far_plane_depth <= 0.1:
            raise RuntimeError(
                "Accumulated point-cloud scene extent is too small for a valid camera frustum: "
                f"extent={fused_scene_extent:.6g} m, far={fused_far_plane_depth:.6g} m"
            )
        log.warning(
            "Datasource scene extent %.6g m gives invalid far plane %.6g m; "
            "using accumulated point-cloud extent %.6g m instead",
            scene_extent,
            far_plane_depth,
            fused_scene_extent,
        )
        return fused_scene_extent, fused_far_plane_depth

    @staticmethod
    def load_ply_with_segment_id(
        filename: str,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.uint8], Optional[NDArray[np.uint8]]]:
        ply_data = pcu.load_triangle_mesh(filename).vertex_data
        ply_metadata = ply_data.custom_attributes

        xyz = ply_data.positions
        normal = ply_data.normals
        color = ply_data.colors
        segment_id = ply_metadata["segment_id"].astype(np.uint8) if "segment_id" in ply_metadata else None

        return xyz, normal, color, segment_id

    def get_point_clouds_and_ego_poses(self, datasource: DataSourceSummary) -> tuple[PointCloud, torch.Tensor, float]:
        """
        Returns points loaded from a ply file and ego poses from the datasource, optionally filtering by class id.
        Subsamples the point cloud to the maximum size defined by `num_point_cloud_points` and returns a sparsity compensation factor
        that can be used to increase the scale of the remaining points.
        """

        assert isinstance(datasource.datasource, (NCOREDataSource)), (
            f"[{self.__class__.__name__}] Only NCOREDataSource is supported for from_accumulated_point_cloud initialization"
        )

        # Get ego positions in NuRec coordinates
        rig_trajectories = unpack_optional(datasource.get_rig_trajectories())
        T_world_to_nurec_np, S_world_to_nurec_np = rig_trajectories.world_to_nurec.get_transformation_matrices()
        T_world_to_nurec = torch.tensor(T_world_to_nurec_np, device=self.device, dtype=torch.float32)
        S_world_to_nurec = torch.tensor(S_world_to_nurec_np, device=self.device, dtype=torch.float32)
        # Ego poses that map from camera_to_nurec
        ego_c2nurec = torch.cat(
            [
                (T_world_to_nurec @ rig_traj.T_rig_worlds.float().to(self.device) @ S_world_to_nurec)
                for rig_traj in rig_trajectories.rig_trajectories
            ]
        )

        def maybe_read_point_cloud():
            xyz_e_world, normal_world, color, segment_id = self.load_ply_with_segment_id(self.config.point_cloud_path)

            xyz_e_nurec = datasource.datasource.world_to_nurec.transform_points(xyz_e_world)
            self._fused_scene_extent = (
                float((xyz_e_nurec.max(axis=0) - xyz_e_nurec.min(axis=0)).max()) if xyz_e_nurec.shape[0] > 0 else None
            )

            normal_nurec = None
            if normal_world is not None:
                # Rotate normal from world to NuRec coordinates, multiply from the right with the transpose is equivalent to multiply from the left
                normal_nurec = normal_world @ datasource.datasource.world_to_nurec.matrix[:3, :3].T
                normal_nurec = normal_nurec / np.maximum(np.linalg.norm(normal_nurec, axis=1)[..., np.newaxis], 1e-5)

            xyz_e_nurec = to_torch(xyz_e_nurec, device=str(self.device), dtype=torch.float32)
            normal_nurec = (
                to_torch(normal_nurec, device=str(self.device), dtype=torch.float32)
                if normal_nurec is not None
                else None
            )
            color = to_torch(color[:, :3], device=str(self.device), dtype=torch.uint8) if color is not None else None
            segment_id = (
                to_torch(segment_id, device=str(self.device), dtype=torch.uint8) if segment_id is not None else None
            )

            # get scales based on distance to ego poses
            dist_to_ego_positions = (
                torch.sqrt(knn_points(xyz_e_nurec.unsqueeze(0), ego_c2nurec[:, :3, 3].unsqueeze(0)).dists)
                .clamp_min(1e-7)
                .squeeze(0)
                .squeeze(-1)
            )
            scale = dist_to_ego_positions * self.config.get("observation_scale_factor", 0.01)

            point_cloud = PointCloud(
                xyz_start=xyz_e_nurec,
                xyz_end=xyz_e_nurec,
                flags=None,
                color=color,
                normal=normal_nurec,
                semantic_class_id=segment_id,
                camera_footprint_scale=scale,
                sensor_type=["camera"],
            )

            # Filter the point cloud based on labels_to_use or labels_to_ignore
            if bool(self.labels_to_ignore) or bool(self.labels_to_use):
                point_cloud = self.filter_point_cloud_by_labels(point_cloud, datasource)
            if self.config.non_dynamic_points_only:
                assert isinstance(datasource.datasource, NCOREDataSource), (
                    "non_dynamic_points_only requires NCOREDataSource"
                )
                cuboidtracks_dynamic = datasource.datasource.get_cuboid_tracks(dynamic_only=True, world_frame=False)
                dynamic_class_ids = None
                if point_cloud.semantic_class_id is not None:
                    sensor_type = unpack_optional(point_cloud.sensor_type)
                    dynamic_class_ids = datasource.datasource.get_dynamic_semantic_class_ids(
                        sensor_type=list(sensor_type),
                        dynamic_class_names=list(datasource.datasource.camera_point_cloud_dynamic_classes),
                    )

                inside_dynamic_tracks_mask = compute_point_cloud_inside_tracks_mask(
                    point_cloud=point_cloud,
                    time_range_us=datasource.datasource.time_range_us,
                    cuboidtracks_dynamic=cuboidtracks_dynamic,
                    track_padding_m=datasource.datasource.valid_points_cuboid_track_params.track_padding_m,
                    batch_size=20,
                    tqdm_disabled=datasource.datasource.tqdm_disabled,
                    restrict_to_class_ids=dynamic_class_ids,
                )
                point_cloud = point_cloud[~inside_dynamic_tracks_mask]

            original_n_points = point_cloud.n_points

            # subsample point cloud to desired size
            if point_cloud.n_points > self.config.num_point_cloud_points:
                random_idxs = torch.from_numpy(
                    np.random.choice(point_cloud.n_points, self.config.num_point_cloud_points, replace=False)
                ).to(self.device)
                point_cloud = point_cloud[random_idxs]

            return point_cloud, original_n_points

        # Actually read the point cloud
        point_cloud, original_n_points = maybe_read_point_cloud()

        # We need to use the original number of points (before subsampling) to compute the sparsity compensation factor
        sparsity_compensation = max(1, original_n_points / self.config.num_point_cloud_points)
        log.info(
            f"[{self.__class__.__name__}] Original number of points: {original_n_points}, number of points: {point_cloud.n_points}, sparsity compensation: {sparsity_compensation}"
        )

        return point_cloud, ego_c2nurec, sparsity_compensation


class AccumulatedPointCloudInitialization(AccumulatedPointCloudInitializationBase, InitializationMixin):
    def initialize_from_datasource(self, datasource: DataSourceSummary, **kwargs) -> None:
        """
        Initialize from a combination of colored point clouds and random points.

        The point cloud is loaded from a ply file and then transformed to NuRec coordinates.
        Random points are sampled from two strategies: near and far.
        Near points are sampled from a set of spheres of radius `near_radius` around each of the ego positions.
        Far points are sampled from a sphere of the size of the scene extent (centered around the ego positions in the middle of the sequence).

        Camera frusta are used to filter out invisible points.
        """
        self._initialize_from_datasource(datasource, **kwargs)


class AccumulatedPointCloudRoadInitialization(AccumulatedPointCloudInitializationBase, RoadInitializationMixin):
    def initialize_from_datasource(self, datasource: DataSourceSummary, **kwargs) -> None:
        """
        Initialize from colored point clouds, which are loaded from a ply file, and with HUGSIM's strategy which encourages a flatter road
        initialization by setting the height of each point to that of the closest ego position. This initialization is intended to be used
        in conjunction with a default_scale configuration that initializes the points to be flat disks
        (via a parameter such as [0.1, 0.1, 0.001])
        """
        self._initialize_from_datasource(datasource, **kwargs)


class InstantNuRecPlyInitialization(BaseInitialization):
    def _resolve_ply_path(self, datasource: DataSourceSummary) -> Path:
        # When `path` is unset, fall back to dataset-directory autodetect.
        configured_path = (
            self.config.path if "path" in self.config and not OmegaConf.is_missing(self.config, "path") else None
        )
        if configured_path is not None:
            return Path(configured_path)

        ncore_ds = datasource.datasource
        assert isinstance(ncore_ds, NCOREDataSource), (
            f"[{self.__class__.__name__}] Only NCOREDataSource is supported for automatic instant_nurec PLY path resolution"
        )

        input_dir = Path(str(ncore_ds.get_path().parent))
        sequence_id = ncore_ds.sequence_id
        return input_dir / "instant_nurec" / "ply" / sequence_id / f"{sequence_id}.ply"

    def _load_ply(self, datasource: DataSourceSummary) -> PLYGaussianLoader:
        if datasource.runtime_instant_nurec_ply is not None:
            raise ValueError("Disk instant_nurec PLY initialization cannot consume an in-process instant_nurec payload")
        assert isinstance(datasource.datasource, NCOREDataSource), (
            f"[{self.__class__.__name__}] Only NCOREDataSource is supported for instant_nurec PLY initialization"
        )
        ply_path = self._resolve_ply_path(datasource)
        log.info("[%s] Loading Gaussians from disk at %s", self.__class__.__name__, ply_path)
        loaded_ply = PLYGaussianLoader(ply_path, device=str(self.device))
        # Disk PLYs are world-space and privately owned by this initializer.
        loaded_ply.transform(
            torch.FloatTensor(datasource.datasource.world_to_nurec.matrix).to(self.device),
        )
        return loaded_ply

    def get_point_clouds_and_ego_poses(self, datasource: DataSourceSummary, road: bool) -> dict[str, torch.Tensor]:
        """
        Returns points and other gaussian properties from the selected source, filtered by a road mask.
        Randomly subsamples the point cloud to the maximum size defined by `num_point_cloud_points`.
        The input pointcloud can be generated by running instant_nurec with the export_ply config option.

        This initialization assumes the following:
        - scale and density are activated.
        - disk position and rotation are in world space and must be converted to NuRec space;
          the in-process bridge publishes an already-converted, read-only payload.
        - rotation is a quaternion in wxyz format.
        - features_albedo is in SH, not RGB.
        """

        loaded_ply = self._load_ply(datasource)
        log.info(f"[{self.__class__.__name__}] Original number of points: {loaded_ply.positions.shape[0]}")
        # PRUNING
        # Remove points with infinite densities
        mask = torch.isfinite(loaded_ply.densities.squeeze())
        log.info(f"[{self.__class__.__name__}] Kept {mask.sum().item()} finite points")

        # Remove points based on road bool (road layer or not)
        assert loaded_ply.road_mask is not None, "Road mask is required for initialization"
        if road:
            mask &= loaded_ply.road_mask
            log.info(f"[{self.__class__.__name__}] Kept {mask.sum().item():,} road points")
        else:
            mask &= ~loaded_ply.road_mask
            log.info(f"[{self.__class__.__name__}] Kept {mask.sum().item():,} non-road points")

        # Remove points based on density
        if self.config.density_pruning.enabled:
            # Only keep the top-k points with the highest densities, where k = self.config.density_pruning.keep
            number_gaussians_to_keep = self.config.density_pruning.keep
            current_n_points = mask.sum().item()
            if number_gaussians_to_keep is not None and number_gaussians_to_keep < current_n_points:
                # Get the indices of the points in the mask that are True
                idx_in_full = mask.nonzero(as_tuple=True)[0]
                # Get the densities of the masked-in points
                masked_densities = loaded_ply.densities.squeeze()[mask]
                # Find the top-k densities and their indices within masked_densities
                # Compute the indices to be removed: smallest densities (not in topk)
                removed_indices_in_masked = torch.topk(
                    masked_densities, current_n_points - number_gaussians_to_keep, largest=False, dim=0
                ).indices
                # Map back to original mask
                mask[idx_in_full[removed_indices_in_masked]] = False
                log.info(f"[{self.__class__.__name__}] Kept {mask.sum().item()} topk density points")
            # only keep points with densities greater than the threshold
            if self.config.density_pruning.density_threshold is not None:
                mask &= (
                    get_activation("sigmoid", inverse=False)(loaded_ply.densities.squeeze())
                    > self.config.density_pruning.density_threshold
                )
                log.info(
                    f"[{self.__class__.__name__}] Kept {mask.sum().item()} density threshold ({self.config.density_pruning.density_threshold}) points"
                )

        # randomly remove points if there are too many
        original_n_points = mask.sum().item()
        if original_n_points > self.config.num_point_cloud_points:
            true_indices = mask.nonzero(as_tuple=False).flatten()
            removal_count = int(original_n_points) - self.config.num_point_cloud_points
            removal_idcs = true_indices[torch.randperm(int(original_n_points), device=self.device)[:removal_count]]
            mask[removal_idcs] = False
            log.info(f"[{self.__class__.__name__}] Kept {mask.sum().item()} points after random subsampling")
        sparsity_compensation = max(1, original_n_points / self.config.num_point_cloud_points)
        log.info(
            f"[{self.__class__.__name__}] Original number of points before random sampling: {original_n_points}, new number of points: {mask.sum().item()}, sparsity compensation: {sparsity_compensation}"
        )

        densities = loaded_ply.densities[mask, :]
        scales = loaded_ply.scales[mask, :]
        positions = loaded_ply.positions[mask, :]
        rotations = loaded_ply.rotations[mask, :]
        features_albedo = loaded_ply.features_albedo[mask, :]  # albedo features are in SH not RGB

        if loaded_ply.features_specular is not None:
            features_specular = loaded_ply.features_specular[mask, :]
        else:
            num_specular_dims = sh_degree_to_specular_dim(self.gaussian_config.progressive_training.max_n_features)
            features_specular = torch.zeros(positions.shape[0], num_specular_dims, device=self.device)

        return {
            "positions": positions,
            "rotations": rotations,
            "densities": densities,
            "scales": scales,
            "features_albedo": features_albedo,
            "features_specular": features_specular,
        }

    def initialize_from_datasource(self, datasource: DataSourceSummary, road: bool = False, **kwargs) -> None:
        """
        Initialize from the selected instant_nurec source.
        """

        point_cloud_dict = self.get_point_clouds_and_ego_poses(datasource, road)
        self.positions = nn.Parameter(point_cloud_dict["positions"])
        self.rotations = nn.Parameter(point_cloud_dict["rotations"])
        self.scales = nn.Parameter(point_cloud_dict["scales"])
        self.densities = nn.Parameter(point_cloud_dict["densities"])
        self.features_albedo = nn.Parameter(point_cloud_dict["features_albedo"])
        self.features_specular = nn.Parameter(point_cloud_dict["features_specular"])


class InstantNuRecPlyRoadInitialization(InstantNuRecPlyInitialization):
    def initialize_from_datasource(self, datasource: DataSourceSummary, road: bool = True, **kwargs) -> None:
        """
        Initialize from one or more ply files obtained from instant_nurec, but with only the points labeled as road
        """
        super().initialize_from_datasource(datasource, road=True, **kwargs)


class InProcessInstantNuRecInitialization(InstantNuRecPlyInitialization):
    """Consume generated NuRec-space Gaussians without falling back to disk."""

    def _load_ply(self, datasource: DataSourceSummary) -> PLYGaussianLoader:
        runtime_ply = datasource.runtime_instant_nurec_ply
        if runtime_ply is None:
            raise ValueError("In-process instant_nurec initialization requires an in-memory instant_nurec payload")
        if not isinstance(runtime_ply, PLYGaussianLoader):
            raise TypeError(f"Unexpected in-memory instant_nurec payload type: {type(runtime_ply).__name__}")
        configured_path = (
            self.config.path if "path" in self.config and not OmegaConf.is_missing(self.config, "path") else None
        )
        if configured_path is not None:
            raise ValueError(
                f"In-process instant_nurec payload cannot be combined with explicit instant_nurec PLY path {configured_path!r}"
            )
        datasource.runtime_instant_nurec_ply_consumed = True
        log.info("[%s] Loading Gaussians from the in-process instant_nurec payload", self.__class__.__name__)
        # The bridge has already moved the payload to the model device and NuRec
        # coordinates. Borrow it read-only; shared filtering creates independent
        # tensors for each Gaussian layer without copying the entire payload.
        return runtime_ply


class InProcessInstantNuRecRoadInitialization(InProcessInstantNuRecInitialization):
    def initialize_from_datasource(self, datasource: DataSourceSummary, road: bool = True, **kwargs) -> None:
        """Initialize from generated instant_nurec Gaussians labeled as road."""
        super().initialize_from_datasource(datasource, road=True, **kwargs)


class PlyInitialization(BaseInitialization):
    def initialize_from_datasource(self, datasource: DataSourceSummary, **kwargs) -> None:
        """
        Initialize from a ply file (ie: one obtained from instant_nurec). The ply positions are assumed to be in world space.
        """

        loaded_ply = PLYGaussianLoader(self.config.path, device=str(self.device))

        if self.config.translate_world_to_nurec:
            assert isinstance(datasource.datasource, NCOREDataSource), (
                f"[{self.__class__.__name__}] Only NCOREDataSource is supported for world to nurec translation"
            )
            # Map from world space to NuRec space
            loaded_ply.transform(
                torch.FloatTensor(datasource.datasource.world_to_nurec.matrix).to(self.device),
            )

        self.positions = nn.Parameter(loaded_ply.positions)
        self.rotations = nn.Parameter(loaded_ply.rotations)
        self.scales = nn.Parameter(loaded_ply.scales)
        self.densities = nn.Parameter(loaded_ply.densities)
        self.features_albedo = nn.Parameter(loaded_ply.features_albedo)
        if loaded_ply.features_specular is not None:
            self.features_specular = nn.Parameter(loaded_ply.features_specular)
        else:
            num_specular_dims = sh_degree_to_specular_dim(self.gaussian_config.progressive_training.max_n_features)
            self.features_specular = nn.Parameter(
                torch.zeros(self.positions.shape[0], num_specular_dims, device=self.device)
            )


class PlyTracksInitialization(BaseInitialization):
    def initialize_from_datasource(self, datasource: DataSourceSummary, **kwargs) -> None:
        """
        Initialize specified tracks with ply files and use a fallback initializer for remaining tracks
        """
        cuboid_tracks = kwargs.get("cuboid_tracks", None)
        assert isinstance(cuboid_tracks, CuboidTracks), (
            f"[{self.__class__.__name__}] cuboid_tracks is required and must be CuboidTracks"
        )

        track_ids = set(self.config.track_ids.keys())
        ply_track_ids = []
        fallback_track_ids = []

        for i, track_id in enumerate(cuboid_tracks.tracks_id):
            if track_id in track_ids:
                ply_track_ids.append(i)
            else:
                fallback_track_ids.append(i)

        positions: List[torch.Tensor] = []
        rotations: List[torch.Tensor] = []
        scales: List[torch.Tensor] = []
        densities: List[torch.Tensor] = []
        features_albedo: List[torch.Tensor] = []
        features_specular: List[torch.Tensor] = []
        gaussian_cuboid_ids: List[torch.Tensor] = []

        if len(fallback_track_ids) > 0:
            log.info(f"Initializing {len(fallback_track_ids)} fallback tracks using {self.config.fallback.name}")
            try:
                fallback_tracks = CuboidTracks.Ops.subset_from_indices(cuboid_tracks, fallback_track_ids)
                fallback = BaseInitialization.factory(
                    self.config.fallback.name,
                    self.config.fallback,
                    self.gaussian_config,
                    self.trainer_config,
                    self.labels_to_ignore,
                    self.labels_to_use,
                )
                fallback.initialize_from_datasource(datasource, cuboid_tracks=fallback_tracks)

                positions.append(fallback.positions)
                rotations.append(fallback.rotations)
                scales.append(fallback.scales)
                densities.append(fallback.densities)
                features_albedo.append(fallback.features_albedo)
                features_specular.append(fallback.features_specular)
                gaussian_cuboid_ids.append(
                    torch.IntTensor(fallback_track_ids).to(self.device)[torch.cat(fallback.gaussian_cuboid_ids)]
                )
            except NoPointsFoundException:
                log.warning("No points found for fallback initialization")
                pass

        object_completion_tracks = CuboidTracks.Ops.subset_from_indices(cuboid_tracks, ply_track_ids)
        transform = torch.FloatTensor(self.config.transform).to(self.device)
        scale_activation = get_activation(self.gaussian_config.scale_activation, inverse=False)
        scale_activation_inv = get_activation(self.gaussian_config.scale_activation, inverse=True)

        for track_index, track_id in enumerate(object_completion_tracks.tracks_id):
            loaded_ply = PLYGaussianLoader(self.config.track_ids[track_id], device=str(self.device))

            loaded_ply.transform(transform)
            if self.config.scale_to_track:
                original_dims = object_completion_tracks.cuboids_dims[track_index]
                # Prefer dataset-configured padding if available; fallback to init/gaussian config or default
                padding_values = None
                if isinstance(datasource.datasource, NCOREDataSource):
                    # First, try lidar-points padding
                    padding_values = list(
                        getattr(
                            datasource.datasource.valid_points_cuboid_track_params,
                            "track_padding_m",
                            [],
                        )
                    )

                if not padding_values:
                    raise ValueError(f"No padding values found for track {track_id}")

                padding_tensor = torch.tensor(padding_values, dtype=original_dims.dtype, device=original_dims.device)
                adjusted_dims = torch.clamp(original_dims - padding_tensor, min=1e-6)
                scale_factor = adjusted_dims.norm().item()

                loaded_ply.scale(
                    scale_factor,
                    scale_activation,
                    scale_activation_inv,
                )
            positions.append(loaded_ply.positions)
            rotations.append(loaded_ply.rotations)
            scales.append(loaded_ply.scales)
            densities.append(loaded_ply.densities)
            features_albedo.append(loaded_ply.features_albedo)
            if loaded_ply.features_specular is not None:
                features_specular.append(loaded_ply.features_specular)
            else:
                num_specular_dims = sh_degree_to_specular_dim(self.gaussian_config.progressive_training.max_n_features)
                features_specular.append(
                    torch.zeros(
                        loaded_ply.features_albedo.shape[0],
                        num_specular_dims,
                        device=self.device,
                    )
                )
            gaussian_cuboid_ids.append(
                torch.full(
                    (loaded_ply.features_albedo.shape[0], 1),
                    fill_value=ply_track_ids[track_index],
                    dtype=torch.int32,
                    device=self.device,
                )
            )

        self.positions = nn.Parameter(torch.cat(positions))
        self.rotations = nn.Parameter(torch.cat(rotations))
        self.scales = nn.Parameter(torch.cat(scales))
        self.densities = nn.Parameter(torch.cat(densities))
        self.features_albedo = nn.Parameter(torch.cat(features_albedo))
        self.features_specular = nn.Parameter(torch.cat(features_specular))
        self.gaussian_cuboid_ids = gaussian_cuboid_ids


def compute_vertex_normals(vertices: NDArray[np.float32], faces: NDArray[np.int32]) -> NDArray[np.float32]:
    """Compute area-weighted vertex normals from a triangle mesh."""
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    face_normals = face_normals / np.maximum(norms, 1e-8)

    vertex_normals = np.zeros_like(vertices)
    for i in range(3):
        np.add.at(vertex_normals, faces[:, i], face_normals)
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    vertex_normals = vertex_normals / np.maximum(norms, 1e-8)
    return vertex_normals


def _lidar_ground_mesh_settings(config: DictConfig) -> GroundMeshReconstructionSettings:
    from apps.av.road_mesh.ground_mesh import GroundMeshReconstructionSettings

    return GroundMeshReconstructionSettings.from_mapping(config)


def compute_face_normals(vertices: NDArray[np.float32], faces: NDArray[np.int32]) -> NDArray[np.float32]:
    """Compute unit face normals from a triangle mesh."""
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    return face_normals / np.maximum(norms, 1e-8)


def compute_triangle_areas(vertices: NDArray[np.float32], faces: NDArray[np.int32]) -> NDArray[np.float32]:
    """Compute 3D triangle areas for a mesh."""
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    return (0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)).astype(np.float32)


def compute_triangle_projected_areas(
    vertices: NDArray[np.float32],
    faces: NDArray[np.int32],
    axis: int = 2,
) -> NDArray[np.float32]:
    """Compute triangle areas projected onto the plane orthogonal to ``axis``."""
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    plane_axes = [i for i in range(3) if i != axis]
    tri = vertices[faces][:, :, plane_axes]
    e0 = tri[:, 1] - tri[:, 0]
    e1 = tri[:, 2] - tri[:, 0]
    return (0.5 * np.abs(e0[:, 0] * e1[:, 1] - e0[:, 1] * e1[:, 0])).astype(np.float32)


def _compute_surface_sampling_areas(
    vertices: NDArray[np.float32],
    faces: NDArray[np.int32],
    area_mode: str,
) -> NDArray[np.float32]:
    if area_mode == "surface":
        return compute_triangle_areas(vertices, faces)
    if area_mode == "projected":
        return compute_triangle_projected_areas(vertices, faces, axis=2)
    raise ValueError(f"Unsupported surface_init_area_mode={area_mode!r}. Expected 'projected' or 'surface'.")


def _sample_mesh_surface_points(
    vertices: NDArray[np.float32],
    faces: NDArray[np.int32],
    num_samples: int,
    seed: int | None,
    area_mode: str = "projected",
    min_samples_per_face: int = 0,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int64], NDArray[np.float32]]:
    """Sample points uniformly by triangle area and return positions, normals, face ids, and bary coords."""
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    if min_samples_per_face < 0:
        raise ValueError(f"min_samples_per_face must be non-negative, got {min_samples_per_face}")

    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    areas = _compute_surface_sampling_areas(vertices, faces, area_mode).astype(np.float64)
    positive_area = np.isfinite(areas) & (areas > 1e-12)
    positive_face_ids = np.flatnonzero(positive_area).astype(np.int64)
    total_area = float(np.sum(areas[positive_area], dtype=np.float64))
    if total_area <= 0.0:
        raise ValueError(f"Cannot sample a surface mesh with zero total {area_mode} triangle area")

    probabilities = np.zeros_like(areas, dtype=np.float64)
    probabilities[positive_area] = areas[positive_area] / total_area
    probability_sum = float(np.sum(probabilities, dtype=np.float64))
    if not np.isfinite(probability_sum) or probability_sum <= 0.0:
        raise ValueError("Cannot sample a surface mesh with invalid triangle-area probabilities")
    probabilities /= probability_sum
    rng = np.random.default_rng(seed)

    if min_samples_per_face > 0:
        required_samples = len(positive_face_ids) * min_samples_per_face
        if required_samples > num_samples:
            log.warning(
                "surface_init_num_gaussians=%d is too small for surface_init_min_per_face=%d; "
                "increasing to %d to cover %d positive-area faces.",
                num_samples,
                min_samples_per_face,
                required_samples,
                len(positive_face_ids),
            )
            num_samples = required_samples
        min_face_ids = np.repeat(positive_face_ids, min_samples_per_face)
        num_remaining = num_samples - required_samples
        if num_remaining > 0:
            extra_face_ids = rng.choice(len(faces), size=num_remaining, replace=True, p=probabilities).astype(np.int64)
            face_ids = np.concatenate([min_face_ids, extra_face_ids])
        else:
            face_ids = min_face_ids
        rng.shuffle(face_ids)
    else:
        face_ids = rng.choice(len(faces), size=num_samples, replace=True, p=probabilities).astype(np.int64)

    u = rng.random(len(face_ids)).astype(np.float32)
    v = rng.random(len(face_ids)).astype(np.float32)
    sqrt_u = np.sqrt(u)
    bary_coords = np.stack(
        [
            1.0 - sqrt_u,
            sqrt_u * (1.0 - v),
            sqrt_u * v,
        ],
        axis=1,
    ).astype(np.float32)

    face_vertices = vertices[faces[face_ids]]
    positions = np.sum(face_vertices * bary_coords[:, :, None], axis=1).astype(np.float32)
    face_normals = compute_face_normals(vertices, faces)
    normals = face_normals[face_ids].astype(np.float32)
    downward = normals[:, 2] < 0.0
    normals[downward] *= -1.0
    return positions, normals, face_ids, bary_coords


def _interpolate_vertex_colors_on_faces(
    vertex_colors: Optional[NDArray],
    faces: NDArray[np.int32],
    face_ids: NDArray[np.int64],
    bary_coords: NDArray[np.float32],
) -> Optional[NDArray[np.uint8]]:
    if vertex_colors is None:
        return None

    colors = np.asarray(vertex_colors)
    face_colors = colors[np.asarray(faces, dtype=np.int64)[face_ids]]
    interpolated = np.sum(face_colors.astype(np.float32) * bary_coords[:, :, None], axis=1)
    return np.clip(np.rint(interpolated), 0, 255).astype(np.uint8)


@dataclass(frozen=True)
class _BevColorGrid:
    bbox_min_xy: NDArray[np.float32]
    cell_size: float
    cell_indices: NDArray[np.int64]
    colors: NDArray[np.uint8]


def _build_bev_color_grid_from_vertices(
    vertices: NDArray[np.float32],
    colors: NDArray[np.uint8],
    cell_size: float,
) -> Optional[_BevColorGrid]:
    """Create a BEV color lookup grid from meshing cell-center vertices."""
    vertices = np.asarray(vertices, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    if len(vertices) == 0 or len(colors) == 0:
        return None
    if len(vertices) != len(colors):
        raise ValueError(f"vertices/colors length mismatch: {len(vertices)} vs {len(colors)}")
    if cell_size <= 0.0:
        raise ValueError(f"cell_size must be positive, got {cell_size}")

    bbox_min_xy = (np.min(vertices[:, :2], axis=0) - 0.5 * cell_size).astype(np.float32)
    cell_indices = np.rint((vertices[:, :2] - bbox_min_xy) / cell_size - 0.5).astype(np.int64)
    return _BevColorGrid(
        bbox_min_xy=bbox_min_xy,
        cell_size=float(cell_size),
        cell_indices=cell_indices,
        colors=colors[:, :3].astype(np.uint8, copy=False),
    )


def _lookup_bev_color_grid(
    points: NDArray[np.float32],
    color_grid: _BevColorGrid,
    neighbor_radius_cells: int = 1,
) -> tuple[NDArray[np.uint8], NDArray[np.bool_]]:
    """Look up BEV cell colors for points, falling back to fixed-radius neighboring cells."""
    points = np.asarray(points, dtype=np.float32)
    if neighbor_radius_cells < 0:
        raise ValueError(f"neighbor_radius_cells must be non-negative, got {neighbor_radius_cells}")

    color_by_cell = {tuple(cell.tolist()): color for cell, color in zip(color_grid.cell_indices, color_grid.colors)}
    query_cells = np.floor((points[:, :2] - color_grid.bbox_min_xy) / color_grid.cell_size).astype(np.int64)

    out_colors = np.zeros((len(points), 3), dtype=np.uint8)
    found_mask = np.zeros(len(points), dtype=bool)
    for i, cell in enumerate(query_cells):
        key = (int(cell[0]), int(cell[1]))
        exact_color = color_by_cell.get(key)
        if exact_color is not None:
            out_colors[i] = exact_color
            found_mask[i] = True
            continue

        if neighbor_radius_cells == 0:
            continue

        weighted_color = np.zeros(3, dtype=np.float64)
        total_weight = 0.0
        query_xy = points[i, :2].astype(np.float64)
        for dx in range(-neighbor_radius_cells, neighbor_radius_cells + 1):
            for dy in range(-neighbor_radius_cells, neighbor_radius_cells + 1):
                neighbor_key = (key[0] + dx, key[1] + dy)
                neighbor_color = color_by_cell.get(neighbor_key)
                if neighbor_color is None:
                    continue
                neighbor_center = (
                    color_grid.bbox_min_xy.astype(np.float64)
                    + (np.array(neighbor_key, dtype=np.float64) + 0.5) * color_grid.cell_size
                )
                dist2 = float(np.sum((query_xy - neighbor_center) ** 2))
                weight = 1.0 / max(dist2, 1.0e-6)
                weighted_color += neighbor_color.astype(np.float64) * weight
                total_weight += weight

        if total_weight > 0.0:
            out_colors[i] = np.clip(np.rint(weighted_color / total_weight), 0, 255).astype(np.uint8)
            found_mask[i] = True

    return out_colors, found_mask


def _format_face_sample_count_stats(face_ids: NDArray[np.int64], num_faces: int) -> str:
    face_ids = np.asarray(face_ids, dtype=np.int64)
    counts = np.bincount(face_ids, minlength=num_faces)
    zero_faces = int(np.sum(counts == 0))
    occupied_faces = int(num_faces - zero_faces)
    percentiles = np.percentile(counts, [50, 90, 99]) if num_faces > 0 else np.zeros(3)
    mean_count = float(np.mean(counts)) if num_faces > 0 else 0.0
    max_count = int(np.max(counts)) if num_faces > 0 else 0
    occupied_pct = 100.0 * occupied_faces / max(num_faces, 1)
    zero_pct = 100.0 * zero_faces / max(num_faces, 1)
    return (
        f"faces={num_faces}, occupied={occupied_faces} ({occupied_pct:.2f}%), "
        f"zero={zero_faces} ({zero_pct:.2f}%), mean={mean_count:.2f}, "
        f"p50={percentiles[0]:.1f}, p90={percentiles[1]:.1f}, "
        f"p99={percentiles[2]:.1f}, max={max_count}"
    )


class LidarGroundMeshRoadInitialization(BaseInitialization):
    """Initialize road Gaussians from a reconstructed colored ground mesh.

    Builds a configured-topology ground mesh from LiDAR point clouds at init time,
    using the same reconstruction pipeline as the ground-mesh artifact export.
    Mesh vertices supply Gaussian positions and (optionally) per-vertex RGB
    colors; face normals orient Gaussians to lie flat on the road surface.
    """

    def _build_ground_mesh_init_state(self, ncore_ds: NCOREDataSource) -> tuple[NDArray, NDArray, NDArray | None]:
        from apps.av.road_mesh.ground_mesh import (
            get_nominal_ground_point_under_lidar,
            reconstruct_ground_mesh_from_points,
        )

        log.info(f"[{self.__class__.__name__}] Building colored ground mesh for road initialization …")
        ground_mesh_settings = _lidar_ground_mesh_settings(self.config)
        respect_rgb_label_validity = ground_mesh_settings.color_type == "camera-rgb-frame-valid-pixels"

        def build_ground_mesh() -> tuple[NDArray[np.float32], NDArray[np.int32], "NDArray[np.uint8] | None"]:
            lidar_ids = list(ground_mesh_settings.lidar_ids) if ground_mesh_settings.lidar_ids is not None else None
            if lidar_ids is not None and len(lidar_ids) != 1:
                raise ValueError("Ground mesh reconstruction supports exactly one LiDAR sensor")
            if respect_rgb_label_validity and ground_mesh_settings.camera_ids:
                raise ValueError("Camera-valid road coloring requires a LiDAR-only ground point cloud")

            # Always request colors for initialization: projected camera RGB improves SH starting values
            # vs random init. The artifact export path has a separate `colored` flag that controls the
            # output format, which is a different concern.
            pc_gen = ncore_ds.get_point_clouds(
                device=torch.device("cpu"),
                lidar_ids=lidar_ids,
                camera_ids=list(ground_mesh_settings.camera_ids) if ground_mesh_settings.camera_ids else None,
                valid_points_only=ground_mesh_settings.valid_points_only,
                color_type=ground_mesh_settings.color_type,
                non_dynamic_points_only=ground_mesh_settings.non_dynamic_points_only,
                step_frame=ground_mesh_settings.step_frame,
            )

            smooth_vertices, triangles, _, _, _, _, _, vertex_colors, _ = reconstruct_ground_mesh_from_points(
                point_cloud_generator=pc_gen,
                min_ray_length=ground_mesh_settings.min_ray_length,
                ground_control_point=get_nominal_ground_point_under_lidar(
                    ncore_ds,
                    rig_z_offset=ground_mesh_settings.z_offset,
                    lidar_id=lidar_ids[0] if lidar_ids is not None else None,
                ),
                ground_compat_max_distance=ground_mesh_settings.ground_compat_max_distance,
                ground_compat_max_angle_deg=ground_mesh_settings.ground_compat_max_angle_deg,
                num_plane_hypotheses=ground_mesh_settings.num_plane_hypotheses,
                plane_max_distance=ground_mesh_settings.plane_max_distance,
                plane_max_angle_deg=ground_mesh_settings.plane_max_angle_deg,
                plane_min_eval_points=ground_mesh_settings.plane_min_eval_points,
                plane_max_eval_points=ground_mesh_settings.plane_max_eval_points,
                plane_min_inliers=ground_mesh_settings.plane_min_inliers,
                enable_plane_extension=ground_mesh_settings.enable_plane_extension,
                voxel_size=ground_mesh_settings.voxel_size,
                min_points_per_voxel=ground_mesh_settings.min_points_per_voxel,
                smoothing_passes=ground_mesh_settings.smoothing_passes,
                grid_hole_fill_min_neighbors=ground_mesh_settings.grid_hole_fill_min_neighbors,
                points_to_world_transf=ncore_ds.world_to_nurec.inverse(),
                verbose=False,
                mesh_topology=ground_mesh_settings.mesh_topology,
                smoothing_method=ground_mesh_settings.smoothing_method,
                height_statistic=ground_mesh_settings.height_statistic,
                color_statistic=ground_mesh_settings.color_statistic,
                outlier_height_threshold=ground_mesh_settings.outlier_height_threshold,
                outlier_neighbor_radius=ground_mesh_settings.outlier_neighbor_radius,
                outlier_min_neighbors=ground_mesh_settings.outlier_min_neighbors,
                max_triangle_edge_length=ground_mesh_settings.max_triangle_edge_length,
                max_triangle_normal_angle_deg=ground_mesh_settings.max_triangle_normal_angle_deg,
                respect_rgb_label_validity=respect_rgb_label_validity,
                spin_mesh_device=ground_mesh_settings.spin_mesh_device,
                # Parallelize the per-spin loop in DominantPlaneRoadSegmenter. ``-1``
                # (default) auto-picks a worker count; ``0`` or ``1`` keeps the legacy
                # serial path; ``N > 1`` uses exactly N workers. See
                # ``_resolve_num_workers`` for the auto-pick rule.
                num_workers=_resolve_num_workers(ground_mesh_settings.num_workers),
                cuboid_tracks=ncore_ds.get_cuboid_tracks(world_frame=True)
                if ground_mesh_settings.extend_to_cuboids.enabled
                else None,
                cuboids_base_margin=ground_mesh_settings.extend_to_cuboids.base_margin,
                cuboids_exclude_points_radius=ground_mesh_settings.extend_to_cuboids.exclude_points_radius,
                cuboids_exclude_points_knns=ground_mesh_settings.extend_to_cuboids.exclude_points_knns,
                cuboid_point_merge_cell_size=ground_mesh_settings.extend_to_cuboids.cuboid_point_merge_cell_size,
            )
            return smooth_vertices, triangles, vertex_colors

        smooth_vertices, triangles, vertex_colors = build_ground_mesh()
        self.ground_mesh_artifact = Mesh(vertices=smooth_vertices, faces=triangles, colors=vertex_colors)
        self.ground_mesh_artifact_settings = ground_mesh_settings

        log.info(
            f"[{self.__class__.__name__}] Ground mesh: {len(smooth_vertices)} vertices, {len(triangles)} triangles, "
            f"colors={'yes' if vertex_colors is not None else 'no'}"
        )

        # Transform mesh vertices from world space back to NuRec space
        positions_nurec = ncore_ds.world_to_nurec.transform_points(smooth_vertices)
        normals_world = compute_vertex_normals(smooth_vertices, triangles)
        normals_nurec = normals_world @ ncore_ds.world_to_nurec.matrix[:3, :3].T
        normals_nurec = normals_nurec / np.maximum(np.linalg.norm(normals_nurec, axis=1, keepdims=True), 1e-8)

        if self.debug_dir is not None:
            from nurec.utils.io.ply import save_ply

            mesh_path = os.path.join(self.debug_dir, "ground_mesh_road_init.ply")
            save_ply(
                mesh_path,
                positions_nurec,
                triangles,
                normals=normals_nurec,
                colors=vertex_colors,
                logger=log,
            )

        positions_nurec = np.asarray(positions_nurec, dtype=np.float32)
        normals_nurec = np.asarray(normals_nurec, dtype=np.float32)
        if vertex_colors is not None:
            vertex_colors = np.asarray(vertex_colors, dtype=np.uint8)

        return positions_nurec, normals_nurec, vertex_colors

    def initialize_from_datasource(self, datasource: DataSourceSummary, **kwargs) -> None:
        ncore_ds = datasource.datasource
        assert isinstance(ncore_ds, NCOREDataSource), f"[{self.__class__.__name__}] Only NCOREDataSource is supported"

        num_random = self.config.get("num_random_points", None)
        assert num_random is None or num_random == 0, (
            f"[{self.__class__.__name__}] does not support random points (got num_random_points={num_random}). "
            f"Set num_random_points to 0 or remove it from the config."
        )

        # Build the road mesh once on rank 0, then share the CPU init state with all ranks.
        positions_nurec, normals_nurec, vertex_colors = sync_objects_or_raise(
            lambda: self._build_ground_mesh_init_state(ncore_ds)
        )

        positions = torch.from_numpy(positions_nurec).float().to(self.device)

        # Build flat-disk scales
        default_scale = self.config.get("default_scale", [0.1, 0.1, 0.001])
        if isinstance(default_scale, (list, ListConfig)):
            scale_tensor = torch.tensor(default_scale, device=self.device, dtype=torch.float32)
            observation_scale = scale_tensor.unsqueeze(0).expand(positions.shape[0], -1)
        else:
            observation_scale = torch.full((positions.shape[0], 3), default_scale, device=self.device)

        normals_tensor = torch.from_numpy(normals_nurec).float().to(self.device)

        if isinstance(default_scale, (list, ListConfig)):
            effective_scale = [s * self.config.scale_multiplier for s in default_scale]
        else:
            effective_scale = default_scale * self.config.scale_multiplier
        log.info(
            f"[{self.__class__.__name__}] scale_multiplier={self.config.scale_multiplier}, "
            f"default_scale={list(default_scale)}, effective_scale={effective_scale}, "
            f"num_positions={positions.shape[0]}"
        )

        self.init_from_positions(
            positions=positions,
            normals=normals_tensor,
            observation_scale=observation_scale,
        )

        # Build a PointCloud so that init_additional_buffers_SH converts
        # mesh vertex colors via RGB2SH.
        mesh_color = None
        if vertex_colors is not None:
            mesh_color = torch.from_numpy(vertex_colors).to(self.device)
        point_cloud = PointCloud(
            xyz_start=positions,
            xyz_end=positions,
            color=mesh_color,
        )
        self.init_additional_buffers(point_cloud)


class CameraGroundMeshRoadInitialization(BaseInitialization):
    """Initialize road Gaussians from a ground mesh built from camera depth point clouds.

    Uses camera metric-depth point clouds with semantic road labels to build a
    ground mesh. Unlike the LiDAR-based ``LidarGroundMeshRoadInitialization``,
    this skips the LiDAR-specific normal estimation and RANSAC road segmentation,
    relying on camera semantic labels (``ROAD_SEMANTIC``) to identify road points
    and feeding them directly into ``DelaunayElevationMeshingAlgorithm``.
    """

    def _load_mesh_from_file(self, mesh_path: str) -> tuple[NDArray, NDArray, NDArray | None]:
        """Load a ground mesh PLY and return (vertices, faces, vertex_colors_or_None).

        The mesh is expected to be in **world space**.  Per-vertex normals in the
        file (if any) are ignored — normals are always recomputed from faces.
        """
        log.info(f"[{self.__class__.__name__}] Loading ground mesh from {mesh_path}")
        mesh_data = pcu.load_triangle_mesh(mesh_path)
        vertices = mesh_data.vertex_data.positions.astype(np.float32)
        faces = mesh_data.face_data.vertex_ids.astype(np.int32)

        vertex_colors = None
        raw_colors = getattr(mesh_data.vertex_data, "colors", None)
        if raw_colors is not None:
            vertex_colors = raw_colors[:, :3].astype(np.uint8)

        log.info(
            f"[{self.__class__.__name__}] Loaded mesh: {len(vertices)} vertices, {len(faces)} faces, "
            f"colors={'yes' if vertex_colors is not None else 'no'}"
        )
        return vertices, faces, vertex_colors

    def _build_mesh_from_fused_point_cloud(
        self, ncore_ds: NCOREDataSource
    ) -> tuple[NDArray[np.float32], NDArray[np.int32], "NDArray[np.uint8] | None", Optional[_BevColorGrid]]:
        """Build a ground mesh from a fused point cloud PLY with segment_id.

        Loads the accumulated point cloud, filters road points by semantic class
        (``segment_id``), then runs Delaunay elevation meshing with voxel-based
        color averaging.

        Returns ``(smooth_vertices, triangles, vertex_colors)`` in **world space**.
        """
        from apps.av.road_mesh.ground_mesh import DelaunayElevationMeshingAlgorithm

        point_cloud_path = self.config.point_cloud_path
        log.info(f"[{self.__class__.__name__}] Loading fused point cloud from {point_cloud_path}")

        ply_data = pcu.load_triangle_mesh(point_cloud_path).vertex_data
        ply_metadata = ply_data.custom_attributes

        xyz_world = ply_data.positions.astype(np.float32)
        color = ply_data.colors
        assert "segment_id" in ply_metadata, f"[{self.__class__.__name__}] segment_id is not present in the point cloud"
        segment_id = ply_metadata["segment_id"]

        log.info(
            f"[{self.__class__.__name__}] Loaded {len(xyz_world)} points, colors={'yes' if color is not None else 'no'}"
        )

        stuff_classes = unpack_optional(ncore_ds.get_semantic_classes_map(camera_semantics=True, lidar_semantics=False))
        assert self.labels_to_use, (
            f"[{self.__class__.__name__}] labels_to_use (from layer class_labels) must be set "
            f"to identify road points in the fused point cloud"
        )
        road_mask = np.zeros(len(xyz_world), dtype=bool)
        for label in self.labels_to_use:
            if label in stuff_classes:
                road_mask |= segment_id == stuff_classes[label]
                log.info(f"[{self.__class__.__name__}] Label '{label}' → class_id={stuff_classes[label]}")
            else:
                log.warning(f"[{self.__class__.__name__}] Label '{label}' not found in semantic classes")

        road_points = xyz_world[road_mask]
        road_colors = None
        if color is not None:
            road_colors = color[road_mask][:, :3].astype(np.uint8)

        log.info(
            f"[{self.__class__.__name__}] Filtered {len(road_points)} road points from {len(xyz_world)} total, "
            f"colors={'yes' if road_colors is not None else 'no'}"
        )

        if len(road_points) < 3:
            raise ValueError(
                f"[{self.__class__.__name__}] Too few road points ({len(road_points)}) to mesh. "
                f"Check point_cloud_path and semantic labels."
            )

        meshing = DelaunayElevationMeshingAlgorithm(
            enable_downsampling=True,
            voxel_size=self.config.voxel_size,
            min_points_per_voxel=self.config.min_points_per_voxel,
            smoothing_passes=self.config.smoothing_passes,
            elevation_axis=2,
            mesh_topology=self.config.mesh_topology,
            smoothing_method=self.config.smoothing_method,
            local_plane_min_neighbors=self.config.local_plane_min_neighbors,
            local_plane_include_self=self.config.local_plane_include_self,
            height_statistic=self.config.height_statistic,
            color_statistic=self.config.color_statistic,
            outlier_height_threshold=self.config.outlier_height_threshold,
            outlier_neighbor_radius=self.config.outlier_neighbor_radius,
            outlier_min_neighbors=self.config.outlier_min_neighbors,
            max_triangle_edge_length=self.config.max_triangle_edge_length,
            max_triangle_normal_angle_deg=self.config.max_triangle_normal_angle_deg,
        )
        smooth_vertices, triangles, mesh_vertices, vertex_colors = meshing.build_mesh_from_points(
            road_points, colors=road_colors
        )

        color_grid = None
        if (
            vertex_colors is not None
            and str(self.config.get("surface_init_color_mode", "vertex_blend")) == "cell_lookup"
        ):
            color_grid = _build_bev_color_grid_from_vertices(
                mesh_vertices,
                vertex_colors,
                cell_size=float(self.config.voxel_size),
            )
            log.info(
                "[%s] Road color cell lookup: %s",
                self.__class__.__name__,
                "disabled (no cells after filtering)" if color_grid is None else f"cells={len(color_grid.colors)}",
            )

        return smooth_vertices, triangles, vertex_colors, color_grid

    def _resolve_surface_init_num_gaussians(
        self,
        mesh_vertices_nurec: NDArray[np.float32],
        triangles: NDArray[np.int32],
        area_mode: str,
    ) -> int:
        requested_count = self.config.get("surface_init_num_gaussians", None)
        requested_density = self.config.get("surface_init_density_per_m2", None)
        if requested_count is not None and requested_density is not None:
            raise ValueError(
                f"[{self.__class__.__name__}] Only one of surface_init_num_gaussians and "
                f"surface_init_density_per_m2 can be set"
            )

        if requested_density is not None:
            density = float(requested_density)
            if density <= 0.0:
                raise ValueError(
                    f"[{self.__class__.__name__}] surface_init_density_per_m2 must be positive, got {density}"
                )
            area = float(np.sum(_compute_surface_sampling_areas(mesh_vertices_nurec, triangles, area_mode)))
            count = int(np.ceil(area * density))
        else:
            face_multiplier = float(self.config.get("surface_init_face_multiplier", 4.0))
            if face_multiplier <= 0.0:
                raise ValueError(
                    f"[{self.__class__.__name__}] surface_init_face_multiplier must be positive, got {face_multiplier}"
                )
            face_cap = int(np.ceil(len(triangles) * face_multiplier))
            if requested_count is None:
                count = face_cap
                log.info(
                    "[%s] Road surface init default count: mesh_faces=%d, "
                    "surface_init_face_multiplier=%.3g, resolved_count=%d",
                    self.__class__.__name__,
                    len(triangles),
                    face_multiplier,
                    count,
                )
            else:
                count = int(requested_count)
                if count <= 0:
                    raise ValueError(
                        f"[{self.__class__.__name__}] surface_init_num_gaussians must be positive, got {count}"
                    )
                count = min(count, face_cap)
                log.info(
                    "[%s] Road surface init requested count: surface_init_num_gaussians=%d, "
                    "mesh_faces=%d, surface_init_face_multiplier=%.3g, face_cap=%d, resolved_count=%d",
                    self.__class__.__name__,
                    int(requested_count),
                    len(triangles),
                    face_multiplier,
                    face_cap,
                    count,
                )

        if count <= 0:
            raise ValueError(f"[{self.__class__.__name__}] surface init Gaussian count must be positive, got {count}")
        return count

    def _prepare_road_gaussian_init_state(
        self,
        mesh_vertices_nurec: NDArray,
        triangles: NDArray,
        mesh_vertex_normals_nurec: NDArray,
        vertex_colors: Optional[NDArray],
        color_grid: Optional[_BevColorGrid] = None,
        nurec_to_color_frame: Optional[FrameConversion] = None,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], Optional[NDArray[np.uint8]]]:
        """Create camera-road Gaussian init arrays for the selected mesh placement mode."""
        mesh_vertices_nurec = np.asarray(mesh_vertices_nurec, dtype=np.float32)
        triangles = np.asarray(triangles, dtype=np.int32)
        mesh_vertex_normals_nurec = np.asarray(mesh_vertex_normals_nurec, dtype=np.float32)
        if vertex_colors is not None:
            vertex_colors = np.asarray(vertex_colors, dtype=np.uint8)

        mode = str(self.config.get("surface_init_mode", "face_area_random"))
        if mode == "vertex":
            log.info(
                "[%s] Road surface init: mode=vertex, num_gaussians=%d, mesh_faces=%d",
                self.__class__.__name__,
                len(mesh_vertices_nurec),
                len(triangles),
            )
            return mesh_vertices_nurec, mesh_vertex_normals_nurec, vertex_colors

        if mode != "face_area_random":
            raise ValueError(
                f"[{self.__class__.__name__}] Unsupported surface_init_mode={mode!r}. "
                f"Expected 'vertex' or 'face_area_random'."
            )

        color_mode = str(self.config.get("surface_init_color_mode", "vertex_blend"))
        if color_mode not in ("vertex_blend", "cell_lookup"):
            raise ValueError(
                f"[{self.__class__.__name__}] Unsupported surface_init_color_mode={color_mode!r}. "
                f"Expected 'vertex_blend' or 'cell_lookup'."
            )

        area_mode = str(self.config.get("surface_init_area_mode", "projected"))
        min_samples_per_face = int(self.config.get("surface_init_min_per_face", 0))
        num_gaussians = self._resolve_surface_init_num_gaussians(
            mesh_vertices_nurec=mesh_vertices_nurec,
            triangles=triangles,
            area_mode=area_mode,
        )
        seed = self.config.get("surface_init_seed", 42)
        positions_nurec, normals_nurec, face_ids, bary_coords = _sample_mesh_surface_points(
            mesh_vertices_nurec,
            triangles,
            num_samples=num_gaussians,
            seed=None if seed is None else int(seed),
            area_mode=area_mode,
            min_samples_per_face=min_samples_per_face,
        )
        vertex_blend_colors = _interpolate_vertex_colors_on_faces(vertex_colors, triangles, face_ids, bary_coords)
        gaussian_colors = vertex_blend_colors
        if color_mode == "cell_lookup":
            if color_grid is None:
                log.info(
                    "[%s] surface_init_color_mode=cell_lookup requested but no color grid is available; "
                    "falling back to vertex_blend.",
                    self.__class__.__name__,
                )
            else:
                color_positions = positions_nurec
                if nurec_to_color_frame is not None:
                    color_positions = nurec_to_color_frame.transform_points(positions_nurec)
                lookup_colors, found_mask = _lookup_bev_color_grid(
                    color_positions,
                    color_grid,
                    neighbor_radius_cells=int(self.config.get("surface_init_color_neighbor_radius_cells", 1)),
                )
                gaussian_colors = lookup_colors
                n_missing = int((~found_mask).sum())
                if n_missing > 0:
                    if vertex_blend_colors is not None:
                        gaussian_colors[~found_mask] = vertex_blend_colors[~found_mask]
                    else:
                        gaussian_colors = None
                    log.warning(
                        "[%s] Road surface color cell lookup missed %d / %d samples; %s.",
                        self.__class__.__name__,
                        n_missing,
                        len(found_mask),
                        "used vertex_blend fallback" if vertex_blend_colors is not None else "using random color init",
                    )
                else:
                    log.info(
                        "[%s] Road surface color cell lookup covered all %d sampled Gaussians.",
                        self.__class__.__name__,
                        len(found_mask),
                    )

        log.info(
            "[%s] Road surface init: mode=face_area_random, requested_num_gaussians=%d, "
            "final_num_gaussians=%d, mesh_vertices=%d, mesh_faces=%d, color_mode=%s, "
            "area_mode=%s, min_per_face=%d, seed=%s",
            self.__class__.__name__,
            num_gaussians,
            len(face_ids),
            len(mesh_vertices_nurec),
            len(triangles),
            color_mode,
            area_mode,
            min_samples_per_face,
            seed,
        )
        log.info(
            "[%s] Road surface init face sample stats: %s",
            self.__class__.__name__,
            _format_face_sample_count_stats(face_ids, len(triangles)),
        )
        return positions_nurec, normals_nurec, gaussian_colors

    def _build_camera_ground_mesh_init_state(
        self, ncore_ds: NCOREDataSource
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], Optional[NDArray[np.uint8]]]:
        """Build and sample the camera road mesh init state in NuRec space."""
        color_grid = None
        mesh_path = self.config.get("mesh_path", None)
        if mesh_path is not None:
            smooth_vertices, triangles, vertex_colors = self._load_mesh_from_file(mesh_path)
        else:
            assert self.config.get("point_cloud_path", None) is not None, (
                f"[{self.__class__.__name__}] Either mesh_path or point_cloud_path must be set"
            )
            smooth_vertices, triangles, vertex_colors, color_grid = self._build_mesh_from_fused_point_cloud(ncore_ds)

        if len(smooth_vertices) < 3 or len(triangles) == 0:
            raise ValueError(
                f"[{self.__class__.__name__}] Loaded mesh is degenerate "
                f"({len(smooth_vertices)} vertices, {len(triangles)} faces). "
                f"Check the input file."
            )

        log.info(
            f"[{self.__class__.__name__}] Ground mesh: {len(smooth_vertices)} vertices, {len(triangles)} triangles, "
            f"colors={'yes' if vertex_colors is not None else 'no'}"
        )

        positions_nurec = ncore_ds.world_to_nurec.transform_points(smooth_vertices)
        normals_world = compute_vertex_normals(smooth_vertices, triangles)
        normals_nurec = normals_world @ ncore_ds.world_to_nurec.matrix[:3, :3].T
        normals_nurec = normals_nurec / np.maximum(np.linalg.norm(normals_nurec, axis=1, keepdims=True), 1e-8)

        if self.debug_dir is not None:
            from nurec.utils.io.ply import save_ply

            debug_mesh_path = os.path.join(self.debug_dir, "camera_ground_mesh_road_init.ply")
            save_ply(
                debug_mesh_path,
                positions_nurec,
                triangles,
                normals=normals_nurec,
                colors=vertex_colors,
                logger=log,
            )

        positions_nurec = np.asarray(positions_nurec, dtype=np.float32)
        normals_nurec = np.asarray(normals_nurec, dtype=np.float32)
        if vertex_colors is not None:
            vertex_colors = np.asarray(vertex_colors, dtype=np.uint8)

        positions_nurec, normals_nurec, vertex_colors = self._prepare_road_gaussian_init_state(
            positions_nurec,
            triangles,
            normals_nurec,
            vertex_colors,
            color_grid=color_grid,
            nurec_to_color_frame=ncore_ds.world_to_nurec.inverse() if color_grid is not None else None,
        )
        positions_nurec = np.asarray(positions_nurec, dtype=np.float32)
        normals_nurec = np.asarray(normals_nurec, dtype=np.float32)
        if vertex_colors is not None:
            vertex_colors = np.asarray(vertex_colors, dtype=np.uint8)
        return positions_nurec, normals_nurec, vertex_colors

    def _init_road_gaussians_from_arrays(
        self,
        positions_nurec: NDArray,
        normals_nurec: NDArray,
        vertex_colors: Optional[NDArray],
    ) -> None:
        positions = torch.from_numpy(np.asarray(positions_nurec, dtype=np.float32)).float().to(self.device)
        normals_tensor = torch.from_numpy(np.asarray(normals_nurec, dtype=np.float32)).float().to(self.device)

        default_scale = self.config.get("default_scale", [0.1, 0.1, 0.001])
        if isinstance(default_scale, (list, ListConfig)):
            scale_tensor = torch.tensor(default_scale, device=self.device, dtype=torch.float32)
        else:
            scale_tensor = torch.full((3,), default_scale, device=self.device, dtype=torch.float32)
        observation_scale = scale_tensor.unsqueeze(0).expand(positions.shape[0], -1)

        effective_scale = (scale_tensor * self.config.scale_multiplier).tolist()
        log.info(
            f"[{self.__class__.__name__}] scale_multiplier={self.config.scale_multiplier}, "
            f"default_scale={scale_tensor.tolist()}, effective_scale={effective_scale}, "
            f"num_positions={positions.shape[0]}"
        )

        self.init_from_positions(
            positions=positions,
            normals=normals_tensor,
            observation_scale=observation_scale,
        )

        mesh_color = None
        if vertex_colors is not None:
            mesh_color = torch.from_numpy(np.asarray(vertex_colors, dtype=np.uint8)).to(self.device)
        point_cloud = PointCloud(
            xyz_start=positions,
            xyz_end=positions,
            color=mesh_color,
        )
        self.init_additional_buffers(point_cloud)

    def initialize_from_datasource(self, datasource: DataSourceSummary, **kwargs) -> None:
        ncore_ds = datasource.datasource
        assert isinstance(ncore_ds, NCOREDataSource), f"[{self.__class__.__name__}] Only NCOREDataSource is supported"

        num_random = self.config.get("num_random_points", None)
        assert num_random is None or num_random == 0, (
            f"[{self.__class__.__name__}] does not support random points (got num_random_points={num_random}). "
            f"Set num_random_points to 0 or remove it from the config."
        )

        positions_nurec, normals_nurec, vertex_colors = sync_objects_or_raise(
            lambda: self._build_camera_ground_mesh_init_state(ncore_ds)
        )
        self._init_road_gaussians_from_arrays(positions_nurec, normals_nurec, vertex_colors)


BaseInitialization.register_to_gaussian_initialization_factory("random", RandomInitialization)
BaseInitialization.register_to_gaussian_initialization_factory(
    "construction-barriers", ConstructionBarriersInitialization
)
BaseInitialization.register_to_gaussian_initialization_factory("lidar-construction", LidarConstructionInitialization)
BaseInitialization.register_to_gaussian_initialization_factory("sfm-point-cloud", SFMPointCloudInitialization)
BaseInitialization.register_to_gaussian_initialization_factory("lidar-rig-trajectory", LidarRigTrajInitialization)
BaseInitialization.register_to_gaussian_initialization_factory(
    "lidar-rig-trajectory-road", LidarRigTrajRoadInitialization
)
BaseInitialization.register_to_gaussian_initialization_factory("lidar-dynamic-tracks", DynamicTracksInitialization)
BaseInitialization.register_to_gaussian_initialization_factory("camera-dynamic-tracks", DynamicTracksInitialization)
BaseInitialization.register_to_gaussian_initialization_factory(
    "accumulated-point-cloud", AccumulatedPointCloudInitialization
)
BaseInitialization.register_to_gaussian_initialization_factory(
    "accumulated-point-cloud-road", AccumulatedPointCloudRoadInitialization
)
BaseInitialization.register_to_gaussian_initialization_factory("instant_nurec-ply", InstantNuRecPlyInitialization)
BaseInitialization.register_to_gaussian_initialization_factory(
    "instant_nurec-road-ply", InstantNuRecPlyRoadInitialization
)
BaseInitialization.register_to_gaussian_initialization_factory(
    "in-process-instant_nurec", InProcessInstantNuRecInitialization
)
BaseInitialization.register_to_gaussian_initialization_factory(
    "in-process-instant_nurec-road", InProcessInstantNuRecRoadInitialization
)
BaseInitialization.register_to_gaussian_initialization_factory("ply", PlyInitialization)
BaseInitialization.register_to_gaussian_initialization_factory("ply-tracks", PlyTracksInitialization)
BaseInitialization.register_to_gaussian_initialization_factory(
    "lidar-ground-mesh-road", LidarGroundMeshRoadInitialization
)
BaseInitialization.register_to_gaussian_initialization_factory(
    "camera-ground-mesh-road", CameraGroundMeshRoadInitialization
)
