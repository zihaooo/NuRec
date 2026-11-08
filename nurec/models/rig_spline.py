# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packed, locally evaluated B-splines for rig trajectory calibration."""

from __future__ import annotations

import math

import torch

from torch import nn


def rotation_vector_to_matrix(rotation_vector: torch.Tensor) -> torch.Tensor:
    """Convert rotation vectors in radians to rotation matrices with a stable exponential map."""
    assert rotation_vector.shape[-1] == 3
    x, y, z = rotation_vector.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    skew = torch.stack(
        [zeros, -z, y, z, zeros, -x, -y, x, zeros],
        dim=-1,
    ).reshape(*rotation_vector.shape[:-1], 3, 3)

    theta = torch.linalg.vector_norm(rotation_vector, dim=-1)
    # sinc is defined at zero, avoiding the NaN gradients produced by a masked sin(theta) / theta branch.
    a = torch.sinc(theta / math.pi)
    b = 0.5 * torch.sinc(theta / (2.0 * math.pi)).square()
    identity = torch.eye(3, dtype=rotation_vector.dtype, device=rotation_vector.device)
    identity = identity.expand(*rotation_vector.shape[:-1], 3, 3)
    return identity + a[..., None, None] * skew + b[..., None, None] * (skew @ skew)


def residual_vectors_to_transforms(residuals: torch.Tensor) -> torch.Tensor:
    """Convert ``[..., translation_m, rotation_vector_rad]`` controls to SE(3) transforms."""
    assert residuals.shape[-1] == 6
    transforms = (
        torch.eye(4, dtype=residuals.dtype, device=residuals.device).expand(*residuals.shape[:-1], 4, 4).clone()
    )
    transforms[..., :3, :3] = rotation_vector_to_matrix(residuals[..., 3:])
    transforms[..., :3, 3] = residuals[..., :3]
    return transforms


class CameraExtrinsicResiduals(nn.Module):
    """Static camera residuals with a mean-zero gauge and trajectory-shared radial bounds.

    Each row is a translation in metres followed by a rotation vector in radians. Centering
    and bounding are evaluated from the raw parameter on every forward pass. A single scale
    is applied to every translation (and separately every rotation) in a trajectory, so the
    radial bounds cannot disturb the projected mean-zero gauge.
    """

    def __init__(
        self,
        trajectory_indices: torch.Tensor,
        *,
        num_trajectories: int,
        max_translation_m: float,
        max_rotation_rad: float,
    ) -> None:
        super().__init__()
        if trajectory_indices.dtype != torch.int64:
            raise TypeError("camera trajectory indices must use int64")
        if trajectory_indices.ndim != 1:
            raise ValueError("camera trajectory indices must be one-dimensional")
        if trajectory_indices.numel() == 0:
            raise ValueError("camera extrinsic calibration requires at least one camera")
        if num_trajectories <= 0:
            raise ValueError("camera extrinsic calibration requires at least one trajectory")
        if torch.any((trajectory_indices < 0) | (trajectory_indices >= num_trajectories)):
            raise IndexError("camera trajectory index is out of range")
        if not math.isfinite(max_translation_m) or max_translation_m <= 0.0:
            raise ValueError("maximum translation must be finite and positive")
        if not math.isfinite(max_rotation_rad) or max_rotation_rad <= 0.0:
            raise ValueError("maximum rotation must be finite and positive")

        for trajectory_idx in trajectory_indices.unique().tolist():
            camera_count = int((trajectory_indices == trajectory_idx).sum().item())
            if camera_count < 2:
                raise ValueError(
                    "mean-zero camera extrinsics require at least two cameras per trajectory; "
                    f"trajectory {trajectory_idx} has {camera_count}"
                )

        self.num_trajectories = num_trajectories
        self.max_translation_m = max_translation_m
        self.max_rotation_rad = max_rotation_rad
        self.trajectory_indices = nn.Buffer(trajectory_indices.detach().clone(), persistent=False)
        self.residuals = nn.Parameter(torch.zeros((trajectory_indices.numel(), 6), dtype=torch.float32))

    @staticmethod
    def _shared_radial_bound(vectors: torch.Tensor, maximum: float) -> torch.Tensor:
        maximum_norm = torch.linalg.vector_norm(vectors, dim=-1).max()
        maximum_tensor = vectors.new_tensor(maximum)
        # Leave one representable float of headroom so a subsequent norm reduction
        # cannot round a vector just above the configured hard bound.
        safe_maximum = torch.nextafter(maximum_tensor, torch.zeros_like(maximum_tensor))
        scale = (safe_maximum / maximum_norm.clamp_min(1e-12)).clamp(max=1.0)
        return vectors * scale

    def projected_residuals(self) -> torch.Tensor:
        """Return per-trajectory centered residuals before applying hard bounds."""
        projected = torch.zeros_like(self.residuals)
        for trajectory_idx in self.trajectory_indices.unique().tolist():
            row_indices = torch.nonzero(self.trajectory_indices == trajectory_idx, as_tuple=False).squeeze(-1)
            trajectory_residuals = self.residuals[row_indices]
            centered = trajectory_residuals - trajectory_residuals.mean(dim=0, keepdim=True)
            projected = projected.index_copy(0, row_indices, centered)
        return projected

    def bounded_residuals(self, projected: torch.Tensor | None = None) -> torch.Tensor:
        """Return gauge-projected residuals with shared translation and rotation scales.

        Pass ``projected`` to reuse an already computed gauge projection.
        """
        if projected is None:
            projected = self.projected_residuals()
        bounded = torch.zeros_like(projected)
        for trajectory_idx in self.trajectory_indices.unique().tolist():
            row_indices = torch.nonzero(self.trajectory_indices == trajectory_idx, as_tuple=False).squeeze(-1)
            trajectory_residuals = projected[row_indices]
            translation = self._shared_radial_bound(trajectory_residuals[:, :3], self.max_translation_m)
            rotation = self._shared_radial_bound(trajectory_residuals[:, 3:], self.max_rotation_rad)
            bounded = bounded.index_copy(0, row_indices, torch.cat([translation, rotation], dim=-1))
        return bounded

    def forward(self, camera_indices: torch.Tensor | int) -> torch.Tensor:
        return residual_vectors_to_transforms(self.bounded_residuals()[camera_indices])

    @torch.no_grad()
    def statistics(self) -> dict[str, torch.Tensor]:
        projected = self.projected_residuals()
        bounded = self.bounded_residuals(projected)
        raw_translation_norm = torch.linalg.vector_norm(projected[:, :3], dim=-1)
        raw_rotation_norm = torch.linalg.vector_norm(projected[:, 3:], dim=-1)
        stats: dict[str, torch.Tensor] = {}
        for trajectory_idx in self.trajectory_indices.unique().tolist():
            mask = self.trajectory_indices == trajectory_idx
            trajectory_mean = bounded[mask].mean(dim=0)
            stats[f"trajectory_{trajectory_idx}_translation_residual_mean_m"] = torch.linalg.vector_norm(
                trajectory_mean[:3]
            )
            stats[f"trajectory_{trajectory_idx}_rotation_residual_mean_rad"] = torch.linalg.vector_norm(
                trajectory_mean[3:]
            )
        for camera_idx in range(self.residuals.shape[0]):
            stats[f"camera_{camera_idx}_translation_norm_m"] = torch.linalg.vector_norm(bounded[camera_idx, :3])
            stats[f"camera_{camera_idx}_rotation_norm_rad"] = torch.linalg.vector_norm(bounded[camera_idx, 3:])
            stats[f"camera_{camera_idx}_translation_bound_saturation"] = (
                raw_translation_norm[camera_idx] >= self.max_translation_m
            ).to(self.residuals.dtype)
            stats[f"camera_{camera_idx}_rotation_bound_saturation"] = (
                raw_rotation_norm[camera_idx] >= self.max_rotation_rad
            ).to(self.residuals.dtype)
        return stats


class PackedRigSpline(nn.Module):
    """Open clamped uniform B-splines packed across independent rig trajectories.

    Controls use translation in metres followed by a rotation vector in radians. Dataset-derived
    origins, durations, and packed offsets are non-persistent buffers: only ``controls`` is saved in
    checkpoints, so knots are reconstructed from the current dataset and configuration.
    """

    def __init__(
        self,
        trajectory_start_timestamps_us: torch.Tensor,
        trajectory_end_timestamps_us: torch.Tensor,
        *,
        degree: int,
        knot_spacing_seconds: float,
        skip_first_control: bool,
        max_translation_m: float,
        max_rotation_rad: float,
    ) -> None:
        super().__init__()
        if trajectory_start_timestamps_us.dtype != torch.int64:
            raise TypeError("trajectory start timestamps must use int64 microseconds")
        if trajectory_end_timestamps_us.dtype != torch.int64:
            raise TypeError("trajectory end timestamps must use int64 microseconds")
        if trajectory_start_timestamps_us.ndim != 1 or trajectory_end_timestamps_us.ndim != 1:
            raise ValueError("trajectory timestamp bounds must be one-dimensional")
        if trajectory_start_timestamps_us.shape != trajectory_end_timestamps_us.shape:
            raise ValueError("trajectory start and end timestamp tensors must have the same shape")
        if trajectory_start_timestamps_us.numel() == 0:
            raise ValueError("at least one trajectory is required")
        if not 1 <= degree <= 5:
            raise ValueError(f"spline degree must be in [1, 5], got {degree}")
        if not math.isfinite(knot_spacing_seconds) or knot_spacing_seconds <= 0.0:
            raise ValueError("knot spacing must be finite and positive")
        if not math.isfinite(max_translation_m) or max_translation_m <= 0.0:
            raise ValueError("maximum translation must be finite and positive")
        if not math.isfinite(max_rotation_rad) or max_rotation_rad <= 0.0:
            raise ValueError("maximum rotation must be finite and positive")
        if torch.any(trajectory_end_timestamps_us < trajectory_start_timestamps_us):
            raise ValueError("trajectory end timestamps must not precede start timestamps")

        spacing_us = round(knot_spacing_seconds * 1_000_000.0)
        if spacing_us <= 0 or not math.isclose(
            spacing_us / 1_000_000.0, knot_spacing_seconds, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError("knot spacing must be representable as an integer number of microseconds")

        starts = trajectory_start_timestamps_us.detach().clone()
        durations_us = trajectory_end_timestamps_us - starts
        span_counts = torch.div(durations_us + spacing_us - 1, spacing_us, rounding_mode="floor").clamp_min(1)
        control_counts = span_counts + degree
        control_offsets = torch.cat(
            [torch.zeros(1, dtype=torch.int64, device=starts.device), control_counts.cumsum(dim=0)]
        )

        self.degree = degree
        self.knot_spacing_seconds = knot_spacing_seconds
        self.knot_spacing_us = spacing_us
        self.skip_first_control = skip_first_control
        self.max_translation_m = max_translation_m
        self.max_rotation_rad = max_rotation_rad

        self.trajectory_origins_us = nn.Buffer(starts, persistent=False)
        self.trajectory_durations_us = nn.Buffer(durations_us, persistent=False)
        self.span_counts = nn.Buffer(span_counts, persistent=False)
        self.control_offsets = nn.Buffer(control_offsets, persistent=False)
        self.controls = nn.Parameter(torch.zeros((int(control_offsets[-1].item()), 6), dtype=torch.float32))

        anchor_mask = torch.ones(self.controls.shape[0], dtype=torch.float32, device=starts.device)
        if skip_first_control:
            anchor_mask[control_offsets[:-1]] = 0.0
        self.anchor_mask = nn.Buffer(anchor_mask, persistent=False)
        # mcity multiclip (2026-11-08): the controls this run optimises (set_active_controls); all by default.
        self.active_controls = nn.Buffer(
            torch.ones(self.controls.shape[0], dtype=torch.bool, device=starts.device), persistent=False
        )
        self._active_hook = None
        self._initial_controls: torch.Tensor | None = None  # anchor_loss's reference, taken at its first call

    @property
    def num_trajectories(self) -> int:
        return int(self.trajectory_origins_us.numel())

    @property
    def control_counts(self) -> torch.Tensor:
        return self.control_offsets[1:] - self.control_offsets[:-1]

    def _apply_radial_bounds(self, controls: torch.Tensor) -> torch.Tensor:
        translation, rotation = controls.split(3, dim=-1)
        translation_norm = torch.linalg.vector_norm(translation, dim=-1, keepdim=True).clamp_min(1e-12)
        rotation_norm = torch.linalg.vector_norm(rotation, dim=-1, keepdim=True).clamp_min(1e-12)
        translation = translation * (self.max_translation_m / translation_norm).clamp(max=1.0)
        rotation = rotation * (self.max_rotation_rad / rotation_norm).clamp(max=1.0)
        return torch.cat([translation, rotation], dim=-1)

    def bounded_controls(self) -> torch.Tensor:
        """Return radially bounded controls with per-trajectory anchors applied exactly."""
        return self._apply_radial_bounds(self.controls) * self.anchor_mask[:, None]

    def _knot_value(self, trajectory_indices: torch.Tensor, knot_indices: torch.Tensor) -> torch.Tensor:
        """Evaluate packed open-uniform knots without materializing a global knot vector."""
        value = (knot_indices - self.degree).to(self.controls.dtype) * self.knot_spacing_seconds
        end = self.span_counts[trajectory_indices].to(self.controls.dtype) * self.knot_spacing_seconds
        return torch.minimum(value.clamp_min(0.0), end)

    def _local_basis_and_control_indices(
        self, timestamps_us: torch.Tensor, trajectory_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        original_shape = timestamps_us.shape
        timestamps_flat = timestamps_us.reshape(-1)
        trajectory_flat = trajectory_indices.reshape(-1).to(torch.int64)
        if timestamps_flat.dtype != torch.int64:
            raise TypeError("spline timestamps must use int64 microseconds")
        if timestamps_flat.shape != trajectory_flat.shape:
            raise ValueError("timestamps and trajectory indices must have matching shapes")
        if torch.any((trajectory_flat < 0) | (trajectory_flat >= self.num_trajectories)):
            raise IndexError("trajectory index is out of range")

        # Subtract in int64 before conversion: epoch-scale timestamps otherwise lose sub-second precision.
        delta_us = timestamps_flat - self.trajectory_origins_us[trajectory_flat]
        # Queries outside the observed frame-start/end range hold the endpoint correction.
        # The final uniform knot span may extend beyond that endpoint when the clip duration
        # is not an exact multiple of the configured spacing.
        max_us = self.trajectory_durations_us[trajectory_flat]
        delta_us = torch.minimum(delta_us.clamp_min(0), max_us)
        u = delta_us.to(self.controls.dtype) / 1_000_000.0

        local_span = torch.div(delta_us, self.knot_spacing_us, rounding_mode="floor")
        local_span = torch.minimum(local_span, self.span_counts[trajectory_flat] - 1) + self.degree

        # Algorithm A2.2 from The NURBS Book, vectorized over queries. It computes only p+1
        # basis values and gathers only the p+1 locally supported controls.
        basis = torch.zeros((timestamps_flat.numel(), self.degree + 1), dtype=self.controls.dtype, device=u.device)
        basis[:, 0] = 1.0
        left = torch.zeros_like(basis)
        right = torch.zeros_like(basis)
        for j in range(1, self.degree + 1):
            left[:, j] = u - self._knot_value(trajectory_flat, local_span + 1 - j)
            right[:, j] = self._knot_value(trajectory_flat, local_span + j) - u
            saved = torch.zeros_like(u)
            for r in range(j):
                denominator = right[:, r + 1] + left[:, j - r]
                term = torch.where(denominator != 0.0, basis[:, r] / denominator, 0.0)
                basis[:, r] = saved + right[:, r + 1] * term
                saved = left[:, j - r] * term
            basis[:, j] = saved

        local_indices = (
            local_span[:, None] - self.degree + torch.arange(self.degree + 1, dtype=torch.int64, device=u.device)
        )
        packed_indices = self.control_offsets[trajectory_flat, None] + local_indices
        return basis.reshape(*original_shape, self.degree + 1), packed_indices.reshape(*original_shape, self.degree + 1)

    def evaluate(self, timestamps_us: torch.Tensor, trajectory_indices: torch.Tensor | int) -> torch.Tensor:
        """Evaluate residual vectors at arbitrary batched timestamps with endpoint-clamped extrapolation."""
        if isinstance(trajectory_indices, int):
            trajectory_indices = torch.full_like(timestamps_us, trajectory_indices, dtype=torch.int64)
        else:
            trajectory_indices = torch.broadcast_to(trajectory_indices, timestamps_us.shape)
        basis, packed_indices = self._local_basis_and_control_indices(timestamps_us, trajectory_indices)
        # Bounds and anchoring are applied after the local gather so evaluation touches only p+1
        # controls per timestamp, independent of the full trajectory length.
        controls = self._apply_radial_bounds(self.controls[packed_indices])
        controls = controls * self.anchor_mask[packed_indices][..., None]
        return torch.sum(basis[..., None] * controls, dim=-2)

    def forward(self, timestamps_us: torch.Tensor, trajectory_indices: torch.Tensor | int) -> torch.Tensor:
        return residual_vectors_to_transforms(self.evaluate(timestamps_us, trajectory_indices))

    def control_mask_for(self, timestamps_us: torch.Tensor, trajectory_idx: int) -> torch.Tensor:
        """Controls with non-zero basis weight at any of the timestamps (int64 microseconds) of one trajectory."""
        mask = torch.zeros(self.controls.shape[0], dtype=torch.bool, device=self.controls.device)
        if timestamps_us.numel() == 0:
            return mask
        timestamps = timestamps_us.reshape(-1).to(device=self.controls.device, dtype=torch.int64)
        trajectories = torch.full_like(timestamps, trajectory_idx)
        basis, packed_indices = self._local_basis_and_control_indices(timestamps, trajectories)
        mask[packed_indices[basis > 0.0]] = True
        return mask

    def set_active_controls(self, mask: torch.Tensor) -> None:
        """Optimise only these controls: the others get no gradient and are left out of the curvature mean."""
        if mask.shape != (self.controls.shape[0],):
            raise ValueError(f"active control mask has shape {tuple(mask.shape)}, expected ({self.controls.shape[0]},)")
        self.active_controls.copy_(mask.to(self.active_controls.device, torch.bool))
        if self._active_hook is None:
            self._active_hook = self.controls.register_hook(
                lambda grad: grad * self.active_controls[:, None].to(grad.dtype)
            )

    def anchor_loss(self, *, translation_normalization_m: float, rotation_normalization_rad: float) -> torch.Tensor:
        """Group-L1 deviation of the active controls from their values at the first call (the loaded trajectory)."""
        bounded = self.bounded_controls()
        if self._initial_controls is None or self._initial_controls.shape != bounded.shape:
            self._initial_controls = bounded.detach().clone()
        active = self.active_controls
        if not bool(active.any()):
            return bounded.sum() * 0.0
        deviation = (bounded - self._initial_controls.to(bounded.device))[active]
        return (
            torch.linalg.vector_norm(deviation[:, :3], dim=-1).mean() / translation_normalization_m
            + torch.linalg.vector_norm(deviation[:, 3:], dim=-1).mean() / rotation_normalization_rad
        )

    def _active_second_difference_mask(self) -> torch.Tensor:
        """Per second difference (as second_differences orders them): its centre control is active."""
        masks: list[torch.Tensor] = []
        for trajectory_idx in range(self.num_trajectories):
            begin = int(self.control_offsets[trajectory_idx].item())
            end = int(self.control_offsets[trajectory_idx + 1].item())
            if end - begin < 3:
                continue
            masks.append(self.active_controls[begin + 1 : end - 1])
        if not masks:
            return torch.zeros(0, dtype=torch.bool, device=self.controls.device)
        return torch.cat(masks)

    def second_differences(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return translation/rotation second control differences in m/s² and rad/s²."""
        controls = self.bounded_controls()
        translation_curvatures: list[torch.Tensor] = []
        rotation_curvatures: list[torch.Tensor] = []
        for trajectory_idx in range(self.num_trajectories):
            begin = int(self.control_offsets[trajectory_idx].item())
            end = int(self.control_offsets[trajectory_idx + 1].item())
            trajectory_controls = controls[begin:end]
            if trajectory_controls.shape[0] < 3:
                continue
            second_difference = torch.diff(trajectory_controls, n=2, dim=0) / self.knot_spacing_seconds**2
            translation_curvatures.append(second_difference[:, :3])
            rotation_curvatures.append(second_difference[:, 3:])
        if not translation_curvatures:
            empty = controls.new_empty((0, 3))
            return empty, empty
        return torch.cat(translation_curvatures), torch.cat(rotation_curvatures)

    def group_l1_curvature_loss(
        self,
        *,
        translation_normalization_m_s2: float,
        rotation_normalization_rad_s2: float,
    ) -> torch.Tensor:
        """Return rotationally invariant group-L1 curvature in physical units."""
        if translation_normalization_m_s2 <= 0.0 or rotation_normalization_rad_s2 <= 0.0:
            raise ValueError("curvature normalizations must be positive")
        translation, rotation = self.second_differences()
        zero = self.controls.sum() * 0.0
        if not bool(self.active_controls.all()):  # only the active part: a run of a subset of the dataset's frames
            active = self._active_second_difference_mask()
            translation, rotation = translation[active], rotation[active]
        translation_loss = (
            torch.linalg.vector_norm(translation, dim=-1).mean() / translation_normalization_m_s2
            if translation.numel()
            else zero
        )
        rotation_loss = (
            torch.linalg.vector_norm(rotation, dim=-1).mean() / rotation_normalization_rad_s2
            if rotation.numel()
            else zero
        )
        return translation_loss + rotation_loss

    @torch.no_grad()
    def statistics(self) -> dict[str, torch.Tensor]:
        bounded = self.bounded_controls()
        translation_norm = torch.linalg.vector_norm(bounded[:, :3], dim=-1)
        rotation_norm = torch.linalg.vector_norm(bounded[:, 3:], dim=-1)
        raw_translation_norm = torch.linalg.vector_norm(self.controls[:, :3], dim=-1)
        raw_rotation_norm = torch.linalg.vector_norm(self.controls[:, 3:], dim=-1)
        translation_curvature, rotation_curvature = self.second_differences()
        zero = self.controls.new_zeros(())
        return {
            "control_count": self.controls.new_tensor(float(self.controls.shape[0])),
            "translation_mean_m": translation_norm.mean(),
            "translation_max_m": translation_norm.max(),
            "rotation_mean_rad": rotation_norm.mean(),
            "rotation_max_rad": rotation_norm.max(),
            "translation_curvature_mean_m_s2": torch.linalg.vector_norm(translation_curvature, dim=-1).mean()
            if translation_curvature.numel()
            else zero,
            "rotation_curvature_mean_rad_s2": torch.linalg.vector_norm(rotation_curvature, dim=-1).mean()
            if rotation_curvature.numel()
            else zero,
            "translation_bound_saturation": (raw_translation_norm >= self.max_translation_m).float().mean(),
            "rotation_bound_saturation": (raw_rotation_norm >= self.max_rotation_rad).float().mean(),
        }
