# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import math

from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Callable, List, Literal, NamedTuple, Optional, Type, TypedDict, cast

import gsplat  # type: ignore[import-untyped]
import numpy as np
import torch

from ncore.data import (
    BivariateWindshieldModelParameters,
    ConcreteCameraModelParametersUnion,
    ConcreteExternalDistortionParametersUnion,
    ConcreteLidarModelParametersUnion,
    FThetaCameraModelParameters,
    OpenCVFisheyeCameraModelParameters,
    OpenCVPinholeCameraModelParameters,
    ReferencePolynomial,
    RowOffsetStructuredSpinningLidarModelParameters,
    ShutterType,
)
from ncore.sensors import CameraModel, camera_model_from_parameters
from nurec.config.base_schema import resolve_per_sensor
from nurec.config.model import (
    BaseRendererConfig,
    GSplatRendererConfig,
    SensorOutputConfig,
    TileSizeStrategy,
)
from nurec.libs.geometry.kernels.pose import se3pose_to_inverse_matrix
from nurec.libs.geometry.kernels.quaternion import quat_slerp
from nurec.models.base import BaseModel
from nurec.models.gaussians.segment_plan import SegmentPlan, assert_sparse_segments_supported
from nurec.utils.batch import FrameMeta, RenderingData
from nurec.utils.cached_device_tensor import CachedDeviceTensorPool, bounded_cache
from nurec.utils.geometry import tquat_to_se3_matrix
from nurec.utils.misc import get_union_types, unpack_optional
from nurec.utils.ncore_types import require_concrete_external_distortion_parameters
from nurec.utils.prober import get_global_prober
from nurec.utils.profiling import ScopedTimer
from nurec.utils.renamed_keys import adopt_renamed_key
from nurec.utils.tile_mask import (
    CameraToLidarTileMapping,
    precompute_from_element_to_tile,
)
from nurec.utils.types import (
    ExtraSignal,
    GaussiansRenderReturn,
)


def _concatenate_extra_signals(parts: list[torch.Tensor]) -> torch.Tensor:
    """Return a lone signal without copying it, or concatenate multiple signals."""
    assert parts
    # Keeping the singleton alias is safe because gsplat's rasterization contract
    # treats extra_signals as read-only and never mutates its storage. It only
    # materializes a contiguous copy when the input layout requires one.
    # NuRec guards that contract against the pinned gsplat build via
    # test_single_camera_extra_signal_real_cuda_matches_cat_and_is_read_only
    # (contiguous + strided). The documented read-only note and gsplat-side
    # regression coverage land with gsplat !432; bump the MODULE.bazel pin
    # once that MR is on nv/main so the shipped ABI matches the docs.
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)


class _CamToLidarMappingCommonKwargs(TypedDict):
    """Backend-independent keyword arguments handed to the camera->lidar tile-mapping
    precompute entry point, ``precompute_from_element_to_tile``.

    Typing-only: annotating ``_camera_to_lidar_mapping_common_kwargs`` with this instead of
    ``dict[str, Any]`` makes the ``**``-splat call site checkable, so a key rename on either
    side is a type error rather than a runtime ``TypeError`` deep in a lidar render. The
    split into a shared, backend-independent part is what keeps a future backend's
    ``build_camera_to_lidar_tile_mapping`` down to its own lidar-grid description.
    """

    camera_model: CameraModel
    lidar_rays_world: torch.Tensor
    T_cam_world: torch.Tensor
    cam_tile_width: int
    cam_tile_height: int
    cam_frame_width: int
    cam_frame_height: int
    depth_shells: tuple[float, ...]
    ray_subsample: int
    n_pose_samples: int
    pose_jitter_translation_m: float
    pose_jitter_rotation_rad: float


class _ECADRenderState:
    """Per-step hand-off between the camera render, the `ecad_confidence` loss and the ECAD strategy.

    When enabled (by `ECADStrategy`), each training camera render appends four per-Gaussian channels to the
    rasterizer's extra signals:
      0. confidence exp(gamma), detached: its rendered map is the pixel confidence C(p);
      1. exp(gamma) - exp(gamma).detach(): renders to zero, but carries d/d gamma of a loss on C(p) without
         sending any gradient to colours or geometry (all channel values are zero);
      2. a zero probe whose gradient becomes sum_p b_i(p) e(p) for the loss's per-pixel SSIM error e(p);
      3. a zero probe whose gradient becomes sum_p b_i(p), the summed blending weight of each Gaussian.
    gamma is channel 0 of the collected `camera_extra_signal`. Single-process training only.
    """

    num_channels: int = 4

    def __init__(self) -> None:
        self.enabled = False
        self.clear()

    def clear(self) -> None:
        self.rendered: Optional[torch.Tensor] = None  # [H, W, 4], keeps the autograd graph for the loss
        self.ssim_probe: Optional[torch.Tensor] = None  # [N] zero leaf
        self.weight_probe: Optional[torch.Tensor] = None  # [N] zero leaf
        self.opacities: Optional[torch.Tensor] = None  # [N] activated opacities used by the render
        self.conics: Optional[torch.Tensor] = None  # [N, 3] inverse 2D covariance (pixels)
        self.radii: Optional[torch.Tensor] = None  # [N] largest projected radius (pixels), 0 = culled
        self.ray_origin: Optional[torch.Tensor] = None  # [3] camera centre of the render

    def attach(self, camera_extra_signal: torch.Tensor, extra_signals: Optional[torch.Tensor]) -> torch.Tensor:
        """Append the ECAD channels to the camera extra signals ([1, N, E] or None)."""
        self.clear()
        log_confidence = camera_extra_signal[:, 0]
        confidence = torch.exp(log_confidence.clamp(-10.0, 5.0))
        self.ssim_probe = torch.zeros_like(confidence.detach()).requires_grad_(True)
        self.weight_probe = torch.zeros_like(confidence.detach()).requires_grad_(True)
        channels = torch.stack(
            [confidence.detach(), confidence - confidence.detach(), self.ssim_probe, self.weight_probe], dim=-1
        ).unsqueeze(0)
        return channels if extra_signals is None else torch.cat([extra_signals, channels], dim=-1)

    def capture(
        self,
        rendered: torch.Tensor,
        meta: dict[str, Any],
        opacities: torch.Tensor,
        rendering_data: RenderingData,
    ) -> None:
        """Keep the rendered ECAD channels [H, W, 4] and the per-Gaussian projection the strategy needs."""
        self.rendered = rendered
        self.opacities = opacities.detach().reshape(-1)
        conics = meta.get("conics")
        self.conics = None if conics is None else conics.detach().reshape(-1, 3)
        radii = meta.get("radii")
        if radii is not None:
            radii = radii.detach()
            if radii.ndim > 1 and radii.shape[-1] == 2:
                radii = radii.amax(dim=-1)
            self.radii = radii.reshape(-1)
        self.ray_origin = rendering_data.rays[0].reshape(-1, 6)[0, :3].detach()


ECAD_RENDER_STATE = _ECADRenderState()


class BaseGaussianRenderer(ABC):
    def __init__(self, config: BaseRendererConfig, model: BaseModel) -> None:
        self.config = config
        # Latch for the one-shot "active tiling requested but no mask supplied" warning that
        # subclasses emit from render(); GaussiansComposite is the sole producer of the mask.
        self._warned_missing_active_tile_mask: bool = False

    def _warn_missing_active_tile_mask_once(self) -> None:
        """Emit the "active tiling is on but no mask arrived" warning at most once.

        The composite is the single owner of the active tile mask: it builds the mask at the
        tile size each renderer reports via ``active_tile_mask_tile_size`` and hands it over on
        ``rendering_data``. Renderers deliberately do NOT compute a fallback -- a second
        implementation is how the mask grid drifts off the kernel grid. Driven standalone (no
        composite), active tiling is simply inactive; say so once.

        Callers own the "should we warn at all" predicate (which sensor kind counts as a
        camera is backend-specific); this method owns the latch and the message.
        """
        if self._warned_missing_active_tile_mask:
            return
        self._warned_missing_active_tile_mask = True
        logging.getLogger(__name__).warning(
            "[tile_mask] active_tile_mask.enabled is set but no active_tile_mask was supplied on the "
            "rendering data; rendering every tile. The mask is produced by GaussiansComposite -- "
            f"standalone {type(self).__name__} callers must set rendering_data.active_tile_mask themselves."
        )

    @abstractmethod
    def render(
        self,
        rendering_data: RenderingData,
        gaussian_parameters: dict[str, torch.Tensor],
        n_active_features: int,
        extra_ray_signal_infos: tuple[list[str], list[int], list[Callable]],
        frame_meta: Optional[List[FrameMeta]] = None,
        segment_plan: SegmentPlan | None = None,
    ) -> GaussiansRenderReturn: ...

    def band0_fold_supported(
        self,
        *,
        is_camera: bool,
        features_dim: int,
        extra_signal_dim: int,
        sensor_extra_signal_dim: int,
    ) -> bool:
        """Would a render of this sensor honor a band-0 Fourier fold?

        The collector asks this before folding a holistic-Fourier layer out of the
        per-frame collect. Answering False means the layer is collected densely every
        frame instead, so a renderer that cannot honor the fold never renders a stale
        band-0. Renderers without fold support answer False and always get the dense
        collect.

        Args:
            is_camera: Which sensor the collect is for. Selects the per-sensor output
                gates and which extra-signal dimension applies.
            features_dim: Width of the collected `features` (albedo + specular).
            extra_signal_dim: Width of the collected sensor-agnostic `extra_signal`.
            sensor_extra_signal_dim: Width of this sensor's extra-signal tensor.
        """
        return False

    # --- Active-tiling geometry surface -------------------------------------------------
    # Everything the composite needs to know about a renderer's tile grid goes through
    # these two methods. Returning ``None`` from either means "this renderer does not
    # support active tiling"; the composite degrades (with a warning) instead of
    # dispatching on concrete renderer classes or reaching into backend internals.

    def active_tile_mask_tile_size(self, width: int, height: int) -> tuple[int, int] | None:
        """(tile_height, tile_width) an active tile mask must be built at so it lands on this
        renderer's tile grid for a ``width`` x ``height`` render, or ``None`` if the renderer
        has no tile grid / does not support active tiling. Subclasses that support active
        tiling override this and return a concrete ``tuple[int, int]``."""
        del width, height
        return None

    def build_camera_to_lidar_tile_mapping(
        self,
        rendering_data_cam: RenderingData,
        rendering_data_lidar: RenderingData,
        cam_tile_height: int,
        cam_tile_width: int,
    ) -> CameraToLidarTileMapping | None:
        """Build the camera→lidar tile mapping on *this* (lidar) renderer's tile grid.

        Called on the lidar renderer, which owns the lidar tile geometry the mask must land
        on. ``cam_tile_height/width`` come from the camera renderer's
        ``active_tile_mask_tile_size`` (they may differ from this renderer's own camera tile
        size when camera and lidar renderers differ).

        Returns ``None`` when the renderer cannot express its lidar tile grid; the composite
        then warns and disables lidar active tiling for that camera.
        """
        del rendering_data_cam, rendering_data_lidar, cam_tile_height, cam_tile_width
        return None

    def _camera_to_lidar_mapping_common_kwargs(
        self,
        rendering_data_cam: RenderingData,
        rendering_data_lidar: RenderingData,
        cam_tile_height: int,
        cam_tile_width: int,
    ) -> _CamToLidarMappingCommonKwargs:
        """Backend-independent arguments for the tile-mapping precompute.

        Used by a concrete ``build_camera_to_lidar_tile_mapping`` implementation, which adds its
        own lidar-tile-grid description on top (``element_to_tile`` + bin counts for
        ``precompute_from_element_to_tile``, the only entry point today). ``GSplatRenderer`` is
        currently the sole caller; this stays on the base as the shared half a future backend
        would reuse, alongside the two hooks above.

        Also carries the ``active_tile_mask.mapping_params`` knobs so any implementation stays
        wired to the same config surface.
        """
        cam_params = cast(ConcreteCameraModelParametersUnion, rendering_data_cam.sensor_model_parameters[0])
        device = rendering_data_cam.rays.device
        mapping_params = self.config.active_tile_mask.mapping_params
        return {
            "camera_model": camera_model_from_parameters(cam_params, device=device, dtype=torch.float32),
            # Use the actual lidar rays (H, W, 6) from the first batch element.
            "lidar_rays_world": rendering_data_lidar.rays[0],
            "T_cam_world": tquat_to_se3_matrix(rendering_data_cam.poses_tquat_startend[0, 0]).to(device),  # (4, 4)
            "cam_tile_width": cam_tile_width,
            "cam_tile_height": cam_tile_height,
            "cam_frame_width": rendering_data_cam.rays.shape[2],  # rays: (B, H, W, 6)
            "cam_frame_height": rendering_data_cam.rays.shape[1],
            "depth_shells": tuple(mapping_params.depth_shells),
            "ray_subsample": mapping_params.ray_subsample,
            "n_pose_samples": mapping_params.n_pose_samples,
            "pose_jitter_translation_m": mapping_params.pose_jitter_translation_m,
            "pose_jitter_rotation_rad": mapping_params.pose_jitter_rotation_rad,
        }

    def render_with_deferred_bp(
        self,
        rendering_data: RenderingData,
        gaussian_parameters: dict[str, torch.Tensor],
        n_active_features: int,
        extra_ray_signal_infos: tuple[list[str], list[int], list[Callable]] = ([], [], []),
        frame_meta: Optional[List[FrameMeta]] = None,
    ) -> GaussiansRenderReturn:
        """
        During training time, when multiple frames are rendered in a single feed-forward pass, the number of intermediate parameters needed to be cached for
        gradient computation can be very large (e.g. the model parameters in the renderer), resulting in large training memory usage.

        Deferred back-propogation is a memory-efficient way to compute backward gradient, first proposed in: **ARF: Artistic Radiance Fields** (https://arxiv.org/abs/2206.06360).
        The idea is to do a no_grad forward pass first. When gradients are needed, we re-do a grad_enabled forward pass followed immediately by a backward pass.
        In ARF, the above process is repeated for each patch separately, while for 3DGS, we do it for each image.
        Ref: GRM: https://github.com/camenduru/GRM/blob/68a7226e9f05a1bcee5d9624dc7336b06c4dd3ed/model/render/deferred_bp.py

        Implementation-wise, this is essentially the same as checkpointing. We just need to flatten/unflatten the dataclasses to list of tensors.
        """
        assert not (rendering_data.rays.requires_grad or rendering_data.poses_tquat_startend.requires_grad), (
            "Gradients w.r.t rendering data are not supported for now."
        )
        kwargs_keys: list[str] = list(gaussian_parameters.keys())

        def render_fn(*args: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
            parameters_dict = dict(zip(kwargs_keys, args))
            return self.render(
                rendering_data, parameters_dict, n_active_features, extra_ray_signal_infos, frame_meta
            ).to_tuple()

        ## when use_reentrant=False:
        # - The forward pass still runs with gradient enabled, but with a hook that replaces ctx.saved_tensors as "weakref" to save memory
        # of immediate activations.
        # - During backward pass, unpacking ctx.saved_tensor will trigger the function to run again until all needed saved_tensors are available.
        # - The function execution can end earlier (through throwing an _StopRecomputationError).
        # - However, some renderer implementations store too many tensors in ctx.undifferentiable_parameters and those are carried in the
        # forward pass, so memory usage there is still high. We have a new checkpoint-friendly implementation for this.
        ## when use_reentrant=True:
        # - During forward pass the function runs with torch.no_grad(), hence no computation graph is built, nor is ctx.undifferentiable_parameters kept,
        # so the memory is under control.
        # - Note that this deprecated though, and has limitations in e.g. torch.compile'd backward pass, or nested input tensors.
        # - Hence we need to "flatten" the input dataclasses into tensor tuples as arguments here.
        result = torch.utils.checkpoint.checkpoint(
            render_fn, *(gaussian_parameters[key] for key in kwargs_keys), use_reentrant=True
        )
        return GaussiansRenderReturn.from_tuple(result)

    @staticmethod
    def factory(name: str, config: BaseRendererConfig, model: BaseModel) -> BaseGaussianRenderer:
        """Factory method to create renderer instances.

        Available renderers:
        - GSplat variants:
          - 3dgs-gsplat: GSplat standard 3DGS (no distortion, pose gradients supported)
          - 3dgut-gsplat: GSplat with 3DGUT (distortion, rolling shutter; pose gradients require ray input, use_rays=True default)

        Args:
            name: Renderer variant name
            config: Renderer configuration
            model: Model instance

        Returns:
            BaseGaussianRenderer instance
        """
        variants: dict[str, Type[BaseGaussianRenderer]] = {
            # GSplat variants
            "3dgs-gsplat": GSplatRenderer,  # Standard 3DGS: with_ut=False, pose grads supported
            "3dgut-gsplat": GSplatRenderer,  # 3DGUT: with_ut=True; pose grads need ray input (use_rays=True, default)
        }
        if name not in variants:
            raise ValueError(f"Unknown renderer variant: {name}. Available variants: {list(variants.keys())}")
        return variants[name](config, model)

    def collect_profilings(self) -> dict[str, float]:
        return {}


class ExternalDistortionAdapter(ABC):
    """Abstract base class for external distortion adapters.

    Provides a unified interface for different external distortion models
    to simplify GSplat rendering code through polymorphism.
    """

    def __init__(self, external_distortion_params: ConcreteExternalDistortionParametersUnion, device: torch.device):
        self.external_distortion_params = external_distortion_params
        self.device = device

    @abstractmethod
    def get_external_distortion_name(self) -> gsplat.ExternalDistortionModelMeta:
        """Get external distortion name.

        Returns:
            External distortion name
        """
        pass

    @abstractmethod
    def get_distortion_params(self) -> gsplat.ExternalDistortionModelParameters:
        """Get distortion parameters for gsplat."""
        pass

    @staticmethod
    def factory(
        sensor_params: ConcreteExternalDistortionParametersUnion, device: torch.device
    ) -> ExternalDistortionAdapter:
        """Factory method to create appropriate external distortion adapter.

        Args:
            sensor_params: External distortion parameters
            device: Torch device

        Returns:
            External distortion adapter instance
        """
        if isinstance(sensor_params, BivariateWindshieldModelParameters):
            return BivariateWindshieldAdapter(sensor_params, device)
        else:
            raise ValueError(f"Unsupported external distortion model: {type(sensor_params)}")


class BivariateWindshieldAdapter(ExternalDistortionAdapter):
    """Adapter for bivariate windshield external distortion model."""

    def get_external_distortion_name(self) -> gsplat.ExternalDistortionModelMeta:
        return "bivariate-windshield"

    def get_distortion_params(self) -> gsplat.ExternalDistortionModelParameters:
        params = cast(BivariateWindshieldModelParameters, self.external_distortion_params)

        match params.reference_poly:
            case ReferencePolynomial.FORWARD:
                reference_poly = gsplat.rendering.ExternalDistortionReferencePolynomial.FORWARD
            case ReferencePolynomial.BACKWARD:
                reference_poly = gsplat.rendering.ExternalDistortionReferencePolynomial.BACKWARD
            case _:
                raise ValueError(f"Unsupported reference polynomial type: {params.reference_poly}")

        assert len(params.horizontal_poly) <= gsplat.rendering.BivariateWindshieldModelParameters.MAX_COEFFS
        assert len(params.vertical_poly) <= gsplat.rendering.BivariateWindshieldModelParameters.MAX_COEFFS
        assert len(params.horizontal_poly_inverse) <= gsplat.rendering.BivariateWindshieldModelParameters.MAX_COEFFS
        assert len(params.vertical_poly_inverse) <= gsplat.rendering.BivariateWindshieldModelParameters.MAX_COEFFS

        # NOTE: We explicitly set the device to "cpu" here becauses gsplat will create
        # std::array's on the host from the tensors to then send in constant cache of
        # the kernels. We don't want to copy the tensor to the GPU here because it would
        # mean a DtoH copy and then another HtoD copy back to the GPU to call the kernel,
        # effectively creating a needless round-trip copy.
        gsplat_params = gsplat.rendering.BivariateWindshieldModelParameters()
        gsplat_params.reference_poly = reference_poly  # type: ignore[attr-defined]
        gsplat_params.horizontal_poly = torch.tensor(  # type: ignore[attr-defined]
            params.horizontal_poly, dtype=torch.float32, device="cpu"
        )
        gsplat_params.vertical_poly = torch.tensor(  # type: ignore[attr-defined]
            params.vertical_poly, dtype=torch.float32, device="cpu"
        )
        gsplat_params.horizontal_poly_inverse = torch.tensor(  # type: ignore[attr-defined]
            params.horizontal_poly_inverse, dtype=torch.float32, device="cpu"
        )
        gsplat_params.vertical_poly_inverse = torch.tensor(  # type: ignore[attr-defined]
            params.vertical_poly_inverse, dtype=torch.float32, device="cpu"
        )
        return gsplat_params


class SensorModelParametersAdapter(ABC):
    """Abstract base class for camera model adapters.

    Provides a unified interface for different camera models (pinhole, fisheye, f-theta)
    to simplify GSplat rendering code through polymorphism.
    """

    def __init__(
        self,
        sensor_params: ConcreteCameraModelParametersUnion,
        device: torch.device,
        tensor_pool: CachedDeviceTensorPool,
    ):
        self.sensor_params = sensor_params
        self.device = device
        self.tensor_pool = tensor_pool

    @staticmethod
    def factory(
        sensor_params: ConcreteCameraModelParametersUnion,
        device: torch.device,
        tensor_pool: CachedDeviceTensorPool,
    ) -> SensorModelParametersAdapter:
        """Factory method to create appropriate camera adapter.

        Args:
            sensor_params: Camera model parameters
            device: Torch device
            tensor_pool: Configured cache for calibration tensors

        Returns:
            Camera adapter instance
        """
        if isinstance(sensor_params, OpenCVPinholeCameraModelParameters):
            return PinholeSensorParametersAdapter(sensor_params, device, tensor_pool)
        elif isinstance(sensor_params, OpenCVFisheyeCameraModelParameters):
            return FisheyeSensorParametersAdapter(sensor_params, device, tensor_pool)
        elif isinstance(sensor_params, FThetaCameraModelParameters):
            return FThetaSensorParametersAdapter(sensor_params, device, tensor_pool)
        else:
            raise TypeError(f"Unsupported camera model: {type(sensor_params)}")

    @abstractmethod
    def get_camera_model_name(self) -> gsplat.CameraModel:
        """Get gsplat camera model name."""
        pass

    @abstractmethod
    def get_gsplat_sensor_params(self) -> dict[str, Any]:
        """Get distortion parameters for gsplat."""
        pass


class PinholeSensorParametersAdapter(SensorModelParametersAdapter):
    """Adapter for OpenCV pinhole camera model."""

    def get_camera_model_name(self) -> gsplat.CameraModel:
        return "pinhole"

    def get_gsplat_sensor_params(self) -> dict[str, Any]:
        params = cast(OpenCVPinholeCameraModelParameters, self.sensor_params)
        radial = self.tensor_pool.get(tuple(float(value) for value in params.radial_coeffs), self.device).view(
            1, 1, -1
        )  # (1, 1, 6)
        tangential = self.tensor_pool.get(tuple(float(value) for value in params.tangential_coeffs), self.device).view(
            1, 1, -1
        )  # (1, 1, 2)

        fx, fy = params.focal_length[0], params.focal_length[1]
        cx, cy = params.principal_point[0], params.principal_point[1]
        Ks = self.tensor_pool.get(
            (float(fx), 0.0, float(cx), 0.0, float(fy), float(cy), 0.0, 0.0, 1.0), self.device
        ).view(1, 1, 3, 3)

        return {"radial_coeffs": radial, "tangential_coeffs": tangential, "Ks": Ks}


class FisheyeSensorParametersAdapter(SensorModelParametersAdapter):
    """Adapter for OpenCV fisheye camera model."""

    def get_camera_model_name(self) -> gsplat.CameraModel:
        return "fisheye"

    def get_gsplat_sensor_params(self) -> dict[str, Any]:
        params = cast(OpenCVFisheyeCameraModelParameters, self.sensor_params)
        radial = self.tensor_pool.get(tuple(float(value) for value in params.radial_coeffs), self.device).view(
            1, 1, -1
        )  # (1, 1, 4)

        fx, fy = params.focal_length[0], params.focal_length[1]
        cx, cy = params.principal_point[0], params.principal_point[1]
        Ks = self.tensor_pool.get(
            (float(fx), 0.0, float(cx), 0.0, float(fy), float(cy), 0.0, 0.0, 1.0), self.device
        ).view(1, 1, 3, 3)

        return {"radial_coeffs": radial, "Ks": Ks}


class FThetaSensorParametersAdapter(SensorModelParametersAdapter):
    """Adapter for F-Theta camera model."""

    def get_camera_model_name(self) -> gsplat.CameraModel:
        return "ftheta"

    def get_gsplat_sensor_params(self) -> dict[str, Any]:
        params = cast(FThetaCameraModelParameters, self.sensor_params)

        # Map polynomial type
        match params.reference_poly:
            case FThetaCameraModelParameters.PolynomialType.PIXELDIST_TO_ANGLE:
                reference_poly = gsplat.rendering.FThetaPolynomialType.PIXELDIST_TO_ANGLE
            case FThetaCameraModelParameters.PolynomialType.ANGLE_TO_PIXELDIST:
                reference_poly = gsplat.rendering.FThetaPolynomialType.ANGLE_TO_PIXELDIST
            case _:
                raise ValueError(f"Unsupported polynomial type: {params.reference_poly}")

        ftheta_params = gsplat.rendering.FThetaCameraDistortionParameters(
            reference_poly=reference_poly,
            pixeldist_to_angle_poly=tuple(params.pixeldist_to_angle_poly.tolist()),
            angle_to_pixeldist_poly=tuple(params.angle_to_pixeldist_poly.tolist()),
            max_angle=float(params.max_angle),
            linear_cde=tuple(params.linear_cde.tolist()),
        )

        # Extract focal length from first-order coefficient
        fx = params.angle_to_pixeldist_poly[1]
        fy = params.angle_to_pixeldist_poly[1]
        cx, cy = params.principal_point[0], params.principal_point[1]
        Ks = self.tensor_pool.get(
            (float(fx), 0.0, float(cx), 0.0, float(fy), float(cy), 0.0, 0.0, 1.0), self.device
        ).view(1, 1, 3, 3)

        return {"ftheta_coeffs": ftheta_params, "Ks": Ks}


class GSplatRenderer(BaseGaussianRenderer):
    """GSplat rendering implementation for Gaussian models.

    Implements the BaseGaussianRenderer interface using the gsplat library
    for 3D Gaussian Splatting with automatic differentiation via PyTorch autograd.
    """

    config: GSplatRendererConfig  # type: ignore[assignment] # Narrow type

    def __init__(self, config: GSplatRendererConfig, model: BaseModel) -> None:
        super().__init__(config, model)
        self.config = config

        self._camera_sensor_tensors = CachedDeviceTensorPool(
            maxsize=config.max_camera_tensor_cache_entries,
            name="gsplat_camera_sensor_tensors",
            overflow_message=(
                "Capacity is measured in entries keyed by tensor values, device, and inference mode. "
                "A distinct pinhole calibration uses up to 3 entries, fisheye 2, and FTheta 1; "
                "transformed intrinsics can add K entries, and normal/inference modes are distinct. "
                "Increase it with the Hydra override "
                "model.renderer.max_camera_tensor_cache_entries=<entries>."
            ),
        )
        self._lidar_geometry_tensors = CachedDeviceTensorPool(
            maxsize=config.max_lidar_geometry_cache_entries,
            name="gsplat_lidar_geometry_tensors",
            overflow_message=(
                "Capacity is measured in entries keyed by tensor values, device, and inference mode. "
                "A distinct LiDAR geometry uses up to 3 entries per device/mode. "
                "Increase it with the Hydra override "
                "model.renderer.max_lidar_geometry_cache_entries=<entries>."
            ),
        )
        self._lidar_frame_contexts = bounded_cache(
            maxsize=config.max_lidar_frame_context_cache_entries,
            name="gsplat_lidar_frame_contexts",
            overflow_message=(
                "Capacity is measured in entries keyed by LiDAR geometry, device, inference mode, "
                "and tiling settings; each distinct combination uses one entry. "
                "Increase it with the Hydra override "
                "model.renderer.max_lidar_frame_context_cache_entries=<entries>."
            ),
        )(partial(GSplatRenderer._build_lidar_frame_context, self._lidar_geometry_tensors))

        # Sibling latch for the one-shot "active tile mask supplied but packed mode is on"
        # warning: rendering still honors the mask, but mask-aware visibility is unavailable.
        self._warned_packed_active_tile_mask: bool = False
        # Same, for the "mask supplied, packed off, but gsplat returned no tiles_per_gauss"
        # third state: the meta key the narrowing reads went missing (gsplat pin drift).
        self._warned_missing_tiles_per_gauss: bool = False
        # Map unique_sensor_idx (int, from FrameMeta) -> lidar_id (str) so the
        # PerSensor config values (lidar_unstructured / cull / box_method) can be
        # resolved per sensor at render time. Populated by set_lidar_idx_to_id from
        # the datasource (which owns lidar_unique_ids); empty until then, in which
        # case only scalar (broadcast) PerSensor values are resolvable.
        self._lidar_idx_to_id: dict[int, str] = {}

        # Parse output settings into a per-sensor dict so a lidar variant can be
        # added alongside without touching this structure.
        def _parse_outputs(sensor_outputs: SensorOutputConfig) -> dict:
            return {
                "enable_features": sensor_outputs.enable_features,
                "enable_normals": sensor_outputs.enable_normals,
                "enable_distance": sensor_outputs.enable_distance,
                "enable_extended_features": sensor_outputs.enable_extended_features,
                "enable_sensor_extended_features": sensor_outputs.enable_sensor_features,
            }

        self._camera_outputs = _parse_outputs(config.outputs.camera)
        self._lidar_outputs = _parse_outputs(config.outputs.lidar)

        # Warn about 3DGUT pose gradient limitation
        if config.name == "3dgut-gsplat" and not config.use_rays:
            logging.getLogger(__name__).warning(
                "GSplatRenderer with 3DGUT and without ray input (use_rays=False) does NOT support "
                "gradients w.r.t. camera/track poses. Pose optimization workflows (DirectTracksCalib, "
                "FreePoseCalib) need ray input — set use_rays=True (the default for 3dgut-gsplat)."
            )

        # Validate and extract tile_size. RESOLUTION mode ignores the
        # configured tile dims (rasterization receives tile_size=None below),
        # so stale/asymmetric tile_width/tile_height in legacy configs must
        # not crash __init__. CUSTOM mode forwards self.tile_size verbatim
        # and gsplat requires square tiles.
        camera_tiling = config.tiling.camera
        self.tile_size = camera_tiling.tile_width
        if config.tiling.strategy == TileSizeStrategy.CUSTOM:
            assert self.tile_size == camera_tiling.tile_height, (
                f"GSplat requires square tiles, got "
                f"tile_width={self.tile_size}, tile_height={camera_tiling.tile_height}"
            )

        # Validate render mode if specified
        render_mode = config.render.mode
        if render_mode == "kbuffer":
            k_buffer_size = config.render.k_buffer_size
            assert k_buffer_size == 0, (
                f"GSplat Phase 1 only supports k_buffer_size=0 (equivalent to classic blending). "
                f"Got k_buffer_size={k_buffer_size}. See docs/gsplat_integration.md Phase 3.7 for k-buffer support."
            )

        # Storage for last rendering meta (for absgrad access by densification strategies)
        self.last_rendering_meta: Optional[dict] = None

        self.use_rays: bool = config.use_rays
        # Log once per process; composite creates camera + lidar renderer with same config
        _log_key = "_logged_rays" if self.use_rays else "_logged_sensor_model"
        if not getattr(GSplatRenderer, _log_key, False):
            if self.use_rays:
                logging.getLogger(__name__).info("GSplatRenderer: rasterization using rays as input")
            else:
                logging.getLogger(__name__).info("GSplatRenderer: rasterization using sensor model as input")
            setattr(GSplatRenderer, _log_key, True)  # type: ignore[attr-defined]

        # Load model config from serialized model dict (from serialize_to_json_dict).
        # Pre-rename artifacts and fixtures still use the nre_data envelope.
        model_dict = model.serialize_to_json_dict(with_state_dict=False)
        adopt_renamed_key(model_dict, "nurec_data", "nre_data")
        model_config = model_dict.get("nurec_data", {}).get("config", {})
        self._saturate_radiance: bool = model_config.get("saturate_radiance", True)
        self._has_post_processing: bool = len(getattr(model, "post_processings", ())) > 0

        # Load extra signal SH degrees from the model's particle config.
        # SH degrees must match across enabled signals because gsplat uses a single degree.
        particle_config = model_config.get("particle", {})
        extra_signal_dim = particle_config.get("extra_signal_dim", 0)
        camera_extra_signal_dim = particle_config.get("camera_extra_signal_dim", 0)

        extra_sph_degree: Optional[int] = (
            particle_config.get("extra_signal_sph_degree", 0) if extra_signal_dim > 0 else None
        )
        camera_extra_sph_degree: Optional[int] = (
            particle_config.get("camera_extra_signal_sph_degree", 0) if camera_extra_signal_dim > 0 else None
        )
        if extra_sph_degree is not None and camera_extra_sph_degree is not None:
            assert extra_sph_degree == camera_extra_sph_degree, (
                f"GSplat requires a single SH degree for all extra signals, but "
                f"extra_signal_sph_degree={extra_sph_degree} != camera_extra_signal_sph_degree={camera_extra_sph_degree}"
            )
        validated_degree = extra_sph_degree if extra_sph_degree is not None else camera_extra_sph_degree
        # gsplat uses None to mean "no SH evaluation" (degree 0)
        self._extra_signals_sh_degree: Optional[int] = validated_degree if validated_degree else None

        # Lidar extra signal SH degree (read from model config, same as camera path above).
        lidar_extra_signal_dim = particle_config.get("lidar_extra_signal_dim", 0)
        lidar_sph_degree: Optional[int] = (
            particle_config.get("lidar_extra_signal_sph_degree", 0) if lidar_extra_signal_dim > 0 else None
        )
        self._lidar_extra_signals_sh_degree: Optional[int] = lidar_sph_degree if lidar_sph_degree else None

    def band0_fold_supported(
        self,
        *,
        is_camera: bool,
        features_dim: int,
        extra_signal_dim: int,
        sensor_extra_signal_dim: int,
    ) -> bool:
        """Whether a render of this sensor reaches gsplat's fold-capable SH path.

        Single source of truth: render() calls this to decide whether to forward the
        fourier_* arguments, and the collector calls it to decide whether a
        holistic-Fourier layer may be folded out of the per-frame collect. Because
        both read the same predicate, a folded layer can never meet a render that
        drops the fold.

        Every term mirrors what render() derives below for the same sensor. All of
        them are functions of the renderer config and the collected tensor widths, so
        the answer is available at collect time.
        """
        use_3dgut = True if not is_camera else self.config.name == "3dgut-gsplat"
        # 3DGUT silently disables both, so the query must apply the same coercion or
        # it would answer False for a config the render actually supports.
        packed = False if use_3dgut else self.config.packed
        sparse_grad = False if use_3dgut else self.config.sparse_grad

        out = self._camera_outputs if is_camera else self._lidar_outputs
        # extra_signals reach gsplat as one concatenated tensor; the fold-capable SH
        # forward does not thread them.
        has_extra_signals = (out["enable_extended_features"] and extra_signal_dim > 0) or (
            out["enable_sensor_extended_features"] and sensor_extra_signal_dim > 0
        )
        # The fold rewrites band 0 of an SH color, so it needs a real SH render:
        # features must be enabled and carry more than one band (a 3-wide features
        # tensor is the post-activation RGB path, which gsplat renders with
        # sh_degree=None and never folds).
        renders_sh = out["enable_features"] and features_dim != 3

        return (
            use_3dgut
            and not packed
            and not sparse_grad
            and not has_extra_signals
            and renders_sh
            and not torch.is_grad_enabled()
        )

    @property
    def _use_3dgut(self) -> bool:
        """Whether this renderer runs gsplat's 3DGUT path (UT projection, rays, packed forced off)."""
        return self.config.name == "3dgut-gsplat"

    def active_tile_mask_tile_size(self, width: int, height: int) -> tuple[int, int]:
        """(tile_height, tile_width) the active tile mask must be built at so it lands on
        gsplat's tile grid for a width x height render. Mirrors gsplat.rasterization's own
        resolution: CUSTOM forwards the configured (square) tile size; RESOLUTION defers to
        gsplat's per-resolution pick (8 below 1080p, 16 at/above, on the eval3d/3DGUT path).
        Building the mask at this size is what lets active tiling work under either strategy
        without the caller pinning a tile size."""
        # `_resolve_tile_size` is a non-public gsplat symbol:
        # - comes from the pinned gsplat commit, not from gsplat's public API
        # - imported, not reimplemented, so the mask lands on gsplat's own resolved tile grid
        # - a pin bump that renames or reorders it breaks active tiling only
        from gsplat.rendering import _resolve_tile_size as _gsplat_resolve_tile_size

        requested = None if self.config.tiling.strategy == TileSizeStrategy.RESOLUTION else self.tile_size
        resolved = _gsplat_resolve_tile_size(requested, self._use_3dgut, width, height)
        return resolved, resolved

    def lidar_tiling_params(self) -> dict[str, int]:
        """LiDAR tiling parameters defining gsplat's spinning-lidar tile grid; used both to
        build the camera->lidar tile mapping for active tiling and by the lidar render path
        itself, so the two mirror by construction. This matters because histogram equalization
        derives ``n_bins_azimuth`` from ``max_pts_per_tile`` — a mismatch would yield a mask on
        the wrong tile grid."""
        lidar = self.config.tiling.lidar
        # RESOLUTION: gsplat's resolution fallback always picks tile=8 for lidar
        # (n_rows < 1080), which caps the in-tile element count at 8*8=64 via the
        # kernel's CTA=32, PPT=2 layout.
        # CUSTOM: trust the configured lidar tile dims.
        if self.config.tiling.strategy == TileSizeStrategy.RESOLUTION:
            max_pts_per_tile = 64
        else:
            max_pts_per_tile = lidar.tile_size_elevation * lidar.tile_size_azimuth
        return {
            "n_bins_elevation": lidar.n_bins_elevation,
            "max_pts_per_tile": max_pts_per_tile,
            "resolution_elevation": lidar.resolution_elevation,
            "densification_factor_azimuth": lidar.densification_factor_azimuth,
        }

    def compute_lidar_element_to_tile(
        self,
        ncore_params: RowOffsetStructuredSpinningLidarModelParameters,
        device: torch.device,
    ) -> tuple[torch.Tensor, int, int]:
        """Per-lidar-pixel gsplat tile id, shape ``(n_rows, n_cols)``, in gsplat's
        elevation-major layout (tile id = elevation * n_bins_azimuth + azimuth), plus
        ``(n_bins_azimuth, n_bins_elevation)``. Uses gsplat's own tile assignment
        (angles_to_tile_indices + histogram-equalized bins) so a camera->lidar mask built
        from it lands exactly on gsplat's lidar tile grid."""
        # `angles_to_tile_indices` is a non-public gsplat symbol, same convention as
        # `_resolve_tile_size` in `active_tile_mask_tile_size`: it comes from the pinned
        # gsplat commit, is imported rather than reimplemented so the mask lands on gsplat's
        # own tile assignment, and a pin bump that renames it breaks active tiling only.
        from gsplat.cuda._lidar import angles_to_tile_indices

        p = self.lidar_tiling_params()
        context = self._build_lidar_params(
            ncore_params,
            device=device,
            n_bins_elevation=p["n_bins_elevation"],
            max_pts_per_tile=p["max_pts_per_tile"],
            resolution_elevation=p["resolution_elevation"],
            densification_factor_azimuth=p["densification_factor_azimuth"],
        )
        ext = context.parameters
        tiling = ext.tiling
        elements = ext.create_elements()  # (n_rows * n_columns,) row-major
        angles = ext.elements_to_sensor_angles(elements)  # (n_rows * n_columns, 2) [az, el]
        tile_ids = angles_to_tile_indices(
            ext,
            angles,
            n_bins_azimuth=int(tiling.n_bins_azimuth),
            n_bins_elevation=int(tiling.n_bins_elevation),
            cdf_elevation=tiling.cdf_elevation,
        )
        element_to_tile = tile_ids.reshape(ext.n_rows, ext.n_columns).long()
        return element_to_tile, int(tiling.n_bins_azimuth), int(tiling.n_bins_elevation)

    def build_camera_to_lidar_tile_mapping(
        self,
        rendering_data_cam: RenderingData,
        rendering_data_lidar: RenderingData,
        cam_tile_height: int,
        cam_tile_width: int,
    ) -> CameraToLidarTileMapping | None:
        """Build the mapping on gsplat's own lidar tile grid (histogram-equalized azimuth
        bins, elevation-major tile ids). ``compute_lidar_element_to_tile`` uses gsplat's own
        tile assignment so the mask lands exactly on gsplat's intersect grid."""
        lidar_params = cast(
            RowOffsetStructuredSpinningLidarModelParameters, rendering_data_lidar.sensor_model_parameters[0]
        )
        device = rendering_data_cam.rays.device
        element_to_tile, n_bins_az, n_bins_el = self.compute_lidar_element_to_tile(lidar_params, device)
        return precompute_from_element_to_tile(
            **self._camera_to_lidar_mapping_common_kwargs(
                rendering_data_cam, rendering_data_lidar, cam_tile_height, cam_tile_width
            ),
            element_to_tile=element_to_tile,
            n_bins_azimuth=n_bins_az,
            n_bins_elevation=n_bins_el,
        )

    def set_lidar_idx_to_id(self, idx_to_id: dict[int, str]) -> None:
        """Provide the unique_sensor_idx -> lidar_id map used to resolve PerSensor
        renderer config (lidar_unstructured / cull / box_method) per sensor.

        Set by the composite model from the datasource's lidar_unique_ids after the
        renderer is constructed (the renderer factory does not receive the datasource).
        """
        self._lidar_idx_to_id = dict(idx_to_id)

    def _lidar_id_from_meta(self, frame_meta: Optional[List[FrameMeta]]) -> Optional[str]:
        """Resolve the lidar_id (str) for the current frame from its FrameMeta idx.

        Returns None when no frame_meta is supplied or the idx is unmapped; callers
        then rely on scalar (broadcast) PerSensor values (a dict value with no
        resolvable id raises in resolve_per_sensor, surfacing the misconfiguration).
        """
        if frame_meta is None or len(frame_meta) == 0:
            return None
        return self._lidar_idx_to_id.get(int(frame_meta[0].unique_sensor_idx))

    def renders_unstructured_lidar(self, frame_meta: Optional[List[FrameMeta]]) -> bool:
        """Whether this frame takes the model-free unstructured LiDAR path.

        The setting is per sensor, so only the renderer can answer it for a given
        frame. Callers that must know before they collect (the sparse-segment path
        does not support this renderer) ask here rather than reading the config.
        """
        lidar_id = self._lidar_id_from_meta(frame_meta)
        return bool(resolve_per_sensor(self.config.lidar_unstructured, lidar_id))

    def _get_unstructured_lidar_intensity(
        self,
        gaussian_parameters: dict[str, torch.Tensor],
        *,
        enable_extended_features: bool,
        enable_sensor_extended_features: bool,
        extra_ray_signal_infos: tuple[list[str], list[int], list[Callable]],
        means: Optional[torch.Tensor] = None,
        viewmat_midframe: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Return the per-Gaussian intensity channel for model-free LiDAR."""
        return self._get_unstructured_lidar_signal(
            "intensity",
            gaussian_parameters,
            enable_extended_features=enable_extended_features,
            enable_sensor_extended_features=enable_sensor_extended_features,
            extra_ray_signal_infos=extra_ray_signal_infos,
            means=means,
            viewmat_midframe=viewmat_midframe,
        )

    def _get_unstructured_lidar_signal(
        self,
        signal_name: str,
        gaussian_parameters: dict[str, torch.Tensor],
        *,
        enable_extended_features: bool,
        enable_sensor_extended_features: bool,
        extra_ray_signal_infos: tuple[list[str], list[int], list[Callable]],
        means: Optional[torch.Tensor] = None,
        viewmat_midframe: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Return the per-Gaussian raw channels for a named LiDAR extra signal.

        The model-free kernel transports flat per-Gaussian channels and composites
        them along each ray (the same as the structured path's extra-signal
        rasterization). For LiDAR SH extra signals, evaluate the coefficients at
        the mid-frame view direction first (matching the structured gsplat path).

        Returns the ``[N, signal_dim]`` RAW (pre-activation) per-Gaussian channels
        for ``signal_name``, or ``None`` if that signal is not configured / its
        feature gate is off. ``signal_dim`` is the signal's configured width (1 for
        intensity, 2 for raydrop) -- the ray-side activation is applied by the
        caller AFTER compositing, matching ExtraSignal.from_packed_tensor.
        """
        signal_names, signal_dims, _ = extra_ray_signal_infos
        if signal_name not in signal_names:
            return None
        signal_idx = signal_names.index(signal_name)

        # lidar_extra_ray_signal_infos are ordered as common signals first, then
        # lidar-specific signals. Split the output layout at the stored common
        # parameter width, rather than packing tensors behind output gates.
        common_extra_signal = gaussian_parameters.get("extra_signal")
        common_dim = 0 if common_extra_signal is None else common_extra_signal.shape[-1]
        common_count = 0
        common_output_dim = 0
        while common_count < len(signal_dims) and common_output_dim < common_dim:
            common_output_dim += signal_dims[common_count]
            common_count += 1
        if common_output_dim != common_dim:
            raise ValueError(
                "Cannot split common and LiDAR extra-signal layouts: "
                f"common_dim={common_dim}, infos={extra_ray_signal_infos[:2]}"
            )

        if signal_idx < common_count:
            if not enable_extended_features or common_extra_signal is None:
                return None
            signal_values = common_extra_signal
            signal_dims_for_source = signal_dims[:common_count]
            signal_idx_for_source = signal_idx
        else:
            if not enable_sensor_extended_features:
                return None
            lidar_extra_signal = gaussian_parameters.get("lidar_extra_signal")
            if lidar_extra_signal is None:
                return None
            signal_values = lidar_extra_signal
            signal_dims_for_source = signal_dims[common_count:]
            signal_idx_for_source = signal_idx - common_count

            if self._lidar_extra_signals_sh_degree is not None:
                if means is None or viewmat_midframe is None:
                    raise ValueError("LiDAR SH extra signal requires Gaussian means and a mid-frame viewmat.")
                sh_degree = self._lidar_extra_signals_sh_degree
                output_dim = sum(signal_dims_for_source)
                n_sh_coefficients = (sh_degree + 1) ** 2
                expected_dim = output_dim * n_sh_coefficients
                if signal_values.shape[-1] != expected_dim:
                    raise ValueError(
                        "LiDAR SH extra-signal layout does not match the configured output channels: "
                        f"expected={expected_dim}, got={signal_values.shape[-1]}, "
                        f"degree={sh_degree}, dims={signal_dims_for_source}"
                    )
                if means.shape[0] != signal_values.shape[0]:
                    raise ValueError(
                        "LiDAR SH extra signal requires one Gaussian mean per extra-signal row: "
                        f"means={tuple(means.shape)}, extra_signals={tuple(signal_values.shape)}"
                    )

                coefficients = signal_values.reshape(signal_values.shape[0], n_sh_coefficients, output_dim)
                camtoworld_midframe = torch.inverse(viewmat_midframe)
                dirs = means - camtoworld_midframe[:3, 3]
                # Match gsplat.rendering._maybe_evaluate_sh(..., clamp=False),
                # which structured LiDAR uses for extra signals.
                signal_values = gsplat.spherical_harmonics(sh_degree, dirs, coefficients) + 0.5

        if signal_values.shape[-1] != sum(signal_dims_for_source):
            raise ValueError(
                "Unstructured LiDAR extra-signal layout does not match its source channels: "
                f"expected={sum(signal_dims_for_source)}, got={signal_values.shape[-1]}, "
                f"dims={signal_dims_for_source}"
            )

        signal_dim = signal_dims_for_source[signal_idx_for_source]
        signal_start = sum(signal_dims_for_source[:signal_idx_for_source])
        return signal_values[:, signal_start : signal_start + signal_dim]

    def _render_unstructured(
        self,
        *,
        rendering_data: RenderingData,
        gaussian_params: Any,
        viewmat_start: torch.Tensor,
        viewmat_end: Optional[torch.Tensor],
        near_plane: float,
        far_plane: float,
        enable_distance: bool,
        intensity: Optional[torch.Tensor],
        raydrop_channels: Optional[torch.Tensor] = None,
        raydrop_activation: Optional[Callable] = None,
        intensity_activation: Optional[Callable] = None,
        device: torch.device,
        lidar_id: Optional[str] = None,
    ) -> GaussiansRenderReturn:
        """Model-free unstructured lidar render (config-gated by
        ``config.lidar_unstructured``).

        Builds an ``UnstructuredLidarFrameContext`` from the measured (N, 1) ray
        list + per-ray timestamps and renders via gsplat's
        ``render_unstructured`` (true ray geometry, no fitted row/col model).
        Returns range/opacity as a flat (n,) vector aligned 1:1 with the ray list.

        This method always renders and never returns None. Camera colors are
        irrelevant to LiDAR, so SH vs RGB-only camera models are both handled by
        dropping them. A flat LiDAR intensity channel, when enabled, is carried as
        a generic gsplat payload alongside the hit-range channel.

        Unlike the structured rasterization (which references its UT projection to
        the mid-frame pose), the unstructured renderer interpolates each ray's
        exact rolling-shutter pose between the FRAME-START and FRAME-END sensor
        poses, anchored to the same ``[frame_start_us, frame_end_us]`` window as
        the per-ray timestamps. So this path takes the true start/end viewmats
        (NOT the mid-frame ``viewmat`` the structured path uses).

        The frame context (tiling) is pose-independent and is built once per frame
        in the dataloader worker, then carried on the batch as
        ``rendering_data.unstructured_lidar_tiling``. Nothing is constructed or
        cached here: this method rebuilds the context from those tensors at zero
        compute, and raises if the batch did not carry them.
        """
        # Resolve the PerSensor cull / box_method for this sensor (scalar broadcast
        # or {lidar_id: value} dict). lidar_id is None only when frame_meta was not
        # supplied or the idx is unmapped, in which case a dict value raises.
        cull_dead_pairs = resolve_per_sensor(self.config.culling.cull_dead_pairs_lidar_unstructured, lidar_id)
        box_method = cast(
            Literal["ut", "pbf", "hybrid"],
            resolve_per_sensor(self.config.culling.box_method_lidar_unstructured, lidar_id),
        )
        pbf_near_range = resolve_per_sensor(self.config.culling.pbf_near_range_lidar_unstructured, lidar_id)
        rs_single_cover = resolve_per_sensor(self.config.culling.rs_single_cover_lidar_unstructured, lidar_id)

        # RS single-cover is a UT-box approximation and MUST be paired with the
        # hybrid box so near gaussians keep the exact PBF box (the RS box can
        # under-cover near / large gaussians -- FINDINGS Phase 42/43). Reject any
        # other box_method to fail loudly instead of silently under-covering.
        if rs_single_cover and box_method != "hybrid":
            raise ValueError(
                "rs_single_cover_lidar_unstructured=True requires "
                "box_method_lidar_unstructured='hybrid' (RS is a UT approximation "
                "that can under-cover near gaussians; hybrid keeps the near PBF box "
                f"exact). Got box_method={box_method!r} for lidar_id={lidar_id!r}."
            )

        # Active tile masking is a STRUCTURED-grid concept and has no meaning here.
        # The mask is defined over gsplat's spinning-lidar tile grid, which
        # `build_camera_to_lidar_tile_mapping` derives from a
        # RowOffsetStructuredSpinningLidarModelParameters via
        # `compute_lidar_element_to_tile` (create_elements() ->
        # elements_to_sensor_angles() -> reshape(n_rows, n_columns)). The
        # model-free path has no sensor model and no (n_rows, n_columns) grid, and
        # `render_unstructured` takes no `masks=` argument, so a mask arriving here
        # could only be silently dropped. NuRecConfig rejects the combination up
        # front (see the lidar_unstructured / active_tile_mask guard); this is the
        # render-time backstop for a non-config producer that sets the field
        # directly.
        if rendering_data.active_tile_mask is not None:
            raise ValueError(
                "active_tile_mask is set on an unstructured (model-free) lidar render, "
                "but the unstructured path has no structured tile grid to apply it to "
                "and gsplat's render_unstructured accepts no mask -- it would be "
                "silently ignored. Use active_tile_mask.strategy='camera_invalid_only' "
                "(camera-only, safe with the unstructured lidar path) or disable active "
                f"tile masking. lidar_id={lidar_id!r}."
            )

        if not getattr(self, "_unstructured_logged", False):
            logging.getLogger(__name__).info(
                "GSplatRenderer: MODEL-FREE UNSTRUCTURED lidar render ACTIVE "
                "(config.lidar_unstructured=True; render_unstructured, "
                "opacity_aware_footprint=True, cull_dead_pairs=%s).",
                cull_dead_pairs,
            )
            self._unstructured_logged = True

        rays = rendering_data.rays[0]  # world origin+dir; (N, 1, 6) model-free grid
        H, W = int(rays.shape[0]), int(rays.shape[1])
        n = H * W  # == N for the model-free (N, 1) grid

        # Per-ray timestamps (us). Required for the unstructured render: each
        # ray's rolling-shutter pose is interpolated from its exact timestamp, so
        # there is no meaningful no-timestamp fallback -- fail loudly instead.
        if rendering_data.rays_timestamps_us is None:
            raise ValueError(
                "Unstructured LiDAR render requires per-ray rays_timestamps_us "
                "(dataset.lidar_ray_mode='unstructured'); none were provided."
            )

        # Frame context: prefer the tiling PREBUILT in the dataloader worker
        # (rendering_data.unstructured_lidar_tiling) so no construction happens in
        # the render loop. Its tensors already rode the batch to this device; we
        # rebuild a gsplat UnstructuredLidarFrameContext from them at zero compute
        # via from_tensors (the worker's own context object was discarded after
        # to_tensors -- only the buffers + scalar metadata travel)
        if (tiling_data := rendering_data.unstructured_lidar_tiling) is None:
            raise ValueError(
                "Unstructured LiDAR render requires prebuilt tiling in "
                "rendering_data.unstructured_lidar_tiling; none was provided. "
                "Datasets must be configured with lidar_ray_mode='unstructured' "
                "and a spinning model to build the tiling."
            )

        frame_context = gsplat.UnstructuredLidarFrameContext.from_tensors(
            directions=tiling_data.directions,
            az=tiling_data.az,
            el=tiling_data.el,
            relative_timestamps=tiling_data.relative_timestamps,
            occupancy=tiling_data.occupancy,
            tiles_pack_info=tiling_data.tiles_pack_info,
            tiles_to_elements_map=tiling_data.tiles_to_elements_map,
            ray_index=tiling_data.ray_index,
            time_mask=tiling_data.time_mask,
            tile_mean_time=tiling_data.tile_mean_time,
            meta=tiling_data.meta,
        )

        # gsplat API (MR !404): rolling-shutter motion-arc bounding mode.
        # "endpoints" (former "single") bounds the arc by the frame start+end
        # poses -- correct for single-cover (spinning / semi-solid-state) sensors;
        # "adaptive" (former "double") adds intermediate pose samples for
        # multi-visit (MEMS / self-crossing) motion.
        arc_sampling: Literal["endpoints", "adaptive"] = (
            "adaptive" if frame_context.multi_visit_fraction() > 0.2 else "endpoints"
        )

        # RS single-cover requires an actual single-cover frame: the per-cell
        # mean-time seed is only meaningful when every ray in a cell shares ~one
        # firing time. Multi-visit (time-tiled / adaptive-arc) frames violate that,
        # so refuse RS there instead of silently seeding the iteration with a
        # meaningless averaged time.
        if rs_single_cover and (arc_sampling != "endpoints" or getattr(frame_context.tiling, "time_tiled", False)):
            raise ValueError(
                "rs_single_cover_lidar_unstructured=True is only valid for "
                "single-cover frames, but this frame is multi-visit "
                f"(arc_sampling={arc_sampling!r}, time_tiled="
                f"{getattr(frame_context.tiling, 'time_tiled', False)}). Disable RS "
                f"for this sensor (lidar_id={lidar_id!r}) or use a single-cover scan."
            )

        # LiDAR signals travel through gsplat's `extra_signals` channel, NOT the
        # `colors` argument: do not pass/inspect gaussian_params.colors here, which
        # may hold camera SH coefficients [N, K, 3] that must never leak into the
        # LiDAR payload. The payload packs the per-Gaussian channels composited
        # along each ray -- [intensity?(1), raydrop_raw?(K)] -- and we track each
        # block's offset (intensity_slice / raydrop_slice) to unpack the result.
        payload_blocks: list[torch.Tensor] = []
        n_gauss = gaussian_params.means.shape[0]
        intensity_slice = None
        raydrop_slice = None
        if intensity is not None:
            expected_intensity_shape = (n_gauss, 1)
            if intensity.shape != expected_intensity_shape:
                raise ValueError(
                    "Unstructured LiDAR intensity must have one value per Gaussian: "
                    f"expected {expected_intensity_shape}, got {tuple(intensity.shape)}."
                )
            intensity_slice = (sum(b.shape[-1] for b in payload_blocks), 1)
            payload_blocks.append(intensity)
        if raydrop_channels is not None:
            if raydrop_channels.shape[0] != n_gauss:
                raise ValueError(
                    "Unstructured LiDAR raydrop channels must have one row per Gaussian: "
                    f"expected {n_gauss}, got {raydrop_channels.shape[0]}."
                )
            raydrop_slice = (sum(b.shape[-1] for b in payload_blocks), raydrop_channels.shape[-1])
            payload_blocks.append(raydrop_channels)
        payload = torch.cat(payload_blocks, dim=-1) if payload_blocks else None

        unstructured_config = gsplat.UnstructuredLidarConfig(
            arc_sampling=arc_sampling,
            opacity_aware_footprint=True,
            cull_dead_pairs=cull_dead_pairs,
            box_method=box_method,
            pbf_near_range=pbf_near_range,
            rs_single_cover=rs_single_cover,
        )
        renderer_config: gsplat.RendererConfig
        if self.config.renderer_config == "mixed_batch":
            renderer_config = gsplat.RendererConfig_MixedBatch(unstructured_lidar=unstructured_config)
        elif self.config.renderer_config == "parallel_batch":
            renderer_config = gsplat.RendererConfig_ParallelBatch(unstructured_lidar=unstructured_config)
        else:
            raise ValueError(f"Unknown renderer_config: {self.config.renderer_config!r}")

        render_colors, render_alphas, meta = gsplat.rasterization(
            means=gaussian_params.means.unsqueeze(0),
            quats=gaussian_params.quats.unsqueeze(0),
            scales=gaussian_params.scales.unsqueeze(0),
            opacities=gaussian_params.opacities.unsqueeze(0),
            colors=None,
            viewmats=viewmat_start.unsqueeze(0).unsqueeze(0),
            Ks=torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
            width=n,
            height=1,
            near_plane=near_plane,
            far_plane=far_plane,
            render_mode="Ed",
            camera_model="lidar",
            with_ut=True,
            with_eval3d=True,
            lidar_frame_context=frame_context,
            viewmats_rs=(viewmat_end if viewmat_end is not None else viewmat_start).unsqueeze(0).unsqueeze(0),
            extra_signals=payload,
            renderer_config=renderer_config,
        )

        opacity = render_alphas.reshape(n)
        distance = render_colors[..., -1].reshape(n)
        # Annotate explicitly: `meta` is an untyped dict, so `meta.get(...)` is
        # `Any | None` and mypy (1.15) cannot bind unpack_optional's TypeVar from it
        # (it degrades to `Never`, making the result non-indexable).
        rendered_extra: Optional[torch.Tensor] = meta.get("render_extra_signals")
        rendered_intensity = None
        if intensity_slice is not None:
            s, w = intensity_slice
            rendered_intensity = unpack_optional(
                rendered_extra,
                msg="gsplat returned no 'render_extra_signals' despite extra_signals being provided (intensity).",
            )[..., s : s + w].reshape(n, w)
            # Match the structured path, which returns extra signals post-activation
            # (ExtraSignal.from_packed_tensor applies each signal's activation).
            if intensity_activation is not None:
                rendered_intensity = intensity_activation(rendered_intensity)

        # Raydrop prediction. PREFERRED: the LEARNED per-Gaussian raydrop head,
        # composited along the ray exactly like the structured path -- the whole
        # point of the raydrop/virtual-ray feature is trainable raydrop, not a
        # geometry-only proxy. The gsplat kernel leaves non-range channels
        # premultiplied by alpha (sum(w*logit)); alpha-normalize them (as range is)
        # to recover the composited logits, then apply the configured ray-side
        # activation (softmax-channel-0) -> drop probability in [0, 1], matching
        # ExtraSignal.from_packed_tensor on the structured path.
        # FALLBACK (no raydrop head configured): the geometric 1 - opacity signal.
        if raydrop_slice is not None and raydrop_activation is not None:
            s, w = raydrop_slice
            # The composited raydrop logits are PREMULTIPLIED by alpha
            # (sum(weight * logit) over a zero background) -- gsplat does NOT
            # alpha-normalize non-range channels, exactly as the structured 3DGUT
            # rasterizer leaves jointly-rasterized extra_signals premultiplied
            # (only 'Ed'/'ED' expected-depth modes normalize). So feed the raw
            # composited logits to the configured activation, matching the
            # structured path's ExtraSignal.from_packed_tensor (no normalization).
            composited = unpack_optional(
                rendered_extra,
                msg="gsplat returned no 'render_extra_signals' despite extra_signals being provided (raydrop).",
            )[..., s : s + w].reshape(n, w)
            rendered_raydrop = raydrop_activation(composited).reshape(n, 1)
        else:
            # Geometric fallback: accumulated opacity is the return/drop evidence.
            # raydrop = 1 - opacity in [0, 1] (1 = likely dropped). Differentiable
            # w.r.t. opacity/geometry. Used only when no learned raydrop head is
            # configured for the lidar extra signal.
            rendered_raydrop = (1.0 - opacity).clamp(0.0, 1.0).reshape(n, 1)

        if rendered_intensity is not None:
            extra_ray_signals = ExtraSignal(intensity=rendered_intensity, raydrop=rendered_raydrop)
        else:
            extra_ray_signals = ExtraSignal(raydrop=rendered_raydrop)
        # RGB is intentionally None: model-free LiDAR returns depth, opacity, and
        # optional intensity separately; camera colors are not used by LiDAR losses.
        rgb = None

        return GaussiansRenderReturn(
            rgb=rgb,
            opacity=opacity,
            distance=distance if enable_distance else None,
            normal=None,
            extra_ray_signals=extra_ray_signals,
            visibility=None,
        )

    def render(
        self,
        rendering_data: RenderingData,
        gaussian_parameters: dict[str, torch.Tensor],
        n_active_features: int,
        extra_ray_signal_infos: tuple[list[str], list[int], list[Callable]],
        frame_meta: Optional[List[FrameMeta]] = None,
        segment_plan: SegmentPlan | None = None,
    ) -> GaussiansRenderReturn:
        """Render Gaussians using gsplat (camera or lidar path; dispatched per sensor).

        Extra-signal concatenation into the rasterizer is controlled by two
        independent `outputs.<sensor>` gates:
          - `enable_extended_features` → common `extra_signal`
          - `enable_sensor_features`   → sensor-specific `<sensor>_extra_signal`
            (`camera_extra_signal` on the camera path, `lidar_extra_signal` on the
            lidar path)
        Disabling a gate lowers CDIM by that tensor's channels and leaves its
        `ExtraSignal` slot `None`.

        Args:
            rendering_data: Ray bundle with camera or lidar parameters and poses
            gaussian_parameters: Dict with means, quats, scales, densities, features,
                and optionally extra_signal, camera_extra_signal (camera path) or
                lidar_extra_signal (lidar path)
            n_active_features: Number of active SH features
            extra_ray_signal_infos: Extra signal (names, dims, activations) for unpacking
            frame_meta: Optional frame metadata
            segment_plan: Optional draw plan. Its segments use shape ``[2, segment_count]``.
                Row 0 stores start rows, and row 1 stores row counts. Its offsets use
                shape ``[segment_count + 1]`` and store packed start rows.

        Returns:
            GaussiansRenderReturn with RGB, ray distance, opacity, normals, and
            `extra_ray_signals` populated for whichever gated tensors had non-zero dim.
        """
        assert rendering_data.b == 1, "Only single-frame batch supported"

        with ScopedTimer("GSplatRenderer/render"):
            # Extract sensor parameters and set sensor-specific state (camera vs lidar).
            # In the model-free unstructured LiDAR mode there is NO sensor model (the
            # ray list is model-independent), so the per-frame entry is None; that
            # denotes a (non-camera) lidar render, routed to _render_unstructured.
            sensor_params = rendering_data.sensor_model_parameters[0]
            device = rendering_data.rays[0].device
            is_camera = sensor_params is not None and isinstance(
                sensor_params, get_union_types(ConcreteCameraModelParametersUnion)
            )

            # Single-owner contract: the composite builds the mask and hands it over on
            # rendering_data (see `_warn_missing_active_tile_mask_once` on the base).
            if self.config.active_tile_mask.enabled and is_camera and rendering_data.active_tile_mask is None:
                self._warn_missing_active_tile_mask_once()

            out = self._camera_outputs if is_camera else self._lidar_outputs
            enable_features: bool = out["enable_features"]
            enable_normals: bool = out["enable_normals"]
            enable_distance: bool = out["enable_distance"]
            enable_extended_features: bool = out["enable_extended_features"]
            enable_sensor_extended_features: bool = out["enable_sensor_extended_features"]

            near_plane = float(self.config.culling.near_clip_distance)
            external_distortion_params = None
            viewmat_rs = None
            tile_size_strategy = self.config.tiling.strategy

            # --- Model-free UNSTRUCTURED lidar render (config-gated) ---
            # dataset.lidar_ray_mode='unstructured' (enforced by NuRecConfig to pair
            # with GSplatRendererConfig.lidar_unstructured=True) routes the lidar
            # branch through gsplat's model-free render_unstructured on the
            # measured (N, 1) ray list. This is dispatched BEFORE the structured
            # sensor-model setup below because that setup requires a
            # RowOffsetStructuredSpinningLidarModelParameters, which is deliberately
            # absent (None) in model-free mode. Active in train+val so the lidar loss
            # back-props through it. Camera path is untouched.
            lidar_id = None if is_camera else self._lidar_id_from_meta(frame_meta)
            lidar_unstructured = (not is_camera) and self.renders_unstructured_lidar(frame_meta)
            if lidar_unstructured:
                far_plane_lidar = float(self.config.culling.far_clip_distance_lidar)
                # The unstructured renderer interpolates each ray's pose over the true
                # [frame_start, frame_end] window, so it needs the FRAME-START viewmat
                # (not the mid-frame viewmat the structured path uses) and the
                # frame-end pose as viewmat_rs.
                viewmat_start = self._build_viewmat(rendering_data, device, use_midframe=False)
                viewmat_end = self._build_rolling_shutter_matrix(rendering_data, device)
                gaussian_params = self._extract_gaussian_parameters(gaussian_parameters, device)
                viewmat_midframe = (
                    self._build_viewmat(rendering_data, device, use_midframe=True)
                    if self._lidar_extra_signals_sh_degree is not None
                    else None
                )
                intensity = self._get_unstructured_lidar_intensity(
                    gaussian_parameters,
                    enable_extended_features=enable_extended_features,
                    enable_sensor_extended_features=enable_sensor_extended_features,
                    extra_ray_signal_infos=extra_ray_signal_infos,
                    means=gaussian_params.means,
                    viewmat_midframe=viewmat_midframe,
                )
                # Learned per-Gaussian raydrop channels (raw, pre-activation). When
                # a 'raydrop' lidar extra-signal head is configured these are
                # composited along the ray like intensity, then the configured
                # ray-side activation (softmax-channel-0) turns them into a drop
                # probability -- matching the structured path's LEARNED raydrop
                # head. When absent, _render_unstructured falls back to the
                # geometric 1 - opacity signal.
                raydrop_channels = self._get_unstructured_lidar_signal(
                    "raydrop",
                    gaussian_parameters,
                    enable_extended_features=enable_extended_features,
                    enable_sensor_extended_features=enable_sensor_extended_features,
                    extra_ray_signal_infos=extra_ray_signal_infos,
                    means=gaussian_params.means,
                    viewmat_midframe=viewmat_midframe,
                )
                raydrop_activation = None
                intensity_activation = None
                _names, _dims, _acts = extra_ray_signal_infos
                if raydrop_channels is not None:
                    raydrop_activation = _acts[_names.index("raydrop")]
                # The structured path returns every extra signal post-activation via
                # ExtraSignal.from_packed_tensor, so intensity must be activated here
                # too -- otherwise a non-identity intensity head yields different
                # values on the structured and unstructured paths.
                if intensity is not None and "intensity" in _names:
                    intensity_activation = _acts[_names.index("intensity")]
                with ScopedTimer("GSplatRenderer/render/render_unstructured"):
                    # _render_unstructured ALWAYS renders (never returns None):
                    # there is deliberately NO fallback to the structured rasterizer.
                    return self._render_unstructured(
                        rendering_data=rendering_data,
                        gaussian_params=gaussian_params,
                        viewmat_start=viewmat_start,
                        viewmat_end=viewmat_end,
                        near_plane=near_plane,
                        far_plane=far_plane_lidar,
                        enable_distance=enable_distance,
                        intensity=intensity,
                        raydrop_channels=raydrop_channels,
                        raydrop_activation=raydrop_activation,
                        intensity_activation=intensity_activation,
                        device=device,
                        lidar_id=lidar_id,
                    )

            if is_camera:
                camera_params = cast(ConcreteCameraModelParametersUnion, sensor_params)
                width = rendering_data.w
                height = rendering_data.h
                # Create camera adapter using polymorphism
                sensor_adapter = SensorModelParametersAdapter.factory(
                    camera_params,
                    device,
                    self._camera_sensor_tensors,
                )

                external_distortion_params = require_concrete_external_distortion_parameters(
                    camera_params.external_distortion_parameters
                )
                external_distortion_adapter = (
                    ExternalDistortionAdapter.factory(external_distortion_params, device)
                    if external_distortion_params is not None
                    else None
                )

                # Get camera model info from adapter
                camera_model = sensor_adapter.get_camera_model_name()
                gsplat_sensor_params = sensor_adapter.get_gsplat_sensor_params()

                external_distortion_params = (
                    external_distortion_adapter.get_distortion_params()
                    if external_distortion_adapter is not None
                    else None
                )

                # Build camera matrices using adapter (start-of-frame pose).
                viewmat = self._build_viewmat(rendering_data, device)

                if self.use_rays:
                    # TODO: add ray footprint support to gsplat
                    # TODO: add support for ray timestamp input to gsplat
                    gsplat_sensor_params["rays"] = rendering_data.rays[0]  # (H, W, 6)
                use_3dgut = self._use_3dgut
                rolling_shutter_type = gsplat.RollingShutterType.GLOBAL
                if self._has_rolling_shutter(camera_params):
                    viewmat_rs = self._build_rolling_shutter_matrix(rendering_data, device)
                    rolling_shutter_type = self._get_rolling_shutter_type(camera_params)

                far_plane = float(self.config.culling.far_clip_distance_camera)

                rasterize_mode = self.config.rasterize_mode

            else:
                lidar_params = cast(ConcreteLidarModelParametersUnion, sensor_params)
                if not isinstance(lidar_params, RowOffsetStructuredSpinningLidarModelParameters):
                    raise NotImplementedError(
                        "GSplatRenderer supports only camera and RowOffsetStructuredSpinningLidarModelParameters lidar."
                    )
                # Single source of truth for the lidar tile grid — the same accessor the
                # camera->lidar mask build uses, so mask and render cannot drift apart.
                lidar_tiling_params = self.lidar_tiling_params()
                lidar_frame_context = self._build_lidar_params(
                    lidar_params,
                    device=device,
                    **lidar_tiling_params,
                )
                height, width = lidar_frame_context.render_shape

                camera_model = cast(gsplat.CameraModel, "lidar")
                use_3dgut = True
                gsplat_sensor_params = {
                    "Ks": torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
                    "lidar_frame_context": lidar_frame_context,
                }

                # For lidar, use mid-frame
                viewmat = self._build_viewmat(rendering_data, device, True)

                viewmat_rs = self._build_rolling_shutter_matrix(rendering_data, device)
                # TODO: this is currently ignored with Lidar, but gsplat expects it as a
                # mandatory parameter. But for giving viewmat_rs (which lidar needs), the
                # shutter type must be different from GLOBAL.
                rolling_shutter_type = gsplat.RollingShutterType.ROLLING_LEFT_TO_RIGHT

                if self.use_rays:
                    # TODO: add ray footprint support to gsplat
                    # TODO: add support for ray timestamp input to gsplat
                    rays = rendering_data.rays[0]  # (H, W, 6)
                    rays = torch.cat(
                        [
                            rays[..., :3],  # origin
                            rays[..., 3:] * self.config.antialiasing.lidar_divergence,  # direction
                        ],
                        dim=-1,
                    )

                    gsplat_sensor_params["rays"] = rays

                far_plane = float(self.config.culling.far_clip_distance_lidar)

                # Lidar must not apply view-dependent compensation to opacity
                rasterize_mode = "classic"

            # Extract Gaussian parameters
            gaussian_params = self._extract_gaussian_parameters(gaussian_parameters, device)

            # Determine rendering mode: RGB-only vs SH
            # RGB-only: n_active_features=0 and colors are (N, 3)
            # SH mode: n_active_features>0 and colors are (N, K, 3)
            colors_shape = gaussian_params.colors.shape
            is_rgb_only = colors_shape[-1] == 3 and len(colors_shape) == 2  # (N, 3) shape

            # Validate n_active_features against stored features (only for SH mode)
            if enable_features and not is_rgb_only:
                n_stored_bands = colors_shape[1]  # (N, K, 3) -> K bands
                n_requested_bands = (n_active_features + 1) ** 2
                assert n_requested_bands <= n_stored_bands, (
                    f"Feature dimension mismatch: requested sh_degree={n_active_features} "
                    f"({n_requested_bands} bands), but stored features only have {n_stored_bands} bands "
                    f"(sh_degree={int(math.sqrt(n_stored_bands)) - 1}). "
                    f"Cannot use more features than stored in model."
                )

            # Calculate eps2d from min_projected_ray_radius
            # eps2d=0 for lidar (no Mip-Splatting dilation needed).
            proj = self.config.projection
            min_ray_radius = proj.min_projected_ray_radius
            eps2d = 0.0 if not is_camera else min_ray_radius**2

            if use_3dgut:
                ut_params = gsplat.rendering.UnscentedTransformParameters(
                    alpha=float(proj.ut_alpha),
                    beta=float(proj.ut_beta),
                    kappa=float(proj.ut_kappa),
                    in_image_margin_factor=float(proj.image_margin_factor),
                    require_all_sigma_points_valid=proj.ut_require_all_sigma_points,
                )

            else:
                if self.use_rays:
                    raise NotImplementedError("Ray input in 3dGS mode isn't supported")

                if camera_model == "ftheta":
                    raise NotImplementedError("GSplat doesn't support ftheta camera model 3dGS mode")

                if rolling_shutter_type != gsplat.RollingShutterType.GLOBAL:
                    raise NotImplementedError(
                        "GSplat doesn't support rolling shutter type other than GLOBAL in 3dGS mode"
                    )

                # No support for distortion coefficients in 3dGS mode
                gsplat_sensor_params_3dgs = {"Ks": gsplat_sensor_params["Ks"]}
                gsplat_sensor_params = gsplat_sensor_params_3dgs
                external_distortion_params = None

                ut_params = None
                viewmat_rs = None

            # Gradients and memory optimization parameters
            sparse_grad = self.config.sparse_grad
            absgrad = self.config.absgrad
            packed = self.config.packed

            # Validate sparse_grad requires packed mode
            if sparse_grad or packed:
                if use_3dgut:
                    logging.getLogger(__name__).warning(
                        "sparse_grad requested but 3DGUT in gsplat currently doesn't support packed=True. "
                    )
                    sparse_grad = False
                    packed = False

            # Select the minimal gsplat render_mode from the output gates. Color is
            # produced when features are enabled; the depth/distance channel is produced
            # when enable_distance is set. Depth-only models (no features) always keep
            # depth. Skipping depth when it is gated off is the CDIM 4 -> 3 win.
            # 3DGUT denotes along-the-ray (hit) distance lowercase 'd'; 3DGS uses z-depth 'D'.
            rgb_with_depth: gsplat.RenderMode = "RGB-d" if use_3dgut else "RGB+D"
            depth_only: gsplat.RenderMode = "d" if use_3dgut else "D"
            render_mode: gsplat.RenderMode
            if not enable_features:
                render_mode = depth_only
            elif enable_distance:
                render_mode = rgb_with_depth
            else:
                render_mode = "RGB"

            global_z_order = not use_3dgut  # True for 3DGS (z-depth), False for 3DGUT (Euclidean distance)

            # RGB vs SH rendering mode:
            # - RGB-only mode (Kelvin/Celsius): sh_degree=None, colors are (N, 3)
            # - SH mode: sh_degree=n_active_features, colors are (N, K, 3)
            sh_degree_param = None if is_rgb_only else (n_active_features if enable_features else None)

            extra_signal_dim = gaussian_parameters["extra_signal"].shape[-1]
            camera_extra_signal_dim = gaussian_parameters["camera_extra_signal"].shape[-1]

            extra_signals = None
            extra_signals_sh_degree = None
            parts = []
            if enable_extended_features and extra_signal_dim > 0:
                parts.append(gaussian_parameters["extra_signal"])
            if enable_sensor_extended_features:
                if is_camera:
                    camera_extra_signal_dim = gaussian_parameters["camera_extra_signal"].shape[-1]
                    if camera_extra_signal_dim > 0:
                        parts.append(gaussian_parameters["camera_extra_signal"])
                else:
                    lidar_extra_signal_dim = gaussian_parameters["lidar_extra_signal"].shape[-1]
                    if lidar_extra_signal_dim > 0:
                        parts.append(gaussian_parameters["lidar_extra_signal"])
            if parts:
                if not is_camera:
                    extra_signals_sh_degree = self._lidar_extra_signals_sh_degree
                else:
                    extra_signals_sh_degree = self._extra_signals_sh_degree
                extra_signals_cat = _concatenate_extra_signals(parts)  # [N, E]
                if extra_signals_sh_degree is not None:
                    # SH mode: gsplat wants [N, K, 3] with the batch dim deduplicated.
                    n_es, e_dim = extra_signals_cat.shape
                    extra_signals = extra_signals_cat.reshape(n_es, e_dim // 3, 3)
                else:
                    extra_signals = extra_signals_cat.unsqueeze(0)  # [1, N, E]

            # ECAD (strategies/ecad.py): training camera renders carry four extra per-Gaussian channels.
            ecad_dim = 0
            if (
                ECAD_RENDER_STATE.enabled
                and is_camera
                and torch.is_grad_enabled()
                and segment_plan is None
                and extra_signals_sh_degree is None
                and gaussian_parameters["camera_extra_signal"].shape[-1] >= 1
            ):
                extra_signals = ECAD_RENDER_STATE.attach(gaussian_parameters["camera_extra_signal"], extra_signals)
                ecad_dim = ECAD_RENDER_STATE.num_channels

            renderer_config: gsplat.RendererConfig
            if self.config.renderer_config == "mixed_batch":
                renderer_config = gsplat.RendererConfig_MixedBatch()
            elif self.config.renderer_config == "parallel_batch":
                renderer_config = gsplat.RendererConfig_ParallelBatch()
            else:
                raise ValueError(f"Unknown renderer_config: {self.config.renderer_config!r}")

            # RESOLUTION: tile_size=None defers to gsplat's resolution-based
            # fallback. CUSTOM: pass the configured camera tile_size verbatim
            # (square).
            tile_size = None if tile_size_strategy == TileSizeStrategy.RESOLUTION else self.tile_size

            render_kwargs: dict[str, Any] = {}

            # Sparse Gaussian segments are an explicit draw plan from the composite; no plan
            # means every row is live and the frame is drawn the dense way. A plan can only
            # arrive from a composite that already agreed the frame is sparse, so anything
            # inconsistent with it is a bug here rather than a configuration to fall back on.
            assert segment_plan is None or self.config.sparse_segments, (
                "received a segment plan but sparse_segments is off"
            )
            if segment_plan is not None:
                assert use_3dgut, "sparse_segments requires the 3DGUT projection path."
                # The same list the composite consulted before it built this plan.
                assert_sparse_segments_supported(
                    has_extra_signals=extra_signals is not None,
                    has_active_tile_mask=rendering_data.active_tile_mask is not None,
                    grad_enabled=torch.is_grad_enabled(),
                )
                n_total = int(gaussian_params.means.shape[0])
                assert segment_plan.total_rows == n_total, (
                    "The segment plan row count does not match the collected Gaussian rows: "
                    f"{segment_plan.total_rows} != {n_total}."
                )
                assert segment_plan.segments.device == gaussian_params.means.device, (
                    "The segment plan and Gaussian rows must be on the same device."
                )
                render_kwargs["segments"] = segment_plan.segments
                render_kwargs["segment_offsets"] = segment_plan.segment_offsets
                render_kwargs["active_count"] = segment_plan.active_count

            # Band-0 Fourier fold: recompute the folded static holistic layer's band-0
            # in the SH kernel, on the dense path as well as the segments one.
            # Forwarded when the collector folded a layer (_fourier_fold_ref set) and
            # this render reaches the fold-capable SH path. theta = 2*pi*remap(time),
            # host fp64 (ground-truth-correct, matching the holistic remap embedding),
            # passed as a scalar.
            #
            # band0_fold_supported is the same predicate the collector consulted before
            # folding, so the two decisions cannot diverge.
            fold_supported = self.band0_fold_supported(
                is_camera=is_camera,
                features_dim=int(gaussian_parameters["features"].shape[-1]),
                extra_signal_dim=extra_signal_dim,
                sensor_extra_signal_dim=int(
                    gaussian_parameters["camera_extra_signal" if is_camera else "lidar_extra_signal"].shape[-1]
                ),
            )
            fold = getattr(self, "_fourier_fold_ref", None)
            if fold is not None and fold_supported:
                ratio = (fold["frame_timestamp_us"] - fold["timestamps_us_min"]) / (
                    fold["timestamps_us_max"] - fold["timestamps_us_min"]
                )
                time_embed = ratio * (fold["remap_max"] - fold["remap_min"]) + fold["remap_min"]
                render_kwargs["fourier_albedo"] = fold["albedo"]
                render_kwargs["fourier_theta"] = 2.0 * math.pi * time_embed
                render_kwargs["fourier_row_range"] = (fold["start"], fold["start"] + fold["size"])

            # Backstop, unreachable in normal operation: the collector only folds after
            # band0_fold_supported answered True for this sensor, and this render calls
            # the same predicate. Reaching here means a fold ref was installed against a
            # render that drops it, which would silently render a stale band-0.
            if getattr(self, "_fourier_fold_ref", None) is not None and "fourier_theta" not in render_kwargs:
                raise AssertionError(
                    "fold divergence: _fourier_fold_ref is set (a band-0 fold was folded "
                    "out of the collect) but this render did not honor it, so band-0 would "
                    "render stale. The fold ref must only be installed on a renderer whose "
                    "band0_fold_supported() answered True for this sensor."
                )

            # Active tile mask. The composite computes a host-side mask and carries it on
            # rendering_data.active_tile_mask as a batched bool tensor over this renderer's
            # tile grid. gsplat skips inactive tiles at intersection (so intersection, sort,
            # and rasterization are all skipped for them; masked tiles render as zeros, since
            # no `backgrounds=` is passed). Both the camera path (mask over [tiles_y, tiles_x])
            # and the lidar path (mask over gsplat's [n_bins_elevation, n_bins_azimuth] grid)
            # honor it. The mask tile grid must match gsplat's resolved tile grid (built via
            # active_tile_mask_tile_size / compute_lidar_element_to_tile).
            # gsplat validates masks against viewmats' batch layout: the mask must be
            # [*batch_dims, C, tile_height, tile_width] (matching meta["isect_offsets"]).
            # This renderer always rasterizes a single view, so viewmats is (1, 1, 4, 4)
            # (batch=1, C=1) and the composite's (B=1, tile_h, tile_w) mask needs the camera
            # dim inserted -> (B, 1, tile_h, tile_w).
            # Both layouts already match gsplat's [tile_h, tile_w] grid, so the same
            # unsqueeze applies: camera masks are (B, tiles_y, tiles_x) and lidar masks are
            # (B, n_bins_elevation, n_bins_azimuth) (tile id = el * n_bins_azimuth + az,
            # i.e. elevation-major, as the composite already lays them out).
            active_tile_mask = (
                None
                if rendering_data.active_tile_mask is None
                else rendering_data.active_tile_mask.unsqueeze(1).contiguous()
            )

            # Call gsplat rasterization
            with ScopedTimer("GSplatRenderer/render/rasterization"):
                render_colors, render_alphas, meta = gsplat.rasterization(
                    **render_kwargs,
                    means=gaussian_params.means.unsqueeze(0),  # (1, N, 3)
                    quats=gaussian_params.quats.unsqueeze(0),  # (1, N, 4)
                    scales=gaussian_params.scales.unsqueeze(0),  # (1, N, 3)
                    opacities=gaussian_params.opacities.unsqueeze(0),  # (1, N)
                    # SH path: pass [N, K, 3] with no batch dim (gsplat dedups across the batch).
                    # Post-activation RGB path: keep the [..., N, D] form.
                    colors=(
                        gaussian_params.colors if sh_degree_param is not None else gaussian_params.colors.unsqueeze(0)
                    ),
                    viewmats=viewmat.unsqueeze(0).unsqueeze(0),  # (1, 1, 4, 4)
                    width=width,
                    height=height,
                    near_plane=near_plane,
                    far_plane=far_plane,
                    radius_clip=float(self.config.radius_clip),
                    eps2d=eps2d,
                    sh_degree=sh_degree_param,
                    tile_size=tile_size,
                    render_mode=render_mode,
                    rasterize_mode=rasterize_mode,
                    camera_model=camera_model,
                    global_z_order=global_z_order,
                    **gsplat_sensor_params,
                    external_distortion_coeffs=external_distortion_params,
                    ut_params=ut_params,
                    rolling_shutter=rolling_shutter_type,
                    viewmats_rs=viewmat_rs.unsqueeze(0).unsqueeze(0) if viewmat_rs is not None else None,
                    # 3DGUT mode: with_ut=True (distortion/RS). Pose grads flow when rays are
                    # passed (use_rays=True, the default for 3dgut-gsplat).
                    # Standard mode: with_ut=False (pose grads via viewmat).
                    with_ut=use_3dgut,
                    with_eval3d=use_3dgut,
                    # Compute normals if enabled (requires with_eval3d=True)
                    return_normals=enable_normals and use_3dgut,
                    packed=packed,  # Memory efficient
                    sparse_grad=sparse_grad,  # COO sparse layout gradients (Experimental)
                    absgrad=absgrad,  # Absolute gradients for splitting
                    extra_signals=extra_signals,
                    extra_signals_sh_degree=extra_signals_sh_degree,
                    renderer_config=renderer_config,
                    masks=active_tile_mask,
                )

                # Store meta dict for densification strategy access (if absgrad enabled)
                # Strategies can access meta["means2d"].absgrad after backward pass
                if absgrad:
                    self.last_rendering_meta = meta

            # Parse outputs
            with ScopedTimer("GSplatRenderer/render/parse_outputs"):
                opacity = render_alphas[0, 0, :, :, 0]  # (H, W)

                # Extract based on render mode
                if enable_features:
                    # "RGB+d"/"RGB-d"/"RGB" mode: first 3 channels are RGB
                    rgb = render_colors[0, 0, :, :, :3]  # (H, W, 3)
                    # Depth is the trailing channel, present only when it was rendered.
                    distance = render_colors[0, 0, :, :, 3] if enable_distance else None
                else:
                    # D/d mode: render_colors only has depth
                    rgb = None
                    distance = render_colors[0, 0, :, :, 0]  # (H, W) depth or hit distance

                # Conditionally compute/return outputs based on config.outputs.camera.enable_* flags

                # Normals: Extract from meta if computed
                # Normal computation: canonical normal (0,0,1) transformed by
                # Gaussian rotation, flipped if facing away from ray
                normal = None
                if enable_normals:
                    render_normals = meta.get("normals")
                    if render_normals is not None:
                        # Shape: (1, 1, H, W, 3) -> (H, W, 3)
                        normal = render_normals[0, 0, :, :, :]

                if extra_signals is not None:  # we passed extra signals to gsplat
                    render_extra_signals = meta.get("render_extra_signals")
                    assert render_extra_signals is not None, "gsplat should have returned meta.render_extra_signals"
                    extra_signals = render_extra_signals[0, 0, :, :, :]  # (H, W, E)
                    if ecad_dim:
                        # Split the ECAD channels off before NuRec parses its own extra signals.
                        ECAD_RENDER_STATE.capture(
                            extra_signals[..., -ecad_dim:], meta, gaussian_params.opacities, rendering_data
                        )
                        extra_signals = extra_signals[..., :-ecad_dim] if extra_signals.shape[-1] > ecad_dim else None

                if extra_signals is not None:
                    E = extra_signals.shape[-1]
                    extra_ray_signals = ExtraSignal.from_packed_tensor(
                        extra_signals.reshape(-1, E), extra_ray_signal_infos
                    )
                else:
                    extra_ray_signals = None

                # Base visibility: the Gaussian projects into the frustum (gsplat radii > 0).
                # This is mask-independent — projection runs for every Gaussian.
                visibility = None
                radii = meta.get("radii")
                if radii is not None:
                    # The pinned gsplat always gives radii a trailing size-2 axis (per-axis
                    # extent): [..., C, N, 2] non-packed, [nnz, 2] packed. Reduce it away to
                    # get one radius per Gaussian.
                    if radii.ndim > 1:
                        radii = radii.amax(dim=-1)
                    # We always render a single batch / single camera and the losses expect
                    # (N,) visibility, so flatten the leading dims to (N,).
                    visibility = radii.reshape(-1) > 0

                # Active tiling refinement: AND in "lands in >= 1 active tile". gsplat counts
                # only ACTIVE tiles in meta["tiles_per_gauss"] (IntersectTile / IntersectTileLidar
                # honor the mask in both the count and emit passes), so this narrows visibility
                # to Gaussians that actually contribute to an active tile — letting SelectiveAdam
                # skip the rest, which get ~zero gradient anyway (their tiles aren't rasterized /
                # their pixels are forced to background). Both the camera and lidar renders pass
                # through here and get_rendered_visibility_mask unions them, so both stay
                # mask-aware. Packed mode is excluded (tiles_per_gauss is then (nnz,), not
                # per-Gaussian; the 3dgut path forces packed=False regardless).
                frustum_visibility: Optional[torch.Tensor] = None
                if active_tile_mask is not None and not packed:
                    tiles_per_gauss = meta.get("tiles_per_gauss")
                    if tiles_per_gauss is not None:
                        # int32 over gsplat's leading batch/camera dims, (B, C, N) for this
                        # single-image non-packed render -> flatten to (N,) bool. Keep the
                        # pre-mask (frustum) visibility so MCMC invisible-steps counting can
                        # ignore the mask narrowing (see GaussiansRenderReturn).
                        active_vis = tiles_per_gauss.reshape(-1) > 0
                        # With no radii (gsplat returned no meta["radii"]) there is no base
                        # visibility to preserve, and leaving `frustum_visibility` None would
                        # hand the counters the mask-narrowed signal below -- the exact failure
                        # this field exists to prevent. Unknown pre-mask visibility must read as
                        # "not narrowed", i.e. all-True, never as "invisible".
                        frustum_visibility = visibility if visibility is not None else torch.ones_like(active_vis)
                        visibility = active_vis if visibility is None else (visibility & active_vis)
                    elif not self._warned_missing_tiles_per_gauss:
                        # Third state: mask on, packed off, but the meta key the narrowing
                        # reads is gone. Not config-gated like the packed case -- a gsplat pin
                        # bump that renames or gates `tiles_per_gauss` lands exactly here, and
                        # silently degrading to frustum-only visibility is how that goes
                        # unnoticed for a whole training run.
                        self._warned_missing_tiles_per_gauss = True
                        logging.getLogger(__name__).warning(
                            "[tile_mask] active_tile_mask is set and packed=False, but gsplat returned no "
                            "meta['tiles_per_gauss']; mask-aware visibility (SelectiveAdam skipping and "
                            "regularizer filtering) is unavailable and visibility stays frustum-only. "
                            "Rendering itself still honors the mask. This usually means the pinned gsplat "
                            "build renamed or gated that meta key -- check the gsplat pin."
                        )
                elif active_tile_mask is not None and not self._warned_packed_active_tile_mask:
                    self._warned_packed_active_tile_mask = True
                    logging.getLogger(__name__).warning(
                        "[tile_mask] active_tile_mask is set but the renderer runs with packed=True; "
                        "mask-aware visibility (SelectiveAdam skipping and regularizer filtering) is "
                        "unavailable because tiles_per_gauss is (nnz,) rather than per-Gaussian in "
                        "packed mode. Rendering itself still honors the mask. Set packed: false (the "
                        "3dgut active-tiling path does so automatically) to get mask-aware visibility."
                    )

                if visibility is not None:
                    visibility = visibility.float().contiguous()
                if frustum_visibility is not None:
                    frustum_visibility = frustum_visibility.float().contiguous()

                # Sparse projection metadata is indexed in active draw-list space,
                # not in the caller's scene-row space. Do not expose it through the
                # scene-level visibility contract without an explicit scatter.
                if segment_plan is not None:
                    visibility = None
                    frustum_visibility = None

                # Saturate final RGB only when no downstream post-processing
                # needs the renderer's HDR output.
                if rgb is not None and self._saturate_radiance and not self._has_post_processing:
                    rgb = rgb.clamp(0.0, 1.0)

                return GaussiansRenderReturn(
                    rgb=rgb.reshape(-1, 3) if rgb is not None else None,
                    opacity=opacity.reshape(-1),
                    distance=distance.reshape(-1) if distance is not None else None,
                    normal=normal.reshape(-1, 3) if normal is not None else None,
                    extra_ray_signals=extra_ray_signals,
                    visibility=visibility,
                    frustum_visibility=frustum_visibility,
                )

    class _GaussianParameters(NamedTuple):
        """Gaussian parameters extracted for rendering (private to GSplatRenderer).

        Attributes:
            means: Gaussian centers (N, 3)
            quats: Quaternion rotations in wxyz format (N, 4)
            scales: Gaussian scales (N, 3)
            opacities: Gaussian opacities post-sigmoid (N,)
            colors: Spherical harmonic coefficients (N, K, 3)
        """

        means: torch.Tensor
        quats: torch.Tensor
        scales: torch.Tensor
        opacities: torch.Tensor
        colors: torch.Tensor

    def _extract_gaussian_parameters(
        self, gaussian_params: dict[str, torch.Tensor], device: torch.device
    ) -> _GaussianParameters:
        """Extract and prepare Gaussian parameters for gsplat.

        Note: gaussian_params from collect_gaussian_parameters() contains
        post-activated values via get_*() methods:
        - positions: raw (no activation)
        - rotations: rotation_activation applied (default: normalize)
        - scales: scale_activation applied (default: exp)
        - densities: density_activation applied (default: sigmoid)

        Args:
            gaussian_params: Dict with Gaussian parameters
            device: Target device for tensors

        Returns:
            _GaussianParameters named tuple with validated and prepared parameters

        Raises:
            AssertionError: If input shapes are invalid
        """
        # Validate required keys
        required_keys = ["positions", "rotations", "scales", "densities", "features"]
        for key in required_keys:
            assert key in gaussian_params, f"Missing required parameter: {key}"

        # Extract parameters (already post-activated)
        means = gaussian_params["positions"]  # (N, 3) raw positions
        quats = gaussian_params["rotations"]  # (N, 4) wxyz, normalized
        scales = gaussian_params["scales"]  # (N, 3) exp-activated
        opacities = gaussian_params["densities"]  # (N,) or (N, 1) sigmoid-activated
        features = gaussian_params["features"]  # (N, K*3) SH coefficients

        # Validate shapes
        n_gaussians = means.shape[0]
        assert means.ndim == 2 and means.shape[1] == 3, f"Expected positions shape (N, 3), got {means.shape}"
        assert quats.ndim == 2 and quats.shape == (n_gaussians, 4), (
            f"Expected rotations shape ({n_gaussians}, 4), got {quats.shape}"
        )
        assert scales.ndim == 2 and scales.shape == (n_gaussians, 3), (
            f"Expected scales shape ({n_gaussians}, 3), got {scales.shape}"
        )
        # Densities can be (N,) or (N, 1) - squeeze to (N,)
        if opacities.ndim == 2 and opacities.shape[1] == 1:
            opacities = opacities.squeeze(1)
        assert opacities.ndim == 1 and opacities.shape[0] == n_gaussians, (
            f"Expected densities shape ({n_gaussians},) or ({n_gaussians}, 1), got {opacities.shape}"
        )
        assert features.ndim == 2 and features.shape[0] == n_gaussians, (
            f"Expected features shape ({n_gaussians}, K*3), got {features.shape}"
        )
        assert features.shape[1] % 3 == 0, (
            f"Features dimension must be divisible by 3 (SH bands), got {features.shape[1]}"
        )

        # Handle RGB-only vs SH mode
        # RGB-only: features are (N, 3) - pass directly for sh_degree=None
        # SH mode: features are (N, K*3) where K>1 - reshape to (N, K, 3)
        num_sh_bands = features.shape[1] // 3
        if num_sh_bands == 1:
            # RGB-only mode: keep as (N, 3) for gsplat's RGB rendering
            colors = features
        else:
            # SH mode: reshape to (N, K, 3) for gsplat's SH rendering
            colors = features.reshape(n_gaussians, num_sh_bands, 3)

        return self._GaussianParameters(
            means=means.to(device),
            quats=quats.to(device),
            scales=scales.to(device),
            opacities=opacities.to(device),
            colors=colors.to(device),
        )

    @staticmethod
    def _build_lidar_frame_context(
        geometry_tensors: CachedDeviceTensorPool,
        row_elevations_rad_values: tuple[float, ...],
        column_azimuths_rad_values: tuple[float, ...],
        row_azimuth_offsets_rad_values: tuple[float, ...],
        spinning_frequency_hz: float,
        spinning_direction: gsplat.SpinningDirection,
        device: torch.device,
        n_bins_elevation: int,
        max_pts_per_tile: int,
        resolution_elevation: int,
        densification_factor_azimuth: int,
        _inference_mode: bool,
    ) -> gsplat.RowOffsetStructuredSpinningLidarFrameContext:
        # Build the context inside the renderer-owned cache so construction
        # itself is memoized. `_inference_mode` is cache-key only: the context
        # wraps tensors stamped by that ambient mode, so a val-first hit must
        # not be served to training.
        gsplat_params = gsplat.RowOffsetStructuredSpinningLidarModelParameters(
            row_elevations_rad=geometry_tensors.get(row_elevations_rad_values, device),
            column_azimuths_rad=geometry_tensors.get(column_azimuths_rad_values, device),
            row_azimuth_offsets_rad=geometry_tensors.get(row_azimuth_offsets_rad_values, device),
            spinning_frequency_hz=spinning_frequency_hz,
            spinning_direction=spinning_direction,
            fov_eps_factor=4,
        )
        angles_to_columns_map = gsplat.compute_lidar_angles_to_columns_map(gsplat_params)
        tiling = gsplat.compute_lidar_tiling(
            gsplat_params,
            n_bins_elevation=n_bins_elevation,
            max_pts_per_tile=max_pts_per_tile,
            resolution_elevation=resolution_elevation,
            densification_factor_azimuth=densification_factor_azimuth,
        )
        return gsplat.RowOffsetStructuredSpinningLidarFrameContext(
            model=gsplat_params,
            angles_to_columns_map=angles_to_columns_map,
            tiling=tiling,
        )

    def _build_lidar_params(
        self,
        ncore_params: RowOffsetStructuredSpinningLidarModelParameters,
        *,
        device: torch.device,
        n_bins_elevation: int,
        max_pts_per_tile: int,
        resolution_elevation: int,
        densification_factor_azimuth: int,
    ) -> gsplat.RowOffsetStructuredSpinningLidarFrameContext:
        """Build gsplat lidar acceleration structures directly from ncore parameters."""

        gsplat_spin = (
            gsplat.SpinningDirection.CLOCKWISE
            if ncore_params.spinning_direction == "cw"
            else gsplat.SpinningDirection.COUNTER_CLOCKWISE
        )
        # Coerce the sensor-geometry arrays to Python-float tuples via
        # `np.asarray(...).tolist()` so the cache key hashes by value
        # regardless of whether ncore stores ndarray, torch.Tensor, or list.
        # (A bare `tuple(tensor)` would yield 0-d tensors whose hash is
        # identity-based — missing the cache every call and reintroducing
        # the per-step H→D allocations this commit is removing.)
        return self._lidar_frame_contexts(
            tuple(np.asarray(ncore_params.row_elevations_rad).tolist()),
            tuple(np.asarray(ncore_params.column_azimuths_rad).tolist()),
            tuple(np.asarray(ncore_params.row_azimuth_offsets_rad).tolist()),
            ncore_params.spinning_frequency_hz,
            gsplat_spin,
            device,
            n_bins_elevation,
            max_pts_per_tile,
            resolution_elevation,
            densification_factor_azimuth,
            torch.is_inference_mode_enabled(),
        )

    def _build_viewmat(
        self, rendering_data: RenderingData, device: torch.device, use_midframe: bool = False
    ) -> torch.Tensor:
        """Build gsplat-compatible viewmat matrix.

        Args:
            rendering_data: Rendering data containing sensor poses
            device: Target device
            use_midframe: If True, interpolate start/end poses at t=0.5 to match
                mid-frame projection pose.

        Returns:
            viewmat: (4, 4) world-to-camera transformation matrix
        """

        if use_midframe:
            pose_start = rendering_data.poses_tquat_startend[0, 0]  # (7,)
            pose_end = rendering_data.poses_tquat_startend[0, 1]  # (7,)
            t_mid = 0.5 * (pose_start[:3] + pose_end[:3])
            q_mid = quat_slerp(pose_start[3:].unsqueeze(0), pose_end[3:].unsqueeze(0), 0.5).squeeze(0)
            pose_tquat = torch.cat([t_mid, q_mid]).unsqueeze(0)  # (1, 7)
        else:
            pose_tquat = rendering_data.poses_tquat_startend[0, 0].unsqueeze(0)  # (1, 7)

        # Convert tquat to inverted 4x4 matrix (world-to-camera)
        viewmat = se3pose_to_inverse_matrix(pose_tquat[..., :3], pose_tquat[..., 3:])

        # Prober integration - save data for testing the viewmat computation
        if (
            prober_result := get_global_prober()(
                0,  # step
                "gsplat_viewmat",
                poses_tquat=pose_tquat,
                viewmat_grad=viewmat,
            )
        ) is not None:
            # Connect the gradient probing to the rest of the computation graph
            (viewmat,) = prober_result

        return viewmat.squeeze(0).to(device)

    def _has_rolling_shutter(self, sensor_params: ConcreteCameraModelParametersUnion) -> bool:
        """Check if sensor has rolling shutter enabled."""
        if hasattr(sensor_params, "shutter_type"):
            return sensor_params.shutter_type != ShutterType.GLOBAL
        return False

    def _get_rolling_shutter_type(self, sensor_params: ConcreteCameraModelParametersUnion) -> gsplat.RollingShutterType:
        """Get rolling shutter type for gsplat.

        Returns:
            gsplat.RollingShutterType enum value
        """
        if not hasattr(sensor_params, "shutter_type"):
            return gsplat.RollingShutterType.GLOBAL

        # Map ncore ShutterType to gsplat RollingShutterType
        shutter_map = {
            ShutterType.GLOBAL: gsplat.RollingShutterType.GLOBAL,
            ShutterType.ROLLING_TOP_TO_BOTTOM: gsplat.RollingShutterType.ROLLING_TOP_TO_BOTTOM,
            ShutterType.ROLLING_LEFT_TO_RIGHT: gsplat.RollingShutterType.ROLLING_LEFT_TO_RIGHT,
            ShutterType.ROLLING_BOTTOM_TO_TOP: gsplat.RollingShutterType.ROLLING_BOTTOM_TO_TOP,
            ShutterType.ROLLING_RIGHT_TO_LEFT: gsplat.RollingShutterType.ROLLING_RIGHT_TO_LEFT,
        }
        return shutter_map.get(sensor_params.shutter_type, gsplat.RollingShutterType.GLOBAL)

    def _build_rolling_shutter_matrix(self, rendering_data: RenderingData, device: torch.device) -> torch.Tensor:
        """Build end viewmat for rolling shutter.

        Returns:
            viewmat_rs: (4, 4) world-to-camera matrix at frame end
        """
        # Extract sensor-to-NuRec pose (end)
        pose_tquat_end = rendering_data.poses_tquat_startend[0, 1].unsqueeze(0)  # (1, 7)

        # Convert tquat to inverted 4x4 matrix (world-to-camera)
        viewmat_rs = se3pose_to_inverse_matrix(pose_tquat_end[..., :3], pose_tquat_end[..., 3:])

        return viewmat_rs.squeeze(0).to(device)

    def collect_profilings(self) -> dict[str, float]:
        """Collect profiling information."""
        return {}
