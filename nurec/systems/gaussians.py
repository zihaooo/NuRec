# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import glob
import logging
import os

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, List, Mapping, Optional, cast

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

from einops import rearrange
from omegaconf import OmegaConf
from point_cloud_utils import TriangleMesh
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from scipy.spatial.transform import Rotation
from torchmetrics import Metric
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from ncore.data import ConcreteLidarModelParametersUnion, FrameTimepoint
from nurec.config.component_only import CameraFrameSelection, CameraLossNormalization
from nurec.config.nurec import NuRecConfig
from nurec.config.systems import (
    GaussiansSystemConfig,
)
from nurec.datasets.ncore import NCOREDataSource, NCORESequentialDataset, NCORETrainDataset
from nurec.datasets.summary import DataSourceSummary
from nurec.datasets.tracks import CuboidTracks
from nurec.harmonizer.training_controller import TrainingHarmonizerController
from nurec.instant_nurec.config.instant_nurec import InstantNuRecConfig
from nurec.libs.losses.models.reduce_functions.reduce_functions import MeanReduceFn
from nurec.libs.losses.orchestration.component_only_normalization import (
    ComponentOnlyLossNormalizationContext,
    configured_camera_loss_kinds_by_name,
    configured_distance_bounds_by_name,
    rescale_camera_mean_losses_to_original_valid_pixels,
)
from nurec.libs.losses.orchestration.config import LossAggregatorReturn, LossReturn, ReduceConfig
from nurec.metrics import AggregationMethod, MetricFactory, MetricManager, MetricType, PSNRMetric
from nurec.models.background import SkyEnvMapBackground
from nurec.models.gaussians.gaussians_composite import GaussiansComposite
from nurec.models.gaussians.initializations import (
    BaseInitialization,
    InProcessInstantNuRecInitialization,
    InstantNuRecPlyInitialization,
)
from nurec.models.gaussians.utils import Asset, PLYGaussianLoader
from nurec.models.post_processing import PPISPPostProcessing
from nurec.systems.active_tile_mapping_prewarmer import ActiveTileMappingPrewarmer
from nurec.systems.base import BaseSystemSO
from nurec.systems.registry import register as register_system
from nurec.utils.batch import DataAndRenderingBatch, DataBatch, generate_grid_2d_indices
from nurec.utils.component_only_mask import (
    apply_zero_mask,
    build_camera_loss_mask,
    mask_has_any_kept_pixels,
)
from nurec.utils.component_only_supervision import (
    find_local_eligible_camera_pairs,
    gather_eligible_camera_pairs,
    install_eligible_camera_plan,
    resolve_supervised_camera_samplers,
)
from nurec.utils.geometry import se3_matrix_inverse, se3_matrix_to_tquat
from nurec.utils.io.checkpoint import reduce_precision_to_fp16, serialize_checkpoint, strip_optimizer_state
from nurec.utils.lidar_post_processing import angle_gradient_filter, distance_based_filter
from nurec.utils.media_log import media_backend
from nurec.utils.misc import crop_mask_border, rank_zero_only, to_numpy, to_torch, unpack_optional
from nurec.utils.profiling import ScopedTimer
from nurec.utils.selective_adam_visibility import AsyncVisibilityUnion
from nurec.utils.trainer import adjust_step_for_world_size, distributed_context
from nurec.utils.types import (
    Checkpoint,
    GaussiansCompositeReturn,
    GaussiansRenderReturn,
    PointCloud,
    RayFlags,
)
from nurec.utils.visualize import (
    DEFAULT_PS_VIEW_JSON_STR,
    draw_text_overlay,
    save_video,
    scalar2img,
    scalar2rgb,
    sem2img,
)
from nurec.visualdebugger import VisualDebugger, get_visualdebugger


log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from nurec.datasets.samplers.holdout import CoveragePlan


# Per-system producer contract: consume the fully materialized nested InstantNuRec
# config and outer datasource summary, then return a device-local,
# NuRec-coordinate payload for the Gaussian layer initializers to borrow.
InProcessInstantNuRecInitializer = Callable[[InstantNuRecConfig, DataSourceSummary], PLYGaussianLoader]


def _validate_in_process_instant_nurec_consumers(config: NuRecConfig) -> None:
    """Reject competing disk sources before running the in-process producer."""
    has_consumer = False
    for layer_name, layer in config.model.layers.items():
        if layer.initialization is None:
            continue
        initializer_type = BaseInitialization.GAUSSIANS_INIT_VARIANTS.get(layer.initialization.name)
        if initializer_type is not None and issubclass(initializer_type, InstantNuRecPlyInitialization):
            if not issubclass(initializer_type, InProcessInstantNuRecInitialization):
                raise ValueError(
                    "in_process_instant_nurec cannot be combined with a disk instant_nurec PLY initializer: "
                    f"model.layers.{layer_name}.initialization.name={layer.initialization.name}"
                )
            has_consumer = True
            configured_path = (
                layer.initialization.path
                if "path" in layer.initialization and not OmegaConf.is_missing(layer.initialization, "path")
                else None
            )
            if configured_path is not None:
                raise ValueError(
                    "in_process_instant_nurec cannot be combined with explicit instant_nurec PLY path "
                    f"model.layers.{layer_name}.initialization.path={configured_path!r}"
                )
    if not has_consumer:
        raise ValueError(
            "in_process_instant_nurec requires at least one InProcessInstantNuRecInitialization-derived layer initializer"
        )


_mean_loss_value = MeanReduceFn(ReduceConfig(name="mean"))


@register_system("gaussians-system")
class GaussiansSystem(BaseSystemSO):
    config: GaussiansSystemConfig
    model: GaussiansComposite
    test_out_types: list[str]

    def __init__(
        self,
        config: NuRecConfig,
        in_process_instant_nurec_initializer: InProcessInstantNuRecInitializer | None = None,
    ) -> None:
        super().__init__(config)
        # Run entrypoints inject this dependency only when in-process InstantNuRec is configured.
        # Store it per system and defer execution until ``setup()``, after
        # Lightning assigns the runtime device; ``None`` preserves the ordinary
        # NuRec path.
        self._in_process_instant_nurec_initializer: InProcessInstantNuRecInitializer | None = (
            in_process_instant_nurec_initializer
        )

        self.config.record_timings.train_interval = adjust_step_for_world_size(
            config.trainer, self.config.record_timings.train_interval
        )
        self.config.record_timings.val_interval = adjust_step_for_world_size(
            config.trainer, self.config.record_timings.val_interval
        )
        self.config.record_quality_metrics.psnr_interval = adjust_step_for_world_size(
            config.trainer, self.config.record_quality_metrics.psnr_interval
        )

        log.info(
            f"GaussiansSystem/record_timings: train_interval={self.config.record_timings.train_interval} val_interval={self.config.record_timings.val_interval}"
        )
        log.info(
            f"GaussiansSystem/record_quality_metrics: psnr_interval={self.config.record_quality_metrics.psnr_interval}"
        )

        # The composite and setup-time in-process InstantNuRec handoff must share this exact
        # object: setup publishes the transient payload into it, and the layer
        # initializers consume that payload during Gaussian materialization.
        self.datasource_summary: DataSourceSummary = DataSourceSummary.from_datasource(self.datamodule.get_datasource())
        # In-process InstantNuRec lifecycle is tracked per GaussiansSystem instance:
        #   (False, False): no payload has been published. This includes the
        #   normal bypass path and a producer failure before it returns, so a
        #   later setup call is not blocked by these flags.
        #   (True, False): a payload was published, but outer setup or Gaussian
        #   materialization failed. The payload is cleared in ``finally``, but
        #   layers or strategies may now be partially mutated, so retrying the
        #   same system instance is unsafe.
        #   (True, True): Gaussian materialization completed and the transient
        #   payload was cleared. Later Lightning setup calls reuse that state
        #   without rerunning the producer.
        # (False, True) is impossible. There is no stable published-but-not-
        # initialized state: the payload is borrowed only during setup and is
        # always cleared before setup returns or raises.
        self._in_process_instant_nurec_initialization_attempted: bool = False
        self._in_process_instant_nurec_initialized: bool = False

        self.model = GaussiansComposite(
            config.model,
            config.trainer,
            self.datasource_summary,
            init_from_datasource="train" in config.mode and not config.resume,
            component_only=config.component_only,
        )
        # Check if we are in the training mode
        is_train = "train" in config.mode

        if is_train:
            # Initialize the loss, semantic loss requires the semantic_classes_map
            self.loss.initialize(self.datamodule.train_dataset)

        # TODO: Clean this up. This is kind of a messy workaround needed for max extent below
        self.max_dist_m: float | None = None
        if config.dataset.name == "ncore":
            self.max_dist_m = config.dataset.max_dist_m

        self._training_harmonizer = TrainingHarmonizerController(self)

        self.track_padding_m = config.dataset.valid_pixels_cuboid_track_params.track_padding_m

        # One-shot guard so the "no visibility mask available" fallback warning is emitted
        # at most once per system instance. Kept as an instance attribute (not a class-level
        # annotation) so per-instance suppression is explicit and not shared across systems.
        self._selective_adam_no_visibility_warned: bool = False
        self._pending_selective_adam_visibility: AsyncVisibilityUnion | None = None
        self._pending_selective_adam_node_ids: tuple[str, ...] = ()
        self._pending_selective_adam_layer_ids: frozenset[str] = frozenset()
        self._serialization_state_dict_hook_handle: Any | None = None
        self._active_tile_mapping_prewarmer = ActiveTileMappingPrewarmer()

    def setup(self, stage: str) -> None:
        # Lightning has established the process group and root-device context at
        # this point, so module initialization may run here. Checkpoint
        # module-state restore and DDPStrategySO.setup/DDP wrapping happen only
        # after this hook returns, so setup must not launch background work that
        # reads model parameters.
        # NOTE: If `init_from_datasource` is not used, this function can be called during construction.
        #       This is done automatically in the `GaussiansComposite` model.
        in_process_instant_nurec = self.cached_config.in_process_instant_nurec
        uses_in_process_instant_nurec = (
            self.model.gaussians_strategy.init_from_datasource and in_process_instant_nurec is not None
        )
        if (
            uses_in_process_instant_nurec
            and self._in_process_instant_nurec_initialization_attempted
            and not self._in_process_instant_nurec_initialized
        ):
            raise RuntimeError(
                "A previous in-process InstantNuRec initialization attempt failed or did not complete; "
                "setup cannot be retried safely on the same GaussiansSystem instance"
            )

        attempting_in_process_instant_nurec = (
            uses_in_process_instant_nurec and not self._in_process_instant_nurec_initialized
        )
        try:
            if attempting_in_process_instant_nurec:
                if self._in_process_instant_nurec_initializer is None:
                    raise RuntimeError(
                        "in_process_instant_nurec requires an in_process_instant_nurec_initializer when constructing GaussiansSystem"
                    )
                _validate_in_process_instant_nurec_consumers(self.cached_config)
                assert in_process_instant_nurec is not None
                self.datasource_summary.runtime_instant_nurec_ply_consumed = False
                runtime_instant_nurec_ply = self._in_process_instant_nurec_initializer(
                    in_process_instant_nurec, self.datasource_summary
                )
                # The selected memory-only consumer requires a payload. Reject
                # an empty producer result before outer setup mutates any state.
                if runtime_instant_nurec_ply is None:
                    raise RuntimeError("in_process_instant_nurec initializer returned no payload")
                self.datasource_summary.runtime_instant_nurec_ply = runtime_instant_nurec_ply
                # From this point onward, outer setup or Gaussian
                # materialization can leave partially initialized state.
                self._in_process_instant_nurec_initialization_attempted = True

            # BaseSystem.setup moves the outer NuRec model to CUDA. Run and tear
            # down Kelvin first so both full models are never resident together.
            super().setup(stage)
            if stage == "fit":
                self._active_tile_mapping_prewarmer.reset()

            # A successful in-memory initialization uses the canonical payload
            # to materialize trainable Gaussian parameters and then releases it.
            # Lightning may call setup again on the same system; do not rerun
            # memory-only initializers after their payload has been released.
            if not self._in_process_instant_nurec_initialized:
                with ScopedTimer("GaussiansSystem/setup/model_initialization"):
                    self.model.initialize_strategy_and_maybe_gaussians()
                if attempting_in_process_instant_nurec:
                    if not self.datasource_summary.runtime_instant_nurec_ply_consumed:
                        raise RuntimeError(
                            "In-process InstantNuRec Gaussian materialization completed without consuming its in-memory payload"
                        )
                    # Initializers have materialized their trainable, rank-local tensors.
                    self._in_process_instant_nurec_initialized = True
        finally:
            if attempting_in_process_instant_nurec:
                # Consumers only borrow the prepared payload during materialization.
                # Release it after either success or failure.
                self.datasource_summary.runtime_instant_nurec_ply = None
                self.datasource_summary.runtime_instant_nurec_ply_consumed = False
        self._validate_training_harmonizer_component_only_compatibility()
        self._validate_component_only_loss_normalization_compatibility()
        self._register_serialization_state_dict_hook()
        if stage == "fit":
            self._apply_init_parts()

        # TODO: Move the metric managers to BaseSystemSO
        # Training PSNR is logged immediately and is never aggregated or exported.
        # Keep it out of MetricManager so the hot path neither stores history nor
        # serializes a CUDA scalar with Tensor.item().
        self.train_psnr_metric = PSNRMetric(device=torch.device("cuda"), data_range=1)

        # Initialize validation metrics
        self.val_metric_manager = MetricManager(device=torch.device("cuda"))
        self.val_metric_manager.register_metric("psnr", MetricFactory[MetricType.PSNR](data_range=1))
        self.val_metric_manager.register_metric(
            "cpsnr", MetricFactory[MetricType.CPSNR](data_range=1, aggregation_methods=AggregationMethod.WEIGHTED_MEAN)
        )
        self.val_metric_manager.register_metric("lidar_common", MetricFactory[MetricType.LIDAR_COMMON]())

        self.criterions: dict[str, Metric] = {
            "ssim": StructuralSimilarityIndexMeasure().to("cuda"),
            "lpips": LearnedPerceptualImagePatchSimilarity(normalize=True).to("cuda"),
        }

        self._training_harmonizer.maybe_init_harmonizer_model()

    def _setup_component_only_camera_supervision(self) -> None:
        """Build and install the fixed eligible-frame index before dataloader workers start."""
        component_only = self.model.component_only
        if (
            component_only is None
            or not component_only.enabled
            or component_only.camera_supervision.frame_selection != CameraFrameSelection.supervised_only
            or "train" not in self.mode
        ):
            return
        if getattr(self, "_component_only_supervised_frames_ready", False):
            return

        dataset = self.datamodule.train_dataset
        if not isinstance(dataset, NCORETrainDataset):
            raise TypeError(
                "component_only camera frame_selection='supervised_only' currently requires NCORETrainDataset"
            )
        samplers = resolve_supervised_camera_samplers(dataset)
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        rank = torch.distributed.get_rank() if distributed else 0
        world_size = torch.distributed.get_world_size() if distributed else 1
        local_pairs: list[tuple[str, int]] = []
        local_error: Exception | None = None
        try:
            local_pairs = find_local_eligible_camera_pairs(
                samplers,
                self.model,
                device=self.device,
                rank=rank,
                world_size=world_size,
            )
        except Exception as error:
            # Every rank must enter the status collective. Re-raising here can strand healthy
            # peers in all_gather_object until the distributed timeout expires.
            local_error = error
        eligible_pairs = gather_eligible_camera_pairs(local_pairs, local_error=local_error)
        restriction = install_eligible_camera_plan(samplers, eligible_pairs)

        scale_samples = (
            component_only.camera_supervision.supervised_only.scale_samples_per_epoch_by_frame_selection_fraction
        )
        configured_samples = dataset.configured_n_samples_per_epoch
        effective_samples = dataset.apply_samples_per_epoch_scaling(
            eligible_pairs=restriction.eligible_total,
            total_pairs=restriction.original_total,
            scale_samples_per_epoch_by_frame_selection_fraction=scale_samples,
        )
        self._lr_scheduler_epoch_steps = None
        if scale_samples:
            samples_per_distributed_batch = world_size * self.train_batch_size
            configured_steps = -(-configured_samples // samples_per_distributed_batch)
            effective_steps = -(-effective_samples // samples_per_distributed_batch)
            if configured_steps > effective_steps:
                if any(
                    lr_config.interval == "step"
                    and isinstance(lr_config.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
                    and configured_steps > effective_steps * lr_config.frequency
                    for lr_config in self.trainer.strategy.lr_scheduler_configs
                ):
                    raise ValueError(
                        "component_only supervised_only sample scaling cannot accelerate a step-interval "
                        "ReduceLROnPlateau scheduler because replaying one metric would change its semantics."
                    )
            self._lr_scheduler_epoch_steps = (configured_steps, effective_steps)
        self._component_only_camera_coverage_plan = restriction.plan
        self._set_component_only_camera_coverage_epoch(self._component_only_camera_coverage_setup_epoch())
        self._component_only_supervised_frames_ready = True
        log.info(
            "Component-only supervised camera frames: retained %d/%d reachable (camera, frame) pairs; "
            "samples_per_epoch=%d (configured=%d, scaled=%s), LR scheduler epoch steps=%s. "
            "Sampling remains uniform per eligible pair.",
            restriction.eligible_total,
            restriction.original_total,
            effective_samples,
            configured_samples,
            scale_samples,
            self._lr_scheduler_epoch_steps,
        )

    def _component_only_camera_coverage_setup_epoch(self) -> int:
        """Return the epoch whose batches Lightning will request after fit setup.

        At an epoch-end checkpoint, Lightning restores ``current_epoch`` to the just-finished epoch,
        then primes the next dataloader iterator before advancing it. ``processed`` already identifies
        that next epoch, so it prevents resumed workers from briefly replaying the previous coverage
        segment. The fallback covers fresh trainers and lightweight test doubles.
        """
        current_epoch = int(self.current_epoch)
        fit_loop = getattr(self.trainer, "fit_loop", None)
        epoch_progress = getattr(fit_loop, "epoch_progress", None)
        current_progress = getattr(epoch_progress, "current", None)
        processed = getattr(current_progress, "processed", current_epoch)
        return max(current_epoch, int(processed))

    def _set_component_only_camera_coverage_epoch(self, epoch: int) -> None:
        """Publish the epoch's global coverage offset to persistent dataloader workers."""
        coverage_plan: CoveragePlan | None = getattr(self, "_component_only_camera_coverage_plan", None)
        if coverage_plan is not None:
            coverage_plan.set_batch_index_offset(epoch * len(self.datamodule.train_dataset))

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        self._set_component_only_camera_coverage_epoch(self.current_epoch)

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        # Publish before Lightning can construct the next epoch's iterator and prefetch worker requests.
        self._set_component_only_camera_coverage_epoch(self.current_epoch + 1)

    def _register_serialization_state_dict_hook(self) -> None:
        if self._serialization_state_dict_hook_handle is None:
            self._serialization_state_dict_hook_handle = self.register_state_dict_post_hook(
                self._apply_gaussian_serialization_overrides
            )

    @staticmethod
    def _apply_gaussian_serialization_overrides(
        module: torch.nn.Module,
        state_dict: dict[str, Any],
        prefix: str,
        _local_metadata: dict[str, Any],
    ) -> None:
        """Bound exported Gaussian state without mutating the live model."""
        system = cast(GaussiansSystem, module)
        overrides = system.model.gaussians_strategy.serialization_state_dict_overrides(system.model.gaussians_nodes)
        key_prefix = f"{prefix}model.gaussians_nodes."
        for relative_key, value in overrides.items():
            key = f"{key_prefix}{relative_key}"
            if key not in state_dict:
                raise KeyError(f"Missing Gaussian serialization state key: {key}")
            state_dict[key] = value

    def _validate_training_harmonizer_component_only_compatibility(self) -> None:
        """Reject novel-view training when its component mask or frame index is source-view-specific.

        Harmonizer novel views shift the camera pose but keep the source-view
        labels. Consequently, source semantic labels do not describe the novel
        view and cannot safely be used to construct a component-only loss mask.
        Bbox-only masks can be projected from a novel pose, but ``supervised_only``
        eligibility was computed for the source pose and cannot guarantee the
        projected mask remains non-empty.
        """
        training_config = self.harmonizer_config.training
        mask_spec = self.model._component_only_mask_spec
        has_novel_view_training = training_config.enabled and training_config.novel_view_poses
        if not has_novel_view_training or mask_spec is None:
            return
        component_only = self.model.component_only
        if (
            component_only is not None
            and component_only.camera_supervision.frame_selection == CameraFrameSelection.supervised_only
        ):
            raise ValueError(
                "component_only camera frame_selection='supervised_only' is incompatible with Harmonizer "
                "novel-view training because source-view frame eligibility does not guarantee a non-empty "
                "mask after the camera pose changes. Disable harmonizer.training, remove "
                "harmonizer.training.novel_view_poses, or use camera frame_selection='all_frames'."
            )
        if mask_spec.needs_semantic:
            raise ValueError(
                "component_only requires per-pixel semantic labels, which are unavailable for Harmonizer "
                "novel-view training. Disable harmonizer.training, remove "
                "harmonizer.training.novel_view_poses, or use bbox/bbox_expanded-only component masks "
                "without sky or residual supervision."
            )

    def _validate_component_only_loss_normalization_compatibility(self) -> None:
        """Reject non-local losses that cannot honor the all-valid-pixels denominator."""
        component_only = self.model.component_only
        if (
            component_only is None
            or not component_only.enabled
            or "train" not in self.mode
            or self.model._component_only_mask_spec is None
            or component_only.camera_supervision.loss_normalization != CameraLossNormalization.all_valid_pixels
        ):
            return
        configured_losses = self.loss.losses + self.loss.in_cuda.losses
        if any(loss.name.startswith("lpips_") for loss in configured_losses):
            raise ValueError(
                "component_only camera loss_normalization='all_valid_pixels' is incompatible with LPIPS. "
                "LPIPS uses non-local feature reductions and cannot preserve the original valid-pixel "
                "denominator after component masking. Disable LPIPS or use "
                "loss_normalization='supervised_pixels'."
            )

    def get_ssim_criterion(self) -> StructuralSimilarityIndexMeasure:
        criterion = self.criterions["ssim"]
        assert isinstance(criterion, StructuralSimilarityIndexMeasure)
        return criterion

    def get_lpips_criterion(self) -> LearnedPerceptualImagePatchSimilarity:
        criterion = self.criterions["lpips"]
        assert isinstance(criterion, LearnedPerceptualImagePatchSimilarity)
        return criterion

    def _apply_init_parts(self) -> None:
        """system.init_parts: copy parameters from another checkpoint and freeze some (once per system).

        Runs in setup for training only (stage "fit"), after the model is built and before Lightning restores any resume
        checkpoint, so a resumed checkpoint still overwrites the copied values (frozen flags stay). Validation and
        rendering restore everything from their checkpoint and skip it: there, lazy Gaussian layers are not
        materialized at setup. Fails loudly on a missing key or shape mismatch rather than training with a silently
        different model."""
        cfg = getattr(self.cached_config.system, "init_parts", None)
        if cfg is None or getattr(self, "_init_parts_applied", False):
            return
        self._init_parts_applied = True
        if cfg.load:
            ckpt = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
            src = ckpt["state_dict"]
            picked = {k: v for k, v in src.items() if any(k.startswith(p) for p in cfg.load)}
            # only the selected tensors: a full state_dict() would touch uninitialized (lazy) parameters
            own = {
                k: v
                for k, v in list(self.named_parameters()) + list(self.named_buffers())
                if any(k.startswith(p) for p in cfg.load)
            }
            if not picked:
                raise ValueError(f"init_parts: no key of {cfg.checkpoint} starts with {cfg.load}")
            for k, v in picked.items():
                if k not in own:
                    raise KeyError(f"init_parts: {k} is not in this model")
                if torch.is_tensor(v) and tuple(own[k].shape) != tuple(v.shape):
                    raise ValueError(f"init_parts: {k} has shape {tuple(v.shape)}, this model {tuple(own[k].shape)}")
            self.load_state_dict(picked, strict=False)
            log.info("init_parts: copied %d tensors from %s: %s", len(picked), cfg.checkpoint, sorted(picked))
            del ckpt
        frozen = []
        for name, param in self.named_parameters():
            if any(name.startswith(p) for p in cfg.freeze):
                param.requires_grad_(False)
                frozen.append(name)
        if cfg.freeze and not frozen:
            raise ValueError(f"init_parts: no parameter starts with {cfg.freeze}")
        if frozen:
            log.info("init_parts: froze %d parameters: %s", len(frozen), frozen)

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # For NuRec's Lightning strategies, checkpoint weights are restored before this hook while
        # FitLoop has not yet built its train dataloader/distributed sampler. The resumed calibration
        # and track state therefore determines eligibility, and the effective dataset length is visible
        # when Lightning snapshots the epoch size.
        self._setup_component_only_camera_supervision()

    def on_train_start(self) -> None:
        super().on_train_start()
        if self.resume:
            return
        self.model.on_train_from_scratch_start(self)
        # Second component-inventory line: lazy Gaussian layers
        # (background, road — initialized from lidar on first use) have
        # materialized by now, so this one shows real parameter counts
        # where the construction-time line showed "0 (lazy)".
        self.model._log_component_inventory(when="post-init")

    def on_train_batch_end(
        self,
        outputs: torch.Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_local_idx: int,
    ) -> None:
        super().on_train_batch_end(outputs, batch, batch_local_idx)
        with self._mcmc_exception_context():
            self.model.raise_if_mcmc_publication_failed()
            with ScopedTimer("GaussiansSystem/model/update_step_train_batch_end"):
                self.model.update_step_train_batch_end(self.current_epoch, self.global_step, batch, self)
        # Memory-based garbage collection
        if self._should_collect_garbage():
            with ScopedTimer("GaussiansSystem/_should_collect_garbage"):
                gc.collect()
                torch.cuda.empty_cache()

    def on_train_end(self) -> None:
        self.model.drain_fused_gather_generation_plan_refresh()
        super().on_train_end()

    def teardown(self, stage: str) -> None:
        self.model.drain_fused_gather_generation_plan_refresh()
        super().teardown(stage)

    def _should_collect_garbage(self) -> bool:
        """Check if garbage collection should be triggered based on memory usage"""

        if not hasattr(self.config, "collect_garbage_mem_usage") or self.config.collect_garbage_mem_usage is None:
            return False

        # Only check memory every N steps to reduce overhead
        check_interval = getattr(self.config, "collect_garbage_check_interval", 250)
        if self.global_step % check_interval != 0:
            return False

        # Get memory information using PyTorch
        free_mem, total_mem = torch.cuda.mem_get_info()
        used_mem = total_mem - free_mem
        usage_ratio = used_mem / total_mem

        # Log GPU memory info when we check
        used_gb = used_mem / (1024**3)
        total_gb = total_mem / (1024**3)
        usage_percent = usage_ratio * 100
        log.info(f"GPU Memory: {usage_percent:.1f}% ({used_gb:.1f}GB / {total_gb:.1f}GB)")

        return usage_ratio > self.config.collect_garbage_mem_usage

    def _mcmc_exception_context(self):
        """Coordinate rank-local MCMC failures before another rank can enter a collective."""
        return distributed_context(cast(Trainer, self._trainer))

    def _raise_if_mcmc_publication_failed(self) -> None:
        with self._mcmc_exception_context():
            self.model.raise_if_mcmc_publication_failed()

    def preflight_distributed_checkpoint_state_dict(self) -> None:
        self._raise_if_mcmc_publication_failed()

    def forward(
        self,
        batch: DataAndRenderingBatch,
        is_training_batch: bool = False,
        gaussian_nodes: Optional[list[str]] = None,
        track_transform_ids: Optional[List[str]] = None,
        track_transforms: Optional[torch.Tensor] = None,
        render_background: bool = True,
        enable_pp: bool = True,
        clamp_output: bool = True,
        prepare_selective_adam_visibility: bool = False,
    ) -> GaussiansCompositeReturn:
        self._raise_if_mcmc_publication_failed()
        return self.model(
            batch,
            is_training_batch=is_training_batch,
            gaussian_nodes=gaussian_nodes,
            track_transform_ids=track_transform_ids,
            track_transforms=track_transforms,
            render_background=render_background,
            enable_pp=enable_pp,
            clamp_output=clamp_output,
            global_step=self.global_step,
            prepare_selective_adam_visibility=prepare_selective_adam_visibility,
        )

    def training_step(self, batch: DataAndRenderingBatch, batch_local_idx: int):
        """Prewarm once after DDP's first buffer sync and before the first render.

        Lightning invokes this method through ``DistributedDataParallel``.
        DDP's pre-forward phase has therefore broadcast ordinary model buffers
        before control reaches here. Neither ``setup`` nor ``on_train_start``
        provides that first-forward synchronization boundary.
        """
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            self._active_tile_mapping_prewarmer.run_once(
                model=self.model,
                train_dataset=self.datamodule.train_dataset,
                device=self.device,
                trainer=self.trainer,
            )
        return super().training_step(batch, batch_local_idx)

    @ScopedTimer("GaussiansSystem/training_losses")
    def training_losses(self, batch: DataAndRenderingBatch, batch_local_idx: int) -> LossAggregatorReturn:
        """Stage one rank-symmetric visibility union for the base optimizer hooks.

        :class:`BaseSystem` launches the staged WORLD collective exactly once
        after coordinated forward-error handling, then resolves it after
        backward and before any optimizer step. A pending value here means a
        caller bypassed that required lifecycle; fail closed rather than
        discarding work that may already be in flight.
        """
        self._raise_if_mcmc_publication_failed()

        assert self._pending_selective_adam_visibility is None, (
            "A previous SelectiveAdam visibility union was not consumed by "
            "BaseSystem.training_step before the next training forward"
        )
        node_ids, selective_layer_ids = self._get_selective_adam_step_layers()
        prepare_selective_adam_visibility = bool(selective_layer_ids) and not self.model.controller_distillation_active

        # Training-time Harmonizer controller
        use_training_harmonizer = self._training_harmonizer.should_use_novel_view()
        maybe_training_batch = None
        if use_training_harmonizer:
            maybe_training_batch = self._training_harmonizer.get_next_novel_batch(batch)
            if maybe_training_batch is not None:
                batch = maybe_training_batch

        # Call the forward pass on all rays and maybe time call
        @ScopedTimer("GaussiansSystem/training_losses/forward_call")
        def forward_call() -> GaussiansCompositeReturn:
            if self.model.controller_distillation_active:
                return self._forward_controller_distillation(batch)
            return self(
                batch,
                is_training_batch=True,
                prepare_selective_adam_visibility=prepare_selective_adam_visibility,
            )

        out = forward_call()

        # If training-time Harmonizer is enabled and used this step, replace gt with harmonizer target computed from dirty render
        if use_training_harmonizer and maybe_training_batch is not None:
            self._training_harmonizer.apply_harmonizer_as_target(batch, out)

        loss_return = self._training_losses(batch, out, batch_local_idx)

        # Publish the unlaunched work only after all forward/loss construction
        # that can raise locally has succeeded. BaseSystem's next coordinated
        # block owns launch, backward, resolve, and clearing.
        if prepare_selective_adam_visibility:
            assert out.selective_adam_visibility is not None, (
                "SelectiveAdam visibility was requested but the Gaussian forward did not prepare it"
            )
            self._stage_pending_selective_adam_visibility(
                out.selective_adam_visibility,
                node_ids=node_ids,
                selective_layer_ids=selective_layer_ids,
            )

        return loss_return

    def _forward_controller_distillation(self, batch: DataAndRenderingBatch) -> GaussiansCompositeReturn:
        """Render the frozen scene, then train only the PPISP controller."""
        with torch.no_grad():
            out = self(
                batch,
                is_training_batch=True,
                enable_pp=False,
                clamp_output=False,
            )

        for post_processing in self.model.post_processings:
            if isinstance(post_processing, PPISPPostProcessing):
                post_processing.forward_rendered_batch(
                    out,
                    batch,
                    global_step=self.global_step,
                    is_training_batch=True,
                )

        self.model._maybe_saturate_rgb(out)

        return out

    def _sync_selective_enabled(self) -> None:
        """Push per-group ``selective_enabled`` flags from config to every Gaussian optimizer.

        Called once per training batch so that flipping ``self.model.config.gaussian_*_selective``
        at runtime takes effect on the next step without swapping optimizer classes.
        Uses :meth:`GaussiansComposite._initial_selective_enabled` as the single source
        of truth for per-group config lookup.
        """
        from nurec.utils.optim import SelectiveAdam, classify_gaussian_optimizer_group

        for node in self.model.gaussians_nodes.values():
            for opt_config in getattr(node, "optimizers", []):
                opt = opt_config["optimizer"]
                if not isinstance(opt, SelectiveAdam):
                    continue
                group = classify_gaussian_optimizer_group(opt.param_groups)
                opt.selective_enabled = self.model._initial_selective_enabled(group)

    def _get_selective_adam_step_layers(self) -> tuple[list[str], list[str]]:
        """Return render-order nodes and layers needing selective updates this step."""
        from nurec.utils.optim import SelectiveAdam

        self._sync_selective_enabled()
        node_ids = self.model.get_gaussians_node_ids()
        selective_layer_ids = [
            node_id
            for node_id in node_ids
            if any(
                isinstance(opt_config["optimizer"], SelectiveAdam) and opt_config["optimizer"].selective_enabled
                for opt_config in getattr(self.model.gaussians_nodes[node_id], "optimizers", [])
            )
        ]
        if not selective_layer_ids:
            return node_ids, selective_layer_ids

        # BaseSystem's manual loop (automatic_optimization=False) runs zero_grad ->
        # forward/stage -> launch -> backward -> resolve/set_visibility -> optimizer.step
        # once per batch. SelectiveAdam consumes and clears that mask in step(). PL
        # gradient accumulation would let the last micro-batch's mask overwrite earlier
        # masks while retaining their gradients, so reject it until visibility is
        # OR-accumulated across micro-batches.
        if self.is_trainer_attached:
            accumulate_grad_batches = getattr(self.trainer, "accumulate_grad_batches", 1)
            assert accumulate_grad_batches == 1, (
                "SelectiveAdam is incompatible with gradient accumulation "
                f"(accumulate_grad_batches={accumulate_grad_batches}). The per-step visibility mask "
                "would only reflect the last micro-batch, so the optimizer step would update "
                "parameters using a mask that does not match the accumulated gradients. Set "
                "accumulate_grad_batches=1, or OR-accumulate per-element visibility across "
                "micro-batches and call set_visibility() once before the optimizer step."
            )
        return node_ids, selective_layer_ids

    def _warn_selective_adam_dense_fallback(
        self,
        selective_layer_ids: list[str],
    ) -> None:
        if self._selective_adam_no_visibility_warned:
            return
        self._selective_adam_no_visibility_warned = True
        log.warning(
            "SelectiveAdam is configured for layers %s, but no valid visibility union could be constructed "
            "(no sources were provided, or a provided source had an unexpected Gaussian count). "
            "Falling back to a dense all-visible update on every rank (equivalent to standard Adam). "
            "Check model.renderer.outputs.scene.enable_visibility and "
            "model.renderer.outputs.scene.enable_cumulated_weights. This warning is only emitted on ranks "
            "whose local sources were missing or malformed; other ranks are forced dense without logging. "
            "The dashboard-visible symptom is gaussians/{node}_visible jumping to 100%% visible.",
            selective_layer_ids,
        )

    def _stage_pending_selective_adam_visibility(
        self,
        pending: AsyncVisibilityUnion,
        *,
        node_ids: list[str] | tuple[str, ...],
        selective_layer_ids: list[str] | frozenset[str],
    ) -> None:
        """Publish one prepared union for the base training-step hooks."""
        assert self._pending_selective_adam_visibility is None, (
            "A previous SelectiveAdam visibility union is still pending"
        )
        self._pending_selective_adam_visibility = pending
        self._pending_selective_adam_node_ids = tuple(node_ids)
        self._pending_selective_adam_layer_ids = frozenset(selective_layer_ids)
        if pending.local_source_missing:
            self._warn_selective_adam_dense_fallback(sorted(selective_layer_ids))

    def _stage_selective_adam_visibility(
        self,
        visibility: torch.Tensor,
        *,
        node_ids: list[str] | tuple[str, ...],
        selective_layer_ids: list[str] | frozenset[str],
    ) -> None:
        """Compact one owner-local mask into layer views and stage optimizers."""
        from nurec.utils.optim import SelectiveAdam

        # `node_ids` is the render/collection order captured before the forward. resolve()
        # concatenates this rank's per-layer ranges in that same order, so total_count and
        # the slicing loop below must keep it unchanged. Shape checks cannot detect an
        # equal-length layer reorder.
        total_count = sum(self.model.gaussians_nodes[node_id].get_num_gaussians() for node_id in node_ids)
        assert visibility.ndim == 1, (
            f"SelectiveAdam visibility must be a flat tensor with shape ({total_count},), "
            f"got {tuple(visibility.shape)}."
        )
        assert total_count == visibility.shape[0], (
            f"SelectiveAdam visibility length {visibility.shape[0]} does not match the total "
            f"Gaussian count {total_count} summed over nodes {node_ids}. The visibility tensor "
            "ordering must match get_gaussians_node_ids() iteration order used for slicing. If the counts "
            "changed since the forward, a topology mutation ran between forward and optimizer step, which "
            "this staging path does not support."
        )

        # Normalize once; per-layer slices are read-only views shared by enabled SelectiveAdam instances.
        visibility_mask = visibility.detach().bool()
        # Every layer advances the offset, but only enabled SelectiveAdam instances receive
        # a mask; dense groups synthesize an all-ones mask inside optimizer.step().
        offset = 0
        for node_id in node_ids:
            node = self.model.gaussians_nodes[node_id]
            count = node.get_num_gaussians()
            layer_vis = visibility_mask[offset : offset + count]
            offset += count
            if node_id not in selective_layer_ids:
                continue
            if self.logger_enabled:
                # Keep telemetry on CUDA; Lightning flushes it with the normal
                # step metrics without an eager scalar conversion here. This
                # reports final cross-rank union owner visibility.
                self.log(
                    f"gaussians/{node_id}_visible",
                    layer_vis.sum(dtype=torch.float32),
                    prog_bar=False,
                )
            for opt_config in getattr(node, "optimizers", []):
                opt = opt_config["optimizer"]
                if isinstance(opt, SelectiveAdam) and opt.selective_enabled:
                    opt.set_visibility(layer_vis)

        assert offset == visibility.numel(), (
            f"SelectiveAdam visibility slicing consumed {offset} of {visibility.numel()} "
            "elements; per-layer Gaussian counts must exactly partition the visibility tensor. If the counts "
            "changed since the forward, a topology mutation ran between forward and optimizer step, which "
            "this staging path does not support."
        )

    def _prepare_optimizers_after_backward(self) -> None:
        pending = self._pending_selective_adam_visibility
        if pending is None:
            return

        try:
            with ScopedTimer("GaussiansSystem/selective_adam_visibility/resolve_after_backward"):
                visibility = pending.resolve()
                self._stage_selective_adam_visibility(
                    visibility,
                    node_ids=self._pending_selective_adam_node_ids,
                    selective_layer_ids=self._pending_selective_adam_layer_ids,
                )
        finally:
            self._pending_selective_adam_visibility = None
            self._pending_selective_adam_node_ids = ()
            self._pending_selective_adam_layer_ids = frozenset()

    def _launch_optimizers_async_preparation(self) -> None:
        """Launch the staged visibility WORLD collective exactly once per step."""
        pending = self._pending_selective_adam_visibility
        if pending is not None:
            with ScopedTimer("GaussiansSystem/selective_adam_visibility/launch_async_all_reduce"):
                pending.launch()

    @ScopedTimer("GaussiansSystem/training_losses/_training_losses")
    def _training_losses(
        self, batch: DataAndRenderingBatch, out: GaussiansCompositeReturn, batch_local_idx: int
    ) -> LossAggregatorReturn:
        self.apply_alpha(batch, out)

        # Component-only optimization: zero pred/gt outside the enabled
        # components' responsible pixels. No-op in full training
        # (model._component_only_mask_spec is None). Training is the only
        # path that records the empty-mask probe.
        normalization_context = self._maybe_apply_component_only_mask(batch, out, record_coverage=True)

        # Compute the loss
        with ScopedTimer("_training_losses/loss_call"):
            loss_return = self.loss(results=out, target=batch, model=self.model, step=self.global_step)
            if normalization_context is not None:
                assert batch.data.camera is not None
                configured_losses = list(self.loss.losses) + list(self.loss.in_cuda.losses)
                rescale_camera_mean_losses_to_original_valid_pixels(
                    loss_return,
                    batch.data.camera.labels,
                    normalization_context,
                    camera_loss_kinds_by_name=configured_camera_loss_kinds_by_name(configured_losses),
                    distance_bounds_by_name=configured_distance_bounds_by_name(configured_losses),
                )
            if (regularization := self.model.calib.regularization_loss()) is not None:
                name, value, weight = regularization
                loss_return.loss_returns[name] = LossReturn(
                    name=name,
                    lambda_=weight,
                    value=value,
                    reduce_fn=_mean_loss_value,
                )
                loss_return.total_value = loss_return.reduce()

        with ScopedTimer("_training_losses/loss_logging"):
            loss_return.log(logging_fn=self.log)

        timer = ScopedTimer("_training_losses/misc_logging")
        timer.start()

        data_camera = batch.data.camera
        if data_camera is not None:
            if self.global_step % self.config.record_quality_metrics.psnr_interval == 0:
                with torch.no_grad():
                    rgb_ray_mask = torch.logical_and(
                        data_camera.labels.get_mask_flags_all(RayFlags.RGB_LABEL),
                        data_camera.labels.get_mask_flags_none(RayFlags.INVALID),
                    ).reshape(-1)
                    rendered_rgb = unpack_optional(unpack_optional(out.rendered_cam).rgb)
                    psnr = self.train_psnr_metric.compute_item_masked(
                        rendered_rgb,
                        unpack_optional(data_camera.labels.rgb).reshape(rendered_rgb.shape),
                        rgb_ray_mask,
                    )
                    self.log("train/psnr", psnr, prog_bar=True)

        assert batch.rendering is not None, "Rendering should not be None at this stage"
        rendering_camera = batch.rendering.camera
        rendering_lidar = batch.rendering.lidar
        total_number_of_rays = (
            rendering_camera.b * rendering_camera.h * rendering_camera.w if rendering_camera is not None else 0
        ) + (rendering_lidar.b * rendering_lidar.h * rendering_lidar.w if rendering_lidar is not None else 0)
        self.log("train/nrays", float(total_number_of_rays), prog_bar=True)

        # Per-node Gaussian size statistics (max / p99 scale and |mean|, count).
        # Pathologically large Gaussians (e.g. tens of metres) project to huge
        # screen footprints and can drive the rasterizer intersect to hundreds of
        # millions of pairs; tracking these lets us compare runs and catch runaway
        # growth early. Cheap (a few reductions per node), gated by an interval.

        timer.stop()

        return loss_return

    # Number of initial training batches probed for mask coverage before
    # concluding the enabled components have no signal in this dataset.
    # Small enough to fail fast; large enough that a handful of unusual
    # frames (e.g. a pedestrian-free stretch) don't false-positive.
    _COMPONENT_ONLY_EMPTY_PROBE_BATCHES = 16

    def _maybe_apply_component_only_mask(
        self,
        batch: DataAndRenderingBatch,
        out: GaussiansCompositeReturn,
        *,
        record_coverage: bool = False,
    ) -> ComponentOnlyLossNormalizationContext | None:
        """Apply the component-only loss mask in-place (camera path).

        ``record_coverage`` gates the empty-mask probe: it must be True
        only on the TRAINING path. The probe's 16-batch budget and its
        "nothing to train" abort are about the train split — feeding it
        validation/sanity batches (a different frame split) would let a
        pedestrian-free val stretch consume the budget or wrongly abort a
        run whose train split does have supervised pixels.

        Two complementary mechanisms:

        * ``RayFlags.INVALID`` is set on every pixel outside the mask.
          ALL camera losses honor it — rgb/ssim (kernel-side), semantic
          cross-entropy, normal, background/alpha, distance — so the
          entire camera loss family is masked uniformly. Fully-masked
          batches are safe: every normalization guards ``n_valid == 0``.
        * The rendered/gt RGB pair is additionally zero-masked, for the
          consumers that do NOT read flags: LPIPS and the quality
          metrics / validation image outputs. ``loss(0, 0) = 0`` — see
          ``nurec.utils.component_only_mask``.

        Lidar-side masking is deferred: ``LidarFrameLabels`` carries no
        per-ray semantic field yet, so lidar rays train unmasked
        (best-effort; low leakage risk).
        """
        spec = self.model._component_only_mask_spec
        if spec is None:
            return None
        if out.rendered_cam is None or batch.data.camera is None:
            return None
        assert batch.rendering is not None and batch.rendering.camera is not None, (
            "component-only mask requires camera rendering data (per-pixel rays)"
        )

        cam = batch.data.camera
        component_only = getattr(self.model, "component_only", None)
        normalization_context = None
        if (
            component_only is not None
            and component_only.camera_supervision.loss_normalization == CameraLossNormalization.all_valid_pixels
            and cam.labels.flags is not None
        ):
            normalization_context = ComponentOnlyLossNormalizationContext.capture(cam.labels.flags)
        track_regions = self.model.build_component_only_bbox_masks(batch.rendering.camera)
        supervision_mask = build_camera_loss_mask(spec, cam.labels.semantic, track_regions)
        if record_coverage:
            # Only pixels that can actually contribute to a loss count toward the
            # "nothing to train" guard: intersect with the camera's pre-existing
            # usable-pixel flags, since a pixel already flagged INVALID is
            # rejected by every camera loss regardless of the supervision mask.
            usable_supervision = supervision_mask
            if cam.labels.flags is not None:
                usable_supervision = supervision_mask & ((cam.labels.flags.squeeze(-1) & RayFlags.INVALID.value) == 0)
            if (
                component_only is not None
                and component_only.camera_supervision.frame_selection == CameraFrameSelection.supervised_only
                and not bool(usable_supervision.flatten(start_dim=1).any(dim=1).all().item())
            ):
                raise RuntimeError(
                    "component_only supervised_only selected at least one camera frame whose current final supervision "
                    "mask is empty. The startup eligibility index may have become stale after calibration or track "
                    "updates; the optimizer step was aborted before backward to protect parameters from a zero-mask "
                    "frame."
                )
            self._record_component_only_mask_coverage(usable_supervision)

        if cam.labels.flags is not None:
            keep_pixels = supervision_mask.unsqueeze(-1)  # flags are (B, H, W, 1)
            cam.labels.flags = torch.where(keep_pixels, cam.labels.flags, cam.labels.flags | RayFlags.INVALID.value)
            # The valid-pixel counts are lazily computed from flags and
            # cached — drop any stale cache so the loss normalization
            # factors see the masked counts.
            cam.labels._n_valid_rgb = None
            cam.labels._n_valid = None
            cam.labels._n_valid_bg = None

        if out.rendered_cam.rgb is None or cam.labels.rgb is None:
            return normalization_context
        pred = out.rendered_cam.rgb
        gt = cam.labels.rgb
        flat_supervision_mask = supervision_mask.reshape(-1)
        if pred.ndim == 2 and pred.shape[0] == flat_supervision_mask.shape[0]:
            # Rendered rgb is flat (N, 3) with N = B*H*W.
            keep = flat_supervision_mask.unsqueeze(-1)
            out.rendered_cam.rgb = pred * keep.to(pred.dtype)
            cam.labels.rgb = (gt.reshape_as(pred) * keep.to(gt.dtype)).reshape_as(gt)
        else:
            out.rendered_cam.rgb, cam.labels.rgb = apply_zero_mask(pred, gt, supervision_mask)
        return normalization_context

    def _record_component_only_mask_coverage(self, mask: torch.Tensor) -> None:
        """Accumulate ``mask.any()`` over the first probe batches.

        Raises once all probe batches yielded zero supervised pixels —
        the enabled components have nothing to train on in this dataset
        (e.g. deformables-only on a clip without any deformable tracks).
        """
        if not hasattr(self, "_component_only_probe_batches_seen"):
            self._component_only_probe_batches_seen = 0
            self._component_only_probe_batches_with_kept = 0
        if self._component_only_probe_batches_seen >= self._COMPONENT_ONLY_EMPTY_PROBE_BATCHES:
            return

        self._component_only_probe_batches_seen += 1
        if mask_has_any_kept_pixels(mask):
            self._component_only_probe_batches_with_kept += 1

        if self._component_only_probe_batches_seen == self._COMPONENT_ONLY_EMPTY_PROBE_BATCHES:
            spec = self.model._component_only_mask_spec
            assert spec is not None  # only called from the masking path
            # Aggregate coverage across DDP ranks: the counters are rank-local
            # and another rank may have sampled a rare target this one didn't.
            with_kept = self._component_only_probe_batches_with_kept
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                counter = torch.tensor([with_kept], device=self.device)
                torch.distributed.all_reduce(counter, op=torch.distributed.ReduceOp.SUM)
                with_kept = int(counter.item())
            log.info(
                "Component-only mask coverage after %d probe batches (aggregated across ranks): "
                "%d had supervised pixels. supervised_layers=%s residual=%s sky=%s",
                self._COMPONENT_ONLY_EMPTY_PROBE_BATCHES,
                with_kept,
                sorted(spec.supervised_layers),
                spec.include_residual,
                spec.include_sky,
            )
            if with_kept == 0:
                # A sampled, camera-only zero-hit is INCONCLUSIVE — not proof of
                # a bad config: batches sample one camera/timestamp with no
                # component-aware coverage guarantee (a rare target may first
                # appear after this window), and LiDAR supervision is left
                # unmasked. Warn rather than abort a possibly-valid run.
                log.warning(
                    "Component-only mode: none of the first %d training batches (across all ranks) "
                    "contained a supervised camera pixel for the enabled components (%s). This MAY "
                    "mean the dataset has no content for them — but per-batch camera/timestamp "
                    "sampling is random and camera-only (LiDAR supervision is unmasked), so it is "
                    "not conclusive. Verify the clip covers the enabled components if training stalls.",
                    self._COMPONENT_ONLY_EMPTY_PROBE_BATCHES,
                    sorted(spec.supervised_layers),
                )

    def get_output_types(self) -> list[str]:
        # Configuration defines what image sequence types (modalities) will be output during validation and test
        out_types = []
        if self.config.test.save_results:
            all_results = ["pred_rgb", "pred_distance", "pred_normal", "pred_opacity"]
            wanted = getattr(self.config.test, "result_types", None)
            out_types.extend([t for t in all_results if wanted is None or t in wanted])
        if self.config.test.save_inputs:
            out_types.extend(
                [
                    "input_rgb",
                    "input_distance",
                    "input_normal",
                    "input_valid_mask",
                    "input_sky_mask",
                    "input_road_mask",
                    "input_semantic_mask",
                ]
            )
        if self.config.test.save_extra_signals:
            out_types.extend(["pred_semantic", "bg_rgb", "rgb_before_post_processing"])
        return out_types

    def get_max_dist_m(self):
        if self.max_dist_m is not None:
            return self.max_dist_m
        else:
            return self.model.scene_extent

    def get_ncore_semantic_classes_map(self) -> Optional[dict[str, int]]:
        if hasattr(self.datamodule, "val_dataset") and isinstance(self.datamodule.val_dataset, NCORESequentialDataset):
            datasource = self.datamodule.val_dataset.get_datasource()
            # Semantic class maps come from aux data; skip when aux is disabled / absent.
            if isinstance(datasource, NCOREDataSource) and not datasource.aux_data:
                return None
            return datasource.get_semantic_classes_map(camera_semantics=True, lidar_semantics=False)
        return None

    def on_validation_start(self) -> None:
        self.model.drain_fused_gather_generation_plan_refresh()
        super().on_validation_start()

        self.val_suffix = []
        if isinstance(self.datamodule.val_dataset, NCORESequentialDataset):
            if self.datamodule.val_dataset.val_sensor_transl_delta_m is not None:
                self.val_suffix.append(
                    "transl_{}".format(
                        "-".join([str(x) for x in self.datamodule.val_dataset.val_sensor_transl_delta_m])
                    )
                )
            if self.datamodule.val_dataset.val_sensor_rot_delta_deg is not None:
                self.val_suffix.append(
                    "rot_{}".format("-".join([str(x) for x in self.datamodule.val_dataset.val_sensor_rot_delta_deg]))
                )

        self.val_out_types = self.get_output_types()

        self.maybe_override_track_plys()
        self.maybe_remove_background_gaussians()
        self.maybe_setup_visualdebugger()

    def maybe_setup_visualdebugger(self) -> None:
        """
        Setup the visualdebugger for pointcloud renders.
        """
        if self.config.test.lidar.save_renders.enabled:
            self.visualdebugger = get_visualdebugger()
            self.visualdebugger.set_properties(
                program_name="Lidar Points",
                up="z_up",
                front="x_front",
                ground_plane_mode="none",
                length_scale=1.0,
                automatically_compute_scene_extents=False,
                window_size=(1920, 1080),
                window_resizable=False,
                view_projection_mode="perspective",
                json_str=DEFAULT_PS_VIEW_JSON_STR,
            )
            self.visualdebugger.set_camera_extrinsics(
                VisualDebugger.RootLookUp(
                    root=self.config.test.lidar.save_renders.root,
                    look_dir=self.config.test.lidar.save_renders.look_dir,
                    up_dir=self.config.test.lidar.save_renders.up_dir,
                )
            )
            self.visualdebugger.clear()

            os.makedirs(os.path.join(self.val_dir, "pc_renders", "pred"), exist_ok=True)
            os.makedirs(os.path.join(self.val_dir, "pc_renders", "gt"), exist_ok=True)
            self.val_out_types += ["pc_renders"]

        if isinstance(self.model.background, SkyEnvMapBackground):
            self.log_and_inpaint_env_map(self.model.background)

        self.maybe_render_track_orbits()

    def maybe_override_track_plys(self) -> None:
        """
        Overrides the gaussians for specified tracks with gaussians from ply files
        """
        if len(self.config.test.track_ply.overrides) == 0:
            return

        transform = torch.FloatTensor(self.config.test.track_ply.transform).to(self.device)
        for track_id, ply_path in self.config.test.track_ply.overrides.items():
            self.model.track_ply_override(
                track_id,
                Asset(Path(ply_path)),
                transform=transform,
                dims_offset=torch.FloatTensor(self.track_padding_m).to(self.device),
            )

    def on_before_checkpoint_capture(self) -> None:
        super().on_before_checkpoint_capture()
        self.maybe_morton_sort_gaussians()

    def maybe_morton_sort_gaussians(self) -> None:
        """
        Reorders the gaussians into Morton order so that gaussians close in space sit close in memory.

        Runs just before a checkpoint is captured, so the .ckpt and the artifact built from it both carry the
        order.
        """
        morton_sort_config = self.cached_config.checkpoint.morton_sort
        if not morton_sort_config.enabled:
            return

        # The reorder replaces parameter objects and save_training_parameters holds references to them, not copies.
        # Sorting after a snapshot was taken would leave restore silently putting back the wrong rows.
        if self.model.has_saved_training_parameters():
            log.warning("Morton sort skipped, a parameter snapshot is already held and restore would undo the order")
            return

        self.model.morton_sort_gaussians(
            sort_track_layers=morton_sort_config.sort_track_layers,
            voxel_size=morton_sort_config.voxel_size,
        )

    def maybe_remove_background_gaussians(self) -> None:
        """
        Removes background gaussians that fall inside the specified cuboid tracks. This ensure that
        objects do not overlap with the background when modifying the trajectories of these objects.
        """
        if len(self.config.test.background_removal.track_ids) == 0:
            return

        track_shift_m = torch.FloatTensor(self.config.test.background_removal.track_shift_m).to(self.device)
        track_padding_m = torch.FloatTensor(self.config.test.background_removal.track_padding_m).to(self.device)

        self.model.remove_background_gaussians(
            self.config.test.background_removal.track_ids, track_shift_m, track_padding_m
        )

    def maybe_render_track_orbits(self) -> None:
        """
        Renders around specified tracks in an orbit trajectory
        """

        track_orbit_ids = set(self.config.test.track_orbit.track_ids)
        if len(track_orbit_ids) == 0:
            return

        render_width = self.config.test.track_orbit.render_width
        render_height = self.config.test.track_orbit.render_height
        assert render_width % 2 == 0, (
            "width should be multiple of 2"
        )  # otherwise videos might not render properly in ffmpeg
        assert render_height % 2 == 0, (
            "weight should be multiple of 2"
        )  # otherwise videos might not render properly in ffmpeg
        distance = self.config.test.track_orbit.distance
        n_frames = self.config.test.track_orbit.n_frames

        tracks = self.model.cuboid_tracks
        for track_idx in range(tracks.n_tracks):
            if tracks.tracks_id[track_idx] in track_orbit_ids:
                cur_track = CuboidTracks.Ops.subset_from_tracks_id(tracks, [tracks.tracks_id[track_idx]])

                for view_idx, view in enumerate(
                    self.get_track_orbit_views(
                        cur_track,
                        distance,
                        render_width,
                        render_height,
                        n_frames,
                    )
                ):
                    if len(self.val_out_types):
                        self.save_images(
                            self.val_dir,
                            None,
                            unpack_optional(self.forward(view.to(self.device)).rendered_cam),
                            view_idx,
                            h=self.config.test.track_orbit.render_height,
                            out_types=self.val_out_types,
                            suffix_list=self.val_suffix + [tracks.tracks_id[track_idx]],
                        )

                if self.config.test.save_videos and len(self.val_out_types):
                    self.save_videos(
                        self.val_dir,
                        self.val_out_types,
                        suffix_list=self.val_suffix + [tracks.tracks_id[track_idx]],
                    )

    def validation_step(self, batch: DataAndRenderingBatch, batch_local_idx: int) -> None:
        self._raise_if_mcmc_publication_failed()
        track_transform_ids: Optional[list[str]] = None
        track_transforms: Optional[torch.Tensor] = None

        batch_idx = unpack_optional(batch.data.idx)
        if len(self.config.test.track_rotation.track_ids) > 0:
            track_transform_ids = self.config.test.track_rotation.track_ids
            track_transforms = (
                torch.eye(4, device=self.device)
                .unsqueeze(0)
                .repeat(len(self.config.test.track_rotation.track_ids), 1, 1)
            )
            track_transforms[:, :3, :3] = torch.FloatTensor(
                Rotation.from_euler(
                    "z",
                    self.config.test.track_rotation.angle_per_frame * batch_idx
                    + np.array(self.config.test.track_rotation.angle_offsets),
                    degrees=True,
                ).as_matrix()
            ).to(self.device)
            track_transforms = se3_matrix_to_tquat(track_transforms)
        else:
            track_transform_ids = None
            track_transforms = None

        if (camera := batch.data.camera) is not None:
            assert camera.b == 1, f"{self.__class__.__name__}: only supports batch size one in validation mode"

        results: GaussiansCompositeReturn

        if batch.data.camera is not None:
            assert batch.data.lidar is None, "Validation batch should not have both rays_cam and rays_lidar"
        else:
            assert batch.data.camera is None, "Validation batch should not have both rays_cam and rays_lidar"

        # Render all rays using forward and maybe time call
        def forward_call() -> GaussiansCompositeReturn:
            return self(
                batch,
                track_transform_ids=track_transform_ids,
                track_transforms=track_transforms,
            )

        results = forward_call()

        # Apply diffusion harmonizer
        if self.harmonizer_config.inference.enabled and results.rendered_cam is not None:
            if unpack_optional(results.rendered_cam).rgb is not None:
                camera = unpack_optional(batch.data.camera)
                w, h = unpack_optional(camera.w), unpack_optional(camera.h)
                unpack_optional(results.rendered_cam).rgb = self.harmonizer_model(
                    unpack_optional(results.rendered_cam).rgb,
                    torch.Size([h, w]),
                    self.harmonizer_config.inference.use_color_transfer,
                )
            else:
                raise ValueError("Cannot apply Harmonizer to an empty render.")

        # Update batch depending on model settings (apply alpha to rgb etc.)
        self.apply_alpha(batch, results)
        # Component-only: mask validation pred/gt the same way training
        # is masked, so metrics reflect only the supervised region. The
        # empty-mask probe is NOT recorded here — it is a train-split
        # guard (see _maybe_apply_component_only_mask).
        self._maybe_apply_component_only_mask(batch, results, record_coverage=False)
        if results.rendered_cam is not None:
            self.validation_camera_step(batch, results.rendered_cam)
        if results.rendered_lidar is not None:
            self.validation_lidar_step(batch, results.rendered_lidar)
        self.validation_selected_nodes_step(batch)

    def _any_rank_has_camera(self, batch: DataAndRenderingBatch) -> bool:
        local_rank_has_camera = batch.data.camera is not None

        if not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1:
            return local_rank_has_camera

        any_rank_has_camera = torch.tensor(local_rank_has_camera, dtype=torch.int32, device=self.device)
        torch.distributed.all_reduce(any_rank_has_camera, op=torch.distributed.ReduceOp.MAX)
        return bool(any_rank_has_camera.item())

    def validation_selected_nodes_step(self, batch: DataAndRenderingBatch) -> None:
        """Render selected nodes while keeping distributed Gaussian gathers rank-symmetric.

        A distributed validation sampler can pad a mixed camera/LiDAR dataset with a
        sample of the other modality. Selected-node forwards must nevertheless run on
        every rank because Gaussian collection contains NCCL collectives.
        """
        if len(self.val_out_types) == 0 or len(self.config.test.val_render_selected_nodes) == 0:
            return

        if not self._any_rank_has_camera(batch):
            return

        for suffix, gaussian_nodes in self.config.test.val_render_selected_nodes.items():
            node_results: GaussiansCompositeReturn = self(
                batch,
                gaussian_nodes=gaussian_nodes,
                render_background=False,
            )

            # A non-camera rank still executes the forward pass so every rank enters
            # the same distributed Gaussian gathers, but it has no image to save.
            if (camera := batch.data.camera) is None:
                continue

            assert node_results.rendered_cam is not None
            if self.harmonizer_config.inference.enabled:
                if node_results.rendered_cam.rgb is not None:
                    node_results.rendered_cam.rgb = self.harmonizer_model(
                        node_results.rendered_cam.rgb,
                        torch.Size([unpack_optional(camera.h), unpack_optional(camera.w)]),
                        self.harmonizer_config.inference.use_color_transfer,
                    )
                else:
                    raise ValueError("Cannot apply Harmonizer to an empty render.")

            self.save_images(
                self.val_dir,
                batch,
                node_results.rendered_cam,
                unpack_optional(batch.data.idx),
                h=unpack_optional(camera.h),
                out_types=self.val_out_types,
                suffix_list=self.val_suffix + [suffix],
            )

    def validation_camera_step(self, batch: DataAndRenderingBatch, rendered: GaussiansRenderReturn) -> None:
        if (camera := batch.data.camera) is None:
            log.warning(f"No camera data for {unpack_optional(batch.data.idx)}, skipping validation metrics")
            return

        w, h = unpack_optional(camera.w), unpack_optional(camera.h)
        # no ground truth when the photo was not loaded (dataset.val_load_rgb=false, render-only): render and save only
        rgb_gt = camera.labels.rgb.squeeze(0) if camera.labels.rgb is not None else None  # (h, w, 3)
        batch_idx = unpack_optional(batch.data.idx)

        # Used for metrics
        timestamps_startend_us = None
        if batch.rendering is not None and batch.rendering.camera is not None:
            timestamps_startend_us = batch.rendering.camera.timestamps_startend_us

        # only validate valid rays (rays that are *not* marked as invalid)
        valid_rays_mask = camera.labels.get_mask_flags_none(RayFlags.INVALID).squeeze()  # h, w
        if rgb_gt is None:
            pass  # render-only: no metrics
        elif valid_rays_mask.any():
            rendered_rgb = rearrange(rendered.rgb, "(h w) c -> h w c", h=h)
            valid_rgb = rendered_rgb[valid_rays_mask]
            valid_rgb_gt = rgb_gt[valid_rays_mask]

            self.val_metric_manager.compute("psnr", valid_rgb, valid_rgb_gt)
            psnr = self.val_metric_manager.get_last("psnr")
            if psnr is not None:
                self.collect_metric(
                    "test/psnr",
                    psnr["psnr"].cpu(),
                    is_lidar=False,
                    frame_meta=camera.meta[0],
                    timestamps_startend_us=timestamps_startend_us,
                    sequence_id=batch.data.sequence_id,
                )
                self.val_metric_manager.reset("psnr")

            if (
                self.config.test.metrics.cpsnr.enabled
                and (semantic_classes_map := self.get_ncore_semantic_classes_map()) is not None
            ):
                semantic = camera.labels.semantic
                if semantic is not None:
                    self.val_metric_manager.compute(
                        "cpsnr",
                        rendered_rgb,
                        rgb_gt,
                        valid_mask=valid_rays_mask.squeeze(),
                        segmentation_frame=semantic.squeeze(),
                        color_dict=semantic_classes_map,
                        include_categories=self.config.test.metrics.cpsnr.classes,
                    )
                    cpsnr = self.val_metric_manager.get_last("cpsnr")
                    if cpsnr is not None:
                        for class_name, value in cpsnr.values.items():
                            pixel_count = cpsnr.metadata["pixel_counts"][class_name]
                            if pixel_count > 0:
                                self.collect_metric(
                                    f"test/cpsnr/{class_name}",
                                    value,
                                    is_lidar=False,
                                    frame_meta=camera.meta[0],
                                    timestamps_startend_us=timestamps_startend_us,
                                    sequence_id=batch.data.sequence_id,
                                )
                        self.val_metric_manager.reset("cpsnr")

            if self.config.test.metrics.ssim.enabled or self.config.test.metrics.lpips.enabled:
                reshaped_rgb = rearrange(
                    unpack_optional(rendered.rgb).clone(), "(h w) c -> h w c", h=h, w=w
                )  # (h, w, 3)
                reshaped_rgb_gt = rgb_gt.clone()  # (h, w, 3)
                reshaped_rgb_gt = rearrange(reshaped_rgb_gt, "h w c -> 1 c h w", h=h, w=w)
                reshaped_rgb = rearrange(reshaped_rgb, "h w c -> 1 c h w", h=h, w=w)

                datasource = self.datamodule.get_datasource()
                camera_mask_border: list[int] | None = None
                if isinstance(datasource, NCOREDataSource) and datasource.camera_mask_border is not None:
                    # unique_sensor_idx indexes the sorted logical sensor list for single-sequence NCORE data.
                    camera_id = datasource.get_camera_sensor_ids(unique_sensors=False)[camera.meta[0].unique_sensor_idx]
                    camera_mask_border = datasource.camera_mask_border.get(camera_id, None)

                if camera_mask_border is not None:
                    # When a camera_mask_border is defined, the valid part of the rgb and rgb_gt images still form a rectangle, so we can crop the border from the images
                    reshaped_rgb = crop_mask_border(reshaped_rgb, camera_mask_border)
                    reshaped_rgb_gt = crop_mask_border(reshaped_rgb_gt, camera_mask_border)
                else:
                    reshaped_rgb[:, :, ~valid_rays_mask] = reshaped_rgb_gt[:, :, ~valid_rays_mask]

                if self.config.test.metrics.ssim.enabled:
                    ssim = self.get_ssim_criterion()(reshaped_rgb, reshaped_rgb_gt)
                    self.validation_step_outputs["ssim"].append(ssim)
                    self.collect_metric(
                        "test/ssim",
                        ssim,
                        is_lidar=False,
                        frame_meta=camera.meta[0],
                        timestamps_startend_us=timestamps_startend_us,
                        sequence_id=batch.data.sequence_id,
                    )
                    self.get_ssim_criterion().reset()

                if self.config.test.metrics.lpips.enabled:
                    lpips = self.get_lpips_criterion()(reshaped_rgb, reshaped_rgb_gt)
                    self.validation_step_outputs["lpips"].append(lpips)
                    self.collect_metric(
                        "test/lpips",
                        lpips,
                        is_lidar=False,
                        frame_meta=camera.meta[0],
                        timestamps_startend_us=timestamps_startend_us,
                        sequence_id=batch.data.sequence_id,
                    )
                    self.get_lpips_criterion().reset()
        else:
            log.warning(f"No valid rays for {batch_idx}, skipping validation metrics")

        if len(self.val_out_types):
            # save image to disk
            self.save_images(
                self.val_dir,
                batch,
                rendered,
                batch_idx,
                h=h,
                out_types=self.val_out_types,
                suffix_list=self.val_suffix,
            )

    def validation_lidar_step(
        self,
        batch: DataAndRenderingBatch,
        rendered: GaussiansRenderReturn,
    ) -> None:
        if (data_lidar := batch.data.lidar) is None:
            log.warning(f"No lidar data for {unpack_optional(batch.data.idx)}, skipping validation metrics")
            return
        assert batch.rendering is not None
        rendering_lidar = batch.rendering.lidar
        assert rendering_lidar is not None

        # flags are shaped (b, h, w, 1)
        is_valid = data_lidar.labels.get_mask_flags_none(RayFlags.INVALID).squeeze(-1)
        has_valid_gt = bool(is_valid.any())

        # When all rays are marked INVALID (e.g. novel-view rendering with a
        # modified lidar pose), GT labels are not aligned with the current ray
        # poses. We still want pred-side visualization, so use all rays
        # downstream; metric computation against stale GT is skipped below.
        if not has_valid_gt:
            log.info(
                f"No valid GT labels for idx={unpack_optional(batch.data.idx)}; "
                "saving pred-side visualization and skipping metrics."
            )
            is_valid = torch.ones_like(is_valid)

        # rendering_lidar has shape (b, h, w, 6)
        rays_lidar = rendering_lidar.rays[is_valid]
        rendered_valid = rendered[is_valid.reshape(-1)]
        height, width = is_valid.shape[1], is_valid.shape[2]

        # In model-free unstructured lidar mode the "grid" is the (N, 1) return list, so
        # there is no spatially-coherent 2D range view: the flat depth/point-cloud
        # metrics work unchanged, the range-view IMAGE saves and the range-view
        # `floating_point_filter` (neighbor_source="rangeview") are undefined (NuRecConfig
        # forbids that combination), but the sensor-agnostic angle filter
        # (neighbor_source="angle") is defined here and works for unstructured lidar.
        lidar_elements = generate_grid_2d_indices((width, height), order="yx", device=is_valid.device)
        lidar_valid_elements = lidar_elements[is_valid.reshape(-1)].to(dtype=torch.int32)

        # filter by predicted raydrop mask
        if rendered_valid.extra_ray_signals is not None and rendered_valid.extra_ray_signals.raydrop is not None:
            save_raydrop = True
            did_return_pred = (
                rendered_valid.extra_ray_signals.raydrop.reshape(-1) < self.config.test.lidar.raydrop_threshold
            )
        else:
            save_raydrop = False
            did_return_pred = torch.ones_like(
                unpack_optional(data_lidar.labels.flags)[is_valid], dtype=torch.bool
            ).reshape(-1)

        # filter by opacity mask (mirrors the gRPC SensorSim rendering path in
        # nurec/render/render.py). Applied before metric computation and visualization
        # export so both reflect the opacity-filtered prediction. 0 disables it.
        opacity_threshold = self.config.test.lidar.opacity_threshold
        if opacity_threshold > 0.0:
            opacity_keep = rendered_valid.opacity.reshape(-1) > opacity_threshold
            did_return_pred = torch.logical_and(did_return_pred, opacity_keep)

        # Sensor pose + sensor-frame ray directions. Computed here (before the metric-affecting
        # distance filter) so the angle neighbour filter can use ray angles; the same tensors are
        # reused below for the sensor-frame point clouds.
        datasource = self.datamodule.get_datasource()
        assert isinstance(datasource, NCOREDataSource), "Only NCOREDataSource supported."
        if any(meta.unique_sensor_idx != data_lidar.meta[0].unique_sensor_idx for meta in data_lidar.meta[1:]):
            raise ValueError("Multiple lidar sensors per validation batch are not supported")
        # unique_sensor_idx indexes the sorted logical sensor list for single-sequence NCORE data.
        lidar_id = datasource.get_lidar_sensor_ids(unique_sensors=False)[data_lidar.meta[0].unique_sensor_idx]
        lidar_sensor = datasource.lidar_sensors[lidar_id]
        # Recover the raw NCore sensor frame index from the global unique_frame_idx.
        # Reverses the computation in NCOREDataset.get_lidar_data_batch().
        lidar_frame_index = (
            data_lidar.meta[0].unique_frame_idx
            - datasource.lidar_linear_start_frame_indices[lidar_id]
            + datasource.lidar_frame_ranges[lidar_id].start
        )
        T_sensor_nurec = to_torch(
            datasource.world_to_nurec.transform_poses(
                lidar_sensor.get_frames_T_sensor_target("world", lidar_frame_index)
            ),
            device=self.device,
        )
        T_nurec_sensor = se3_matrix_inverse(T_sensor_nurec)
        xyz_start_sensor = (T_nurec_sensor[:3, :3] @ rays_lidar[:, 0:3].T + T_nurec_sensor[:3, 3:4]).T
        ray_direction_sensor = (T_nurec_sensor[:3, :3] @ rays_lidar[:, 3:6].T).T

        # Distance-discontinuity ("flying pixel") filter. `neighbor_source` selects how ray neighbours
        # are found: "rangeview" (structured 2D grid) or "angle" (sensor-agnostic: elevation-binned
        # rows + azimuth-sorted neighbours). The angle path works for model-free unstructured lidar.
        # Both share the relative-gradient `filter_threshold` and drop flagged rays from
        # `did_return_pred` (flows into metrics + clouds).
        if self.config.test.lidar.floating_point_filter.enabled:
            fpf = self.config.test.lidar.floating_point_filter
            threshold = fpf.threshold
            filter_mask_valid = None
            if fpf.neighbor_source == "angle":
                try:
                    filter_mask_valid = angle_gradient_filter(
                        rendered_valid,
                        ray_direction_sensor,
                        did_return_pred,
                        threshold,
                        fpf.elevation_resolution_deg,
                        fpf.azimuth_max_gap_deg,
                    )
                except ValueError as e:
                    # The angle filter fails loud when it cannot auto-assign scanline rows (e.g. an
                    # interleaved-elevation sensor with elevation_resolution_deg unset). During
                    # validation, skip the filter for this run rather than crashing the whole val
                    # loop; the fix is to set floating_point_filter.elevation_resolution_deg
                    # explicitly. Warn once so it does not flood the val logs.
                    if not getattr(self, "_floating_point_filter_skip_warned", False):
                        log.warning(
                            "Skipping floating_point_filter (neighbor_source='angle'): %s Set "
                            "system.test.lidar.floating_point_filter.elevation_resolution_deg "
                            "explicitly to enable it. (shown once)",
                            e,
                        )
                        self._floating_point_filter_skip_warned = True
            else:
                filter_mask = distance_based_filter(
                    rendered_valid, lidar_valid_elements, did_return_pred, threshold, height, width
                )
                filter_mask_valid = filter_mask[is_valid.reshape(-1)]
            # remove points flagged as flying pixels
            if filter_mask_valid is not None:
                did_return_pred = torch.logical_and(did_return_pred, ~filter_mask_valid)

        # Used for metrics
        timestamps_startend_us = None
        if batch.rendering is not None and batch.rendering.lidar is not None:
            timestamps_startend_us = batch.rendering.lidar.timestamps_startend_us

        if has_valid_gt:
            self.val_metric_manager.compute(
                "lidar_common", data_lidar, rendering_lidar, rendered, did_return_pred, self.config
            )
            lidar_common = self.val_metric_manager.get_last("lidar_common")
        else:
            # No usable GT (e.g. novel-view rendering). Skip metric computation;
            # downstream collect_metric calls are already guarded by None checks.
            lidar_common = None

        dist_gt = unpack_optional(data_lidar.labels.distance)[is_valid].reshape(-1)
        did_return_gt = data_lidar.labels.get_mask_flags_none(RayFlags.DROPPED)[is_valid].reshape(-1)
        if self.config.test.lidar.ROI.min_m is not None:
            did_return_gt = torch.logical_and(did_return_gt, dist_gt >= self.config.test.lidar.ROI.min_m)
        if self.config.test.lidar.ROI.max_m is not None:
            did_return_gt = torch.logical_and(did_return_gt, dist_gt <= self.config.test.lidar.ROI.max_m)

        if lidar_common is not None and "raydrop_accuracy" in lidar_common:
            self.collect_metric(
                "test/raydrop_accuracy",
                lidar_common["raydrop_accuracy"].cpu(),
                is_lidar=True,
                frame_meta=data_lidar.meta[0],
                timestamps_startend_us=timestamps_startend_us,
                sequence_id=batch.data.sequence_id,
            )
        if lidar_common is not None and "raydrop_precision" in lidar_common:
            self.collect_metric(
                "test/raydrop_precision",
                lidar_common["raydrop_precision"].cpu(),
                is_lidar=True,
                frame_meta=data_lidar.meta[0],
                timestamps_startend_us=timestamps_startend_us,
                sequence_id=batch.data.sequence_id,
            )
        if lidar_common is not None and "raydrop_recall" in lidar_common:
            self.collect_metric(
                "test/raydrop_recall",
                lidar_common["raydrop_recall"].cpu(),
                is_lidar=True,
                frame_meta=data_lidar.meta[0],
                timestamps_startend_us=timestamps_startend_us,
                sequence_id=batch.data.sequence_id,
            )
        if lidar_common is not None and "raydrop_IoU" in lidar_common:
            self.collect_metric(
                "test/raydrop_IoU",
                lidar_common["raydrop_IoU"].cpu(),
                is_lidar=True,
                frame_meta=data_lidar.meta[0],
                timestamps_startend_us=timestamps_startend_us,
                sequence_id=batch.data.sequence_id,
            )

        dist_pred = unpack_optional(rendered_valid.distance)

        if lidar_common is not None and "depth_median_l2" in lidar_common:
            self.collect_metric(
                "test/depth_median_l2",
                lidar_common["depth_median_l2"].cpu(),
                is_lidar=True,
                frame_meta=data_lidar.meta[0],
                timestamps_startend_us=timestamps_startend_us,
                sequence_id=batch.data.sequence_id,
            )
        if lidar_common is not None and "depth_mean_rel_l2" in lidar_common:
            self.collect_metric(
                "test/depth_mean_rel_l2",
                lidar_common["depth_mean_rel_l2"].cpu(),
                is_lidar=True,
                frame_meta=data_lidar.meta[0],
                timestamps_startend_us=timestamps_startend_us,
                sequence_id=batch.data.sequence_id,
            )
        if lidar_common is not None and "depth_rmse" in lidar_common:
            self.collect_metric(
                "test/depth_rmse",
                lidar_common["depth_rmse"].cpu(),
                is_lidar=True,
                frame_meta=data_lidar.meta[0],
                timestamps_startend_us=timestamps_startend_us,
                sequence_id=batch.data.sequence_id,
            )
        if lidar_common is not None and "depth_mae" in lidar_common:
            self.collect_metric(
                "test/depth_mae",
                lidar_common["depth_mae"].cpu(),
                is_lidar=True,
                frame_meta=data_lidar.meta[0],
                timestamps_startend_us=timestamps_startend_us,
                sequence_id=batch.data.sequence_id,
            )
        if lidar_common is not None and "depth_medae" in lidar_common:
            self.collect_metric(
                "test/depth_medae",
                lidar_common["depth_medae"].cpu(),
                is_lidar=True,
                frame_meta=data_lidar.meta[0],
                timestamps_startend_us=timestamps_startend_us,
                sequence_id=batch.data.sequence_id,
            )
        if lidar_common is not None and "depth_recall50" in lidar_common:
            self.collect_metric(
                "test/depth_recall50",
                lidar_common["depth_recall50"].cpu(),
                is_lidar=True,
                frame_meta=data_lidar.meta[0],
                timestamps_startend_us=timestamps_startend_us,
                sequence_id=batch.data.sequence_id,
            )

        xyz_end_pred = torch.addcmul(rays_lidar[:, 0:3], rays_lidar[:, 3:6], dist_pred.unsqueeze(1)).clone()
        xyz_end_gt = torch.addcmul(rays_lidar[:, 0:3], rays_lidar[:, 3:6], dist_gt.unsqueeze(1)).clone()

        if lidar_common is not None and "chamfer_distance" in lidar_common:
            self.collect_metric(
                "test/chamfer_distance",
                lidar_common["chamfer_distance"].cpu(),
                is_lidar=True,
                frame_meta=data_lidar.meta[0],
                timestamps_startend_us=timestamps_startend_us,
                sequence_id=batch.data.sequence_id,
            )

        intensity_gt = (
            data_lidar.labels.intensity[is_valid].reshape(-1) if data_lidar.labels.intensity is not None else None
        )
        if rendered_valid.extra_ray_signals is not None and rendered_valid.extra_ray_signals.intensity is not None:
            intensity_pred = rendered_valid.extra_ray_signals.intensity.squeeze()

            if lidar_common is not None and "intensity_mae" in lidar_common:
                self.collect_metric(
                    "test/intensity_mae",
                    lidar_common["intensity_mae"].cpu(),
                    is_lidar=True,
                    frame_meta=data_lidar.meta[0],
                    timestamps_startend_us=timestamps_startend_us,
                    sequence_id=batch.data.sequence_id,
                )

            if lidar_common is not None and "intensity_rmse" in lidar_common:
                self.collect_metric(
                    "test/intensity_rmse",
                    lidar_common["intensity_rmse"].cpu(),
                    is_lidar=True,
                    frame_meta=data_lidar.meta[0],
                    timestamps_startend_us=timestamps_startend_us,
                    sequence_id=batch.data.sequence_id,
                )
        else:
            intensity_pred = None

        xyz_end_pred_sensor = (T_nurec_sensor[:3, :3] @ xyz_end_pred.T + T_nurec_sensor[:3, 3:4]).T
        xyz_end_gt_sensor = (T_nurec_sensor[:3, :3] @ xyz_end_gt.T + T_nurec_sensor[:3, 3:4]).T

        flags = data_lidar.labels.flags[is_valid].reshape(-1) if data_lidar.labels.flags is not None else None
        pc_gt = PointCloud(
            xyz_start=xyz_start_sensor,
            xyz_end=xyz_end_gt_sensor,
            intensity=intensity_gt,
            flags=flags.clone() if flags is not None else None,
        )
        if pc_gt.flags is not None:
            pc_gt.flags[~did_return_gt] |= RayFlags.DROPPED
        # Clear GT's DROPPED bit on pred so that pc_pred.flags.DROPPED reflects
        # the model prediction (plus ROI / distance filters) rather than the
        # union of GT and pred drop states.
        pred_flags = flags.clone() & ~RayFlags.DROPPED.value if flags is not None else None
        pc_pred = PointCloud(
            xyz_start=xyz_start_sensor,
            xyz_end=xyz_end_pred_sensor,
            intensity=intensity_pred,
            flags=pred_flags,
        )
        if pc_pred.flags is not None:
            pc_pred.flags[~did_return_pred] |= RayFlags.DROPPED
        batch_idx = unpack_optional(batch.data.idx)

        self.save_pointcloud(batch_idx, pc_pred, pc_gt)

        # Range-view IMAGE saves need a real 2D (n_rows, n_columns) range view + the
        # sensor model parameters; both are absent in model-free unstructured mode
        # (sensor_model_parameters is None, grid is (N, 1)). Detect that per-frame
        # from the data (model-free => no sensor model) rather than a global config
        # flag, since lidar_ray_mode can now be per-sensor.
        lidar_unstructured_frame = rendering_lidar.sensor_model_parameters[0] is None
        if not lidar_unstructured_frame:
            lidar_model_parameters = cast(ConcreteLidarModelParametersUnion, rendering_lidar.sensor_model_parameters[0])
            self.save_rangeview(batch_idx, pc_pred, pc_gt, lidar_valid_elements, lidar_model_parameters)
            if save_raydrop:
                self.save_raydrop(
                    batch_idx, did_return_pred, did_return_gt, lidar_valid_elements, lidar_model_parameters
                )
        if self.config.test.lidar.save_renders.enabled:
            self.save_pointcloud_renders(batch, pc_pred, pc_gt)

    def save_pointcloud_renders(self, batch: DataAndRenderingBatch, pc_pred: PointCloud, pc_gt: PointCloud) -> None:
        """
        Save the pointcloud renders to the visualdebugger and save the images to disk.
        """
        idx = unpack_optional(batch.data.idx)

        xyz_end_pred_sensor = pc_pred.xyz_end
        xyz_end_gt_sensor = pc_gt.xyz_end

        self.visualdebugger.clear()
        self.visualdebugger.add_point_cloud(
            "Pred",
            to_numpy(xyz_end_pred_sensor),
            radius=self.config.test.lidar.save_renders.radius,
            color=(0, 0, 1),
        )
        image_name = os.path.join(self.val_dir, "pc_renders", "pred", f"{idx:06d}.png")
        pred_im = self.visualdebugger.screenshot_to_buffer()
        imageio.v2.imsave(image_name, pred_im)

        self.visualdebugger.clear()
        self.visualdebugger.add_point_cloud(
            "GT", to_numpy(xyz_end_gt_sensor), radius=self.config.test.lidar.save_renders.radius, color=(1, 0, 0)
        )
        image_name = os.path.join(self.val_dir, "pc_renders", "gt", f"{idx:06d}.png")
        gt_im = self.visualdebugger.screenshot_to_buffer()
        imageio.v2.imsave(image_name, gt_im)

        if self.config.save_logger:
            for image_name, image_im in [("pred", pred_im), ("gt", gt_im)]:
                image_group = f"pc_renders-{image_name}-rank_{self.global_rank}"
                self.log_image(image_im, image_group, idx)

    def save_pointcloud(self, batch_idx: int, pc_pred: PointCloud, pc_gt: PointCloud) -> None:
        image_name = f"{batch_idx:06d}"
        os.makedirs(out_dir := os.path.join(self.val_dir, "pred_pc"), exist_ok=True)
        image_name = os.path.join(out_dir, image_name)

        pcd_pred = TriangleMesh()
        valid_mask_pred = torch.bitwise_and(unpack_optional(pc_pred.flags), RayFlags.DROPPED).eq(0)
        pcd_pred.vertex_data.positions = pc_pred.xyz_end[valid_mask_pred].cpu().numpy()
        if pc_pred.intensity is not None:
            pcd_pred.vertex_data.colors = scalar2rgb(pc_pred.intensity[valid_mask_pred].cpu().numpy().squeeze())
        pcd_pred.save(image_name + "output.ply")

        pcd_gt = TriangleMesh()
        valid_mask_gt = torch.bitwise_and(unpack_optional(pc_gt.flags), RayFlags.DROPPED).eq(0)
        pcd_gt.vertex_data.positions = pc_gt.xyz_end[valid_mask_gt].cpu().numpy()
        if pc_gt.intensity is not None:
            pcd_gt.vertex_data.colors = scalar2rgb(pc_gt.intensity[valid_mask_gt].cpu().numpy())
        pcd_gt.save(image_name + "output_gt.ply")

    def save_rangeview(
        self,
        batch_idx: int,
        pc_pred: PointCloud,
        pc_gt: PointCloud,
        model_element: torch.Tensor,
        lidar_parameters: ConcreteLidarModelParametersUnion,
    ) -> None:
        n_vertical_bins = lidar_parameters.n_rows
        n_horizontal_bins = lidar_parameters.n_columns

        rangeview_range_gt = torch.full((n_vertical_bins, n_horizontal_bins), torch.inf, device=self.device)
        rangeview_range_pred = torch.full((n_vertical_bins, n_horizontal_bins), torch.inf, device=self.device)

        pc_lidar_gt = pc_gt.xyz_end - pc_gt.xyz_start
        r_gt = pc_lidar_gt.norm(dim=1)

        pc_lidar_pred = pc_pred.xyz_end - pc_pred.xyz_start
        r_pred = pc_lidar_pred.norm(dim=1)
        valid_mask_pred = torch.bitwise_and(unpack_optional(pc_pred.flags), RayFlags.DROPPED).eq(0)
        r_pred[~valid_mask_pred] = 0

        valid_mask_gt = torch.bitwise_and(unpack_optional(pc_gt.flags), RayFlags.DROPPED).eq(0)
        rangeview_range_gt[model_element[..., 0][valid_mask_gt], model_element[..., 1][valid_mask_gt]] = r_gt[
            valid_mask_gt
        ]

        rangeview_range_pred[model_element[..., 0][valid_mask_pred], model_element[..., 1][valid_mask_pred]] = r_pred[
            valid_mask_pred
        ]

        rangeview_range_pred[rangeview_range_gt == torch.inf] = 0
        rangeview_range_gt[rangeview_range_gt == torch.inf] = 0

        image_name = f"{unpack_optional(batch_idx):06d}"
        os.makedirs(out_dir_rangeview := os.path.join(self.val_dir, "pred_pc"), exist_ok=True)
        image_name = os.path.join(out_dir_rangeview, image_name)

        rangeview_range_gt_numpy = rangeview_range_gt.cpu().numpy()
        im_gt = scalar2img(rangeview_range_gt_numpy, vmin=0.0, vmax=self.get_max_dist_m())
        im_gt[rangeview_range_gt_numpy == 0] = 0
        imageio.v2.imsave(image_name + "_gt.png", im_gt)

        rangeview_range_pred_numpy = rangeview_range_pred.cpu().numpy()
        im_pred = scalar2img(rangeview_range_pred_numpy, vmin=0.0, vmax=self.get_max_dist_m())
        im_pred[rangeview_range_pred_numpy == 0] = 0
        imageio.v2.imsave(image_name + ".png", im_pred)

    def save_raydrop(
        self,
        batch_idx: int,
        raydrop_pred: torch.Tensor,
        raydrop_gt: torch.Tensor,
        model_element: torch.Tensor,
        lidar_parameters: ConcreteLidarModelParametersUnion,
    ) -> None:
        n_vertical_bins = lidar_parameters.n_rows
        n_horizontal_bins = lidar_parameters.n_columns

        rangeview_dropped_gt = torch.ones((n_vertical_bins, n_horizontal_bins), device=self.device, dtype=torch.bool)
        rangeview_dropped_gt[model_element[..., 0][~raydrop_gt], model_element[..., 1][~raydrop_gt]] = False
        rangeview_dropped_pred = torch.ones((n_vertical_bins, n_horizontal_bins), device=self.device, dtype=torch.bool)
        rangeview_dropped_pred[model_element[..., 0][~raydrop_pred], model_element[..., 1][~raydrop_pred]] = False

        os.makedirs(out_dir := os.path.join(self.val_dir, "ray_drop"), exist_ok=True)
        plt.imsave(
            f"{out_dir}/{batch_idx}_gt.png",
            rangeview_dropped_gt.cpu().numpy(),
            cmap="gray",
        )
        plt.imsave(
            f"{out_dir}/{batch_idx}.png",
            rangeview_dropped_pred.cpu().numpy(),
            cmap="gray",
        )

    def _write_result_png(self, path: str, im: np.ndarray) -> None:
        """Write one result PNG, in a background thread if image_writer_threads > 0 (decoupled from rendering)."""
        n_threads = int(getattr(self.config.test, "image_writer_threads", 0) or 0)
        if n_threads <= 0:
            imageio.v2.imsave(path, im)
            return
        if getattr(self, "_png_pool", None) is None:
            from concurrent.futures import ThreadPoolExecutor

            self._png_pool = ThreadPoolExecutor(max_workers=n_threads)
            self._png_futures = []
        self._png_futures.append(self._png_pool.submit(imageio.v2.imsave, path, im.copy()))
        while len(self._png_futures) > 4 * n_threads:  # bound the queue (memory); surfaces write errors early
            self._png_futures.pop(0).result()

    def _finish_result_pngs(self) -> None:
        for f in getattr(self, "_png_futures", None) or []:
            f.result()
        self._png_futures = []

    def on_validation_epoch_end(self) -> None:
        self._finish_result_pngs()  # every result PNG on disk before videos / the caller read them
        if self.config.test.save_videos and len(self.val_out_types):
            self.save_videos(self.val_dir, self.val_out_types, suffix_list=self.val_suffix)
            for suffix in self.config.test.val_render_selected_nodes:
                self.save_videos(self.val_dir, self.val_out_types, suffix_list=self.val_suffix + [suffix])

        for key in self.validation_step_outputs:
            if len(vals := self.validation_step_outputs[key]) > 0:
                self.log(f"test/{key}", torch.stack(vals).mean(), prog_bar=True)
                vals.clear()  # free memory

        if len(self.val_metric_manager.get_metric("psnr")) > 0:
            aggregated_psnr = self.val_metric_manager.aggregate("psnr")["psnr"]
            for method in aggregated_psnr:
                self.log(f"test/psnr/{method.name.lower()}", aggregated_psnr[method]["psnr"], prog_bar=True)
            self.val_metric_manager.clear("psnr")  # free memory

        if (
            self.config.test.metrics.cpsnr.enabled
            and (semantic_classes_map := self.get_ncore_semantic_classes_map()) is not None
        ):
            aggregated_cpsnr = self.val_metric_manager.aggregate("cpsnr")["cpsnr"]
            for method in aggregated_cpsnr:
                for class_name in semantic_classes_map.keys():
                    if class_name in aggregated_cpsnr[method].values:
                        value = aggregated_cpsnr[method][class_name]
                        self.log(
                            f"test/cpsnr/{method.name.lower()}/{class_name}",
                            value,
                            prog_bar=True,
                        )
            self.val_metric_manager.clear("cpsnr")  # free memory

        if isinstance(self.model.background, SkyEnvMapBackground):
            self.model.background.restore_original_textures()

        self.model.restore_training_parameters()

    def on_test_start(self) -> None:
        super().on_test_start()

        if isinstance(self.model.background, SkyEnvMapBackground):
            self.log_and_inpaint_env_map(self.model.background)

    def test_step(self, batch: DataAndRenderingBatch, batch_local_idx: int) -> None:
        self._raise_if_mcmc_publication_failed()
        if len(self.test_out_types):
            batch_idx = unpack_optional(batch.data.idx)

            results: GaussiansCompositeReturn = self(batch)

            if batch.data.camera is not None:
                self.save_images(
                    self.test_dir,
                    batch,
                    unpack_optional(results.rendered_cam),
                    batch_idx,
                    h=unpack_optional(batch.data.camera.h),
                    out_types=self.test_out_types,
                    suffix_list=self.val_suffix,
                )

    def on_test_epoch_end(self):
        if isinstance(self.model.background, SkyEnvMapBackground):
            self.model.background.restore_original_textures()

    def serialize_artifact_checkpoint(self, checkpoint: Checkpoint) -> None:
        """
        Set the precision and loading type for the artifacts checkpoint.

        For Gaussian systems we:
            - do not cast the checkpoint to fp16
            - set the load_in_place flag to false to enable enable dealing with dynamic number of Gaussians.
        """

        self.model.raise_if_mcmc_publication_failed()

        # Most checkpoints can be loaded in place, but if the dimensions of the buffers are not know in advance, they need to be assigned to the uninitialized tesnors instead.
        checkpoint["load_in_place"] = False

        ckpt = (
            strip_optimizer_state(checkpoint)
            if self.artifact_cache.artifact_config.checkpoint.strip_optimizer
            else checkpoint
        )
        if (
            self.artifact_cache.artifact_config.checkpoint.fp16
            and not self.artifact_cache.artifact_config.compression.enabled
        ):
            ckpt = reduce_precision_to_fp16(ckpt)
        self.artifact_cache.checkpoint = serialize_checkpoint(ckpt)  # TODO update when is_resumable is available

    def get_camera_and_frame_export_names(
        self, camera: DataBatch.Camera, datasource: NCOREDataSource, batch_idx: int
    ) -> tuple[str, str]:
        """Compute the output directory name and frame filename stem for a validation/test frame.

        Returns (cam_dir, image_name) based on the configured naming scheme.
        Only called when camera data is available (not for synthetic views like track_orbit).
        """
        unique_sensor_idx = camera.meta[0].unique_sensor_idx
        camera_id = datasource.all_camera_ids[unique_sensor_idx]
        cam_dir = camera_id if self.config.test.use_camera_name_dirs else f"cam_{unique_sensor_idx:02d}"
        match self.config.test.frame_naming:
            case "frame-end-timestamp":
                unique_frame_idx = camera.meta[0].unique_frame_idx
                sensor_frame_idx = (
                    unique_frame_idx
                    - datasource.camera_linear_start_frame_indices[camera_id]
                    + datasource.camera_frame_ranges[camera_id].start
                )
                frame_end_ts = datasource.camera_sensors[camera_id].get_frame_timestamp_us(
                    sensor_frame_idx, frame_timepoint=FrameTimepoint.END
                )
                # Zero-pad to 20 digits so a lexical sort over filenames (used by
                # save_videos() and any `ls`/`glob` consumer) matches numeric
                # ordering. 20 digits covers full-range int64 microseconds.
                # Prior unpadded form produced spliced/jumping val videos because
                # e.g. "1038406" (1.04s) lex-sorted between "10338377" and
                # "10438403" (10.34s and 10.44s).
                image_name = f"{frame_end_ts:020d}"
            case "batch-index":
                image_name = f"{batch_idx:06d}"
            case _:
                raise ValueError(f"Invalid frame naming scheme: {self.config.test.frame_naming}")
        return cam_dir, image_name

    def save_images(
        self,
        save_dir: str,
        batch: Optional[DataAndRenderingBatch],
        rendered: GaussiansRenderReturn,
        batch_idx: int,
        h: int,
        out_types: list[str],
        suffix_list: list[str],
        overlay_text: Optional[str] = None,
    ) -> None:
        camera = batch.data.camera if batch is not None else None
        datasource = self.datamodule.get_datasource()
        if camera is not None:
            assert isinstance(datasource, NCOREDataSource), "Only NCOREDataSource supported."
            cam_dir, image_name = self.get_camera_and_frame_export_names(camera, datasource, batch_idx)
        else:
            # Fallback for synthetic views (e.g. track_orbit) where batch/camera are not available.
            # Hardcode cam_00 and use a sequential index.
            cam_dir = "cam_00"
            image_name = f"{batch_idx:06d}"

        for out_type in out_types:
            im = None

            match out_type:
                case "pred_rgb":
                    if rendered.rgb is not None:
                        im = rearrange(rendered.rgb.clamp(0, 1).cpu().numpy(), "(h w) c -> h w c", h=h)
                        im = (im * 255).astype(np.uint8)
                        if overlay_text:
                            draw_text_overlay(im, overlay_text)

                case "pred_distance":
                    im = scalar2img(
                        rearrange(unpack_optional(rendered.distance).cpu().numpy(), "(h w) -> h w", h=h),
                        vmin=0.0,
                        vmax=self.get_max_dist_m(),
                    )

                case "pred_normal":
                    if rendered.normal is not None:
                        im = rearrange(rendered.normal.cpu().numpy(), "(h w) c -> h w c", h=h)
                        im = (im + 1) / 2 * 255
                        im = im.astype(np.uint8)

                case "pred_opacity":
                    im = rendered.opacity.cpu().numpy().reshape((h, -1))
                    im = (im * 255).astype(np.uint8)

                case "pred_semantic":
                    if (
                        rendered.extra_ray_signals is not None
                        and rendered.extra_ray_signals.semantic_logits is not None
                    ):
                        n_classes = rendered.extra_ray_signals.semantic_logits.shape[-1]

                        im = rearrange(
                            rendered.extra_ray_signals.semantic_logits.reshape((-1, n_classes))
                            .argmax(-1)
                            .cpu()
                            .numpy(),
                            "(h w) -> h w",
                            h=h,
                        )

                        colormap = self.datamodule.val_dataset.get_datasource().get_semantic_colormap(
                            camera_semantics=True, lidar_semantics=False
                        )
                        if colormap is not None:
                            im = sem2img(im, color_remap=colormap)

                case "input_rgb":
                    if camera is not None:
                        im = unpack_optional(camera.labels.rgb).cpu().numpy().squeeze()
                        im = (im * 255).astype(np.uint8)

                case "input_distance":
                    if camera is not None and camera.labels.metric_distance is not None:
                        im = scalar2img(
                            camera.labels.metric_distance.cpu().numpy().squeeze(),
                            vmin=0.0,
                            vmax=self.get_max_dist_m(),
                        )

                case "input_normal":
                    if camera is not None and camera.labels.normals is not None:
                        im = camera.labels.normals.cpu().numpy().squeeze()
                        im = (im + 1) / 2 * 255
                        im = im.astype(np.uint8)

                case "input_valid_mask":
                    if camera is not None:
                        im = camera.labels.get_mask_flags_none(RayFlags.INVALID).cpu().numpy().reshape((h, -1))
                        im = (im * 255).astype(np.uint8)

                case "input_sky_mask":
                    if camera is not None and isinstance(datasource, NCOREDataSource) and datasource.aux_data:
                        im = camera.labels.get_mask_flags_all(RayFlags.SKY_SEMANTIC).cpu().numpy().reshape((h, -1))
                        im = (im * 255).astype(np.uint8)

                case "input_road_mask":
                    if camera is not None and isinstance(datasource, NCOREDataSource) and datasource.aux_data:
                        im = camera.labels.get_mask_flags_all(RayFlags.ROAD_SEMANTIC).cpu().numpy().reshape((h, -1))
                        im = (im * 255).astype(np.uint8)

                case "input_semantic_mask":
                    if camera is not None and isinstance(datasource, NCOREDataSource) and datasource.aux_data:
                        semantic_colormap = datasource.get_semantic_colormap(
                            camera_semantics=True, lidar_semantics=False
                        )
                        semantic = camera.labels.semantic
                        if semantic_colormap is not None and semantic is not None:
                            semantic_class_id = semantic.cpu().numpy().reshape((h, -1))
                            im = semantic_colormap[semantic_class_id]

                case "bg_rgb":
                    if rendered.extra_ray_signals is not None and rendered.extra_ray_signals.rgb_background is not None:
                        rgb_background = rendered.extra_ray_signals.rgb_background * (1.0 - rendered.opacity[..., None])
                        im = rearrange(rgb_background.cpu().numpy(), "(h w) c -> h w c", h=h)
                        im = (im * 255).astype(np.uint8)

                case "rgb_before_post_processing":
                    if (
                        rendered.extra_ray_signals is not None
                        and rendered.extra_ray_signals.rgb_before_post_processing is not None
                    ):
                        im = rearrange(
                            rendered.extra_ray_signals.rgb_before_post_processing.cpu().numpy(), "(h w) c -> h w c", h=h
                        )
                        im = (im * 255).astype(np.uint8)

                case "pc_renders":
                    # saved in save_pointcloud_renders
                    continue

                case _:
                    raise ValueError(f"{self.__class__.__name__}: Unsupported output name")

            if not isinstance(im, np.ndarray):
                continue

            out_type_name = out_type
            if len(suffix_list) > 0:
                out_type_name = out_type_name + "-" + "-".join(suffix_list)

            os.makedirs(out_dir := os.path.join(save_dir, out_type_name, cam_dir), exist_ok=True)
            self._write_result_png(os.path.join(out_dir, image_name + ".png"), im)

            if self.config.save_logger:
                # log image sequences as batch-numerated '<image-type>(-suffix)-cam_<cam-idx>-step_<global-step>-rank_<rank>' metrics in UIs
                image_group = (
                    out_type_name
                    + "-cam_"
                    + cam_dir
                    + "-step_"
                    + str(self.global_step)
                    + "-rank_"
                    + str(self.global_rank)
                )

                self.log_image(im, image_group, batch_idx)

    @rank_zero_only
    def save_videos(self, save_dir: str, out_seq_types: list[str], suffix_list: list[str]) -> None:
        # Only rank 0 should create videos in distributed training/validation
        for out_seq_type in out_seq_types:
            out_type_name = out_seq_type
            if len(suffix_list) > 0:
                out_type_name = out_type_name + "-" + "-".join(suffix_list)

            if not os.path.exists(out_type_dir := os.path.join(save_dir, out_type_name)):
                continue
            for cam_dir in os.listdir(out_type_dir):
                video_file = os.path.join(save_dir, out_type_name + "-" + cam_dir + ".mp4")
                im_files = sorted(glob.glob(os.path.join(save_dir, out_type_name, cam_dir, "*.png")))

                # Loads all images at once - can run out of memory if too many frames
                ims: list[np.ndarray] = [imageio.v2.imread(f) for f in im_files]

                save_video(video_file, ims, fps=self.config.test.video_fps)

                if self.config.save_logger and isinstance(self.logger, WandbLogger):
                    # log video sequences as '<video-type>(-suffix)_video-cam_<cam-idx>-step_<global-step>-rank_<rank>' metrics in UIs
                    backend = media_backend()
                    if backend is None:
                        raise RuntimeError(
                            f"{type(self.logger).__name__} needs a media backend; none is registered. "
                            "Run through an entrypoint that ships one."
                        )
                    backend.log_video(
                        self.logger,
                        out_type_name
                        + "_video-"
                        + cam_dir
                        + "-step_"
                        + str(self.global_step)
                        + "-rank_"
                        + str(self.global_rank),
                        video_file,
                    )

    def load_state_dict(self, state_dict: Mapping[str, Any], *args, **kwargs):
        # The model's post-load hook invalidates pose-derived mappings; let the
        # next mGPU training step rebuild them before rendering.
        self._active_tile_mapping_prewarmer.reset()
        # The sensor models in calib aren't stored in the state_dict, let's keep on using the one that
        # we have created in the constructor.

        # TODO: ideally this code should be in the load_state_dict of the model that
        # contains the sensor models (CameraFreePoseViewGeometry or LidarFreePoseViewGeometry)
        # and somehow called from here (through GaussiansComposite and BaseCalib)

        updated_state_dict = dict(state_dict)

        if hasattr(self.model, "calib"):
            calib_sensor_models = {
                f"model.calib.{k}": v for k, v in self.model.calib.state_dict().items() if ".sensor_models." in k
            }
            updated_state_dict.update(calib_sensor_models)
            # mcity multiclip: a checkpoint without the per-image rotation table (e.g. the coarse start of a cell)
            # loads it as the module's zeros
            image_pose = getattr(self.model.calib, "image_pose", None)
            key = "model.calib.image_pose.residuals"
            if image_pose is not None and key not in updated_state_dict:
                updated_state_dict[key] = image_pose.residuals.detach().clone()

        return super().load_state_dict(updated_state_dict, *args, **kwargs)

    def on_save_checkpoint(self, checkpoint: Checkpoint) -> None:
        self._raise_if_mcmc_publication_failed()
        super().on_save_checkpoint(checkpoint)
