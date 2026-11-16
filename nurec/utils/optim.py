# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# TODO: Move this file to nurec/libs/math/optim.py.

from __future__ import annotations

import copy
import logging
import math

from collections import OrderedDict
from typing import Callable, NotRequired, Optional, Required, Type, TypedDict

import apex
import lietorch as lt
import numpy as np
import torch

from gsplat import has_adam  # type: ignore[import-not-found]
from gsplat.optimizers import SelectiveAdam as _GsplatSelectiveAdam  # type: ignore[import-not-found]
from omegaconf import DictConfig
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import (
    ExponentialLR,
    LinearLR,
    LRScheduler,
    PolynomialLR,
    ReduceLROnPlateau,
    _warn_get_lr_called_within_step,
)
from torch.optim.optimizer import Optimizer

from nurec.config.base_schema import BaseConfigSchema
from nurec.config.optim import OptimizerConfig, SchedulerConfig
from nurec.config.trainer import TrainerConfig
from nurec.utils.trainer import (
    adjust_betas_for_batch_size,
    adjust_gamma_for_world_size,
    adjust_learning_rate_for_batch_size,
    adjust_step_for_world_size,
)


log = logging.getLogger(__name__)

# Types parameterizing optimizers / lr schedulers

LRSchedulerTypeUnion = LRScheduler | ReduceLROnPlateau


def lr_scheduler_step_update_count(
    *,
    batch_local_idx: int,
    frequency: int,
    configured_steps_per_epoch: int,
    effective_steps_per_epoch: int,
) -> int:
    """Map one effective training batch to configured scheduler updates.

    This cumulative integer mapping reaches exactly the configured scheduler
    position at the end of each shortened epoch without floating-point drift.
    Equal epoch lengths reduce to the existing modulo-frequency behavior.
    """
    if batch_local_idx < 0:
        raise ValueError(f"batch_local_idx must be non-negative, got {batch_local_idx}")
    if frequency <= 0:
        raise ValueError(f"frequency must be positive, got {frequency}")
    if configured_steps_per_epoch <= 0:
        raise ValueError(f"configured_steps_per_epoch must be positive, got {configured_steps_per_epoch}")
    if effective_steps_per_epoch <= 0:
        raise ValueError(f"effective_steps_per_epoch must be positive, got {effective_steps_per_epoch}")

    denominator = effective_steps_per_epoch * frequency
    updates_before = batch_local_idx * configured_steps_per_epoch // denominator
    updates_after = (batch_local_idx + 1) * configured_steps_per_epoch // denominator
    return updates_after - updates_before


class LRSchedulerConfigType(TypedDict, total=False):
    """A typed dictionary representing an initialized LRScheduler for consumption by PL"""

    scheduler: Required[LRSchedulerTypeUnion]
    name: str | None
    interval: str
    frequency: int
    reduce_on_plateau: bool
    monitor: str | None
    strict: bool


class OptimizerLRSchedulerConfig(TypedDict):
    """A typed dictionary representing an initialized optimizer with (optional) initialized LRScheduler for consumption by PL"""

    optimizer: Optimizer
    lr_scheduler: NotRequired[LRSchedulerConfigType]


class SelectiveAdam(_GsplatSelectiveAdam):
    """PL-compatible wrapper around :class:`gsplat.optimizers.SelectiveAdam`.

    gsplat stages each visibility mask with :meth:`set_visibility` and consumes
    it through the standard ``optimizer.step(closure)`` interface. This subclass
    adds runtime switching between selective and dense updates.

    Requires gsplat's fused CUDA ``adam`` kernel (``GSPLAT_BUILD_ADAM=1``).
    Raises ``RuntimeError`` at construction time if the kernel is not available.

    The ``selective_enabled`` flag toggles between two update modes at runtime:
      * ``True``  — real selective update using the visibility mask supplied
                    via :meth:`set_visibility`.
      * ``False`` — dense update equivalent to plain Adam; an all-ones mask is
                    synthesised per step. ``set_visibility`` calls are still
                    accepted but the stored mask is ignored.

    Usage::

        optimizer = SelectiveAdam(params, lr=0.01, eps=1e-15, betas=(0.9, 0.99))

        # Each training step:
        visibility_mask = ...  # (N,) float tensor, >0 for visible
        optimizer.set_visibility(visibility_mask)
        optimizer.step()  # uses and clears the stored mask
    """

    # Maximum number of (N, device) all-ones masks retained in the dense-path LRU cache.
    _ONES_CACHE_CAPACITY: int = 2

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        selective_enabled: bool = True,
    ):
        if not has_adam():
            raise RuntimeError(
                "SelectiveAdam requires gsplat's fused CUDA adam kernel but it is not compiled. "
                "Ensure ADAM is included in the --@gsplat//bazel:kernel_families flag (see .bazelrc)."
            )
        super().__init__(params, lr=lr, betas=betas, eps=eps)
        self.selective_enabled: bool = selective_enabled
        # Fixed-capacity LRU cache of all-ones masks keyed by (N, device), populated
        # lazily in the dense path. Bounding it prevents changing Gaussian counts
        # (e.g. MCMC add/relocate) from accumulating stale masks and leaking GPU memory.
        self._ones_cache: OrderedDict[tuple[int, torch.device], torch.Tensor] = OrderedDict()

    def set_visibility(self, visibility: Optional[torch.Tensor]) -> None:
        """Stage a Boolean visibility mask for the next optimizer step.

        Args:
            visibility: Boolean or numeric tensor with one entry per Gaussian.
                Empty tensors and ``None`` clear the staged mask.
        """
        if visibility is not None:
            if visibility.numel() == 0:
                visibility = None
            elif visibility.dtype != torch.bool:
                visibility = visibility.bool()
        super().set_visibility(visibility)

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero gradients, always setting to ``None`` (matches FusedAdam behaviour)."""
        super().zero_grad(set_to_none=True)

    def _get_ones_mask(self) -> torch.Tensor:
        """Return a cached all-ones mask sized to this optimizer's first param group.

        Backed by a fixed-capacity LRU cache (``_ONES_CACHE_CAPACITY``) keyed by
        ``(N, device)``. Gaussian counts change during training (e.g. MCMC add/relocate),
        so evicting the least-recently-used entry once over capacity keeps memory bounded
        instead of retaining a mask for every count ever seen.
        """
        # All param groups in a single SelectiveAdam share the same Gaussian count N
        # (the mask is indexed by Gaussian, not by param). Verify that invariant on every
        # lookup rather than only at construction: densification/relocation (e.g. MCMC
        # add/relocate) can replace parameter tensors between steps, so a one-time check
        # could go stale. A mismatch means the cached mask — sized from the first param —
        # would be silently wrong for the other groups, so fail loudly instead.
        grouped_params = [(group, param) for group in self.param_groups for param in group["params"]]
        assert grouped_params, "SelectiveAdam requires at least one parameter to build a visibility mask."
        for group, param in grouped_params:
            assert param.ndim > 0, (
                "SelectiveAdam parameters must have a leading Gaussian dimension N; "
                f"got scalar parameter in group {group.get('name', '?')!r}."
            )

        first_param = grouped_params[0][1]
        n = first_param.shape[0]
        for group, param in grouped_params:
            assert param.shape[0] == n, (
                "SelectiveAdam requires all parameters across all param groups to share the "
                f"same leading Gaussian dimension N; got {param.shape[0]} in group "
                f"{group.get('name', '?')!r} vs {n}. The visibility mask is indexed by "
                "Gaussian, so mixing different N in one SelectiveAdam would mask wrong rows."
            )
        key = (n, first_param.device)
        mask = self._ones_cache.get(key)
        if mask is None:
            mask = torch.ones(key[0], device=key[1], dtype=torch.bool)
            self._ones_cache[key] = mask
            # Evict least-recently-used entries until back within capacity.
            while len(self._ones_cache) > self._ONES_CACHE_CAPACITY:
                self._ones_cache.popitem(last=False)
        else:
            # Refresh recency: mark this key as most-recently-used.
            self._ones_cache.move_to_end(key)
        return mask

    def _num_gaussians(self) -> int:
        """Return this optimizer's shared leading (Gaussian) dimension.

        Every parameter in a SelectiveAdam shares the same leading dimension because the
        visibility mask and gsplat's fused kernel are indexed by Gaussian, not by parameter,
        so the first parameter is authoritative (same access as :meth:`_get_ones_mask`).
        Returns 0 when the optimizer holds no parameters or when its parameters have been
        emptied (leading dim 0) by densification/relocation. Read fresh each step because
        densification/relocation can resize parameters between steps.
        """
        params = self.param_groups[0]["params"]
        if not params:
            return 0
        first_param = params[0]
        return first_param.shape[0] if first_param.ndim > 0 else 0

    @torch.no_grad()
    def step(self, closure=None):
        """Perform an optimisation step using gsplat's fused CUDA kernel.

        With ``selective_enabled=True``, requires :meth:`set_visibility` to have
        been called since the previous step. With ``selective_enabled=False``,
        any stored mask is ignored and a dense update is performed.

        A zero-Gaussian optimizer (leading dim 0, e.g. a layer emptied by
        densification/relocation) is a no-op: there is nothing to update, and both
        the visibility mask and the fused kernel are indexed by Gaussian.
        """
        # Skip layers that currently hold no Gaussians. Densification/relocation can empty a
        # layer's parameters (leading dim 0), in which case the post-backward visibility staging
        # hook provides an empty (numel()==0) slice. There is nothing to step, so treat it as a
        # genuine no-op instead of letting gsplat reject the cleared mask — otherwise training
        # aborts the moment any selective layer is pruned to empty.
        if self._num_gaussians() == 0:
            super().set_visibility(None)
            if closure is not None:
                with torch.enable_grad():
                    return closure()
            return None

        if not self.selective_enabled:
            # Dense path: synthetic all-ones mask, equivalent to plain Adam.
            super().set_visibility(self._get_ones_mask())

        if self._visibility is None:
            param_names = [group.get("name", "?") for group in self.param_groups]
            raise RuntimeError(
                f"SelectiveAdam.step() called with no visibility mask for params {param_names}. "
                "Call set_visibility() before each step, set selective_enabled=False, "
                "or use FusedAdam/Adam instead."
            )

        return super().step(closure)


optim_fns: dict[str, Callable] = {
    "fused_adam": apex.optimizers.FusedAdam,
    "selective_adam": SelectiveAdam,
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}


def classify_gaussian_optimizer_group(param_groups: list[dict]) -> str:
    """Classify a Gaussian-layer optimizer as density, scale, or general.

    Densities and scales each get their own optimizer behavior; positions, rotations,
    and features fall into general optimizer behavior. Reading the suffix of the first param
    group is therefore enough to tell them apart.

    Unnamed or empty inputs fall back to general.
    """
    if not param_groups:
        return "general"

    name = param_groups[0].get("name")
    if name:
        suffix = name.rsplit(".", 1)[-1]
        if suffix == "densities":
            return "density"
        if suffix == "scales":
            return "scale"
    return "general"


def get_model_parameters(model: torch.nn.Module, config: OptimizerConfig, name_prefix: str = "") -> list[dict]:
    params = []
    name_prefix = (name_prefix + ".") if name_prefix else ""

    def getattr_recursive(m: object, attr: str) -> object:
        for name in attr.split("."):
            m = getattr(m, name)
        return m

    def append_param_recursive(param_name: str, param: object, param_options: dict):
        """Recursively register parameter group if parameter options contains 'args' key - otherwise recurse down the model's hierarchy"""
        if param_args := param_options.get("args"):
            match param:
                case None:
                    # intentionally skip a parameter group if it is not initialized to a value - this
                    # will result in optimizers not being instantiated if none of it's parameters is
                    # set to an optimizable value (e.g., if all parameters should be unconditionally frozen)
                    pass
                case torch.nn.Parameter():
                    params.append({"params": [param], "name": name_prefix + param_name, **param_args})
                case torch.nn.ParameterList():
                    params.append({"params": list(param), "name": name_prefix + param_name, **param_args})
                case torch.nn.ParameterDict():
                    params.append({"params": list(param.values()), "name": name_prefix + param_name, **param_args})
                case torch.nn.Module():
                    module_parameters = filter(lambda p: p.requires_grad, param.parameters())
                    n_params = sum([np.prod(p.size(), dtype=int) for p in module_parameters])

                    if n_params > 0:
                        params.append(
                            {"params": list(param.parameters()), "name": name_prefix + param_name, **param_args}
                        )
                case lt.LieGroupParameter():
                    params.append({"params": [param], "name": name_prefix + param_name, **param_args})
        else:
            # deepen one level to sub-parameters within current model level
            for param_subname, param_suboptions in param_options.items():
                append_param_recursive(
                    param_name + "." + param_subname, getattr_recursive(param, param_subname), param_suboptions
                )

    # process top-level list of parameter groups within config
    assert config.params is not None
    for param_name, param_options in config.params.items():
        if param_options is not None:
            append_param_recursive(param_name, getattr_recursive(model, param_name), param_options)

    return params


def adjust_param_group_learning_rates(
    trainer_config: TrainerConfig, param_options: dict, config_name: str, group_name: str = ""
) -> None:
    """Recursively adjust learning rates in parameter group configurations to handle both flat and nested structures."""
    if param_args := param_options.get("args"):
        # Flat structure: has direct args with lr
        if "lr" in param_args:
            param_args["lr"] = adjust_learning_rate_for_batch_size(trainer_config, param_args["lr"])
            log.info("Optimizer/%s/%s: lr=%f", config_name, group_name, param_args["lr"])
    else:
        # Nested structure: recurse into sub-parameters
        for subkey, suboptions in param_options.items():
            if isinstance(suboptions, dict):
                sub_group_name = f"{group_name}.{subkey}" if group_name else subkey
                adjust_param_group_learning_rates(trainer_config, suboptions, config_name, sub_group_name)


def parse_scheduler_config(
    config: SchedulerConfig,
    optimizer: Optimizer,
    trainer_config: TrainerConfig,
    recursion_depth: int = 0,
) -> LRSchedulerConfigType:
    log.info("%sScheduler/%s", "  " * recursion_depth, config.name)

    # This function is used to instantiate the scheduler.
    def init_scheduler(interval: str, name: str, *args, **kwargs) -> LRSchedulerTypeUnion:
        """
        In this function, we first adjust the scheduler parameters based on the distributed world size,
        then we instantiate the scheduler.
        """

        def logfunc(msg):
            log.info("  └─%s", msg)

        if name in _StepFunSchedulers:
            kwargs["interval"] = interval
            kwargs["trainer"] = trainer_config
            return _StepFunSchedulers[name](*args, **kwargs)

        elif name in ["ExponentialLR"]:
            if "gamma" in kwargs:
                kwargs["gamma"] = adjust_gamma_for_world_size(trainer_config, kwargs["gamma"])
                logfunc(f"gamma={kwargs['gamma']}")

        elif name in ["ConstantLR", "LinearLR", "PolynomialLR"]:
            if "total_iters" in kwargs:
                kwargs["total_iters"] = adjust_step_for_world_size(trainer_config, kwargs["total_iters"])
                logfunc(f"total_iters={kwargs['total_iters']}")

        elif name in ["OneCycleLR"]:
            if "steps_per_epoch" in kwargs:
                kwargs["steps_per_epoch"] = adjust_step_for_world_size(trainer_config, kwargs["steps_per_epoch"])
                logfunc(f"steps_per_epoch={kwargs['steps_per_epoch']}")

            if "total_steps" in kwargs:
                kwargs["total_steps"] = adjust_step_for_world_size(trainer_config, kwargs["total_steps"])
                logfunc(f"total_steps={kwargs['total_steps']}")

            if "max_lr" in kwargs:
                if isinstance(kwargs["max_lr"], list):
                    kwargs["max_lr"] = [
                        adjust_learning_rate_for_batch_size(trainer_config, lr) for lr in kwargs["max_lr"]
                    ]
                else:
                    kwargs["max_lr"] = adjust_learning_rate_for_batch_size(trainer_config, kwargs["max_lr"])
                logfunc(f"max_lr={kwargs['max_lr']}")

        elif name in ["CosineAnnealingLR"]:
            if "T_max" in kwargs:
                kwargs["T_max"] = adjust_step_for_world_size(trainer_config, kwargs["T_max"])
                logfunc(f"T_max={kwargs['T_max']}")

        elif name in ["SequentialLR"]:
            if "milestones" in kwargs:
                if interval == "step":
                    kwargs["milestones"] = [
                        adjust_step_for_world_size(trainer_config, milestone) for milestone in kwargs["milestones"]
                    ]
                logfunc(f"milestones={kwargs['milestones']}")

        # Now we instantiate the scheduler
        if hasattr(lr_scheduler, name):
            return getattr(lr_scheduler, name)(*args, **kwargs)
        else:
            raise NotImplementedError(f"Scheduler {name} not implemented")

    # Construct the top-level scheduler and build the nested schedulers recursively.
    interval: str = getattr(config, "interval", "epoch")
    assert interval in ["epoch", "step"]

    if config.name == "SequentialLR":
        assert config.schedulers is not None, "SequentialLR requires nested 'schedulers'"
        return {
            "scheduler": init_scheduler(
                interval,
                config.name,
                optimizer,
                schedulers=[
                    parse_scheduler_config(conf, optimizer, trainer_config, recursion_depth + 1)["scheduler"]
                    for conf in config.schedulers
                ],
                milestones=config.milestones,
            ),
            "interval": interval,
        }
    elif config.name == "ChainedScheduler":
        assert config.schedulers is not None, "ChainedScheduler requires nested 'schedulers'"
        return {
            "scheduler": getattr(lr_scheduler, config.name)(
                [
                    parse_scheduler_config(conf, optimizer, trainer_config, recursion_depth + 1)["scheduler"]
                    for conf in config.schedulers
                ]
            ),
            "interval": interval,
        }
    else:
        return {"scheduler": init_scheduler(interval, config.name, optimizer, **config.args), "interval": interval}


def parse_optimizer(
    config: OptimizerConfig,
    model: torch.nn.Module,
    trainer_config: TrainerConfig,
    name_prefix: str = "",
) -> Optional[Optimizer]:
    """Instantiate optimizer for a given model and (optional) config-provided parameter group parametrizations

    Skip instantiation of optimizer if *no* parameter groups would be associated with the new instance
    (e.g., for optimizers that would only affect frozen model components)
    """

    config = config.model_copy(deep=True) if isinstance(config, BaseConfigSchema) else copy.deepcopy(config)

    if config.name in ("fused_adam", "selective_adam"):
        config.args["lr"] = adjust_learning_rate_for_batch_size(trainer_config, config.args["lr"])
        log.info("Optimizer/%s: lr=%f", config.name, config.args["lr"])

        config.args["betas"] = adjust_betas_for_batch_size(trainer_config, config.args["betas"])
        log.info("Optimizer/%s: betas=%s", config.name, config.args["betas"])

        config_params = getattr(config, "params", None)
        if config_params is not None:
            for key, group in config_params.items():
                adjust_param_group_learning_rates(trainer_config, group, config.name, key)
    else:
        raise NotImplementedError(f"Optimizer {config.name} not supported")

    params: list[dict] | list[torch.nn.Parameter]
    if config_params is not None:
        # parse config-provided parameter group parametrizations and associated with model
        params = get_model_parameters(model, config, name_prefix)
    else:
        # config doesn't provide parameter groups parametrizations, use all parameter groups from model as is
        params = list(model.parameters())

    if not len(params):
        # don't initialize an optimizer if no parameter groups are going to be associated with it
        return None

    return optim_fns[config.name](params, **config.args)


def configure_optimizers(
    config: BaseConfigSchema | DictConfig,
    trainer_config: TrainerConfig,
    model: torch.nn.Module | torch.nn.ModuleList,
    name_prefix: str = "",
) -> list[OptimizerLRSchedulerConfig]:
    """
    Construct PL parameters for single optimizer (stored in 'system.optimizer' config) or
    multiple optimizers (stored in 'system.optimizers' config list), associated with the
    model's parameter groups (both with optional lr_schedulers if present)

    Note: config accepts BaseConfigSchema | DictConfig during the transition to typed configs.
    Some callers (e.g. calib.py, base.py, post_processing.py) still pass DictConfig via
    .to_dictconfig(). The inner functions (parse_optimizer, parse_scheduler_config) are typed
    for OptimizerConfig/SchedulerConfig but work with DictConfig at runtime thanks to duck typing.
    These callers will be migrated in follow-up MRs.
    """

    ret: list[OptimizerLRSchedulerConfig] = []

    def parse_append_single_optimizer(
        optimizer_config: OptimizerConfig,
        scheduler_config: SchedulerConfig | None,
    ) -> None:
        """
        Parses and instantiate an optimizer and associated parameter group configuration
        of a single optimizer, and appends it to the output list of optimizer dictionaries.
        """
        if (optim := parse_optimizer(optimizer_config, model, trainer_config, name_prefix)) is None:
            return

        if scheduler_config is not None:
            lr_scheduler = parse_scheduler_config(scheduler_config, optim, trainer_config)
            ret.append({"optimizer": optim, "lr_scheduler": lr_scheduler})
        else:
            ret.append({"optimizer": optim})

    if hasattr(config, "optimizers") and config.optimizers is not None:
        # multiple optimizers case
        for optimizer_config in config.optimizers:
            parse_append_single_optimizer(optimizer_config, getattr(optimizer_config, "scheduler", None))

    elif hasattr(config, "optimizer") and config.optimizer is not None:
        # single optimizer case
        parse_append_single_optimizer(config.optimizer, getattr(config, "scheduler", None))

    return ret


class StepFunLRSchedulerMixin(LRScheduler):
    """
    A mixin class for `LRScheduler` subclasses that discretize the continuous lr functions into step functions.
    Overrides the internal step mechanisms with `update_every_n_steps`.
    """

    update_every_n_steps: int

    def _initial_step(self):
        self._step_count_actual = 0
        super()._initial_step()

    def step(self, epoch: Optional[int] = None):
        if epoch is None:
            self._step_count_actual += 1
        else:
            self._step_count_actual = epoch

        if self._step_count_actual % self.update_every_n_steps == 0:
            epoch_internal = (epoch // self.update_every_n_steps) if epoch is not None else None
            super().step(epoch_internal)


class StepFunCosineAnnealingLR(StepFunLRSchedulerMixin):
    """
    Modified version of `torch.optim.lr_scheduler.CosineAnnealingLR`.
    Support either of:
    - Setting an absolute final lr for all the optimizer groups using `eta_min`
    - Setting a relative final lr factor based on the different initial lrs of different optimizer groups using `min_factor`
    """

    T_max: float
    eta_mins: list[float]

    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int,
        interval: str,
        trainer: TrainerConfig,
        eta_min: Optional[float] = None,
        min_factor: Optional[float] = None,
        last_epoch: int = -1,
        update_every_n_steps: int = 1,
    ):
        if interval == "step":
            T_max = adjust_step_for_world_size(trainer, T_max)
            update_every_n_steps = adjust_step_for_world_size(trainer, update_every_n_steps)

        last_epoch = (last_epoch // update_every_n_steps) if last_epoch > 0 else last_epoch
        self.update_every_n_steps = update_every_n_steps
        LRScheduler.__init__(self, optimizer, last_epoch)

        # Adding the numerator with (update_every_n_steps-1) to prevent over flooring too much in the division
        self.T_max = max((T_max + update_every_n_steps - 1) // update_every_n_steps, 1)
        if eta_min is None:
            assert min_factor is not None, "Please specify `min_factor` if `eta_min` is not provided"
            self.eta_mins = [(float(group["initial_lr"]) * min_factor) for group in optimizer.param_groups]
        else:
            assert min_factor is None, "Please do not specify `min_factor` since `eta_min` is provided"
            self.eta_mins = [eta_min for _ in optimizer.param_groups]

    def get_lr(self):
        _warn_get_lr_called_within_step(self)

        if self.last_epoch == 0:
            return [group["lr"] for group in self.optimizer.param_groups]
        elif self._step_count == 1 and self.last_epoch > 0:
            return [
                eta_min + (base_lr - eta_min) * (1 + math.cos((self.last_epoch) * math.pi / self.T_max)) / 2
                for base_lr, eta_min, group in zip(self.base_lrs, self.eta_mins, self.optimizer.param_groups)
            ]
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [
                group["lr"] + (base_lr - eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
                for base_lr, eta_min, group in zip(self.base_lrs, self.eta_mins, self.optimizer.param_groups)
            ]
        return [
            (1 + math.cos(math.pi * self.last_epoch / self.T_max))
            / (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max))
            * (group["lr"] - eta_min)
            + eta_min
            for eta_min, group in zip(self.eta_mins, self.optimizer.param_groups)
        ]

    def _get_closed_form_lr(self):
        return [
            eta_min + (base_lr - eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
            for base_lr, eta_min in zip(self.base_lrs, self.eta_mins)
        ]


class StepFunExponentialLR(StepFunLRSchedulerMixin, ExponentialLR):
    def __init__(
        self,
        optimizer: Optimizer,
        gamma: float,
        interval: str,
        trainer: TrainerConfig,
        last_epoch: int = -1,
        update_every_n_steps: int = 1,
    ):
        if interval == "step":
            update_every_n_steps = adjust_step_for_world_size(trainer, update_every_n_steps)

        last_epoch = (last_epoch // update_every_n_steps) if last_epoch > 0 else last_epoch
        self.update_every_n_steps = update_every_n_steps
        super().__init__(optimizer, gamma, last_epoch)


class StepFunPolynomialLR(StepFunLRSchedulerMixin, PolynomialLR):
    def __init__(
        self,
        optimizer: Optimizer,
        interval: str,
        trainer: TrainerConfig,
        total_iters: int = 5,
        power: float = 1.0,
        last_epoch: int = -1,
        update_every_n_steps: int = 1,
    ):
        if interval == "step":
            update_every_n_steps = adjust_step_for_world_size(trainer, update_every_n_steps)
            total_iters = adjust_step_for_world_size(trainer, total_iters)

        last_epoch = (last_epoch // update_every_n_steps) if last_epoch > 0 else last_epoch
        self.update_every_n_steps = update_every_n_steps
        # Adding the numerator with (update_every_n_steps-1) to prevent over flooring too much in the division
        total_iters = max((total_iters + update_every_n_steps - 1) // update_every_n_steps, 1)
        super().__init__(optimizer, total_iters, power, last_epoch)


class StepFunLinearLR(StepFunLRSchedulerMixin, LinearLR):
    def __init__(
        self,
        optimizer: Optimizer,
        interval: str,
        trainer: TrainerConfig,
        start_factor: float = 1.0 / 3,
        end_factor: float = 1.0,
        total_iters: int = 5,
        last_epoch: int = -1,
        update_every_n_steps: int = 1,
    ):
        if interval == "step":
            update_every_n_steps = adjust_step_for_world_size(trainer, update_every_n_steps)
            total_iters = adjust_step_for_world_size(trainer, total_iters)

        last_epoch = (last_epoch // update_every_n_steps) if last_epoch > 0 else last_epoch
        self.update_every_n_steps = update_every_n_steps
        # Adding the numerator with (update_every_n_steps-1) to prevent over flooring too much in the division
        total_iters = max((total_iters + update_every_n_steps - 1) // update_every_n_steps, 1)
        super().__init__(optimizer, start_factor, end_factor, total_iters, last_epoch)


class WarmupExponentialLR(LRScheduler):
    """
    A linear warm-up from `start_factor` to 1 over `warmup_steps` steps, multiplied by an exponential decay `gamma^t`
    from step 0: lr(t) = base_lr * (start_factor + (1 - start_factor) * min(t, warmup_steps) / warmup_steps) * gamma^t.
    For a run that starts from a trained model (a warm start), whose fresh Adam state otherwise moves every parameter by
    about one learning rate per step at first. Closed form on `base_lrs` (unlike Sequential/ChainedScheduler), so the
    Gaussian model's scene-extent scaling of the position group's `base_lrs` carries through. gamma=1: warm-up only.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        interval: str,
        trainer: TrainerConfig,
        warmup_steps: int,
        start_factor: float,
        gamma: float = 1.0,
        last_epoch: int = -1,
    ):
        if interval == "step":
            warmup_steps = adjust_step_for_world_size(trainer, warmup_steps)
            gamma = adjust_gamma_for_world_size(trainer, gamma)
        assert warmup_steps >= 1 and 0.0 < start_factor <= 1.0, (warmup_steps, start_factor)
        self.warmup_steps, self.start_factor, self.gamma = int(warmup_steps), float(start_factor), float(gamma)
        super().__init__(optimizer, last_epoch)

    def _factor(self, t: int) -> float:
        ramp = min(t, self.warmup_steps) / self.warmup_steps
        return (self.start_factor + (1.0 - self.start_factor) * ramp) * self.gamma**t

    def get_lr(self) -> list[float]:
        return [base_lr * self._factor(self.last_epoch) for base_lr in self.base_lrs]

    def _get_closed_form_lr(self) -> list[float]:
        return self.get_lr()


class WarmupStableDecayLR(LRScheduler):
    """
    Warm-up, stable, decay: a linear ramp from `start_factor` to 1 over `warmup_steps`, the full rate until
    `decay_start`, then a decay to `final_factor` at `total_steps` (`shape`: exp, cos or linear), held after it.
    For a warm start that should explore at its peak rate for most of the run and refine in a short anneal at the end
    (mcity 2026-11-16). Closed form on `base_lrs`, like WarmupExponentialLR, so the scene-extent scaling still applies.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        interval: str,
        trainer: TrainerConfig,
        warmup_steps: int,
        start_factor: float,
        decay_start: int,
        total_steps: int,
        final_factor: float,
        shape: str = "exp",
        last_epoch: int = -1,
    ):
        if interval == "step":
            warmup_steps = adjust_step_for_world_size(trainer, warmup_steps)
            decay_start = adjust_step_for_world_size(trainer, decay_start)
            total_steps = adjust_step_for_world_size(trainer, total_steps)
        assert 1 <= warmup_steps <= decay_start < total_steps, (warmup_steps, decay_start, total_steps)
        assert 0.0 < start_factor <= 1.0 and 0.0 < final_factor <= 1.0 and shape in ("exp", "cos", "linear")
        self.warmup_steps, self.start_factor = int(warmup_steps), float(start_factor)
        self.decay_start, self.total_steps = int(decay_start), int(total_steps)
        self.final_factor, self.shape = float(final_factor), shape
        super().__init__(optimizer, last_epoch)

    def _factor(self, t: int) -> float:
        if t < self.warmup_steps:
            return self.start_factor + (1.0 - self.start_factor) * t / self.warmup_steps
        if t < self.decay_start:
            return 1.0
        x = min((t - self.decay_start) / (self.total_steps - self.decay_start), 1.0)
        f = self.final_factor
        if self.shape == "exp":
            return f**x
        if self.shape == "cos":
            return f + (1.0 - f) * 0.5 * (1.0 + math.cos(math.pi * x))
        return 1.0 - (1.0 - f) * x

    def get_lr(self) -> list[float]:
        return [base_lr * self._factor(self.last_epoch) for base_lr in self.base_lrs]

    def _get_closed_form_lr(self) -> list[float]:
        return self.get_lr()


_StepFunSchedulers: dict[str, Type[LRScheduler]] = {
    cls.__name__: cls
    for cls in (
        StepFunCosineAnnealingLR,
        StepFunPolynomialLR,
        StepFunExponentialLR,
        StepFunLinearLR,
        WarmupExponentialLR,
        WarmupStableDecayLR,
    )
}
