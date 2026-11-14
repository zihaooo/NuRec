# SPDX-License-Identifier: Apache-2.0
"""Efficient Confidence-Aware Densification (ECAD) on top of NuRec's MCMC strategy.

Re-implements Hofer et al., "Efficient Confidence-Aware Densification for 3DGS" (arXiv 2610.13541; no reference code
was released) with NuRec's MCMC bookkeeping and NVIDIA gsplat's MCMC row plans. Like 3DGS-MCMC it keeps a fixed
budget, grows 5% per densification step and applies MCMC's opacity/scale split correction to every copied source.
Unlike MCMC it

* draws 80% of the sources with probability r(grad) * ((1 - eta) + eta * (1.2 - r(confidence))), where r() is the rank
  in [0, 1] and eta = min(t / 5000, 1), and 20% in proportion to the SSIM error each Gaussian carries;
* relocates, every 400 steps, the Gaussians that contribute least to the images: before 5k only those that contribute
  nothing (at most 10%), then the lowest 10%, decaying as exp(-((t - 10k) / 5k)^2 / 2) from 10k until 25k;
  Gaussians moved by one relocation are exempt from the next;
* places each copy at a draw from N(mu, Sigma') of its source and uses no SGLD position noise.

Per-Gaussian confidence gamma lives in channel 0 of `camera_extra_signal` (copied with its source by every MCMC row
plan) and is trained by the `ecad_confidence` loss; see `_ECADRenderState` in renderers.py for the probes.

Departures from the paper, forced by gsplat's 3DGUT renderer:
* Gradient term. The paper ranks Gaussians by their absolute screen-space position gradient (AbsGS). 3DGUT evaluates
  Gaussians in 3D and has no screen-space gradient, so this uses the 3D position-gradient norm times the distance to
  the camera, averaged over the views that saw the Gaussian since the previous densification step, the substitute
  NuRec's own 3DGUT densifier uses.
* Contribution. The paper's w_i is the maximum blending weight over pixels and views since the previous relocation.
  gsplat exposes no per-Gaussian maximum, so each view estimates it as sum_p b_i(p) / (I_i * m_i), with I_i the
  integral of the projected 2D Gaussian and m_i the share of it inside gsplat's opacity-aware cut-off: opacity times
  the mean transmittance in front of the Gaussian. That equals the maximum weight when transmittance is uniform over
  the footprint; w_i is the maximum of it over views.
* Colour initialisation of new Gaussians from a later training view is not implemented.
"""

from __future__ import annotations

import math

from typing import Optional

import torch

from gsplat.strategy import MCMCOperation, MCMCRowPlan
from torch import nn
from tqdm import tqdm

from nurec.config.model import ECADStrategyConfig
from nurec.models.gaussians.gaussians_model import BaseGaussianModel
from nurec.models.gaussians.renderers import ECAD_RENDER_STATE
from nurec.models.gaussians.strategies.base import BaseGaussianStrategy
from nurec.models.gaussians.strategies.gsplat_mcmc_adapter import _apply_and_publish, _build_owners
from nurec.models.gaussians.strategies.mcmc import MCMCStrategy
from nurec.models.gaussians.strategies.selected_gaussian_nodes import SelectedGaussianNodes
from nurec.models.nn_extensions import TypedModuleDict
from nurec.utils.batch import DataAndRenderingBatch
from nurec.utils.cuda_rng import get_cuda_generator
from nurec.utils.profiling import ScopedTimer
from nurec.utils.trainer import TrainerConfig


_ALPHA_THRESHOLD = 1.0 / 255.0  # gsplat culls Gaussians below this opacity and skips pixels below this alpha
_GAUSSIAN_EXTEND = 3.33  # gsplat's cap on the opacity-aware cut-off, in standard deviations


def _rank01(values: torch.Tensor) -> torch.Tensor:
    """Rank of each value in [0, 1]: the smallest gets 0, the largest 1."""
    n = values.numel()
    ranks = torch.empty_like(values, dtype=torch.float32)
    ranks[torch.argsort(values, stable=True)] = torch.arange(n, device=values.device, dtype=torch.float32)
    return ranks / max(n - 1, 1)


def _quat_to_rotmat(quats: torch.Tensor) -> torch.Tensor:
    """[K, 4] wxyz quaternions (gsplat's convention) to [K, 3, 3] rotation matrices."""
    w, x, y, z = torch.nn.functional.normalize(quats, dim=-1).unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(-1, 3, 3)


class ECADStrategy(MCMCStrategy):
    """Efficient Confidence-Aware Densification; see the module docstring."""

    config: ECADStrategyConfig

    def __init__(
        self,
        config: ECADStrategyConfig,
        trainer_config: TrainerConfig,
        init_from_datasource: bool,
        gaussians_nodes: TypedModuleDict[BaseGaussianModel],
    ) -> None:
        super().__init__(config, trainer_config, init_from_datasource, gaussians_nodes)
        if trainer_config.world_size != 1:
            raise ValueError("ECAD supports a single GPU")
        if self.config.independent_layer_max_n_gaussians or self.config.cuboid_grouped_layer_ids:
            raise ValueError("ECAD does not support independent or cuboid-grouped MCMC pools")
        if self.config.relocate.same_track_layer_ids:
            raise ValueError("ECAD does not support same-track relocation")
        for layer_id in self.config.ecad.layer_ids:
            node = gaussians_nodes[layer_id]
            if node.config.particle.camera_extra_signal_dim < 1:
                raise ValueError(
                    f"ECAD stores confidence in camera_extra_signal[:, 0]; set "
                    f"model.layers.{layer_id}.particle.camera_extra_signal_dim >= 1"
                )
        self._fresh_start = init_from_datasource
        self._confidence_initialized = False
        # Per-layer, row-aligned statistics; plain tensors, not checkpointed (a resumed run restarts its windows).
        self._grad_sum: dict[str, torch.Tensor] = {}
        self._grad_count: dict[str, torch.Tensor] = {}
        self._ssim_sum: dict[str, torch.Tensor] = {}
        self._contribution: dict[str, torch.Tensor] = {}
        self._exempt: dict[str, torch.Tensor] = {}
        self._last_log: dict[str, float] = {}
        self._warned: set[str] = set()
        ECAD_RENDER_STATE.enabled = True
        e = self.config.ecad
        tqdm.write(
            f"ECADStrategy: layers={e.layer_ids} relocate every {self.config.relocate.frequency} steps "
            f"[{self.config.relocate.start_iteration}, {self.config.relocate.end_iteration}), add every "
            f"{self.config.add.frequency} up to {self.config.add.max_n_gaussians}, gradient share {e.gradient_share}, "
            f"relocation share {e.relocation_share}, position sampling {e.sample_positions}, "
            f"MCMC noise {e.keep_mcmc_noise}"
        )

    # ------------------------------------------------------------------ setup

    @torch.no_grad()
    def maybe_initialize_buffers(self, gaussians_nodes: TypedModuleDict[BaseGaussianModel]) -> None:
        super().maybe_initialize_buffers(gaussians_nodes)
        if not self._fresh_start or self._confidence_initialized:
            return
        for layer_id in self.config.ecad.layer_ids:
            node = gaussians_nodes[layer_id]
            if isinstance(node.camera_extra_signal, nn.UninitializedParameter):
                return
        for layer_id in self.config.ecad.layer_ids:
            gaussians_nodes[layer_id].camera_extra_signal[:, 0].fill_(self.config.ecad.init_confidence)
        self._confidence_initialized = True

    def _warn_once(self, key: str, message: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            tqdm.write(f"ECADStrategy WARNING: {message}")

    # ------------------------------------------------------------- statistics

    def _stat_rows(self, layer_id: str, num: int, device: torch.device) -> None:
        """Make every per-layer statistic row-aligned with `num` rows, zero-filling new rows."""

        def fit(store: dict[str, torch.Tensor], dtype: torch.dtype) -> None:
            old = store.get(layer_id)
            if old is None or old.numel() != num or old.device != device:
                new = torch.zeros(num, dtype=dtype, device=device)
                if old is not None and old.device == device:
                    keep = min(num, old.numel())
                    new[:keep] = old[:keep]
                store[layer_id] = new

        fit(self._grad_sum, torch.float32)
        fit(self._grad_count, torch.float32)
        fit(self._ssim_sum, torch.float32)
        fit(self._contribution, torch.float32)
        fit(self._exempt, torch.bool)

    def _single_layer(self, gaussians_nodes: TypedModuleDict[BaseGaussianModel]) -> Optional[str]:
        layers = [layer_id for layer_id, node in gaussians_nodes.items() if node.get_num_gaussians() > 0]
        if len(layers) != 1 or layers[0] not in self.config.ecad.layer_ids:
            raise NotImplementedError(
                f"ECAD currently needs exactly one non-empty Gaussian layer, listed in ecad.layer_ids; got {layers}"
            )
        return layers[0]

    @torch.no_grad()
    def _accumulate(self, gaussians_nodes: TypedModuleDict[BaseGaussianModel]) -> None:
        """Fold this step's camera render into the per-Gaussian statistics."""
        state = ECAD_RENDER_STATE
        ssim_probe, weight_probe = state.ssim_probe, state.weight_probe
        radii, conics, opacities, origin = state.radii, state.conics, state.opacities, state.ray_origin
        state.clear()
        if ssim_probe is None or radii is None:
            return
        layer_id = self._single_layer(gaussians_nodes)
        node = gaussians_nodes[layer_id]
        num = node.get_num_gaussians()
        if radii.numel() != num:
            self._warn_once("rows", f"rendered rows {radii.numel()} != layer rows {num}; skipping ECAD statistics")
            return
        device = node.positions.device
        self._stat_rows(layer_id, num, device)
        visible = radii > 0

        # Gradient term: distance-scaled 3D position gradient (substitute for the screen-space absgrad).
        grad = node.positions.grad
        if grad is not None and origin is not None:
            distance = torch.linalg.vector_norm(node.positions.detach() - origin, dim=-1)
            grad_norm = torch.linalg.vector_norm(grad, dim=-1) * distance
            seen = visible & (grad_norm > 0)
            self._grad_sum[layer_id] += torch.where(seen, grad_norm, 0.0)
            self._grad_count[layer_id] += seen.float()
        else:
            self._warn_once("grad", "positions.grad is unavailable at batch end; gradient term stays empty")

        # SSIM term: per-pixel 1 - SSIM shared out by blending weight.
        if ssim_probe.grad is not None:
            self._ssim_sum[layer_id] += ssim_probe.grad.float().clamp_min(0.0)

        # Contribution: estimated maximum blending weight, max over views since the previous relocation.
        if weight_probe.grad is not None and conics is not None and opacities is not None:
            weight_sum = weight_probe.grad.float().clamp_min(0.0)
            det = (conics[:, 0] * conics[:, 2] - conics[:, 1] * conics[:, 1]).clamp_min(1e-20)
            integral = 2.0 * math.pi / torch.sqrt(det)
            cut2 = (2.0 * torch.log((opacities / _ALPHA_THRESHOLD).clamp_min(1.0))).clamp(max=_GAUSSIAN_EXTEND**2)
            inside = 1.0 - torch.exp(-0.5 * cut2)
            estimate = weight_sum / (integral * inside).clamp_min(1e-12)
            estimate = torch.where(visible & (inside > 0), estimate, 0.0).clamp(max=1.0)
            torch.maximum(self._contribution[layer_id], estimate, out=self._contribution[layer_id])
        else:
            self._warn_once("weights", "no blending-weight probe or conics; contribution stays zero")

    # --------------------------------------------------------------- sampling

    def _source_weights(
        self, layer_id: str, node: BaseGaussianModel, step: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gradient-and-confidence and SSIM source weights over the layer's current rows."""
        count = self._grad_count[layer_id]
        mean_grad = torch.where(count > 0, self._grad_sum[layer_id] / count.clamp_min(1.0), 0.0)
        warmup = self.config.ecad.confidence_warmup_steps
        eta = 1.0 if warmup == 0 else min(step / warmup, 1.0)
        confidence_rank = _rank01(node.camera_extra_signal[:, 0].detach().float())
        weight_grad = _rank01(mean_grad) * ((1.0 - eta) + eta * (1.2 - confidence_rank))
        weight_grad = torch.where(count > 0, weight_grad, 0.0)
        weight_ssim = self._ssim_sum[layer_id].clone()
        return weight_grad, weight_ssim

    def _draw_sources(self, weight_grad: torch.Tensor, weight_ssim: torch.Tensor, count: int) -> torch.Tensor:
        generator = get_cuda_generator(weight_grad.device)
        has_grad, has_ssim = bool(weight_grad.sum() > 0), bool(weight_ssim.sum() > 0)
        if not has_grad and not has_ssim:
            return torch.empty(0, dtype=torch.int64, device=weight_grad.device)
        n_grad = round(self.config.ecad.gradient_share * count) if has_grad and has_ssim else (count if has_grad else 0)
        parts = []
        if n_grad > 0:
            parts.append(torch.multinomial(weight_grad, n_grad, replacement=True, generator=generator))
        if count - n_grad > 0:
            parts.append(torch.multinomial(weight_ssim, count - n_grad, replacement=True, generator=generator))
        return torch.cat(parts)

    def _plan(
        self,
        operation: MCMCOperation,
        owners,
        keep: torch.Tensor,
        sources: torch.Tensor,
    ) -> MCMCRowPlan:
        """MCMC row plan for the given sources, with MCMC's split correction (Eq. 9) by occurrence count."""
        from gsplat.relocation import compute_relocation

        opacity_logits = torch.cat([owner.scene.splats["opacities"].detach().reshape(-1) for owner in owners])
        log_scales = torch.cat([owner.scene.splats["scales"].detach() for owner in owners])
        occurrences = torch.bincount(sources, minlength=opacity_logits.numel())[sources]
        ratios = (occurrences + 1).clamp(max=self.binoms.shape[0]).to(torch.int32)
        scales = torch.exp(log_scales[sources]).clamp_min(torch.finfo(torch.float32).tiny)
        revised_opacities, revised_scales = compute_relocation(
            torch.sigmoid(opacity_logits[sources]).float().contiguous(),
            scales.float().contiguous(),
            ratios,
            self.binoms,
            self.config.opacity_threshold,
        )
        return MCMCRowPlan(
            operation=operation,
            keep=keep,
            source_indices=sources,
            revised_opacities=revised_opacities,
            revised_scales=revised_scales,
            storage_min_opacity=float(self.config.opacity_threshold),
        )

    @torch.no_grad()
    def _apply_row_plan(
        self,
        gaussians_nodes: TypedModuleDict[BaseGaussianModel],
        layer_id: str,
        operation: MCMCOperation,
        keep: torch.Tensor,
        sources: torch.Tensor,
    ) -> int:
        """Apply one plan; returns the number of retained rows (new rows follow them)."""
        selected = SelectedGaussianNodes(gaussians_nodes, [k for k in gaussians_nodes.keys() if k != layer_id])
        invisible_steps = self._get_gsplat_invisible_steps(selected)
        owners, snapshots = _build_owners(selected, invisible_steps)
        plan = self._plan(operation, owners, keep, sources)
        _apply_and_publish(selected, owners, snapshots, plan, invisible_steps)
        retained = int(keep.sum().item())
        node = gaussians_nodes[layer_id]
        if node.get_num_gaussians() != retained + sources.numel():
            raise RuntimeError(
                f"ECAD: {layer_id} has {node.get_num_gaussians()} rows after the plan, "
                f"expected {retained} + {sources.numel()}"
            )
        if self.config.ecad.sample_positions and sources.numel() > 0:
            rows = slice(retained, retained + sources.numel())
            rotation = _quat_to_rotmat(node.rotations[rows].detach().float())
            scale = torch.exp(node.scales[rows].detach().float())
            noise = torch.randn(scale.shape, device=scale.device, generator=get_cuda_generator(scale.device))
            offset = torch.einsum("kij,kj->ki", rotation, scale * noise)
            node.positions[rows] += offset.to(node.positions.dtype)
        return retained

    # ------------------------------------------------------------- topology

    def _relocation_count(self, num: int, step: int, contribution: torch.Tensor, exempt: torch.Tensor) -> torch.Tensor:
        """Boolean mask of the rows to relocate at `step`."""
        e = self.config.ecad
        eligible = ~exempt
        dead = torch.zeros(num, dtype=torch.bool, device=contribution.device)
        if step < e.relocation_dead_only_until:
            candidates = torch.nonzero(eligible & (contribution <= 0), as_tuple=False).squeeze(-1)
            limit = int(e.relocation_share * num)
            if candidates.numel() > limit:
                pick = torch.randperm(candidates.numel(), device=candidates.device)[:limit]
                candidates = candidates[pick]
            dead[candidates] = True
            return dead
        share = e.relocation_share
        if step >= e.relocation_decay_start:
            share *= math.exp(-0.5 * ((step - e.relocation_decay_start) / e.relocation_decay_sigma) ** 2)
        n = min(int(share * num), int(eligible.sum().item()))
        if n <= 0:
            return dead
        scored = torch.where(eligible, contribution, torch.full_like(contribution, float("inf")))
        dead[torch.topk(scored, n, largest=False).indices] = True
        return dead

    @ScopedTimer("ECADStrategy/densify")
    @torch.no_grad()
    def _densify(
        self, gaussians_nodes: TypedModuleDict[BaseGaussianModel], step: int, relocate: bool, add: bool
    ) -> dict[str, float]:
        layer_id = self._single_layer(gaussians_nodes)
        node = gaussians_nodes[layer_id]
        num = node.get_num_gaussians()
        device = node.positions.device
        self._stat_rows(layer_id, num, device)
        weight_grad, weight_ssim = self._source_weights(layer_id, node, step)
        stats: dict[str, float] = {}

        if relocate:
            dead = self._relocation_count(num, step, self._contribution[layer_id], self._exempt[layer_id])
            n_dead = int(dead.sum().item())
            if 0 < n_dead < num:
                sources = self._draw_sources(
                    torch.where(dead, 0.0, weight_grad), torch.where(dead, 0.0, weight_ssim), n_dead
                )
                if sources.numel() == n_dead:
                    keep = ~dead
                    retained = self._apply_row_plan(gaussians_nodes, layer_id, MCMCOperation.RELOCATION, keep, sources)
                    weight_grad = torch.cat([weight_grad[keep], torch.zeros(n_dead, device=device)])
                    weight_ssim = torch.cat([weight_ssim[keep], torch.zeros(n_dead, device=device)])
                    exempt = torch.zeros(retained + n_dead, dtype=torch.bool, device=device)
                    exempt[retained:] = True
                    self._exempt[layer_id] = exempt
                    stats["relocated"] = float(n_dead)
                    num = retained + n_dead
                else:
                    self._warn_once("sources", "no source weights yet; relocation skipped")
            self._contribution[layer_id] = torch.zeros(num, device=device)

        if add:
            target = min(self.config.add.max_n_gaussians, int(1.05 * num))
            n_add = max(0, target - num)
            if n_add > 0:
                sources = self._draw_sources(weight_grad, weight_ssim, n_add)
                if sources.numel() == n_add:
                    keep = torch.ones(num, dtype=torch.bool, device=device)
                    self._apply_row_plan(gaussians_nodes, layer_id, MCMCOperation.ADDITION, keep, sources)
                    stats["added"] = float(n_add)
                    num += n_add
                else:
                    self._warn_once("sources", "no source weights yet; growth skipped")

        # New gradient / SSIM windows start after every densification step.
        self._grad_sum[layer_id] = torch.zeros(num, device=device)
        self._grad_count[layer_id] = torch.zeros(num, device=device)
        self._ssim_sum[layer_id] = torch.zeros(num, device=device)
        self._stat_rows(layer_id, num, device)
        return stats

    # -------------------------------------------------------------- per step

    def _update_step_train_batch_end_impl(
        self,
        epoch: int,
        global_step: int,
        batch: DataAndRenderingBatch,
        system,
        gaussians_nodes: TypedModuleDict[BaseGaussianModel],
        **kwargs,
    ) -> None:
        for gaussian_model in gaussians_nodes.values():
            gaussian_model.raise_if_mcmc_publication_failed()

        if self._fresh_start and not self._confidence_initialized and global_step <= 1:
            self.maybe_initialize_buffers(gaussians_nodes)

        self._accumulate(gaussians_nodes)

        relocate = self._is_topology_mutation_due(global_step, self.config.relocate)
        add = self._is_topology_mutation_due(global_step, self.config.add)
        if relocate or add:
            stats = self._densify(gaussians_nodes, global_step, relocate, add)
            if stats and self._empty_cache_after_parameter_update:
                torch.cuda.empty_cache()
            for key, value in stats.items():
                self._last_log[key] = self._last_log.get(key, 0.0) + value

        if self.config.ecad.keep_mcmc_noise and self._check_step_condition(
            step=global_step,
            start=self.config.perturb.start_iteration,
            end=self.config.perturb.end_iteration,
            freq=self.config.perturb.frequency,
        ):
            for layer_id, gaussian_model in gaussians_nodes.items():
                if layer_id not in self.config.exclude_layer_ids:
                    self.perturb_gaussians(layer_id, gaussian_model)

        for layer_id, max_offset_m in self.config.perturb.max_position_offset_m.items():
            if layer_id not in self.config.exclude_layer_ids and layer_id in gaussians_nodes:
                self._clamp_position_offset(gaussians_nodes[layer_id], max_offset_m)

        if self._is_max_scale_guard_due(global_step):
            self.enforce_max_scale_guard(gaussians_nodes)

        log_every = self.config.ecad.log_every
        if log_every and global_step % log_every == 0:
            self._log(gaussians_nodes, global_step)

    @torch.no_grad()
    def _log(self, gaussians_nodes: TypedModuleDict[BaseGaussianModel], step: int) -> None:
        layer_id = self._single_layer(gaussians_nodes)
        node = gaussians_nodes[layer_id]
        gamma = node.camera_extra_signal[:, 0].float()
        opacity = node.get_densities().view(-1).float()
        contribution = self._contribution.get(layer_id)
        zero_share = float((contribution <= 0).float().mean()) if contribution is not None else float("nan")
        contribution_text = ""
        if contribution is not None and bool((contribution > 0).any()):
            seen = contribution[contribution > 0]
            seen = seen[torch.randperm(seen.numel(), device=seen.device)[:1_000_000]]
            contribution_text = (
                f"; contribution of seen median={float(seen.median()):.3f} p90={float(seen.quantile(0.9)):.3f} "
                f"at clamp {float((seen >= 0.999).float().mean()):.1%}"
            )
        tqdm.write(
            f"ECAD step {step}: {layer_id} n={node.get_num_gaussians()} "
            f"relocated={int(self._last_log.get('relocated', 0))} added={int(self._last_log.get('added', 0))} "
            f"since last log; confidence median={float(torch.exp(gamma.median())):.3f} "
            f"p10={float(torch.exp(gamma.quantile(0.1) if gamma.numel() < 16_000_000 else gamma.median())):.3f}; "
            f"opacity<1/255 {float((opacity < _ALPHA_THRESHOLD).float().mean()):.1%}; "
            f"zero-contribution share {zero_share:.1%}{contribution_text}"
        )
        self._last_log.clear()


BaseGaussianStrategy.register_to_gaussian_strategy_factory("ecad", ECADStrategy)
