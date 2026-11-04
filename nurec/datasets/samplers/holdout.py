# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import json
import logging

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import torch

from omegaconf import DictConfig


if TYPE_CHECKING:
    from nurec.datasets.ncore import NCORETrainDataset
from nurec.datasets.samplers.base import (
    BaseFrameSampler,
    BaseSensorSampler,
    FrameSamplerReturn,
    SensorSamplerReturn,
)


def compute_holdout_training_frame_mask(
    n_frames: int,
    frame_start: int | None,
    frame_step: int | None,
    exclude_frame_start: int | None,
    exclude_frame_step: int | None,
) -> npt.NDArray[np.bool_]:
    """Return a boolean mask of length *n_frames* where True = training frame,
    matching the train/val partition that ``HoldOutFrameSampler.sample_frame``
    samples from.

    Two mutually exclusive holdout modes are supported:

    1. **val-frame-step** (``frame_step`` is not None and > 1):
       Validation frames are at indices ``[frame_start :: frame_step]``.
       Training frames are the complement.

    2. **val-exclude-frame-step** (``exclude_frame_step`` is not None):
       Training frames are at indices ``[exclude_frame_start :: exclude_frame_step]``.
       All other frames are validation.

    If neither mode applies, every frame is treated as a training frame.
    """
    mask = np.ones(n_frames, dtype=np.bool_)
    if frame_step is not None and frame_step > 1:
        # val frames are at [start::step]  →  training = complement
        start = frame_start if frame_start is not None else 0
        mask[start::frame_step] = False
    elif exclude_frame_step is not None:
        # val frames EXCLUDE [start::step]  →  training = [start::step]
        mask[:] = False
        start = exclude_frame_start if exclude_frame_start is not None else 0
        mask[start::exclude_frame_step] = True
    # else: no holdout → all frames are training
    return mask


class HoldOutFrameSampler(BaseFrameSampler):
    """Implements a hold-out sampling of sensor frames for training (every N frames will be excluded)"""

    include_frames_start: int | None
    exclude_frames_start: int | None
    include_every_n_frames: int | None
    exclude_every_n_frames: int | None

    def __init__(self, config: DictConfig, dataset: NCORETrainDataset):
        super().__init__(config, dataset)

        # Initialization to work even if one of these keys are missing from the config (backward compatibility).
        # It is not allowed to use HoldOutFrameSampler if both keys are missing from the config.
        self.include_every_n_frames = config.get("include_every_n_frames", None)
        self.exclude_every_n_frames = config.get("exclude_every_n_frames", None)

        # Initialization to work even if the keys are missing from the config (backward compatibility).
        # include_frames_start was introduced in the config after include_every_n_frames.
        if "include_frames_start" in config:
            self.include_frames_start = config.include_frames_start
        elif self.include_every_n_frames is None:
            self.include_frames_start = None
        else:
            self.include_frames_start = 0

        # Initialization to work even if the keys are missing from the config (backward compatibility).
        # include_frames_start was introduced in the config after include_every_n_frames.
        if "exclude_frames_start" in config:
            self.exclude_frames_start = config.exclude_frames_start
        elif self.exclude_every_n_frames is None:
            self.exclude_frames_start = None
        else:
            self.exclude_frames_start = 0

        assert (
            # Specifying frames to exclude
            self.include_frames_start is None
            and self.include_every_n_frames is None
            and self.exclude_frames_start is not None
            and self.exclude_every_n_frames is not None
        ) or (
            self.include_frames_start is not None
            and self.include_every_n_frames is not None
            and self.exclude_frames_start is None
            and self.exclude_every_n_frames is None
        ), (
            f"{self.__class__.__name__} Either specify "
            "frames to include (via include_frames_start and include_every_n_frames) or "
            "frames to exclude (via exclude_frames_start and exclude_every_n_frames) but not both"
        )

    @classmethod
    def training_frame_mask(cls, config: DictConfig, n_frames: int) -> npt.NDArray[np.bool_]:
        """Mirror of ``sample_frame``'s probability vector: the frames this
        sampler can actually return for training. This — not the validation
        holdout configuration — is what the all-frame -> training-frame
        mapping must be derived from."""
        include_every_n_frames = config.get("include_every_n_frames", None)
        exclude_every_n_frames = config.get("exclude_every_n_frames", None)
        if include_every_n_frames is not None:
            include_frames_start = config.get("include_frames_start", 0) or 0
            mask = np.zeros(n_frames, dtype=np.bool_)
            mask[include_frames_start::include_every_n_frames] = True
            # An include-start beyond this sensor's range selects nothing.
            # sample_frame would then divide by p.sum() == 0, so fail fast
            # here — before the frame mapping is silently built empty.
            if not mask.any():
                raise IndexError(
                    "HoldOutFrameSampler include-mode selects no training frame for a frame "
                    f"range of length {n_frames} (include_frames_start={include_frames_start})"
                )
            return mask
        if exclude_every_n_frames is not None:
            exclude_frames_start = config.get("exclude_frames_start", 0) or 0
            mask = np.ones(n_frames, dtype=np.bool_)
            mask[exclude_frames_start::exclude_every_n_frames] = False
            return mask
        raise ValueError("HoldOutFrameSampler config must set include_every_n_frames or exclude_every_n_frames")

    def sample_frame(
        self,
        rng: np.random.Generator,
        batch_idx: int,
        frame_range: range,
        unique_sensor_id: str,
    ) -> FrameSamplerReturn:
        """Implementation of 'FrameSampler' protocol sampling from frame_range except every Nth frame"""

        if self.include_every_n_frames is not None:
            assert self.include_frames_start is not None
            p = np.zeros((len(frame_range),))
            p[self.include_frames_start :: self.include_every_n_frames] = 1
        else:
            assert self.exclude_frames_start is not None and self.exclude_every_n_frames is not None
            p = np.ones((len(frame_range),))
            p[self.exclude_frames_start :: self.exclude_every_n_frames] = 0

        p = p / p.sum()
        return FrameSamplerReturn(
            sampled_frame_idx=rng.choice(frame_range, size=1, replace=False, shuffle=False, p=p).item()
        )


_COVERAGE_BASE_SEED = 0

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=8)
def _load_include_frames_file(path: str) -> dict[str, tuple[int, ...]]:
    """Per-sensor training-frame lists for :class:`CoverageFrameSampler` (``include_frames_file``): a JSON object
    mapping each sensor id (e.g. ``camera_front``) to the LOCAL frame indices (positions in that sensor's
    ``frame_range``) it may train on. Used to give each spatial cell of a multi-cell reconstruction its own
    per-camera image set (VastGaussian-style visibility-based camera selection)."""
    with open(path) as f:
        raw = json.load(f)
    return {str(k): tuple(int(i) for i in v) for k, v in raw.items()}


@functools.lru_cache(maxsize=16)
def _coverage_permutation(base_seed: int, total: int, epoch: int) -> npt.NDArray[np.int64]:
    """Deterministic per-pass permutation of ``range(total)``.

    The ``(base_seed, epoch)`` pair is handed to ``default_rng`` as entropy; numpy's ``SeedSequence``
    mixes it, so successive passes get decorrelated permutations without any hand-rolled seed spreading.
    Cached so the sensor and frame halves of a step -- and the camera and lidar plans -- reuse one array
    per ``(seed, size, pass)`` instead of re-permuting on every ``sample_*`` call. The returned array is
    read-only by contract.
    """
    perm = np.random.default_rng([base_seed, epoch]).permutation(total)
    perm.setflags(write=False)
    return perm


@dataclass(frozen=True, eq=False)
class CoverageDecode:
    """A single scheduled work item: which sensor to sample and which frame to draw from it."""

    sensor_id: str  # sensor key, as returned by the sensor sampler / used to index frame ranges
    unique_sensor_id: str  # unique sensor id, as passed to the frame sampler (used for the coupling assert)
    frame_idx: int  # frame index within that sensor's ``frame_range``


@dataclass(frozen=True, eq=False)
class CoveragePlanRestriction:
    """Result of restricting a coverage plan to an eligible set of work items."""

    plan: CoveragePlan
    original_total: int
    eligible_total: int

    @property
    def eligible_fraction(self) -> float:
        """Fraction of the original work items retained by :attr:`plan`."""
        return self.eligible_total / self.original_total


@dataclass(frozen=True, eq=False)
class CoveragePlan:
    """Joint ``(sensor, frame)`` coverage schedule, shared by the coverage sensor and frame
    samplers so their choices are coupled by construction.

    A "work item" is one ``(sensor, training-frame)`` pair. The plan enumerates every training frame of
    every ACTIVE sensor exactly once, so a full pass of ``total`` steps touches every ``(sensor, frame)``
    pair exactly once, in a fresh per-pass random order. ``decode(batch_idx)`` is deterministic and
    worker-safe; because BOTH samplers decode the SAME instance, the sensor they pick and the frame they
    draw are guaranteed to agree. A configured batch-index offset lives in shared memory so map-style
    dataloader indices can restart at zero each training epoch without restarting this coverage schedule.
    """

    sensor_ids: tuple[str, ...]
    unique_sensor_ids: tuple[str, ...]  # parallel to sensor_ids
    per_sensor_frame_idxs: tuple[npt.NDArray[np.int64], ...]  # training frame values per sensor (in its frame_range)
    # cumulative work-item counts, length == n_sensors + 1; prefix[-1] == total work items
    prefix: npt.NDArray[np.int64]
    base_seed: int = _COVERAGE_BASE_SEED
    _batch_index_offset: torch.Tensor = field(
        default_factory=lambda: torch.zeros((), dtype=torch.int64),
        repr=False,
        compare=False,
    )

    @property
    def total(self) -> int:
        """Number of work items in one full coverage pass (== steps per pass)."""
        return int(self.prefix[-1])

    def set_batch_index_offset(self, offset: int) -> None:
        """Set the epoch's global index offset, propagating it to persistent dataloader workers."""
        if offset < 0:
            raise ValueError(f"CoveragePlan batch-index offset must be non-negative, got {offset}.")
        if not self._batch_index_offset.is_shared():
            self._batch_index_offset.share_memory_()
        self._batch_index_offset.fill_(offset)

    def decode(self, batch_idx: int) -> CoverageDecode:
        """Map an epoch-local ``batch_idx`` to the ``(sensor, frame)`` work item for that global step."""
        total = self.total
        assert total > 0, "CoveragePlan: no work items (no active sensor has a training frame)."
        global_batch_idx = int(self._batch_index_offset.item()) + int(batch_idx)
        epoch, pos = divmod(global_batch_idx, total)
        item = int(_coverage_permutation(self.base_seed, total, epoch)[pos])
        sensor = int(np.searchsorted(self.prefix, item, side="right")) - 1
        offset = item - int(self.prefix[sensor])
        return CoverageDecode(
            sensor_id=self.sensor_ids[sensor],
            unique_sensor_id=self.unique_sensor_ids[sensor],
            frame_idx=int(self.per_sensor_frame_idxs[sensor][offset]),
        )

    def iter_pairs(self) -> Iterator[tuple[str, int]]:
        """Yield every scheduled ``(sensor_id, frame_idx)`` pair in stable plan order."""
        for sensor_id, frame_idxs in zip(self.sensor_ids, self.per_sensor_frame_idxs):
            for frame_idx in frame_idxs:
                yield sensor_id, int(frame_idx)

    def restrict_to_pairs(self, eligible_pairs: Iterable[tuple[str, int]]) -> CoveragePlanRestriction:
        """Return a plan containing exactly *eligible_pairs*.

        Work items retain their original stable order, and the resulting coverage permutation remains
        uniform over eligible ``(sensor, frame)`` pairs. Sensors with no eligible frames remain represented
        by an empty frame array; only a globally empty eligible set is rejected.
        """
        original_total = self.total
        if original_total == 0:
            raise ValueError("Cannot restrict an empty CoveragePlan.")

        available_pairs = set(self.iter_pairs())
        eligible_set = {(sensor_id, int(frame_idx)) for sensor_id, frame_idx in eligible_pairs}
        unknown_pairs = eligible_set - available_pairs
        if unknown_pairs:
            raise ValueError(f"Eligible pairs are not present in the CoveragePlan: {sorted(unknown_pairs)!r}")
        if not eligible_set:
            raise ValueError("CoveragePlan restriction must retain at least one eligible pair.")

        if len(eligible_set) == original_total:
            restricted_plan = self
        else:
            restricted_frame_idxs = tuple(
                frame_idxs[
                    np.fromiter(
                        ((sensor_id, int(frame_idx)) in eligible_set for frame_idx in frame_idxs),
                        dtype=np.bool_,
                        count=frame_idxs.shape[0],
                    )
                ]
                for sensor_id, frame_idxs in zip(self.sensor_ids, self.per_sensor_frame_idxs)
            )
            counts = [int(frame_idxs.shape[0]) for frame_idxs in restricted_frame_idxs]
            prefix = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64)])
            restricted_plan = CoveragePlan(
                sensor_ids=self.sensor_ids,
                unique_sensor_ids=self.unique_sensor_ids,
                per_sensor_frame_idxs=restricted_frame_idxs,
                prefix=prefix,
                base_seed=self.base_seed,
            )

        return CoveragePlanRestriction(
            plan=restricted_plan,
            original_total=original_total,
            eligible_total=len(eligible_set),
        )

    @classmethod
    def single_sensor(cls, unique_sensor_id: str, frame_idxs: npt.NDArray[np.int64]) -> CoveragePlan:
        """A trivial one-sensor plan -- the single-sensor fall-back used when no shared plan is injected."""
        return cls(
            sensor_ids=(unique_sensor_id,),
            unique_sensor_ids=(unique_sensor_id,),
            per_sensor_frame_idxs=(frame_idxs,),
            prefix=np.array([0, int(frame_idxs.shape[0])], dtype=np.int64),
        )

    @classmethod
    def build(cls, dataset: NCORETrainDataset, sensor_kind: str, frame_sampler_config: DictConfig) -> CoveragePlan:
        """Build the joint coverage plan over the ACTIVE sensors of *dataset*.

        The per-sensor training subset comes from :meth:`CoverageFrameSampler.training_frame_mask`
        applied to *frame_sampler_config* -- the SAME reachable-frame definition used to size the PPISP /
        bilateral-grid per-frame parameters -- so the schedule covers exactly the frames those parameters
        exist for.
        """
        assert sensor_kind in ("camera", "lidar"), f"CoveragePlan: unknown sensor_kind {sensor_kind!r}"
        datasource = dataset.datasource
        if sensor_kind == "camera":
            active_ids = dataset.camera_ids
            frame_ranges = datasource.camera_frame_ranges
            unique_ids = datasource.camera_unique_ids
        else:
            active_ids = dataset.lidar_ids
            frame_ranges = datasource.lidar_frame_ranges
            unique_ids = datasource.lidar_unique_ids

        sensor_ids: list[str] = []
        unique_sensor_ids: list[str] = []
        per_sensor_frame_idxs: list[npt.NDArray[np.int64]] = []
        for sensor_id in active_ids:
            frame_range = frame_ranges[sensor_id]
            mask = CoverageFrameSampler.training_frame_mask(
                frame_sampler_config, n_frames=len(frame_range), sensor_id=sensor_id
            )
            frame_idxs = np.asarray(frame_range, dtype=np.int64)[mask]
            assert frame_idxs.shape[0] > 0, (
                f"CoveragePlan: {sensor_kind} sensor '{sensor_id}' has no training frames over "
                f"{len(frame_range)} frames; check the hold-out config."
            )
            sensor_ids.append(sensor_id)
            unique_sensor_ids.append(unique_ids[sensor_id].id)
            per_sensor_frame_idxs.append(frame_idxs)

        counts = [int(v.shape[0]) for v in per_sensor_frame_idxs]
        if frame_sampler_config.get("include_frames_file", None):
            logger.info(
                "CoveragePlan (%s, include_frames_file=%s): training frames per sensor %s",
                sensor_kind,
                frame_sampler_config.get("include_frames_file"),
                dict(zip(sensor_ids, counts)),
            )
        prefix = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64)])
        return cls(
            sensor_ids=tuple(sensor_ids),
            unique_sensor_ids=tuple(unique_sensor_ids),
            per_sensor_frame_idxs=tuple(per_sensor_frame_idxs),
            prefix=prefix,
        )


class CoverageFrameSampler(BaseFrameSampler):
    # training_frame_mask takes the sensor id (for include_frames_file); see BaseFrameSampler.group_training_frame_mask
    TRAINING_FRAME_MASK_PER_SENSOR = True

    """Coverage-guaranteeing frame sampler, with an OPTIONAL hold-out split.

    The uniform / timestamp samplers draw one frame per step i.i.d. (with replacement across steps), so
    they give no guarantee that every frame is touched early -- the latest first-sampled frame is
    ~N*ln(N) steps in, which can leave a per-frame parameter (e.g. PPISP exposure) unanchored for
    thousands of steps and free to drift. This sampler instead cycles through a fresh random permutation
    of its eligible frames each pass, so every frame is sampled exactly once per pass and no frame can
    starve. Deterministic and worker-safe: the sampled frame is a pure function of ``batch_idx``.

    Hold-out is OPTIONAL (unlike :class:`HoldOutFrameSampler`, which requires it):

    * with no include/exclude params, cover ALL frames (coverage, no train/val split);
    * with ``include_every_n_frames`` / ``exclude_every_n_frames``, cover only the training
      subset (coverage + hold-out), using the same partition semantics as :class:`HoldOutFrameSampler`;
    * with ``include_frames_file`` (JSON: sensor id -> local frame indices), additionally restrict each sensor
      to its listed frames -- a per-sensor image set, e.g. for VastGaussian-style cells. Every consumer of
      :meth:`training_frame_mask` must then pass the sensor id.

    MULTI-SENSOR coverage. The per-sensor "once per pass" property only holds if this sampler is invoked
    on (almost) every step for its sensor. With a random sensor sampler that is NOT true: each sensor is
    only drawn on a random subset of steps, so ``batch_idx`` does not sweep its frames and coverage
    degrades to i.i.d.. To fix this, the batch sampler pairs the coverage frame sampler with the coverage
    SENSOR sampler (:class:`CoverageSensorSampler`) and injects a shared :class:`CoveragePlan`: the sensor
    sampler returns the sensor the plan scheduled and this sampler returns that sensor's scheduled frame,
    so every ``(sensor, frame)`` pair is covered once per pass across ALL active sensors. When no plan is
    injected (single-sensor training, or direct unit use) it falls back to a one-sensor coverage cycle,
    which is correct precisely because the single sensor is drawn every step.
    """

    def __init__(self, config: DictConfig, dataset: NCORETrainDataset):
        super().__init__(config, dataset)
        # Hold-out params are all optional here (coverage may run over all frames); validate the pair now.
        self._validate_holdout_config(
            config.get("include_every_n_frames", None), config.get("exclude_every_n_frames", None)
        )
        # Kept for the single-sensor fall-back (rebuilds the reachable set via training_frame_mask).
        self._holdout_config = config
        # Shared joint plan, injected by the batch sampler for multi-sensor coverage (None => fall-back).
        self._coverage_plan: CoveragePlan | None = None

    def set_coverage_plan(self, plan: CoveragePlan) -> None:
        """Inject the shared joint ``(sensor, frame)`` plan (called by the batch sampler)."""
        self._coverage_plan = plan

    @property
    def coverage_plan(self) -> CoveragePlan:
        """Return the authoritative shared plan, failing loudly if it was not injected."""
        if self._coverage_plan is None:
            raise RuntimeError(
                "CoverageFrameSampler used without an injected CoveragePlan -- the batch sampler must wire the "
                "shared plan into the paired coverage samplers."
            )
        return self._coverage_plan

    def build_and_inject_coverage_plan(
        self, dataset: NCORETrainDataset, sensor_kind: str, sensor_sampler: BaseSensorSampler
    ) -> None:
        """Build the joint ``(sensor, frame)`` plan over *dataset*'s active sensors (using this sampler's
        own hold-out config) and inject the SAME instance into this frame sampler and the paired sensor
        sampler, so their choices are coupled. Called by ``BaseBatchSampler._wire_coverage_plan``; kept in
        this module so ``base`` never imports :class:`CoveragePlan`."""
        plan = CoveragePlan.build(dataset, sensor_kind, self._holdout_config)
        self.set_coverage_plan(plan)
        sensor_sampler.set_coverage_plan(plan)

    @staticmethod
    def _validate_holdout_config(include_every_n_frames: int | None, exclude_every_n_frames: int | None) -> None:
        """Validate the (optional) hold-out config -- shared by the constructor and the
        ``training_frame_mask`` classmethod so BOTH paths reject the same misconfigurations:

        * ``include_*`` and ``exclude_*`` are mutually exclusive (else the mask silently uses include-only);
        * strides must be positive -- a 0 stride raises a cryptic ValueError in the slice, and a
          NEGATIVE stride is a valid reverse slice that silently selects the wrong subset.

        No-op when a stride is unset (coverage over all frames)."""
        assert not (include_every_n_frames is not None and exclude_every_n_frames is not None), (
            "CoverageFrameSampler: specify hold-out via include_* OR exclude_* params, not both."
        )
        for name, stride in (
            ("include_every_n_frames", include_every_n_frames),
            ("exclude_every_n_frames", exclude_every_n_frames),
        ):
            assert stride is None or stride >= 1, (
                f"CoverageFrameSampler: {name} must be a positive integer stride, got {stride}"
            )

    @classmethod
    def training_frame_mask(
        cls, config: DictConfig, n_frames: int, sensor_id: str | None = None
    ) -> npt.NDArray[np.bool_]:
        """Frames this sampler can return for training (mirror of ``sample_frame``): the hold-out
        training subset if configured, else every frame (coverage over all frames); with
        ``include_frames_file``, further restricted to *sensor_id*'s listed frames."""
        mask = cls._holdout_training_frame_mask(config, n_frames)
        include_file = config.get("include_frames_file", None)
        if include_file:
            if sensor_id is None:
                raise ValueError(
                    "CoverageFrameSampler: include_frames_file is set, so training_frame_mask needs the sensor id"
                )
            lists = _load_include_frames_file(str(include_file))
            if sensor_id not in lists:
                raise KeyError(f"CoverageFrameSampler: sensor '{sensor_id}' is missing from {include_file}")
            idx = np.asarray(lists[sensor_id], dtype=np.int64)
            if idx.size and (idx.min() < 0 or idx.max() >= n_frames):
                raise IndexError(
                    f"CoverageFrameSampler: {include_file} lists frames outside sensor '{sensor_id}''s range of "
                    f"{n_frames} frames"
                )
            keep = np.zeros(n_frames, dtype=np.bool_)
            keep[idx] = True
            mask = mask & keep
            if not mask.any():
                raise IndexError(
                    f"CoverageFrameSampler: no training frame left for sensor '{sensor_id}' after {include_file}"
                )
        return mask

    @classmethod
    def _holdout_training_frame_mask(cls, config: DictConfig, n_frames: int) -> npt.NDArray[np.bool_]:
        """The hold-out part of :meth:`training_frame_mask`."""
        include_every_n_frames = config.get("include_every_n_frames", None)
        exclude_every_n_frames = config.get("exclude_every_n_frames", None)
        cls._validate_holdout_config(include_every_n_frames, exclude_every_n_frames)
        if include_every_n_frames is not None:
            include_frames_start = config.get("include_frames_start", 0) or 0
            mask = np.zeros(n_frames, dtype=np.bool_)
            mask[include_frames_start::include_every_n_frames] = True
            if not mask.any():
                raise IndexError(
                    "CoverageFrameSampler include-mode selects no training frame for a frame range of "
                    f"length {n_frames} (include_frames_start={include_frames_start})"
                )
            return mask
        if exclude_every_n_frames is not None:
            exclude_frames_start = config.get("exclude_frames_start", 0) or 0
            mask = np.ones(n_frames, dtype=np.bool_)
            mask[exclude_frames_start::exclude_every_n_frames] = False
            # Symmetric with include-mode: a degenerate exclude (e.g. exclude_every_n_frames=1) can strip
            # every training frame. Fail fast here -- the single choke point all consumers pass through --
            # rather than letting mapping setup absorb the all-empty mask and only failing later.
            if not mask.any():
                raise IndexError(
                    "CoverageFrameSampler exclude-mode selects no training frame for a frame range of "
                    f"length {n_frames} (exclude_frames_start={exclude_frames_start}, "
                    f"exclude_every_n_frames={exclude_every_n_frames})"
                )
            return mask
        # No hold-out configured: every frame is a training frame.
        return np.ones(n_frames, dtype=np.bool_)

    def sample_frame(
        self,
        rng: np.random.Generator,
        batch_idx: int,
        frame_range: range,
        unique_sensor_id: str,
    ) -> FrameSamplerReturn:
        del rng  # coverage is deterministic in batch_idx
        if self._coverage_plan is not None:
            decode = self.coverage_plan.decode(batch_idx)
            # The sensor the pipeline chose MUST be the one the shared plan scheduled for this step; if it
            # is not, the sensor sampler is not the coverage sensor sampler and the guarantee is broken.
            # Fail loudly rather than silently mis-covering.
            assert decode.unique_sensor_id == unique_sensor_id, (
                f"CoverageFrameSampler: sensor sampler selected '{unique_sensor_id}', but the coverage plan "
                f"scheduled '{decode.unique_sensor_id}' for step {batch_idx}. Multi-sensor coverage requires "
                "the coverage sensor sampler so the sensor and frame choices stay coupled."
            )
            return FrameSamplerReturn(sampled_frame_idx=decode.frame_idx)

        # No shared plan: single-sensor coverage over just this sensor's frame range.
        if self._holdout_config.get("include_frames_file", None):
            raise NotImplementedError(
                "CoverageFrameSampler: include_frames_file needs the multi-sensor coverage plan (sensor ids)"
            )
        mask = self.training_frame_mask(self._holdout_config, n_frames=len(frame_range))
        frame_idxs = np.asarray(frame_range, dtype=np.int64)[mask]
        assert frame_idxs.shape[0] > 0, (
            f"{type(self).__name__}: no eligible training frames for sensor '{unique_sensor_id}' over "
            f"{len(frame_range)} frames; check the hold-out config."
        )
        decode = CoveragePlan.single_sensor(unique_sensor_id, frame_idxs).decode(batch_idx)
        return FrameSamplerReturn(sampled_frame_idx=decode.frame_idx)


class CoverageSensorSampler(BaseSensorSampler):
    """Sensor half of the coverage schedule.

    Returns the single sensor the shared :class:`CoveragePlan` scheduled for this step, coupling the
    sensor choice to the frame choice so multi-sensor coverage holds (see :class:`CoverageFrameSampler`).
    ``sample_all_sensors`` is irrelevant (exactly one sensor per step by design), which also matches the
    NCORE pipeline's one-sensor-per-batch requirement.
    """

    def __init__(self, config: DictConfig, dataset: NCORETrainDataset):
        super().__init__(config, dataset)
        self._coverage_plan: CoveragePlan | None = None

    def set_coverage_plan(self, plan: CoveragePlan) -> None:
        """Inject the shared joint ``(sensor, frame)`` plan (called by the batch sampler)."""
        self._coverage_plan = plan

    @property
    def coverage_plan(self) -> CoveragePlan:
        assert self._coverage_plan is not None, (
            "CoverageSensorSampler used without an injected CoveragePlan -- the batch sampler must wire the "
            "shared plan into the paired coverage samplers."
        )
        return self._coverage_plan

    def sample_sensor(
        self,
        rng: np.random.Generator,
        batch_idx: int,
        sensor_ids: list[str],
    ) -> SensorSamplerReturn:
        del rng  # coverage is deterministic in batch_idx
        decode = self.coverage_plan.decode(batch_idx)
        assert decode.sensor_id in sensor_ids, (
            f"CoverageSensorSampler: planned sensor '{decode.sensor_id}' is not in the active set "
            f"{list(sensor_ids)}; the coverage plan and dataloader disagree on the active sensors."
        )
        return SensorSamplerReturn(sampled_sensor_ids=[decode.sensor_id])


BaseFrameSampler.register_to_frame_sampler_factory("holdout", HoldOutFrameSampler)
BaseFrameSampler.register_to_frame_sampler_factory("coverage", CoverageFrameSampler)
BaseSensorSampler.register_to_sensor_sampler_factory("coverage", CoverageSensorSampler)
