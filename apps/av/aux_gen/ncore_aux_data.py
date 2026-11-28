# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import multiprocessing
import multiprocessing.synchronize
import os
import queue
import threading
import time
import traceback

from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    cast,
)

import click
import numpy as np
import PIL.Image as PILImage
import torch
import tqdm

from numpy.typing import NDArray
from upath import UPath

import ncore.data
import ncore.impl.data.stores as ncore_stores
import nurec.utils.cli as nurec_utils_cli

from apps.av.aux_gen.depth_estimator import DepthEstimator
from apps.av.aux_gen.device_frame_service import (
    DeviceFrameServiceReady,
    DeviceFrameSharedArray,
    DeviceFrameSharedArrayDescriptor,
    DeviceFrameSharedBuffers,
    SharedDeviceFrameClient,
    run_device_frame_service_process,
)
from apps.av.aux_gen.segmentation_device_pipeline import (
    CAMERA_MASK_INPUT,
    MASK2FORMER_PROCESSOR,
    SAM2_MASK_OUTPUT,
    SAM2_PROCESSOR,
    AuxiliaryProcessorFactory,
    SegmentationDeviceFramePipelineFactory,
)
from apps.av.aux_gen.segmentation_presentation import is_dark_image
from apps.av.aux_gen.shared_segmentation import (
    AuxiliarySegmentationOutputs,
    SegmentationFrameOutput,
    SharedSegmentationFrameProcessor,
    save_auxiliary_mask,
)
from ncore.data import ConcreteCameraModelParametersUnion
from ncore.impl.common.transformations import time_bounds
from nurec.ncore_backends import (
    DatasetBackend,
    DatasetRequest,
    ExplicitSequenceInputProvider,
    ResolvedDataset,
    require_explicit_sequence_input_provider,
    resolve_dataset,
)
from nurec.utils.debug.remote_debug import breakpoint_env
from nurec.utils.files import parse_universal_path
from nurec.utils.ncore_types import require_concrete_camera_parameters
from nurec.utils.ncore_utils import (
    PCA,
    AuxDataCameraSemSegProvider,
    AuxDataWriter,
    AuxShardDataLoader,
    CamAuxDataLoader,
    Feature2ColorTransform,
    get_camera_sensor_mask,
)
from nurec.utils.profiling import ScopedTimer, TimingTag
from nurec.utils.types import HalfClosedInterval


if TYPE_CHECKING:
    from apps.av.aux_gen.estimators import (
        DINOv2Estimator,
        LidarSegmentationByProjectEstimator,
        Mask2FormerSegmentationEstimator,
        PromptlessSAM2EgoMaskEstimator,
    )


# Conditionally activate remote debugging based on environment variables
breakpoint_env()


@dataclass(kw_only=True, slots=True, frozen=True)
class SensorRanges:
    camera_frame_ranges: dict[str, range]
    lidar_frame_ranges: dict[str, range]


SensorRangeCallback = Callable[
    [ncore.data.SequenceLoaderProtocol], SensorRanges
]  # represents frame range for each enabled camera / lidar sensor


@dataclass(kw_only=True, slots=True, frozen=True)
class CLIBaseParams:
    """Parameters passed to non-command-based CLI part"""

    # Optional low-level input supplied by an extending entrypoint.
    explicit_path_pattern: Optional[str]

    # Sequence metadata entry point
    dataset_path: Optional[str]  # path to a NCore meta-file

    # Named component groups considered for data loading
    poses_component_group: str
    intrinsics_component_group: str
    masks_component_group: str
    cuboids_component_group: str

    output_dir: str
    camera_ids: list[str]
    lidar_ids: list[str]
    segmentation_backend: str
    seg_logits: bool
    enable_trt: bool
    dinov2_backend: str
    dinov2_pca_dim: int
    dinov2_width: int
    lidar_seg_camvis: bool
    lidar_seg_ensemble_cuda: bool
    zarr_store_type: str
    open_consolidated: bool
    num_threads: str
    debug: bool
    visualize: bool
    store_meta: bool
    ego_mask: bool
    ego_mask_samples_per_second: float
    ego_mask_aggregation_method: Literal["majority"]
    ego_mask_camera_ids: tuple[str, ...] | None = None
    execution_mode: Literal["serial", "parallel"]
    workers_per_gpu: int

    @property
    def parallel_mode(self) -> bool:
        """Return whether camera and lidar work runs in parallel processes.

        Returns:
            Whether ``execution_mode`` selects shared parallel processing.
        """
        return self.execution_mode == "parallel"


@dataclass(kw_only=True, slots=True, frozen=True)
class CLIState:
    params: CLIBaseParams
    depth_generation: DepthGenerationRequest | None
    auxiliary_segmentation: AuxiliarySegmentationRequest | None
    extension_cli_params: dict[str, Any]
    instance_generation: InstanceGenerationRequest | None = None


@dataclass(kw_only=True, slots=True, frozen=True)
class DepthGenerationRequest:
    """Model-agnostic depth generation configured by an extending entrypoint."""

    create_estimator: Callable[[], DepthEstimator]
    store_as_png: bool
    description: str


DepthGenerationConfigurator = Callable[[dict[str, Any]], DepthGenerationRequest | None]


class InstanceGenerationRequest(Protocol):
    """Optional instance-mask generation owned by an extending entrypoint."""

    def preflight(self) -> None:
        """Validate model inputs before opening the sequence."""
        ...

    def generate(
        self,
        sequence_loader: ncore.data.SequenceLoaderProtocol,
        backend: DatasetBackend,
        writer: AuxDataWriter,
        camera_frame_ranges: Mapping[str, range],
    ) -> None:
        """Write instances for selected cameras, releasing model state before returning."""
        ...


InstanceGenerationConfigurator = Callable[[dict[str, Any]], InstanceGenerationRequest | None]


class AuxiliarySegmentationRequest(Protocol):
    """Model-agnostic auxiliary segmentation configured by an entrypoint.

    Shared segmentation runs one extension-owned device processor beside
    Mask2Former on a sampled subset of camera frames, stores its mask output as
    a PNG per sampled frame, and hands the extension its own per-camera
    counters when the run completes. This module stays independent of which
    model does that work: an extending entrypoint owns option parsing, frame
    selection, processor construction, and result finalization.
    """

    def preflight(self) -> None:
        """Validate extension-owned inputs before any sequence is opened.

        Raises:
            Exception: If the extension's own configuration is invalid or its
                required inputs are missing.
        """
        ...

    def frame_outputs(self) -> AuxiliarySegmentationOutputs:
        """Describe how the extension processor's frame outputs are consumed.

        Returns:
            A picklable description forwarded to every camera worker.
        """
        ...

    def processor_factory(self) -> AuxiliaryProcessorFactory:
        """Return the picklable factory creating the device processor.

        Returns:
            A factory invoked once per GPU service process.
        """
        ...

    def select_camera_frames(
        self,
        sequence_loader: ncore.data.SequenceLoaderProtocol,
        camera_frame_ranges: Sequence[tuple[str, range]],
        /,
    ) -> Mapping[int, tuple[int, ...]]:
        """Select the frames each camera worker submits to the processor.

        Args:
            sequence_loader: Loader for the single sequence being processed.
            camera_frame_ranges: Active ``(camera_id, frame_range)`` pairs,
                positionally indexed by worker identity.

        Returns:
            Worker-local frame indices keyed by worker identity. Workers absent
            from the mapping submit no auxiliary frames.
        """
        ...

    def finalize(
        self,
        *,
        camera_ids: tuple[str, ...],
        camera_statistics: tuple[Mapping[str, Any], ...],
        processor_metadata: Mapping[str, Any],
        wall_time_seconds: float,
    ) -> None:
        """Publish extension-owned results after every camera worker exits.

        Args:
            camera_ids: Cameras processed by this run, in worker order.
            camera_statistics: One record per camera that produced counters.
            processor_metadata: The device processor's storage metadata as
                reported at service readiness.
            wall_time_seconds: Total shared-segmentation wall time.
        """
        ...


AuxiliarySegmentationConfigurator = Callable[[dict[str, Any]], Optional[AuxiliarySegmentationRequest]]


@dataclass(kw_only=True, slots=True, frozen=True)
class ManifestLoaderWorkerPayload:
    """Resolved manifest input passed to a spawned worker."""

    resolved_dataset: ResolvedDataset


@dataclass(kw_only=True, slots=True, frozen=True)
class ExplicitPathLoaderWorkerPayload:
    """Explicit sequence path passed to a spawned worker."""

    path: str
    open_consolidated: bool
    loader_max_workers: int | None
    explicit_input_provider: ExplicitSequenceInputProvider


LoaderWorkerPayload = ManifestLoaderWorkerPayload | ExplicitPathLoaderWorkerPayload


@dataclass(kw_only=True, slots=True, frozen=True)
class OpenedSequence:
    backend: DatasetBackend
    loader: ncore.data.SequenceLoaderProtocol
    worker_payload: LoaderWorkerPayload


def _open_worker_sequence(
    payload: LoaderWorkerPayload,
) -> tuple[DatasetBackend, ncore.data.SequenceLoaderProtocol]:
    if isinstance(payload, ManifestLoaderWorkerPayload):
        resolved = payload.resolved_dataset
        return resolved.backend, resolved.create_sequence_loader()
    return payload.explicit_input_provider.reopen_input(
        UPath(payload.path),
        open_consolidated=payload.open_consolidated,
        loader_max_workers=payload.loader_max_workers,
    )


def _run_ego_mask_for_camera(
    ego_mask: bool,
    ego_mask_camera_ids: Optional[Sequence[str]],
    camera_id: str,
) -> bool:
    """Return whether ego-mask estimation should run for the given camera.

    Used in the per-camera loop to respect --ego-mask and --ego-mask-camera-id:
    when ego_mask_camera_ids is None or empty, all cameras get ego mask (when
    ego_mask is True); otherwise only cameras listed in ego_mask_camera_ids do.

    Args:
        ego_mask: Whether ego-mask is enabled (e.g. from --ego-mask).
        ego_mask_camera_ids: Optional tuple of camera IDs from Click (or any Sequence) to
            restrict ego-mask to. None or empty means "all cameras".
        camera_id: The current camera ID.

    Returns:
        True if ego-mask estimation should run for this camera, False otherwise.
    """
    return bool(ego_mask and (not ego_mask_camera_ids or camera_id in ego_mask_camera_ids))


def _preflight_auxiliary_segmentation(
    params: CLIBaseParams,
    auxiliary_segmentation: AuxiliarySegmentationRequest | None,
) -> None:
    """Check the topology auxiliary segmentation needs, then its own inputs.

    Args:
        params: Validated CLI parameters.
        auxiliary_segmentation: Optional extension-owned request.

    Raises:
        ValueError: If the execution topology or sequence input cannot host an
            auxiliary processor.
    """
    if auxiliary_segmentation is None:
        return
    if not params.parallel_mode:
        raise ValueError("Auxiliary segmentation requires --parallel-mode")
    if params.dataset_path is None:
        raise ValueError("Auxiliary segmentation requires --dataset-path")
    auxiliary_segmentation.preflight()


def _select_ego_mask_sample_indices(
    frame_timestamps_us: Sequence[int],
    *,
    samples_per_second: float,
    camera_id: str,
) -> tuple[int, ...]:
    """Select sparse ego-mask samples consistently across execution topologies.

    Args:
        frame_timestamps_us: Ordered camera timestamps in microseconds.
        samples_per_second: Requested temporal sampling rate.
        camera_id: Camera identifier used in diagnostics.

    Returns:
        Zero-based frame indices, excluding boundary frames when enough input
        frames are available.

    Raises:
        ValueError: If the active camera range contains no frames.
    """
    total_frames = len(frame_timestamps_us)
    if total_frames == 0:
        raise ValueError(f"No frames available for ego-mask estimation on camera {camera_id}")

    sequence_duration_s = (frame_timestamps_us[-1] - frame_timestamps_us[0]) / 1e6
    num_samples = max(1, int(samples_per_second * sequence_duration_s))
    if num_samples >= total_frames:
        logging.warning(
            "Requested %d samples but only %d frames are available for camera %s; using all frames",
            num_samples,
            total_frames,
            camera_id,
        )
        return tuple(range(total_frames))

    chunk_size = total_frames // (num_samples + 1)
    return tuple((sample_index + 1) * chunk_size - 1 for sample_index in range(num_samples))


def merge_zarr_itar(
    root_dir: Path,
    pattern: str,
    output_dir: Path,
):
    """Merged the zarr itar files together."""
    itar_files = list(root_dir.rglob(pattern))
    if not itar_files:
        raise ValueError(f"No files found matching pattern '{pattern}' in {root_dir}")
    return _merge_zarr_itar_files(itar_files, output_dir / itar_files[0].name)


def _merge_zarr_itar_files(itar_files: Sequence[Path], output_fp: Path) -> Path:
    """Merge an exact set of indexed-tar Zarr archives.

    Args:
        itar_files: Input archives that belong to one signal and sequence.
        output_fp: Destination archive path.

    Returns:
        The destination archive path.

    Raises:
        ValueError: If no inputs are provided or an input archive is missing.
    """
    if not itar_files:
        raise ValueError("No indexed-tar archives were provided for merging")
    missing_files = [path for path in itar_files if not path.is_file()]
    if missing_files:
        raise ValueError(f"Missing indexed-tar archives: {missing_files}")

    zarr_store_list = []
    for path in itar_files:
        zarr_store = ncore_stores.IndexedTarStore(path, mode="r")
        zarr_store_list.append(zarr_store)

    with ncore_stores.IndexedTarStore(output_fp, mode="w") as s_itar_out:
        for store in zarr_store_list:
            for k in store.keys():
                if k.startswith(".zmetadata"):
                    continue

                if k not in s_itar_out:
                    s_itar_out[k] = store[k]

        ncore_stores.consolidate_compressed_metadata(s_itar_out)

    return output_fp


def _merge_mask2former_outputs(
    output_path: Path,
    camera_ids: Sequence[str],
    store_base_name: str,
    seg_logits: bool,
    ego_mask_camera_ids: Sequence[str] = (),
    include_mask2former: bool = True,
) -> Path | None:
    """Merge per-camera Mask2Former indexed-tar archives.

    Args:
        output_path: Root containing per-camera archives under ``cam/``.
        camera_ids: Cameras processed for the current sequence.
        store_base_name: Current sequence or shard archive basename.
        seg_logits: Whether to merge optional logits archives.
        ego_mask_camera_ids: Cameras that produced shared SAM2-prepass
            archives.
        include_mask2former: Whether semantic archives were generated.

    Returns:
        Path to the merged semantic-label archive, or None when disabled.

    Raises:
        ValueError: If a requested archive family has no per-camera inputs.
    """
    camera_output_path = output_path / "cam"
    semantic_archive_name = f"{store_base_name}.aux.sseg.zarr.itar"
    semantic_output_path = None
    if include_mask2former:
        semantic_output_path = _merge_zarr_itar_files(
            [camera_output_path / camera_id / semantic_archive_name for camera_id in camera_ids],
            output_path / semantic_archive_name,
        )
    if seg_logits:
        logits_archive_name = f"{store_base_name}.aux.sseg-logits.zarr.itar"
        _merge_zarr_itar_files(
            [camera_output_path / camera_id / logits_archive_name for camera_id in camera_ids],
            output_path / logits_archive_name,
        )
    if ego_mask_camera_ids:
        ego_mask_archive_name = f"{store_base_name}.aux.egomask.zarr.itar"
        _merge_zarr_itar_files(
            [camera_output_path / camera_id / ego_mask_archive_name for camera_id in ego_mask_camera_ids],
            output_path / ego_mask_archive_name,
        )
    return semantic_output_path


def sequence_loader_shard_info(
    backend: DatasetBackend,
    sequence_loader: ncore.data.SequenceLoaderProtocol,
) -> Tuple[str, int, int]:
    """Get the output-part information for a sequence loader.

    Returns:
        base_name: str - base name of the sequence
        shard_id: int - shard id
        shard_count: int - shard count
    """

    part_info = backend.get_sequence_part_info(sequence_loader)
    return part_info.base_name, part_info.part_index, part_info.part_count


def _run_segmentation_camera_frames(
    *,
    writer: AuxDataWriter,
    camera_id: str,
    timestamped_image_frame_handles: Sequence[
        tuple[int, ncore.data.CameraSensorProtocol.EncodedImageDataHandleProtocol]
    ],
    semantic_metadata: dict[str, Any] | None,
    visualize: bool,
    process_frame: Callable[[int, int, Any, Path | None], Any],
    resolve_frame: Callable[[Any], SegmentationFrameOutput] | None = None,
    max_in_flight: int | None = 1,
    selected_frame_indices: frozenset[int] | None = None,
) -> None:
    """Decode and store camera frames independently of inference topology.

    Args:
        writer: Destination auxiliary-data writer.
        camera_id: Camera identifier used for metadata and frame outputs.
        timestamped_image_frame_handles: Ordered timestamp and encoded-frame
            handle pairs.
        semantic_metadata: Metadata stored before the first frame.
        visualize: Whether to construct per-frame visualization paths.
        process_frame: Inference callback returning either compact frame
            outputs or a token resolved by ``resolve_frame``.
        resolve_frame: Optional callback that waits for and resolves a
            submitted frame token.
        max_in_flight: Maximum submitted frames awaiting ordered resolution,
            or ``None`` when the frame processor owns dynamic capacity.
        selected_frame_indices: Optional worker-local subset to decode.

    Returns:
        None.

    Raises:
        RuntimeError: If capacity becomes unavailable, resolved output is
            invalid, or asynchronous storage fails.
        TypeError: If a resolved frame has an unexpected type.
        ValueError: If ``max_in_flight`` is not positive.
    """
    logger = logging.getLogger(__name__)
    if semantic_metadata is not None:
        logger.debug(f"Storing semantic meta {(camera_id, *semantic_metadata.values())}")
        writer.store_semantic_meta(camera_id, **semantic_metadata)

    if max_in_flight is not None and max_in_flight < 1:
        raise ValueError(f"max_in_flight must be positive, got {max_in_flight}")

    # AuxDataWriter initializes stores lazily without locking. Metadata
    # initialization above completes before this single consumer starts.
    with AsyncShardWriter() as shard_writer:
        pending_frames: deque[tuple[int, Any]] = deque()

        def store_frame(frame_timestamp_us: int, token: Any) -> None:
            """Resolve, own, and enqueue one frame's output buffers.

            Args:
                frame_timestamp_us: Timestamp used as the writer key.
                token: Completed frame output, or a submission token accepted
                    by ``resolve_frame``.

            Returns:
                None.

            Raises:
                TypeError: If the resolved value is not a Mask2Former frame
                    output.
                RuntimeError: If asynchronous storage previously failed.
            """
            output = resolve_frame(token) if resolve_frame is not None else token
            if not isinstance(output, SegmentationFrameOutput):
                raise TypeError(f"Unexpected segmentation frame output type: {type(output)}")

            if output.semantic_seg is not None:
                owned_seg = output.semantic_seg.copy()
                logger.debug(f"Storing semantic {(camera_id, frame_timestamp_us, owned_seg, 'png')}")
                shard_writer.submit(
                    writer.store_semantic_segmentation,
                    camera_id,
                    frame_timestamp_us,
                    owned_seg,
                    "png",
                    nbytes=owned_seg.width * owned_seg.height * len(owned_seg.getbands()),
                )
            if output.semantic_seg_logits is not None:
                owned_logits = np.ascontiguousarray(output.semantic_seg_logits, dtype=np.float16)
                if owned_logits is output.semantic_seg_logits:
                    owned_logits = owned_logits.copy()
                logger.debug(f"Storing semantic logits {(camera_id, frame_timestamp_us)}")
                shard_writer.submit(
                    writer.store_semantic_logits,
                    camera_id,
                    frame_timestamp_us,
                    owned_logits,
                    nbytes=owned_logits.nbytes,
                )
            if output.auxiliary_mask is not None:
                if output.auxiliary_mask_path is None:
                    raise RuntimeError("Auxiliary mask output omitted its destination path")
                owned_auxiliary_mask = np.ascontiguousarray(output.auxiliary_mask, dtype=np.uint8)
                if owned_auxiliary_mask is output.auxiliary_mask:
                    owned_auxiliary_mask = owned_auxiliary_mask.copy()
                logger.debug(f"Storing auxiliary mask {output.auxiliary_mask_path}")
                shard_writer.submit(
                    save_auxiliary_mask,
                    output.auxiliary_mask_path,
                    owned_auxiliary_mask,
                    nbytes=owned_auxiliary_mask.nbytes,
                )

        for frame_index, (frame_timestamp_us, frame_image_handle) in enumerate(
            tqdm.tqdm(timestamped_image_frame_handles)
        ):
            if selected_frame_indices is not None and frame_index not in selected_frame_indices:
                continue
            image = frame_image_handle.get_data().get_decoded_image()
            vis_path = (
                writer.output_dir_path / "seg_logits" / camera_id / writer.store_base_name / str(frame_timestamp_us)
                if visualize
                else None
            )
            while True:
                token = process_frame(frame_index, frame_timestamp_us, image, vis_path)
                if token is not None:
                    pending_frames.append((frame_timestamp_us, token))
                    break
                if max_in_flight is not None:
                    raise RuntimeError("Mask2Former frame processor unexpectedly reported unavailable capacity")
                if pending_frames:
                    store_frame(*pending_frames.popleft())
                else:
                    time.sleep(0.001)

            if max_in_flight is not None and len(pending_frames) >= max_in_flight:
                store_frame(*pending_frames.popleft())

        while pending_frames:
            store_frame(*pending_frames.popleft())


def _run_egomask_samples(
    *,
    writer: AuxDataWriter,
    camera_id: str,
    resolution: tuple[int, int],
    metadata: dict[str, Any],
    samples: Sequence[tuple[int, int, Any]],
    aggregation_method: Literal["majority"],
    visualize: bool,
    infer_mask: Callable[[int, Any], NDArray[np.uint8]],
) -> NDArray[np.uint8] | None:
    """Infer, store, aggregate, and optionally visualize sampled ego masks."""
    if not samples:
        return None
    if aggregation_method != "majority":
        raise ValueError(f"Unsupported ego-mask aggregation method: {aggregation_method}")

    writer.store_egomask_meta(
        camera_id,
        resolution=list(resolution),
        dataset_name=metadata["dataset_name"],
        method=metadata["method"],
        pretrained_checkpoint=metadata["pretrained_checkpoint"],
        aggregation_method=aggregation_method,
    )
    width, height = resolution
    mask_sum = np.zeros((height, width), dtype=np.uint64)
    sampled_frames: list[tuple[int, Any]] = []
    vis_folder = writer.output_dir_path / "ego_mask" / camera_id / writer.store_base_name if visualize else None
    if vis_folder is not None:
        os.makedirs(vis_folder, exist_ok=True)

    for request_id, frame_timestamp_us, frame_image_handle in tqdm.tqdm(samples):
        image = frame_image_handle.get_data().get_decoded_image()
        ego_mask = infer_mask(request_id, image)
        writer.store_egomask(
            camera_id,
            frame_timestamp_us,
            ego_mask,
            image_file_format="png",
        )
        mask_sum += ego_mask
        if vis_folder is not None:
            visualize_egomask_overlay(
                ego_mask,
                np.asarray(image),
                vis_folder,
                frame_timestamp_us,
                "raw_ego",
            )
            sampled_frames.append((frame_timestamp_us, frame_image_handle))

    aggregate = (mask_sum >= len(samples) / 2).astype(np.uint8)
    writer.store_egomask(
        camera_id,
        0,
        aggregate,
        image_file_format="png",
    )
    if vis_folder is not None:
        for frame_timestamp_us, frame_image_handle in sampled_frames:
            image = frame_image_handle.get_data().get_decoded_image()
            visualize_egomask_overlay(
                aggregate,
                np.asarray(image),
                vis_folder,
                frame_timestamp_us,
                "agg_ego",
            )
    return aggregate


def _run_shared_sam2_prepass(
    *,
    writer: AuxDataWriter,
    camera_id: str,
    timestamped_image_frame_handles: Sequence[
        tuple[int, ncore.data.CameraSensorProtocol.EncodedImageDataHandleProtocol]
    ],
    sample_indices: tuple[int, ...],
    resolution: tuple[int, int],
    sam2_metadata: dict[str, Any] | None,
    aggregation_method: Literal["majority"],
    visualize: bool,
    client: SharedDeviceFrameClient,
) -> NDArray[np.uint8] | None:
    """Generate raw and aggregate SAM2 masks before semantic segmentation.

    Args:
        writer: Destination auxiliary-data writer.
        camera_id: Camera identifier used for outputs and visualization.
        timestamped_image_frame_handles: All active camera frames.
        sample_indices: Frame indices selected for promptless SAM2.
        resolution: Camera resolution as ``(width, height)``.
        sam2_metadata: Serializable metadata reported by the SAM2 processor.
        aggregation_method: Aggregation applied across sampled masks.
        visualize: Whether to save raw and aggregate mask overlays.
        client: Shared device-frame client used for SAM2-only requests.

    Returns:
        Majority aggregate as ``uint8[H, W]``, or ``None`` when no samples
        were selected.

    Raises:
        RuntimeError: If metadata or a SAM2 output is missing.
        ValueError: If the aggregation method is unsupported.
    """
    if not sample_indices:
        return None
    if sam2_metadata is None:
        raise RuntimeError("SAM2 prepass requires estimator metadata")

    def infer_mask(sample_index: int, image: Any) -> NDArray[np.uint8]:
        if is_dark_image(image):
            width, height = resolution
            return np.zeros((height, width), dtype=np.uint8)
        while True:
            pending = client.submit(
                request_id=sample_index,
                image=image,
                enabled_processors=frozenset({SAM2_PROCESSOR}),
            )
            if pending is not None:
                break
            time.sleep(0.001)
        resolved = client.resolve(pending)
        ego_mask = resolved.outputs.get(SAM2_MASK_OUTPUT)
        if ego_mask is None:
            raise RuntimeError("SAM2 prepass completed without a writable mask output")
        return ego_mask

    samples = tuple(
        (
            sample_index,
            timestamped_image_frame_handles[sample_index][0],
            timestamped_image_frame_handles[sample_index][1],
        )
        for sample_index in sample_indices
    )
    return _run_egomask_samples(
        writer=writer,
        camera_id=camera_id,
        resolution=resolution,
        metadata=sam2_metadata,
        samples=samples,
        aggregation_method=aggregation_method,
        visualize=visualize,
        infer_mask=infer_mask,
    )


@dataclass(frozen=True, slots=True)
class _SegmentationHostWorkerConfig:
    """Picklable configuration for one segmentation host worker."""

    worker_id: int
    loader_payload: LoaderWorkerPayload
    zarr_store_type: str
    output_path: Path
    camera_id: str
    camera_frame_range: range
    semantic_metadata: dict[str, Any] | None
    resolution: tuple[int, int]
    visualize: bool
    slot_queue: Any
    sam2_sample_indices: tuple[int, ...]
    sam2_metadata: dict[str, Any] | None
    ego_mask_aggregation_method: Literal["majority"]
    request_queue: Any
    result_queue: Any
    error_queue: Any
    stop_event: Any
    auxiliary_sample_indices: tuple[int, ...] = ()
    auxiliary_outputs: AuxiliarySegmentationOutputs | None = None
    auxiliary_camera_stats: Any | None = None


def _segmentation_host_worker(config: _SegmentationHostWorkerConfig) -> None:
    """Decode and store one camera while a separate process owns CUDA.

    Args:
        config: Serializable loader, output, model, and IPC configuration.

    Returns:
        None. The worker raises failures to its process boundary and always
        closes its shared-memory attachments.
    """
    worker_id = config.worker_id
    loader_payload = config.loader_payload
    zarr_store_type = config.zarr_store_type
    output_path = config.output_path
    camera_id = config.camera_id
    camera_frame_range = config.camera_frame_range
    semantic_metadata = config.semantic_metadata
    resolution = config.resolution
    visualize = config.visualize
    slot_queue = config.slot_queue
    sam2_sample_indices = config.sam2_sample_indices
    sam2_metadata = config.sam2_metadata
    auxiliary_sample_indices = config.auxiliary_sample_indices
    auxiliary_outputs = config.auxiliary_outputs
    auxiliary_camera_stats = config.auxiliary_camera_stats
    ego_mask_aggregation_method = config.ego_mask_aggregation_method
    request_queue = config.request_queue
    result_queue = config.result_queue
    error_queue = config.error_queue
    stop_event = config.stop_event
    client: SharedDeviceFrameClient | None = None
    frame_processor: SharedSegmentationFrameProcessor | None = None
    static_processor_input: DeviceFrameSharedArray | None = None
    writer: AuxDataWriter | None = None
    writer_finalized = False
    try:
        backend, sequence_loader = _open_worker_sequence(loader_payload)
        base_name, shard_id, shard_count = sequence_loader_shard_info(backend, sequence_loader)
        writer = AuxDataWriter(
            output_path,
            base_name,
            sequence_loader.sequence_id,
            shard_id,
            shard_count,
            zarr_store_type,
        )
        metadata = None if semantic_metadata is None else {**semantic_metadata, "resolution": list(resolution)}
        stuff_colors = np.asarray(metadata["stuff_colors"] if metadata is not None else [], dtype=np.uint8)

        camera_sensor = sequence_loader.get_camera_sensor(camera_id)
        timestamped_handles = list(
            zip(
                camera_sensor.get_frames_timestamps_us(),
                [camera_sensor.get_frame_handle(i) for i in range(camera_sensor.frames_count)],
            )
        )[camera_frame_range.start : camera_frame_range.stop]
        client = SharedDeviceFrameClient(
            worker_id=worker_id,
            slot_queue=slot_queue,
            request_queue=request_queue,
            result_queue=result_queue,
            stop_event=stop_event,
        )
        generated_camera_mask = _run_shared_sam2_prepass(
            writer=writer,
            camera_id=camera_id,
            timestamped_image_frame_handles=timestamped_handles,
            sample_indices=sam2_sample_indices,
            resolution=resolution,
            sam2_metadata=sam2_metadata,
            aggregation_method=ego_mask_aggregation_method,
            visualize=visualize,
            client=client,
        )

        camera_mask = get_camera_sensor_mask(camera_sensor, backend=backend)
        if camera_mask is None:
            camera_mask = generated_camera_mask
        processor_inputs: tuple[DeviceFrameSharedArrayDescriptor, ...] = ()
        if camera_mask is not None:
            width, height = resolution
            if camera_mask.shape != (height, width):
                raise ValueError(
                    f"Ego mask shape {camera_mask.shape} does not match camera resolution {(height, width)}"
                )
            static_processor_input = DeviceFrameSharedArray.create(
                CAMERA_MASK_INPUT,
                np.asarray(camera_mask, dtype=np.bool_),
            )
            processor_inputs = (static_processor_input.descriptor,)

        frame_processor = SharedSegmentationFrameProcessor(
            stuff_colors=stuff_colors,
            processor_inputs=processor_inputs,
            client=client,
            include_mask2former=metadata is not None,
            auxiliary_outputs=auxiliary_outputs,
            auxiliary_sample_indices=frozenset(auxiliary_sample_indices),
            camera_id=camera_id,
            camera_frame_range_start=camera_frame_range.start,
        )

        _run_segmentation_camera_frames(
            writer=writer,
            camera_id=camera_id,
            timestamped_image_frame_handles=timestamped_handles,
            semantic_metadata=metadata,
            visualize=visualize and metadata is not None,
            process_frame=frame_processor,
            resolve_frame=frame_processor.resolve,
            max_in_flight=None,
            selected_frame_indices=frozenset(auxiliary_sample_indices) if metadata is None else None,
        )
        if auxiliary_camera_stats is not None:
            camera_statistics = frame_processor.auxiliary_camera_statistics()
            if camera_statistics is not None:
                auxiliary_camera_stats.append(dict(camera_statistics))

        writer.finalize()
        writer_finalized = True
    except Exception:
        stop_event.set()
        error_queue.put((camera_id, traceback.format_exc()))
        raise
    finally:
        if frame_processor is not None:
            frame_processor.close()
        elif client is not None:
            client.close()
        if static_processor_input is not None:
            static_processor_input.close()
        if writer is not None and not writer_finalized:
            try:
                writer.finalize()
            except Exception:
                logging.exception("Failed to finalize partial auxiliary outputs for camera %s", camera_id)


def _wait_for_segmentation_services(
    service_processes: dict[int, Any],
    ready_queue: Any,
    *,
    sam2_enabled: bool,
    auxiliary_processor_name: str | None = None,
    mask2former_enabled: bool = True,
    timeout_s: float = 600.0,
) -> DeviceFrameServiceReady:
    """Wait for a consistent readiness report from every segmentation service.

    Args:
        service_processes: Service processes keyed by stable service identity.
        ready_queue: Queue receiving one readiness payload from each service.
        sam2_enabled: Whether every service must report SAM2 metadata.
        auxiliary_processor_name: Extension processor every service must
            report, or ``None`` when no extension processor is configured.
        mask2former_enabled: Whether every service must report Mask2Former.
        timeout_s: Maximum total time to wait for all readiness reports.

    Returns:
        The service-zero readiness payload used as common model metadata.

    Raises:
        RuntimeError: If a service exits, reports an error, duplicates a
            service identity, reports an unknown service identity, omits
            required metadata, or disagrees with another service.
        TypeError: If the queue returns an unexpected payload type.
    """
    deadline = time.monotonic() + timeout_s
    ready_by_service: dict[int, DeviceFrameServiceReady] = {}
    while len(ready_by_service) < len(service_processes):
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0:
            missing_service_ids = sorted(service_processes.keys() - ready_by_service.keys())
            raise RuntimeError(
                f"Segmentation services did not report readiness within {timeout_s:g}s: {missing_service_ids}"
            )
        try:
            ready = ready_queue.get(timeout=min(0.2, remaining_s))
        except queue.Empty:
            dead_unreported_services = [
                service_id
                for service_id, service_process in service_processes.items()
                if service_id not in ready_by_service and not service_process.is_alive()
            ]
            if dead_unreported_services:
                details = ", ".join(
                    f"service {service_id} (exit code {service_processes[service_id].exitcode})"
                    for service_id in dead_unreported_services
                )
                raise RuntimeError(f"Segmentation services exited during startup: {details}")
            continue

        if not isinstance(ready, DeviceFrameServiceReady):
            raise TypeError(f"Unexpected device frame startup result type: {type(ready)}")
        if ready.service_id not in service_processes:
            raise RuntimeError(f"Segmentation service reported unexpected identity {ready.service_id}")
        if ready.service_id in ready_by_service:
            raise RuntimeError(f"Segmentation service {ready.service_id} reported readiness more than once")
        ready_by_service[ready.service_id] = ready

    startup_failures = [
        f"service {service_id} on CUDA device {ready.device_id}:\n{ready.error}"
        for service_id, ready in sorted(ready_by_service.items())
        if ready.error is not None or ready.processor_metadata is None
    ]
    if startup_failures:
        raise RuntimeError("Segmentation services failed to start:\n" + "\n".join(startup_failures))

    reference_ready = ready_by_service[0]
    assert reference_ready.processor_metadata is not None
    for service_id, ready in sorted(ready_by_service.items()):
        if ready.processor_metadata != reference_ready.processor_metadata:
            assert ready.processor_metadata is not None
            reference_mask2former = reference_ready.processor_metadata.get(MASK2FORMER_PROCESSOR)
            service_mask2former = ready.processor_metadata.get(MASK2FORMER_PROCESSOR)
            if (
                reference_mask2former is not None
                and service_mask2former is not None
                and service_mask2former.backend != reference_mask2former.backend
            ):
                raise RuntimeError(
                    f"Segmentation service {service_id} uses Mask2Former backend "
                    f"{service_mask2former.backend}, but service 0 uses {reference_mask2former.backend}; "
                    "mixed inference backends are not supported"
                )
            raise RuntimeError(f"Segmentation service {service_id} returned inconsistent processor metadata")
    if mask2former_enabled and MASK2FORMER_PROCESSOR not in reference_ready.processor_metadata:
        raise RuntimeError("Segmentation services started without Mask2Former metadata")
    if sam2_enabled and SAM2_PROCESSOR not in reference_ready.processor_metadata:
        raise RuntimeError("Segmentation services started without required SAM2 metadata")
    if auxiliary_processor_name is not None and auxiliary_processor_name not in reference_ready.processor_metadata:
        raise RuntimeError(f"Segmentation services started without required {auxiliary_processor_name} metadata")
    return reference_ready


def _stop_processes_if_alive(processes: Iterable[Any], *, timeout_s: float = 10.0) -> None:
    """Stop child processes that remain alive during failure cleanup.

    Args:
        processes: Child processes that may still be running.
        timeout_s: Maximum wait for each join, terminate, and kill stage.

    Returns:
        None.
    """
    for process in processes:
        if not process.is_alive():
            continue
        try:
            process.join(timeout=timeout_s)
            if process.is_alive():
                logging.warning("Terminating unresponsive process %s", process.name)
                process.terminate()
                process.join(timeout=timeout_s)
            if process.is_alive():
                logging.error("Killing unresponsive process %s", process.name)
                process.kill()
                process.join(timeout=timeout_s)
            if process.is_alive():
                logging.error("Process %s remained alive after kill", process.name)
        except Exception:
            logging.exception("Failed to stop process %s during cleanup", process.name)


def _run_shared_segmentation(
    sensor_ranges: SensorRanges,
    opened_sequence: OpenedSequence,
    params: CLIBaseParams,
    output_path: Path,
    auxiliary_segmentation: AuxiliarySegmentationRequest | None = None,
) -> Path | None:
    """Run camera workers against a shared pool of device-frame services.

    Args:
        sensor_ranges: Active frame ranges keyed by camera and lidar ID.
        opened_sequence: Source sequence and worker-safe reopening payload.
        params: Validated CLI parameters controlling models and outputs.
        output_path: Root output directory.
        auxiliary_segmentation: Optional extension-owned processor running
            beside Mask2Former on a sampled subset of frames.

    Returns:
        Path to the merged semantic-label archive.

    Raises:
        RuntimeError: If CUDA is unavailable or any child process fails.
        ValueError: If no cameras are selected or required outputs are absent.
        TypeError: If a service process returns an invalid startup payload.
    """
    backend = opened_sequence.backend
    sequence_loader = opened_sequence.loader
    cuda_device_count = torch.cuda.device_count()
    if cuda_device_count == 0:
        raise RuntimeError("No CUDA device available")

    camera_items = list(sensor_ranges.camera_frame_ranges.items())
    if not camera_items:
        raise ValueError("Shared segmentation requires at least one camera")
    total_frame_count = sum(len(camera_frame_range) for _, camera_frame_range in camera_items)
    if total_frame_count == 0:
        raise ValueError("Shared segmentation requires at least one frame")
    service_count = min(
        cuda_device_count * params.workers_per_gpu,
        total_frame_count,
    )
    transport_slot_count = max(service_count, len(camera_items))
    logging.info(
        "Starting %d shared segmentation service(s) across %d visible CUDA device(s) "
        "(up to %d per device), "
        "with %d dynamically shared frame transport slot(s)",
        service_count,
        cuda_device_count,
        params.workers_per_gpu,
        transport_slot_count,
    )
    sam2_sample_indices_by_worker: dict[int, tuple[int, ...]] = {}
    for worker_id, (camera_id, camera_frame_range) in enumerate(camera_items):
        if len(camera_frame_range) == 0:
            sam2_sample_indices_by_worker[worker_id] = ()
            continue
        if _run_ego_mask_for_camera(params.ego_mask, params.ego_mask_camera_ids, camera_id):
            camera_timestamps = sequence_loader.get_camera_sensor(camera_id).get_frames_timestamps_us()[
                camera_frame_range.start : camera_frame_range.stop
            ]
            sam2_sample_indices_by_worker[worker_id] = _select_ego_mask_sample_indices(
                [int(timestamp_us) for timestamp_us in camera_timestamps],
                samples_per_second=params.ego_mask_samples_per_second,
                camera_id=camera_id,
            )
        else:
            sam2_sample_indices_by_worker[worker_id] = ()
    sam2_enabled = any(sam2_sample_indices_by_worker.values())
    mask2former_enabled = params.segmentation_backend == "mask2former"
    sam2_camera_ids = tuple(
        camera_items[worker_id][0]
        for worker_id, sample_indices in sam2_sample_indices_by_worker.items()
        if sample_indices
    )

    ctx = multiprocessing.get_context("spawn")
    auxiliary_started = time.perf_counter()
    auxiliary_outputs: AuxiliarySegmentationOutputs | None = None
    auxiliary_processor: AuxiliaryProcessorFactory | None = None
    auxiliary_manager: Any | None = None
    auxiliary_camera_stats: Any | None = None
    auxiliary_camera_stats_snapshot: tuple[Mapping[str, Any], ...] = ()
    auxiliary_sample_indices_by_worker: Mapping[int, tuple[int, ...]] = {}
    if auxiliary_segmentation is not None:
        auxiliary_outputs = auxiliary_segmentation.frame_outputs()
        auxiliary_processor = auxiliary_segmentation.processor_factory()
        auxiliary_sample_indices_by_worker = auxiliary_segmentation.select_camera_frames(
            sequence_loader,
            camera_items,
        )
    request_queue = ctx.Queue(maxsize=transport_slot_count)
    result_queues = [ctx.Queue(maxsize=transport_slot_count) for _ in camera_items]
    ready_queue = ctx.Queue(maxsize=service_count)
    error_queue = ctx.Queue()
    stop_event = ctx.Event()
    pipeline_factory = SegmentationDeviceFramePipelineFactory(
        estimate_logits=params.seg_logits,
        enable_trt=params.enable_trt,
        include_sam2=sam2_enabled,
        include_mask2former=mask2former_enabled,
        output_uncertainty_mask=params.visualize,
        auxiliary_processor=auxiliary_processor,
    )

    resolutions_by_worker = {
        worker_id: cast(
            tuple[int, int],
            tuple(int(value) for value in sequence_loader.get_camera_sensor(camera_id).model_parameters.resolution),
        )
        for worker_id, (camera_id, _) in enumerate(camera_items)
    }
    initial_resolution = resolutions_by_worker[0]
    service_processes = {
        service_id: ctx.Process(
            name=f"segmentation-service-{service_id}-device-{service_id % cuda_device_count}",
            target=run_device_frame_service_process,
            kwargs={
                "service_id": service_id,
                "device_id": service_id % cuda_device_count,
                "initial_resolution": initial_resolution,
                "request_queue": request_queue,
                "result_queues": result_queues,
                "ready_queue": ready_queue,
                "stop_event": stop_event,
                "pipeline_factory": pipeline_factory,
                "max_staging_slots": min(3, transport_slot_count),
                "allowed_resolutions": frozenset(resolutions_by_worker.values()),
            },
        )
        for service_id in range(service_count)
    }
    owned_slots: list[DeviceFrameSharedBuffers] = []
    worker_processes: dict[int, Any] = {}
    premature_service_ids: set[int] = set()
    worker_error_failures: list[str] = []
    auxiliary_processor_metadata: dict[str, Any] | None = None

    try:
        if auxiliary_outputs is not None:
            auxiliary_manager = ctx.Manager()
            auxiliary_camera_stats = auxiliary_manager.list()
        for service_process in service_processes.values():
            service_process.start()

        reference_ready = _wait_for_segmentation_services(
            service_processes,
            ready_queue,
            sam2_enabled=sam2_enabled,
            mask2former_enabled=mask2former_enabled,
            auxiliary_processor_name=None if auxiliary_outputs is None else auxiliary_outputs.processor_name,
        )

        assert reference_ready.processor_metadata is not None
        logging.info("Shared segmentation processors: %s", sorted(reference_ready.processor_metadata))
        semantic_metadata = (
            reference_ready.processor_metadata[MASK2FORMER_PROCESSOR].storage if mask2former_enabled else None
        )
        sam2_processor_metadata = reference_ready.processor_metadata.get(SAM2_PROCESSOR)
        sam2_metadata = None if sam2_processor_metadata is None else sam2_processor_metadata.storage
        if auxiliary_outputs is not None:
            auxiliary_processor_metadata = reference_ready.processor_metadata[auxiliary_outputs.processor_name].storage
        num_classes = len(semantic_metadata["stuff_classes"]) if semantic_metadata is not None else 0
        worker_ids_by_resolution: dict[tuple[int, int], list[int]] = {}
        for worker_id, resolution in resolutions_by_worker.items():
            worker_ids_by_resolution.setdefault(resolution, []).append(worker_id)
        slot_counts_by_resolution = {
            resolution: len(worker_ids) for resolution, worker_ids in worker_ids_by_resolution.items()
        }
        for extra_slot_index in range(transport_slot_count - len(camera_items)):
            resolution = resolutions_by_worker[extra_slot_index % len(camera_items)]
            slot_counts_by_resolution[resolution] += 1

        slot_queues_by_resolution: dict[tuple[int, int], Any] = {}
        for resolution, slot_count in slot_counts_by_resolution.items():
            slot_queue = ctx.Queue(maxsize=slot_count)
            slot_queues_by_resolution[resolution] = slot_queue
            for _ in range(slot_count):
                output_specs = pipeline_factory.output_specs(
                    resolution,
                    num_classes=num_classes,
                )
                owned_slot = DeviceFrameSharedBuffers.create(resolution, output_specs)
                owned_slots.append(owned_slot)
                slot_queue.put(owned_slot.slot)

        for worker_id, (camera_id, camera_frame_range) in enumerate(camera_items):
            resolution = resolutions_by_worker[worker_id]
            sam2_sample_indices = sam2_sample_indices_by_worker[worker_id]
            auxiliary_sample_indices = auxiliary_sample_indices_by_worker.get(worker_id, ())
            camera_output_path = output_path / "cam" / camera_id
            worker_process = ctx.Process(
                name=f"segmentation-host-{camera_id}",
                target=_segmentation_host_worker,
                args=(
                    _SegmentationHostWorkerConfig(
                        worker_id=worker_id,
                        loader_payload=opened_sequence.worker_payload,
                        zarr_store_type=params.zarr_store_type,
                        output_path=camera_output_path,
                        camera_id=camera_id,
                        camera_frame_range=camera_frame_range,
                        semantic_metadata=semantic_metadata,
                        resolution=resolution,
                        visualize=params.visualize,
                        slot_queue=slot_queues_by_resolution[resolution],
                        sam2_sample_indices=sam2_sample_indices,
                        sam2_metadata=sam2_metadata,
                        auxiliary_sample_indices=auxiliary_sample_indices,
                        auxiliary_outputs=auxiliary_outputs,
                        auxiliary_camera_stats=auxiliary_camera_stats,
                        ego_mask_aggregation_method=params.ego_mask_aggregation_method,
                        request_queue=request_queue,
                        result_queue=result_queues[worker_id],
                        error_queue=error_queue,
                        stop_event=stop_event,
                    ),
                ),
            )
            worker_process.start()
            worker_processes[worker_id] = worker_process

        while any(worker_process.is_alive() for worker_process in worker_processes.values()):
            for worker_process in worker_processes.values():
                worker_process.join(timeout=0.05)
            if any(worker_process.exitcode not in (None, 0) for worker_process in worker_processes.values()):
                stop_event.set()
            for service_id, service_process in service_processes.items():
                if not service_process.is_alive():
                    premature_service_ids.add(service_id)
            if premature_service_ids:
                stop_event.set()
            while True:
                try:
                    camera_id, error = error_queue.get_nowait()
                except queue.Empty:
                    break
                worker_error_failures.append(f"{camera_id} worker failed:\n{error}")

        if not stop_event.is_set():
            for _ in service_processes:
                try:
                    request_queue.put(None, timeout=0.2)
                except queue.Full:
                    stop_event.set()
                    break

        for worker_process in worker_processes.values():
            worker_process.join()
        for service_process in service_processes.values():
            service_process.join()

        failures = [
            f"{worker_process.name} exited with code {worker_process.exitcode}"
            for worker_process in worker_processes.values()
            if worker_process.exitcode != 0
        ]
        failures.extend(worker_error_failures)
        while True:
            try:
                camera_id, error = error_queue.get_nowait()
            except queue.Empty:
                break
            failures.append(f"{camera_id} worker failed:\n{error}")
        failures.extend(
            f"{service_process.name} exited with code {service_process.exitcode}"
            for service_process in service_processes.values()
            if service_process.exitcode != 0
        )
        failures.extend(
            f"segmentation-service-{service_id} exited before shutdown" for service_id in sorted(premature_service_ids)
        )
        if failures:
            raise RuntimeError("\n".join(failures))
        if auxiliary_camera_stats is not None:
            auxiliary_camera_stats_snapshot = tuple(auxiliary_camera_stats)
    finally:
        stop_event.set()
        for _ in service_processes:
            try:
                request_queue.put_nowait(None)
            except queue.Full:
                break
        _stop_processes_if_alive(worker_processes.values())
        _stop_processes_if_alive(service_processes.values())
        for owned_slot in owned_slots:
            owned_slot.close()
        if auxiliary_manager is not None:
            auxiliary_manager.shutdown()

    if auxiliary_segmentation is not None:
        assert auxiliary_processor_metadata is not None
        auxiliary_segmentation.finalize(
            camera_ids=tuple(camera_id for camera_id, _ in camera_items),
            camera_statistics=auxiliary_camera_stats_snapshot,
            processor_metadata=auxiliary_processor_metadata,
            wall_time_seconds=time.perf_counter() - auxiliary_started,
        )
    store_base_name, _, _ = sequence_loader_shard_info(backend, sequence_loader)
    return _merge_mask2former_outputs(
        output_path,
        tuple(sensor_ranges.camera_frame_ranges),
        store_base_name,
        params.seg_logits,
        ego_mask_camera_ids=sam2_camera_ids,
        include_mask2former=mask2former_enabled,
    )


def lidar_segmentation_worker(
    device_id: int,
    gpu_sem: multiprocessing.synchronize.BoundedSemaphore,
    lidar_id: str,
    camera_ids: list,
    loader_payload: LoaderWorkerPayload,
    zarr_store_type: str,
    lidar_seg_ensemble_cuda: bool,
    cam_seg_fp: Path,
    output_path: Path,
    frame_timestamps_us: list[int],
    visualize: bool,
):
    """Worker: Lidar Segmentation."""
    from apps.av.aux_gen.estimators import LidarSegmentationByProjectEstimator

    with gpu_sem:
        torch.cuda.set_device(device_id)
        logging.info(
            "\n*********** Device Sanity Check ************\n"
            f"visible device counts: {torch.cuda.device_count()} \n"
            f"current device: {torch.cuda.current_device()} \n"
            f"device name: {torch.cuda.get_device_name(0)} \n"
            "*********** End of Device Sanity Check ************"
        )

        # Init the cam seg aux loader
        cam_seg_aux_loader = CamAuxDataLoader(cam_seg_fp)

        # construct the current loader and writer for each process
        backend, sequence_loader = _open_worker_sequence(loader_payload)

        base_name, shard_id, shard_count = sequence_loader_shard_info(backend, sequence_loader)

        estimator = LidarSegmentationByProjectEstimator(
            sequence_loader,
            cam_seg_aux_loader,
            camera_ids=camera_ids,
            ensemble_cuda=lidar_seg_ensemble_cuda,
        )

        # init the writer
        writer = AuxDataWriter(
            output_path,
            base_name,
            sequence_loader.sequence_id,
            shard_id,
            shard_count,
            zarr_store_type,
        )

        run_lidar_segmentation(
            estimator,
            sequence_loader,
            writer,
            lidar_id,
            frame_timestamps_us,
            visualize,
        )

        del estimator
        torch.cuda.empty_cache()

        # properly close permanent annotation shard
        writer.finalize()


def run_lidar_segmentation_parallel(
    opened_sequence: OpenedSequence,
    params: CLIBaseParams,
    output_path: Path,
    cam_seg_fp: Path,
    lidar_id: str,
    camera_ids: list[str],
    frame_timestamps_us: list[int],
    num_threads: int,
):
    """Split the jobs based on the number of available GPU"""

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No CUDA device available")

    ctx = multiprocessing.get_context("spawn")
    gpu_sems = [ctx.BoundedSemaphore(int(np.ceil(num_threads / num_gpus))) for _ in range(num_gpus)]
    procs = []
    output_path_list = []

    def split_chunks(lst, n):
        """Split list into n roughly equal parts."""
        return [list(chunk) for chunk in np.array_split(lst, n)]

    frame_timestamp_chunks = split_chunks(frame_timestamps_us, num_threads)

    for idx, ts_list in enumerate(frame_timestamp_chunks):
        device_id = idx % num_gpus

        logging.info(f"Assign chunk {idx} -> GPU {device_id}")
        lidar_output_path = output_path / "lidar_seg" / f"chunk_{idx}"
        output_path_list.append(lidar_output_path)
        p = ctx.Process(
            target=lidar_segmentation_worker,
            args=(
                device_id,
                gpu_sems[device_id],
                lidar_id,
                camera_ids,
                opened_sequence.worker_payload,
                params.zarr_store_type,
                params.lidar_seg_ensemble_cuda,
                cam_seg_fp,
                lidar_output_path,
                ts_list,
                params.visualize,
            ),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Worker process failed with exit code {p.exitcode}")

    final_output_fp = merge_zarr_itar(
        root_dir=Path(output_path / "lidar_seg"),
        pattern="*.aux.lidar-sseg.zarr.itar",
        output_dir=output_path,
    )
    # the workers also write the lidar-camera visibility; merge it like the semantics (serial mode writes both)
    merge_zarr_itar(
        root_dir=Path(output_path / "lidar_seg"),
        pattern="*.aux.lidar-camvis.zarr.itar",
        output_dir=output_path,
    )

    return final_output_fp


class AsyncShardWriter:
    """Run shard writes on one background thread, preserving submission order.

    Encoding the semantic PNG (``optimize=True`` on a palette image) and writing
    it into the shard is the largest non-inference cost of a Mask2Former frame,
    while TensorRT enqueue and stream waits release the GIL. The consumer can
    therefore overlap shard writes with inference.

    Exactly one consumer thread, so writes reach the store in submission order
    and the stored bytes do not change.

    Two preconditions belong to the caller:

    * Submitted objects must be safe to own. ``PILImage.fromarray`` aliases the
      array it wraps for 8-bit modes, so a palette image built over a buffer the
      next frame overwrites has to be copied before it is submitted.
    * The target ``AuxDataWriter`` must not be used from another thread while the
      queue is open. Its store initialisation is a lazy check-then-act and would
      race. Submit, then ``close()``, then read or finalize the writer.

    Backpressure is a byte budget rather than an item count because payload sizes
    differ by orders of magnitude: a 4K palette label image is about 8 MB while a
    4K float16 logits array is about 330 MB. A payload larger than the whole
    budget is still accepted, it simply waits until it is the only one in flight.
    """

    _logger = logging.getLogger(__name__)
    _QueueItem = Tuple[Callable[..., None], Tuple[object, ...], int]

    def __init__(self, max_pending_bytes: int = 32 * 1024 * 1024) -> None:
        self._queue: queue.Queue[Optional["AsyncShardWriter._QueueItem"]] = queue.Queue()
        self._budget = max_pending_bytes
        self._pending_bytes = 0
        self._space = threading.Condition()
        self._error: Optional[BaseException] = None
        self._failed_call: Optional[str] = None
        self._closed = False
        self._thread = threading.Thread(target=self._consume, name="aux-shard-writer", daemon=True)
        self._thread.start()

    def _consume(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            func, args, nbytes = item
            try:
                # After a failure, keep draining without writing so the producer
                # never blocks waiting for budget that is never released.
                if self._first_error() is None:
                    func(*args)
            except BaseException as exc:  # surfaced on the producer thread
                with self._space:
                    if self._error is None:
                        self._error = exc
                        self._failed_call = f"{getattr(func, '__name__', func)}{args[:2]!r}"
                        self._logger.exception("Shard write failed: %s", self._failed_call)
            finally:
                with self._space:
                    self._pending_bytes -= nbytes
                    self._space.notify_all()

    def submit(self, func: Callable[..., None], *args: object, nbytes: int) -> None:
        """Queue ``func(*args)``, blocking while ``nbytes`` would exceed the budget."""
        if self._closed:
            raise RuntimeError("AsyncShardWriter is closed")
        self._raise_if_failed()
        with self._space:
            while self._pending_bytes and self._pending_bytes + nbytes > self._budget:
                self._space.wait()
            self._pending_bytes += nbytes
        self._queue.put((func, args, nbytes))

    def _first_error(self) -> Optional[BaseException]:
        with self._space:
            return self._error

    def _raise_if_failed(self) -> None:
        if (error := self._first_error()) is not None:
            raise RuntimeError(f"async shard write failed: {self._failed_call}") from error

    def _drain(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self._space:
            pending = self._pending_bytes
        if pending:
            # tqdm counts submissions, so the bar is already at 100% here.
            self._logger.info("Draining %.1f MB of pending shard writes", pending / 1e6)
        self._queue.put(None)
        self._thread.join()

    def close(self) -> None:
        """Drain all pending writes and stop the thread. Must precede finalize()."""
        self._drain()
        self._raise_if_failed()

    def __enter__(self) -> "AsyncShardWriter":
        return self

    def __exit__(
        self, exc_type: Optional[type[BaseException]], exc: Optional[BaseException], tb: Optional[TracebackType]
    ) -> None:
        self._drain()
        if exc_type is None:
            self._raise_if_failed()
        elif self._error is not None:
            # The body already failed; reporting the write error here would
            # replace the traceback that explains why.
            self._logger.error("Shard write also failed: %s", self._failed_call, exc_info=self._error)


def _run_serial_mask2former_segmentation(
    segmentation_estimator: Mask2FormerSegmentationEstimator,
    writer: AuxDataWriter,
    camera_id: str,
    camera_model_parameters: ConcreteCameraModelParametersUnion,
    camera_sensor: ncore.data.CameraSensorProtocol,
    timestamped_image_frame_handles: list[Tuple[int, ncore.data.CameraSensorProtocol.EncodedImageDataHandleProtocol]],
    *,
    backend: DatasetBackend,
    visualize: bool = False,
    has_ego_mask: bool = False,
) -> None:
    """Run Mask2Former semantic segmentation for a camera's image frames.

    Predicts per-pixel semantic labels (and optionally logits) for each frame,
    stores metadata and results via the writer. When no camera sensor mask is
    available and has_ego_mask is True, uses the writer's aggregated egomask.

    Args:
        segmentation_estimator: Mask2Former model used for prediction.
        writer: Aux data writer for storing semantic meta, segmentations, and logits.
        camera_id: Identifier of the camera (used for writer keys and paths).
        camera_model_parameters: Camera intrinsics/resolution for the estimator.
        camera_sensor: NCore camera sensor; used to obtain sensor mask if available.
        timestamped_image_frame_handles: List of (timestamp_us, image_handle) to process.
        visualize: If True, write segmentation logits for debugging visualization.
        has_ego_mask: If True and no camera sensor mask is available, use the writer's
            aggregated egomask (writer.get_egomask). If False, do not request an
            egomask, avoiding creation of an empty egomask store (e.g. with --no-ego-mask).

    Returns:
        None.

    Raises:
        ValueError: If estimator inputs, model outputs, masks, or frame
            capacity violate their contracts.
        RuntimeError: If inference omits required outputs or asynchronous
            storage fails.
    """
    logger = logging.getLogger(__name__)
    segmentation_estimator.set_resolution(camera_model_parameters.resolution.tolist())

    camera_mask = get_camera_sensor_mask(camera_sensor, backend=backend)
    if camera_mask is None and has_ego_mask:
        logger.info("Using aux generated ego-mask (aggregated) from writer.")
        camera_mask = writer.get_egomask(camera_id, 0)

    def process_frame(
        _frame_index: int,
        _frame_timestamp_us: int,
        image: Any,
        vis_path: Path | None,
    ) -> SegmentationFrameOutput:
        """Run serial inference for one decoded camera frame.

        Args:
            _frame_index: Unused frame index required by the common callback
                contract.
            _frame_timestamp_us: Unused timestamp required by the common
                callback contract.
            image: Decoded RGB image accepted by the segmentation estimator.
            vis_path: Optional path for MMSeg visualization output.

        Returns:
            Caller-owned paletted ``uint8[H, W]`` semantic labels and optional
            ``float16[H, W, C]`` logits.

        Raises:
            RuntimeError: If the estimator omits semantic labels.
            ValueError: If estimator input or output contracts are invalid.
        """
        segmentation = segmentation_estimator.predict(image, vis_path=vis_path, ego_mask=camera_mask)

        if segmentation.semantic_seg is None:
            raise RuntimeError("Mask2Former prediction did not return semantic labels")
        return SegmentationFrameOutput(
            semantic_seg=segmentation.semantic_seg,
            semantic_seg_logits=segmentation.semantic_seg_logits,
        )

    _run_segmentation_camera_frames(
        writer=writer,
        camera_id=camera_id,
        timestamped_image_frame_handles=timestamped_image_frame_handles,
        semantic_metadata=segmentation_estimator.metadata().storage,
        visualize=visualize,
        process_frame=process_frame,
    )


def run_dinov2_feature_precomputation(
    dinov2_backend: str,
    dinov2_width: int,
    dinov2_pca_dim: int,
    loader: ncore.data.SequenceLoaderProtocol,
    sensor_ranges: SensorRanges,
    *,
    backend: DatasetBackend,
) -> tuple[Optional[PCA], Optional[Feature2ColorTransform]]:
    dinov2_pca: Optional[PCA] = None
    dinov2_color_transform: Optional[Feature2ColorTransform] = None
    logger = logging.getLogger(__name__)

    match dinov2_backend:
        case "none":
            pass

        case _:
            from apps.av.aux_gen.estimators import DINOv2Estimator

            dinov2_features_all = []

            for camera_id, camera_frame_range in sensor_ranges.camera_frame_ranges.items():
                camera_sensor = loader.get_camera_sensor(camera_id)

                # Collect all image handles with timestamps and restrict to active frames
                # Sample every 15 frames here to reduce computation burden.
                timestamped_image_frame_handles_subset: list[
                    Tuple[int, ncore.data.CameraSensorProtocol.EncodedImageDataHandleProtocol]
                ] = list(
                    zip(
                        camera_sensor.get_frames_timestamps_us(),
                        [camera_sensor.get_frame_handle(i) for i in range(camera_sensor.frames_count)],
                    )
                )[camera_frame_range.start : camera_frame_range.stop : 15]

                camera_model_parameters = require_concrete_camera_parameters(camera_sensor.model_parameters)
                dinov2_estimator = DINOv2Estimator(
                    DINOv2Estimator.resolution_from_dino_width(
                        camera_model_parameters.resolution.tolist(), dinov2_width
                    ),
                    backend=dinov2_backend,
                )

                logger.debug(f"Precompute DINOv2 for {(camera_id, *dinov2_estimator.get_metadata().values())}")

                camera_mask: Optional[np.ndarray] = get_camera_sensor_mask(camera_sensor, backend=backend)

                dinov2_feats_list: list[np.ndarray] = []

                for _, frame_image_handle in tqdm.tqdm(timestamped_image_frame_handles_subset):
                    encoded_img_data = frame_image_handle.get_data()
                    img = encoded_img_data.get_decoded_image()

                    dinov2_feats, dinov2_mask = dinov2_estimator.predict(img, camera_mask)
                    if dinov2_mask is not None:
                        dinov2_feats = dinov2_feats[~dinov2_mask]
                    dinov2_feats_list.append(dinov2_feats)

                dinov2_features_all.extend(dinov2_feats_list)

                del dinov2_estimator
                torch.cuda.empty_cache()

            dinov2_features_sample = np.concatenate(dinov2_features_all)

            if dinov2_pca_dim > 0:
                # Parameter following EmerNeRF practice (n_iter = 20, no additional centralizing)
                dinov2_pca = PCA.from_data(dinov2_features_sample, q=dinov2_pca_dim, n_iter=20, data_centralized=True)
                dinov2_features_sample = dinov2_pca.transform(dinov2_features_sample)

            assert isinstance(dinov2_features_sample, np.ndarray)
            dinov2_color_transform = Feature2ColorTransform.from_feature(dinov2_features_sample)

    return dinov2_pca, dinov2_color_transform


def run_dinov2_feature_extraction(
    dinov2_estimator: DINOv2Estimator,
    writer: AuxDataWriter,
    camera_id: str,
    camera_sensor: ncore.data.CameraSensorProtocol,
    timestamped_image_frame_handles: list[Tuple[int, ncore.data.CameraSensorProtocol.EncodedImageDataHandleProtocol]],
    pca: Optional[PCA],
    color_transform: Feature2ColorTransform,
    *,
    backend: DatasetBackend,
    visualize: bool = False,
) -> None:
    if visualize:
        import matplotlib.pyplot as plt

    logger = logging.getLogger(__name__)
    logger.debug(f"Storing DINOv2 feature for {(camera_id, *dinov2_estimator.get_metadata().values())}")

    writer.store_dinov2_meta(
        camera_id, dinov2_estimator.get_metadata(), color_transform.to_dict(), pca.to_dict() if pca else None
    )

    # Ego mask is used to determine if DINOv2 feature is valid for reconstruction.
    camera_mask: Optional[np.ndarray] = get_camera_sensor_mask(camera_sensor, backend=backend)

    for frame_timestamp_us, frame_image_handle in tqdm.tqdm(timestamped_image_frame_handles):
        encoded_img_data = frame_image_handle.get_data()
        img = encoded_img_data.get_decoded_image()
        dinov2_feats, dino_ego_mask = dinov2_estimator.predict(img, ego_mask=camera_mask)

        if pca is not None:
            dinov2_feats = cast(np.ndarray, pca.transform(dinov2_feats))

        dino_valid_mask = None
        if dino_ego_mask is not None:
            dino_valid_mask = ~dino_ego_mask

        logger.debug(f"Storing semantic dinov2 feats {(camera_id, frame_timestamp_us)}")
        writer.store_dinov2(camera_id, frame_timestamp_us, dinov2_feats, dino_valid_mask)

        if visualize:
            vis_path = writer.output_dir_path / "dinov2_feats" / camera_id / writer.store_base_name
            os.makedirs(vis_path, exist_ok=True)
            dinov2_feats_rgb = color_transform.transform(dinov2_feats)
            plt.imsave(str(vis_path / f"{frame_timestamp_us}.jpg"), dinov2_feats_rgb)


def run_depth_estimation(
    depth_estimator: DepthEstimator,
    writer: AuxDataWriter,
    store_depth_as_png: bool,
    camera_id: str,
    timestamped_image_frame_handles: list[Tuple[int, ncore.data.CameraSensorProtocol.EncodedImageDataHandleProtocol]],
    visualize: bool = False,
) -> None:
    logger = logging.getLogger(__name__)

    logger.debug("Storing depth estimation meta")

    if visualize:
        import cv2
        import matplotlib

        cmap = matplotlib.colormaps.get_cmap("Spectral_r")  # type: ignore

    for frame_idx, (frame_timestamp_us, frame_image_handle) in enumerate(tqdm.tqdm(timestamped_image_frame_handles)):
        image = np.asarray(frame_image_handle.get_data().get_decoded_image())
        depth = depth_estimator.predict(image).squeeze(axis=0)

        if frame_idx == 0:
            # Store the depth metadata
            writer.store_depth_meta(
                camera_id,
                resolution=list([depth.shape[1], depth.shape[0]]),
                store_depth_as_png=store_depth_as_png,
                max_depth_m=depth_estimator.max_depth_m,
                method=depth_estimator.method,
            )

        if visualize:
            vis_folder = writer.output_dir_path / "depth" / camera_id / writer.store_base_name
            os.makedirs(str(vis_folder), exist_ok=True)

            # Convert the predicted depth to color and store it
            vis_depth = np.copy(depth)
            vis_depth = (vis_depth - vis_depth.min()) / (vis_depth.max() - vis_depth.min()) * 255.0
            vis_depth = vis_depth.astype(np.uint8)
            vis_depth = (cmap(vis_depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            cv2.imwrite(str((vis_folder / (str(frame_timestamp_us) + ".jpeg"))), vis_depth)

        # writing results to the shard
        logger.debug(
            f"Storing depth estimation {(camera_id, frame_timestamp_us, depth, 'png' if store_depth_as_png else 'npy')}"
        )
        writer.store_depth(camera_id, frame_timestamp_us, depth)


def run_lidar_segmentation(
    estimator: LidarSegmentationByProjectEstimator,
    loader: ncore.data.SequenceLoaderProtocol,
    writer: AuxDataWriter,
    lidar_id: str,
    frame_timestamps_us: list[int],
    visualize: bool = False,
) -> None:
    logger = logging.getLogger(__name__)

    lidar_sensor = loader.get_lidar_sensor(lidar_id)

    estimator.set_meta()
    logger.debug(f"Storing semantic meta {(lidar_id, *estimator.get_semantic_metadata().values())}")
    writer.store_lidar_semantic_meta(lidar_id, **estimator.get_semantic_metadata())
    logger.debug(f"Storing lidar-camera-visibility meta {(lidar_id, estimator.get_visibility_metadata())}")
    writer.store_lidar_camera_visibility_meta(lidar_id, **estimator.get_visibility_metadata())

    if visualize:
        vis_path = writer.output_dir_path / "lidar_seg" / lidar_id / writer.store_base_name
        os.makedirs(vis_path, exist_ok=True)
        for camera_id in estimator.camera_ids:
            os.makedirs(vis_path / camera_id, exist_ok=True)
    else:
        vis_path = None

    if not estimator.ensemble_cuda:
        logging.info(
            "Start lidar segmentation. If you find the speed is too slow (< 1it/s), please switch to cuda version, lower "
            " the number of numba threads (using CLI --num-threads), or stop other tasks with high CPU usage."
        )
    for frame_timestamp_us in tqdm.tqdm(frame_timestamps_us):
        lidar_seg, visibility_mask = estimator.predict(
            lidar_sensor=lidar_sensor,
            frame_timestamp_us=frame_timestamp_us,
            vis_path=vis_path,
        )
        # writing results to shard
        logger.debug(f"Storing semantic {(lidar_id, frame_timestamp_us, lidar_seg)}")
        writer.store_lidar_semantic_segmentation(lidar_id, frame_timestamp_us, lidar_seg, "png")
        writer.store_lidar_camera_visibility(lidar_id, frame_timestamp_us, visibility_mask)


def visualize_egomask_overlay(
    ego_mask: np.ndarray,
    frame_np: np.ndarray,
    vis_folder: Path,
    frame_timestamp_us: int,
    vis_name: str,
) -> None:
    import cv2

    color = np.array([0, 255, 0], dtype=np.uint8)
    mask_color = np.where(ego_mask[..., None], color, frame_np)
    # blend with original frame using alpha blending
    out_frame = cv2.addWeighted(frame_np, 0.8, mask_color, 0.2, 0)
    # save overlay image
    PILImage.fromarray(out_frame).save(vis_folder / f"{frame_timestamp_us}_{vis_name}.png", optimize=True)


def run_egomask_estimation(
    estimator: PromptlessSAM2EgoMaskEstimator,
    writer: AuxDataWriter,
    camera_id: str,
    camera_model_parameters: ConcreteCameraModelParametersUnion,
    timestamped_image_frame_handles: list[Tuple[int, ncore.data.CameraSensorProtocol.EncodedImageDataHandleProtocol]],
    aggregation_method: Literal["majority"] = "majority",
    visualize: bool = False,
) -> None:
    """Generate and store ego-mask instance-segmentation with finetuned promptless SAM2 estimator"""
    resolution = tuple(int(value) for value in camera_model_parameters.resolution)
    _run_egomask_samples(
        writer=writer,
        camera_id=camera_id,
        resolution=cast(tuple[int, int], resolution),
        metadata=estimator.metadata().storage,
        samples=tuple(
            (sample_index, frame_timestamp_us, frame_image_handle)
            for sample_index, (frame_timestamp_us, frame_image_handle) in enumerate(timestamped_image_frame_handles)
        ),
        aggregation_method=aggregation_method,
        visualize=visualize,
        infer_mask=lambda _sample_index, image: estimator.predict(image),
    )


def _with_cli_options(command: Callable[..., Any]) -> Callable[..., Any]:
    """Apply public aux-data options to a newly created command callback."""
    decorators = (
        click.option("--dataset-path", type=str, help="Path to an NCore sequence metadata file", required=False),
        click.option("--poses-component-group", type=str, help="V4 component group for 'poses'", default="default"),
        click.option(
            "--intrinsics-component-group", type=str, help="V4 component group for 'intrinsics'", default="default"
        ),
        click.option("--masks-component-group", type=str, help="V4 component group for 'masks'", default="default"),
        click.option("--cuboids-component-group", type=str, help="V4 component group for 'cuboids'", default="default"),
        click.option("--output-dir", type=str, help="Path to the output folder", required=True),
        click.option(
            "--camera-id",
            "camera_ids",
            multiple=True,
            type=str,
            help="Cameras to be used (multiple value option, all if not specified)",
            default=None,
        ),
        click.option(
            "--lidar-id",
            "lidar_ids",
            multiple=True,
            type=str,
            help="Lidars to be used (multiple value option, all if not specified)",
            default=None,
        ),
        click.option(
            "--segmentation-backend",
            default="mask2former",
            type=click.Choice(["none", "mask2former"], case_sensitive=False),
            help="Perform segmentation, please choose a backend. \
                (0)None, not to perform segmentation.\
                (1)Mask2Former, supporting semantic segmentation and logits saving via --seg-logits (optional).",
        ),
        click.option(
            "--seg-logits/--no-seg-logits",
            is_flag=True,
            default=False,
            help="Perform semantic segmentation and save logits",
        ),
        click.option(
            "--enable-trt/--disable-trt", is_flag=True, default=True, help="Enable running TRT optimized models"
        ),
        click.option(
            "--dinov2-backend",
            default="none",
            type=click.Choice(
                ["none", "nv_dinov2", "dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"],
                case_sensitive=False,
            ),
            help="DINOv2 backbone to be used for feature extraction",
        ),
        click.option(
            "--dinov2-pca-dim",
            type=int,
            default=-1,
            help="PCA dimension for the features to be extracted (-1 means not to apply PCA).",
        ),
        click.option(
            "--dinov2-width",
            type=int,
            default=256,
            help="DINOv2 feature width (default 256)",
        ),
        click.option(
            "--lidar-seg-camvis/--no-lidar-seg-camvis",
            is_flag=True,
            default=True,
            help="Perform lidar segmentation and point-in-cameras visibility determination",
        ),
        click.option(
            "--lidar-seg-ensemble-cuda/--no-lidar-seg-ensemble-cuda",
            is_flag=True,
            default=True,
            help="Wether to use cuda-based ensemble function for lidar segmentation",
        ),
        click.option(
            "--ego-mask/--no-ego-mask", is_flag=True, default=True, help="Perform automatic ego-mask estimation"
        ),
        click.option(
            "--ego-mask-samples-per-second",
            type=click.FloatRange(min=0.0002, max=30.0),
            default=0.2,
            help="Number of frames to sample for ego-mask estimation per second (default: 0.2).",
        ),
        click.option(
            "--ego-mask-aggregation-method",
            type=click.Choice(["majority"], case_sensitive=False),
            default="majority",
            help="Aggregation method for ego-mask estimation when using multiple samples.",
        ),
        click.option(
            "--ego-mask-camera-id",
            "ego_mask_camera_ids",
            multiple=True,
            type=str,
            default=None,
            help="Restrict ego-mask estimation to these camera IDs (default: all cameras in the run). Can be passed multiple times.",
        ),
        click.option(
            "--zarr-store-type",
            type=click.Choice(["itar", "directory"], case_sensitive=False),
            default="itar",
            help="Zarr store type to store the aux data in",
        ),
        click.option(
            "--open-consolidated/--no-open-consolidated",
            is_flag=True,
            default=True,
            help="Open shards consolidated meta-data",
        ),
        click.option(
            "--num-threads",
            type=str,
            help="Number of threads to use (use 'auto' to determine number of threads from current cpu count)",
            is_flag=False,
            flag_value="auto",
            default="8",
        ),
        click.option("--serial-mode", is_flag=True, help="Process camera and lidar tasks serially (default)."),
        click.option(
            "--parallel-mode/--no-parallel-mode",
            is_flag=True,
            default=False,
            help="Process tasks in parallel; GPU workers are shared between CPU workers.",
        ),
        click.option(
            "--workers-per-gpu",
            type=click.IntRange(min=1),
            default=1,
            help="Number of shared GPU workers to run on each visible CUDA device.",
        ),
        click.option("--debug", is_flag=True, default=False, help="Enable debug logging outputs"),
        click.option("--visualize", is_flag=True, default=False, help="Enable outputting visualization results"),
        click.option(
            "--store-meta",
            is_flag=True,
            default=False,
            help="Store meta-file per shard with CLI arguments and maglev runtime logging (if available)",
        ),
        nurec_utils_cli.scopedtimer_cli_options(print_func=logging.info),
    )
    for decorator in reversed(decorators):
        command = decorator(command)
    return command


def _run_cli(
    ctx: click.Context,
    configure_depth_generation: DepthGenerationConfigurator | None,
    configure_auxiliary_segmentation: AuxiliarySegmentationConfigurator | None,
    configure_instance_generation: InstanceGenerationConfigurator | None,
    **kwargs: Any,
) -> None:
    """Process inferred auxiliary-data signals for NCore shards.

    Args:
        ctx: Click context receiving validated parameters for subcommands.
        configure_depth_generation: Optional internal depth-option consumer.
        configure_auxiliary_segmentation: Optional internal consumer of
            auxiliary segmentation options.
        configure_instance_generation: Optional internal consumer of instance-generation options.
        **kwargs: Parsed top-level command options.

    Returns:
        None.

    Raises:
        click.UsageError: If execution-mode options are mutually exclusive or
            would otherwise be ignored.
        ValueError: If validated options or selected data are inconsistent.
        RuntimeError: If model inference or output generation fails.
    """

    selected_execution_modes = [
        mode
        for mode, selected in (
            ("serial", kwargs.pop("serial_mode")),
            ("parallel", kwargs.pop("parallel_mode")),
        )
        if selected
    ]
    if len(selected_execution_modes) > 1:
        raise click.UsageError("--serial-mode and --parallel-mode are mutually exclusive")
    kwargs["execution_mode"] = selected_execution_modes[0] if selected_execution_modes else "serial"
    if kwargs["execution_mode"] == "serial" and kwargs["workers_per_gpu"] != 1:
        raise click.UsageError("--workers-per-gpu requires --parallel-mode")
    if kwargs["execution_mode"] == "parallel" and kwargs["zarr_store_type"] != "itar":
        raise click.UsageError("--parallel-mode requires --zarr-store-type=itar")
    kwargs.setdefault("explicit_path_pattern", None)
    # Preserve extension-owned options after the configurators consume them.
    unconfigured_kwargs = kwargs.copy()
    depth_generation = None if configure_depth_generation is None else configure_depth_generation(kwargs)
    auxiliary_segmentation = (
        None if configure_auxiliary_segmentation is None else configure_auxiliary_segmentation(kwargs)
    )
    if (
        kwargs["execution_mode"] == "parallel"
        and kwargs["segmentation_backend"] != "mask2former"
        and auxiliary_segmentation is None
    ):
        raise click.UsageError("--parallel-mode requires --segmentation-backend=mask2former or auxiliary segmentation")
    instance_generation = None if configure_instance_generation is None else configure_instance_generation(kwargs)
    extension_cli_params = {name: value for name, value in unconfigured_kwargs.items() if name not in kwargs}
    params = CLIBaseParams(**kwargs)

    # Initialize basic top-level logger configuration
    logging.basicConfig(
        level=logging.DEBUG if params.debug else logging.INFO,
        format="<%(asctime)s|%(levelname)s|%(filename)s:%(lineno)d|%(name)s> %(message)s",
    )
    state = CLIState(
        params=params,
        depth_generation=depth_generation,
        auxiliary_segmentation=auxiliary_segmentation,
        extension_cli_params=extension_cli_params,
        instance_generation=instance_generation,
    )
    ctx.obj = state

    if ctx.invoked_subcommand is None:
        # No subcommand was called - run default processing without range restriction for all enabled sensors

        def range_callback(sequence_loader: ncore.data.SequenceLoaderProtocol):
            camera_ids = params.camera_ids if params.camera_ids else sequence_loader.camera_ids
            lidar_ids = params.lidar_ids if params.lidar_ids else sequence_loader.lidar_ids

            return SensorRanges(
                camera_frame_ranges={
                    camera_id: range(sequence_loader.get_camera_sensor(camera_id).frames_count)
                    for camera_id in camera_ids
                },
                lidar_frame_ranges={
                    lidar_id: range(sequence_loader.get_lidar_sensor(lidar_id).frames_count) for lidar_id in lidar_ids
                },
            )

        try:
            ncore_aux_data(
                params,
                range_callback,
                state.depth_generation,
                state.extension_cli_params,
                state.auxiliary_segmentation,
                state.instance_generation,
            )
        finally:
            ScopedTimer.print_summary()


def _run_offset(
    ctx: click.Context,
    sequence_seek_sec: float | None,
    sequence_duration_sec: float | None,
) -> None:
    """Perform per-sequence time range offset to all sensor"""
    state: CLIState = ctx.obj
    params = state.params

    def range_callback(loader: ncore.data.SequenceLoaderProtocol):
        camera_ids = params.camera_ids if params.camera_ids else loader.camera_ids
        lidar_ids = params.lidar_ids if params.lidar_ids else loader.lidar_ids

        start_timestamp_us, end_timestamp_us = time_bounds(
            [loader.sequence_timestamp_interval_us.start, loader.sequence_timestamp_interval_us.stop + 1],
            sequence_seek_sec,
            sequence_duration_sec,
        )

        shard_restriction_interval_us = HalfClosedInterval(
            start_timestamp_us, end_timestamp_us + 1
        )  # make sure to include end-timestamp in interval

        return SensorRanges(
            camera_frame_ranges={
                camera_id: shard_restriction_interval_us.cover_range(
                    loader.get_camera_sensor(camera_id).get_frames_timestamps_us()
                )
                for camera_id in camera_ids
            },
            lidar_frame_ranges={
                lidar_id: shard_restriction_interval_us.cover_range(
                    loader.get_lidar_sensor(lidar_id).get_frames_timestamps_us()
                )
                for lidar_id in lidar_ids
            },
        )

    ncore_aux_data(
        params,
        range_callback,
        state.depth_generation,
        state.extension_cli_params,
        state.auxiliary_segmentation,
        state.instance_generation,
    )
    ScopedTimer.print_summary()


def _run_sensor_frames(
    ctx: click.Context,
    main_sensor_ids: list[str],
    start_frame: int | None,
    stop_frame: int | None,
) -> None:
    """Perform per-shard time range restriction based on sensor frame numbers of main sensors

    Minimum / maximum time ranges will be inferred if multiple main sensors are specified

    """

    assert len(main_sensor_ids), "At least a single main sensor needs to be provided"

    state: CLIState = ctx.obj
    params = state.params

    def range_callback(sequence_loader: ncore.data.SequenceLoaderProtocol):
        def sensor_timestamps(sensor_id: str) -> np.ndarray:
            if sensor_id in sequence_loader.camera_ids:
                return sequence_loader.get_camera_sensor(sensor_id).get_frames_timestamps_us()
            elif sensor_id in sequence_loader.lidar_ids:
                return sequence_loader.get_lidar_sensor(sensor_id).get_frames_timestamps_us()
            elif sensor_id in sequence_loader.radar_ids:
                return sequence_loader.get_radar_sensor(sensor_id).get_frames_timestamps_us()
            else:
                raise ValueError(f"Unknown main sensor id {sensor_id}")

        # determine combined ranges of all main time-sensors
        main_sensors_timestamps_us = [
            sensor_timestamps(sensor_id)[start_frame:stop_frame] for sensor_id in main_sensor_ids
        ]
        start_timestamp_us = min(
            [main_sensor_timestamps_us.min() for main_sensor_timestamps_us in main_sensors_timestamps_us]
        )
        end_timestamp_us = max(
            [main_sensor_timestamps_us.max() for main_sensor_timestamps_us in main_sensors_timestamps_us]
        )

        shard_restriction_interval_us = HalfClosedInterval(
            start_timestamp_us, end_timestamp_us + 1
        )  # make sure to include end-timestamp in interval

        camera_ids = params.camera_ids if params.camera_ids else sequence_loader.camera_ids
        lidar_ids = params.lidar_ids if params.lidar_ids else sequence_loader.lidar_ids

        return SensorRanges(
            camera_frame_ranges={
                camera_id: shard_restriction_interval_us.cover_range(
                    sequence_loader.get_camera_sensor(camera_id).get_frames_timestamps_us()
                )
                for camera_id in camera_ids
            },
            lidar_frame_ranges={
                lidar_id: shard_restriction_interval_us.cover_range(
                    sequence_loader.get_lidar_sensor(lidar_id).get_frames_timestamps_us()
                )
                for lidar_id in lidar_ids
            },
        )

    ncore_aux_data(
        params,
        range_callback,
        state.depth_generation,
        state.extension_cli_params,
        state.auxiliary_segmentation,
        state.instance_generation,
    )
    ScopedTimer.print_summary()


def build_ncore_aux_data_cli(
    configure_depth_generation: DepthGenerationConfigurator | None = None,
    configure_auxiliary_segmentation: AuxiliarySegmentationConfigurator | None = None,
    configure_instance_generation: InstanceGenerationConfigurator | None = None,
) -> click.Group:
    """Build an independent aux-data CLI with optional extension configuration.

    Args:
        configure_depth_generation: Optional callback consuming depth options
            from the mutable parsed-options dictionary. Called first; returns
            a depth request, or ``None`` when depth generation is disabled.
        configure_auxiliary_segmentation: Optional callback consuming auxiliary
            segmentation options from the same dictionary after the depth
            callback. Returns a request, or ``None`` when disabled. A missing
            callback leaves that extension unconfigured. Each callback must
            remove its extension-owned keys, including disabled options, before
            the remaining values are passed to ``CLIBaseParams``. Consumed
            values are retained separately as invocation metadata.
        configure_instance_generation: Optional instance-generation configurator,
            following the same option-consumption contract.

    Returns:
        An independent Click group with the public options and range
        subcommands. The extending entrypoint must register its own options
        on this group before invoking it. Callback exceptions propagate when
        the command runs, not when this group is built.
    """

    @click.group(name="cli", invoke_without_command=True)
    @_with_cli_options
    @click.pass_context
    def command(ctx: click.Context, **kwargs: Any) -> None:
        """Process inferred auxiliary-data signals for NCore shards."""
        _run_cli(
            ctx, configure_depth_generation, configure_auxiliary_segmentation, configure_instance_generation, **kwargs
        )

    @command.command(name="offset")
    @click.option(
        "--sequence-seek-sec",
        type=click.FloatRange(min=0.0, max_open=True),
        default=None,
        help="Time to skip for processing each *individual* sequence (in seconds)",
    )
    @click.option(
        "--sequence-duration-sec",
        type=click.FloatRange(min=0.0, max_open=True),
        default=None,
        help="Restrict total duration of each *individual* sequence (in seconds)",
    )
    @click.pass_context
    def offset_command(
        ctx: click.Context,
        sequence_seek_sec: float | None,
        sequence_duration_sec: float | None,
    ) -> None:
        """Perform per-sequence time range offset to all sensors."""
        _run_offset(ctx, sequence_seek_sec, sequence_duration_sec)

    @command.command(name="sensor-frames")
    @click.option(
        "--main-sensor-id",
        "main_sensor_ids",
        multiple=True,
        type=str,
        help="Main sensors to be used (multiple value option, requires at least one)",
        default=None,
    )
    @click.option(
        "--start-frame",
        type=click.IntRange(min=0, max_open=True),
        help="If provided, the initial frame index of *all* main sensors to be exported",
        default=None,
    )
    @click.option(
        "--stop-frame",
        type=click.IntRange(min=0, max_open=True),
        help="If provided, the past-the-end frame index of *all* main sensors to be exported",
        default=None,
    )
    @click.pass_context
    def sensor_frames_command(
        ctx: click.Context,
        main_sensor_ids: list[str],
        start_frame: int | None,
        stop_frame: int | None,
    ) -> None:
        """Restrict each shard's time range using main-sensor frame numbers."""
        _run_sensor_frames(ctx, main_sensor_ids, start_frame, stop_frame)

    return command


cli = build_ncore_aux_data_cli()


def ncore_aux_data(
    params: CLIBaseParams,
    range_callback: SensorRangeCallback,
    depth_generation: DepthGenerationRequest | None,
    extension_cli_params: dict[str, Any],
    auxiliary_segmentation: AuxiliarySegmentationRequest | None = None,
    instance_generation: InstanceGenerationRequest | None = None,
) -> None:
    """Generate requested auxiliary data for every selected sequence.

    Args:
        params: Validated CLI parameters controlling inputs, models, execution
            topology, and outputs.
        range_callback: Callable selecting camera and lidar frame ranges for
            each decoded sequence.
        depth_generation: Optional model-agnostic depth generation request.
        extension_cli_params: CLI arguments consumed by an extending
            entrypoint and retained for invocation metadata.
        auxiliary_segmentation: Optional model-agnostic auxiliary segmentation
            request run beside shared Mask2Former segmentation.
        instance_generation: Optional instance-mask pass run before the other models.

    Returns:
        None.

    Raises:
        click.UsageError: If neither or both supported input selectors are set.
        ValueError: If option combinations or selected data are invalid.
        RuntimeError: If inference workers, model services, or output writing
            fail.
    """
    # Determine and set number of numba threads to use
    num_threads = (
        multiprocessing.cpu_count() if (num_threads_param := params.num_threads) == "auto" else int(num_threads_param)
    )
    num_threads_str = str(num_threads)
    logging.info(f"Setting number of threads to {num_threads_str}")
    _preflight_auxiliary_segmentation(params, auxiliary_segmentation)
    if instance_generation is not None:
        instance_generation.preflight()

    ## Initialize the sequence data loaders
    opened_sequences: list[OpenedSequence] = []
    if bool(params.dataset_path) == bool(params.explicit_path_pattern):
        raise click.UsageError("Exactly one sequence input must be provided")
    if params.explicit_path_pattern:
        # Treat each shard as an individual self-contained sequence.
        explicit_input_provider = require_explicit_sequence_input_provider()
        opened_sequences = [
            OpenedSequence(
                backend=backend,
                loader=loader,
                worker_payload=ExplicitPathLoaderWorkerPayload(
                    path=str(shard_path),
                    open_consolidated=params.open_consolidated,
                    loader_max_workers=None,
                    explicit_input_provider=explicit_input_provider,
                ),
            )
            for backend, loader, shard_path in explicit_input_provider.open_inputs(
                params.explicit_path_pattern,
                open_consolidated=params.open_consolidated,
                loader_max_workers=None,
            )
        ]
    elif params.dataset_path:
        # Load the dataset manifest and treat it as a single sequence.
        request = DatasetRequest(
            metadata_path=parse_universal_path(params.dataset_path, s3_cache_type="blockcache"),
            open_consolidated=params.open_consolidated,
            loader_max_workers=None,
            poses_component_group=params.poses_component_group,
            intrinsics_component_group=params.intrinsics_component_group,
            masks_component_group=params.masks_component_group,
            cuboids_component_group=params.cuboids_component_group,
        )
        resolved = resolve_dataset(request)
        opened_sequences = [
            OpenedSequence(
                backend=resolved.backend,
                loader=resolved.create_sequence_loader(),
                worker_payload=ManifestLoaderWorkerPayload(
                    resolved_dataset=resolved,
                ),
            )
        ]

    if auxiliary_segmentation is not None and len(opened_sequences) != 1:
        raise ValueError("Auxiliary segmentation requires exactly one NCore sequence")

    # Set up output path
    (output_path := Path(params.output_dir)).mkdir(parents=True, exist_ok=True)

    # Check estimation dependencies
    if params.segmentation_backend != "mask2former" and params.seg_logits:
        raise ValueError("segmentation logits requires 'mask2former' segmentation backend")
    if params.parallel_mode and params.segmentation_backend != "mask2former" and auxiliary_segmentation is None:
        raise ValueError("Parallel mode requires 'mask2former' segmentation backend or auxiliary segmentation")
    if not params.parallel_mode and params.workers_per_gpu != 1:
        raise ValueError("Workers per GPU requires parallel mode")
    if params.parallel_mode and params.zarr_store_type != "itar":
        raise ValueError("Parallel modes require --zarr-store-type=itar for per-worker archive merging")

    # Process each input sequence individually
    for opened_sequence in opened_sequences:
        backend = opened_sequence.backend
        sequence_loader = opened_sequence.loader
        sensor_ranges = range_callback(sequence_loader)

        base_name, shard_id, shard_count = sequence_loader_shard_info(backend, sequence_loader)

        # Initialize the output writer for current annotation container
        writer = AuxDataWriter(
            output_path,
            base_name,
            sequence_loader.sequence_id,
            shard_id,
            shard_count,
            params.zarr_store_type,
        )

        logging.info(
            f"Processing aux data for sequence {sequence_loader.sequence_id} ({sequence_loader.sequence_paths}) on cameras {sensor_ranges.camera_frame_ranges.keys()}"
        )

        if instance_generation is not None:
            with ScopedTimer("aux_gen/instance_segmentation", tag=TimingTag.DEFAULT):
                instance_generation.generate(sequence_loader, backend, writer, sensor_ranges.camera_frame_ranges)

        # If PCA is needed for DINOv2 feature extraction, pre-compute the transformation with **all cameras** jointly.
        #   TODO [JH]: In the future, we might want to perform PCA on all shards provided.
        with ScopedTimer("aux_gen/dinov2_precomputation", tag=TimingTag.DEFAULT):
            dinov2_pca, dinov2_color_transform = run_dinov2_feature_precomputation(
                params.dinov2_backend,
                params.dinov2_width,
                params.dinov2_pca_dim,
                sequence_loader,
                sensor_ranges,
                backend=backend,
            )

        cam_seg_aux_loader = None
        cam_seg_output_fp = None
        shared_segmentation = params.parallel_mode and (
            params.segmentation_backend == "mask2former" or auxiliary_segmentation is not None
        )
        if shared_segmentation:
            logging.info(
                f"Running shared segmentation (backend={params.segmentation_backend}, "
                f"auxiliary={auxiliary_segmentation is not None}) for {sensor_ranges.camera_frame_ranges.keys()}"
            )
            with ScopedTimer("aux_gen/mask2former_parallel", tag=TimingTag.DEFAULT):
                cam_seg_output_fp = _run_shared_segmentation(
                    sensor_ranges=sensor_ranges,
                    opened_sequence=opened_sequence,
                    params=params,
                    output_path=output_path,
                    auxiliary_segmentation=auxiliary_segmentation,
                )
            if cam_seg_output_fp is not None:
                cam_seg_aux_loader = CamAuxDataLoader(cam_seg_output_fp)

        shared_sam2_prepass = shared_segmentation
        for camera_id, camera_frame_range in sensor_ranges.camera_frame_ranges.items():
            camera_sensor = sequence_loader.get_camera_sensor(camera_id)

            # Collect all image handles with timestamps and restrict to active frames
            timestamped_image_frame_handles: list[
                Tuple[int, ncore.data.CameraSensorProtocol.EncodedImageDataHandleProtocol]
            ] = list(
                zip(
                    camera_sensor.get_frames_timestamps_us(),
                    [camera_sensor.get_frame_handle(i) for i in range(camera_sensor.frames_count)],
                )
            )[camera_frame_range.start : camera_frame_range.stop]

            camera_model_parameters = require_concrete_camera_parameters(camera_sensor.model_parameters)

            run_ego_mask = _run_ego_mask_for_camera(params.ego_mask, params.ego_mask_camera_ids, camera_id)
            if run_ego_mask and not shared_sam2_prepass:
                from apps.av.aux_gen.estimators import PromptlessSAM2EgoMaskEstimator

                sample_indices = _select_ego_mask_sample_indices(
                    [timestamp_us for timestamp_us, _ in timestamped_image_frame_handles],
                    samples_per_second=params.ego_mask_samples_per_second,
                    camera_id=camera_id,
                )
                sampled_handles = [timestamped_image_frame_handles[index] for index in sample_indices]

                ego_mask_estimator = PromptlessSAM2EgoMaskEstimator()
                logging.info(
                    f"Running automatic ego-mask estimation (samples={len(sampled_handles)}, aggregation={params.ego_mask_aggregation_method}) on camera {camera_id}"
                )
                with ScopedTimer(
                    f"aux_gen/ego_mask_estimation/{camera_id}(samples={len(sampled_handles)})",
                    tag=TimingTag.DEFAULT,
                ):
                    run_egomask_estimation(
                        ego_mask_estimator,
                        writer,
                        camera_id,
                        camera_model_parameters,
                        sampled_handles,
                        aggregation_method=params.ego_mask_aggregation_method,
                        visualize=params.visualize,
                    )
                del ego_mask_estimator
                torch.cuda.empty_cache()

            match params.segmentation_backend:
                case "mask2former":
                    if not params.parallel_mode:
                        from apps.av.aux_gen.estimators import Mask2FormerSegmentationEstimator

                        logging.info(
                            f"Running semantic segmentation using mask2former on {len(timestamped_image_frame_handles)} images of camera {camera_id}"
                        )
                        mask2former_segmentation_estimator = Mask2FormerSegmentationEstimator(
                            camera_model_parameters.resolution.tolist(),
                            estimate_logits=params.seg_logits,
                            enable_trt=params.enable_trt,
                            output_uncertainty_mask=params.visualize,
                        )
                        with ScopedTimer(
                            f"aux_gen/mask2former_segmentation/{camera_id}(n={len(timestamped_image_frame_handles)})",
                            tag=TimingTag.DEFAULT,
                        ):
                            _run_serial_mask2former_segmentation(
                                mask2former_segmentation_estimator,
                                writer,
                                camera_id,
                                camera_model_parameters,
                                camera_sensor,
                                timestamped_image_frame_handles,
                                backend=backend,
                                visualize=params.visualize,
                                has_ego_mask=run_ego_mask,
                            )
                        del mask2former_segmentation_estimator
                        torch.cuda.empty_cache()

                case "none":
                    pass

                case _:
                    raise ValueError("segmentation_backend need to be set")

            if depth_generation is not None:
                logging.info(
                    f"Running {depth_generation.description} on "
                    f"{len(timestamped_image_frame_handles)} images of camera {camera_id}"
                )
                depth_estimator = depth_generation.create_estimator()

                with ScopedTimer(
                    f"aux_gen/depth_estimation/{camera_id}(n={len(timestamped_image_frame_handles)})",
                    tag=TimingTag.DEFAULT,
                ):
                    run_depth_estimation(
                        depth_estimator,
                        writer,
                        depth_generation.store_as_png,
                        camera_id,
                        timestamped_image_frame_handles,
                        visualize=params.visualize,
                    )

                del depth_estimator
                torch.cuda.empty_cache()

            match params.dinov2_backend:
                case "none":
                    pass

                case _:
                    from apps.av.aux_gen.estimators import DINOv2Estimator

                    dinov2_estimator = DINOv2Estimator(
                        DINOv2Estimator.resolution_from_dino_width(
                            camera_model_parameters.resolution.tolist(), params.dinov2_width
                        ),
                        backend=params.dinov2_backend,
                    )
                    logging.info(
                        f"Running DINOv2 feature extraction on {len(timestamped_image_frame_handles)} images of camera {camera_id}"
                    )
                    assert dinov2_color_transform is not None
                    with ScopedTimer(
                        f"aux_gen/dinov2_feature_extraction/{camera_id}(n={len(timestamped_image_frame_handles)})",
                        tag=TimingTag.DEFAULT,
                    ):
                        run_dinov2_feature_extraction(
                            dinov2_estimator,
                            writer,
                            camera_id,
                            camera_sensor,
                            timestamped_image_frame_handles,
                            pca=dinov2_pca,
                            color_transform=dinov2_color_transform,
                            backend=backend,
                            visualize=params.visualize,
                        )
                    del dinov2_estimator
                    torch.cuda.empty_cache()

        for lidar_id, lidar_frame_range in sensor_ranges.lidar_frame_ranges.items():
            lidar_sensor = sequence_loader.get_lidar_sensor(lidar_id)

            # Collect timestamps of all frames to process
            frame_timestamps_us = list(
                lidar_sensor.get_frames_timestamps_us()[lidar_frame_range.start : lidar_frame_range.stop]
            )

            if params.lidar_seg_camvis:
                aux_loader: AuxDataCameraSemSegProvider
                if params.segmentation_backend != "none":
                    aux_loader = writer
                    if params.parallel_mode:
                        if cam_seg_aux_loader is None:
                            raise RuntimeError("cam_seg_aux_loader not initialized for parallel mode")
                        aux_loader = cam_seg_aux_loader
                else:
                    aux_loader = AuxShardDataLoader.from_sequence_loader(sequence_loader)
                    # prerequisite for lidar seg: check that all enabled cameras have semantic segmentation
                    if camera_ids_without_semantic_segmentation := [
                        camera_id
                        for camera_id in list(sensor_ranges.camera_frame_ranges.keys())
                        if not aux_loader.has_semantic_segmentation(camera_id)
                    ]:
                        raise RuntimeError(
                            f"Semantic segmentation aux_data not found for the following camera_id: {', '.join(camera_ids_without_semantic_segmentation)}"
                        )

                if params.parallel_mode and cam_seg_output_fp is not None:
                    cam_ids = list(sensor_ranges.camera_frame_ranges.keys())
                    with ScopedTimer(
                        f"aux_gen/lidar_segmentation/{lidar_id}(n={len(frame_timestamps_us)})",
                        tag=TimingTag.DEFAULT,
                    ):
                        run_lidar_segmentation_parallel(
                            opened_sequence=opened_sequence,
                            params=params,
                            output_path=output_path,
                            cam_seg_fp=cam_seg_output_fp,
                            lidar_id=lidar_id,
                            camera_ids=cam_ids,
                            frame_timestamps_us=frame_timestamps_us,
                            num_threads=num_threads,
                        )

                else:
                    from apps.av.aux_gen.estimators import LidarSegmentationByProjectEstimator

                    lidar_seg_estimator = LidarSegmentationByProjectEstimator(
                        sequence_loader,
                        aux_loader,
                        camera_ids=list(sensor_ranges.camera_frame_ranges.keys()),
                        ensemble_cuda=params.lidar_seg_ensemble_cuda,
                    )
                    logging.info(
                        f"Running lidar semantic segmentation and point-in-cameras visibility on {len(frame_timestamps_us)} frames of point cloud {lidar_id}"
                    )
                    with ScopedTimer(
                        f"aux_gen/lidar_segmentation/{lidar_id}(n={len(frame_timestamps_us)})",
                        tag=TimingTag.DEFAULT,
                    ):
                        run_lidar_segmentation(
                            lidar_seg_estimator,
                            sequence_loader,
                            writer,
                            lidar_id=lidar_id,
                            frame_timestamps_us=frame_timestamps_us,
                            visualize=params.visualize,
                        )
                    del lidar_seg_estimator
                    torch.cuda.empty_cache()

        # properly close permanent annotation shard
        writer.finalize()

        # store meta-data of invocation arguments / cluster runtime information (if available)
        if params.store_meta:
            with open(writer.output_dir_path / f"{writer.store_base_name}.aux-meta.json", "w") as meta_json:
                cli_metadata = asdict(params)
                cli_metadata.update(extension_cli_params)
                meta = {"cli": cli_metadata}
                # store workflow runtime information if available
                if (workflow_identifier := os.environ.get("WORKFLOW_IDENTIFIER")) is not None:
                    meta["workflow"] = {
                        "id": workflow_identifier,
                        "date-time": os.environ["WORKFLOW_DATE_TIME"],
                    }
                json.dump(meta, meta_json, indent=4, sort_keys=True)


if __name__ == "__main__":
    cli(show_default=True)
