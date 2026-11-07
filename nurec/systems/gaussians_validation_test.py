# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for distributed validation behavior in ``GaussiansSystem``."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import imageio
import numpy as np
import pytest
import torch

from nurec.config.systems import SystemTestConfig
from nurec.systems.gaussians import GaussiansSystem


def _bare_system() -> GaussiansSystem:
    system = object.__new__(GaussiansSystem)
    torch.nn.Module.__init__(system)
    system.cpu()
    return system


def test_camera_rank_reports_camera_when_another_rank_has_lidar() -> None:
    system = _bare_system()
    batch: Any = SimpleNamespace(data=SimpleNamespace(camera=object()))

    with (
        mock.patch("torch.distributed.is_initialized", return_value=True),
        mock.patch("torch.distributed.get_world_size", return_value=2),
        mock.patch("torch.distributed.all_reduce") as all_reduce,
    ):
        assert system._any_rank_has_camera(batch)

    all_reduce.assert_called_once()


def test_lidar_rank_reports_camera_when_another_rank_has_camera() -> None:
    system = _bare_system()
    batch: Any = SimpleNamespace(data=SimpleNamespace(camera=None))

    def report_camera_peer(has_camera: torch.Tensor, **_kwargs: Any) -> None:
        has_camera.fill_(1)

    with (
        mock.patch("torch.distributed.is_initialized", return_value=True),
        mock.patch("torch.distributed.get_world_size", return_value=2),
        mock.patch("torch.distributed.all_reduce", side_effect=report_camera_peer) as all_reduce,
    ):
        assert system._any_rank_has_camera(batch)

    all_reduce.assert_called_once()


def test_lidar_rank_runs_selected_node_forwards_when_camera_peer_exists() -> None:
    system = _bare_system()
    system.val_out_types = ["rgb"]
    system_config: Any = SimpleNamespace(
        test=SimpleNamespace(
            val_render_selected_nodes={
                "road_only": ["road"],
                "background_only": ["background"],
            }
        )
    )
    system.config = system_config
    system.forward = mock.MagicMock(return_value=SimpleNamespace(rendered_cam=None))
    batch: Any = SimpleNamespace(data=SimpleNamespace(camera=None))

    with mock.patch.object(system, "_any_rank_has_camera", return_value=True):
        system.validation_selected_nodes_step(batch)

    assert system.forward.call_count == 2


def _system_with_test_config(**test_config: Any) -> GaussiansSystem:
    system = _bare_system()
    system_config: Any = SimpleNamespace(test=SimpleNamespace(**test_config))
    system.config = system_config
    return system


def test_result_types_select_the_saved_predictions() -> None:
    flags = {"save_results": True, "save_inputs": False, "save_extra_signals": False}

    assert _system_with_test_config(result_types=None, **flags).get_output_types() == [
        "pred_rgb",
        "pred_distance",
        "pred_normal",
        "pred_opacity",
    ]
    assert _system_with_test_config(result_types=["pred_rgb"], **flags).get_output_types() == ["pred_rgb"]
    assert _system_with_test_config(result_types=["pred_opacity", "pred_rgb"], **flags).get_output_types() == [
        "pred_rgb",
        "pred_opacity",
    ]


def test_test_config_validates_result_types_and_writer_threads() -> None:
    required: dict[str, Any] = {
        "video_fps": 30,
        "track_ply": None,
        "track_orbit": None,
        "track_rotation": None,
        "background_removal": None,
        "val_render_selected_nodes": None,
        "metrics": None,
    }

    config = SystemTestConfig(**required)
    assert config.result_types is None
    assert config.image_writer_threads == 0
    assert SystemTestConfig(result_types=["pred_rgb"], image_writer_threads=4, **required).result_types == ["pred_rgb"]
    with pytest.raises(ValueError):
        SystemTestConfig(result_types=["rgb"], **required)  # type: ignore[list-item]  # deliberately invalid
    with pytest.raises(ValueError):
        SystemTestConfig(image_writer_threads=-1, **required)


def test_threaded_result_pngs_are_identical_to_synchronous_writes(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    images = [rng.integers(0, 256, size=(24, 32, 3), dtype=np.uint8) for _ in range(20)]

    for n_threads in (0, 2):
        system = _system_with_test_config(image_writer_threads=n_threads)
        out_dir = tmp_path / f"threads_{n_threads}"
        out_dir.mkdir()
        buffer = np.empty_like(images[0])  # a caller may reuse its image buffer right after queuing a write
        for i, image in enumerate(images):
            buffer[...] = image
            system._write_result_png(str(out_dir / f"{i:03d}.png"), buffer)
            if n_threads > 0:
                assert len(system._png_writes) <= 4 * n_threads
        system._finish_result_pngs()
        assert system._png_writer is None
        if n_threads > 0:
            assert not system._png_writes

    for i, image in enumerate(images):
        threaded = tmp_path / "threads_2" / f"{i:03d}.png"
        assert threaded.read_bytes() == (tmp_path / "threads_0" / f"{i:03d}.png").read_bytes()
        np.testing.assert_array_equal(imageio.v2.imread(threaded), image)


def test_threaded_result_png_errors_surface_when_finishing(tmp_path: Path) -> None:
    system = _system_with_test_config(image_writer_threads=1)
    system._write_result_png(str(tmp_path / "missing_dir" / "000.png"), np.zeros((4, 4, 3), dtype=np.uint8))

    with pytest.raises(OSError):
        system._finish_result_pngs()


def test_save_videos_waits_for_queued_pngs(tmp_path: Path) -> None:
    system = _system_with_test_config(image_writer_threads=1)
    path = tmp_path / "000.png"
    system._write_result_png(str(path), np.zeros((4, 4, 3), dtype=np.uint8))

    system.save_videos(str(tmp_path), [], [])

    assert path.exists()
    assert system._png_writer is None
    assert not system._png_writes


def test_validation_camera_step_without_ground_truth_saves_the_render_only() -> None:
    system = _bare_system()
    metric_manager = mock.MagicMock()
    save_images = mock.MagicMock()
    system.val_metric_manager = metric_manager
    system.save_images = save_images  # type: ignore[method-assign]
    system.val_out_types = ["pred_rgb"]
    system.val_dir = "val"
    system.val_suffix = []
    labels = SimpleNamespace(rgb=None, get_mask_flags_none=lambda _flags: torch.ones(1, 2, 4, 1, dtype=torch.bool))
    batch: Any = SimpleNamespace(
        data=SimpleNamespace(camera=SimpleNamespace(w=4, h=2, labels=labels, meta=[object()]), idx=7, sequence_id="s"),
        rendering=None,
    )
    rendered: Any = SimpleNamespace(rgb=torch.rand(8, 3))

    system.validation_camera_step(batch, rendered)

    metric_manager.compute.assert_not_called()
    save_images.assert_called_once()
