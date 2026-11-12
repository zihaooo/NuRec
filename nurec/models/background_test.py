# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile

from unittest.mock import patch

import pytest
import torch

from omegaconf import DictConfig

from nurec.config.model import ExtraSignalSkyMlpBackgroundConfig, SkyEnvMapBackgroundConfig
from nurec.config.trainer import TrainerConfig
from nurec.models.background import (
    ExtraSignalSkyMlpBackground,
    SkyEnvMapBackground,
    erode_mask,
    ground_fill,
    swap_sky_for_ground,
)
from nurec.models.background_sampling import sample_cubemap_seamless
from nurec.utils.geometry import cartesian_to_spherical
from nurec.utils.misc import unpack_optional
from nurec.utils.types import ExtraSignal, GaussiansCompositeReturn, GaussiansRenderReturn


class MockTrainerConfig(TrainerConfig):
    def __init__(self):
        super().__init__(
            max_epochs=1,
            check_val_every_n_epoch=1,
            precision="32",
            log_every_n_steps=1,
            enable_progress_bar=False,
            num_sanity_val_steps=0,
        )


def _make_render_results(extras: ExtraSignal | None, *, device: str = "cpu") -> GaussiansCompositeReturn:
    h, w = 4, 4
    return GaussiansCompositeReturn(
        rendered_cam=GaussiansRenderReturn(
            rgb=torch.zeros((h, w, 3), device=device),
            opacity=torch.ones((h, w), device=device),
            distance=torch.zeros((h, w), device=device),
            extra_ray_signals=extras,
        ),
    )


def test_skyenvmapbackground_checkpoint_save_and_load() -> None:
    config = DictConfig(
        {
            "name": "sky-env-map",
            "envmap_type": "cubemap",
            "width": 512,
            "height": 512,
            "composite_in_linear_space": False,
            "min_grad_updates": 1000,
            "should_inpaint": False,
            "inpaint_threshold": 5e-2,
            "inpaint_kernel_size": 10,
        }
    )
    trainer_config = MockTrainerConfig()
    background = SkyEnvMapBackground(SkyEnvMapBackgroundConfig.model_validate(config), trainer_config)
    TARGET_N_GRAD_UPDATES = 42
    background.n_grad_updates = TARGET_N_GRAD_UPDATES
    background.to("cuda")

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "checkpoint.pt")
        torch.save(background.state_dict(), checkpoint_path)
        background_loaded = SkyEnvMapBackground(SkyEnvMapBackgroundConfig.model_validate(config), trainer_config)
        assert background_loaded.n_grad_updates == 1

        background_loaded.load_state_dict(torch.load(checkpoint_path))
        assert background_loaded.n_grad_updates == TARGET_N_GRAD_UPDATES
        # texture_grads is a persistent buffer and must survive the round-trip.
        torch.testing.assert_close(background_loaded.texture_grads.cpu(), background.texture_grads.cpu())


# ------------------------------------------------------------------------------
# SkyEnvMapBackground subclasses gsplat's EnvMapBackground. The tests below pin
# the behaviour required by existing checkpoints:
#   * cubemap sampling uses seamless OpenGL cube-map filtering;
#   * equirectangular sampling is overridden to keep NuRec's historical layout
#     (polar -> width, azimuth -> height), so existing checkpoints keep rendering
#     identically;
#   * compositing and radiance gradient tracking are unchanged.
# The convention/behaviour tests run on CPU; cubemap sampling requires CUDA.
# ------------------------------------------------------------------------------


def _make_sky_env_map_config(
    envmap_type: str,
    size: int = 32,
    *,
    saturate_radiance: bool = False,
    grad_track_interval: int = 1,
) -> SkyEnvMapBackgroundConfig:
    return SkyEnvMapBackgroundConfig.model_validate(
        DictConfig(
            {
                "name": "sky-env-map",
                "envmap_type": envmap_type,
                "width": size,
                "height": size,
                "composite_in_linear_space": False,
                "min_grad_updates": 0,
                "should_inpaint": False,
                "inpaint_threshold": 5e-2,
                "inpaint_kernel_size": 10,
                "saturate_radiance": saturate_radiance,
                "grad_track_interval": grad_track_interval,
            }
        )
    )


def test_skyenvmapbackground_sky_texture_filter_defaults_to_disabled() -> None:
    config = _make_sky_env_map_config("equirectangular")

    assert config.filter_texture_updates_by_sky is False


def _forward_results(n_rays: int, opacity: float, *, device: str = "cpu") -> GaussiansCompositeReturn:
    return GaussiansCompositeReturn(
        rendered_cam=GaussiansRenderReturn(
            rgb=torch.zeros((n_rays, 3), device=device),
            opacity=torch.full((n_rays,), opacity, device=device),
            distance=torch.zeros((n_rays,), device=device),
            extra_ray_signals=None,
        ),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat cubemap sampler is CUDA-only")
def test_skyenvmapbackground_forward_composites_background_behind_render() -> None:
    bg = SkyEnvMapBackground(_make_sky_env_map_config("cubemap"), MockTrainerConfig()).to("cuda")
    with torch.no_grad():
        bg.textures.copy_(torch.rand_like(bg.textures))
    rays_d = torch.nn.functional.normalize(torch.randn(64, 3, device="cuda"), dim=-1)
    opacity = 0.25
    # forward() rotates rays into OpenGL before sampling; mirror that for the reference.
    expected_bg = bg.sample(rays_d @ bg.to_opengl.T)

    results = _forward_results(64, opacity, device="cuda")
    bg.forward(rays_d, results, is_training=False)

    rendered = results.rendered_cam
    assert rendered is not None
    # Over-composite behind a zero render: final = 0 + bg * (1 - opacity).
    torch.testing.assert_close(rendered.rgb, expected_bg * (1.0 - opacity))
    assert rendered.extra_ray_signals is not None
    torch.testing.assert_close(rendered.extra_ray_signals.rgb_background, expected_bg)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat cubemap sampler is CUDA-only")
def test_skyenvmapbackground_tracks_gradients_only_after_warmup() -> None:
    bg = SkyEnvMapBackground(_make_sky_env_map_config("cubemap"), MockTrainerConfig()).to("cuda")
    rays_d = torch.nn.functional.normalize(torch.randn(256, 3, device="cuda"), dim=-1)

    # During warm-up (n_grad_updates <= min_grad_updates) texture_grads stays zero,
    # but the counter still advances.
    bg.min_grad_updates = 5
    bg.n_grad_updates = 1
    bg.forward(rays_d, _forward_results(256, 0.3, device="cuda"), is_training=True)
    assert torch.count_nonzero(bg.texture_grads) == 0
    assert bg.n_grad_updates == 2

    # After warm-up the running-max per-texel gradient magnitude is populated.
    bg.min_grad_updates = 0
    bg.n_grad_updates = 10
    bg.forward(rays_d, _forward_results(256, 0.3, device="cuda"), is_training=True)
    assert torch.count_nonzero(bg.texture_grads) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat cubemap sampler is CUDA-only")
@pytest.mark.parametrize("ray_dtype", [torch.float32, torch.float16])
def test_skyenvmapbackground_direct_coverage_matches_autograd_tracking(ray_dtype: torch.dtype) -> None:
    bg = SkyEnvMapBackground(_make_sky_env_map_config("cubemap", size=16), MockTrainerConfig()).to("cuda")
    rays_d = torch.nn.functional.normalize(torch.randn(512, 3, device="cuda"), dim=-1)
    opacity = torch.linspace(0.0, 0.9, rays_d.shape[0], device="cuda")
    mask = torch.arange(rays_d.shape[0], device="cuda") % 3 != 0
    # Under autocast the rays reach forward in fp16 while the texture is sampled in fp32.
    sampling_rays = (rays_d @ bg.to_opengl.T).to(ray_dtype)
    raw_rgb = bg._sample_raw(sampling_rays)
    masked_rgb = torch.where(mask[:, None], raw_rgb, raw_rgb.detach())
    reference = (
        torch.autograd.grad(
            masked_rgb,
            bg.textures,
            (1 - opacity[:, None]).expand_as(masked_rgb),
        )[0]
        .squeeze(0)
        .mean(dim=-1)
    )
    bg._track_texture_grads(masked_rgb, sampling_rays, opacity, mask, is_training=True)

    torch.testing.assert_close(bg.texture_grads, reference, atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat cubemap sampler is CUDA-only")
def test_skyenvmapbackground_grad_track_interval_uses_absolute_modulo_cadence() -> None:
    bg = SkyEnvMapBackground(
        _make_sky_env_map_config("cubemap", size=8, grad_track_interval=3), MockTrainerConfig()
    ).to("cuda")
    rays_d = torch.nn.functional.normalize(torch.randn(64, 3, device="cuda"), dim=-1)

    for expected_counter in (2, 3):
        bg.forward(rays_d, _forward_results(64, 0.2, device="cuda"), is_training=True)
        assert bg.n_grad_updates == expected_counter
        assert torch.count_nonzero(bg.texture_grads) == 0
    bg.forward(rays_d, _forward_results(64, 0.2, device="cuda"), is_training=True)

    assert bg.n_grad_updates == 4
    assert torch.count_nonzero(bg.texture_grads) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat cubemap sampler is CUDA-only")
@pytest.mark.parametrize(
    ("mode", "envmap_type"),
    [("evaluation", "cubemap"), ("no_grad", "cubemap"), ("frozen", "cubemap"), ("frozen", "equirectangular")],
)
def test_skyenvmapbackground_disabled_tracking_does_not_update(mode: str, envmap_type: str) -> None:
    bg = SkyEnvMapBackground(_make_sky_env_map_config(envmap_type, size=8), MockTrainerConfig()).to("cuda")
    rays_d = torch.nn.functional.normalize(torch.randn(64, 3, device="cuda", requires_grad=mode == "frozen"), dim=-1)
    is_training = mode != "evaluation"
    if mode == "frozen":
        bg.textures.requires_grad_(False)

    context = torch.no_grad() if mode == "no_grad" else torch.enable_grad()
    with context:
        bg.forward(rays_d, _forward_results(64, 0.2, device="cuda"), is_training=is_training)

    assert bg.n_grad_updates == 1
    assert torch.count_nonzero(bg.texture_grads) == 0


def test_skyenvmapbackground_tracks_raw_radiance_gradients() -> None:
    bg = SkyEnvMapBackground(
        _make_sky_env_map_config("equirectangular", saturate_radiance=True),
        MockTrainerConfig(),
    )
    with torch.no_grad():
        bg.textures.fill_(2.0)

    rays_d = torch.nn.functional.normalize(torch.randn(256, 3), dim=-1)
    bg.forward(rays_d, _forward_results(256, 0.3), is_training=True)

    assert torch.count_nonzero(bg.texture_grads) > 0


def test_skyenvmapbackground_blocks_non_sky_texture_gradients() -> None:
    config = _make_sky_env_map_config("equirectangular")
    rays_d = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    filtered_bg = SkyEnvMapBackground(config, MockTrainerConfig())
    filtered_results = _forward_results(2, opacity=0.0)
    filtered_bg.forward(
        rays_d,
        filtered_results,
        is_training=True,
        texture_update_mask=torch.tensor([True, False]),
    )
    assert filtered_results.rendered_cam is not None
    unpack_optional(filtered_results.rendered_cam.rgb).sum().backward()

    sky_only_bg = SkyEnvMapBackground(config, MockTrainerConfig())
    sky_only_results = _forward_results(1, opacity=0.0)
    sky_only_bg.forward(rays_d[:1], sky_only_results, is_training=True)
    assert sky_only_results.rendered_cam is not None
    unpack_optional(sky_only_results.rendered_cam.rgb).sum().backward()

    assert filtered_bg.textures.grad is not None
    assert sky_only_bg.textures.grad is not None
    torch.testing.assert_close(filtered_bg.textures.grad, sky_only_bg.textures.grad)


def test_skyenvmapbackground_equirect_uses_nurec_convention() -> None:
    # NuRec maps polar angle -> width (columns) and azimuth -> height (rows). Pin this
    # by encoding the texture along one axis and checking which spherical angle drives
    # the sampled value. This is the contract that keeps equirect checkpoints valid.
    size = 16
    bg = SkyEnvMapBackground(_make_sky_env_map_config("equirectangular", size), MockTrainerConfig())
    H = W = size

    with torch.no_grad():  # value depends only on the width (column) index
        col = torch.arange(W).float().view(1, 1, W, 1).expand(1, H, W, 3).clone()
        bg.textures.copy_(col)

    # Fixing polar and sweeping azimuth must NOT change a column-encoded texture
    # (azimuth maps to rows).
    phis = torch.linspace(-3.0, 3.0, 6)
    z0 = 0.3
    r = (1.0 - z0**2) ** 0.5
    dirs_azimuth = torch.stack([r * torch.cos(phis), r * torch.sin(phis), torch.full_like(phis, z0)], dim=-1)
    out_azimuth = bg.sample(dirs_azimuth)[:, 0]
    assert out_azimuth.std().item() < 1e-4, "azimuth must not affect a column-encoded texture"

    # Sweeping polar at fixed azimuth MUST change a column-encoded texture.
    zs = torch.linspace(0.8, -0.8, 6)
    dirs_polar = torch.stack([torch.sqrt(1.0 - zs**2), torch.zeros_like(zs), zs], dim=-1)
    out_polar = bg.sample(dirs_polar)[:, 0]
    assert out_polar.std().item() > 1.0, "polar angle must drive a column-encoded texture"


def _smooth_texture(shape: tuple[int, ...]) -> torch.Tensor:
    # Low-frequency, strictly-positive pattern (so relu activation is a no-op and
    # sub-texel interpolation differences between samplers stay negligible).
    *lead, H, W, _ = shape
    ys = torch.linspace(0.0, 1.0, H).view(-1, 1)
    xs = torch.linspace(0.0, 1.0, W).view(1, -1)
    base = 0.5 + 0.35 * torch.sin(2.0 * torch.pi * xs) * torch.cos(2.0 * torch.pi * ys)  # [H, W]
    return base.view(*([1] * len(lead)), H, W, 1).expand(shape).contiguous()


def _equirectangular_reference(bg: SkyEnvMapBackground, rays_d_world: torch.Tensor) -> torch.Tensor:
    """Sample the historical periodic texture layout with direct bilinear interpolation."""
    rotated = rays_d_world @ bg.to_opengl.T
    phi, theta, _ = cartesian_to_spherical(rotated)
    fractional_x = (phi / torch.pi * 2 - 1) % 1
    fractional_y = (theta / torch.pi) % 1
    height, width = bg.textures.shape[1:3]
    x = fractional_x * width - 0.5
    y = fractional_y * height - 0.5
    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x_weight = (x - x0).unsqueeze(-1)
    y_weight = (y - y0).unsqueeze(-1)
    texture = bg.textures[0]
    top = texture[y0 % height, x0 % width] * (1 - x_weight) + texture[y0 % height, (x0 + 1) % width] * x_weight
    bottom = (
        texture[(y0 + 1) % height, x0 % width] * (1 - x_weight)
        + texture[(y0 + 1) % height, (x0 + 1) % width] * x_weight
    )
    return bg.activation(top * (1 - y_weight) + bottom * y_weight)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cubemap sampling requires CUDA")
def test_skyenvmapbackground_cubemap_uses_shared_seamless_sampler() -> None:
    torch.manual_seed(0)
    bg = SkyEnvMapBackground(_make_sky_env_map_config("cubemap", 64), MockTrainerConfig()).to("cuda")
    with torch.no_grad():
        bg.textures.copy_(_smooth_texture(tuple(bg.textures.shape)).to("cuda"))

    rays = torch.nn.functional.normalize(torch.randn(8192, 3, device="cuda"), dim=-1)
    open_gl_rays = rays @ bg.to_opengl.T

    actual = bg.sample(open_gl_rays)
    expected = bg.activation(sample_cubemap_seamless(bg.textures.squeeze(0), open_gl_rays))

    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="equirectangular parity test requires CUDA")
def test_skyenvmapbackground_equirect_matches_periodic_reference() -> None:
    torch.manual_seed(0)
    bg = SkyEnvMapBackground(_make_sky_env_map_config("equirectangular", 128), MockTrainerConfig()).to("cuda")
    with torch.no_grad():
        bg.textures.copy_(_smooth_texture(tuple(bg.textures.shape)).to("cuda"))

    rays = torch.nn.functional.normalize(torch.randn(8192, 3, device="cuda"), dim=-1)
    actual = bg.sample(rays @ bg.to_opengl.T)
    expected = _equirectangular_reference(bg, rays)
    diff = (actual - expected).abs()
    assert diff.mean().item() < 2e-3
    assert torch.quantile(diff.flatten(), 0.99).item() < 1e-2


# ------------------------------------------------------------------------------
# ExtraSignalSkyMlpBackground: training-mode assert + inference-mode early-return.
# These two tests are CPU-only — they bypass tcnn init via `__new__` and only
# need to exercise the missing selected-signal slot branch.
# ------------------------------------------------------------------------------


def test_extra_signal_sky_mlp_background_asserts_when_slot_missing_during_training() -> None:
    bg = ExtraSignalSkyMlpBackground.__new__(ExtraSignalSkyMlpBackground)
    bg.signal_type = "semantic_logits"
    with patch("nurec.models.background.SkyMlpBackground.forward"):
        with pytest.raises(AssertionError, match="during training"):
            bg.forward(
                torch.zeros((16, 3)),
                _make_render_results(extras=ExtraSignal(semantic_logits=None)),
                is_training=True,
            )


def test_extra_signal_sky_mlp_background_does_not_raise_at_inference_when_slot_missing() -> None:
    bg = ExtraSignalSkyMlpBackground.__new__(ExtraSignalSkyMlpBackground)
    bg.signal_type = "semantic_logits"
    with patch("nurec.models.background.SkyMlpBackground.forward"):
        bg.forward(
            torch.zeros((16, 3)),
            _make_render_results(extras=ExtraSignal(semantic_logits=None)),
            is_training=False,
        )


# ------------------------------------------------------------------------------
# Positive path: full CUDA + tcnn instance. With a populated `semantic_logits`
# slot during training, forward must run the MLP and actually modify the slot
# via `extras.semantic_logits += bg_pred * (1 - opacity)`. Catches a regression
# that turns the post-rasterizer composition step into a no-op when it should
# run (which would silently degrade training quality).
# ------------------------------------------------------------------------------


def _make_extra_signal_sky_mlp_background_config(n_output_dims: int) -> ExtraSignalSkyMlpBackgroundConfig:
    return ExtraSignalSkyMlpBackgroundConfig.model_validate(
        DictConfig(
            {
                "name": "extra-signal-sky-mlp",
                "saturate_radiance": False,
                "composite_in_linear_space": False,
                "enable_jit_if_supported": False,
                "seed": 0,
                "dir_encoding_config": {
                    "otype": "HashGrid",
                    "n_levels": 16,
                    "n_features_per_level": 2,
                    "log2_hashmap_size": 19,
                    "base_resolution": 16,
                    "max_resolution": 4096,
                },
                "mlp_network_config": {
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "ReLU",
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                },
                "extra_signal": {
                    "type": "semantic_logits",
                    "n_output_dims": n_output_dims,
                    "mlp_network_config": {
                        "otype": "FullyFusedMLP",
                        "activation": "ReLU",
                        "output_activation": "None",
                        "n_neurons": 64,
                        "n_hidden_layers": 2,
                    },
                },
            }
        )
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="tcnn FullyFusedMLP requires CUDA")
def test_extra_signal_sky_mlp_background_adds_to_semantic_logits_during_training() -> None:
    n_dims = 20  # matches production semantic_logits channel count
    bg = ExtraSignalSkyMlpBackground(
        _make_extra_signal_sky_mlp_background_config(n_dims),
        MockTrainerConfig(),
    ).to("cuda")
    bg.train()

    n_rays = 16
    rays_d = torch.zeros((n_rays, 3), device="cuda")
    rays_d[:, 2] = 1.0  # all rays +z; values don't matter, just need non-degenerate dirs
    semantic_logits = torch.zeros((n_rays, n_dims), device="cuda")
    initial = semantic_logits.clone()
    results = GaussiansCompositeReturn(
        rendered_cam=GaussiansRenderReturn(
            rgb=torch.zeros((n_rays, 3), device="cuda"),
            opacity=torch.full((n_rays,), 0.5, device="cuda"),  # (1 - opacity) = 0.5 so the addend is non-zero
            distance=torch.zeros((n_rays,), device="cuda"),
            extra_ray_signals=ExtraSignal(semantic_logits=semantic_logits),
        ),
    )

    bg.forward(rays_d, results, is_training=True)

    # Forward must modify the slot via the additive composite. If the MLP path
    # were silently skipped (regression), the slot would remain zero.
    rendered_cam = results.rendered_cam
    assert rendered_cam is not None
    extras_after = rendered_cam.extra_ray_signals
    assert extras_after is not None
    semantic_logits_after = extras_after.semantic_logits
    assert semantic_logits_after is not None
    assert not torch.equal(semantic_logits_after, initial)


def test_ground_fill_replaces_only_rays_below_the_horizon() -> None:
    # elevations 45, 0, -0.5, -1, -45 and -90 degrees about +z
    rays_d = torch.tensor([[1.0, 0.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, -0.0087268], [1.0, 0.0, -0.0174551],
                           [1.0, 0.0, -1.0], [0.0, 0.0, -1.0]])
    sky = torch.full((6, 3), 0.25)
    out = ground_fill(rays_d, sky, torch.tensor([0.75, 0.5, 0.5]), up=[0.0, 0.0, 1.0], horizon_deg=0.0, blend_deg=1.0)
    weight = (out[:, 0] - 0.25) / 0.5
    assert torch.allclose(weight, torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0, 1.0]), atol=1e-3)
    assert torch.equal(out[:2], sky[:2])  # the sky at and above the horizon is untouched
    assert torch.allclose(out[3:], torch.tensor([0.75, 0.5, 0.5]).expand(3, 3))


def test_ground_fill_follows_the_up_axis_and_unnormalised_rays() -> None:
    # with y as up, a ray pointing down y is ground and one along z is the horizon
    rays_d = torch.tensor([[0.0, -3.0, 0.0], [0.0, 0.0, 2.0]])
    out = ground_fill(rays_d, torch.zeros(2, 3), torch.ones(3), up=[0.0, 2.0, 0.0], horizon_deg=0.0, blend_deg=1.0)
    assert torch.allclose(out, torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]))


def test_ground_fill_with_a_mask_changes_only_masked_rays() -> None:
    # two rays straight down; only the masked one (the ego hood) is filled
    rays_d = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]])
    out = ground_fill(rays_d, torch.zeros(2, 3), torch.ones(3), up=[0.0, 0.0, 1.0], horizon_deg=0.0, blend_deg=1.0,
                      mask=torch.tensor([True, False]))
    assert torch.equal(out, torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]))


def test_ground_fill_weight_scales_the_fill() -> None:
    # a fractional mask weight moves the colour only that far
    out = ground_fill(torch.tensor([[0.0, 0.0, -1.0]]), torch.zeros(1, 3), torch.ones(3), up=[0.0, 0.0, 1.0],
                      horizon_deg=0.0, blend_deg=1.0, mask=torch.tensor([0.25]))
    assert torch.allclose(out, torch.full((1, 3), 0.25))


def test_erode_mask_shrinks_from_unset_pixels_but_not_from_the_border() -> None:
    mask = torch.zeros(6, 6, dtype=torch.bool)
    mask[3:, :] = True  # a hood along the bottom edge
    eroded = erode_mask(mask, 1)
    assert not eroded[3].any()  # the row next to unset pixels is dropped
    assert eroded[4:].all()  # rows below it survive, including those at the image border
    assert torch.equal(erode_mask(mask, 0), mask)


def test_swap_sky_for_ground_replaces_only_the_sky_share() -> None:
    # foreground 0.2, sky 0.6, opacity 0.5: before post-processing 0.2 + 0.5 * 0.6 = 0.5
    pre = torch.full((1, 3), 0.5)
    sky = torch.full((1, 3), 0.6)
    see_through = torch.full((1, 1), 0.5)
    # identity post-processing: the result is foreground + see-through share of the ground colour
    assert torch.allclose(swap_sky_for_ground(pre, pre, sky, see_through, [1.0, 1.0, 1.0]), torch.full((1, 3), 0.7))
    # a post-processing gain of 1.5 scales the removed sky share with it: 0.75 - 1.5 * 0.3 + 0.5 = 0.8
    assert torch.allclose(swap_sky_for_ground(pre * 1.5, pre, sky, see_through, [1.0, 1.0, 1.0]), torch.full((1, 3), 0.8))
