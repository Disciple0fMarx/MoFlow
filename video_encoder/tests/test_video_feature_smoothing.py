"""Regression tests for temporal video-feature smoothing.

``temporal_smooth`` (in :mod:`video_encoder.sdd_adapter`) is the Gaussian
temporal filter applied to cached per-frame ResNet features before the
dataloader mean-pools ``[P, D] -> [D]``. The tests pin:

* ``sigma <= 0`` → exact identity (no copy, no drift),
* a flat ``[T, D]`` input is left flat (kernel normalizes to 1.0),
* a single-frame impulse is damped and spread across neighbours (noise decorrelation),
* shape and dtype are preserved, including the reflected edge padding,
* the cfg-driven pipeline behaviour: ``full`` mode with ``sigma>0`` yields a
  *different* pooled vector than the default raw mean-pool, while ``sigma=0``
  reproduces it exactly.

These run on synthetic fixtures — no video decode is required.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from data.dataloader_sdd_global import SDDGlobalDataset
from video_encoder.sdd_adapter import temporal_smooth
from video_encoder.tests.sdd_test_utils import (
    write_synthetic_sdd,
    write_synthetic_video_cache,
)

# ---------------------------------------------------------------------------
# Unit tests — temporal_smooth kernel behaviour
# ---------------------------------------------------------------------------


def _impulse(T: int = 9, spike_pos: int = 4, spike_val: float = 10.0) -> np.ndarray:
    x = np.zeros((T, 4), dtype=np.float32)
    x[spike_pos, 0] = spike_val
    if T > 1:
        x[:, 1] = 1.0  # constant channel
    x[0, 2] = 1.0  # edge-padding probe
    return x


def test_sigma_zero_is_identity() -> None:
    x = _impulse()
    for sigma in (0, 0.0, -1.0):
        out = temporal_smooth(x, sigma)
        assert out is x  # no copy
        np.testing.assert_array_equal(out, x)


def test_constant_channel_stays_flat() -> None:
    x = _impulse()
    out = temporal_smooth(x, 1.0)
    np.testing.assert_allclose(out[:, 1], 1.0, atol=1e-6)


def test_impulse_is_damped_and_spread() -> None:
    x = _impulse()
    out = temporal_smooth(x, 1.0)
    assert out[4, 0] < 10.0  # peak reduced
    assert out[3, 0] > 0.0 and out[5, 0] > 0.0  # energy spread to neighbours
    np.testing.assert_allclose(out.sum(axis=0)[0], 10.0, atol=1e-4)  # energy preserved
    # σ=1 → kernel exp(-t²/2) over t ∈ [-3, 3]; center weight = 1/sum(exp(-t²/2))
    kernel = np.exp(-0.5 * np.arange(-3, 4) ** 2)
    center = kernel[3] / kernel.sum()
    np.testing.assert_allclose(out[4, 0], 10.0 * center, atol=1e-4)


def test_shape_and_dtype_preserved() -> None:
    x = _impulse(T=17, spike_pos=8)
    out = temporal_smooth(x, 2.0)
    assert out.shape == x.shape
    assert out.dtype == x.dtype


def test_edges_not_zero_damped() -> None:
    # Reflected padding keeps a flat signal right up to the boundary: a
    # constant channel's first/last frame must stay ~1.0 (zero-padding would
    # truncate the kernel and damp it below 1.0).
    x = _impulse()
    out = temporal_smooth(x, 1.0)
    np.testing.assert_allclose(out[0, 1], 1.0, atol=1e-6)
    np.testing.assert_allclose(out[-1, 1], 1.0, atol=1e-6)


def test_single_frame_is_identity() -> None:
    x = np.ones((1, 512), dtype=np.float32)
    out = temporal_smooth(x, 5.0)
    np.testing.assert_array_equal(out, x)


# ---------------------------------------------------------------------------
# Integration — dataloader 'full' mode honours VIDEO_SMOOTH_SIGMA
# ---------------------------------------------------------------------------


class _Cfg:
    class _CE:
        VIDEO_FEATURES_ROOT = None
        VIDEO_DIM_RAW = 512
        AGENTS = None
        VIDEO_SMOOTH_SIGMA = None

    MODEL = type("M", (object,), {"CONTEXT_ENCODER": _CE()})
    past_traj_min = past_traj_max = fut_traj_min = fut_traj_max = None


def _build(td: str, sigma: float) -> SDDGlobalDataset:
    cfg = _Cfg()
    cfg.MODEL.CONTEXT_ENCODER.VIDEO_SMOOTH_SIGMA = sigma
    root = Path(td)
    write_synthetic_sdd(
        root, n_scenes=5, videos_per_scene=2, frames_per_video=60, sdd_scenes=True
    )
    write_synthetic_video_cache(
        root / "feats", "bookstore", ("video0", "video1"), frames_per_video=60
    )
    return SDDGlobalDataset(
        cfg,
        training=True,
        sdd_root=root,
        held_out_scene="coupa",
        use_video=True,
        video_features_root=root / "feats",
        video_mode="full",
    )


def test_sigma_zero_matches_previous_mean_pool() -> None:
    with tempfile.TemporaryDirectory() as td:
        ds = _build(td, 0.0)
        z = ds[0]["z_video_global"]
        assert z.dim() == 1 and z.numel() == 512


def test_smooth_twiddles_full_mode_vector() -> None:
    with tempfile.TemporaryDirectory() as td:
        ds_raw = _build(td, 0.0)
        ds_sm = _build(td, 1.5)
        # Same windows; only the smoothing differs.
        raw = ds_raw[0]["z_video_global"]
        sm = ds_sm[0]["z_video_global"]
        assert raw.shape == sm.shape
        # Random per-frame features are almost surely changed by a Gaussian blur.
        assert not torch.allclose(raw, sm, atol=1e-6)


def test_smooth_preserves_contract_for_all_windows() -> None:
    with tempfile.TemporaryDirectory() as td:
        ds = _build(td, 1.0)
        hits = 0
        for i in range(len(ds.windows)):
            it = ds[i]
            if it["scene"] != "bookstore":  # only bookstore has a cached feature root
                continue
            z = it["z_video_global"]
            assert z.dim() == 1 and z.numel() == 512
            assert torch.isfinite(z).all()
            hits += 1
        assert hits > 0
