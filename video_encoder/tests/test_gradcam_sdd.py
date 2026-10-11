"""Regression tests for ``tools/gradcam_sdd.py`` (Grad-CAM scene-video attribution).

Covers the formula-level math (``alpha_k`` = spatial GAP of the gradients,
``CAM = ReLU(sum_k alpha_k A^k)``), the numpy colormap/overlay stack, and an
end-to-end ``main()`` run on a synthetic SDD tree with a random-weight
FlowMatcher (offline: ``--no-pretrained`` backbone, monkeypatched frame
lookup). No ImageNet weights, no real video, no writes outside ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tools import gradcam_sdd as gc
from utils.config import Config

from .sdd_test_utils import write_synthetic_sdd, write_synthetic_video_cache


@pytest.fixture(autouse=True)
def _hermetic_norm_stats(monkeypatch):
    """Keep every test off the repo's real norm-stats cache."""

    import tools.visualize_trajectory_comparison as viz

    monkeypatch.setattr(
        viz,
        "_ensure_norm_stats",
        lambda scene, args: (-10.0, 10.0, -10.0, 10.0),
    )


class _Log:
    def info(self, *a, **k) -> None:  # noqa: ANN002
        pass

    def warning(self, *a, **k) -> None:  # noqa: ANN002
        pass


def test_jet_lut_range_and_coloring():
    lut = gc._jet_lut()
    assert lut.shape == (256, 3)
    assert lut.dtype == np.float32
    assert lut.min() >= 0.0 and lut.max() <= 1.0
    # cold end is blue-ish, hot end is red-ish
    assert lut[0, 2] > lut[0, 0]
    assert lut[255, 0] > lut[255, 2]
    # scalar map through the lut
    cam01 = np.zeros((7, 7), dtype=np.float32)
    cam01[3, 3] = 1.0
    color = gc.CAM_to_color(cam01, lut)
    assert color.shape == (7, 7, 3)
    hot = color[3, 3]
    assert hot[0] > hot[2]  # red-dominant at the peak


def test_overlay_upscales_to_frame_resolution():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, size=(96, 128, 3), dtype=np.uint8)
    cam = np.zeros((7, 7), dtype=np.float32)
    cam[3, 3] = 1.0
    out = gc.overlay_heatmap(frame, cam, alpha=0.5, lut=gc._jet_lut())
    assert out.shape == (96, 128, 3)
    assert out.dtype == np.uint8
    # the peak region is visibly tinted relative to the untouched frame
    assert not np.array_equal(out[40:56, 60:76], frame[40:56, 60:76])


def test_grad_cam_maps_formula():
    """alpha_k == spatial GAP of grads; CAM == ReLU(sum_k alpha_k A^k)."""
    torch.manual_seed(0)
    A = torch.randn(4, 8, 7, 7, requires_grad=True)
    weights = torch.randn(8)
    score = -(A * weights.view(1, 8, 1, 1)).pow(2).sum()

    # reference gradient first (kept alive so the live call below can rerun)
    g = torch.autograd.grad(score, A, retain_graph=True)[0]
    alpha_ref = g.mean(dim=(2, 3), keepdim=True)
    cam_ref = torch.clamp((alpha_ref * A).sum(dim=1), min=0.0)

    alphas, cam = gc.grad_cam_maps(A, score)
    assert alphas.shape == (4, 8, 1, 1)
    assert cam.shape == (4, 7, 7)
    assert (cam >= 0).all()
    assert torch.allclose(alphas, alpha_ref, atol=1e-6)
    assert torch.allclose(cam, cam_ref, atol=1e-5)


def _fake_anchor_frame(root, scene, video_id, frame_idx, override=None):
    """Stand-in for ``load_anchor_frame``: deterministic synthetic frame."""
    h, w = 96, 128
    r = np.linspace(0, 255, w, dtype=np.float32)[None, :, None]
    r = np.repeat(r, h, axis=0)
    g = np.repeat((np.arange(h) * 255 / h)[:, None, None], w, axis=1)
    b = np.full((h, w, 1), 120, dtype=np.float32)
    img = np.concatenate([r, g, b], axis=-1).astype(np.uint8)
    return img, (0.0, float(w), float(h), 0.0), "synthetic"


def test_main_end_to_end(tmp_path, monkeypatch):
    from models.backbone_eth_ucy import ETHMotionTransformer
    from models.flow_matching import FlowMatcher

    cfg = Config("cfg/sdd/cor_fm.yml", "gradcam-test")
    cfg.MODEL.CONTEXT_ENCODER.USE_VIDEO = True  # match a video-trained ckpt
    model = ETHMotionTransformer(model_config=cfg.MODEL, logger=_Log(), config=cfg)
    denoiser = FlowMatcher(cfg, model, logger=_Log())
    ckpt = tmp_path / "ck.pt"
    torch.save({"model": denoiser.state_dict()}, ckpt)

    sdd_root = write_synthetic_sdd(
        tmp_path / "sdd",
        n_scenes=5,
        videos_per_scene=2,
        frames_per_video=60,
        sdd_scenes=True,
    )
    feats = write_synthetic_video_cache(
        tmp_path / "feats", "coupa", ("video0", "video1"), 60, dim=512
    )

    # offline frame lookup: no real videos in the synthetic tree
    monkeypatch.setattr(gc, "load_anchor_frame", _fake_anchor_frame)

    argv = [
        "--ckpt",
        str(ckpt),
        "--sdd-root",
        str(sdd_root),
        "--video-features-root",
        str(feats),
        "--scene",
        "coupa",
        "--window-index",
        "0",
        "--sampling_steps",
        "4",
        "--seed",
        "42",
        "--no-pretrained",
        "--out-dir",
        str(tmp_path / "out"),
    ]
    gc.main(argv)

    out = tmp_path / "out" / "coupa"
    assert (out / "manifest.csv").exists()
    assert (out / "w00000").is_dir()

    overlays = sorted((out / "w00000").glob("*_overlay.jpg"))
    cams = sorted((out / "w00000").glob("*_cam.png"))
    assert len(overlays) == 8  # the 8 observed frames [a-7 .. a]
    assert len(cams) == 8

    art = np.load(out / "w00000" / "_gradcam.npz")
    assert art["cam"].shape == (8, 7, 7)  # SDD_PAST_FRAMES x layer4 7x7
    assert art["alphas"].shape == (8, 512, 1, 1)
    assert art["activations"].shape == (8, 512, 7, 7)
    assert (art["cam"] >= 0).all()
    assert np.isfinite(art["cam"]).all()
    assert np.isfinite(art["score"])
    assert art["past_frame_ids"].shape == (8,)
    assert np.isfinite(art["per_hyp_ade"]).all()

    from PIL import Image

    with Image.open(overlays[0]) as im:
        assert im.size == (128, 96)  # upscaled to the synthetic frame size
