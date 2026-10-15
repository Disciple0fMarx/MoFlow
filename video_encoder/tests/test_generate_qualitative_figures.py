"""Regression tests for ``tools/generate_qualitative_figures.py``.

Hermetic end-to-end: a synthetic SDD tree + synthetic video-feature cache, a
random-weight FlowMatcher checkpoint dropped into the *convention* run dirs
(``_SDD_ho<scene>_<novid|vmstatic|vmfull>[_vid<video0>]/models/checkpoint_best.pt``),
and a monkeypatched frame lookup + norm stats. No ImageNet download, no real
video decode, no writes outside ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import tools.generate_qualitative_figures as q
from utils.config import Config
from video_encoder.tests.sdd_test_utils import (
    write_synthetic_sdd,
    write_synthetic_video_cache,
)


class _Log:
    def info(self, *a, **k) -> None:  # noqa: ANN002
        pass

    def warning(self, *a, **k) -> None:  # noqa: ANN002
        pass


@pytest.fixture(autouse=True)
def _hermetic_norm_stats(monkeypatch):
    import tools.visualize_trajectory_comparison as viz

    monkeypatch.setattr(
        viz, "_ensure_norm_stats", lambda scene, args: (-10.0, 10.0, -10.0, 10.0)
    )


def _fake_anchor_frame(root, scene, video_id, frame_idx, override=None):
    h, w = 90, 120
    r = np.repeat(np.linspace(0, 255, w, dtype=np.float32)[None, :, None], h, 0)
    g = np.repeat((np.arange(h) * 255 / h)[:, None, None], w, 1)
    b = np.full((h, w, 1), 100, dtype=np.float32)
    img = np.concatenate([r, g, b], axis=-1).astype(np.uint8)
    return img, (0.0, float(w), float(h), 0.0), "synthetic"


def _save_checkpoint(path: Path, use_video: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = Config("cfg/sdd/cor_fm.yml", "qual-test")
    cfg.MODEL.CONTEXT_ENCODER.USE_VIDEO = use_video
    model = q.ETHMotionTransformer(model_config=cfg.MODEL, logger=_Log(), config=cfg)
    denoiser = q.FlowMatcher(cfg, model, logger=_Log())
    torch.save({"model": denoiser.state_dict()}, path)


def _stage_run_dirs(results_root: Path, scene: str, video_id: str | None) -> None:
    suf = f"_vid{video_id}" if video_id else ""
    for variant, suffix in q.VARIANT_SUFFIX.items():
        ckpt = (
            results_root
            / f"_SDD_ho{scene}_{suffix}{suf}"
            / "models"
            / "checkpoint_best.pt"
        )
        _save_checkpoint(ckpt, use_video=q.VARIANT_USE_VIDEO[variant])


def test_variant_checkpoint_resolution(tmp_path, monkeypatch):
    monkeypatch.setattr(q, "RESULTS_DIR", tmp_path)
    _stage_run_dirs(tmp_path, "coupa", None)
    _stage_run_dirs(tmp_path, "coupa", "video0")

    assert (
        q.variant_checkpoint("coupa", "off", None).parent.parent.name
        == "_SDD_hocoupa_novid"
    )
    assert (
        q.variant_checkpoint("coupa", "full", "video0").parent.parent.name
        == "_SDD_hocoupa_vmfull_vidvideo0"
    )
    # missing run dir resolves to the (nonexistent) canonical best path
    missing = q.variant_checkpoint("coupa", "static", "videoX")
    assert missing.name == "checkpoint_best.pt"
    assert not missing.exists()


def test_select_window_indices():
    import argparse

    a = argparse.Namespace(window_indices=None, num_samples=5)
    assert q.select_window_indices(100, a) == [0, 25, 50, 74, 99]
    assert q.select_window_indices(3, a) == [0, 1, 2]

    a2 = argparse.Namespace(window_indices=[2, 99, 5], num_samples=5)
    assert q.select_window_indices(10, a2) == [2, 5]  # 99 clamped out, sorted


def test_observed_frame_ids():
    assert q.observed_frame_ids(10) == [3, 4, 5, 6, 7, 8, 9, 10]
    # clamped at zero near the start of a video
    assert q.observed_frame_ids(2) == [0, 0, 0, 0, 0, 0, 1, 2]


def test_main_end_to_end(tmp_path, monkeypatch):
    scene = "coupa"
    results_root = tmp_path / "results"
    _stage_run_dirs(results_root, scene, None)
    _stage_run_dirs(results_root, scene, "video0")
    monkeypatch.setattr(q, "RESULTS_DIR", results_root)
    monkeypatch.setattr(q, "load_anchor_frame", _fake_anchor_frame)

    sdd_root = write_synthetic_sdd(
        tmp_path / "sdd",
        n_scenes=5,
        videos_per_scene=2,
        frames_per_video=60,
        sdd_scenes=True,
    )
    feats = write_synthetic_video_cache(
        tmp_path / "feats", scene, ("video0", "video1"), 60, dim=512
    )
    out_dir = tmp_path / "out"

    q.main(
        [
            "--scenes",
            scene,
            "--protocol",
            "both",
            "--variants",
            "off",
            "static",
            "full",
            "--num-samples",
            "1",
            "--sampling_steps",
            "3",
            "--seed",
            "42",
            "--no-pretrained",
            "--sdd-root",
            str(sdd_root),
            "--video-features-root",
            str(feats),
            "--out-dir",
            str(out_dir),
        ]
    )

    # global protocol: 1 sample, 5 panels (off x1, static x2, full x2)
    gdir = out_dir / "global" / scene
    assert (out_dir / "global" / "manifest.csv").exists()
    assert (gdir / "sample_001_no_video.png").exists()
    assert (gdir / "sample_001_static_video.png").exists()
    assert (gdir / "sample_001_static_gradcam.png").exists()
    assert (gdir / "sample_001_full_video.png").exists()
    assert (gdir / "sample_001_full_gradcam.png").exists()

    # video0 protocol writes into <scene>_video0/
    vdir = out_dir / "video0" / f"{scene}_video0"
    assert (out_dir / "video0" / "manifest.csv").exists()
    assert (vdir / "sample_001_no_video.png").exists()
    assert (vdir / "sample_001_full_gradcam.png").exists()

    # every panel is a real 300-DPI PNG of non-trivial size
    from PIL import Image

    for png in sorted(gdir.glob("*.png")):
        with Image.open(png) as im:
            assert im.width > 100 and im.height > 100

    extra = sorted(p.name for p in gdir.glob("sample_001_*.png"))
    assert extra == [
        "sample_001_full_gradcam.png",
        "sample_001_full_video.png",
        "sample_001_no_video.png",
        "sample_001_static_gradcam.png",
        "sample_001_static_video.png",
    ]


def test_main_skips_missing_variant(tmp_path, monkeypatch, capsys):
    scene = "coupa"
    results_root = tmp_path / "results"
    _stage_run_dirs(results_root, scene, None)
    # remove static + full checkpoints to emulate the Q6 cleanup
    for suffix in ("vmstatic", "vmfull"):
        d = results_root / f"_SDD_ho{scene}_{suffix}" / "models"
        for f in d.glob("checkpoint_*.pt"):
            f.unlink()
    monkeypatch.setattr(q, "RESULTS_DIR", results_root)
    monkeypatch.setattr(q, "load_anchor_frame", _fake_anchor_frame)

    sdd_root = write_synthetic_sdd(
        tmp_path / "sdd",
        n_scenes=5,
        videos_per_scene=2,
        frames_per_video=60,
        sdd_scenes=True,
    )
    feats = write_synthetic_video_cache(
        tmp_path / "feats", scene, ("video0", "video1"), 60, dim=512
    )
    out_dir = tmp_path / "out"

    q.main(
        [
            "--scenes",
            scene,
            "--protocol",
            "global",
            "--num-samples",
            "1",
            "--sampling_steps",
            "3",
            "--no-pretrained",
            "--sdd-root",
            str(sdd_root),
            "--video-features-root",
            str(feats),
            "--out-dir",
            str(out_dir),
        ]
    )

    gdir = out_dir / "global" / scene
    # off survived; static/full were skipped without crashing
    assert (gdir / "sample_001_no_video.png").exists()
    assert not (gdir / "sample_001_full_video.png").exists()
    captured = capsys.readouterr().out
    assert "SKIP global/coupa/static" in captured
    assert "SKIP global/coupa/full" in captured
