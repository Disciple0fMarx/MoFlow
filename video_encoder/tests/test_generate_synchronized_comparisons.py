"""Regression tests for ``tools/generate_synchronized_comparisons.py``.

Fully hermetic: synthetic SDD tree + synthetic video-feature cache, random-weight
FlowMatcher checkpoints in the convention run dirs, and a monkeypatched frame
lookup + norm stats. The synchronized tool imports helpers from
``generate_qualitative_figures``, so the anchor-frame patch must cover both
namespaces (the helper module decodes frames, the new module draws backgrounds).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import tools.generate_qualitative_figures as q
import tools.generate_synchronized_comparisons as sync
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
    cfg = Config("cfg/sdd/cor_fm.yml", "sync-test")
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


def _bundle(points: list[list[float]], ade: float = 1.0) -> dict:
    arr = np.asarray(points, dtype=float)
    return {
        "obs": arr,
        "obs_abs": arr,
        "gt": arr,
        "preds": arr[None],
        "best": arr,
        "best_idx": 0,
        "ade": ade,
    }


def test_shared_limits_unions_every_variant():
    bundles = {
        "off": _bundle([[0, 0], [10, 10]]),
        "static": _bundle([[0, 0], [20, 5]]),
        "full": _bundle([[-5, 0], [10, 20]]),
    }
    (x0, x1), (y0, y1) = sync.shared_limits(bundles, None, None, margin=0.0)
    assert x0 <= -5 and x1 >= 20
    assert y0 <= 0 and y1 >= 20
    assert y1 > y0  # no background → normal (non-inverted) Y


def test_shared_limits_clamps_and_inverts_with_background():
    bundles = {"off": _bundle([[-50, -50], [400, 400]])}
    bg = np.zeros((90, 120, 3), dtype=np.uint8)
    (x0, x1), (y0, y1) = sync.shared_limits(
        bundles, bg, (0.0, 120.0, 90.0, 0.0), margin=50.0
    )
    assert (x0, x1) == (0.0, 120.0)
    assert (y0, y1) == (90.0, 0.0)  # inverted image coordinates


def test_main_end_to_end(tmp_path, monkeypatch):
    scene = "coupa"
    results_root = tmp_path / "results"
    _stage_run_dirs(results_root, scene, None)
    _stage_run_dirs(results_root, scene, "video0")
    monkeypatch.setattr(q, "RESULTS_DIR", results_root)
    monkeypatch.setattr(q, "load_anchor_frame", _fake_anchor_frame)
    monkeypatch.setattr(sync, "load_anchor_frame", _fake_anchor_frame)

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

    sync.main(
        [
            "--scenes",
            scene,
            "--protocol",
            "both",
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

    import json

    from PIL import Image

    gdir = out_dir / "global" / scene / "sample_001"
    vdir = out_dir / "video0" / f"{scene}_video0" / "sample_001"
    for d in (gdir, vdir):
        png = d / "synchronized_comparison.png"
        assert png.exists()
        with Image.open(png) as im:
            assert im.width > 100 and im.height > 100
        meta = json.loads((d / "metadata.json").read_text())
        assert set(meta["variants"]) == {"off", "static", "full"}
        assert meta["gradcam_variant"] == "full"
        assert meta["cam_available"] is True
        assert meta["agent_index"] == 0
        assert meta["figure"] == "synchronized_comparison.png"

    assert (out_dir / "global" / "manifest.csv").exists()
    assert (out_dir / "video0" / "manifest.csv").exists()


def test_main_skips_scene_when_a_variant_is_missing(tmp_path, monkeypatch, capsys):
    scene = "coupa"
    results_root = tmp_path / "results"
    _stage_run_dirs(results_root, scene, None)
    # emulate the Q6 cleanup: static + full arm checkpoints were deleted
    for suffix in ("vmstatic", "vmfull"):
        d = results_root / f"_SDD_ho{scene}_{suffix}" / "models"
        for f in d.glob("checkpoint_*.pt"):
            f.unlink()
    monkeypatch.setattr(q, "RESULTS_DIR", results_root)
    monkeypatch.setattr(q, "load_anchor_frame", _fake_anchor_frame)
    monkeypatch.setattr(sync, "load_anchor_frame", _fake_anchor_frame)

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

    sync.main(
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

    # No 4-panel figure is faked when an arm is absent.
    assert not (out_dir / "global" / scene / "sample_001").exists()
    captured = capsys.readouterr().out
    assert "SKIP global/coupa/static" in captured
    assert "no checkpoint" in captured


# ---------------------------------------------------------------------------
# Strategic selection (best_gain / worst_gain)
# ---------------------------------------------------------------------------
def _make_args(sdd_root: Path, feats: Path, scene: str) -> object:
    args = sync._parse_args(
        [
            "--scenes",
            scene,
            "--sdd-root",
            str(sdd_root),
            "--video-features-root",
            str(feats),
            "--no-pretrained",
            "--sampling_steps",
            "3",
            "--seed",
            "42",
        ]
    )
    args.video_id = None
    args._protocol = "global"
    args.device = "cpu"
    return args


def _write_metrics_csv(path: Path, dataset, gain_fn) -> None:
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "scene",
                "video_id",
                "anchor_frame",
                "window_index",
                "ade_min_baseline",
                "ade_min_zeroed",
            ],
        )
        writer.writeheader()
        for i in range(len(dataset)):
            gain = float(gain_fn(i))
            writer.writerow(
                {
                    "scene": dataset.windows.scene_id(i),
                    "video_id": dataset.windows.video_id(i),
                    "anchor_frame": int(dataset.windows.anchor_frame(i)),
                    "window_index": i,
                    "ade_min_baseline": f"{1.0:.6f}",
                    "ade_min_zeroed": f"{1.0 + gain:.6f}",
                }
            )


def test_row_gain_supports_both_csv_schemas():
    aggregate = {
        "gain_ade": "0.42",
        "ade_baseline": "1.00",
        "ade_zeroed": "1.42",
    }
    per_window = {"ade_min_baseline": "1.00", "ade_min_zeroed": "1.25"}
    assert sync.row_gain(aggregate) == pytest.approx(0.42)
    assert sync.row_gain(per_window) == pytest.approx(0.25)
    # video hurts ⇒ negative ΔADE
    assert sync.row_gain(
        {"ade_min_baseline": "1.00", "ade_min_zeroed": "0.75"}
    ) == pytest.approx(-0.25)
    assert sync.row_gain({"ade_min_baseline": "nan"}) is None


def test_rank_from_metrics_aligns_by_video_and_anchor(tmp_path, monkeypatch):
    scene = "coupa"
    results_root = tmp_path / "results"
    _stage_run_dirs(results_root, scene, None)
    monkeypatch.setattr(q, "RESULTS_DIR", results_root)

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
    args = _make_args(sdd_root, feats, scene)
    dataset, _ = sync.build_variant_dataset(scene, "off", args)
    assert len(dataset) >= 2

    # Deliberately corrupt window_index so the (video_id, anchor_frame) lookup
    # has to do the alignment.
    chosen = {0: 0.10, len(dataset) - 1: 0.90}
    csv_path = tmp_path / "metrics.csv"
    with csv_path.open("w", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "scene",
                "video_id",
                "anchor_frame",
                "window_index",
                "ade_min_baseline",
                "ade_min_zeroed",
            ],
        )
        writer.writeheader()
        for i, gain in chosen.items():
            writer.writerow(
                {
                    "scene": scene,
                    "video_id": dataset.windows.video_id(i),
                    "anchor_frame": int(dataset.windows.anchor_frame(i)),
                    "window_index": -1,
                    "ade_min_baseline": "1.0",
                    "ade_min_zeroed": f"{1.0 + gain:.6f}",
                }
            )

    args.metrics_csv = str(csv_path)
    ranked, used = sync.rank_from_metrics(scene, args, dataset)
    assert used == [csv_path]
    assert set(ranked) == set(chosen)
    for i, gain in chosen.items():
        assert ranked[i] == pytest.approx(gain, abs=1e-6)


def test_main_best_gain_from_csv(tmp_path, monkeypatch):
    scene = "coupa"
    results_root = tmp_path / "results"
    _stage_run_dirs(results_root, scene, None)
    monkeypatch.setattr(q, "RESULTS_DIR", results_root)
    monkeypatch.setattr(q, "load_anchor_frame", _fake_anchor_frame)
    monkeypatch.setattr(sync, "load_anchor_frame", _fake_anchor_frame)

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
    args = _make_args(sdd_root, feats, scene)
    dataset, _ = sync.build_variant_dataset(scene, "off", args)
    n = len(dataset)

    # Monotone gain so the two highest indices win deterministically.
    csv_path = tmp_path / "metrics.csv"
    _write_metrics_csv(csv_path, dataset, lambda i: (i + 1) / 100.0)

    out_dir = tmp_path / "out"
    sync.main(
        [
            "--scenes",
            scene,
            "--protocol",
            "global",
            "--selection-mode",
            "best_gain",
            "--rank-source",
            "csv",
            "--metrics-csv",
            str(csv_path),
            "--top-k",
            "2",
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

    metas = [
        json.loads(p.read_text())
        for p in sorted((out_dir / "global" / scene).glob("sample_*/metadata.json"))
    ]
    assert len(metas) == 2
    assert [m["window_index"] for m in metas] == [n - 1, n - 2]
    for rank, meta in enumerate(metas, start=1):
        assert meta["selection_mode"] == "best_gain"
        assert meta["rank_source"] == "metrics.csv"
        assert meta["selection_rank"] == rank
        expected = (meta["window_index"] + 1) / 100.0
        assert meta["selection_gain"] == pytest.approx(expected, abs=1e-6)
        # rendered_gain is the exact ΔADE for the drawn panels (not the CSV value)
        assert np.isfinite(meta["rendered_gain"])


def test_main_best_gain_compute_matches_rendered_ade(tmp_path, monkeypatch):
    scene = "coupa"
    results_root = tmp_path / "results"
    _stage_run_dirs(results_root, scene, None)
    monkeypatch.setattr(q, "RESULTS_DIR", results_root)
    monkeypatch.setattr(q, "load_anchor_frame", _fake_anchor_frame)
    monkeypatch.setattr(sync, "load_anchor_frame", _fake_anchor_frame)

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

    sync.main(
        [
            "--scenes",
            scene,
            "--protocol",
            "global",
            "--selection-mode",
            "best_gain",
            "--rank-source",
            "compute",
            "--top-k",
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

    metas = list((out_dir / "global" / scene).glob("sample_*/metadata.json"))
    assert len(metas) == 1
    meta = json.loads(metas[0].read_text())
    assert meta["selection_mode"] == "best_gain"
    assert meta["rank_source"] == "computed"
    rendered_gain = (
        meta["variants"]["off"]["ade_min"] - meta["variants"]["full"]["ade_min"]
    )
    assert meta["rendered_gain"] == pytest.approx(rendered_gain, abs=1e-6)
    # With rank_source=compute the ranking metric and the drawn panels agree.
    assert meta["selection_gain"] == pytest.approx(rendered_gain, abs=1e-6)


def test_metric_paths_keeps_protocols_separate(tmp_path):
    metrics = tmp_path / "q4"
    metrics.mkdir()
    for name in (
        "q4_window_gains.csv",
        "q4_window_gains_vidvideo0.csv",
        "coupa_windows.csv",
        "coupa_windows_vidvideo0.csv",
    ):
        (metrics / name).write_text("scene\n")

    args = _make_args(tmp_path, tmp_path, "coupa")
    args.metrics_dir = str(metrics)
    args.metrics_csv = None
    args.rank_source = "auto"

    args.video_id = None
    assert [p.name for p in sync._metric_paths("coupa", args)] == [
        "q4_window_gains.csv",
        "coupa_windows.csv",
    ]

    args.video_id = "video0"
    assert [p.name for p in sync._metric_paths("coupa", args)] == [
        "q4_window_gains_vidvideo0.csv",
        "coupa_windows_vidvideo0.csv",
    ]
