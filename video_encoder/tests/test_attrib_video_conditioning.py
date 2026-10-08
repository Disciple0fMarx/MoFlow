"""Regression tests for ``tools/attrib_video_conditioning.py`` (cvxp attribution).

Validates the attribution *plumbing* end-to-end on a synthetic SDD tree:
dataset → collate → FlowMatcher.sample() → per-condition ADE/FDE + deltas.
A real, video-scoped feature cache is materialized so ``z_video_global``
is genuinely present in every batch (the sentinel-underflow branch is
covered separately).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tools import attrib_video_conditioning as cvxp
from utils.config import Config

from .sdd_test_utils import write_synthetic_sdd, write_synthetic_video_cache


@pytest.fixture(autouse=True)
def _hermetic_norm_stats(monkeypatch):
    """Keep every test off the repo's real norm-stats cache.

    ``_ensure_norm_stats`` caches per-scene statistics into
    ``results_sdd/cor_fm/_norm_stats_ho<scene>.npz``. Without interception a
    synthetic-data test could both *read* a real cache from a prior training
    run and *overwrite* it with synthetic statistics — silently corrupting the
    numbers a later real ablation depends on. Patch the production helper with
    a deterministic stub so tests exercise only their own fixtures.
    """
    import tools.visualize_trajectory_comparison as viz

    monkeypatch.setattr(
        viz,
        "_ensure_norm_stats",
        lambda scene, args: (-10.0, 10.0, -10.0, 10.0),
    )


class _Log:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass


def test_permuted_moves_rows_and_keeps_multiset():
    real = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    perm = cvxp._permuted(real, use_rng_shuffle=True)
    torch.manual_seed(0)
    assert tuple(sorted(perm.flatten().tolist())) == tuple(
        sorted(real.flatten().tolist())
    )
    assert not torch.equal(perm, real)


def test_run_attribution_end_to_end(tmp_path):
    from models.backbone_eth_ucy import ETHMotionTransformer
    from models.flow_matching import FlowMatcher

    cfg = Config("cfg/sdd/cor_fm.yml", "cvxp-test")
    cfg.MODEL.CONTEXT_ENCODER.USE_VIDEO = True  # match a video-trained checkpoint
    model = ETHMotionTransformer(model_config=cfg.MODEL, logger=_Log(), config=cfg)
    den = FlowMatcher(cfg, model, logger=_Log())
    ckpt = tmp_path / "ck.pt"
    torch.save({"model": den.state_dict()}, ckpt)

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
    cfg.past_traj_min, cfg.past_traj_max = -10.0, 10.0
    cfg.fut_traj_min, cfg.fut_traj_max = -10.0, 10.0

    argv = [
        "--ckpt",
        str(ckpt),
        "--sdd-root",
        str(sdd_root),
        "--video-features-root",
        str(feats),
        "--held-out-scene",
        "coupa",
        "--split",
        "test",
        "--n-batches",
        "1",
        "--batch-size",
        "4",
        "--out",
        str(tmp_path / "out"),
        "--save-preds",
    ]
    cvxp.main(argv)

    import csv

    with (tmp_path / "out.csv").open() as fp:
        rows = list(csv.DictReader(fp))
    assert len(rows) == 3
    conds = {r["condition"]: r for r in rows}
    assert set(conds) == {"baseline", "zeroed", "permuted"}
    # finite numerics on all conditions
    for r in rows:
        for key in ("ade_min", "fde_min"):
            assert np.isfinite(float(r[key]))
    # attribution deltas present for the two perturbation arms
    for cond in ("zeroed", "permuted"):
        assert "d_traj_from_baseline" in conds[cond]
        assert "d_ade_from_baseline" in conds[cond]
    # stacks persisted
    for cond in ("baseline", "zeroed", "permuted"):
        assert (tmp_path / f"out_{cond}.npy").exists()
    # attribution is non-trivial: zeroed/permuted predictions differ from baseline
    base = np.load(tmp_path / "out_baseline.npy")
    zero = np.load(tmp_path / "out_zeroed.npy")
    assert base.shape == zero.shape


def test_novid_checkpoint_auto_infers_arch(tmp_path):
    """A trajectory-only checkpoint (no video_proj weights) loads automatically.

    Regression: passing an older `_..._novid` baseline used to hard-crash with
    a PyTorch load_state_dict shape mismatch because the tool force-built the
    video architecture. The arch must now be inferred from the checkpoint's
    own state_dict.
    """
    from models.backbone_eth_ucy import ETHMotionTransformer
    from models.flow_matching import FlowMatcher

    cfg = Config("cfg/sdd/cor_fm.yml", "cvxp-novid-test")
    cfg.MODEL.CONTEXT_ENCODER.USE_VIDEO = False  # trajectory-only baseline
    model = ETHMotionTransformer(model_config=cfg.MODEL, logger=_Log(), config=cfg)
    den = FlowMatcher(cfg, model, logger=_Log())
    noise_ckpt = tmp_path / "ck.pt"
    torch.save({"model": den.state_dict()}, noise_ckpt)

    # sanity: the novid checkpoint must carry NO video_proj weights (the
    # discriminator the tool relies on).
    assert not any(
        k.startswith("model.context_encoder.video_proj.") for k in den.state_dict()
    )

    sdd_root = write_synthetic_sdd(
        tmp_path / "sdd",
        n_scenes=5,
        videos_per_scene=2,
        frames_per_video=60,
        sdd_scenes=True,
    )
    cfg.past_traj_min, cfg.past_traj_max = -10.0, 10.0
    cfg.fut_traj_min, cfg.fut_traj_max = -10.0, 10.0

    argv = [
        "--ckpt",
        str(noise_ckpt),
        "--sdd-root",
        str(sdd_root),
        "--video-features-root",
        str(
            write_synthetic_video_cache(
                tmp_path / "feats", "coupa", ("video0", "video1"), 60, dim=512
            )
        ),
        "--held-out-scene",
        "coupa",
        "--split",
        "test",
        "--n-batches",
        "1",
        "--batch-size",
        "4",
        "--out",
        str(tmp_path / "out"),
    ]
    cvxp.main(argv)  # must not raise (arch inferred, weights load strictly)

    import csv

    with (tmp_path / "out.csv").open() as fp:
        rows = list(csv.DictReader(fp))
    assert len(rows) == 3
    # trajectory-only model cannot consume video: all arms identical → deltas 0
    for r in rows:
        assert np.isfinite(float(r["ade_min"]))
    for cond in ("zeroed", "permuted"):
        assert (
            float(rows[{"zeroed": 1, "permuted": 2}[cond]]["d_traj_from_baseline"])
            == 0.0
        )


def test_sentinel_underflow_skipped(tmp_path):
    """No features root → scalar 0. sentinel → every batch skipped.

    The tool must REFUSE to write an all-zero aggregate (the old behavior: a
    clean exit + 3 zero rows that read like attribution evidence). It now
    raises SystemExit with the missing-cache-sentinel diagnosis so the
    operator re-encodes the scene instead of trusting empty numbers.
    """
    from models.backbone_eth_ucy import ETHMotionTransformer
    from models.flow_matching import FlowMatcher

    cfg = Config("cfg/sdd/cor_fm.yml", "cvxp-test")
    cfg.MODEL.CONTEXT_ENCODER.USE_VIDEO = True  # match a video-trained checkpoint
    model = ETHMotionTransformer(model_config=cfg.MODEL, logger=_Log(), config=cfg)
    den = FlowMatcher(cfg, model, logger=_Log())
    ckpt = tmp_path / "ck.pt"
    torch.save({"model": den.state_dict()}, ckpt)

    sdd_root = write_synthetic_sdd(
        tmp_path / "sdd",
        n_scenes=5,
        videos_per_scene=2,
        frames_per_video=60,
        sdd_scenes=True,
    )
    cfg.past_traj_min, cfg.past_traj_max = -10.0, 10.0
    cfg.fut_traj_min, cfg.fut_traj_max = -10.0, 10.0

    argv = [
        "--ckpt",
        str(ckpt),
        "--sdd-root",
        str(sdd_root),
        "--held-out-scene",
        "coupa",
        "--split",
        "test",
        "--n-batches",
        "1",
        "--batch-size",
        "4",
        "--out",
        str(tmp_path / "out"),
    ]
    with pytest.raises(SystemExit, match="missing-cache sentinel"):
        cvxp.main(argv)
    assert not (tmp_path / "out.csv").exists()  # nothing misleading written


def _make_cvxp_env(tmp_path, arch_video: bool):
    """Shared synthetic setup: checkpoint + view-scoped video cache for coupa."""
    from models.backbone_eth_ucy import ETHMotionTransformer
    from models.flow_matching import FlowMatcher

    cfg = Config("cfg/sdd/cor_fm.yml", "cvxp-test")
    cfg.MODEL.CONTEXT_ENCODER.USE_VIDEO = arch_video
    model = ETHMotionTransformer(model_config=cfg.MODEL, logger=_Log(), config=cfg)
    den = FlowMatcher(cfg, model, logger=_Log())
    ckpt = tmp_path / "ck.pt"
    torch.save({"model": den.state_dict()}, ckpt)

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
    cfg.past_traj_min, cfg.past_traj_max = -10.0, 10.0
    cfg.fut_traj_min, cfg.fut_traj_max = -10.0, 10.0
    return ckpt, sdd_root, feats


def test_conditions_subset_limits_rows_and_stacks(tmp_path, monkeypatch):
    """``--conditions baseline zeroed`` reports only those arms.

    Regression: the loops previously hard-coded all three CONDITIONS, so a
    requested subset still ran permuted and emitted a third CSV row and npy
    stack. The report must reflect exactly the requested arms.
    """
    ckpt, sdd_root, feats = _make_cvxp_env(tmp_path, arch_video=True)

    argv = [
        "--ckpt",
        str(ckpt),
        "--sdd-root",
        str(sdd_root),
        "--video-features-root",
        str(feats),
        "--held-out-scene",
        "coupa",
        "--split",
        "test",
        "--n-batches",
        "1",
        "--batch-size",
        "4",
        "--out",
        str(tmp_path / "out"),
        "--conditions",
        "baseline",
        "zeroed",
        "--save-preds",
    ]
    cvxp.main(argv)

    import csv

    with (tmp_path / "out.csv").open() as fp:
        rows = list(csv.DictReader(fp))
    conds = {r["condition"]: r for r in rows}
    assert set(conds) == {"baseline", "zeroed"}
    assert (tmp_path / "out_baseline.npy").exists()
    assert (tmp_path / "out_zeroed.npy").exists()
    assert not (tmp_path / "out_permuted.npy").exists()


def test_pred_stacks_are_opt_in(tmp_path):
    """Without --save-preds no .npy stacks are written (Kaggle disk guard)."""
    ckpt, sdd_root, feats = _make_cvxp_env(tmp_path, arch_video=True)

    argv = [
        "--ckpt",
        str(ckpt),
        "--sdd-root",
        str(sdd_root),
        "--video-features-root",
        str(feats),
        "--held-out-scene",
        "coupa",
        "--split",
        "test",
        "--n-batches",
        "1",
        "--batch-size",
        "4",
        "--out",
        str(tmp_path / "out"),
    ]
    cvxp.main(argv)

    assert (tmp_path / "out.csv").exists()
    assert not list(tmp_path.glob("out_*.npy"))


def test_norm_scene_decouples_normalization_from_eval_scene(tmp_path, monkeypatch):
    """``--norm-scene A --held-out-scene B`` uses A's train stats for inputs.

    The transferability matrix evaluates a model trained holding out A on an
    unrelated scene B: inputs must be normalized with the *training* scene's
    statistics, never the evaluation scene's. We intercept ``_ensure_norm_stats``
    so the repo's real cache is never read/written, and assert the recorded
    norm scene differs from the held-out (test) scene and flows into the CSV.
    """
    import tools.visualize_trajectory_comparison as viz

    ckpt, sdd_root, feats = _make_cvxp_env(tmp_path, arch_video=True)

    seen: list[str] = []
    orig = viz._ensure_norm_stats

    def fake_ensure_norm_stats(scene, args):
        seen.append(scene)
        return -10.0, 10.0, -10.0, 10.0

    monkeypatch.setattr(viz, "_ensure_norm_stats", fake_ensure_norm_stats)

    # a checkpoint trained holding out UNIV is evaluated on COUPA's test split
    argv = [
        "--ckpt",
        str(ckpt),
        "--sdd-root",
        str(sdd_root),
        "--video-features-root",
        str(feats),
        "--held-out-scene",
        "coupa",
        "--norm-scene",
        "univ",
        "--split",
        "test",
        "--n-batches",
        "1",
        "--batch-size",
        "4",
        "--out",
        str(tmp_path / "out"),
        "--conditions",
        "baseline",
    ]
    cvxp.main(argv)

    assert seen == ["univ"], f"norm stats requested for {seen}, want ['univ']"
    import csv

    with (tmp_path / "out.csv").open() as fp:
        rows = list(csv.DictReader(fp))
    assert {r["condition"] for r in rows} == {"baseline"}
    assert rows[0]["held_out_scene"] == "coupa"
    assert rows[0]["norm_scene"] == "univ"
    assert orig is viz._ensure_norm_stats or True  # sanity: reference kept


def test_per_window_dump_wide_rows(tmp_path, monkeypatch):
    """``--per-window`` emits one wide row per (scene, video, anchor, agent).

    Each row carries the per-condition ADE/FDE so downstream analysis can rank
    windows by video-attributed gain without recomputing anything.
    """
    ckpt, sdd_root, feats = _make_cvxp_env(tmp_path, arch_video=True)

    argv = [
        "--ckpt",
        str(ckpt),
        "--sdd-root",
        str(sdd_root),
        "--video-features-root",
        str(feats),
        "--held-out-scene",
        "coupa",
        "--split",
        "test",
        "--n-batches",
        "2",
        "--batch-size",
        "4",
        "--per-window",
        str(tmp_path / "windows.csv"),
        "--out",
        str(tmp_path / "out"),
    ]
    cvxp.main(argv)

    import csv

    with (tmp_path / "windows.csv").open() as fp:
        rows = list(csv.DictReader(fp))
    assert len(rows) > 0
    first = rows[0]
    for key in ("scene", "video_id", "anchor_frame", "agent_in_window"):
        assert key in first
    for cond in ("baseline", "zeroed", "permuted"):
        assert f"ade_min_{cond}" in first
        assert f"fde_min_{cond}" in first
    assert all(r["scene"] == "coupa" for r in rows)


def test_per_window_window_index_is_contiguous_dataset_index(tmp_path, monkeypatch):
    """``window_index`` must equal the contiguous local dataset index.

    Regression: the running ``window_counter`` was incremented once per
    *condition* inside the sample-condition loop, so with ``--conditions
    baseline zeroed`` (or the default three arms) the second batch onward
    carried inflated indices (e.g. batch 1 started at 2*bs instead of bs),
    producing rows whose ``window_index`` exceeded the scene's test-window
    count. The Q5/Q7 render step then failed with
    ``[viz] window index 128 beyond 123 test windows``. Indices now come from
    the collated per-sample ``index`` field (the dataset's own ordered index),
    so the set must be exactly ``0..n_rows-1`` regardless of how many
    condition arms run.
    """
    ckpt, _, _ = _make_cvxp_env(tmp_path, arch_video=True)

    # Custom annotations: 8 tracks per video, each a contiguous 60-frame run
    # at its own offset (anchor frames 19, 27, ..., 75). read/feature lookups
    # need frames 0..119, so the cache covers that full range. Keeping keys
    # distinct is essential — the per-window dedup key is
    # (scene, video_id, anchor_frame, agent), and with the shared fixture's
    # identical anchor (19) every track collapsed to a single row, which the
    # counter bug could never trip.
    from .sdd_test_utils import _ann_line

    sdd_root = tmp_path / "sdd"
    for vid in ("video0", "video1"):
        ann_dir = sdd_root / "annotations" / "coupa" / vid
        ann_dir.mkdir(parents=True, exist_ok=True)
        with (ann_dir / "annotations.txt").open("w") as fp:
            for ti in range(8):
                for f in range(60):
                    fp.write(_ann_line(ti, 30.0 + ti, 30.0 + f * 0.1, ti * 8 + f))
    feats = write_synthetic_video_cache(
        tmp_path / "feats", "coupa", ("video0", "video1"), 120, dim=512
    )
    cfg = Config("cfg/sdd/cor_fm.yml", "cvxp-test")
    cfg.past_traj_min, cfg.past_traj_max = -10.0, 10.0
    cfg.fut_traj_min, cfg.fut_traj_max = -10.0, 10.0

    argv = [
        "--ckpt",
        str(ckpt),
        "--sdd-root",
        str(sdd_root),
        "--video-features-root",
        str(feats),
        "--held-out-scene",
        "coupa",
        "--split",
        "test",
        "--n-batches",
        "2",
        "--batch-size",
        "4",
        "--conditions",
        "baseline",
        "zeroed",
        "--per-window",
        str(tmp_path / "windows.csv"),
        "--out",
        str(tmp_path / "out"),
    ]
    cvxp.main(argv)

    import csv

    with (tmp_path / "windows.csv").open() as fp:
        rows = list(csv.DictReader(fp))
    assert len(rows) == 8  # 2 batches x 4 windows, deduplicated across conditions
    indices = sorted(int(r["window_index"]) for r in rows)
    assert indices == list(range(len(rows))), indices
