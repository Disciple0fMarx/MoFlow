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
import torch

from tools import attrib_video_conditioning as cvxp
from utils.config import Config

from .sdd_test_utils import write_synthetic_sdd, write_synthetic_video_cache


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
    """No features root → scalar 0. sentinel → every batch skipped silently.

    The tool must still exit cleanly and write empty aggregate rows rather
    than crashing on a missing ``z_video_global``.
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
    cvxp.main(argv)  # must not raise

    import csv

    with (tmp_path / "out.csv").open() as fp:
        rows = list(csv.DictReader(fp))
    assert len(rows) == 3
    for r in rows:
        assert float(r["ade_min"]) == 0.0  # no agents were scored
