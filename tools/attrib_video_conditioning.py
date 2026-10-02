"""cvxp attribution for the z_video_global conditioning channel.

What this measures
------------------
For a **trained** FlowMatching denoiser, run conditional-velocity eXperiment /
**P**ermutation sampling (:meth:`FlowMatcher.sample`, ``objective='pred_data'``)
over a set of batches while perturbing the global video conditioning
``z_video_global`` that is fed to the ``ETHEncoder``:

* ``baseline`` — the real per-window scene vector.
* ``zeroed``   — the same channel replaced with the zero vector.
* ``permuted`` — the batch's vectors shuffled across samples, so every window
  still sees a *valid* video vector but the wrong one.

Comparing samples across the three conditions attributes how strongly the
trained model actually uses the video channel, and whether the usage is
content-sensitive or effectively a constant prior:

* baseline ≈ zeroed     → the model ignores the video channel entirely.
* zeroed ≃ permuted     → the model relies on "some video" far more than on
  "*this* video" (a scene constant), i.e. per-window content barely matters.
* permuted ≫ zeroed     → the model genuinely consumes per-window content.

The RNG is re-seeded to ``--seed`` immediately before *every* ``sample()`` call,
so the three conditions draw **identical** initial noise and differ only in
``z_video_global`` — trajectory deltas are a clean measure of video influence,
not sampling stochasticity. (This also makes a trajectory-only ``_novid``
checkpoint produce byte-identical predictions across arms, ``d_traj = 0``.)

Metrics (per condition, aggregated over batches / agents):
* ``ADE_min`` / ``FDE_min`` (K=20 best-of-20, in original scale),
* ``d_traj`` = mean L2 displacement of the predicted best trajectory relative
  to the **baseline** best trajectory (attribution magnitude),
* ``d_ade``  = mean absolute ADE shift relative to baseline.

This is a pure-inference script: it never decodes video and never trains.

Example::

    python tools/attrib_video_conditioning.py \
        --cfg cfg/sdd/cor_fm.yml \
        --ckpt <results_sdd>/.../_SDD_ho<scene>_<variant>/models/checkpoint_best.pt \
        --sdd-root /home/efrei_stage/Desktop/Datasets/SDD \
        --video-features-root ./features/resnet18 \
        --held-out-scene <scene> \
        --n-batches 20 --batch-size 64 --seed 0

Outputs ``--out`` stem as ``<stem>_<condition>.npy`` prediction stacks plus an
aggregate ``<stem>.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Allow running as a plain script (`python tools/attrib_video_conditioning.py`)
# from anywhere: the script's own directory is tools/, but the project imports
# (data/, models/, utils/) live one level up at the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import DataLoader

try:
    import matplotlib

    matplotlib.use("Agg")
except Exception:
    pass

from data.dataloader_sdd_global import SDDGlobalDataset, collate_sdd_global
from models.backbone_eth_ucy import ETHMotionTransformer
from models.flow_matching import FlowMatcher
from tools.visualize_trajectory_comparison import load_checkpoint_state
from utils.config import Config
from utils.normalization import unnormalize_min_max

CONDITIONS: tuple[str, ...] = ("baseline", "zeroed", "permuted")


class _NullLogger:
    """Silent logger for offline inference (the backbone calls ``logger.info``)."""

    def info(self, *args, **kwargs) -> None:  # noqa: ANN002
        pass

    def warning(self, *args, **kwargs) -> None:  # noqa: ANN002
        pass

    def debug(self, *args, **kwargs) -> None:  # noqa: ANN002
        pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="cvxp attribution of z_video_global.")
    p.add_argument("--cfg", default="cfg/sdd/cor_fm.yml")
    p.add_argument("--ckpt", required=True, help="Trainer checkpoint (.pt).")
    p.add_argument("--exp", default="cvxp-attrib", help="Experiment tag (run dir).")
    p.add_argument("--sdd-root", default=None)
    p.add_argument("--video-features-root", default=None)
    p.add_argument("--held-out-scene", required=True)
    p.add_argument(
        "--norm-scene",
        default=None,
        help=(
            "Scene whose LOSO train-split normalization the checkpoint expects. "
            "Defaults to --held-out-scene. For cross-scene transferability "
            "(evaluate a model trained holding out A on scene B) pass "
            "--norm-scene A --held-out-scene B so inputs are normalized with the "
            "model's OWN train statistics, never the target scene's."
        ),
    )
    p.add_argument(
        "--conditions",
        nargs="+",
        choices=list(CONDITIONS),
        default=list(CONDITIONS),
        help="Subset of analytics to run: baseline zeroed permuted (default: all).",
    )
    p.add_argument("--split", choices=["test", "train"], default="test")
    p.add_argument(
        "--video-id",
        default=None,
        help=(
            "Restrict the (LOSO-selected) windows to a single video folder "
            "name (e.g. 'video0'). Single-video ablation; default: all videos."
        ),
    )
    p.add_argument(
        "--per-window",
        default=None,
        metavar="CSV",
        help=(
            "Also dump per-window (per-agent) ADE_min/FDE_min for every "
            "requested condition plus ``gain`` columns, used to rank windows "
            "by video contribution (Q5/Q7 geometry analysis)."
        ),
    )
    p.add_argument(
        "--n-batches",
        type=int,
        default=None,
        help="Max batches to process (None = all).",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out", default="report/cvxp_attrib", help="Output stem.")
    p.add_argument("--use-ema", action="store_true", help="Load EMA weights.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--no-video-arch",
        action="store_true",
        help=(
            "Force a trajectory-only architecture (USE_VIDEO=False) even "
            "when the checkpoint carries video weights. By default the "
            "architecture is inferred from the checkpoint's own video_proj "
            "weights, so _novid baselines load automatically."
        ),
    )
    p.add_argument(
        "--no-shuffle-permute",
        action="store_true",
        help="Permute via shift instead of rng shuffle (deterministic).",
    )
    args = p.parse_args(argv)
    return args


def _permuted(real: torch.Tensor, use_rng_shuffle: bool = True) -> torch.Tensor:
    """Return a per-sample shuffled copy with an identical row multiset.

    Random shuffle breaks the window→video correspondence across arbitrary
    pairs; an index-shift is a deterministic, fixed-point-free alternative that
    also preserves the marginal vector distribution while guaranteeing no
    window keeps its own vector.
    """
    B = real.shape[0]
    if B <= 1:
        return real.clone()
    if use_rng_shuffle:
        return real[torch.randperm(B)]
    idx = torch.arange(B)
    idx = torch.roll(idx, 1)
    assert not torch.equal(idx, torch.arange(B)), "shift must move every row"
    return real[idx]


def _assert_per_window_records(
    args: argparse.Namespace,
    batches: int,
    skipped_batches: int,
    window_rows: dict[tuple, dict[str, object]],
) -> None:
    """Guarantee the per-window evidence file will be written when requested.

    If ``--per-window`` was asked for but no per-window rows were recorded,
    raise -- with the *concrete* cause -- instead of writing only the
    aggregate (which silently breaks script 04's aggregation downstream).
    """
    if not args.per_window or window_rows:
        return
    if skipped_batches and batches == 0:
        reason = (
            f"all {skipped_batches} batch(es) were skipped because "
            "z_video_global was the missing-cache sentinel"
        )
    elif batches == 0:
        reason = "the dataset produced no batches (empty loader)"
    else:
        reason = (
            f"{batches} batch(es) ran but the per-window block never "
            "recorded rows — please report this bug"
        )
    raise SystemExit(
        "[cvxp] FATAL: --per-window requested but nothing could be "
        f"recorded ({reason}). Aggregate output would be misleading, so "
        "no files were written. Verify the video feature cache for the "
        "filtered scene/video exists under --video-features-root and that "
        "the dataset is non-empty (see '[cvxp] dataset=... windows=N')."
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = Config(args.cfg, args.exp)
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = cfg.device

    # ---- FM / arch defaults (mirror run_visualize_trajectories.py) ----------
    cfg.denoising_method = "fm"
    cfg.sampling_steps = cfg.sampling_steps
    cfg.t_schedule = "logit_normal"
    cfg.logit_norm_mean = -0.5
    cfg.logit_norm_std = 1.5
    cfg.fm_wrapper = "direct"
    cfg.fm_rew_sqrt = False
    cfg.fm_in_scaling = False
    cfg.perturb_ctx = 0.0
    cfg.drop_method = "emb"
    cfg.drop_logi_k = 20.0
    cfg.drop_logi_m = 0.5
    cfg.MODEL.USE_PRE_NORM = False
    cfg.tied_noise = False
    cfg.LOSS_NN_MODE = "agent"
    cfg.LOSS_REG_REDUCTION = "sum"
    cfg.LOSS_REG_SQUARED = False
    cfg.LOSS_VELOCITY = False
    cfg.rotate = False
    cfg.rotate_aug = False
    cfg.data_norm = "min_max"

    # ---- training-time normalization stats (train split, cached) ------------
    if args.sdd_root is None:
        raise SystemExit("--sdd-root is required (raw annotations tree).")
    from tools.visualize_trajectory_comparison import _ensure_norm_stats

    norm_scene = args.norm_scene or args.held_out_scene
    past_min, past_max, fut_min, fut_max = _ensure_norm_stats(norm_scene, args)
    cfg.past_traj_min, cfg.past_traj_max = past_min, past_max
    cfg.fut_traj_min, cfg.fut_traj_max = fut_min, fut_max

    # ---- dataset (video-scoped lookup; re-encoded caches required) --------
    # Test split always targets --held-out-scene; --norm-scene only dictates
    # which scene's train-split statistics normalize the inputs (decoupled for
    # the cross-scene transferability matrix).
    print(
        f"[cvxp] norm-scene={norm_scene} (stats ho{norm_scene}) "
        f"data-split={args.split} held_out={args.held_out_scene}"
    )
    dset = SDDGlobalDataset(
        cfg,
        training=False,
        sdd_root=args.sdd_root,
        held_out_scene=args.held_out_scene,
        split=args.split,
        use_video=True,
        video_features_root=args.video_features_root,
        video_mode="full",
        video_ids=[args.video_id] if getattr(args, "video_id", None) else None,
    )
    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=False,
        # num_workers=0 keeps the per-worker feature mmap unique to the main
        # process (same rationale as fm_sdd_global.py); multi-worker doubles
        # the mmap'd feature buffers in RAM for no measurable throughput here.
        num_workers=0,
        collate_fn=collate_sdd_global,
        pin_memory=True,
    )
    print(
        f"[cvxp] dataset={args.split} held_out={args.held_out_scene} "
        f"windows={len(dset)} mode=full"
    )

    # ---- model ------------------------------------------------------------
    # Infer the architecture from the checkpoint's own weights, so a
    # trajectory-only baseline (_..._novid) and a video-conditioned run both
    # load without shape mismatches. The video branch is materialized only
    # when the checkpoint actually carries its projection weights.
    state = load_checkpoint_state(Path(args.ckpt), use_ema=args.use_ema)
    has_video_proj = any(
        isinstance(k, str) and k.startswith("model.context_encoder.video_proj.")
        for k in state
    )
    if args.no_video_arch:
        if has_video_proj:
            print(
                "[cvxp] WARNING: checkpoint carries video weights but "
                "--no-video-arch forces a trajectory-only architecture; "
                "attribution will NOT exercise the z_video_global channel."
            )
        use_video = False
    else:
        use_video = has_video_proj
        if not use_video:
            print(
                "[cvxp] NOTE: checkpoint has NO video_proj weights "
                "(trajectory-only baseline). The model cannot consume "
                "z_video_global, so baseline == zeroed == permuted is "
                "expected; run any _*_vm{static,full} checkpoint to "
                "attribute the real video channel."
            )
    cfg.MODEL.CONTEXT_ENCODER.USE_VIDEO = use_video
    model = ETHMotionTransformer(
        model_config=cfg.MODEL, logger=_NullLogger(), config=cfg
    )
    denoiser = FlowMatcher(cfg, model, logger=_NullLogger())
    try:
        denoiser.load_state_dict(state)
    except RuntimeError as exc:
        raise SystemExit(
            "[cvxp] checkpoint state_dict does not match the model architecture "
            f"built from {args.cfg} (USE_VIDEO={use_video}). "
            "This usually means the checkpoint was trained with a different "
            "MODEL config (e.g. D_MODEL / layer counts / video flag). "
            "Pass --no-video-arch to force the trajectory-only architecture, "
            "or point --cfg at the config that produced the checkpoint.\n"
            f"  {exc}"
        ) from exc
    denoiser.to(device)
    denoiser.eval()
    print(f"[cvxp] loaded {args.ckpt} (use_ema={args.use_ema})")

    # ---- accumulators -----------------------------------------------------
    # ``baseline`` is always computed internally (the d_* attribution columns
    # for zeroed/permuted need it); ``run_conditions`` is what the user asked
    # to report.
    run_conditions: list[str] = list(args.conditions)
    sample_conditions = list(dict.fromkeys([*run_conditions, "baseline"]))
    agg: dict[str, dict[str, float]] = {
        c: {
            "ade_min": 0.0,
            "fde_min": 0.0,
            "n_agents": 0,
            "d_traj_sum": 0.0,
            "d_ade_sum": 0.0,
        }
        for c in sample_conditions
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    window_rows: dict[tuple, dict[str, object]] = {}
    window_counter = 0

    batches = 0
    skipped_batches = 0
    for b in loader:
        if b.get("z_video_global") is None:
            skipped_batches += 1
            print("[cvxp] batch has no z_video_global (sentinel) — skipping")
            continue
        z_real = b["z_video_global"].to(device)  # [B, D_raw]
        X = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in b.items()}

        preds: dict[str, np.ndarray | None] = {}
        for cond in sample_conditions:
            Xc_ = dict(X)
            if cond == "zeroed":
                z = torch.zeros_like(z_real)
            elif cond == "permuted":
                z = _permuted(z_real, use_rng_shuffle=not args.no_shuffle_permute)
            else:
                z = z_real
            Xc_["z_video_global"] = z
            # Reset the RNG to a fixed seed before every sample so the three
            # conditions draw IDENTICAL initial noise (FlowMatcher.sample uses
            # the global torch RNG at y_t ~ N(0,I)). The only thing that varies
            # across arms is then z_video_global — trajectory differences become
            # a clean measure of video influence, not sampling noise.
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            with torch.no_grad():
                pred_traj, *_ = denoiser.sample(
                    Xc_, num_trajs=cfg.denoising_head_preds, return_all_states=False
                )
            # [B, K, A, F*D] -> [B*A, K, F, 2] in original scale (min_max)
            pred_traj = rearrange(
                pred_traj, "b k a (f d) -> (b a) k f d", f=cfg.future_frames
            )[..., :2]
            pred_traj = unnormalize_min_max(
                pred_traj, cfg.fut_traj_min, cfg.fut_traj_max, -1, 1
            )
            preds[cond] = pred_traj.detach().cpu().numpy()

        # ground truth [B, A, F, 2] -> [B*A, F, 2] original scale
        fut_gt = rearrange(b["fut_traj_original_scale"], "b a f d -> (b a) f d").numpy()

        for cond in sample_conditions:
            pt = preds[cond]  # [M, K, F, 2]
            ade = (
                np.linalg.norm(pt - fut_gt[:, None], axis=-1).mean(axis=-1).min(axis=-1)
            )  # [M]
            fde = np.linalg.norm(pt[:, :, -1] - fut_gt[:, None, -1], axis=-1).min(
                axis=-1
            )  # [M]
            agg[cond]["ade_min"] += float(ade.sum())
            agg[cond]["fde_min"] += float(fde.sum())
            agg[cond]["n_agents"] += int(fut_gt.shape[0])
            if args.per_window:
                scene_names = list(b["scene"])
                video_ids = list(b["video_id"])
                anchor = b["anchor_frame"]
                n_agents_per_window = int(b["fut_traj_original_scale"].shape[1])
                for w in range(int(b["batch_size"])):
                    for a in range(n_agents_per_window):
                        m = w * n_agents_per_window + a
                        key = (
                            scene_names[w],
                            video_ids[w],
                            int(anchor[w]),
                            a,
                        )
                        wr = window_rows.get(key)
                        if wr is None:
                            wr = {
                                "scene": scene_names[w],
                                "video_id": video_ids[w],
                                "anchor_frame": int(anchor[w]),
                                "agent_in_window": a,
                                "window_index": window_counter + w,
                            }
                            window_rows[key] = wr
                        wr[f"ade_min_{cond}"] = float(ade[m])
                        wr[f"fde_min_{cond}"] = float(fde[m])
                window_counter += int(b["batch_size"])

        # attribution relative to baseline
        base = preds["baseline"]  # [M, K, F, 2]
        for cond in ("zeroed", "permuted"):
            if cond not in preds:
                continue
            pt = preds[cond]
            # per-agent min-ADE trajectory divergence from baseline
            b_ade = (
                np.linalg.norm(base - fut_gt[:, None], axis=-1)
                .mean(axis=-1)
                .min(axis=-1)
            )
            c_ade = (
                np.linalg.norm(pt - fut_gt[:, None], axis=-1).mean(axis=-1).min(axis=-1)
            )
            agg[cond]["d_ade_sum"] += float(np.abs(c_ade - b_ade).sum())
            # mean L2 displacement between best trajectories (baseline vs cond)
            b_best = base[
                np.arange(base.shape[0]),
                np.argmin(
                    np.linalg.norm(base - fut_gt[:, None], axis=-1).mean(axis=-1),
                    axis=-1,
                ),
            ]
            c_best = pt[
                np.arange(pt.shape[0]),
                np.argmin(
                    np.linalg.norm(pt - fut_gt[:, None], axis=-1).mean(axis=-1), axis=-1
                ),
            ]
            agg[cond]["d_traj_sum"] += float(
                np.linalg.norm(c_best - b_best, axis=-1).mean(axis=-1).sum()
            )

        batches += 1
        if args.n_batches is not None and batches >= args.n_batches:
            break
        if batches and batches % 5 == 0:
            print(f"[cvxp] processed {batches} batches")
    print(
        f"[cvxp] batches processed={batches} skipped={skipped_batches} "
        f"per-window rows={len(window_rows)}"
    )

    # --/-- per-window isolation guarantee ---------------------------------
    # The per-window CSV is the Q5/Q7 evidence file. If --per-window was
    # requested but NOTHING could be recorded, writing only the aggregate
    # would silently produce a "missing windows file" downstream (script 04
    # aggregation). Fail loudly with the actual cause instead.
    _assert_per_window_records(args, batches, skipped_batches, window_rows)

    # ---- aggregate + persist ---------------------------------------------
    rows = []
    for cond in run_conditions:
        a = agg[cond]
        n = max(1, a["n_agents"])
        row = {
            "condition": cond,
            "split": args.split,
            "held_out_scene": args.held_out_scene,
            "norm_scene": norm_scene,
            "n_agents": a["n_agents"],
            "ade_min": a["ade_min"] / n,
            "fde_min": a["fde_min"] / n,
        }
        if cond in ("zeroed", "permuted"):
            row["d_traj_from_baseline"] = a["d_traj_sum"] / n
            row["d_ade_from_baseline"] = a["d_ade_sum"] / n
        rows.append(row)
        print(
            f"  {cond:10s} ADE_min={row['ade_min']:.4f} FDE_min={row['fde_min']:.4f}"
            + (
                f" d_traj={row['d_traj_from_baseline']:.4f} d_ade={row['d_ade_from_baseline']:.4f}"
                if cond in ("zeroed", "permuted")
                else ""
            )
        )

    with open(out_path.with_suffix(".csv"), "w", newline="") as f:
        fieldnames = list(rows[0].keys())
        for r in rows:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    # per-condition prediction stacks for downstream statistical testing
    for cond in run_conditions:
        if batches and preds.get(cond) is not None:
            np.save(out_path.with_name(f"{out_path.stem}_{cond}.npy"), preds[cond])
    with open(out_path.with_suffix(".json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"[cvxp] wrote {out_path}.csv/.json (+ _<condition>.npy stacks)")

    # ---- per-window dump (window-level attribution evidence) -------------
    # window_rows is guaranteed non-empty here (checked above).
    if args.per_window:
        pw_path = Path(args.per_window)
        pw_path.parent.mkdir(parents=True, exist_ok=True)
        window_list = list(window_rows.values())
        pw_fieldnames = sorted({k for r in window_list for k in r}, key=str.lower)
        with open(pw_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=pw_fieldnames)
            w.writeheader()
            w.writerows(window_list)
        print(f"[cvxp] wrote {len(window_list)} per-window rows -> {pw_path}")


if __name__ == "__main__":
    main()
