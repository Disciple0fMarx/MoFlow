"""Publication-ready trajectory comparison: No Video baseline vs Global VE.

For each held-out SDD scene this script loads the trained no-video baseline
(`_SDD_ho<scene>_novid`) and the global-video model (`_SDD_ho<scene>_vid`),
samples future trajectories from both on the *same* observation window drawn
from the native ``SDDGlobalDataset`` test split, selects the best-of-K
hypothesis by min-ADE, and renders a single comparison figure.

Design notes
------------
* Model construction mirrors ``fm_sdd_global.init_basics`` exactly (same FM /
  loss / optimization overrides and SDD knobs injected into ``cfg``).
* Normalization statistics are identical to production evaluation: the
  dataset derives them from the training windows of the LOSO split. They are
  cached once per scene under ``RESULTS_DIR/_norm_stats_ho<scene>.npz``
  because they are deterministic (data-derived, seed-independent).
* Memory discipline (zero-OOM sequential scene evaluation):
  - one CPU batch per scene, shared verbatim by both checkpoints;
  - only ONE model resident on the GPU at any time (built, sampled, freed);
  - sampling wrapped in ``torch.inference_mode()``;
  - predictions are detached, unnormalized, reduced to best-of-K and moved
    to host NumPy arrays immediately after inference;
  - ``torch.cuda.empty_cache()`` (and ``gc.collect()``) are called between
    the two checkpoints and again at the end of every scene.

Usage::

    # styling self-check (no checkpoints / dataset required)
    python tools/visualize_trajectory_comparison.py --demo

    # real evaluation (lab machine)
    python tools/visualize_trajectory_comparison.py \
        --scenes gates quad \
        --video_features_root ./features/resnet18

Figures are written to ``visualizations/<scene>_trajectory_comparison.png``
at 300 DPI with serif fonts sized for print readability.
"""
from __future__ import annotations

import argparse
import gc
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Publication-quality rcParams (scientific serif styling, print-readable sizes)
# ---------------------------------------------------------------------------
matplotlib.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "stix",
        "font.size": 13,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "lines.linewidth": 1.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        # editable text (not outlined paths) when exporting PDF/EPS figures
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from einops import rearrange  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from data.dataloader_sdd_global import (  # noqa: E402
    SDD_SCENES,
    SDDGlobalDataset,
    collate_sdd_global,
)
from models.backbone_eth_ucy import ETHMotionTransformer  # noqa: E402
from models.flow_matching import FlowMatcher  # noqa: E402
from utils.config import Config  # noqa: E402
from utils.normalization import unnormalize_min_max  # noqa: E402
from video_encoder.sdd_adapter import expand_sdd_root  # noqa: E402

CFG_PATH = REPO_ROOT / "cfg" / "sdd" / "cor_fm.yml"
RESULTS_DIR = REPO_ROOT / "results_sdd" / "cor_fm"
DEFAULT_OUT_DIR = REPO_ROOT / "visualizations"

STYLE = {
    "history": {"color": "black", "linestyle": "--", "linewidth": 1.6, "label": "History"},
    "gt": {"color": "green", "linestyle": "-", "linewidth": 1.8, "label": "Ground truth"},
    "baseline": {"color": "blue", "linestyle": "-", "linewidth": 2.0, "alpha": 0.7, "label": "No Video"},
    "ours": {"color": "red", "linestyle": "-", "linewidth": 3.0, "alpha": 0.9, "label": "Global VE"},
}

logger = logging.getLogger("viz")


# ---------------------------------------------------------------------------
# Plotting core
# ---------------------------------------------------------------------------
def plot_trajectory_comparison(
    obs_traj: np.ndarray,
    gt_traj: np.ndarray,
    pred_baseline: np.ndarray,
    pred_video: np.ndarray,
    title: str | None = None,
    save_path: str | Path | None = None,
    ax=None,
):
    """Render history vs ground truth vs both model predictions.

    All inputs are coerced with :func:`_as_xy`, i.e. any of ``[T, 2]``,
    ``[A, T, 2]`` or ``[B, A, T, 2]`` array-likes (NumPy or Torch) are accepted;
    agent 0 is plotted.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = ax.figure

    obs = _as_xy(obs_traj)
    gt = _as_xy(gt_traj)
    base = _as_xy(pred_baseline)
    ours = _as_xy(pred_video)

    ax.plot(obs[:, 0], obs[:, 1], **STYLE["history"])
    ax.scatter(obs[-1, 0], obs[-1, 1], color="black", marker="s", s=70, zorder=5)

    ax.plot(gt[:, 0], gt[:, 1], **STYLE["gt"])
    ax.plot(base[:, 0], base[:, 1], **STYLE["baseline"])
    ax.plot(ours[:, 0], ours[:, 1], **STYLE["ours"])

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title if title is not None else "Trajectory Comparison")
    ax.legend(loc="best", frameon=True, framealpha=0.9)
    ax.set_aspect("equal", adjustable="datalim")

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[viz] wrote {save_path}")
    return ax


def _as_xy(traj) -> np.ndarray:
    """Coerce torch/numpy trajectory to a ``[T, 2]`` float NumPy array."""
    if hasattr(traj, "detach"):
        traj = traj.detach().cpu().numpy()
    traj = np.asarray(traj, dtype=np.float64)
    while traj.ndim > 2:  # [B, A, T, 2] / [A, T, 2] -> [T, 2]
        traj = traj[0]
    return traj


# ---------------------------------------------------------------------------
# Best-of-K selection (pure function; unit-testable without torch/GPU)
# ---------------------------------------------------------------------------
def select_best_of_k(
    preds: np.ndarray, gt: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Select the hypothesis minimising average displacement error.

    Args:
        preds: ``[N, K, F, 2]]`` predicted futures (original scale).
        gt: ``[N, F, 2]`` ground-truth futures (original scale).

    Returns:
        ``(best [N, F, 2], argmin indices [N])`` where ``best[n]`` is the
        hypothesis with the lowest mean L2 distance to ``gt[n]``.
    """
    distances = np.linalg.norm(preds - gt[:, None, :, :], axis=-1)  # [N, K, F]
    ade = distances.mean(axis=-1)                                   # [N, K]
    idx = ade.argmin(axis=-1)                                       # [N]
    best = preds[np.arange(preds.shape[0]), idx]                    # [N, F, 2]
    return best, idx


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------
def load_checkpoint_state(ckpt_path: Path, use_ema: bool = False) -> dict:
    """Return the raw model ``state_dict`` stored in a trainer checkpoint.

    The trainer saves ``{'step', 'model', 'opt', 'ema', 'scheduler', ...}``;
    ``'model'`` is the unwrapped denoiser ``state_dict`` and ``'ema'`` holds
    the exponential-moving-average weights (smoother, preferred for visual
    comparisons when available).
    """
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and ("model" in ckpt or "ema" in ckpt):
        if use_ema and ckpt.get("ema") is not None:
            return ckpt["ema"]
        return ckpt["model"]
    return ckpt


# ---------------------------------------------------------------------------
# Config construction (mirror of fm_sdd_global.init_basics defaults)
# ---------------------------------------------------------------------------
def _build_cfg(scene: str, vid_status: str, args: argparse.Namespace) -> Config:
    """Build the project ``Config`` for one (scene, checkpoint) pair.

    Mirrors every override applied by ``fm_sdd_global.init_basics`` using the
    same default values as the training CLI, then injects the SDD knobs and
    the normalization statistics required at evaluation time.
    """
    cfg = Config(str(CFG_PATH), tag="viz")

    # ---- FM / arch / loss / optimization (defaults of fm_sdd_global.py) ----
    cfg.denoising_method = "fm"
    cfg.sampling_steps = args.sampling_steps
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
    cfg.rotate = False
    cfg.rotate_aug = False
    cfg.data_norm = "min_max"

    # ---- SDD knobs ----------------------------------------------------------
    ce = cfg.MODEL.CONTEXT_ENCODER
    ce.SDD_ROOT = str(expand_sdd_root(args.sdd_root))
    ce.HELD_OUT_SCENE = scene
    if args.video_features_root is not None:
        ce.VIDEO_FEATURES_ROOT = str(Path(args.video_features_root).expanduser())
    # NOTE: the flag mirrors the CHECKPOINT's modality, not the local YAML
    # default: the vid run trained with the video branch enabled (the YAML
    # value is toggled per experiment), so forcing it here keeps the built
    # architecture in sync with the stored weights.
    ce.USE_VIDEO = vid_status == "vid"

    # ---- Normalization statistics (production-eval semantics) --------------
    past_min, past_max, fut_min, fut_max = _ensure_norm_stats(scene, args)
    cfg.past_traj_min = past_min
    cfg.past_traj_max = past_max
    cfg.fut_traj_min = fut_min
    cfg.fut_traj_max = fut_max

    cfg.device = args.device
    return cfg


def _ensure_norm_stats(scene: str, args: argparse.Namespace) -> tuple[float, float, float, float]:
    """Return ``(past_min, past_max, fut_min, fut_max)`` for the LOSO split.

    Statistics are derived from the training windows exactly as in production
    (``SDDGlobalDataset(training=True)``), are deterministic across runs, and
    are cached to ``RESULTS_DIR/_norm_stats_ho<scene>.npz`` so repeated
    invocations skip the (relatively expensive) train-split index scan.
    """
    cache = RESULTS_DIR / f"_norm_stats_ho{scene}.npz"
    if cache.exists():
        data = np.load(cache)
        return (
            float(data["past_min"]),
            float(data["past_max"]),
            float(data["fut_min"]),
            float(data["fut_max"]),
        )

    logger.info("[%s] computing norm stats from train split (one-time)...", scene)
    cfg_stats = Config(str(CFG_PATH), tag="viz-stats")
    dset = SDDGlobalDataset(
        cfg_stats,
        training=True,
        sdd_root=args.sdd_root,
        held_out_scene=scene,
        split="train",
        use_video=False,  # stats are video-independent
    )
    stats = (
        float(dset.past_traj_min),
        float(dset.past_traj_max),
        float(dset.fut_traj_min),
        float(dset.fut_traj_max),
    )
    del dset, cfg_stats  # free window index + materialized tensors
    gc.collect()
    torch.cuda.empty_cache()

    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(cache, past_min=stats[0], past_max=stats[1], fut_min=stats[2], fut_max=stats[3])
    except OSError as exc:  # read-only results dir -> skip caching silently
        logger.warning("could not cache norm stats (%s)", exc)
    return stats


def _quiet_logger() -> logging.Logger:
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.INFO)
    return logger


# ---------------------------------------------------------------------------
# Native hooks
# ---------------------------------------------------------------------------
def build_model_and_cfg(scene: str, vid_status: str, args: argparse.Namespace):
    """Instantiate ``ETHMotionTransformer`` + ``FlowMatcher`` natively.

    Mirrors ``fm_sdd_global.build_network``; loads the matching checkpoint
    (``_SDD_ho<scene>_<vid|novid>/models/checkpoint_best.pt``) onto the
    device and switches the denoiser to eval mode.
    """
    suffix = "vid" if vid_status == "vid" else "novid"
    ckpt_path = RESULTS_DIR / f"_SDD_ho{scene}_{suffix}" / "models" / "checkpoint_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    cfg = _build_cfg(scene, vid_status, args)
    log = _quiet_logger()

    model = ETHMotionTransformer(model_config=cfg.MODEL, logger=log, config=cfg)
    denoiser = FlowMatcher(cfg, model, logger=log)

    state = load_checkpoint_state(ckpt_path, use_ema=args.use_ema)
    denoiser.load_state_dict(state)
    denoiser.to(cfg.device)
    denoiser.eval()
    logger.info("[%s/%s] loaded %s", scene, vid_status, ckpt_path.name)
    return denoiser, cfg


def get_batch_for_scene(scene: str, args: argparse.Namespace) -> dict:
    """Fetch exactly one evaluation batch for the held-out scene.

    Uses the native ``SDDGlobalDataset`` test split (LOSO held-out scene),
    ``batch_size=1``, ``shuffle=False`` → deterministic window selection.
    The batch is fetched once with the video branch enabled and shared
    verbatim by BOTH checkpoints; the baseline simply ignores
    ``z_video_global`` because it was trained with ``USE_VIDEO=False``.
    """
    cfg = _build_cfg(scene, "vid", args)
    dset = SDDGlobalDataset(
        cfg,
        training=False,
        sdd_root=args.sdd_root,
        held_out_scene=scene,
        split="test",
        use_video=True,
        video_features_root=args.video_features_root,
    )
    loader = DataLoader(
        dset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_sdd_global,
    )
    batch = next(iter(loader))
    del dset, loader, cfg  # release the window index immediately
    gc.collect()
    return batch


def _move_to_device(batch: dict, device: str) -> dict:
    """Transfer tensors only; lists/strings/None pass through untouched."""
    return {
        k: v.to(device) if hasattr(v, "to") else v
        for k, v in batch.items()
    }


def sample_prediction(denoiser, cfg: Config, batch_cpu: dict) -> dict[str, np.ndarray]:
    """Sample best-of-K futures for one scene batch and return host arrays.

    Everything runs under ``torch.inference_mode()``; GPU tensors are
    detached and converted to NumPy before returning so no CUDA memory
    outlives this call.
    """
    device = cfg.device
    with torch.inference_mode():
        x_gpu = _move_to_device(batch_cpu, device)
        # returns (final_y [B,K,A,F*D], states [B,S,K,A,F*D], t, y_t, score)
        sample_out = denoiser.sample(x_gpu, num_trajs=cfg.denoising_head_preds)
        pred = sample_out[0]

        pred = rearrange(
            pred, "b k a (f d) -> b k a f d", f=cfg.future_frames
        )[..., :2]  # [B, K, A, F, 2]

        if cfg.get("data_norm", "min_max") == "min_max":
            pred = unnormalize_min_max(
                pred, cfg.fut_traj_min, cfg.fut_traj_max, -1, 1
            )

        # detach + offload to host RAM immediately
        preds_np = rearrange(
            pred.detach().cpu().numpy(), "b k a f d -> (b a) k f d"
        )

        del x_gpu, sample_out, pred

    # ground truth / history live on the CPU batch already
    gt_np = rearrange(
        batch_cpu["fut_traj_original_scale"].numpy(), "b a f d -> (b a) f d"
    ).astype(np.float64)
    # past_full columns: [abs_x, abs_y, rel_x, rel_y, vel_x, vel_y];
    # the relative frame shares its origin with the GT/predictions.
    obs_np = rearrange(
        batch_cpu["past_traj_original_scale"].numpy()[..., 2:4],
        "b a p d -> (b a) p d",
    ).astype(np.float64)

    best, _ = select_best_of_k(preds_np, gt_np)
    return {"obs": obs_np, "gt": gt_np, "preds_all": preds_np, "pred_best": best}


# ---------------------------------------------------------------------------
# Scene orchestration (strict memory hygiene)
# ---------------------------------------------------------------------------
def run_scene(scene: str, args: argparse.Namespace) -> Path | None:
    """Evaluate both checkpoints on ``scene`` and render the comparison."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_cpu = get_batch_for_scene(scene, args)
    results: dict[str, dict[str, np.ndarray]] = {}
    try:
        for vid_status in ("novid", "vid"):
            denoiser, cfg = build_model_and_cfg(scene, vid_status, args)
            try:
                results[vid_status] = sample_prediction(denoiser, cfg, batch_cpu)
            finally:
                # free the model before the next one is built
                del denoiser, cfg
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        del batch_cpu
        gc.collect()
        torch.cuda.empty_cache()

    obs = results["novid"]["obs"][0]
    gt = results["novid"]["gt"][0]
    base = results["novid"]["pred_best"][0]
    ours = results["vid"]["pred_best"][0]

    save_path = out_dir / f"{scene}_trajectory_comparison.png"
    plot_trajectory_comparison(
        obs,
        gt,
        base,
        ours,
        title=f"SDD '{scene}' — No Video vs Global VE",
        save_path=save_path,
    )
    return save_path


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scenes", nargs="+", default=["gates", "quad"], choices=SDD_SCENES)
    p.add_argument("--device", default=None, help="'cuda', 'cpu' (default: auto)")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--sdd-root", default=None)
    p.add_argument("--video_features_root", dest="video_features_root", default=None)
    p.add_argument("--sampling_steps", type=int, default=10)
    p.add_argument("--use-ema", action="store_true",
                   help="prefer the EMA weights over the raw model weights")
    p.add_argument("--demo", action="store_true",
                   help="render a synthetic styling demo (no checkpoints needed)")
    return p.parse_args(argv)


def run_demo(out_dir: str | Path) -> Path:
    """Styling self-check with synthetic geometry (no repo deps required)."""
    t = np.linspace(0, 4 * np.pi, 12)
    obs = np.stack([t / 8, np.sin(t / 2)], axis=-1)
    gt = np.stack([obs[-1, 0] + np.linspace(0, 3, 12), obs[-1, 1] + np.linspace(0, 2.5, 12)], axis=-1)
    noise_b = lambda: gt + np.random.randn(12, 2) * 0.35
    save_path = Path(out_dir) / "demo_trajectory_comparison.png"
    plot_trajectory_comparison(obs, gt, noise_b(), noise_b(),
                               title="Demo — No Video vs Global VE",
                               save_path=save_path)
    return save_path


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.demo:
        path = run_demo(args.out_dir)
        print(f"[viz] demo figure at {path}")
        return

    for scene in args.scenes:
        print(f"[viz] === scene: {scene} ===")
        saved = run_scene(scene, args)
        print(f"[viz] done: {saved}")


if __name__ == "__main__":
    main()
