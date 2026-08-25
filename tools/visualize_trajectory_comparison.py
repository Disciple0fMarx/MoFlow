"""Publication-ready trajectory comparison: No Video baseline vs Global VE.

For each held-out SDD scene this script loads the trained no-video baseline
(``_SDD_ho<scene>_novid``) and the global-video model (``_SDD_ho<scene>_vid``),
samples the full ``K``-hypothesis future distribution from both on the *same*
observation window drawn from the native ``SDDGlobalDataset`` test split, and
renders a single comparison figure showing:

* the multimodal prediction bundles (all ``K`` heads, low-alpha lines),
* the min-ADE best hypothesis of each model (thick line),
* discrete timestep markers along every path,
* optionally the SDD reference frame of the source video as background —
  trajectories live in the annotation pixel space of that exact frame, so
  overlaying is an exact coordinate match (origin top-left, y down).

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
  - predictions are detached, unnormalized and moved to host NumPy arrays
    immediately after inference;
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

# Bundled samples: low-alpha lines exposing the multimodal distribution;
# best hypotheses: the previously-established headline styling, kept thick.
SAMPLE_STYLE_BASE = {"linestyle": "-", "linewidth": 0.9, "marker": "o",
                     "markersize": 2}
STYLE = {
    "history": {"color": "black", "linestyle": "--", "linewidth": 1.6,
                "marker": "o", "markersize": 3, "label": "History"},
    "gt": {"color": "green", "linestyle": "-", "linewidth": 1.8,
           "marker": "o", "markersize": 3, "label": "Ground truth"},
    "baseline_best": {"color": "blue", "linewidth": 2.0, "alpha": 0.7,
                      "marker": "o", "markersize": 3,
                      "label": "No Video (best-of-K)"},
    "ours_best": {"color": "red", "linewidth": 3.0, "alpha": 0.9,
                  "marker": "o", "markersize": 3,
                  "label": "Global VE (best-of-K)"},
}

logger = logging.getLogger("viz")


# ---------------------------------------------------------------------------
# Plotting core
# ---------------------------------------------------------------------------
def plot_trajectory_comparison(
    obs_traj: np.ndarray,
    gt_traj: np.ndarray,
    baseline_samples: np.ndarray,
    ours_samples: np.ndarray,
    baseline_best: np.ndarray | None = None,
    ours_best: np.ndarray | None = None,
    title: str | None = None,
    save_path: str | Path | None = None,
    ax=None,
    background: np.ndarray | None = None,
    extent: tuple[float, float, float, float] | None = None,
    units: str = "m",
):
    """Render history vs ground truth vs both prediction distributions.

    Args:
        obs_traj: ``[T, 2]`` observed history.
        gt_traj: ``[F, 2]`` ground-truth future.
        baseline_samples / ours_samples: ``[K, F, 2]`` hypothesis bundles;
            every head is drawn as a low-alpha line so the multimodality of
            the FlowMatcher output is visible.
        baseline_best / ours_best: optional ``[F, 2]`` min-ADE trajectories
            drawn as thick highlight lines on top of their bundles.
        background: optional HxW(x3) image drawn behind everything; requires
            ``extent=(x0, x1, y1, y0)`` in trajectory coordinates (for SDD
            pixel-space overlays: ``(0, W, H, 0)``, origin top-left).
        units: axis-unit suffix for the labels (``px`` when a background is
            overlaid, ``m`` otherwise).
    Inputs accept torch tensors or arrays shaped ``[.., T, 2]`` (agent 0 is
    used when extra leading dims are present).
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = ax.figure

    obs = _as_xy(obs_traj)
    gt = _as_xy(gt_traj)
    base = _as_bundle(baseline_samples)
    ours = _as_bundle(ours_samples)

    if background is not None:
        if extent is None:  # default: full image, origin top-left (y down)
            extent = (0, background.shape[1], background.shape[0], 0)
        ax.imshow(background, extent=extent, origin="upper", zorder=0,
                  interpolation="bilinear")
        ax.set_xlim(extent[0], extent[1])
        # (H, 0): pixel-space orientation — y increases downward so the
        # annotation coordinates land exactly on their image locations.
        ax.set_ylim(extent[2], extent[3])

    def draw_bundle(bundle, color, label):
        for i, hyp in enumerate(bundle):
            ax.plot(hyp[:, 0], hyp[:, 1], color=color, alpha=0.15,
                    label=label if i == 0 else "_nolegend_",
                    zorder=2, **SAMPLE_STYLE_BASE)

    ax.plot(obs[:, 0], obs[:, 1], zorder=4, **STYLE["history"])
    ax.scatter(obs[-1, 0], obs[-1, 1], color="black", marker="s", s=70,
               zorder=5)
    ax.plot(gt[:, 0], gt[:, 1], zorder=4, **STYLE["gt"])

    draw_bundle(base, "blue", "No Video samples")
    draw_bundle(ours, "red", "Global VE samples")
    if baseline_best is not None:
        b = _as_xy(baseline_best)
        ax.plot(b[:, 0], b[:, 1], zorder=5, **STYLE["baseline_best"])
    if ours_best is not None:
        o = _as_xy(ours_best)
        ax.plot(o[:, 0], o[:, 1], zorder=5, **STYLE["ours_best"])

    ax.set_xlabel(f"x [{units}]")
    ax.set_ylabel(f"y [{units}]")
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


def _as_bundle(samples) -> np.ndarray:
    """Coerce predictions to ``[K, F, 2]`` (leading B/A dims collapsed)."""
    if hasattr(samples, "detach"):
        samples = samples.detach().cpu().numpy()
    samples = np.asarray(samples, dtype=np.float64)
    while samples.ndim > 3:  # [B, A, K, F, 2] / [A, K, F, 2]
        samples = samples[0]
    return samples


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
# Background loading (SDD reference frames)
# ---------------------------------------------------------------------------
def _imread_rgb(path: Path) -> np.ndarray:
    """Read an image as ``[H, W, 3]`` uint8-ish array (matplotlib or cv2)."""
    try:
        img = plt.imread(str(path))
    except Exception:
        import cv2  # local fallback; matplotlib handles most formats anyway

        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    return img


def load_background_image(
    sdd_root: str | Path | None, scene: str, video_id: str,
    override: str | Path | None = None,
) -> tuple[np.ndarray, tuple[float, float, float, float]] | None:
    """Locate and read the reference frame backing the annotations.

    SDD annotations (and therefore the dataloader's absolute trajectories)
    are pixel coordinates in the reference frame of each video, so the
    returned ``extent=(0, W, H, 0)`` aligns trajectories exactly on top of
    the image (origin top-left, y pointing down).

    Candidate locations, first hit wins:
      1. explicit ``override`` path (--background);
      2. ``<sdd_root>/frames/<scene>/<video_id>/referenceframe.jpg``
         (official SDD layout);
      3. ``<sdd_root>/videos/<scene>/<video_id>/referenceframe.jpg``;
      4. first extracted frame ``<sdd_root>/videos/<scene>/<video_id>/
         frames/frame000001.jpg`` (layout produced by ``video_encoder``).
    Returns ``None`` when nothing is found (figure falls back to white).
    """
    candidates: list[Path] = []
    if override is not None:
        candidates.append(Path(override))
    if sdd_root is not None:
        root = Path(expand_sdd_root(sdd_root))
        candidates.extend(
            [
                root / "frames" / scene / video_id / "referenceframe.jpg",
                root / "videos" / scene / video_id / "referenceframe.jpg",
                root / "videos" / scene / video_id / "frames" / "frame000001.jpg",
            ]
        )
    for cand in candidates:
        if cand.exists():
            img = _imread_rgb(cand)
            logger.info("[%s/%s] background: %s", scene, video_id, cand)
            return img, (0.0, float(img.shape[1]), float(img.shape[0]), 0.0)
    logger.warning(
        "[%s/%s] no reference frame found; plotting without background",
        scene, video_id,
    )
    return None


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
    The batch carries ``scene``/``video_id``/``anchor_frame`` metadata used
    for background lookup.
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
    """Sample full K-hypothesis futures for one batch and return host arrays.

    Everything runs under ``torch.inference_mode()``; GPU tensors are
    detached and converted to NumPy before returning so no CUDA memory
    outlives this call. Predictions are returned in the relative frame
    (origin = last observed position); ``obs_abs`` provides the absolute
    pixel anchor required to map them onto the reference frame.
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

        # detach + offload to host RAM immediately: [N=(b a), K, F, 2]
        preds_np = rearrange(
            pred.detach().cpu().numpy(), "b k a f d -> (b a) k f d"
        )

        del x_gpu, sample_out, pred

    # ground truth / history live on the CPU batch already
    gt_np = rearrange(
        batch_cpu["fut_traj_original_scale"].numpy(), "b a f d -> (b a) f d"
    ).astype(np.float64)
    # past_full columns: [abs_x, abs_y, rel_x, rel_y, vel_x, vel_y];
    # the relative frame shares its origin (last observed point) with the
    # GT/predictions, while cols 0:2 give the absolute pixel anchor.
    past_full = batch_cpu["past_traj_original_scale"].numpy()
    obs_np = rearrange(past_full[..., 2:4], "b a p d -> (b a) p d").astype(np.float64)
    obs_abs = rearrange(past_full[..., 0:2], "b a p d -> (b a) p d")[0].astype(np.float64)

    best, _ = select_best_of_k(preds_np, gt_np)
    return {
        "obs": obs_np,
        "obs_abs": obs_abs,
        "gt": gt_np,
        "preds_all": preds_np,
        "pred_best": best,
    }


# ---------------------------------------------------------------------------
# Scene orchestration (strict memory hygiene)
# ---------------------------------------------------------------------------
def _to_absolute(arrs: list[np.ndarray], anchor: np.ndarray) -> list[np.ndarray]:
    """Shift relative-frame arrays into annotation pixel space."""
    return [arr + anchor[None, :] for arr in arrs]


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

        video_id = batch_cpu.get("video_id", [""])[0]
        background = None
        if not getattr(args, "no_background", False):
            loaded = load_background_image(
                args.sdd_root, scene, video_id,
                override=getattr(args, "background", None),
            )
            if loaded is not None:
                background, _ = loaded
    finally:
        del batch_cpu
        gc.collect()
        torch.cuda.empty_cache()

    novid, vid = results["novid"], results["vid"]
    obs, gt = novid["obs"][0], novid["gt"][0]
    base_all, ours_all = novid["preds_all"][0], vid["preds_all"][0]
    base_best, ours_best = novid["pred_best"][0], vid["pred_best"][0]

    if background is not None:
        # Trajectories are stored relative to the last observed point whose
        # ABSOLUTE position (annotation pixels) anchors them on the frame.
        anchor = novid["obs_abs"][-1]
        obs = obs + anchor[None, :]
        gt, base_best, ours_best = _to_absolute([gt, base_best, ours_best], anchor)
        base_all, ours_all = _to_absolute([base_all, ours_all], anchor)
        units = "px"
    else:
        units = "m"

    save_path = out_dir / f"{scene}_trajectory_comparison.png"
    plot_trajectory_comparison(
        obs,
        gt,
        base_all,
        ours_all,
        baseline_best=base_best,
        ours_best=ours_best,
        title=f"SDD '{scene}' — No Video vs Global VE",
        save_path=save_path,
        background=background,
        units=units,
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
    p.add_argument("--background", default=None,
                   help="explicit path to a background image (overrides the "
                        "automatic reference-frame lookup)")
    p.add_argument("--no-background", action="store_true",
                   help="skip background lookup and plot on white")
    p.add_argument("--demo", action="store_true",
                   help="render a synthetic styling demo (no checkpoints needed)")
    return p.parse_args(argv)


def run_demo(out_dir: str | Path) -> Path:
    """Styling self-check: synthetic distribution over a synthetic backdrop."""
    rng = np.random.default_rng(0)
    t = np.linspace(0, 4 * np.pi, 12)
    obs = np.stack([t / 8, np.sin(t / 2)], axis=-1)
    gt = np.stack(
        [obs[-1, 0] + np.linspace(0, 3, 12), obs[-1, 1] + np.linspace(0, 2.5, 12)],
        axis=-1,
    )

    def bundle(scale: float) -> np.ndarray:
        modes = np.array([[0.0, 0.6], [1.4, 0.0]])  # bimodal ground truth-ish
        out = np.empty((20, 12, 2))
        for k in range(20):
            mode = modes[k % 2] + rng.normal(scale=scale, size=2)
            out[k] = gt + mode[None, :] * np.linspace(0, 1, 12)[:, None]
            out[k] += rng.normal(scale=scale * 0.3, size=(12, 2))
        return out

    grad = np.linspace(0, 255, 64 * 48, dtype=np.uint8).reshape(48, 64)
    background = np.repeat(grad[..., None], 3, axis=-1)  # fake backdrop
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_path = out / "demo_trajectory_comparison.png"
    plot_trajectory_comparison(
        obs, gt, bundle(0.35), bundle(0.35),
        baseline_best=bundle(0.35)[0], ours_best=bundle(0.35)[0],
        title="Demo — No Video vs Global VE",
        save_path=save_path, background=background, units="px",
    )
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
