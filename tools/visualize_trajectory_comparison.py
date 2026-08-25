"""Publication-ready trajectory comparison: No Video baseline vs Global VE.

For each held-out SDD scene this script loads the trained no-video baseline
(``_SDD_ho<scene>_novid``) and the global-video model (``_SDD_ho<scene>_vid``),
samples the full ``K``-hypothesis future distribution from both on the *same*
observation window drawn from the native ``SDDGlobalDataset`` test split, and
renders a side-by-side comparison figure:

* **left panel** — No Video baseline vs ground truth,
* **right panel** — Global VE vs ground truth,

both overlaid on the EXACT video frame at the present timestep (the final
observation step, ``anchor_frame`` of the plotted window). Trajectories live
in the annotation pixel space of the source video, so overlaying is an exact
coordinate match (origin top-left, y down). Each panel shows the observed
history, the ground-truth future, the full multimodal prediction bundle
(all ``K`` heads, low alpha) and the min-ADE best hypothesis highlighted
with a thick line; discrete timesteps are marked along every path. Both
panels share identical axis limits and the same background frame so the
layouts are directly comparable.

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
    the two checkpoints and again at the end of every scene;
  - the background frame is decoded once per scene (cv2 seek, released
    immediately) and reused by both panels.

Usage::

    # styling self-check (no checkpoints / dataset required)
    python tools/visualize_trajectory_comparison.py --demo

    # real evaluation (lab machine)
    python tools/visualize_trajectory_comparison.py \\
        --scenes gates quad \\
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
        "axes.titlesize": 13,
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
from video_encoder.sdd_adapter import (  # noqa: E402
    expand_sdd_root,
    video_mov_path,
)

CFG_PATH = REPO_ROOT / "cfg" / "sdd" / "cor_fm.yml"
RESULTS_DIR = REPO_ROOT / "results_sdd" / "cor_fm"
DEFAULT_OUT_DIR = REPO_ROOT / "visualizations"

# Bundled samples: low-alpha lines exposing the multimodal distribution;
# best hypotheses: thick highlight lines, one colour per model.
SAMPLE_STYLE_BASE = {"linestyle": "-", "linewidth": 0.9, "marker": "o",
                     "markersize": 2}
STYLE = {
    "history": {"color": "black", "linestyle": "--", "linewidth": 1.6,
                "marker": "o", "markersize": 3, "label": "History"},
    "gt": {"color": "green", "linestyle": "-", "linewidth": 1.8,
           "marker": "o", "markersize": 3, "label": "Ground truth"},
    "baseline_best": {"color": "blue", "linewidth": 2.4, "alpha": 0.85,
                      "marker": "o", "markersize": 3,
                      "label": "Best-of-K (min-ADE)"},
    "ours_best": {"color": "red", "linewidth": 2.4, "alpha": 0.85,
                  "marker": "o", "markersize": 3,
                  "label": "Best-of-K (min-ADE)"},
}

logger = logging.getLogger("viz")


# ---------------------------------------------------------------------------
# Plotting core (side-by-side panels)
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
    background: np.ndarray | None = None,
    extent: tuple[float, float, float, float] | None = None,
    units: str = "m",
    panel_titles: tuple[str, str] = ("No Video (baseline)", "Global VE"),
):
    """Render a 1x2 figure: each model's prediction vs the shared ground truth.

    Args:
        obs_traj: ``[T, 2]`` observed history (identical in both panels).
        gt_traj: ``[F, 2]`` ground-truth future (identical in both panels).
        baseline_samples / ours_samples: ``[K, F, 2]`` hypothesis bundles;
            every head is drawn as a low-alpha marker line so the multimodal
            FlowMatcher output stays visible.
        baseline_best / ours_best: optional ``[F, 2]`` min-ADE trajectories,
            drawn as thick highlight lines on top of their bundles.
        background: optional HxW(x3) image drawn behind BOTH panels; requires
            ``extent=(x0, x1, y_bottom, y_top)`` in trajectory coordinates
            (SDD pixel-space overlays use ``(0, W, H, 0)``: origin top-left,
            y pointing down).
        units: axis-unit suffix (``px`` when a background is overlaid).
        panel_titles: per-panel titles identifying the two checkpoints.

    Both axes end up with strictly identical limits: image-extent-derived
    when a background is present, otherwise a common data bounding box.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)

    obs = _as_xy(obs_traj)
    gt = _as_xy(gt_traj)
    base_all = _as_bundle(baseline_samples)
    ours_all = _as_bundle(ours_samples)

    _, labels_r = _draw_panel(
        axes[0], obs, gt, base_all,
        best=_as_xy(baseline_best) if baseline_best is not None else None,
        best_style=STYLE["baseline_best"], panel_title=panel_titles[0],
        units=units,
    )
    handles_r, _ = _draw_panel(
        axes[1], obs, gt, ours_all,
        best=_as_xy(ours_best) if ours_best is not None else None,
        best_style=STYLE["ours_best"], panel_title=panel_titles[1],
        units=units,
    )

    if background is not None:
        if extent is None:  # default: full image, origin top-left (y down)
            extent = (0.0, float(background.shape[1]),
                      float(background.shape[0]), 0.0)
        for ax_ in axes:
            ax_.imshow(background, extent=extent, origin="upper", zorder=0,
                       interpolation="bilinear")
        x0, x1, yb, yt = extent
        for ax_ in axes:
            ax_.set_xlim(x0, x1)
            # pixel-space orientation: y increases downward
            ax_.set_ylim(yb, yt)
    else:
        _set_shared_data_limits(axes, [obs, gt, base_all, ours_all])

    axes[1].set_ylabel("")
    fig.suptitle(title if title is not None else "Trajectory Comparison")
    if handles_r:
        fig.legend(handles_r, labels_r, loc="lower center", ncol=4,
                   frameon=True, framealpha=0.9)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[viz] wrote {save_path}")
    return fig, axes


def _draw_panel(ax, obs, gt, samples, best, best_style, panel_title, units):
    """Draw history + GT + sample bundle + highlighted best on one axis.

    Returns the artist handles and labels used for the figure-level legend.
    """
    ax.set_title(panel_title)
    (h_hist,) = ax.plot(obs[:, 0], obs[:, 1], zorder=4, **STYLE["history"])
    ax.scatter(obs[-1, 0], obs[-1, 1], color="black", marker="s", s=70,
               zorder=5)
    (h_gt,) = ax.plot(gt[:, 0], gt[:, 1], zorder=4, **STYLE["gt"])

    h_sample = None
    for i, hyp in enumerate(np.asarray(samples)):
        kwargs = {"label": "Predicted samples" if i == 0 else "_nolegend_"}
        (artist,) = ax.plot(hyp[:, 0], hyp[:, 1],
                            color=best_style["color"], alpha=0.15,
                            zorder=2, **kwargs, **SAMPLE_STYLE_BASE)
        if h_sample is None:
            h_sample = artist

    handles = [h_hist, h_gt]
    if h_sample is not None:
        handles.append(h_sample)
    if best is not None:
        (h_best,) = ax.plot(best[:, 0], best[:, 1], zorder=5, **best_style)
        handles.append(h_best)
    labels = [h.get_label() for h in handles]

    ax.set_xlabel(f"x [{units}]")
    return handles, labels


def _set_shared_data_limits(axes, arrays) -> None:
    """Give every axis the same limits spanning all plotted trajectories."""
    pts = np.concatenate([np.asarray(a).reshape(-1, 2) for a in arrays])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    lo, hi = lo - 0.08 * span, hi + 0.08 * span
    for ax_ in axes:
        ax_.set_xlim(lo[0], hi[0])
        ax_.set_ylim(lo[1], hi[1])


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
# Background loading: the EXACT video frame at the present timestep
# ---------------------------------------------------------------------------
def _imread_rgb(path: Path) -> np.ndarray:
    """Read an image as ``[H, W, 3]`` array (matplotlib or cv2 fallback)."""
    try:
        img = plt.imread(str(path))
    except Exception:
        import cv2  # matplotlib covers most formats; cv2 is the safety net

        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    return img


def _decode_video_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    """Decode a single frame via cv2 seek (no whole-video loading)."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx = min(max(int(frame_idx), 0), max(total - 1, 0))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
        if not ok:
            raise IOError(f"failed to read frame {idx} of {video_path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()  # free the handle immediately


def load_anchor_frame(
    sdd_root: str | Path | None,
    scene: str,
    video_id: str,
    anchor_frame: int,
    override: str | Path | None = None,
) -> tuple[np.ndarray, tuple[float, float, float, float], str] | None:
    """Load the video frame backing the plotted window's coordinates.

    SDD annotations (and therefore the dataloader's absolute trajectories)
    are pixel coordinates of the source video, whose resolution never changes
    within a clip — so the frame at ``anchor_frame`` (the last observation
    step) is the exact spatial context for this sample.

    Resolution order (first hit wins):
      1. explicit ``override`` path (--background);
      2. pre-extracted JPEG ``<root>/videos/<scene>/<vid>/frames/
         frame{anchor:06d}.jpg`` (+1 variant tolerates 0-based annotation
         ids against 1-based extraction);
      3. static ``referenceframe.jpg`` (official SDD layout, then videos/);
      4. direct cv2 seek-decode from the raw video at ``anchor_frame``.

    Returns ``(image, extent=(0, W, H, 0), source_tag)`` or ``None``.
    """
    candidates: list[tuple[str, Path]] = []
    if override is not None:
        candidates.append(("override", Path(override)))
    root: Path | None = None
    if sdd_root is not None:
        root = Path(expand_sdd_root(sdd_root))
        frames_dir = root / "videos" / scene / video_id / "frames"
        candidates.extend(
            [
                ("exact-extracted", frames_dir / f"frame{int(anchor_frame):06d}.jpg"),
                ("exact-extracted+1", frames_dir / f"frame{int(anchor_frame) + 1:06d}.jpg"),
                ("reference", root / "frames" / scene / video_id / "referenceframe.jpg"),
                ("reference-videos", root / "videos" / scene / video_id / "referenceframe.jpg"),
            ]
        )
    for source, cand in candidates:
        if cand.exists():
            img = _imread_rgb(cand)
            logger.info("[%s/%s] background (%s): %s", scene, video_id,
                        source, cand)
            return img, (0.0, float(img.shape[1]), float(img.shape[0]), 0.0), source

    if root is not None:
        video = video_mov_path(root, scene, video_id)
        if not video.exists():  # tolerate non-canonical extensions
            alt = video.with_suffix(".mp4")
            video = alt if alt.exists() else video
        try:
            img = _decode_video_frame(video, int(anchor_frame))
            logger.info("[%s/%s] background (video-decode @%d): %s",
                        scene, video_id, int(anchor_frame), video)
            h, w = img.shape[:2]
            return img, (0.0, float(w), float(h), 0.0), "video-decode"
        except Exception as exc:
            logger.warning("[%s/%s] video frame extraction failed: %s",
                           scene, video_id, exc)

    logger.warning(
        "[%s/%s] no frame found; plotting without background", scene, video_id
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
    to resolve the exact background frame.
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
    pixel anchor required to map them onto the video frame.
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
        anchor_frame = batch_cpu.get("anchor_frame", None)
        anchor_frame = int(anchor_frame[0]) if anchor_frame is not None else 0
        background = extent = None
        if not getattr(args, "no_background", False):
            loaded = load_anchor_frame(
                args.sdd_root, scene, video_id, anchor_frame,
                override=getattr(args, "background", None),
            )
            if loaded is not None:
                background, extent, _ = loaded
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
        extent=extent,
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
                        "automatic anchor-frame lookup)")
    p.add_argument("--no-background", action="store_true",
                   help="skip frame lookup and plot on white")
    p.add_argument("--demo", action="store_true",
                   help="render a synthetic styling demo (no checkpoints needed)")
    return p.parse_args(argv)


def run_demo(out_dir: str | Path) -> Path:
    """Styling self-check: synthetic distributions over a synthetic frame."""
    rng = np.random.default_rng(0)
    t = np.linspace(0, 4 * np.pi, 12)
    obs = np.stack([t / 8 + 20, np.sin(t / 2) + 30], axis=-1)
    gt = np.stack(
        [obs[-1, 0] + np.linspace(0, 3, 12), obs[-1, 1] + np.linspace(0, 2.5, 12)],
        axis=-1,
    )

    def bundle(scale: float) -> np.ndarray:
        modes = np.array([[0.0, 0.6], [1.4, 0.0]])  # bimodal spread
        out = np.empty((20, 12, 2))
        for k in range(20):
            mode = modes[k % 2] + rng.normal(scale=scale, size=2)
            out[k] = gt + mode[None, :] * np.linspace(0, 1, 12)[:, None]
            out[k] += rng.normal(scale=scale * 0.3, size=(12, 2))
        return out

    grad = np.linspace(0, 255, 96 * 72, dtype=np.uint8).reshape(72, 96)
    background = np.repeat(grad[..., None], 3, axis=-1)  # fake video frame
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_path = out / "demo_trajectory_comparison.png"
    plot_trajectory_comparison(
        obs, gt, bundle(0.35), bundle(0.35),
        baseline_best=bundle(0.35)[rng.integers(20)],
        ours_best=bundle(0.35)[rng.integers(20)],
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
