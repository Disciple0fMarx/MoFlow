"""Publication-ready trajectory visualization for the agent-centric SDD encoder.

Companion to ``tools/visualize_trajectory_comparison.py`` (the global-video
variant): given a scene, a pedestrian ``--track-id`` and a trained checkpoint,
this script samples the full ``K``-hypothesis future distribution from the
agent-centric model (``ETHMotionTransformer`` + ``FlowMatcher`` trained by
``fm_sdd_agent.py``) on that single pedestrian's observation window and renders
one figure:

* observed history (black dashed, last step marked with a square),
* the ground-truth future (green),
* the full multimodal prediction bundle (all ``K`` heads, low alpha),
* the min-ADE best hypothesis highlighted with a thick red line,

all overlaid on the EXACT video frame at the present timestep (the final
observation step, ``anchor_frame`` of the plotted window). Trajectories live
in the annotation pixel space of the source video, so overlaying is an exact
coordinate match (origin top-left, y down); discrete timesteps are marked
along every path.

The checkpoint's agent-crop backbone is auto-detected from its state dict
(``context_encoder.agent_video_encoder.features.*`` → ``CompactAgentVideoEncoder``,
``...backbone.*`` → ``AgentVideoEncoder``) or set explicitly via ``--agent-encoder``.
Both backbones are selected inside :class:`~models.context_encoder.eth_encoder.ETHEncoder`
through the ``AGENT_ENCODER_TYPE`` config knob; the tri-modal fusion module
(``cross_attn_agent``/``cross_attn_scene``) is part of the checkpoint and is
loaded automatically.

Crops are fetched by :class:`~data.agent_crop_sdd.SDDAgentCropDataset` on the
window dictated by the LOSO window index, so the trajectory features and the
visual crops refer to the exact same (scene, video, track, anchor) — the strict
coordinate synchronization that keeps crop centers aligned with the predicted
trajectory points. Trajectory features and
normalization statistics come from the native ``SDDGlobalDataset`` test split
(identical semantics to production evaluation; stats cached once per scene
under ``RESULTS_DIR/_norm_stats_ho<scene>.npz``).

Usage (lab machine)::

    # auto-detect the encoder from the checkpoint weights
    python viz_agent_sdd.py --scene coupa --track-id 3 \\
        --checkpoint results_sdd/cor_fm/_SDD_hocoupa_agent/models/checkpoint_best.pt

    # explicit compact encoder + tighter zoom + custom output dir
    python viz_agent_sdd.py --scene coupa --track-id 3 --video-id video0 \\
        --agent-encoder compact --zoom-margin 80 \\
        --checkpoint <path>/checkpoint_best.pt --out-dir visualizations

    # styling self-check (no checkpoint / dataset required)
    python viz_agent_sdd.py --demo

The figure is written to ``visualizations/<scene>_track<id>_agentvid_trajectory.png``
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
# Publication-quality rcParams (identical to the global-video viz script)
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
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from einops import rearrange  # noqa: E402

from data.agent_crop_sdd import SDDAgentCropDataset  # noqa: E402
from data.dataloader_sdd_global import (  # noqa: E402
    SDD_SCENES,
    SDDGlobalDataset,
    SDD_PAST_FRAMES,
)
from models.backbone_eth_ucy import ETHMotionTransformer  # noqa: E402
from models.flow_matching import FlowMatcher  # noqa: E402
from utils.config import Config  # noqa: E402
from utils.normalization import unnormalize_min_max  # noqa: E402
from video_encoder.sdd_adapter import expand_sdd_root  # noqa: E402

# Reuse the plotting core / checkpoint / frame helpers of the global-video viz
# script verbatim so both visualizations stay pixel-for-pixel consistent.
from tools.visualize_trajectory_comparison import (  # noqa: E402
    _as_bundle,
    _as_xy,
    _move_to_device,
    _quiet_logger,
    load_anchor_frame,
    load_checkpoint_state,
    select_best_of_k,
)

CFG_PATH = REPO_ROOT / "cfg" / "sdd" / "cor_fm.yml"
RESULTS_DIR = REPO_ROOT / "results_sdd" / "cor_fm"
DEFAULT_OUT_DIR = REPO_ROOT / "visualizations"

# Bundled samples: low-alpha lines exposing the multimodal distribution;
# the best hypothesis is a thick highlight line.
SAMPLE_STYLE_BASE = {"linestyle": "-", "linewidth": 0.9, "marker": "o",
                     "markersize": 2}
STYLE = {
    "history": {"color": "black", "linestyle": "--", "linewidth": 1.6,
                "marker": "o", "markersize": 3, "label": "History"},
    "gt": {"color": "green", "linestyle": "-", "linewidth": 1.8,
           "marker": "o", "markersize": 3, "label": "Ground truth"},
    "samples": {"color": "red", "linestyle": "-", "linewidth": 0.9,
                "marker": "o", "markersize": 2, "label": "Predicted samples"},
    "best": {"color": "red", "linewidth": 2.4, "alpha": 0.85,
             "marker": "o", "markersize": 3,
             "label": "Best-of-K (min-ADE)"},
}

logger = logging.getLogger("viz-agent")


def _quiet_logger() -> logging.Logger:
    """Ensure the module logger never spews to stdout/stderr by default."""
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.INFO)
    return logger


# ---------------------------------------------------------------------------
# Plotting core (single model panel)
# ---------------------------------------------------------------------------
def plot_agent_trajectory(
    obs_traj: np.ndarray,
    gt_traj: np.ndarray,
    samples: np.ndarray,
    best: np.ndarray | None = None,
    title: str | None = None,
    save_path: str | Path | None = None,
    background: np.ndarray | None = None,
    extent: tuple[float, float, float, float] | None = None,
    units: str = "m",
    margin: float = 150.0,
):
    """Render the agent-centric prediction vs ground truth on one panel.

    Args:
        obs_traj: ``[T, 2]`` observed history (pixel/annotation space).
        gt_traj: ``[F, 2]`` ground-truth future.
        samples: ``[K, F, 2]`` hypothesis bundle; every head is drawn as a
            low-alpha marker line so the multimodal FlowMatcher output stays
            visible. Predictions must already be in the same coordinate frame
            as ``obs_traj`` (absolute annotation pixels when a background is
            shown).
        best: optional ``[F, 2]`` min-ADE trajectory, drawn as the thick red
            highlight on top of the bundle.
        background: optional HxW(x3) image drawn behind the panel; requires
            ``extent=(x0, x1, y_bottom, y_top)`` in trajectory coordinates
            (SDD pixel overlays use ``(0, W, H, 0)``: origin top-left, y down).
        units: axis-unit suffix (``px`` when a background is overlaid).
        margin: contextual margin (image pixels) added around the union
            bounding box of ALL plotted trajectories before zooming; the
            padded box is clamped to the frame extent.

    In image space the Y-axis is inverted (limits handed to matplotlib
    bottom-value-first) so the frame renders upright, exactly like the
    global-video comparison panels.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    obs = _as_xy(obs_traj)
    gt = _as_xy(gt_traj)
    all_samples = _as_bundle(samples)

    # ---- background FIRST: imshow before any trajectory is plotted --------
    if background is not None:
        if extent is None:  # default: full image, origin top-left (y down)
            extent = (0.0, float(background.shape[1]),
                      float(background.shape[0]), 0.0)
        ax.imshow(background, extent=extent, origin="upper", zorder=0,
                  interpolation="bilinear", aspect="equal")

    # ---- trajectories ------------------------------------------------------
    (h_hist,) = ax.plot(obs[:, 0], obs[:, 1], zorder=4, **STYLE["history"])
    ax.scatter(obs[-1, 0], obs[-1, 1], color="black", marker="s", s=70,
               zorder=5)
    (h_gt,) = ax.plot(gt[:, 0], gt[:, 1], zorder=4, **STYLE["gt"])

    h_sample = None
    sample_style = {k: v for k, v in STYLE["samples"].items() if k != "label"}
    for i, hyp in enumerate(np.asarray(all_samples)):
        label = STYLE["samples"]["label"] if i == 0 else "_nolegend_"
        (artist,) = ax.plot(
            hyp[:, 0], hyp[:, 1], alpha=0.15, zorder=2, label=label,
            **sample_style,
        )
        if h_sample is None:
            h_sample = artist

    handles = [h_hist, h_gt]
    if h_sample is not None:
        handles.append(h_sample)
    if best is not None:
        (h_best,) = ax.plot(best[:, 0], best[:, 1], zorder=5, **STYLE["best"])
        handles.append(h_best)

    # ---- limits LAST: dynamic zoom ----------------------------------------
    if background is not None:
        x0, x1, yb, yt = extent
        y_lo, y_hi = min(yb, yt), max(yb, yt)
        lo, hi = _padded_bbox([obs, gt, all_samples], margin)
        xmin = max(x0, lo[0])
        xmax = min(x1, hi[0])
        ymin = max(y_lo, lo[1])
        ymax = min(y_hi, hi[1])
        if xmax - xmin < 1.0:  # degenerate after clamping -> keep sane span
            c = 0.5 * (xmin + xmax)
            xmin, xmax = c - 0.5, c + 0.5
        if ymax - ymin < 1.0:
            c = 0.5 * (ymin + ymax)
            ymin, ymax = c - 0.5, c + 0.5
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymax, ymin)  # inverted Y so the frame renders upright
        ax.set_adjustable("box")
    else:
        _set_data_limits(ax, [obs, gt, all_samples])

    ax.set_xlabel(f"x [{units}]")
    ax.set_ylabel(f"y [{units}]")
    if title is not None:
        ax.set_title(title)
    ax.legend(handles=handles, loc="best", frameon=True, framealpha=0.9)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[viz] wrote {save_path}")
    return fig, ax


def _padded_bbox(
    arrays: list[np.ndarray], margin: float
) -> tuple[np.ndarray, np.ndarray]:
    """Union bounding box of all arrays (+ ``margin`` on every side)."""
    pts = np.concatenate([np.asarray(a).reshape(-1, 2) for a in arrays])
    lo = pts.min(axis=0) - margin
    hi = pts.max(axis=0) + margin
    return lo, hi


def _set_data_limits(ax, arrays) -> None:
    """Give the axis limits spanning all plotted trajectories."""
    pts = np.concatenate([np.asarray(a).reshape(-1, 2) for a in arrays])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    lo, hi = lo - 0.08 * span, hi + 0.08 * span
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])


# ---------------------------------------------------------------------------
# Config construction (mirror of fm_sdd_agent.init_basics defaults)
# ---------------------------------------------------------------------------
def _build_cfg(scene: str, args: argparse.Namespace) -> Config:
    """Build the project ``Config`` for the agent-centric model.

    Mirrors every override applied by ``fm_sdd_agent.init_basics`` using the
    same default values as the training CLI, then injects the SDD knobs, the
    agent-crop backbone selection and the normalization statistics required
    at evaluation time. The checkpoint's agent encoder must be resolved BEFORE
    this call (via ``--agent-encoder`` or auto-detection) so the built
    architecture matches the stored weights.
    """
    cfg = Config(str(CFG_PATH), tag="viz")

    # ---- FM / arch / loss / optimization (defaults of fm_sdd_agent.py) ----
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
    cfg.LOSS_REG_SQUARED = False
    cfg.LOSS_VELOCITY = False
    cfg.rotate = False
    cfg.rotate_aug = False
    cfg.data_norm = "min_max"

    # ---- SDD knobs ----------------------------------------------------------
    ce = cfg.MODEL.CONTEXT_ENCODER
    ce.SDD_ROOT = str(expand_sdd_root(args.sdd_root))
    ce.HELD_OUT_SCENE = scene

    # ---- Agent-centric branch (mirror of fm_sdd_agent.init_basics) ----------
    ce.USE_VIDEO = False
    ce.USE_AGENT_VIDEO = True
    ce.USE_TRI_MODAL_FUSION = True
    ce.AGENT_CROP_SIZE = [args.crop_size, args.crop_size]
    ce.AGENT_ENCODER_TYPE = args.agent_encoder

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

    Identical semantics (and the SAME on-disk cache) as the global-video viz
    script: statistics are derived from the training windows exactly as in
    production, are deterministic across runs, and are cached to
    ``RESULTS_DIR/_norm_stats_ho<scene>.npz``.
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


# ---------------------------------------------------------------------------
# Agent-encoder auto-detection + checkpoint loading
# ---------------------------------------------------------------------------
def _resolve_agent_encoder(
    args: argparse.Namespace, ckpt_state: dict
) -> str:
    """Return the ``ETHEncoder.AGENT_ENCODER_TYPE`` for the checkpoint.

    ``--agent-encoder compact|resnet18`` wins; ``auto`` inspects the stored
    state dict keys: ``context_encoder.agent_video_encoder.features.*`` belongs
    to ``CompactAgentVideoEncoder``, ``...backbone.*`` to ``AgentVideoEncoder``.
    """
    if args.agent_encoder != "auto":
        return args.agent_encoder
    keys = ckpt_state.keys() if isinstance(ckpt_state, dict) else ()
    features = any(
        k.startswith("context_encoder.agent_video_encoder.features.") for k in keys
    )
    backbone = any(
        k.startswith("context_encoder.agent_video_encoder.backbone.") for k in keys
    )
    if features:
        print("[viz] checkpoint uses CompactAgentVideoEncoder (features.* keys).")
        return "compact"
    if backbone:
        print("[viz] checkpoint uses AgentVideoEncoder (backbone.* keys).")
        return "resnet18"
    print("[viz] no agent-video weights found in checkpoint; defaulting to resnet18.")
    return "resnet18"


def build_model_and_cfg(scene: str, args: argparse.Namespace):
    """Instantiate ``ETHMotionTransformer`` + ``FlowMatcher`` natively.

    Mirrors ``fm_sdd_agent.build_network``; loads the given checkpoint onto
    the device and switches the denoiser to eval mode. The agent-crop backbone
    is auto-detected from the checkpoint before the model is built.
    """
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    ckpt_state = load_checkpoint_state(ckpt_path, use_ema=args.use_ema)
    encoder_type = _resolve_agent_encoder(args, ckpt_state)
    args.agent_encoder = encoder_type

    cfg = _build_cfg(scene, args)
    log = _quiet_logger()

    model = ETHMotionTransformer(model_config=cfg.MODEL, logger=log, config=cfg)
    denoiser = FlowMatcher(cfg, model, logger=log)

    denoiser.load_state_dict(ckpt_state)
    denoiser.to(cfg.device)
    denoiser.eval()
    logger.info("[%s/%s] loaded %s", scene, encoder_type, ckpt_path.name)
    return denoiser, cfg


# ---------------------------------------------------------------------------
# Window batch construction for one (scene, track_id)
# ---------------------------------------------------------------------------
def get_window_batch(
    scene: str, track_id: int, args: argparse.Namespace
) -> tuple[dict, str, str, int]:
    """Assemble the single-window model batch for ``track_id`` in ``scene``.

    Finds the first LOSO test-split window owned by ``track_id`` from the
    native ``SDDGlobalDataset`` window index (optionally disambiguated by
    ``--video-id``), then fetches its agent-centric crops through
    :class:`data.agent_crop_sdd.SDDAgentCropDataset` on the EXACT observed
    frame ids. Returns ``(batch, video_id, anchor_frame, track_id)`` where the
    batch mirrors ``fm_sdd_agent.collate_sdd_agent`` output:
    ``agent_crops`` [1, 1, T_obs, C, S, S] float32, plus the shared trajectory
    tensors and metadata.
    """
    cfg = _build_cfg(scene, args)  # carries norm stats
    dset = SDDGlobalDataset(
        cfg,
        training=False,
        sdd_root=args.sdd_root,
        held_out_scene=scene,
        split="test",
        use_video=False,
    )

    # ---- locate the window ------------------------------------------------
    rows = dset.windows.rows["track_id"]
    idcs = np.where(rows == int(track_id))[0]
    row_pos = None
    for i in idcs:
        vid = dset.windows.video_id(int(i))
        if args.video_id is not None and vid != args.video_id:
            continue
        row_pos = int(i)
        break
    if row_pos is None:
        del dset
        raise ValueError(
            f"track_id={track_id} not found in scene {scene} "
            f"(video_id={args.video_id!r}); run --scene with an existing track."
        )

    video_id = dset.windows.video_id(row_pos)
    anchor = dset.windows.anchor_frame(row_pos)
    past_fids = dset.windows.past_frame_ids(row_pos)

    item = dset[row_pos]  # trajectory features, normalized per cfg stats
    del dset  # free window index + materialized trajectory tensors
    gc.collect()

    # ---- agent-centric crops on the exact observed frames ------------------
    crop_ds = SDDAgentCropDataset(
        expand_sdd_root(args.sdd_root),
        scene,
        video_id,
        track_ids=[int(track_id)],
        past_frames_per_window=[past_fids.tolist()],
        crop_size=int(args.crop_size),
        obs_frames=SDD_PAST_FRAMES,
        padding=int(args.padding),
        drop_lost=args.drop_lost,
    )
    crop_item = crop_ds[0]
    crops = crop_item["agent_crops"]  # [T_obs, C, S, S] uint8
    del crop_ds
    gc.collect()

    # ---- pack the exact collate_sdd_agent contract (B=1, A=1) ---------------
    batch = {
        "past_traj": item["past_traj"].unsqueeze(0),                    # [1,1,P,6]
        "fut_traj": item["fut_traj"].unsqueeze(0),                     # [1,1,F,2]
        "past_traj_original_scale": item["past_traj_original_scale"].unsqueeze(0),
        "fut_traj_original_scale": item["fut_traj_original_scale"].unsqueeze(0),
        "fut_traj_vel": item["fut_traj_vel"].unsqueeze(0),
        "index": torch.tensor([row_pos], dtype=torch.int32),
        "batch_size": torch.tensor(1),
        "scene": [scene],
        "video_id": [video_id],
        "anchor_frame": torch.tensor([anchor], dtype=torch.int32),
        # [B, A=1, T_obs, C, S, S] float32, values in [0, 255] (like training)
        "agent_crops": crops.unsqueeze(0).unsqueeze(1).to(dtype=torch.float32),
        "agent_crops_valid": torch.tensor([True]),
        "z_video_global": None,  # agent branch: scene-level video OFF
    }
    return batch, video_id, anchor


# ---------------------------------------------------------------------------
# Inference + best-of-K
# ---------------------------------------------------------------------------
def sample_prediction(denoiser, cfg: Config, batch_cpu: dict) -> dict[str, np.ndarray]:
    """Sample the full K-hypothesis futures for the single-window batch.

    Everything runs under ``torch.inference_mode()``; GPU tensors are
    detached and converted to NumPy before returning so no CUDA memory
    outlives this call. Predictions are returned in the relative frame
    (origin = last observed position); ``obs_abs`` provides the absolute
    pixel anchor required to map them onto the video frame.
    """
    device = cfg.device
    with torch.inference_mode():
        x_gpu = _move_to_device(batch_cpu, device)
        # returns (final_y [B,K,A,F*D], states, t, y_t, score)
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


def _to_absolute(arrs: list[np.ndarray], anchor: np.ndarray) -> list[np.ndarray]:
    """Shift relative-frame arrays into annotation pixel space."""
    return [arr + anchor[None, :] for arr in arrs]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_track(scene: str, track_id: int, args: argparse.Namespace) -> Path | None:
    """Render one agent-centric trajectory figure for (scene, track_id)."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the agent-crop backbone from the checkpoint ONCE so both the
    # dataset cfg and the built model use the identical AGENT_ENCODER_TYPE.
    if args.agent_encoder == "auto":
        state = load_checkpoint_state(Path(args.checkpoint), use_ema=args.use_ema)
        args.agent_encoder = _resolve_agent_encoder(args, state)

    batch_cpu, video_id, anchor_frame = get_window_batch(scene, track_id, args)
    try:
        denoiser, cfg = build_model_and_cfg(scene, args)
        try:
            result = sample_prediction(denoiser, cfg, batch_cpu)
        finally:
            # free the model before loading the next one (single-track anyway)
            del denoiser, cfg
            gc.collect()
            torch.cuda.empty_cache()

        background = extent = None
        if not args.no_background:
            # NOTE: resolve the platform default HERE — args.sdd_root is None
            # when relying on the lab/Kaggle default.
            resolved_root = expand_sdd_root(args.sdd_root)
            loaded = load_anchor_frame(
                resolved_root, scene, video_id, anchor_frame,
                override=args.background,
            )
            if loaded is not None:
                background, extent, _ = loaded
    finally:
        del batch_cpu
        gc.collect()
        torch.cuda.empty_cache()

    obs, gt = result["obs"][0], result["gt"][0]
    all_samples = result["preds_all"][0]
    best = result["pred_best"][0]

    if background is not None:
        # Relative trajectories anchored by the last observed (ABSOLUTE) point.
        anchor = result["obs_abs"][-1]
        obs = obs + anchor[None, :]
        gt, best = _to_absolute([gt, best], anchor)
        all_samples = _to_absolute([all_samples], anchor)[0]
        units = "px"
    else:
        units = "m"

    save_path = out_dir / f"{scene}_track{track_id}_agentvid_trajectory.png"
    plot_agent_trajectory(
        obs,
        gt,
        all_samples,
        best=best,
        title=f"SDD '{scene}' — track {track_id} ({video_id}) — Agent-centric VE",
        save_path=save_path,
        background=background,
        extent=extent,
        units=units,
        margin=float(args.zoom_margin),
    )
    return save_path


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scene", default=None, choices=SDD_SCENES,
                   help="held-out SDD scene (default: first test scene)")
    p.add_argument("--track-id", type=int, default=None,
                   help="pedestrian track_id in the scene's annotation windows")
    p.add_argument("--video-id", default=None,
                   help="disambiguate the track when the id appears in several videos")
    p.add_argument("--checkpoint", default=None,
                   help="path to the trained checkpoint (checkpoint_best.pt)")
    p.add_argument("--agent-encoder", dest="agent_encoder",
                   default="auto", choices=["auto", "resnet18", "compact"],
                   help="agent-crop backbone; auto = detected from the checkpoint keys")
    p.add_argument("--device", default=None, help="'cuda', 'cpu' (default: auto)")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--sdd-root", default=None)
    p.add_argument("--crop-size", type=int, default=64)
    p.add_argument("--padding", type=int, default=0)
    p.add_argument("--drop-lost", action="store_true",
                   help="Strict opt-in: refusal of crops around 'lost' frames "
                        "(default keeps them, matching training).")
    p.add_argument("--sampling-steps", type=int, default=10)
    p.add_argument("--use-ema", action="store_true",
                   help="prefer the EMA weights over the raw model weights")
    p.add_argument("--background", default=None,
                   help="explicit path to a background image (overrides the "
                        "automatic anchor-frame lookup)")
    p.add_argument("--no-background", action="store_true",
                   help="skip frame lookup and plot on white")
    p.add_argument("--zoom-margin", type=float, default=150.0,
                   help="contextual margin in pixels added around the union "
                        "bounding box of all plotted trajectories when a "
                        "video frame background is shown (default: 150)")
    p.add_argument("--demo", action="store_true",
                   help="render a synthetic styling demo (no checkpoints needed)")
    return p.parse_args(argv)


def run_demo(out_dir: str | Path) -> Path:
    """Styling self-check: synthetic distribution over a synthetic frame."""
    rng = np.random.default_rng(0)
    t = np.linspace(0, 4 * np.pi, SDD_PAST_FRAMES)
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
    save_path = out / "demo_agent_trajectory.png"
    plot_agent_trajectory(
        obs, gt, bundle(0.35), best=bundle(0.35)[rng.integers(20)],
        title="Demo — Agent-centric VE",
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

    if args.scene is None:
        args.scene = SDD_SCENES[0]
    if args.track_id is None:
        raise SystemExit("--track-id is required (or run with --demo).")
    if args.checkpoint is None:
        raise SystemExit("--checkpoint is required (or run with --demo).")

    print(f"[viz] === scene: {args.scene}, track: {args.track_id} ===")
    saved = run_track(args.scene, args.track_id, args)
    print(f"[viz] done: {saved}")


if __name__ == "__main__":
    main()