"""Offline qualitative figure generator for the Video-MoFlow SDD ablation.

For each selected test window this tool renders five publication-grade, 300-DPI
PNGs comparing the three SDD model variants:

* ``off``    — trajectory-only baseline (``_SDD_ho<scene>_novid``), no video;
* ``static`` — static scene-prior video model (``_SDD_ho<scene>_vmstatic``);
* ``full``   — per-window video model (``_SDD_ho<scene>_vmfull``).

Every figure shows the observed history (``P=8``), the full best-of-``K=20``
prediction bundle from the flow-matching sampler, the min-ADE best hypothesis
and the ground-truth future (``F=12``). For ``static``/``full`` two figures are
produced per sample — one over the raw scene frame (``*_video.png``) and one
with a ResNet-18 Grad-CAM heatmap alpha-blended on top (``*_gradcam.png``).

Two evaluation protocols are supported and written to disjoint trees::

    results_sdd/visualizations/
    ├── global/<scene>/sample_001_no_video.png ...
    └── video0/<scene>_video0/sample_001_no_video.png ...

Coordinates & units
-------------------
SDD annotations are pixel coordinates of the source video and the project uses
an identity homography (pixels ≈ metres, see ``SDDGlobalDataset``'s note), so
axis limits are the physical scene coordinates in the same units used by the
benchmark; they are labelled ``[m]`` accordingly. Figures with a video
background are zoomed to the union bounding box of all plotted trajectories
(clamped to the frame extent); background-free figures use a padded data box.

Fidelity notes
--------------
* Predictions are produced by the *native* ``SDDGlobalDataset`` test split per
  variant (``video_mode`` = ``off``/``static``/``full``) and the correct
  checkpoint, so the conditioning vector equals production evaluation.
* ``static`` is NOT a literal single keyframe: the dataloader serves one fixed
  scene-level vector (the mean ResNet-18 feature over up to 8 coarsely-spaced
  frames of the first reachable video). The Grad-CAM pass therefore reproduces
  those exact keyframes, feeds their mean, and renders the mean per-keyframe
  CAM over the window's anchor frame (same static scene geometry). If the
  feature cache is unavailable it degrades to a single anchor keyframe.
* The research suite deletes ``vmstatic`` checkpoints after the Q6 aggregation
  (``scripts/research/00_common.sh:cleanup_run_checkpoints``); missing variants
  are skipped with an actionable message instead of crashing.

This is a standalone evaluation/plotting utility: it never trains and never
touches the training loops.

Example::

    python tools/generate_qualitative_figures.py \\
        --protocol both --num-samples 5 \\
        --sdd-root /home/efrei_stage/Desktop/Datasets/SDD \\
        --video-features-root ./features/resnet18
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange
from PIL import Image

# Importing the viz module installs the shared publication rcParams (serif
# fonts, print-readable sizes, 300-DPI savefig) used by every figure here.
import tools.visualize_trajectory_comparison as viz
from data.dataloader_sdd_global import (
    SDD_PAST_FRAMES,
    SDD_SCENES,
    SDDGlobalDataset,
    collate_sdd_global,
)
from models.backbone_eth_ucy import ETHMotionTransformer
from models.flow_matching import FlowMatcher
from tools import gradcam_sdd as gcam
from tools.visualize_trajectory_comparison import (
    RESULTS_DIR,
    SAMPLE_STYLE_BASE,
    STYLE,
    _build_cfg,
    load_anchor_frame,
    load_checkpoint_state,
    select_best_of_k,
)
from utils.normalization import unnormalize_min_max
from video_encoder.sdd_adapter import expand_sdd_root, get_lookup

SCENES: tuple[str, ...] = tuple(SDD_SCENES)
VARIANTS: tuple[str, ...] = ("off", "static", "full")
VARIANT_SUFFIX = {"off": "novid", "static": "vmstatic", "full": "vmfull"}
VARIANT_USE_VIDEO = {"off": False, "static": True, "full": True}
VARIANT_LABEL = {
    "off": "Trajectory-only (No Video)",
    "static": "Static Video-MoFlow",
    "full": "Full Video-MoFlow",
}
VARIANT_COLOR = {"off": "#1f77b4", "static": "#ff7f0e", "full": "#d62728"}
VARIANT_TAG = {
    "off": "no_video",
    "static": "static_video",
    "full": "full_video",
}
VARIANT_GRADCAM_TAG = {"static": "static_gradcam", "full": "full_gradcam"}

DEFAULT_OUT_DIR = _REPO_ROOT / "results_sdd" / "visualizations"


class _NullLogger:
    def info(self, *a, **k) -> None:  # noqa: ANN002, ARG002
        pass

    def warning(self, *a, **k) -> None:  # noqa: ANN002, ARG002
        pass


def _console(msg: str) -> None:
    print(f"[qual] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Checkpoint / model / dataset construction (one variant at a time)
# ---------------------------------------------------------------------------
def variant_checkpoint(scene: str, variant: str, video_id: str | None) -> Path:
    """``results_sdd/cor_fm/_SDD_ho<scene>_<suffix>[_vid<id>]/models/checkpoint_best.pt``."""
    suffix = VARIANT_SUFFIX[variant]
    out_suf = f"_vid{video_id}" if video_id else ""
    run_dir = RESULTS_DIR / f"_SDD_ho{scene}_{suffix}{out_suf}" / "models"
    best = run_dir / "checkpoint_best.pt"
    if best.exists():
        return best
    # The Q6 cleanup removes *all* checkpoints; before that, the trainer keeps
    # ``checkpoint_best.pt``. Fall back to any surviving checkpoint so a
    # partially-cleaned run dir still works.
    for cand in sorted(run_dir.glob("checkpoint_*.pt")):
        return cand
    return best


def load_variant_denoiser(
    scene: str, variant: str, args: argparse.Namespace
) -> tuple[FlowMatcher, object]:
    """Build + load the denoiser for one variant (``off``/``static``/``full``)."""
    vid_status = "novid" if variant == "off" else "vid"
    cfg = _build_cfg(scene, vid_status, args)
    model = ETHMotionTransformer(
        model_config=cfg.MODEL, logger=_NullLogger(), config=cfg
    )
    denoiser = FlowMatcher(cfg, model, logger=_NullLogger())
    ckpt = variant_checkpoint(scene, variant, getattr(args, "video_id", None))
    state = load_checkpoint_state(ckpt, use_ema=args.use_ema)
    denoiser.load_state_dict(state)
    denoiser.to(cfg.device)
    denoiser.eval()
    return denoiser, cfg


def build_variant_dataset(scene: str, variant: str, args: argparse.Namespace):
    """Native test split for one variant (conditioning == production eval)."""
    cfg = _build_cfg(scene, "novid" if variant == "off" else "vid", args)
    dset = SDDGlobalDataset(
        cfg,
        training=False,
        sdd_root=args.sdd_root,
        held_out_scene=scene,
        split="test",
        use_video=VARIANT_USE_VIDEO[variant],
        video_features_root=args.video_features_root,
        video_mode=variant,
        video_ids=[args.video_id] if getattr(args, "video_id", None) else None,
    )
    return dset, cfg


# ---------------------------------------------------------------------------
# Sampling a window's K-hypothesis bundle
# ---------------------------------------------------------------------------
def sample_window_predictions(
    denoiser: FlowMatcher,
    cfg,
    batch: dict,
    device: str,
    seed: int,
) -> dict:
    """Sample K futures for one collated window and return host NumPy arrays."""
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    with torch.inference_mode():
        x_gpu = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
        out = denoiser.sample(x_gpu, num_trajs=int(cfg.denoising_head_preds))
        pred = rearrange(out[0], "b k a (f d) -> b k a f d", f=cfg.future_frames)[
            ..., :2
        ]
        if cfg.get("data_norm", "min_max") == "min_max":
            pred = unnormalize_min_max(pred, cfg.fut_traj_min, cfg.fut_traj_max, -1, 1)
        preds_np = rearrange(
            pred.detach().cpu().numpy(), "b k a f d -> (b a) k f d"
        ).astype(np.float64)

    gt_np = rearrange(
        batch["fut_traj_original_scale"].numpy(), "b a f d -> (b a) f d"
    ).astype(np.float64)
    # past_full columns: [abs_x, abs_y, rel_x, rel_y, vel_x, vel_y]; obs/gt/preds
    # share the RELATIVE frame (origin = last observed point), while cols 0:2
    # give the absolute pixel anchor used to place them on a video frame.
    past_full = batch["past_traj_original_scale"].numpy()
    obs_np = rearrange(past_full[..., 2:4], "b a p d -> (b a) p d").astype(np.float64)[
        0
    ]
    obs_abs = rearrange(past_full[..., 0:2], "b a p d -> (b a) p d").astype(np.float64)[
        0
    ]

    best, best_idx = select_best_of_k(preds_np, gt_np)
    ade = float(
        np.linalg.norm(preds_np[0] - gt_np[0], axis=-1).mean(axis=-1)[best_idx[0]]
    )
    return {
        "obs": obs_np,
        "obs_abs": obs_abs,
        "gt": gt_np[0],
        "preds": preds_np[0],
        "best": best[0],
        "best_idx": int(best_idx[0]),
        "ade": ade,
    }


def _to_absolute(res: dict, anchor: np.ndarray) -> dict:
    """Shift relative-frame arrays into absolute annotation pixel space."""
    out = dict(res)
    for key in ("obs", "gt", "preds", "best"):
        out[key] = res[key] + anchor[None, :]
    return out


# ---------------------------------------------------------------------------
# Frame handling (observed window + static keyframes)
# ---------------------------------------------------------------------------
def _decode_frame_ids(
    sdd_root, scene: str, video_id: str, frame_ids
) -> list[np.ndarray | None]:
    """Decode frames by id (replicating the last success across gaps)."""
    frames: list[np.ndarray | None] = []
    latest: np.ndarray | None = None
    for fid in frame_ids:
        got = load_anchor_frame(sdd_root, scene, video_id, int(fid))
        if got is not None:
            latest = got[0]
        frames.append(latest)
    return frames


def observed_frame_ids(anchor: int) -> list[int]:
    """The ``P=8`` past frame ids ``[anchor-7 .. anchor]`` (clamped at 0)."""
    return [
        max(int(anchor) - SDD_PAST_FRAMES + 1 + i, 0) for i in range(SDD_PAST_FRAMES)
    ]


def static_keyframe_ids(
    scene: str, video_id: str, args: argparse.Namespace
) -> list[int]:
    """Reproduce ``SDDGlobalDataset._static_video``'s coarse keyframe sample.

    Returns the exact frame ids whose cached ResNet-18 feature mean forms the
    static scene prior for ``(scene, video_id)`` (up to 8, stride ``N//8``).
    """
    root = args.video_features_root
    if root is None:
        return []
    try:
        lookup = get_lookup(root, scene, sdd_root=expand_sdd_root(args.sdd_root))
        ids = lookup.video_sorted_ids.get(video_id)
    except Exception as exc:  # missing/corrupt cache → caller falls back
        _console(f"static keyframe lookup failed for {scene}/{video_id}: {exc}")
        return []
    if ids is None or not len(ids):
        return []
    sample = ids[:: max(1, len(ids) // 8)][:8]
    return [int(x) for x in sample]


def frames_to_tensor(
    frames_rgb: list[np.ndarray | None],
    preprocess,
    expected: int | None = None,
) -> torch.Tensor | None:
    """Stack decoded RGB frames into a normalized ``[T, 3, 224, 224]`` tensor."""
    imgs: list[np.ndarray] = []
    latest: np.ndarray | None = None
    for f in frames_rgb:
        if f is not None:
            latest = f
        if latest is None:
            return None
        imgs.append(latest)
    if expected is not None and len(imgs) != expected:
        while len(imgs) < expected:
            imgs.append(imgs[-1])
    return torch.stack([preprocess(Image.fromarray(im)) for im in imgs])


# ---------------------------------------------------------------------------
# Grad-CAM for a variant (full: 8 observed frames; static: keyframe prior)
# ---------------------------------------------------------------------------
def variant_gradcam(
    variant: str,
    scene: str,
    batch: dict,
    anchor: int,
    video_id: str,
    denoiser: FlowMatcher,
    cfg,
    backbone,
    preprocess,
    device: str,
    seed: int,
    static_frames: list[np.ndarray | None] | None,
    sdd_root,
) -> tuple[np.ndarray | None, int | None]:
    """Return ``(cam2d, gradcam_best_idx)`` or ``(None, None)`` if no frames.

    * ``full``   — CAM of the anchor (last observed) frame from the 8-frame batch.
    * ``static`` — mean CAM over the reproduced scene-prior keyframes.
    """
    if variant == "full":
        fids = observed_frame_ids(anchor)
        frames = _decode_frame_ids(sdd_root, scene, video_id, fids)
        frames_t = frames_to_tensor(frames, preprocess, expected=SDD_PAST_FRAMES)
        cam_pick = "last"
    else:  # static
        frames = static_frames
        if frames is None or not any(f is not None for f in frames):
            # Fall back to a single anchor keyframe (user-facing "P=1" case).
            frames = _decode_frame_ids(sdd_root, scene, video_id, [anchor])
        frames_t = frames_to_tensor(frames, preprocess)
        cam_pick = "mean"

    if frames_t is None:
        return None, None

    x_data = {
        "past_traj_original_scale": batch["past_traj_original_scale"].to(device),
        "fut_traj": batch["fut_traj"].to(device),
    }
    gt = batch["fut_traj_original_scale"].to(device).float()

    res = gcam.gradcam_pass(
        backbone,
        denoiser,
        cfg,
        x_data,
        frames_t,
        gt,
        int(cfg.future_frames),
        device,
        seed,
    )
    cam = res["cam"].numpy()  # [T, 7, 7]
    cam2d = cam[-1] if cam_pick == "last" else cam.mean(axis=0)
    return cam2d, int(res["best_idx"])


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _zoom_limits(ax, arrays, background, extent, margin):
    pts = np.concatenate([np.asarray(a).reshape(-1, 2) for a in arrays])
    if background is not None:
        x0, x1, yb, yt = extent
        y_lo, y_hi = min(yb, yt), max(yb, yt)
        lo = pts.min(axis=0) - margin
        hi = pts.max(axis=0) + margin
        xmin, xmax = max(x0, lo[0]), min(x1, hi[0])
        ymin, ymax = max(y_lo, lo[1]), min(y_hi, hi[1])
        if xmax - xmin < 1.0:
            c = 0.5 * (xmin + xmax)
            xmin, xmax = c - 0.5, c + 0.5
        if ymax - ymin < 1.0:
            c = 0.5 * (ymin + ymax)
            ymin, ymax = c - 0.5, c + 0.5
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymax, ymin)  # inverted Y (image coordinates)
    else:
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        span = np.maximum(hi - lo, 1e-6)
        pad = 0.08 * span
        ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
        ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])


def plot_sample_figure(
    obs: np.ndarray,
    gt: np.ndarray,
    samples: np.ndarray,
    best: np.ndarray,
    *,
    title: str,
    save_path: Path,
    background: np.ndarray | None = None,
    extent: tuple[float, float, float, float] | None = None,
    color: str = "#d62728",
    units: str = "m",
    margin: float = 150.0,
) -> None:
    """Render one single-panel sample figure (history, K bundle, best, GT)."""
    fig, ax = plt.subplots(figsize=(7.5, 7.5))

    if background is not None:
        if extent is None:
            extent = (0.0, float(background.shape[1]), float(background.shape[0]), 0.0)
        ax.imshow(
            background,
            extent=extent,
            origin="upper",
            zorder=0,
            interpolation="bilinear",
            aspect="equal",
        )

    ax.plot(obs[:, 0], obs[:, 1], zorder=4, **STYLE["history"])
    ax.scatter(obs[-1, 0], obs[-1, 1], color="black", marker="s", s=60, zorder=5)
    for i, hyp in enumerate(np.asarray(samples)):
        ax.plot(
            hyp[:, 0],
            hyp[:, 1],
            color=color,
            alpha=0.15,
            zorder=2,
            label="Predicted samples (K=20)" if i == 0 else "_nolegend_",
            **SAMPLE_STYLE_BASE,
        )
    ax.plot(
        best[:, 0],
        best[:, 1],
        color=color,
        linewidth=2.6,
        alpha=0.9,
        marker="o",
        markersize=3,
        zorder=5,
        label="Best-of-K (min-ADE)",
    )
    ax.plot(gt[:, 0], gt[:, 1], zorder=4, **STYLE["gt"])

    _zoom_limits(ax, [obs, gt, samples, best], background, extent, margin)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel(f"x [{units}]")
    ax.set_ylabel(f"y [{units}]")
    ax.legend(loc="best", fontsize=10, framealpha=0.9)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Window selection
# ---------------------------------------------------------------------------
def select_window_indices(n_windows: int, args: argparse.Namespace) -> list[int]:
    if n_windows <= 0:
        return []
    if getattr(args, "window_indices", None):
        idx = [int(i) for i in args.window_indices if 0 <= int(i) < n_windows]
        return idx
    k = min(int(args.num_samples), n_windows)
    if k == n_windows:
        return list(range(n_windows))
    return sorted({int(round(x)) for x in np.linspace(0, n_windows - 1, k)})


# ---------------------------------------------------------------------------
# Per-variant driver
# ---------------------------------------------------------------------------
def run_variant(
    out_root: Path,
    protocol: str,
    scene: str,
    variant: str,
    args: argparse.Namespace,
    backbone,
    preprocess,
    device: str,
) -> list[dict]:
    ckpt = variant_checkpoint(scene, variant, getattr(args, "video_id", None))
    if not ckpt.exists():
        _console(
            f"SKIP {protocol}/{scene}/{variant}: no checkpoint at {ckpt.parent}. "
            "Re-train with "
            f"`python fm_sdd_global.py --held_out_scene {scene} "
            f"--video_mode {variant}"
            + (f" --video-id {args.video_id}" if args.video_id else "")
            + "` to regenerate it."
        )
        return []

    denoiser, cfg = load_variant_denoiser(scene, variant, args)
    dataset, _dcfg = build_variant_dataset(scene, variant, args)
    indices = select_window_indices(len(dataset), args)
    if not indices:
        _console(f"  no test windows for {protocol}/{scene}/{variant}; skipping")
        del denoiser, dataset
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        return []
    _console(
        f"{protocol}/{scene}/{variant}: {len(dataset)} test windows, "
        f"rendering {len(indices)} -> {out_root}"
    )

    # The dataloader caches its scene prior on the FIRST ``__getitem__`` call, so
    # production (shuffle=False) locks it to window 0's video. Force that here
    # too so the prior — and the keyframes we reproduce for Grad-CAM — always
    # come from the same video regardless of which windows the user selects.
    static_frames = None
    if variant == "static" and not args.no_gradcam:
        first_vid = dataset.windows.video_id(0)
        _ = dataset[0]  # populate/cache _static_video_vec from window 0
        fids = static_keyframe_ids(scene, first_vid, args)
        if fids:
            static_frames = _decode_frame_ids(
                expand_sdd_root(args.sdd_root), scene, first_vid, fids
            )
            _console(f"  static prior from {first_vid}: {len(fids)} keyframes")

    lut = gcam._jet_lut()
    rows: list[dict] = []
    for out_i, widx in enumerate(indices, start=1):
        try:
            batch = collate_sdd_global([dataset[widx]])
            video_id = batch["video_id"][0]
            anchor = int(batch["anchor_frame"][0])
            res = sample_window_predictions(denoiser, cfg, batch, device, args.seed)

            stem = f"sample_{out_i:03d}"
            bg = None
            extent = None
            if variant != "off":
                got = load_anchor_frame(
                    expand_sdd_root(args.sdd_root), scene, video_id, anchor
                )
                if got is not None:
                    bg, extent, _src = got

            # Shift the relative-frame bundle into absolute pixels so it lands
            # on the video frame; the project's identity homography means
            # absolute annotation pixels equal physical metres.
            if bg is not None:
                res = _to_absolute(res, res["obs_abs"][-1])

            common_title = (
                f"{scene}/{video_id} | w{widx} | {VARIANT_LABEL[variant]} | "
                f"ADE$_{{min}}$={res['ade']:.2f}"
            )
            plot_sample_figure(
                res["obs"],
                res["gt"],
                res["preds"],
                res["best"],
                title=common_title,
                save_path=out_root / f"{stem}_{VARIANT_TAG[variant]}.png",
                background=bg,
                extent=extent,
                color=VARIANT_COLOR[variant],
            )

            if variant in ("static", "full") and not args.no_gradcam:
                cam, gcam_idx = variant_gradcam(
                    variant,
                    scene,
                    batch,
                    anchor,
                    video_id,
                    denoiser,
                    cfg,
                    backbone,
                    preprocess,
                    device,
                    args.seed,
                    static_frames,
                    expand_sdd_root(args.sdd_root),
                )
                if gcam_idx is not None and gcam_idx != res["best_idx"]:
                    _console(
                        f"  note: grad-cam best_idx={gcam_idx} != "
                        f"sampled best_idx={res['best_idx']} on {stem} "
                        "(CAM may not explain the shown best trajectory)"
                    )
                if cam is not None and bg is not None:
                    overlay = gcam.overlay_heatmap(bg, cam, alpha=args.alpha, lut=lut)
                    plot_sample_figure(
                        res["obs"],
                        res["gt"],
                        res["preds"],
                        res["best"],
                        title=common_title + " | Grad-CAM",
                        save_path=out_root
                        / f"{stem}_{VARIANT_GRADCAM_TAG[variant]}.png",
                        background=overlay,
                        extent=extent,
                        color=VARIANT_COLOR[variant],
                    )
                else:
                    _console(
                        f"  no Grad-CAM for {stem} ({scene}/{video_id}) — no frame"
                    )
                del cam

            rows.append(
                {
                    "protocol": protocol,
                    "scene": scene,
                    "variant": variant,
                    "window_index": widx,
                    "sample": stem,
                    "video_id": video_id,
                    "anchor_frame": anchor,
                    "ade_min": f"{res['ade']:.6f}",
                    "best_hyp": res["best_idx"],
                }
            )
            _console(
                f"  {stem} w{widx} {variant}: ADE_min={res['ade']:.3f} "
                f"(video={video_id}, anchor={anchor})"
            )
            del batch, res, bg
        except Exception as exc:  # keep going: one bad window never kills the run
            _console(f"  ERROR {scene}/{variant} w{widx}: {exc!r}")
        finally:
            gc.collect()

    del denoiser, dataset
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return rows


# ---------------------------------------------------------------------------
# Protocol driver
# ---------------------------------------------------------------------------
def run_protocol(
    args: argparse.Namespace, protocol: str, backbone, preprocess, device: str
) -> list[dict]:
    video_id = "video0" if protocol == "video0" else None
    args.video_id = video_id  # consumed by _build_cfg / _ensure_norm_stats

    scenes = args.scenes
    rows: list[dict] = []
    for scene in scenes:
        dir_name = f"{scene}_video0" if protocol == "video0" else scene
        out_root = Path(args.out_dir) / protocol / dir_name
        _console(f"=== {protocol}/{dir_name} ===")
        for variant in args.variants:
            rows.extend(
                run_variant(
                    out_root,
                    protocol,
                    scene,
                    variant,
                    args,
                    backbone,
                    preprocess,
                    device,
                )
            )
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cfg", default=str(_REPO_ROOT / "cfg" / "sdd" / "cor_fm.yml"))
    p.add_argument("--scenes", nargs="+", default=list(SCENES), choices=list(SCENES))
    p.add_argument("--protocol", choices=["global", "video0", "both"], default="both")
    p.add_argument(
        "--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS)
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Evenly spaced test windows per scene (default 5).",
    )
    p.add_argument(
        "--window-indices",
        nargs="+",
        type=int,
        default=None,
        help="Explicit window indices (overrides --num-samples).",
    )
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--sdd-root", default=None)
    p.add_argument("--video-features-root", dest="video_features_root", default=None)
    p.add_argument("--sampling_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Grad-CAM overlay blend fraction (default 0.45).",
    )
    p.add_argument("--backbone", choices=["resnet18", "resnet50"], default="resnet18")
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", default=None, help="'cuda' or 'cpu' (default: auto).")
    p.add_argument("--use-ema", action="store_true")
    p.add_argument(
        "--no-gradcam",
        action="store_true",
        help="Skip Grad-CAM panels (fast trajectory-only figures).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    backbone, _dim, preprocess = gcam._make_backbone(
        args.backbone, pretrained=args.pretrained, device=args.device
    )

    protocols = ["global", "video0"] if args.protocol == "both" else [args.protocol]
    all_rows: list[dict] = []
    for protocol in protocols:
        all_rows.extend(run_protocol(args, protocol, backbone, preprocess, args.device))

    # per-protocol manifest (best-effort; never fatal)
    for protocol in protocols:
        rows = [r for r in all_rows if r["protocol"] == protocol]
        if not rows:
            continue
        manifest = Path(args.out_dir) / protocol / "manifest.csv"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        import csv

        with manifest.open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        _console(f"wrote {manifest} ({len(rows)} rows)")

    _console(f"done — {len(all_rows)} figures registered under {args.out_dir}")


if __name__ == "__main__":
    main()
