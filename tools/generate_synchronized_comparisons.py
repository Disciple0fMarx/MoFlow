"""Synchronized 4-panel qualitative comparison for the SDD video-conditioning ablation.

For every selected test window this tool renders a single publication-grade
figure with four horizontally aligned panels that all describe the **same
scene, window index, pedestrian agent, and spatial extent**:

1. ``off``    — trajectory prediction with the video branch disabled (``z_vid = 0``);
2. ``static`` — prediction conditioned on the fixed scene prior (one vector,
   ``P = 1``; the prior is the mean of up to 8 coarse keyframes, see
   :func:`tools.generate_qualitative_figures.static_keyframe_ids`);
3. ``full``   — prediction conditioned on the per-window multi-frame temporal
   mean (``P = 8`` observed frames);
4. ``Grad-CAM`` — the full encoder's spatial attention heatmap for the window's
   exact anchor frame, alpha-blended over that frame.

Why this exists
---------------
The per-variant tools produce separate PNGs, which makes apples-to-apples reading
hard: a reader has to eyeball two files and hope the axes agree. Here the three
predictions come from the *same* window (same ``obs_abs`` origin, same GT, same
best-of-K seed) and **share one bounding box**, so the only difference across
panels is the conditioning. A single color language is used everywhere
(GT = green, K=20 bundle = faint red, best-of-K = thick red).

Output layout
-------------
::

    results_sdd/synchronized_comparisons/
    ├── global/<scene>/sample_001/
    │   ├── synchronized_comparison.png
    │   └── metadata.json
    ├── video0/<scene>_video0/sample_001/
    │   ├── synchronized_comparison.png
    │   └── metadata.json
    └── <protocol>/manifest.csv

Notes
-----
* Predictions are sampled with the native ``SDDGlobalDataset`` test split per
  variant (same conditioning as production evaluation) and the same seed, so
  the shown bundle and the Grad-CAM Euler replica explain the same hypotheses.
* ``static`` is a *fixed scene-level vector*, not a literal single keyframe; the
  panel is labelled ``P = 1`` because the model receives one vector per window.
* Missing checkpoints (e.g. ``vmstatic`` removed by the Q6 cleanup) skip the
  whole scene/protocol with an actionable retrain message — a 4-panel figure
  cannot be faked when one arm is absent.

Example::

    python tools/generate_synchronized_comparisons.py \\
        --protocol both --num-samples 5 \\
        --sdd-root /home/efrei_stage/Desktop/Datasets/SDD \\
        --video-features-root ./features/resnet18
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
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
from matplotlib.lines import Line2D

from data.dataloader_sdd_global import collate_sdd_global
from tools import gradcam_sdd as gcam
from tools.generate_qualitative_figures import (
    SCENES,
    VARIANTS,
    _console,
    _decode_frame_ids,
    _to_absolute,
    build_variant_dataset,
    load_variant_denoiser,
    sample_window_predictions,
    select_window_indices,
    static_keyframe_ids,
    variant_checkpoint,
    variant_gradcam,
)
from tools.visualize_trajectory_comparison import (
    SAMPLE_STYLE_BASE,
    STYLE,
    load_anchor_frame,
)
from video_encoder.sdd_adapter import expand_sdd_root

DEFAULT_OUT_DIR = _REPO_ROOT / "results_sdd" / "synchronized_comparisons"

PANEL_TITLES = {
    "off": "off  ($z_{vid}=0$)",
    "static": "static  (fixed prior, $P=1$)",
    "full": "full  ($P=8$ frames)",
}
PRED_COLOR = "#d62728"


# ---------------------------------------------------------------------------
# Geometry: one shared bounding box for every panel
# ---------------------------------------------------------------------------
def shared_limits(
    bundles: dict[str, dict],
    background: np.ndarray | None,
    extent: tuple[float, float, float, float] | None,
    margin: float = 150.0,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Union bounding box across all variants' obs/GT/predictions.

    Returns ``(xlim, ylim)`` ready for ``ax.set_xlim(*xlim); ax.set_ylim(*ylim)``.
    With a frame background the box is clamped to the image and the y-axis is
    inverted (image coordinates); without one it is padded in data space.
    """
    arrays = []
    for b in bundles.values():
        arrays.extend(
            [
                np.asarray(b["obs"], dtype=float).reshape(-1, 2),
                np.asarray(b["gt"], dtype=float).reshape(-1, 2),
                np.asarray(b["preds"], dtype=float).reshape(-1, 2),
                np.asarray(b["best"], dtype=float).reshape(-1, 2),
            ]
        )
    pts = np.concatenate(arrays, axis=0)

    if background is not None and extent is not None:
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
        return (xmin, xmax), (ymax, ymin)  # inverted Y (image coordinates)

    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    pad = 0.08 * span
    return (lo[0] - pad[0], hi[0] + pad[0]), (lo[1] - pad[1], hi[1] + pad[1])


def _draw_bundle(ax, bundle: dict, background, extent, pred_color: str) -> None:
    """Draw history / K=20 bundle / best-of-K / GT for one variant panel."""
    if background is not None:
        ax.imshow(
            background,
            extent=extent,
            origin="upper",
            zorder=0,
            aspect="equal",
            interpolation="bilinear",
        )
    ax.plot(bundle["obs"][:, 0], bundle["obs"][:, 1], **STYLE["history"], zorder=4)
    ax.scatter(
        bundle["obs"][-1, 0],
        bundle["obs"][-1, 1],
        color="black",
        marker="s",
        s=55,
        zorder=5,
    )
    for hyp in np.asarray(bundle["preds"]):
        ax.plot(
            hyp[:, 0],
            hyp[:, 1],
            color=pred_color,
            alpha=0.12,
            zorder=2,
            **SAMPLE_STYLE_BASE,
        )
    ax.plot(
        bundle["best"][:, 0],
        bundle["best"][:, 1],
        color=pred_color,
        linewidth=2.6,
        alpha=0.9,
        marker="o",
        markersize=3,
        zorder=5,
    )
    ax.plot(bundle["gt"][:, 0], bundle["gt"][:, 1], **STYLE["gt"], zorder=4)


def _legend_handles(pred_color: str) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            linewidth=1.6,
            marker="o",
            markersize=3,
            label="Observed history",
        ),
        Line2D(
            [0],
            [0],
            color="green",
            linewidth=1.8,
            marker="o",
            markersize=3,
            label="Ground truth",
        ),
        Line2D(
            [0],
            [0],
            color=pred_color,
            linewidth=1.0,
            alpha=0.5,
            label="Predicted (K=20)",
        ),
        Line2D([0], [0], color=pred_color, linewidth=2.6, label="Best-of-K (min-ADE)"),
    ]


def plot_synchronized_figure(
    bundles: dict[str, dict],
    cam: np.ndarray | None,
    background: np.ndarray | None,
    extent: tuple[float, float, float, float] | None,
    *,
    title: str,
    save_path: Path,
    gradcam_variant: str,
    anchor_frame: int,
    alpha: float = 0.45,
    margin: float = 150.0,
    dpi: int = 300,
) -> None:
    """Render the 1x4 synchronized comparison (off | static | full | Grad-CAM)."""
    order = [v for v in VARIANTS if v in bundles]
    fig, axes = plt.subplots(1, 4, figsize=(22.0, 5.6))
    xlim, ylim = shared_limits(bundles, background, extent, margin)
    lut = gcam._jet_lut()

    for ax, variant in zip(axes[:3], order):
        b = bundles[variant]
        _draw_bundle(ax, b, background, extent, PRED_COLOR)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(
            f"{PANEL_TITLES[variant]}\nADE$_{{min}}$ = {b['ade']:.2f}",
            fontsize=11,
        )
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.tick_params(labelsize=8)

    # ---- panel 4: Grad-CAM ------------------------------------------------
    ax = axes[3]
    if cam is not None and background is not None:
        overlay = gcam.overlay_heatmap(background, cam, alpha=alpha, lut=lut)
        ax.imshow(
            overlay,
            extent=extent,
            origin="upper",
            zorder=0,
            aspect="equal",
            interpolation="bilinear",
        )
        # Relate attention to the prediction: GT + full best-of-K, thin.
        fb = bundles.get(gradcam_variant) or bundles.get("full")
        if fb is not None:
            ax.plot(
                fb["gt"][:, 0],
                fb["gt"][:, 1],
                color="green",
                linewidth=1.4,
                alpha=0.9,
                zorder=4,
            )
            ax.plot(
                fb["best"][:, 0],
                fb["best"][:, 1],
                color="white",
                linewidth=1.6,
                alpha=0.9,
                linestyle="--",
                zorder=5,
            )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(
            f"Grad-CAM ({gradcam_variant})\nanchor frame $f_{{{anchor_frame}}}$",
            fontsize=11,
        )
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.tick_params(labelsize=8)
    else:
        ax.text(
            0.5,
            0.5,
            "Grad-CAM unavailable\n(no decodable anchor frame)",
            ha="center",
            va="center",
            fontsize=11,
            color="0.35",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Grad-CAM ({gradcam_variant})", fontsize=11)

    fig.suptitle(title, fontsize=13)
    fig.legend(
        handles=_legend_handles(PRED_COLOR),
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-variant collection (one model resident at a time)
# ---------------------------------------------------------------------------
def _free_model() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def collect_variant(
    scene: str,
    variant: str,
    indices: list[int],
    args: argparse.Namespace,
    backbone,
    preprocess,
    device: str,
) -> dict[int, dict] | None:
    """Sample one variant for ``indices``; optionally compute its Grad-CAM."""
    ckpt = variant_checkpoint(scene, variant, getattr(args, "video_id", None))
    if not ckpt.exists():
        _console(
            f"SKIP {args._protocol}/{scene}/{variant}: no checkpoint at "
            f"{ckpt.parent}. Re-train with `python fm_sdd_global.py "
            f"--held_out_scene {scene} --video_mode {variant}"
            + (f" --video-id {args.video_id}" if args.video_id else "")
            + "`."
        )
        return None

    denoiser, cfg = load_variant_denoiser(scene, variant, args)
    dataset, _ = build_variant_dataset(scene, variant, args)

    static_frames = None
    if variant == "static":
        # Lock the scene prior to window 0's video (production shuffle=False).
        first_vid = dataset.windows.video_id(0)
        _ = dataset[0]
        fids = static_keyframe_ids(scene, first_vid, args)
        if fids:
            static_frames = _decode_frame_ids(
                expand_sdd_root(args.sdd_root), scene, first_vid, fids
            )

    want_cam = (not args.no_gradcam) and variant == args.gradcam_variant
    out: dict[int, dict] = {}
    try:
        for widx in indices:
            if widx >= len(dataset):
                continue
            batch = collate_sdd_global([dataset[widx]])
            video_id = batch["video_id"][0]
            anchor = int(batch["anchor_frame"][0])
            res = sample_window_predictions(
                denoiser,
                cfg,
                batch,
                device,
                args.seed,
                agent_index=args.agent_index,
            )
            rec = {
                "bundle": _to_absolute(res, res["obs_abs"][-1]),
                "video_id": video_id,
                "anchor_frame": anchor,
                "cam": None,
            }
            if want_cam:
                cam, _gidx = variant_gradcam(
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
                rec["cam"] = cam
            out[widx] = rec
            del batch
    finally:
        del denoiser, dataset
        _free_model()
    return out


def collect_scene(
    scene: str,
    args: argparse.Namespace,
    backbone,
    preprocess,
    device: str,
) -> tuple[list[int], dict[int, dict]] | None:
    """Collect all three variants for one scene, keyed by the same window index."""
    indices: list[int] | None = None
    records: dict[int, dict] = {}

    for variant in VARIANTS:
        ckpt = variant_checkpoint(scene, variant, getattr(args, "video_id", None))
        if not ckpt.exists():
            # collect_variant prints the actionable message
            collect_variant(scene, variant, [], args, backbone, preprocess, device)
            return None
        dataset, _ = build_variant_dataset(scene, variant, args)
        idxs = select_window_indices(len(dataset), args)
        del dataset
        _free_model()
        if indices is None:
            indices = idxs
        elif idxs != indices:
            common = sorted(set(indices) & set(idxs))
            _console(
                f"  note: {scene}/{variant} window indices differ; "
                f"using the {len(common)} shared windows"
            )
            indices = common

    if not indices:
        _console(f"  no test windows for {scene}/{getattr(args, '_protocol', '?')}")
        return None

    for variant in VARIANTS:
        part = collect_variant(
            scene, variant, indices, args, backbone, preprocess, device
        )
        if part is None:
            return None
        for widx, rec in part.items():
            slot = records.setdefault(
                widx,
                {
                    "window_index": widx,
                    "video_id": rec["video_id"],
                    "anchor_frame": rec["anchor_frame"],
                    "bundles": {},
                    "cam": None,
                },
            )
            slot["bundles"][variant] = rec["bundle"]
            if (
                rec["video_id"] != slot["video_id"]
                or rec["anchor_frame"] != slot["anchor_frame"]
            ):
                _console(f"  WARNING: {scene} w{widx} window mismatch across variants")
            if rec["cam"] is not None:
                slot["cam"] = rec["cam"]
    return indices, records


# ---------------------------------------------------------------------------
# Rendering driver
# ---------------------------------------------------------------------------
def render_scene(
    scene: str,
    protocol: str,
    indices: list[int],
    records: dict[int, dict],
    args: argparse.Namespace,
) -> list[dict]:
    dir_name = f"{scene}_video0" if protocol == "video0" else scene
    out_root = Path(args.out_dir) / protocol / dir_name
    rows: list[dict] = []
    sdd_root = expand_sdd_root(args.sdd_root)

    for widx in indices:
        rec = records.get(widx)
        if rec is None:
            continue
        bundles = rec["bundles"]
        if set(VARIANTS) - set(bundles):
            _console(f"  skip {scene} w{widx}: incomplete variants {sorted(bundles)}")
            continue

        sample_i = len(rows) + 1
        background = extent = None
        got = load_anchor_frame(
            sdd_root, scene, rec["video_id"], int(rec["anchor_frame"])
        )
        if got is not None:
            background, extent, _src = got

        sample_dir = out_root / f"sample_{sample_i:03d}"
        fig_path = sample_dir / "synchronized_comparison.png"
        title = (
            f"SDD '{scene}' | w{widx} | video {rec['video_id']} | "
            f"anchor f{rec['anchor_frame']} | agent {args.agent_index}"
        )
        try:
            plot_synchronized_figure(
                bundles,
                rec["cam"],
                background,
                extent,
                title=title,
                save_path=fig_path,
                gradcam_variant=args.gradcam_variant,
                anchor_frame=int(rec["anchor_frame"]),
                alpha=args.alpha,
                margin=args.margin,
                dpi=args.dpi,
            )
        except Exception as exc:  # one bad window never kills the run
            _console(f"  ERROR rendering {scene} w{widx}: {exc!r}")
            continue

        meta = {
            "scene": scene,
            "protocol": protocol,
            "sample": sample_i,
            "window_index": widx,
            "video_id": rec["video_id"],
            "anchor_frame": int(rec["anchor_frame"]),
            "agent_index": args.agent_index,
            "gradcam_variant": args.gradcam_variant,
            "cam_available": rec["cam"] is not None,
            "variants": {
                v: {
                    "ade_min": float(bundles[v]["ade"]),
                    "best_hyp": int(bundles[v]["best_idx"]),
                }
                for v in VARIANTS
            },
            "figure": fig_path.name,
        }
        meta_path = sample_dir / "metadata.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        rows.append(meta)

        _console(
            f"  {sample_dir.relative_to(Path(args.out_dir))} "
            f"w{widx}: ADE off={bundles['off']['ade']:.2f} "
            f"static={bundles['static']['ade']:.2f} "
            f"full={bundles['full']['ade']:.2f}"
        )
    return rows


# ---------------------------------------------------------------------------
# Protocol / CLI
# ---------------------------------------------------------------------------
def run_protocol(
    args: argparse.Namespace, protocol: str, backbone, preprocess, device: str
) -> list[dict]:
    args.video_id = "video0" if protocol == "video0" else None
    args._protocol = protocol  # used only in log messages

    rows: list[dict] = []
    for scene in args.scenes:
        _console(f"=== {protocol}/{scene} ===")
        collected = collect_scene(scene, args, backbone, preprocess, device)
        if collected is None:
            continue
        indices, records = collected
        rows.extend(render_scene(scene, protocol, indices, records, args))
    return rows


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scenes", nargs="+", default=list(SCENES), choices=list(SCENES))
    p.add_argument("--protocol", choices=["global", "video0", "both"], default="both")
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
        "--agent-index",
        type=int,
        default=0,
        help="Pedestrian index within the window (SDD AGENTS=1 → 0).",
    )
    p.add_argument(
        "--gradcam-variant",
        choices=["full", "static"],
        default="full",
        help="Which video arm's Grad-CAM to show in panel 4 (default full).",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Grad-CAM overlay blend fraction (default 0.45).",
    )
    p.add_argument(
        "--margin",
        type=float,
        default=150.0,
        help="Context margin (annotation px) around the shared bbox.",
    )
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--backbone", choices=["resnet18", "resnet50"], default="resnet18")
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", default=None, help="'cuda' or 'cpu' (default: auto).")
    p.add_argument("--use-ema", action="store_true")
    p.add_argument(
        "--no-gradcam",
        action="store_true",
        help="Skip panel 4 (3-panel figure; debugging only).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.video_id = None
    args._protocol = "global"

    backbone, _dim, preprocess = gcam._make_backbone(
        args.backbone, pretrained=args.pretrained, device=args.device
    )

    protocols = ["global", "video0"] if args.protocol == "both" else [args.protocol]
    all_rows: list[dict] = []
    for protocol in protocols:
        all_rows.extend(run_protocol(args, protocol, backbone, preprocess, args.device))

    for protocol in protocols:
        rows = [r for r in all_rows if r["protocol"] == protocol]
        if not rows:
            continue
        manifest = Path(args.out_dir) / protocol / "manifest.csv"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "protocol",
            "scene",
            "sample",
            "window_index",
            "video_id",
            "anchor_frame",
            "agent_index",
            "gradcam_variant",
            "cam_available",
            "ade_off",
            "ade_static",
            "ade_full",
            "figure",
        ]
        with manifest.open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fields)
            writer.writeheader()
            for r in rows:
                writer.writerow(
                    {
                        "protocol": r["protocol"],
                        "scene": r["scene"],
                        "sample": r["sample"],
                        "window_index": r["window_index"],
                        "video_id": r["video_id"],
                        "anchor_frame": r["anchor_frame"],
                        "agent_index": r["agent_index"],
                        "gradcam_variant": r["gradcam_variant"],
                        "cam_available": r["cam_available"],
                        "ade_off": f"{r['variants']['off']['ade_min']:.6f}",
                        "ade_static": f"{r['variants']['static']['ade_min']:.6f}",
                        "ade_full": f"{r['variants']['full']['ade_min']:.6f}",
                        "figure": (
                            f"{protocol}/"
                            f"{(r['scene'] + '_video0') if protocol == 'video0' else r['scene']}/"
                            f"sample_{int(r['sample']):03d}/{r['figure']}"
                        ),
                    }
                )
        _console(f"wrote {manifest} ({len(rows)} rows)")

    _console(f"done — {len(all_rows)} synchronized figures under {args.out_dir}")


if __name__ == "__main__":
    main()
