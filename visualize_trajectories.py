"""Publication-quality trajectory visualization module.

Standalone, importable rendering pipeline for academic trajectory-prediction
figures. The module is deliberately free of project state: the only project
dependency is ``numpy``/``matplotlib`` and the public function
:func:`render_scene_trajectories` is fully unit-testable on synthetic data.

The core contract (see :func:`render_scene_trajectories`):

* ``frame_img`` : the RGB video frame at ``t = T_obs`` (the observation end),
  rendered as the figure background;
* ``past_gt`` : the ``T_obs`` observed ground-truth positions;
* ``future_gt`` : the ``T_pred`` ground-truth positions;
* ``predictions`` : the ``K`` sampled prediction heads ``[K, T_pred, 2]``.

All coordinates are ABSOLUTE pixel coordinates in the same frame space as the
background image (origin top-left, y growing downwards) — exactly the space
the SDD annotation pipeline emits. If trajectories arrive in world
coordinates, project them with :func:`world_to_pixel` first (homography-aware,
identity by default so SDD pixel-space data passes through untouched).

Each scene+pedestrian+variant is rendered as a standalone 300-DPI PNG:

    ``{save_dir}/{scene_id}_ped{agent_id}_{variant_name}.png``

where ``variant_name`` is one of ``no_video | global_video | agent_centric``.

Run ``python visualize_trajectories.py --demo`` for a checkpoint-free styling
self-check that writes one figure per variant to ``visualizations/``.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

# Guarded import: matplotlib is an optional runtime dependency; the pure
# geometry/metrics helpers below must remain importable without it.
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    _PLOT_OK = True
except Exception as exc:  # pragma: no cover - environment dependent
    plt = None  # type: ignore[assignment]
    Circle = None  # type: ignore[assignment]
    _PLOT_OK = False
    _MPL_IMPORT_ERROR = exc

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------
VARIANT_LABELS: dict[str, str] = {
    "no_video": "No Video Encoder",
    "global_video": "Global Video Encoder",
    "agent_centric": "Agent-Centric Video Encoder",
}
VARIANTS: tuple[str, ...] = ("no_video", "global_video", "agent_centric")

# Styling (public so downstream scripts can tweak before calling render).
COLOR_PAST = "#00E5FF"      # bright cyan: visible on light and dark frames
COLOR_FUTURE = "#00FF00"    # bright green (spec-mandated)
COLOR_SAMPLES = "#FF8C42"   # warm coral/orange distribution bundle
COLOR_BEST = "#E6217F"      # bold magenta best-of-K highlight
COLOR_HIGHLIGHT = "#FFD700"  # target-agent marker ring

PAST_STYLE: dict = {"linestyle": "-", "linewidth": 2.0, "marker": "o", "markersize": 4.5}
FUTURE_STYLE: dict = {"linestyle": "-", "linewidth": 2.0, "marker": "o", "markersize": 4.5}
SAMPLE_STYLE: dict = {"linestyle": "-", "linewidth": 0.9, "marker": "o", "markersize": 2.0}
BEST_STYLE: dict = {
    "linestyle": "--",
    "linewidth": 2.6,
    "marker": "o",
    "markersize": 5.0,
}
SAMPLE_ALPHA = 0.30          # spec: alpha 0.25–0.35
BEST_ALPHA = 0.95
DEFAULT_ZOOM_MARGIN = 90.0   # pixel margin around plotted content


def _set_rcparams() -> None:
    """Apply publication print style (idempotent, Agg-friendly)."""
    if not _PLOT_OK:
        return
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 13,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "legend.fontsize": 11,
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


_set_rcparams()


# ---------------------------------------------------------------------------
# Coordinate geometry (vectorized, homography-aware)
# ---------------------------------------------------------------------------
def apply_homography(points: np.ndarray, H: np.ndarray | None) -> np.ndarray:
    """Transform ``(..., 2)`` points by a ``(3, 3)`` homography ``H``.

    The input is vectorized over the leading dimensions: any shape whose
    final axis is 2 is supported. When ``H`` is ``None`` the identity is
    used (SDD annotation/pixel space is an identity mapped world frame), so
    the call stays a no-op in that case — matching the pipeline's "pixels ≈
    meters" convention documented in ``data/dataloader_sdd_global.py``.

    Projective division by the homogenous ``w`` component is applied; ``w``
    values within epsilon of zero are clamped to avoid NaNs.
    """
    pts = np.asarray(points, dtype=np.float64)
    flat = pts.reshape(-1, 2)
    if H is None:
        return flat.reshape(pts.shape).astype(np.float64)

    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3):
        raise ValueError(f"H must be (3, 3), got {H.shape}")
    hom = np.column_stack([flat, np.ones(len(flat), dtype=np.float64)])
    proj = hom @ H.T                       # [N, 3]
    w = proj[:, 2:3]
    w = np.where(np.abs(w) < 1e-9, 1e-9, w)
    xy = proj[:, :2] / w
    return xy.reshape(pts.shape).astype(np.float64)


def world_to_pixel(
    points: np.ndarray,
    H: np.ndarray | None = None,
    image_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Project world coordinates to pixel coordinates (vectorized).

    Args:
        points:     ``(..., 2)`` world coordinates.
        H:          optional ``(3, 3)`` homography from world to pixel.
                    ``None`` (default) = identity — SDD trajectories already
                    live in annotation pixel space and pass through unchanged.
        image_size: optional ``(W, H)``; when given, the result is clipped to
                    the valid image extent so overlay plotting cannot escape
                    the frame.

    Returns:
        Pixel coordinates with the same leading shape as ``points``.
    """
    px = apply_homography(points, H)
    if image_size is not None:
        w, h = int(image_size[0]), int(image_size[1])
        px = np.clip(px, np.array([0.0, 0.0]), np.array([float(w - 1), float(h - 1)]))
    return px


# ---------------------------------------------------------------------------
# Metrics (best-of-K ADE/FDE)
# ---------------------------------------------------------------------------
def ade_fde_per_sample(preds: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-hypothesis ADE/FDE.

    Args:
        preds: ``[K, T_pred, 2]`` predicted futures (original scale).
        gt:    ``[T_pred, 2]`` ground-truth future (original scale).

    Returns:
        ``(ade [K], fde [K])`` — average and final displacement errors
        (Euclidean) per sampled hypothesis.
    """
    preds = np.asarray(preds, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    dist = np.linalg.norm(preds - gt[None, :, :], axis=-1)   # [K, T_pred]
    ade = dist.mean(axis=-1)
    fde = dist[:, -1]
    return ade, fde


def best_of_k(preds: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, int]:
    """Select the hypothesis with minimum average displacement (min-ADE).

    Returns ``(best [T_pred, 2], argmin_index)``.
    """
    ade, _ = ade_fde_per_sample(preds, gt)
    idx = int(ade.argmin())
    return np.asarray(preds, dtype=np.float64)[idx], idx


def ade_fde_best(preds: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """Best-of-K ADE/FDE for the whole bundle."""
    ade, fde = ade_fde_per_sample(preds, gt)
    i = int(ade.argmin())
    return float(ade[i]), float(fde[i])


# ---------------------------------------------------------------------------
# Shape coercers
# ---------------------------------------------------------------------------
def _as_xy(traj) -> np.ndarray:
    """Coerce to a ``[T, 2]`` float64 numpy array (leading dims collapsed)."""
    traj = np.asarray(traj, dtype=np.float64)
    while traj.ndim > 2:
        traj = traj[0]
    return traj


def _as_bundle(preds) -> np.ndarray:
    """Coerce to ``[K, T_pred, 2]`` float64 numpy (leading B/A dims collapsed)."""
    preds = np.asarray(preds, dtype=np.float64)
    while preds.ndim > 3:
        preds = preds[0]
    return preds


# ---------------------------------------------------------------------------
# Core public renderer
# ---------------------------------------------------------------------------
def render_scene_trajectories(
    frame_img: np.ndarray,
    past_gt: np.ndarray,
    future_gt: np.ndarray,
    predictions: np.ndarray,
    agent_id: int | str,
    scene_id: str,
    variant_name: str,
    save_dir: str | Path,
    *,
    ade: float | None = None,
    fde: float | None = None,
    zoom_margin: float = DEFAULT_ZOOM_MARGIN,
    show_metrics: bool = True,
    fname_suffix: str | None = None,
    overwrite: bool = True,
) -> Path:
    """Render one publication-quality trajectory figure over a video frame.

    Args:
        frame_img:  ``(H, W, 3)`` RGB frame at ``t = T_obs`` (uint8 0–255).
        past_gt:    ``(T_obs, 2)`` observed ground-truth positions (pixels).
        future_gt:  ``(T_pred, 2)`` future ground-truth positions (pixels).
        predictions: ``(K, T_pred, 2)`` sampled predicted futures (pixels;
            prediction heads are drawn as a semi-transparent bundle, and the
            min-ADE hypothesis is highlighted on top).
        agent_id:   pedestrian identifier, used in the filename + title.
        scene_id:   SDD scene name, used in the filename + title.
        variant_name: ``"no_video"``, ``"global_video"`` or ``"agent_centric"``.
        save_dir:   destination directory; the PNG is written to
            ``{save_dir}/{scene_id}_ped{agent_id}_{variant_name}.png`` (or with
            ``fname_suffix``: ``..._ped{agent_id}_{fname_suffix}_{variant_name}.png``).
        ade / fde:  optional precomputed best-of-K metrics (pixels). When
            omitted they are computed internally from ``predictions`` vs
            ``future_gt``.
        zoom_margin: pixel margin around the union of all plotted content
            (clamped to the frame) so trajectories stay readable.
        show_metrics: draw the best-of-K ADE/FDE text box.
        fname_suffix: optional extra token embedded in the filename to
            disambiguate multiple windows per pedestrian.
        overwrite:   silently replace an existing file (True, default) or
            raise ``FileExistsError``.

    Returns:
        The absolute path of the written PNG (``bbox_inches="tight"``,
        ``pad_inches=0``, ``dpi=300``).
    """
    if not _PLOT_OK:
        raise RuntimeError(
            f"matplotlib is required for rendering: {_MPL_IMPORT_ERROR}"
        )
    if variant_name not in VARIANTS:
        raise ValueError(
            f"variant_name must be one of {VARIANTS}, got {variant_name!r}"
        )

    # ---- validate / coerce -------------------------------------------------
    frame = np.asarray(frame_img)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"frame_img must be (H, W, 3), got {frame.shape}")
    past = _as_xy(past_gt)
    future = _as_xy(future_gt)
    preds = _as_bundle(predictions)
    if past.shape[-1] != 2 or future.shape[-1] != 2 or preds.shape[-1] != 2:
        raise ValueError("all trajectories must have a final (x, y) axis of size 2")
    if preds.shape[1] != future.shape[0]:
        raise ValueError(
            f"prediction horizon {preds.shape[1]} != future_gt horizon "
            f"{future.shape[0]}"
        )

    best, best_idx = best_of_k(preds, future)
    if ade is None or fde is None:
        ade, fde = ade_fde_best(preds, future)

    # ---- safe filename tokens ----------------------------------------------
    safe_scene = re.sub(r"[^A-Za-z0-9_]+", "_", str(scene_id)).strip("_")
    safe_ped = re.sub(r"[^A-Za-z0-9_]+", "_", str(agent_id)).strip("_")
    safe_suffix = re.sub(r"[^A-Za-z0-9_]+", "_", fname_suffix or "").strip("_")
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if safe_suffix:
        save_path = out_dir / f"{safe_scene}_ped{safe_ped}_{safe_suffix}_{variant_name}.png"
    else:
        save_path = out_dir / f"{safe_scene}_ped{safe_ped}_{variant_name}.png"
    if save_path.exists() and not overwrite:
        raise FileExistsError(save_path)

    h, w = frame.shape[0], frame.shape[1]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    extent = (0.0, float(w), float(h), 0.0)                 # origin top-left
    ax.imshow(frame, extent=extent, origin="upper", zorder=0,
              interpolation="bilinear", aspect="equal")

    # ---- target-agent highlight at t = T_obs ------------------------------
    pc = past[-1]
    ring = Circle((float(pc[0]), float(pc[1])), radius=8.0, fill=False,
                  edgecolor=COLOR_HIGHLIGHT, linewidth=2.2, zorder=6)
    ax.add_patch(ring)
    ax.scatter(pc[0], pc[1], s=14, color=COLOR_HIGHLIGHT, zorder=6)

    # ---- past GT -----------------------------------------------------------
    (h_past,) = ax.plot(past[:, 0], past[:, 1], color=COLOR_PAST, zorder=4,
                        label="Observed (GT)", **PAST_STYLE)

    # ---- future GT ----------------------------------------------------------
    (h_gt,) = ax.plot(future[:, 0], future[:, 1], color=COLOR_FUTURE, zorder=4,
                      label="Future GT", **FUTURE_STYLE)

    # ---- prediction bundle ---------------------------------------------------
    h_samples = None
    sample_style = dict(SAMPLE_STYLE)
    for i, hyp in enumerate(preds):
        label = "Predicted (K samples)" if i == 0 else "_nolegend_"
        (artist,) = ax.plot(
            hyp[:, 0], hyp[:, 1], color=COLOR_SAMPLES, alpha=SAMPLE_ALPHA,
            zorder=2, label=label, **sample_style,
        )
        if h_samples is None:
            h_samples = artist

    # ---- best-of-K highlight -------------------------------------------------
    (h_best,) = ax.plot(
        best[:, 0], best[:, 1], color=COLOR_BEST, alpha=BEST_ALPHA, zorder=5,
        label=f"Best-of-K (min-ADE, k={best_idx})", **BEST_STYLE,
    )

    # ---- limits: padded union, clamped to frame ------------------------------
    pts = np.concatenate(
        [past.reshape(-1, 2), future.reshape(-1, 2), preds.reshape(-1, 2)]
    )
    lo = pts.min(axis=0) - zoom_margin
    hi = pts.max(axis=0) + zoom_margin
    xmin, xmax = max(0.0, lo[0]), min(float(w), hi[0])
    ymin, ymax = max(0.0, lo[1]), min(float(h), hi[1])
    if xmax - xmin < 1.0:
        c = 0.5 * (xmin + xmax)
        xmin, xmax = c - 0.5, c + 0.5
    if ymax - ymin < 1.0:
        c = 0.5 * (ymin + ymax)
        ymin, ymax = c - 0.5, c + 0.5
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)  # inverted so the frame renders upright
    ax.set_adjustable("box")

    # ---- annotations ---------------------------------------------------------
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    ax.set_title(
        f"SDD '{safe_scene}' \u2014 PedID {safe_ped} \u2014 "
        f"{VARIANT_LABELS[variant_name]}"
    )
    ax.legend(handles=[h_past, h_gt, h_samples, h_best], loc="upper left",
              frameon=True, framealpha=0.85)

    if show_metrics:
        k = preds.shape[0]
        txt = (
            f"$\\mathrm{{ADE}}_\\mathrm{{best}}(K={k})$ = {ade:.2f} px\n"
            f"$\\mathrm{{FDE}}_\\mathrm{{best}}(K={k})$ = {fde:.2f} px"
        )
        ax.text(
            0.98, 0.02, txt, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=11, zorder=7,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="black", boxstyle="round,pad=0.4"),
        )

    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return save_path.absolute()


# ---------------------------------------------------------------------------
# Demo (checkpoint-free styling self-check)
# ---------------------------------------------------------------------------
def _synthetic_bundle(future_gt: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Synthetic multimodal bundle around a ground-truth future (demo only)."""
    k, f = 24, future_gt.shape[0]
    modes = np.array([[0.0, 0.7], [1.6, -0.2], [-1.1, 0.5]], dtype=np.float64)
    t = np.linspace(0, 1, f)[None, :, None]
    out = np.empty((k, f, 2), dtype=np.float64)
    for i in range(k):
        mode = modes[i % len(modes)] + rng.normal(scale=0.25, size=2)
        out[i] = future_gt + mode * t
        out[i] += rng.normal(scale=0.12, size=(f, 2))
    return out


def run_demo(out_dir: str | Path) -> list[Path]:
    """Render one figure per variant on synthetic data (no checkpoint needed)."""
    if not _PLOT_OK:
        raise RuntimeError("matplotlib is required for the demo.")
    rng = np.random.default_rng(0)
    w, h = 640, 480
    grad = np.linspace(0, 180, w, dtype=np.uint8)
    frame = np.repeat(grad[None, :, None], h, axis=0)
    frame = np.repeat(frame, 3, axis=-1)

    t_obs = np.linspace(0, 2 * np.pi, 8)
    past = np.stack([120 + 3 * t_obs, 300 + 1.8 * np.sin(t_obs) * 3], axis=-1)
    future = np.stack(
        [past[-1, 0] + np.linspace(0, 60, 12),
         past[-1, 1] + np.linspace(0, 35, 12)],
        axis=-1,
    )
    preds = _synthetic_bundle(future, rng)

    # Exercise the homography helpers (identity path + a projective PE test).
    assert np.allclose(world_to_pixel(past, H=None), past)
    shear = np.array([[1.0, 0.3, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    sheared = world_to_pixel(past, H=shear)
    assert np.allclose(sheared[:, 0] - past[:, 0], 0.3 * past[:, 1])

    saved: list[Path] = []
    for variant in VARIANTS:
        path = render_scene_trajectories(
            frame, past, future, preds,
            agent_id="demo", scene_id="demo_scene", variant_name=variant,
            save_dir=out_dir,
        )
        saved.append(path)
        print(f"[vis] demo {variant}: {path}")
    return saved


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", default="visualizations",
                   help="where the demo PNGs are written (default: visualizations/)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    if not _PLOT_OK:
        raise RuntimeError(
            f"matplotlib import failed: {_MPL_IMPORT_ERROR}. "
            "Install matplotlib>=3.9.2 or set MPLBACKEND=Agg."
        )
    args = _parse_args(argv)
    saved = run_demo(args.out_dir)
    print(f"[vis] wrote {len(saved)} variant figures to {args.out_dir}")


if __name__ == "__main__":
    main()