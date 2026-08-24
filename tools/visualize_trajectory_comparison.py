#!/usr/bin/env python3
"""Trajectory comparison visualizer for Video-MoFlow (SDD LOSO).

Loads the best checkpoints of the trajectory-only baseline (`novid`) and the
global video-conditioned model (`vid`), samples predictions for the same
observation window, and plots them against ground truth:

    history      black dashed + square marker on final observed timestep
    ground truth solid green
    no-video     solid blue   (alpha 0.7)
    global VE    solid red    (alpha 0.9, thicker line)

Figures are written to ``visualizations/<scene>_trajectory_comparison.png``.

Checkpoint layout (relative to repo root)::

    results_sdd/cor_fm/_SDD_ho<scene>_<vid|novid>/models/checkpoint_best.pt

Usage::

    python tools/visualize_trajectory_comparison.py --scenes gates quad
    python tools/visualize_trajectory_comparison.py --demo   # styling check only

NOTE (SDD): coordinates are in *pixel* space under identity homography —
label axes accordingly if you change the dataloader.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results_sdd" / "cor_fm"
DEFAULT_OUT_DIR = REPO_ROOT / "visualizations"

SCENES = (
    "bookstore", "coupa", "deathCircle", "gates",
    "hyang", "little", "nexus", "quad",
)

# Styling contract (single source of truth for all plots).
STYLE = {
    "history": dict(color="black", linestyle="--", linewidth=1.6),
    "history_endpoint": dict(marker="s", s=70, color="black", zorder=5),
    "gt": dict(color="green", linestyle="-", linewidth=1.8),
    "novid": dict(color="blue", linestyle="-", linewidth=1.8, alpha=0.7),
    "vid": dict(color="red", linestyle="-", linewidth=3.0, alpha=0.9),
}


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------
def load_checkpoint_state(scene: str, vid_status: str,
                          results_dir: Path = RESULTS_DIR,
                          map_location: str = "cpu") -> dict:
    """Return the model state_dict stored in ``checkpoint_best.pt``.

    The trainer saves a payload dict
        {'step', 'model', 'opt', 'ema', 'scheduler', 'scaler'}
    where ``model`` is the unwrapped FlowMatcher(denoiser) state. Falls back
    gracefully if a bare state_dict was saved instead.
    """
    ckpt_path = results_dir / f"_SDD_ho{scene}_{vid_status}" / "models" / "checkpoint_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    import torch

    payload = torch.load(ckpt_path, map_location=map_location, weights_only=True)
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"]
    return payload


def build_model_and_cfg(scene: str, vid_status: str, device: str):
    """Construct cfg + FlowMatcher and load best weights.

    This is the ONLY integration point left open: wire in your native
    config/model constructors exactly as training does.
    """
    raise NotImplementedError(
        "Wire your native pipeline here — mirror fm_sdd_global.init_basics().\n"
        "\n"
        "Sketch (matches the committed teacher stack):\n"
        "    from utils.config import Config\n"
        "    from models.backbone_eth_ucy import ETHMotionTransformer\n"
        "    from models.flow_matching import FlowMatcher\n"
        "\n"
        f"    cfg = Config(str(REPO_ROOT / 'cfg/sdd/cor_fm.yml'), 'viz_{scene}')\n"
        "    # ... mirror the init_basics() attribute overrides here ...\n"
        "    logger = logging.getLogger('viz')\n"
        "    model = ETHMotionTransformer(model_config=cfg.MODEL,\n"
        "                                 logger=logger, config=cfg)\n"
        f"    denoiser = FlowMatcher(cfg, model, logger=logger).to('{device}').eval()\n"
        "\n"
        "    state = load_checkpoint_state(scene, vid_status)\n"
        "    denoiser.load_state_dict(state)          # strict=True by design;\n"
        "    # tip: swap in payload['ema'] if you prefer EMA weights for viz.\n"
    )


def get_batch_for_scene(scene: str):
    """Fetch one evaluation batch through YOUR native dataloader.

    Returns ``(obs_traj, x_data)`` where
        obs_traj : tensor [B, A, T_obs, 2+]  observation window
        x_data   : dict consumed by FlowMatcher.sample / p_losses
                    (must include 'past_traj_original_scale' etc.)
    """
    # TODO: plug in SDDGlobalDataset / ETH-UCY loader with held-out scene.
    raise NotImplementedError("Plug in your native dataloader here.")


def sample_prediction(denoiser, x_data: dict):
    """Run flow-matching inference and return a [T_fut, 2] trajectory.

    Reference API (committed models/flow_matching.py)::

        y_pred, *_ = denoiser.sample(x_data, num_trajs=K)
        # y_pred: [B, K, A, F, 2] — reduce K via min-ADE vs gt or take [:, 0].
    """
    # TODO: call your FlowMatcher sampling loop; select the trajectory to draw.
    raise NotImplementedError(
        "Call denoiser.sample(x_data, num_trajs=...) and select best-of-K."
    )


# ---------------------------------------------------------------------------
# Plotting core (torch-optional: accepts tensors or array-likes)
# ---------------------------------------------------------------------------
def _as_xy(traj) -> np.ndarray:
    """Coerce a trajectory to float ndarray [T, 2].

    Accepts torch.Tensor or array-like with shape [T, 2], [A, T, 2] or
    [B, A, T, 2]; leading singleton/agent dims are squeezed (agent 0 wins).
    """
    if hasattr(traj, "detach"):                      # torch.Tensor
        traj = traj.detach().cpu().numpy()
    arr = np.asarray(traj, dtype=float)
    while arr.ndim > 2:                              # [B?, A?, T, 2] -> [T, 2]
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[-1] < 2:
        raise ValueError(f"expected [..., T, >=2] trajectory, got {arr.shape}")
    return arr


def plot_trajectory_comparison(
    obs_traj,
    gt_future,
    pred_novid,
    pred_vid,
    title: str = "",
    save_path: Path | None = None,
    ax=None,
):
    """Plot history vs ground truth vs both predictions on one axis.

    Parameters are tensors/array-likes shaped [..., T, 2+] (see _as_xy).
    Returns the Matplotlib Axes for further tweaking.
    """
    hist = _as_xy(obs_traj)
    gt = _as_xy(gt_future)
    novid = _as_xy(pred_novid)
    vid = _as_xy(pred_vid)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 6))

    # History: black dashed line + distinct marker at the FINAL timestep.
    ax.plot(hist[:, 0], hist[:, 1], label="History", **STYLE["history"])
    ax.scatter(hist[-1, 0], hist[-1, 1], label="_final obs.", **STYLE["history_endpoint"])

    ax.plot(gt[:, 0], gt[:, 1], label="Ground Truth", **STYLE["gt"])
    ax.plot(novid[:, 0], novid[:, 1], label="No Video Prediction", **STYLE["novid"])
    ax.plot(vid[:, 0], vid[:, 1], label="Global VE Prediction", **STYLE["vid"])

    ax.set_title(title or "Trajectory Comparison")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", framealpha=0.9)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig = ax.figure
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[viz] wrote {save_path}")

    if standalone:
        plt.close(fig)
    return ax


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_scene(scene: str, device: str, out_dir: Path) -> None:
    """End-to-end per-scene flow: load models -> sample -> plot."""
    obs_traj, x_data = get_batch_for_scene(scene)                     # TODO hook
    gt_future = x_data["fut_traj_original_scale"]                     # adjust key if needed

    denoiser_novid = build_model_and_cfg(scene, "novid", device)      # TODO hooks below
    pred_novid = sample_prediction(denoiser_novid, x_data)

    denoiser_vid = build_model_and_cfg(scene, "vid", device)
    pred_vid = sample_prediction(denoiser_vid, x_data)

    plot_trajectory_comparison(
        obs_traj,
        gt_future,
        pred_novid,
        pred_vid,
        title=f"SDD '{scene}' — No Video vs Global VE",
        save_path=out_dir / f"{scene}_trajectory_comparison.png",
    )


def run_demo(out_dir: Path) -> None:
    """Styling smoke-test with synthetic geometry (no checkpoints needed)."""
    t = np.linspace(0, np.pi, 8)
    obs = np.stack([t, -0.4 * t], axis=1)
    tf = np.linspace(np.pi, 2 * np.pi, 12)
    gt = np.stack([tf, -0.4 * tf], axis=1)
    drift = lambda k: np.stack([tf + 0.08 * np.sin(k * tf), -0.4 * tf + k], axis=1)
    plot_trajectory_comparison(
        obs, gt, drift(-0.55), drift(0.30),
        title="DEMO — styling reference",
        save_path=out_dir / "demo_trajectory_comparison.png",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scenes", nargs="*", default=list(SCENES))
    p.add_argument("--device", default="cuda" if _torch_cuda() else "cpu")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--demo", action="store_true",
                   help="Plot synthetic trajectories to verify styling only.")
    return p


def _torch_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.demo:
        run_demo(out_dir)
        return
    for scene in args.scenes:
        run_scene(scene, args.device, out_dir)


if __name__ == "__main__":
    main()
