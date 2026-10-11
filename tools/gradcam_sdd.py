"""Grad-CAM attribution for the SDD ResNet-18 scene video encoder.

What this produces
------------------
For a **trained** FlowMatching denoiser conditioned on ``z_video_global``
(the mean-pooled ResNet-18 features of the 8 observed frames), compute
per-frame gradient-weighted class-activation-mapping heatmaps over the
last convolutional feature maps of the backbone:

1. The 8 observed frames ``[a-7 .. a]`` are batched through the *frozen*
   ResNet-18 backbone with gradient tracking enabled; a forward hook
   captures the ``layer4`` feature maps ``A^f in [8, 512, 7, 7]``.
2. ``z_video_global`` is taken to be the temporal mean of the backbone's
   pooled features (identical to the training-time feature pipeline), so
   gradients can flow back all the way into ``A``.
3. A **gradient-enabled** Euler integration of the trained velocity field
   (a re-implementation of ``FlowMatcher.sample`` that is NOT wrapped in
   ``torch.inference_mode``) produces the K=20 candidate futures.
4. The target score is ``y^c = -ADE_min`` (negated best-of-20 min-ADE in
   the original pixel scale — *higher is better*).
5. Per frame, ``alpha_k = (1/|HW|) sum_ij d y^c/dA^k_ij`` (spatial GAP of
   the gradients) and ``CAM = ReLU(sum_k alpha_k A^k)``.

Heatmaps are bilinearly upsampled to source-frame resolution, colormapped
and alpha-blended over the SDD frames, and persisted under
``results_sdd/gradcam_visualizations/<scene>/w<idx>/`` together with the
raw ``A`` / ``alpha`` / ``CAM`` tensors.

Scope / guarantees
------------------
* Pure inference: no training, no optimizer; the backbone and the denoiser
  are both frozen (the backbone's weights get ``requires_grad=False`` — only
  the *activations* from the input frames are tracked).
* Follows the exact evaluation semantics of ``sample_prediction`` in
  ``tools/visualize_trajectory_comparison.py``: same min-max normalization,
  same best-of-K ADE selection, same RNG seeding (``--seed``, default 42).
* Does not modify any core training loop.

Example::

    python tools/gradcam_sdd.py \\
        --scene gates \\                          # LOSO held-out scene
        --ckpt <results_sdd>/.../_SDD_ho<scene>_vmfull/models/checkpoint_best.pt \\
        --sdd-root /home/efrei_stage/Desktop/Datasets/SDD \\
        --video-features-root ./features/resnet18 \\
        --window-index 104
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
from pathlib import Path
from typing import Callable

# Allow running as a plain script (`python tools/gradcam_sdd.py`) from anywhere.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
from einops import rearrange
from torch import nn

from data.dataloader_sdd_global import SDD_PAST_FRAMES, SDD_SCENES
from models.backbone_eth_ucy import ETHMotionTransformer
from models.flow_matching import FlowMatcher
from tools.visualize_trajectory_comparison import (_build_cfg,
                                                   build_model_and_cfg,
                                                   get_batch_for_scene,
                                                   load_anchor_frame,
                                                   load_checkpoint_state)
from utils.normalization import unnormalize_min_max
from video_encoder.sdd_adapter import expand_sdd_root

CFG_PATH = _REPO_ROOT / "cfg" / "sdd" / "cor_fm.yml"
DEFAULT_OUT_DIR = _REPO_ROOT / "results_sdd" / "gradcam_visualizations"


class _NullLogger:
    """Silent logger for offline inference."""

    def info(self, *a, **k) -> None:  # noqa: ANN002, ARG002
        pass

    def warning(self, *a, **k) -> None:  # noqa: ANN002, ARG002
        pass


# ---------------------------------------------------------------------------
# Backbone construction (replica of video_encoder.backbones minus the
# mandatory pretrained weights — tests run offline with --no-pretrained)
# ---------------------------------------------------------------------------
def _make_backbone(
    name: str = "resnet18",
    pretrained: bool = True,
    device: str = "cpu",
) -> tuple[nn.Module, int, Callable]:
    """Return ``(frozen_resnet, feat_dim, preprocess_fn)`` on ``device``.

    ``preprocess_fn`` consumes a PIL.Image and returns the normalized
    ``[3, 224, 224]`` float tensor used at feature-extraction time.
    """
    import torchvision
    from torchvision import transforms

    if name == "resnet18":
        m = torchvision.models.resnet18(
            weights=(
                torchvision.models.ResNet18_Weights.IMAGENET1K_V1
                if pretrained
                else None
            )
        )
        dim = 512
    elif name == "resnet50":
        m = torchvision.models.resnet50(
            weights=(
                torchvision.models.ResNet50_Weights.IMAGENET1K_V2
                if pretrained
                else None
            )
        )
        dim = 2048
    else:
        raise ValueError(f"Unsupported backbone '{name}' (use resnet18/resnet50).")

    m.fc = nn.Identity()
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    m = m.to(device)
    return m, dim, preprocess


# ---------------------------------------------------------------------------
# Gradient-enabled Euler integration of the trained velocity field
# ---------------------------------------------------------------------------
def euler_sample_predictions(
    denoiser: FlowMatcher,
    cfg,
    x_data: dict,
    num_frames: int,
    device: str,
    seed: int,
) -> torch.Tensor:
    """Gradient-enabled replica of ``FlowMatcher.sample`` (euler solver).

    Returns the final ``y_t`` tensor ``[B, K, A, F*D]`` — every operation is
    differentiable w.r.t. ``x_data['z_video_global']``. The RNG is re-seeded
    to ``seed`` immediately before the initial noise draw so each run is
    reproducible.
    """
    K = int(cfg.denoising_head_preds)
    S = int(cfg.sampling_steps)
    dt = 1.0 / S

    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    y_t = torch.randn((1, K, 1, num_frames * 2), device=device, dtype=torch.float)
    y_t.requires_grad_(True)

    for idx in range(S):
        cur_t = dt * idx  # t_ls = dt * arange(S); starts at 0 (pure noise)
        batched_t = torch.full((1,), float(cur_t), device=device)
        preds = denoiser.model_predictions(y_t, x_data, batched_t, flag_print=False)
        y_t = y_t + preds.pred_vel * dt

    return y_t


def best_of_k_target_score(
    pred: torch.Tensor, gt: torch.Tensor, cfg
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(score, best_idx, per_hyp_ade)``.

    ``pred`` is ``[B, K, A, F*D]`` in *normalized* space; both are mapped back
    to the original (pixel) scale with the checkpoint's own train statistics
    before the best-of-K ADE is computed. The target score is the **negated**
    min-ADE, so it is maximised by the better prediction.
    """
    pred_metric = rearrange(pred, "b k a (f d) -> b k a f d", f=cfg.future_frames)[
        ..., :2
    ]
    pred_metric = unnormalize_min_max(
        pred_metric, cfg.fut_traj_min, cfg.fut_traj_max, -1, 1
    )
    # A == 1 for the SDD argumentation — flatten the agent axis so distances
    # are [B, K] cleanly.
    pred_metric = pred_metric[:, :, 0, ...]  # [B, K, F, 2]
    gt_metric = gt[:, 0, ...]  # [B, F, 2] pixel scale
    distances = torch.linalg.norm(pred_metric - gt_metric[:, None, ...], dim=-1).mean(
        dim=-1
    )  # [B, K]
    per_hyp_ade = distances[0]  # [K]
    best_idx = per_hyp_ade.argmin()
    score = -per_hyp_ade[best_idx]
    return score, best_idx, per_hyp_ade


def grad_cam_maps(
    A: torch.Tensor, score: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Grad-CAM for a batch of feature maps (the core attribution step).

    Args:
        A: ``[T, C, h, w]`` last-conv activations of the encoder.
        score: scalar target ``y^c`` with a graph path into ``A``.

    Returns:
        ``(alphas, cam)`` where ``alphas`` is ``[T, C, 1, 1]``
        (spatial GAP of the gradients, Eq. 1) and ``cam`` is ``[T, h, w]``
        with ``ReLU(sum_k alpha_k A^k)`` (Eq. 2).
    """
    grads = torch.autograd.grad(score, A)[0]
    alphas = grads.mean(dim=(2, 3), keepdim=True)  # [T, C, 1, 1]
    cam = torch.clamp((alphas * A).sum(dim=1), min=0.0)  # [T, h, w]
    return alphas, cam


def gradcam_pass(
    backbone: nn.Module,
    denoiser: FlowMatcher,
    cfg,
    x_data: dict,
    frames_t: torch.Tensor,
    gt: torch.Tensor,
    num_frames: int,
    device: str,
    seed: int,
) -> dict:
    """Full differentiable Grad-CAM pass for one window.

    Args:
        backbone: frozen ResNet (``fc=Identity``), already on ``device``.
        denoiser: loaded FlowMatcher in eval mode.
        cfg: built config (norm stats + K / S / horizon).
        x_data: web of trajectory conditioning; ``z_video_global`` is
            replaced here by the gradient-carrying mean-pooled features.
        frames_t: ``[T, 3, 224, 224]`` normalized observed frames.
        gt: ``[1, A, F, 2]`` ground-truth future in the original (pixel) scale.
        num_frames: future horizon ``F``.
        device: ``'cuda'`` / ``'cpu'``.
        seed: RNG seed for the initial noise draw.

    Returns a dict with ``cam [T, h, w]``, ``alphas [T, C]``,
    ``activations [T, C, h, w]``, ``score``, ``best_idx``, ``per_hyp_ade``,
    ``z_video_global [1, D_raw]``.
    """
    frames_t = frames_t.to(device)
    # Make the input a grad-requiring tensor: with the backbone's weights
    # frozen, gradients must travel from the score back through the input
    # pixels into the layer4 features. ``torch.stack`` of ToTensor outputs
    # yields a leaf (no grad_fn), so requires_grad_ is legal here.
    frames_t.requires_grad_(True)

    captured: list[torch.Tensor] = []

    def _hook(_mod, _inp, out) -> None:
        captured.append(out)

    with torch.enable_grad():
        hook = backbone.layer4.register_forward_hook(_hook)
        try:
            feats = backbone(frames_t)  # [T, dim] (fc = Identity)
        finally:
            hook.remove()
        A = captured[0]  # [T, 512, 7, 7]

        z_video = feats.mean(dim=0, keepdim=True).contiguous()  # [1, dim]
        x_data = dict(x_data)
        x_data["z_video_global"] = z_video

        y_t = euler_sample_predictions(denoiser, cfg, x_data, num_frames, device, seed)
        score, best_idx, per_hyp_ade = best_of_k_target_score(y_t, gt, cfg)

        alphas, cam = grad_cam_maps(A, score)  # [T, C, 1, 1] and [T, h, w]

    return {
        "cam": cam.detach().cpu(),
        "alphas": alphas.detach().cpu(),
        "activations": A.detach().cpu(),
        "z_video_global": z_video.detach().cpu(),
        "score": float(score.detach().cpu()),
        "best_idx": int(best_idx.detach().cpu()),
        "per_hyp_ade": per_hyp_ade.detach().cpu().numpy(),
    }


# ---------------------------------------------------------------------------
# Heatmap rendering utilities (pure NumPy — independently testable)
# ---------------------------------------------------------------------------
def _jet_lut() -> np.ndarray:
    """Return the 256-entry jet colormap as ``float32 [256, 3]`` (0..1)."""
    n = 256
    r = np.clip(
        np.minimum(
            np.minimum(4.0 * (np.arange(n) / n) - 1.5, -4.0 * (np.arange(n) / n) + 4.5),
            1.0,
        ),
        0.0,
        1.0,
    )
    g = np.clip(
        np.minimum(
            np.minimum(4.0 * (np.arange(n) / n) - 0.5, -4.0 * (np.arange(n) / n) + 3.5),
            1.0,
        ),
        0.0,
        1.0,
    )
    b = np.clip(
        np.minimum(
            np.minimum(4.0 * (np.arange(n) / n) + 0.5, -4.0 * (np.arange(n) / n) + 1.5),
            1.0,
        ),
        0.0,
        1.0,
    )
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def normalize_heatmap(cam: np.ndarray) -> np.ndarray:
    """Min-max normalise a single camera map to ``[0, 1]`` (safe for flat maps)."""
    lo, hi = float(cam.min()), float(cam.max())
    if not np.isfinite(hi - lo) or hi - lo <= 1e-12:
        return np.zeros_like(cam, dtype=np.float32)
    return ((cam - lo) / (hi - lo)).astype(np.float32)


def CAM_to_color(cam01: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Map a ``[H, W]`` 0..1 map through a colormap (``[H, W, 3]`` in 0..1)."""
    idx = np.clip(np.rint(cam01 * 255.0).astype(np.int32), 0, 255)
    return lut[idx]  # [H, W, 3]


def overlay_heatmap(
    frame_rgb: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.45,
    lut: np.ndarray | None = None,
) -> np.ndarray:
    """Blend ``cam`` over ``frame_rgb``, upscaled to the frame resolution."""
    if lut is None:
        lut = _jet_lut()
    from PIL import Image

    h, w = frame_rgb.shape[:2]
    cam01 = normalize_heatmap(cam)
    img = Image.fromarray((cam01 * 255.0).astype(np.uint8), mode="L")
    cam_resized = (
        np.asarray(img.resize((w, h), Image.BILINEAR)).astype(np.float32) / 255.0
    )
    color = CAM_to_color(cam_resized, lut)  # [H, W, 3] 0..1
    frame = frame_rgb.astype(np.float32) / 255.0
    blended = frame * (1.0 - alpha) + color * alpha
    return (np.clip(blended, 0.0, 1.0) * 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cfg", default=str(CFG_PATH))
    p.add_argument(
        "--scene",
        required=True,
        choices=list(SDD_SCENES),
        help="LOSO held-out scene whose vmfull checkpoint is attributed.",
    )
    p.add_argument(
        "--ckpt",
        default=None,
        help="Explicit checkpoint path. Defaults to the auto-derived vmfull "
        "run dir for --scene.",
    )
    p.add_argument(
        "--window-index",
        type=int,
        default=0,
        help="0-based test-split window index (Q1 evaluation sample).",
    )
    p.add_argument(
        "--max-windows",
        type=int,
        default=1,
        help="Render windows [--window-index, --window-index+max-windows).",
    )
    p.add_argument("--sampling_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="overlay blend fraction in [0, 1] (default 0.45).",
    )
    p.add_argument("--backbone", choices=["resnet18", "resnet50"], default="resnet18")
    p.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load ImageNet weights (disable for offline tests).",
    )
    p.add_argument("--device", default=None, help="'cuda' or 'cpu' (default: auto).")
    p.add_argument("--sdd-root", default=None)
    p.add_argument("--video-features-root", dest="video_features_root", default=None)
    p.add_argument(
        "--video-id",
        default=None,
        help="Restrict LOSO windows to one video folder name.",
    )
    p.add_argument(
        "--use-ema",
        action="store_true",
        help="prefer EMA weights over the raw model weights.",
    )
    p.add_argument(
        "--num-workers", type=int, default=None, help="DataLoader workers (default 0)."
    )
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    return p.parse_args(argv)


def _build_denoiser(scene: str, args: argparse.Namespace) -> tuple[FlowMatcher, object]:
    if args.ckpt:
        cfg = _build_cfg(scene, "vid", args)
        model = ETHMotionTransformer(
            model_config=cfg.MODEL, logger=_NullLogger(), config=cfg
        )
        denoiser = FlowMatcher(cfg, model, logger=_NullLogger())
        state = load_checkpoint_state(Path(args.ckpt), use_ema=args.use_ema)
        denoiser.load_state_dict(state)
        denoiser.to(cfg.device)
        denoiser.eval()
        return denoiser, cfg
    denoiser, cfg = build_model_and_cfg(scene, "vid", args)
    return denoiser, cfg


def _save_single(
    out: Path,
    frame_id: int,
    frame_rgb: np.ndarray,
    cam: np.ndarray,
    lut: np.ndarray,
    alpha: float,
) -> None:
    from PIL import Image

    if frame_rgb is not None:
        blended = overlay_heatmap(frame_rgb, cam, alpha=alpha, lut=lut)
        Image.fromarray(blended).save(out / f"frame_f{int(frame_id):06d}_overlay.jpg")
        heat = (CAM_to_color(normalize_heatmap(cam), lut) * 255.0).astype(np.uint8)
        Image.fromarray(heat).save(out / f"frame_f{int(frame_id):06d}_cam.png")
    else:
        # No scene frame available — persist the raw heatmap as an image anyway.
        heat = (CAM_to_color(normalize_heatmap(cam), lut) * 255.0).astype(np.uint8)
        Image.fromarray(heat).save(out / f"frame_f{int(frame_id):06d}_cam.png")


def _save_manifest_rows(out_root: Path, rows: list[dict[str, str]]) -> Path:
    fieldnames = [
        "scene",
        "window_index",
        "video_id",
        "anchor_frame",
        "seed",
        "backbone",
        "ade_min",
        "best_hyp",
        "cam_score",
        "out_dir",
    ]
    manifest = out_root / "manifest.csv"
    new = not manifest.exists()
    with manifest.open("a", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        if new:
            writer.writeheader()
        writer.writerows(rows)
    return manifest


def run_gradcam(scene: str, args: argparse.Namespace, out_dir: Path) -> list[Path]:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    denoiser, cfg = _build_denoiser(scene, args)
    backbone, _dim, preprocess = _make_backbone(
        args.backbone, pretrained=args.pretrained, device=device
    )
    num_frames = int(cfg.future_frames)
    lut = _jet_lut()

    out_root = out_dir / scene
    out_root.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    manifest_rows: list[dict[str, str]] = []

    for off in range(int(args.max_windows)):
        args.window_index = args.window_index + off
        batch = get_batch_for_scene(scene, args)  # CPU; window_index selector

        video_id = batch["video_id"][0]
        anchor = int(batch["anchor_frame"][0])

        # The 8 observed frames are past_frame_ids = [a-7 .. a].
        past_ids = [
            max(int(anchor) - SDD_PAST_FRAMES + 1 + i, 0)
            for i in range(SDD_PAST_FRAMES)
        ]

        sdd_resolved = expand_sdd_root(args.sdd_root)
        from PIL import Image as _PIL

        # Strict coordinate synchronization: past_ids spans exactly the 8
        # observed frames [a-7 .. a]. Frames the raw video cannot provide
        # (window near frame 0) are filled by replicating the most recent
        # available frame so the *backbone batch* always has 8 entries and
        # every CAM row maps 1:1 onto past_ids / frames_rgb.
        frames_rgb: list[np.ndarray] = []
        frames_pil: list = []
        missing = 0
        latest: np.ndarray | None = None
        for fid in past_ids:
            got = load_anchor_frame(sdd_resolved, scene, video_id, fid)
            if got is not None:
                img_rgb, _extent, _src = got
                latest = img_rgb
            else:
                missing += 1
            if latest is None:
                raise RuntimeError(
                    f"[gradcam:{scene}] no observable frame decoded before "
                    f"past frame {fid} — cannot attribute the video branch."
                )
            frames_rgb.append(latest.copy())
            frames_pil.append(_PIL.fromarray(latest))
        if missing:
            print(
                f"[gradcam:{scene}] {missing}/{len(past_ids)} observed frames were "
                "unavailable and replicated from the nearest decoded frame "
                "(window near video start)."
            )
        frames_t = torch.stack([preprocess(im) for im in frames_pil])
        if frames_t.shape[0] != SDD_PAST_FRAMES:
            raise RuntimeError(
                f"[gradcam:{scene}] expected {SDD_PAST_FRAMES} observed frames, "
                f"got {frames_t.shape[0]}."
            )

        x_data = {
            "past_traj_original_scale": batch["past_traj_original_scale"].to(device),
            "fut_traj": batch["fut_traj"].to(device),
        }
        gt = batch["fut_traj_original_scale"].to(device).float()

        res = gradcam_pass(
            backbone,
            denoiser,
            cfg,
            x_data,
            frames_t,
            gt,
            num_frames,
            device,
            args.seed,
        )

        win_dir = out_root / f"w{args.window_index:05d}"
        win_dir.mkdir(parents=True, exist_ok=True)

        cam = np.asarray(res["cam"])  # [T, 7, 7]
        for i, fid in enumerate(past_ids):
            if i >= cam.shape[0]:
                break
            _save_single(win_dir, fid, frames_rgb[i], cam[i], lut, args.alpha)
            saved.append(win_dir / f"frame_f{fid:06d}_overlay.jpg")

        np.savez(
            win_dir / "_gradcam.npz",
            cam=np.asarray(res["cam"]),
            alphas=np.asarray(res["alphas"]),
            activations=np.asarray(res["activations"]),
            z_video_global=np.asarray(res["z_video_global"]),
            per_hyp_ade=res["per_hyp_ade"],
            past_frame_ids=np.asarray(past_ids, dtype=np.int64),
            score=np.asarray(res["score"]),
            best_idx=np.asarray(res["best_idx"]),
        )

        manifest_rows.append(
            {
                "scene": scene,
                "window_index": str(args.window_index),
                "video_id": video_id,
                "anchor_frame": str(anchor),
                "seed": str(args.seed),
                "backbone": args.backbone,
                "ade_min": f"{float(res['per_hyp_ade'][res['best_idx']]):.6f}",
                "best_hyp": str(res["best_idx"]),
                "cam_score": f"{res['score']:.6f}",
                "out_dir": str(win_dir),
            }
        )
        print(
            f"[gradcam:{scene}] window {args.window_index} "
            f"(video={video_id}, anchor={anchor}) "
            f"ade_min={float(res['per_hyp_ade'][res['best_idx']]):.4f} "
            f"best_hyp={res['best_idx']} frames={past_ids}"
        )

        # release window tensors + force GC between windows
        del batch, frames_t, frames_rgb, frames_pil, x_data, res
        gc.collect()

    manifest = _save_manifest_rows(out_root, manifest_rows)
    saved.append(manifest)
    return saved


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    out_dir = Path(args.out_dir)
    saved = run_gradcam(args.scene, args, out_dir)
    print(f"[gradcam:{args.scene}] {len(saved)} files written under {out_dir}")


if __name__ == "__main__":
    main()
