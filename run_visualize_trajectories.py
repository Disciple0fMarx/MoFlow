"""Main evaluation/visualization loop: render academic SDD trajectory figures.

By default **one random window per scene** is selected with a seedable RNG
(``--num-samples`` / ``--seed``) and rendered across the configured model
variants, so a (scene, pedestrian, frame) tuple produces one standalone
300-DPI PNG per variant into
``{out_dir}/{scene_id}_ped{agent_id}_{fname_suffix}_{variant_name}.png`` (see
:mod:`visualize_trajectories` for the rendering contract).  The same window
set is rendered for every variant so the figures are directly comparable.

Target selection can be overridden with explicit filters: ``--ped-id``
(track id) and/or ``--frame-id`` (anchor frame number).  When provided,
random sampling is skipped and the single matching window is rendered.

Variants (each rendered independently, never combined in one panel):

* ``no_video``      — trajectory-only model (``USE_VIDEO=False``);
* ``global_video``  — scene-level video encoder (``z_video_global``);
* ``agent_centric`` — agent-crop video encoder + tri-modal fusion
  (mirrors ``viz_agent_sdd.py``).

Checkpoints may be given explicitly (``--variant-ckpt no_video=/path.pt``,
repeated) or discovered under ``--checkpoint-dir`` by scene run-tag:

    "no_video":      `<dir>/_SDD_ho<scene>_{novid,novideo,baseline}/models/checkpoint_best.pt`
    global_video:   `<dir>/_SDD_ho<scene>_{vid,globalvideo,global}/models/checkpoint_best.pt`
    agent_centric:  `<dir>/_SDD_ho<scene>_agent/models/checkpoint_best.pt`

When a checkpoint is missing for a variant it is skipped with a printed
message — run with ``--variants agent_centric --variant-ckpt ...`` to be
explicit. Normalization statistics are reused from the same on-disk cache as
``viz_agent_sdd.py`` (``RESULTS_DIR/_norm_stats_ho<scene>.npz``) so metrics
are pixel-identical to production evaluation.

Example (lab machine)::

    # 1 random window per scene, deterministic (1 PNG per variant)
    python run_visualize_trajectories.py \\
        --checkpoint-dir results_sdd/cor_fm \\
        --video-features-root results_sdd/video_features \\
        --num-samples 1 --seed 42

    # specific pedestrian + frame in one scene
    python run_visualize_trajectories.py \\

    # checkpoint-free styling self-check
    python run_visualize_trajectories.py --demo
"""
from __future__ import annotations

import argparse
import gc
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from einops import rearrange  # noqa: E402

from data.agent_crop_sdd import SDDAgentCropDataset  # noqa: E402
from data.dataloader_sdd_global import (  # noqa: E402
    SDD_AGENTS_PER_WINDOW,
    SDD_PAST_FRAMES,
    SDD_SCENES,
    SDDGlobalDataset,
)
from models.backbone_eth_ucy import ETHMotionTransformer  # noqa: E402
from models.flow_matching import FlowMatcher  # noqa: E402
from tools.visualize_trajectory_comparison import (  # noqa: E402
    _move_to_device,
    _quiet_logger,
    load_anchor_frame,
    load_checkpoint_state,
)
from utils.config import Config  # noqa: E402
from utils.normalization import unnormalize_min_max  # noqa: E402
from video_encoder.sdd_adapter import expand_sdd_root  # noqa: E402
from visualize_trajectories import (  # noqa: E402
    VARIANTS,
    ade_fde_best,
    render_scene_trajectories,
    world_to_pixel,
)

CFG_PATH = REPO_ROOT / "cfg" / "sdd" / "cor_fm.yml"
RESULTS_DIR = REPO_ROOT / "results_sdd" / "cor_fm"
DEFAULT_OUT_DIR = REPO_ROOT / "visualizations"
DEFAULT_VIDEO_RESOLUTION = (1280, 720)  # white placeholder when no frame resolves

logger = logging.getLogger("viz-loop")


# ---------------------------------------------------------------------------
# Config construction (mirrors fm_sdd_agent / viz_agent_sdd defaults)
# ---------------------------------------------------------------------------
def _variant_flags(variant: str) -> dict:
    """Per-variant architecture settings forced INTO the built model."""
    if variant == "no_video":
        return {"USE_VIDEO": False, "USE_AGENT_VIDEO": False, "USE_TRI_MODAL_FUSION": False}
    if variant == "global_video":
        return {"USE_VIDEO": True, "USE_AGENT_VIDEO": False, "USE_TRI_MODAL_FUSION": False}
    if variant == "agent_centric":
        return {"USE_VIDEO": False, "USE_AGENT_VIDEO": True, "USE_TRI_MODAL_FUSION": True}
    raise ValueError(f"unknown variant {variant!r}")


def _norm_stats(scene: str, args: argparse.Namespace) -> tuple[float, float, float, float]:
    """Return LOSO normalization statistics (cached, shared with viz_agent_sdd)."""
    cache = RESULTS_DIR / f"_norm_stats_ho{scene}.npz"
    if cache.exists():
        data = np.load(cache)
        return (float(data["past_min"]), float(data["past_max"]),
                float(data["fut_min"]), float(data["fut_max"]))
    logger.info("[%s] computing norm stats (one-time)...", scene)
    cfg = Config(str(CFG_PATH), tag="viz-stats")
    dset = SDDGlobalDataset(cfg, training=True, sdd_root=args.sdd_root,
                            held_out_scene=scene, split="train", use_video=False)
    stats = (float(dset.past_traj_min), float(dset.past_traj_max),
             float(dset.fut_traj_min), float(dset.fut_traj_max))
    del dset, cfg
    gc.collect()
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(cache, past_min=stats[0], past_max=stats[1],
                 fut_min=stats[2], fut_max=stats[3])
    except OSError as exc:
        logger.warning("could not cache norm stats (%s)", exc)
    return stats


def _build_variant_cfg(scene: str, variant: str, args: argparse.Namespace) -> Config:
    """Build the project ``Config`` matching the checkpoint's modality."""
    cfg = Config(str(CFG_PATH), tag="viz")

    # ---- FM / arch defaults (fm_sdd_agent.py defaults) ----------------------
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
    ce.AGENTS = SDD_AGENTS_PER_WINDOW
    ce.AGENT_ENCODER_TYPE = args.agent_encoder if variant == "agent_centric" else "compact"
    if args.video_features_root is not None:
        ce.VIDEO_FEATURES_ROOT = str(Path(args.video_features_root).expanduser())

    # ---- per-variant modality ----------------------------------------------
    for key, value in _variant_flags(variant).items():
        setattr(ce, key, value)

    # ---- normalization statistics -------------------------------------------
    past_min, past_max, fut_min, fut_max = _norm_stats(scene, args)
    cfg.past_traj_min, cfg.past_traj_max = past_min, past_max
    cfg.fut_traj_min, cfg.fut_traj_max = fut_min, fut_max

    cfg.device = args.device
    return cfg


# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------
# Run-tag naming scheme: _SDD_ho<scene>_novid | _SDD_ho<scene>_vid | _SDD_ho<scene>_agent.
# The preferred tag is listed first; older aliases are kept as fallbacks.
_DISCOVERY_SUFFIXES = {
    "no_video": ["_SDD_ho{scene}_novid", "_SDD_ho{scene}_novideo",
                 "_SDD_ho{scene}_baseline"],
    "global_video": ["_SDD_ho{scene}_vid", "_SDD_ho{scene}_globalvideo",
                     "_SDD_ho{scene}_global"],
    "agent_centric": ["_SDD_ho{scene}_agent"],
}

#: Filenames probed inside a run directory, ``<dir>/models/`` first.
CKPT_NAMES = ("checkpoint_best.pt", "checkpoint_best.pth",
              "model_best.pt", "model_best.pth", "model.pt", "model.pth")


def _find_checkpoint(run_dir: Path) -> Path | None:
    """First existing checkpoint inside ``run_dir`` (``models/`` then flat)."""
    for sub in (run_dir / "models", run_dir):
        if not sub.is_dir():
            continue
        for name in CKPT_NAMES:
            cand = sub / name
            if cand.is_file() and cand.stat().st_size > 0:
                return cand
    return None


def resolve_variant_checkpoints(
    args: argparse.Namespace, scene: str
) -> dict[str, Path | None]:
    """Map each requested variant to its checkpoint path (or ``None``)."""
    out: dict[str, Path | None] = {}
    for variant in args.variants:
        # 1) explicit --variant-ckpt variant=path override
        if variant in args.variant_ckpt:
            out[variant] = Path(args.variant_ckpt[variant])
            continue
        # 2) discovery under --checkpoint-dir by scene run-tag.
        #    Explicit mapping: no_video -> <dir>/_SDD_ho<scene>_novid/models/...
        #    global_video -> <dir>/_SDD_ho<scene>_vid/models/...
        #    agent_centric -> <dir>/_SDD_ho<scene>_agent/models/...
        if args.checkpoint_dir is not None:
            base = Path(args.checkpoint_dir)
            found: Path | None = None
            tried: list[str] = []
            for tag in _DISCOVERY_SUFFIXES[variant]:
                run_dir = base / tag.format(scene=scene)
                tried.append(str(run_dir))
                ck = _find_checkpoint(run_dir)
                if ck is not None:
                    found = ck
                    break
            out[variant] = found
            if found is None:
                print(f"[viz-loop] {scene}/{variant}: no checkpoint found; "
                      f"tried build dirs:\n" + "\n".join(f"  {t}" for t in tried))
            continue
        print(f"[viz-loop] {scene}/{variant}: no --variant-ckpt and no --checkpoint-dir")
        out[variant] = None
    return out


# ---------------------------------------------------------------------------
# Model + batch construction
# ---------------------------------------------------------------------------
def build_model(cfg: Config, ckpt_state: dict, args: argparse.Namespace):
    """Instantiate the denoiser for this variant and load ``ckpt_state``."""
    log = _quiet_logger()
    model = ETHMotionTransformer(model_config=cfg.MODEL, logger=log, config=cfg)
    denoiser = FlowMatcher(cfg, model, logger=log)
    denoiser.load_state_dict(ckpt_state)
    denoiser.to(cfg.device)
    denoiser.eval()
    return denoiser


def _build_crop_index(
    dset: SDDGlobalDataset, scene: str, args: argparse.Namespace
) -> dict[str, tuple[SDDAgentCropDataset, np.ndarray]]:
    """Group test windows by video and build one crop dataset per video."""
    groups: dict[str, list[int]] = defaultdict(list)
    for idx in range(len(dset)):
        groups[dset.windows.video_id(idx)].append(idx)

    index: dict[str, tuple[SDDAgentCropDataset, np.ndarray]] = {}
    resolved_root = expand_sdd_root(args.sdd_root)
    for video_id, idcs in groups.items():
        windows = [
            (
                int(dset.windows.track_id(i)),
                dset.windows.past_frame_ids(i).tolist(),
            )
            for i in idcs
        ]
        crop_ds = SDDAgentCropDataset(
            resolved_root,
            scene,
            video_id,
            track_ids=[t for t, _ in windows],
            past_frames_per_window=[f for _, f in windows],
            crop_size=int(args.crop_size),
            obs_frames=SDD_PAST_FRAMES,
            padding=int(args.padding),
            drop_lost=args.drop_lost,
        )
        index[video_id] = (crop_ds, np.asarray(idcs, dtype=np.int64))
    return index


def _as_batch_frame_id(anchor_frame: int | np.ndarray | torch.Tensor) -> torch.Tensor:
    """Coerce an anchor frame id (int / numpy scalar / tensor) to ``[1]`` int32."""
    if isinstance(anchor_frame, torch.Tensor):
        arr = anchor_frame.detach().cpu().numpy()
    else:
        arr = np.asarray(anchor_frame)
    arr = np.ravel(np.asarray(arr, dtype=np.int32))
    if arr.size != 1:
        arr = arr[:1]
    return torch.from_numpy(arr)


def make_window_batch(
    dset: SDDGlobalDataset,
    idx: int,
    variant: str,
    crop_index: dict[str, tuple[SDDAgentCropDataset, np.ndarray]] | None,
) -> dict:
    """Assemble a B=1 batch for window ``idx`` (mirror of the collate contract)."""
    item = dset[idx]
    batch = {
        "past_traj": item["past_traj"].unsqueeze(0),
        "fut_traj": item["fut_traj"].unsqueeze(0),
        "past_traj_original_scale": item["past_traj_original_scale"].unsqueeze(0),
        "fut_traj_original_scale": item["fut_traj_original_scale"].unsqueeze(0),
        "fut_traj_vel": item["fut_traj_vel"].unsqueeze(0),
        "index": item["index"].unsqueeze(0),
        "batch_size": torch.tensor(1),
        "scene": [item["scene"]],
        "video_id": [item["video_id"]],
        "anchor_frame": _as_batch_frame_id(item["anchor_frame"]),
    }

    zglob = item.get("z_video_global")
    if zglob is not None and zglob.numel() == 1:  # missing-features sentinel
        zglob = None
    if variant == "global_video":
        batch["z_video_global"] = (
            torch.stack([zglob], dim=0) if zglob is not None else None
        )
    else:
        batch["z_video_global"] = None

    if variant == "agent_centric":
        video_id = batch["video_id"][0]
        crop_ds, idcs = crop_index[video_id]
        pos = int(np.where(idcs == idx)[0][0])
        crop_item = crop_ds[pos]
        crops = crop_item["agent_crops"]  # [T_obs, C, S, S] uint8
        batch["agent_crops"] = (
            crops.unsqueeze(0).unsqueeze(1).to(dtype=torch.float32)
        )
        batch["agent_crops_valid"] = torch.tensor([True])

    return batch


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def sample_prediction(denoiser, cfg: Config, batch_cpu: dict) -> dict:
    """Sampling for a B=1 batch → host-side numpy arrays (original scale).

    Predictions come back as *relative* displacement; ``obs_abs`` carries the
    absolute last-observed pixel anchor needed to map them onto the frame.
    """
    with torch.inference_mode():
        x_gpu = _move_to_device(batch_cpu, cfg.device)
        sample_out = denoiser.sample(x_gpu, num_trajs=cfg.denoising_head_preds)
        pred = sample_out[0]
        pred = rearrange(pred, "b k a (f d) -> b k a f d",
                         f=cfg.future_frames)[..., :2]
        if cfg.get("data_norm", "min_max") == "min_max":
            pred = unnormalize_min_max(pred, cfg.fut_traj_min, cfg.fut_traj_max, -1, 1)
        preds_np = rearrange(pred.detach().cpu().numpy(), "b k a f d -> (b a) k f d")
        del x_gpu, sample_out, pred

    gt_np = rearrange(
        batch_cpu["fut_traj_original_scale"].numpy(), "b a f d -> (b a) f d"
    ).astype(np.float64)
    past_full = batch_cpu["past_traj_original_scale"].numpy()
    obs_rel = rearrange(past_full[..., 2:4], "b a p d -> (b a) p d").astype(np.float64)
    obs_abs = rearrange(past_full[..., 0:2], "b a p d -> (b a) p d")[0].astype(np.float64)
    return {"obs_rel": obs_rel, "obs_abs": obs_abs, "gt": gt_np, "preds": preds_np}


# ---------------------------------------------------------------------------
# Target selection (random sampling / explicit ped+frame override)
# ---------------------------------------------------------------------------
def _scene_seed(seed: int, scene: str) -> int:
    """Deterministic per-scene seed so picks stay reproducible for a scene
    regardless of which scenes are run together."""
    scene_hash = sum((i + 1) * ord(ch) for i, ch in enumerate(scene))
    return int(seed) + scene_hash


def _select_targets(scene: str, args: argparse.Namespace) -> list[int]:
    """Choose the dataset window indices to render for ``scene``.

    Explicit ``--ped-id``/``--frame-id`` override random sampling: the
    candidates are filtered and the single (first, deterministic) match wins.
    Otherwise ``--num-samples`` window indices are drawn per scene with the
    seedable RNG (default 1).  The same target list is reused for every model
    variant so figures are directly comparable.
    """
    cfg = Config(str(CFG_PATH), tag="viz-targets")
    dset = SDDGlobalDataset(
        cfg, training=False, sdd_root=args.sdd_root, held_out_scene=scene,
        split="test", use_video=False,
    )
    windows = dset.windows
    candidates = list(range(len(windows)))
    if args.max_agents and args.max_agents > 0:
        candidates = candidates[: int(args.max_agents)]
    del dset, cfg
    gc.collect()

    ped_id = getattr(args, "ped_id", None)
    frame_id = getattr(args, "frame_id", None)
    if ped_id is not None or frame_id is not None:
        if ped_id is not None:
            candidates = [i for i in candidates if int(windows.track_id(i)) == ped_id]
        if frame_id is not None:
            candidates = [
                i for i in candidates if int(windows.anchor_frame(i)) == frame_id
            ]
        targets = sorted(candidates)
        if not targets:
            print(f"[viz-loop] {scene}: no window matches ped_id={ped_id} "
                  f"frame_id={frame_id}; skipping")
            return []
        if len(targets) > 1:
            print(f"[viz-loop] {scene}: {len(targets)} windows match "
                  f"ped_id={ped_id} frame_id={frame_id}; rendering first "
                  f"(window idx {targets[0]}); pass --frame-id to disambiguate")
        return targets[:1]

    rng = random.Random(_scene_seed(args.seed, scene))
    n = min(int(args.num_samples), len(candidates))
    if n <= 0:
        print(f"[viz-loop] {scene}: no candidate windows; skipping")
        return []
    targets = sorted(rng.sample(candidates, n))
    return targets


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _resolve_video_features_root(args: argparse.Namespace) -> str | None:
    """Locate per-scene ``<scene>.npy`` video-features when not explicit.

    ``global_video`` needs cached global features for the scene; without them the
    model receives zero placeholders and renders meaningless futures.  When
    ``--video-features-root`` is omitted we probe the conventional layouts next to
    the discovered checkpoints before giving up.
    """
    if args.video_features_root is not None:
        return str(Path(args.video_features_root).expanduser())
    bases = []
    if args.checkpoint_dir is not None:
        bases.append(Path(args.checkpoint_dir))
    bases.append(Path(DEFAULT_OUT_DIR).parent / "results_sdd" / "cor_fm")
    for base in bases:
        for cand in ("features", "video_features", "results", "results/video_features"):
            p = base / cand
            if p.is_dir() and any(p.glob("*.npy")):
                return str(p)
    return None


def run_scene_variant(
    scene: str, variant: str, ckpt_path: Path, args: argparse.Namespace,
    targets: list[int],
) -> int:
    """Render every selected window in ``scene`` for ``variant``."""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{variant} checkpoint missing: {ckpt_path}")

    ckpt_state = load_checkpoint_state(ckpt_path, use_ema=args.use_ema)
    cfg = _build_variant_cfg(scene, variant, args)
    denoiser = build_model(cfg, ckpt_state, args)

    video_features_root = (
        _resolve_video_features_root(args) if variant == "global_video" else None
    )
    if variant == "global_video" and video_features_root is None:
        print("[viz-loop] WARNING global_video: no video features root found; "
              "model receives zero placeholders. Pass --video-features-root "
              "(dir of per-scene <scene>.npy) for meaningful renderings.")
    dset = SDDGlobalDataset(
        cfg, training=False, sdd_root=args.sdd_root, held_out_scene=scene,
        split="test", use_video=(variant == "global_video"),
        video_features_root=video_features_root,
    )

    crop_index = None
    if variant == "agent_centric":
        crop_index = _build_crop_index(dset, scene, args)

    resolved_root = expand_sdd_root(args.sdd_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_windows = len(dset)
    valid = [int(i) for i in targets if 0 <= int(i) < n_windows]
    skipped = len(targets) - len(valid)
    if skipped:
        print(f"[viz-loop] {scene}/{variant}: skipping {skipped} out-of-range "
              f"window indices (dataset has {n_windows})")
    if not valid:
        print(f"[viz-loop] {scene}/{variant}: no windows to render; skipping")
        del dset, cfg, ckpt_state
        gc.collect()
        torch.cuda.empty_cache()
        return 0
    print(f"[viz-loop] {scene}/{variant}: rendering {len(valid)}/{n_windows} "
          f"selected windows (checkpoint {ckpt_path.name})")

    rendered = 0
    for idx in valid:
        batch_cpu = make_window_batch(dset, idx, variant, crop_index)
        try:
            result = sample_prediction(denoiser, cfg, batch_cpu)
        finally:
            del batch_cpu
            gc.collect()
            torch.cuda.empty_cache()

        anchor = result["obs_abs"][-1]
        obs_abs = result["obs_rel"][0] + anchor[None, :]
        gt_abs = result["gt"][0] + anchor[None, :]
        preds = np.asarray(result["preds"][0], dtype=np.float64) + anchor[None, None, :]

        # Homography projection (identity for SDD pixel space; pure helper use
        # so world-coordinate experiments remain supported).
        obs_abs = world_to_pixel(obs_abs, H=args.homography)
        gt_abs = world_to_pixel(gt_abs, H=args.homography)
        preds = world_to_pixel(preds, H=args.homography)

        video_id = dset.windows.video_id(idx)
        anchor_frame = int(dset.windows.anchor_frame(idx))
        agent_id = int(dset.windows.track_id(idx))

        background = None
        if not args.no_background:
            loaded = load_anchor_frame(
                resolved_root, scene, video_id, anchor_frame,
                override=args.background,
            )
            if loaded is not None:
                background, _, _ = loaded
        if background is None:
            w, h = DEFAULT_VIDEO_RESOLUTION
            background = np.full((h, w, 3), 255, dtype=np.uint8)

        ade_v, fde_v = ade_fde_best(preds, gt_abs)
        fname_suffix = f"{video_id}_f{anchor_frame}"
        save_path = render_scene_trajectories(
            background, obs_abs, gt_abs, preds,
            agent_id=agent_id, scene_id=scene, variant_name=variant,
            save_dir=out_dir, ade=ade_v, fde=fde_v,
            zoom_margin=float(args.zoom_margin), fname_suffix=fname_suffix,
        )
        rendered += 1
        if rendered % 50 == 0:
            print(f"[viz-loop] {scene}/{variant}: {rendered}/{len(valid)} rendered")

    del dset
    if crop_index is not None:
        del crop_index
    del denoiser, cfg, ckpt_state
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[viz-loop] {scene}/{variant}: done ({rendered} figures)")
    return rendered


def run_pipeline(args: argparse.Namespace) -> int:
    scenes = args.scenes if args.scenes else list(SDD_SCENES)

    if args.discover_only:
        missing = 0
        for scene in scenes:
            ckpts = resolve_variant_checkpoints(args, scene)
            for variant in args.variants:
                path = ckpts[variant]
                if path is None:
                    missing += 1
                print(f"[viz-loop] {scene:12s} {variant:14s} -> {path if path else 'MISSING'}")
        print(f"[viz-loop] discover: {missing} missing checkpoint(s) in {len(scenes)} scene(s)")
        return 0 if missing == 0 else 1

    total = 0
    for scene in scenes:
        ckpts = resolve_variant_checkpoints(args, scene)
        if not any(ckpts[v] for v in args.variants):
            continue
        targets = _select_targets(scene, args)
        if not targets:
            continue
        for variant in args.variants:
            path = ckpts[variant]
            if path is None:
                continue
            total += run_scene_variant(scene, variant, path, args, targets)
    print(f"[viz-loop] TOTAL: {total} figures written to {args.out_dir}")
    return total


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scene", dest="scenes", action="append", choices=SDD_SCENES,
                   default=None, help="held-out SDD scene (repeatable; default: all)")
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS),
                   help="which model variants to render (default: all three)")
    p.add_argument("--variant-ckpt", dest="variant_ckpt", action="append", default=None,
                   metavar="VARIANT=PATH",
                   help="explicit checkpoint per variant (repeatable), e.g. "
                        "no_video=/ckpt_a.pt global_video=/ckpt_b.pt")
    p.add_argument("--checkpoint-dir", default=None,
                   help="base results dir for run-tag discovery (results_sdd/cor_fm)")
    p.add_argument("--discover-only", action="store_true",
                   help="print the resolved scene/variant/checkpoint plan and exit "
                        "(no model/data loading)")
    p.add_argument("--num-samples", type=int, default=1,
                   help="random windows rendered per scene (default: 1)")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for deterministic random sampling (default: 42)")
    p.add_argument("--ped-id", "--agent-id", dest="ped_id", type=int,
                   default=None,
                   help="only render this pedestrian (track id); overrides "
                        "random sampling")
    p.add_argument("--frame-id", "--window-idx", dest="frame_id", type=int,
                   default=None,
                   help="only render the window anchored at this frame number; "
                        "overrides random sampling")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--sdd-root", default=None)
    p.add_argument("--video-features-root", default=None,
                   help="required for the global_video variant (dir of per-scene .npy)")
    p.add_argument("--device", default=None)
    p.add_argument("--sampling-steps", type=int, default=10)
    p.add_argument("--use-ema", action="store_true")
    p.add_argument("--agent-encoder", default="auto",
                   choices=["auto", "compact", "resnet18"])
    p.add_argument("--crop-size", type=int, default=64)
    p.add_argument("--padding", type=int, default=0)
    p.add_argument("--drop-lost", action="store_true",
                   help="strict opt-in: crop only windows without 'lost' frames "
                        "(default keeps them)")
    p.add_argument("--no-background", action="store_true",
                   help="skip frame lookup; render over a white placeholder")
    p.add_argument("--background", default=None,
                   help="explicit background image (overrides frame lookup)")
    p.add_argument("--zoom-margin", type=float, default=90.0)
    p.add_argument("--max-agents", type=int, default=-1,
                   help="cap on windows rendered per scene-variant "
                        "(-1 = all pedestrians)")
    p.add_argument("--homography", default=None,
                   help="optional path to a (3,3) homography .npy/.txt applied to "
                        "trajectories before overlay (identity when omitted)")
    p.add_argument("--demo", action="store_true",
                   help="checkpoint-free styling self-check via visualize_trajectories")
    args = p.parse_args(argv)

    parsed_ckpts: dict[str, Path] = {}
    for spec in args.variant_ckpt or []:
        variant, _, path = spec.partition("=")
        if variant not in VARIANTS or not path:
            raise SystemExit(f"--variant-ckpt must be VARIANT=PATH, got {spec!r}")
        parsed_ckpts[variant] = Path(path)
    args.variant_ckpt = parsed_ckpts

    if args.homography is not None:
        path = Path(args.homography)
        if path.suffix == ".npy":
            H = np.load(path)
        else:
            H = np.loadtxt(path)
        args.homography = np.asarray(H, dtype=np.float64)
        if args.homography.shape != (3, 3):
            raise SystemExit(f"homography must be (3, 3), got {args.homography.shape}")
    return args


def run_demo(out_dir: str | Path) -> int:
    """Render the three synthetic variant figures (no checkpoint required)."""
    import visualize_trajectories as vis

    saved = vis.run_demo(out_dir)
    print(f"[viz-loop] demo wrote {len(saved)} figures to {out_dir}")
    return len(saved)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.demo:
        run_demo(args.out_dir)
        return
    if args.variants and not args.variant_ckpt and args.checkpoint_dir is None:
        raise SystemExit(
            "set --variant-ckpt VARIANT=PATH per variant or --checkpoint-dir "
            "for run-tag discovery (or use --demo)."
        )
    total = run_pipeline(args)
    print(f"[viz-loop] finished: {total} figures")


if __name__ == "__main__":
    main()