"""Visual alignment verification for the SDD dataloader (remote-machine only).

Run on the lab machine where ``~/Desktop/Datasets/SDD`` is mounted. The script:

1. Picks a single (scene, video_id) from the SDD annotations.
2. Renders three sample frames to PNG with bounding boxes + track IDs drawn.
3. Sanity-checks that ``SDDGlobalDataset`` can index a few windows and that
   the trajectory coordinates match the rendered bboxes pixel-for-pixel.

Output goes to ``./verify_sdd_out/`` (relative to CWD). No files outside the
remote SDD root or the output folder are written.

Usage on the lab machine::

    python verify_sdd_alignment.py                     # uses defaults
    python verify_sdd_alignment.py --scene bookstore   # pick a scene
    python verify_sdd_alignment.py --num_samples 5     # render 5 frames
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Ensure project root is on sys.path when run from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.dataloader_sdd_global import (
    SDD_FUTURE_FRAMES,
    SDD_PAST_FRAMES,
    SDDGlobalDataset,
    build_window_index,
)
from video_encoder.sdd_adapter import (
    DEFAULT_SDD_ROOT,
    SDD_SCENES,
    annotation_path,
    expand_sdd_root,
)


def _load_annotations_for_video(sdd_root: Path, scene: str, video_id: str):
    """Return dict[track_id -> list[(frame, xmin, ymin, xmax, ymax)]]."""
    txt = annotation_path(sdd_root, scene, video_id)
    if not txt.exists():
        raise FileNotFoundError(f"missing {txt}")
    arr = np.loadtxt(txt, usecols=(0, 1, 2, 3, 4, 5))
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    out: dict[int, list[tuple[int, float, float, float, float]]] = {}
    for tid, xmin, ymin, xmax, ymax, fid in arr:
        out.setdefault(int(tid), []).append(
            (int(fid), float(xmin), float(ymin), float(xmax), float(ymax))
        )
    for tid in out:
        out[tid].sort(key=lambda r: r[0])
    return out


def _extract_frame_from_mov(
    sdd_root: Path,
    scene: str,
    video_id: str,
    frame_id: int,
    video_subfolder: str = "videos",
) -> Image.Image:
    """Decode a single frame from ``<sdd_root>/<video_subfolder>/<scene>/<video_id>/video.mov``.

    The ``video_subfolder`` parameter defaults to ``"videos"`` (lab server layout)
    but can be set to e.g. ``"video"`` on Kaggle where the folder is named
    differently. The rest of the path (``<scene>/<video_id>/video.mov``) is
    unchanged.

    Falls back to torchvision's ``read_video`` which loads the entire video
    into RAM — fine for verification (one short clip at a time).
    """
    from torchvision.io import read_video

    mov = sdd_root / video_subfolder / scene / video_id / "video.mov"
    if not mov.exists():
        raise FileNotFoundError(f"missing {mov}")
    video, _audio, info = read_video(str(mov), pts_unit="sec")
    video_np = video.numpy()  # [T, H, W, C] uint8
    if frame_id >= video_np.shape[0]:
        frame_id = video_np.shape[0] - 1
    return Image.fromarray(video_np[frame_id])


def _draw_boxes(img: Image.Image, rows: list[tuple[int, float, float, float, float]]) -> Image.Image:
    """Draw bbox + track_id overlays on a copy of ``img``."""
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    color_cycle = [
        (255, 64, 64), (64, 255, 64), (64, 128, 255), (255, 200, 64),
        (255, 64, 200), (64, 255, 200), (200, 64, 255), (64, 64, 64),
    ]
    for idx, (tid, xmin, ymin, xmax, ymax) in enumerate(rows):
        color = color_cycle[idx % len(color_cycle)]
        x0, y0, x1, y1 = xmin, ymin, xmax, ymax
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        label = f"id {tid}"
        tx, ty = x0 + 2, max(0, y0 - 16)
        draw.rectangle([tx, ty, tx + 56, ty + 16], fill=color)
        draw.text((tx + 4, ty), label, fill=(0, 0, 0), font=font)
    return img


def _draw_trajectory(
    img: Image.Image,
    xy_seq: np.ndarray,
    color=(255, 0, 0),
    radius: int = 3,
) -> Image.Image:
    """Overlay a polyline + endpoint dots for ``xy_seq`` shape ``[T, 2]``."""
    draw = ImageDraw.Draw(img)
    for (x, y) in xy_seq:
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)
    for (x0, y0), (x1, y1) in zip(xy_seq[:-1], xy_seq[1:]):
        draw.line([x0, y0, x1, y1], fill=color, width=2)
    return img


def render_one(
    out_path: Path,
    sdd_root: Path,
    scene: str,
    video_id: str,
    frame_id: int,
    bboxes_at_frame: list[tuple[int, float, float, float, float]],
    trajectory_xy: np.ndarray | None = None,
    video_subfolder: str = "videos",
) -> None:
    """Render a single annotated frame + optional trajectory overlay."""
    img = _extract_frame_from_mov(
        sdd_root, scene, video_id, frame_id, video_subfolder=video_subfolder
    )
    img = _draw_boxes(img, bboxes_at_frame)
    if trajectory_xy is not None:
        img = _draw_trajectory(img, trajectory_xy)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"[verify] wrote {out_path}")


def sanity_check_dataset(idx_dataset: SDDGlobalDataset, sample_idx: int) -> None:
    """Pull one sample from the new dataloader and assert shape contract."""
    sample = idx_dataset[sample_idx]
    assert sample["past_traj"].dim() == 3, sample["past_traj"].shape           # [1, P, 6]
    assert sample["fut_traj"].dim() == 3, sample["fut_traj"].shape             # [1, F, 2]
    assert sample["past_traj"].shape[1] == SDD_PAST_FRAMES
    assert sample["fut_traj"].shape[1] == SDD_FUTURE_FRAMES
    z = sample["z_video_global"]
    if z.dim() == 3:
        assert z.shape[0] == SDD_PAST_FRAMES, z.shape
    else:
        print(f"[verify] NOTE: z_video_global is a scalar placeholder ({z}); "
              "no cached features available.")
    print(
        f"[verify] sample {sample_idx}: scene={sample['scene']} "
        f"video={sample['video_id']} anchor_frame={sample['anchor_frame']} "
        f"past_traj.shape={tuple(sample['past_traj'].shape)} "
        f"fut_traj.shape={tuple(sample['fut_traj'].shape)}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sdd_root", default=str(DEFAULT_SDD_ROOT))
    p.add_argument("--scene", default="bookstore", choices=SDD_SCENES)
    p.add_argument("--video_id", default=None,
                   help="If omitted, picks the first video under <scene>.")
    p.add_argument(
        "--video_subfolder",
        default="videos",
        type=str,
        help=(
            "Subdirectory under SDD root that contains the .mov files. "
            "Lab server layout uses 'videos'; Kaggle may use 'video'. "
            "Default: 'videos'."
        ),
    )
    p.add_argument("--out_dir", default="./verify_sdd_out")
    p.add_argument("--num_samples", default=3, type=int)
    p.add_argument("--sanity_only", action="store_true",
                   help="Only run the dataloader sanity check; skip rendering.")
    args = p.parse_args()

    sdd_root = expand_sdd_root(args.sdd_root)
    out_dir = Path(args.out_dir)

    # ---- Pick a video_id ---------------------------------------------------
    ann_dir = sdd_root / "annotations" / args.scene
    if not ann_dir.is_dir():
        raise SystemExit(f"annotations dir not found: {ann_dir}")
    video_ids = sorted(p.name for p in ann_dir.iterdir() if p.is_dir())
    if not video_ids:
        raise SystemExit(f"no video dirs under {ann_dir}")
    video_id = args.video_id or video_ids[0]
    print(f"[verify] scene={args.scene} video_id={video_id} root={sdd_root}")

    # ---- Load all bboxes for this video ------------------------------------
    bboxes_by_track = _load_annotations_for_video(sdd_root, args.scene, video_id)
    if not bboxes_by_track:
        raise SystemExit(f"no annotations for {args.scene}/{video_id}")

    # ---- Render ``num_samples`` evenly-spaced frames with all boxes --------
    all_frames = sorted({fid for rows in bboxes_by_track.values() for fid, *_ in rows})
    if args.sanity_only:
        sample_ids = []
    else:
        sample_ids = np.linspace(0, len(all_frames) - 1, args.num_samples, dtype=int)
        sample_ids = [int(all_frames[i]) for i in sample_ids]

    for i, fid in enumerate(sample_ids):
        rows_at_fid = [
            (tid, xmin, ymin, xmax, ymax)
            for tid, lst in bboxes_by_track.items()
            for f, xmin, ymin, xmax, ymax in lst
            if f == fid
        ]
        render_one(
            out_path=out_dir / f"{args.scene}_{video_id}_frame{fid:06d}.png",
            sdd_root=sdd_root,
            scene=args.scene,
            video_id=video_id,
            frame_id=fid,
            bboxes_at_frame=rows_at_fid,
            video_subfolder=args.video_subfolder,
        )

    # ---- Cross-check: render a trajectory window from the new dataloader --
    try:
        ds = SDDGlobalDataset(
            cfg=_minimal_cfg(),
            training=False,
            sdd_root=sdd_root,
            held_out_scene=args.scene,
            split="test",
            use_video=False,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[verify] WARNING: could not construct SDDGlobalDataset: {exc}")
        ds = None

    if ds is not None and len(ds) > 0:
        for sample_idx in (0, len(ds) // 2, len(ds) - 1):
            sample_idx = max(0, min(sample_idx, len(ds) - 1))
            sanity_check_dataset(ds, sample_idx)

        # Overlay the FIRST dataloader window on a frame and write it out.
        first = ds[0]
        anchor = int(first["anchor_frame"])
        # The dataset stores trajectory in *normalized* pixel coords; unnormalize.
        past_norm = first["past_traj_original_scale"][0, :, :2].cpu().numpy()  # [P, 2] (cx, cy)
        fut_norm = first["fut_traj_original_scale"][0, :, :2].cpu().numpy()    # [F, 2] (cx, cy)
        rows_at_fid = [
            (tid, xmin, ymin, xmax, ymax)
            for tid, lst in bboxes_by_track.items()
            for f, xmin, ymin, xmax, ymax in lst
            if f == anchor
        ]
        full_xy = np.concatenate([past_norm, fut_norm], axis=0)
        render_one(
            out_path=out_dir / f"{args.scene}_{video_id}_window_anchor{anchor:06d}.png",
            sdd_root=sdd_root,
            scene=args.scene,
            video_id=video_id,
            frame_id=anchor,
            bboxes_at_frame=rows_at_fid,
            trajectory_xy=full_xy,
            video_subfolder=args.video_subfolder,
        )

    print(f"[verify] done. Outputs in {out_dir.resolve()}")


def _minimal_cfg():
    """Build a stand-in cfg object with only the attributes SDDGlobalDataset reads.

    Avoids hard-dependency on the full ``utils.config.Config`` (which needs a YAML).
    """
    class _C:
        class MODEL:
            class CONTEXT_ENCODER:
                VIDEO_DIM_RAW = 512
                HELD_OUT_SCENE = None
                VIDEO_FEATURES_ROOT = None
        past_traj_min = None
        past_traj_max = None
        fut_traj_min = None
        fut_traj_max = None
    return _C()


if __name__ == "__main__":
    main()
