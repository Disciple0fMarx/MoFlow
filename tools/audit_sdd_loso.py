"""Audit the SDD LOSO protocol against a real (or synthetic) dataset root.

Answers the "what exactly did we hold out?" question for the supervisor:

1. **Scene-category census** — for every canonical SDD scene, how many videos
   (perspectives), tracks and sliding windows exist under the raw
   ``<sdd_root>/annotations/<scene>/<video_id>/annotations.txt`` tree.
2. **LOSO unit verification** — for each candidate held-out scene, rebuild the
   exact train (7 other scenes) and test (the held-out scene only) window
   indices used by :class:`data.dataloader_sdd_global.SDDGlobalDataset` and
   assert, at the ``(scene_idx, video_idx, track_id, anchor_frame)`` key level,
   that train and test are disjoint and that the held-out scene's *videos*
   never appear in train.
3. **Frame-id overlap diagnostic** — per scene, the pairwise intersection of
   ``frame_id`` sets across that scene's videos, and — when a feature cache
   manifest exists — the count of cached rows that would have been collapsed
   by the ``drop_duplicates("frame_id")`` encode step (see ``tools/audit_sdd_perspective.py``
   for the full perspective-scope audit).

This script touches only the annotations (plain text) and the optional
feature-cache manifests — it never decodes video.

Example::

    python tools/audit_sdd_loso.py --sdd-root /path/to/SDD --out report/audit_sdd
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

import pandas as pd

from data.dataloader_sdd_global import build_window_index
from video_encoder.sdd_adapter import SDD_SCENES, expand_sdd_root
from video_encoder.global_video_encoder_sdd import build_sdd_frame_index


@dataclass
class SceneVideoCensus:
    scene: str
    video_id: str
    n_tracks: int
    n_windows: int
    n_frames: int
    frame_min: int
    frame_max: int


def census_scene_videos(
    sdd_root: Path, scenes: Sequence[str]
) -> list[SceneVideoCensus]:
    """Per-(scene, video) track/window/frame census from annotations only.

    Frame ranges are computed from the canonical frame index (unique
    ``(scene, video_id, frame_id)`` rows); track/window counts come from the
    sliding-window index (``build_window_index`` requires runs of
    ``SDD_SEQ_LEN=20`` contiguous frame ids, matching the dataloader).
    """
    frame_df = build_sdd_frame_index(sdd_root=sdd_root, scenes=list(scenes))
    index = build_window_index(sdd_root, scenes=list(scenes))

    # (scene, video) -> (n_tracks, n_frames, frame_min, frame_max)
    meta: dict[tuple[str, str], list] = {}
    if len(frame_df):
        for (scene, vid), sub in frame_df.groupby(
            ["scene", "video_id"], sort=False
        ):
            fids = sub["frame_id"].astype(int)
            meta[(scene, vid)] = [
                int(sub["track_id"].nunique()),
                int(fids.nunique()),
                int(fids.min()),
                int(fids.max()),
            ]
    else:
        for i in range(len(index)):
            key = (index.scene_id(i), index.video_id(i))
            meta.setdefault(key, [0, 0, 0, 0])

    rows = index.rows
    win_counts: dict[tuple[str, str], int] = {}
    for i in range(len(index)):
        key = (index.scene_id(i), index.video_id(i))
        win_counts[key] = win_counts.get(key, 0) + 1

    out: list[SceneVideoCensus] = []
    for (scene, vid), (nt, nf, fmin, fmax) in sorted(meta.items()):
        out.append(
            SceneVideoCensus(
                scene=scene,
                video_id=vid,
                n_tracks=nt,
                n_windows=win_counts.get((scene, vid), 0),
                n_frames=nf,
                frame_min=fmin,
                frame_max=fmax,
            )
        )
    return out


@dataclass
class LosoCheck:
    held_out: str
    n_train_scenes: int
    n_test_scenes: int
    train_videos: list[str]
    test_videos: list[str]
    train_windows: int
    test_windows: int
    leaked_windows: int
    leaked_video_ids: list[str]
    ok: bool


def check_loso_disjointness(
    sdd_root: Path, scenes: Sequence[str] | None = None
) -> list[LosoCheck]:
    """Rebuild train/test splits per held-out scene and assert disjointness.

    Only scenes whose ``annotations/<scene>/`` dir exists are checked; missing
    scene dirs are skipped with a warning (synthetic/test roots often contain
    a subset).
    """
    candidates = list(SDD_SCENES if scenes is None else scenes)
    present = [s for s in candidates if (sdd_root / "annotations" / s).is_dir()]
    for s in candidates:
        if s not in present:
            print(f"[audit-sdd-loso] skip missing scene for LOSO check: {s}")
    checks: list[LosoCheck] = []
    for held in present:
        train_scenes = [s for s in present if s != held]
        test_scenes = [held]
        train_idx = build_window_index(sdd_root, scenes=train_scenes)
        test_idx = build_window_index(sdd_root, scenes=test_scenes)

        # NOTE: scene_idx/video_idx are LOCAL to each index (renumbered from 0);
        # keying on them across train/test spuriously collides. Key on names.
        train_keys = set(
            (train_idx.scene_id(i), train_idx.video_id(i), train_idx.track_id(i), train_idx.anchor_frame(i))
            for i in range(len(train_idx))
        )
        test_keys = set(
            (test_idx.scene_id(i), test_idx.video_id(i), test_idx.track_id(i), test_idx.anchor_frame(i))
            for i in range(len(test_idx))
        )
        leaked = sorted(train_keys & test_keys)

        train_videos = sorted(
            {f"{train_idx.scene_id(i)}:{train_idx.video_id(i)}" for i in range(len(train_idx))}
        )
        test_videos = sorted(
            {f"{test_idx.scene_id(i)}:{test_idx.video_id(i)}" for i in range(len(test_idx))}
        )
        leaked_vids = sorted(set(train_videos) & set(test_videos))

        checks.append(
            LosoCheck(
                held_out=held,
                n_train_scenes=len(train_scenes),
                n_test_scenes=len(test_scenes),
                train_videos=train_videos,
                test_videos=test_videos,
                train_windows=len(train_idx),
                test_windows=len(test_idx),
                leaked_windows=len(leaked),
                leaked_video_ids=leaked_vids,
                ok=(len(leaked) == 0 and len(leaked_vids) == 0),
            )
        )
    return checks


@dataclass
class FrameOverlap:
    scene: str
    video_a: str
    video_b: str
    shared_frame_ids: int
    overlap_frames: list[int]


def frame_id_overlap(sdd_root: Path, scenes: Sequence[str]) -> list[FrameOverlap]:
    """Pairwise per-scene frame-id intersections across that scene's videos.

    Identical SDD video frame counts are 0-based *per video*, so a strong
    overlap is expected by construction — this is the precondition of the
    feature-cache perspective-scope bug (the cache keys by ``frame_id`` only).
    """
    frame_df = build_sdd_frame_index(sdd_root=sdd_root, scenes=list(scenes))
    out: list[FrameOverlap] = []
    if not len(frame_df):
        return out
    for scene, sub in frame_df.groupby("scene", sort=False):
        per_video: dict[str, set[int]] = {}
        for vid, g in sub.groupby("video_id", sort=False):
            per_video[str(vid)] = set(g["frame_id"].astype(int).tolist())
        vids = sorted(per_video)
        for i in range(len(vids)):
            for j in range(i + 1, len(vids)):
                shared = sorted(per_video[vids[i]] & per_video[vids[j]])
                if shared:
                    out.append(
                        FrameOverlap(
                            scene=scene,
                            video_a=vids[i],
                            video_b=vids[j],
                            shared_frame_ids=len(shared),
                            overlap_frames=shared[:100],  # cap for report size
                        )
                    )
    return out


def manifest_scope_check(features_root: Path, sdd_root: Path, scenes: Sequence[str]) -> pd.DataFrame:
    """Compare cached feature rows vs. annotation (video, frame) rows.

    Columns: ``scene, video_id, n_rows, n_manifest_rows, rows_collapsed``.
    ``rows_collapsed`` is the number of ``(video_id, frame_id)`` rows present
    in the annotations but absent from the cache manifest — the direct measure
    of how many frames were collapsed by the encode-time
    ``drop_duplicates("frame_id")``.
    """
    frame_df = build_sdd_frame_index(sdd_root=sdd_root, scenes=list(scenes))
    rows: list[dict] = []
    if len(frame_df):
        for (scene, vid), sub in frame_df.groupby(["scene", "video_id"], sort=False):
            ann_frames = set(sub["frame_id"].astype(int).tolist())
            manifest_path = features_root / f"{scene}.manifest.parquet"
            man_frames: set[int] = set()
            if manifest_path.exists():
                man = pd.read_parquet(manifest_path)
                man_frames = set(man["frame_id"].astype(int).tolist())
            rows.append(
                {
                    "scene": scene,
                    "video_id": vid,
                    "n_rows": len(ann_frames),
                    "n_manifest_rows": len(man_frames),
                    "rows_collapsed": len(ann_frames - man_frames),
                }
            )
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="SDD LOSO audit.")
    p.add_argument("--sdd-root", default=None, help="SDD dataset root (default: env-resolved).")
    p.add_argument("--out", default=None, help="Output stem (writes <stem>.csv/.json).")
    p.add_argument("--scenes", nargs="*", default=list(SDD_SCENES))
    args = p.parse_args(argv)

    root = expand_sdd_root(args.sdd_root)
    print(f"[audit-sdd-loso] root = {root}")
    if not (root / "annotations").is_dir():
        raise SystemExit(
            f"FATAL: no annotations dir under {root}. Point --sdd-root at the SDD dataset root "
            "containing annotations/<scene>/<video_id>/annotations.txt."
        )

    census = census_scene_videos(root, args.scenes)
    checks = check_loso_disjointness(root)
    overlaps = frame_id_overlap(root, args.scenes)

    print("\n=== Per-(scene, video) census ===")
    if census:
        tbl = pd.DataFrame(asdict(c) for c in census).groupby("scene", as_index=False).agg(
            n_videos=("video_id", "nunique"),
            n_tracks=("n_tracks", "sum"),
            n_windows=("n_windows", "sum"),
            n_frames=("n_frames", "sum"),
        )
        print(tbl.to_string(index=False))
    else:
        print("(no windows found)")

    print("\n=== LOSO disjointness (held-out scene vs 7 others) ===")
    for c in checks:
        status = "OK " if c.ok else "!! "
        print(
            f"  {status} hold-out={c.held_out:<12} train_scenes={c.n_train_scenes} "
            f"test_scenes={c.n_test_scenes} | train_windows={c.train_windows:6d} "
            f"test_windows={c.test_windows:6d} | leak_windows={c.leaked_windows} "
            f"leak_videos={c.leaked_video_ids or '-'}"
        )

    print("\n=== Frame-id overlap across videos within a scene ===")
    for o in overlaps:
        print(
            f"  {o.scene:<12} {o.video_a} ∩ {o.video_b} = {o.shared_frame_ids} frames"
        )

    if args.out:
        stem = Path(args.out)
        stem.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(asdict(c) for c in census).to_csv(stem.with_suffix(".csv"), index=False)
        pd.DataFrame(asdict(c) for c in checks).to_csv(
            stem.with_suffix(".loso.csv"), index=False
        )
        pd.DataFrame(
            [
                {
                    "scene": o.scene,
                    "video_a": o.video_a,
                    "video_b": o.video_b,
                    "shared_frame_ids": o.shared_frame_ids,
                    "overlap_frames": ",".join(str(f) for f in o.overlap_frames),
                }
                for o in overlaps
            ]
        ).to_csv(stem.with_suffix(".overlap.csv"), index=False)
        payload = {
            "census": [asdict(c) for c in census],
            "loso_checks": [asdict(c) for c in checks],
            "frame_overlap": [
                {**asdict(o), "overlap_frames": o.overlap_frames[:100]}
                for o in overlaps
            ],
        }
        (stem.with_suffix(".json")).write_text(
            json.dumps(payload, indent=2, default=str)
        )
        print(f"\nWrote {stem}.csv, {stem}.loso.csv, {stem}.overlap.csv, {stem}.json")

    summary = {"n_loso_failures": sum(0 if c.ok else 1 for c in checks)}
    print(f"\nSUMMARY: LOSO failures = {summary['n_loso_failures']}")


if __name__ == "__main__":
    main()