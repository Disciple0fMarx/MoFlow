"""Audit GV-feature *perspective scope* — the frame-id collision bug.

Background
----------
SDD ``frame_id`` values are **0-based per video**. The global-video feature
cache is built **per scene** and keyed on ``frame_id`` only (see the
``drop_duplicates("frame_id")`` step in
:meth:`video_encoder.global_video_encoder_sdd.SDDGlobalVideoEncoder.encode_split_to_cache`
and the ``frame_id``-only key in :class:`video_encoder.sdd_adapter.SDDFrameFeatureLookup`).

If a scene has several *videos* whose frame-id ranges overlap (which is
structural in SDD — every video starts at 0), then a window from the *later
sorted* video resolves its ``frame_id`` to the **first video's** feature rows
even though the pixels came from a different camera. The LOSO train set mixes
videos within each scene, so most windows are conditioned on *wrong-perspective*
features whenever the manifest is not scoped by ``video_id``.

What this tool computes (no video decoding, annotations + manifests only)
-------------------------------------------------------------------------
For every sliding window in the dataloader's flat index:

* **old** — resolution under the pre-fix semantics: manifest keyed by
  ``frame_id`` only, first-video-wins on collisions (exactly what the current
  shipped encoder produces).
* **new** — resolution under the fixed semantics: manifest keyed by
  ``(video_id, frame_id)``, so a window always resolves inside *its own*
  video's rows.

Then per (scene, video) it counts the windows whose 8 past frames would have
resolved to **a different video** under ``old`` vs ``new`` — i.e. how many
training/eval windows are (were) conditioned on wrong-camera features.

Caveat: on a dev checkout with no SDD caches present, this runs in
``--no-manifest`` mode which computes the collision **as if** a cache existed
(annotation frame-id rows directly). Point ``--features-root`` at a real SDD
feature cache to compare against actual served manifests.

Example::

    python tools/audit_sdd_perspective.py \
        --sdd-root /home/efrei_stage/Desktop/Datasets/SDD \
        --features-root ./features/resnet18_sdd \
        --out report/audit_sdd_perspective
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

import pandas as pd

from data.dataloader_sdd_global import SDD_PAST_FRAMES, build_window_index
from video_encoder.sdd_adapter import SDD_SCENES, expand_sdd_root
from video_encoder.global_video_encoder_sdd import build_sdd_frame_index


@dataclass
class _ScopeStats:
    n_windows: int = 0
    n_wrong_old: int = 0
    n_wrong_new: int = 0

    @property
    def frac_wrong_old(self) -> float:
        return self.n_wrong_old / self.n_windows if self.n_windows else 0.0

    @property
    def frac_wrong_new(self) -> float:
        return self.n_wrong_new / self.n_windows if self.n_windows else 0.0


def _resolve_bits(
    rows_map: dict[tuple[str, int], str],  # (video_id, frame_id) -> owning video
    window_video: str,
    past_frame_ids: Sequence[int],
) -> list[bool]:
    """Per-frame 'resolved to a different video than the window's own' flags."""
    bad = []
    for fid in past_frame_ids:
        owner = rows_map.get((window_video, int(fid)))
        if owner is None:
            # frame never indexed for this video — nearest-snap semantics;
            # flag as ambiguous (conservative), same as the old lookup.
            bad.append(True)
        else:
            bad.append(owner != window_video)
    return bad


def scope_stats_for_scene(
    scene: str,
    frame_df: pd.DataFrame,
    videos: Sequence[str],
    features_root: Path | None,
    root: Path,
) -> pd.DataFrame:
    """Per-(scene, video) wrong-camera window counts under old and new cache semantics.

    ``videos`` order matters for the *old* (first-wins) semantics — the encoder's
    ``sort_values(["video_id", "frame_id"])`` makes the first sorted video the
    collision winner.
    """
    sub = frame_df[frame_df["scene"] == scene]
    ann_rows: dict[tuple[str, int], str] = {
        (str(v), int(f)): str(v)
        for v, f in zip(sub["video_id"].astype(str), sub["frame_id"].astype(int))
    }
    # only rows that the INDEX actually uses (contiguous runs of SEQ_LEN=20)
    index = build_window_index(root, scenes=[scene])

    def _manifest_rows() -> dict[tuple[str, int], str]:
        """Rows as the encoder would write them: new=per-(video,frame), old=frame-only first-wins."""
        man_path = (features_root or Path("")) / f"{scene}.manifest.parquet"
        new_rows: dict[tuple[str, int], str] = {}
        if features_root is not None and man_path.exists():
            man = pd.read_parquet(man_path)
            for v, f in zip(man["video_id"].astype(str), man["frame_id"].astype(int)):
                new_rows[(v, int(f))] = v
        return new_rows if new_rows else ann_rows

    man_rows = _manifest_rows()
    # old semantics: key by frame_id, first sorted video wins (encoder sort order).
    old_winner: dict[int, str] = {}
    for (v, f), _ in sorted(man_rows.items(), key=lambda kv: kv[0]):
        old_winner.setdefault(f, v)

    def _in_old(vid: str, fid: int) -> bool:
        # Correct only when the frame that won the collision belongs to `vid`.
        owner = old_winner.get(int(fid))
        bit = owner is None or owner != vid
        return bit

    rows: list[dict] = []
    per_video: dict[str, _ScopeStats] = {}
    for v in videos:
        per_video[str(v)] = _ScopeStats()

    for i in range(len(index)):
        vid = str(index.video_id(i))
        past = [int(f) for f in index.past_frame_ids(i)]
        if vid not in per_video:
            continue
        stats = per_video[vid]
        stats.n_windows += 1
        # old: frame-id-only keying -> compare resolved owner vs window video
        stats.n_wrong_old += sum(_in_old(vid, f) for f in past)
        # new: (video_id, frame_id) keying
        stats.n_wrong_new += sum(_resolve_bits(man_rows, vid, past))

    for v, s in per_video.items():
        rows.append(
            {
                "scene": scene,
                "video_id": v,
                "n_windows": s.n_windows,
                "n_frames_wrong_old": s.n_wrong_old,
                "n_frames_wrong_new": s.n_wrong_new,
                "pct_frames_wrong_old": round(100.0 * s.frac_wrong_old, 2),
                "pct_frames_wrong_new": round(100.0 * s.frac_wrong_new, 2),
            }
        )
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="SDD GV-feature perspective-scope audit.")
    p.add_argument("--sdd-root", default=None)
    p.add_argument("--features-root", default=None,
                   help="Directory with <scene>.npy + <scene>.manifest.parquet (default: none -> annotation-derived).")
    p.add_argument("--out", default=None)
    p.add_argument("--scenes", nargs="*", default=list(SDD_SCENES))
    args = p.parse_args(argv)

    root = expand_sdd_root(args.sdd_root)
    feat = Path(args.features_root) if args.features_root else None
    print(f"[audit-sdd-perspective] root={root} features_root={feat or '(none, annotation-derived)'}")
    if not (root / "annotations").is_dir():
        raise SystemExit(f"FATAL: no annotations dir under {root}.")

    frame_df = build_sdd_frame_index(sdd_root=root, scenes=list(args.scenes))
    if not len(frame_df):
        raise SystemExit("No annotation rows parsed — nothing to audit.")

    all_rows: list[pd.DataFrame] = []
    for scene in args.scenes:
        videos = sorted(frame_df[frame_df["scene"] == scene]["video_id"].astype(str).unique().tolist())
        df = scope_stats_for_scene(scene, frame_df, videos, feat, root)
        all_rows.append(df)
        print(f"\n--- scene={scene}: {len(videos)} video(s) ---")
        if not df.empty:
            print(df.to_string(index=False))
        else:
            print("  (no windows)")

    merged = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    if not merged.empty:
        tot_old = int(merged["n_frames_wrong_old"].sum())
        tot_new = int(merged["n_frames_wrong_new"].sum())
        tot_win = int((merged["n_windows"] * SDD_PAST_FRAMES).sum())
        print("\n=== Aggregate (window-frames) ===")
        print(f"  total window-frames              : {tot_win}")
        print(f"  frames resolving to WRONG video  : old={tot_old} ({100.0*tot_old/max(tot_win,1):.2f}%)  new={tot_new} ({100.0*tot_new/max(tot_win,1):.2f}%)")

    if args.out:
        stem = Path(args.out)
        stem.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(stem.with_suffix(".csv"), index=False)
        (stem.with_suffix(".json")).write_text(
            json.dumps(
                {
                    "rows": merged.to_dict(orient="records"),
                    "n_window_frames": tot_win,
                    "n_frames_wrong_old": tot_old,
                    "n_frames_wrong_new": tot_new,
                },
                indent=2,
                default=str,
            )
        )
        print(f"\nWrote {stem}.csv, {stem}.json")


if __name__ == "__main__":
    main()