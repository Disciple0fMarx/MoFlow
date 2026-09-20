"""Regression tests for the agent-crop fetcher (``data/agent_crop_window.py``).

Covers the guarantee that **every** trajectory window carries a real, non-black
agent crop:

* ``lost`` frames keep a crop centered on the annotation box the trajectory
  branch already consumes (default ``drop_lost=False``); strict mode is opt-in;
* a group whose raw video is missing falls back to ``reference.jpg``;
* a group with no pixel source at all raises instead of black-filling.

A tiny synthetic SDD mirror (annotations + generated ``.avi``/``reference.jpg``)
is built under a temp dir; no lab dataset is required.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np

from data.agent_crop_sdd import AgentTrack, parse_annotations
from data.agent_crop_window import SDDAgentWindowDataset, _ordinal_for_frames
from data.dataloader_sdd_global import build_window_index

SCENE = "bookstore"
CROP = 16
OBS = 8
VIDEO_SIZE = 96


# ---------------------------------------------------------------------------
# Synthetic mirror builders
# ---------------------------------------------------------------------------

def _track_rows(
    tid: int, centers: list[tuple[float, float]], lost_frames: set[int]
) -> list[str]:
    rows = []
    for f, (cx, cy) in enumerate(centers):
        lost = 1 if f in lost_frames else 0
        rows.append(
            f"{tid} {cx - 5.0:.1f} {cy - 5.0:.1f} {cx + 5.0:.1f} {cy + 5.0:.1f} "
            f"{f} {lost} 0 0 \"pedestrian\""
        )
    return rows


def _write_annotations(video_dir: Path, rows: list[str]) -> None:
    video_dir.mkdir(parents=True, exist_ok=True)
    lines = "\n".join(rows) + "\n"
    (video_dir / "annotations.txt").write_text(lines, encoding="utf-8")


def _write_video(video_dir: Path, markers: dict[int, list[tuple[float, float]]]) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(str(video_dir / "video.avi"), fourcc, 30.0, (VIDEO_SIZE, VIDEO_SIZE))
    for i in range(24):
        img = np.full((VIDEO_SIZE, VIDEO_SIZE, 3), (i * 4 + 30) % 256, dtype=np.uint8)
        for cx, cy in markers.get(i, []):
            cv2.rectangle(
                img, (int(cx) - 3, int(cy) - 3), (int(cx) + 3, int(cy) + 3),
                (255, 255, 255), -1,
            )
        out.write(img)
    out.release()


def _marker_map(*tracks: list[tuple[float, float]]) -> dict[int, list[tuple[float, float]]]:
    markers: dict[int, list[tuple[float, float]]] = {}
    for centers in tracks:
        for f, (cx, cy) in enumerate(centers):
            markers.setdefault(f, []).append((cx, cy))
    return markers


def _build_mirror(
    root: Path, video_id: str, with_video: bool, with_reference: bool
) -> None:
    """Create ``annotations/<scene>/<video_id>/`` (two tracks) + optional pixels."""
    t0 = [(10.0 + 2 * f, 20.0 + f) for f in range(24)]
    t1 = [(70.0 - 2 * f, 60.0 + f) for f in range(24)]
    rows = _track_rows(0, t0, lost_frames={18, 19}) + _track_rows(1, t1, lost_frames=set())
    video_ann = root / "annotations" / SCENE / video_id
    _write_annotations(video_ann, rows)
    if with_video or with_reference:
        video_dir = root / "videos" / SCENE / video_id
        video_dir.mkdir(parents=True, exist_ok=True)
        if with_video:
            _write_video(video_dir, _marker_map(t0, t1))
        if with_reference:
            ref = np.zeros((VIDEO_SIZE, VIDEO_SIZE, 3), dtype=np.uint8)
            ref[:, :] = (0, 0, 255)  # solid red (BGR); PNG keeps it lossless
            cv2.imwrite(str(video_dir / "reference.png"), ref)


def _load_windows(root: Path):
    index = build_window_index(root, [SCENE])
    if len(index) == 0:
        raise AssertionError("synthetic mirror produced no windows")
    return index


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_lost_frame_windows_keep_crops(tmp: Path) -> None:
    """Default (drop_lost=False): lost-frame windows still get real crops.

    Track 0's observed window covers annotation frames 12..19 (frames 18/19 are
    ``lost``) — the same boxes the trajectory branch consumes.
    """
    _build_mirror(tmp, "video0", with_video=True, with_reference=False)
    windows = _load_windows(tmp)

    dset = SDDAgentWindowDataset(
        tmp, windows, crop_size=CROP, obs_frames=OBS, padding=0, drop_lost=False
    )
    assert dset.valid.all(), f"{int((~dset.valid).sum())} window(s) reported invalid"
    crops = dset.crops
    assert crops.shape == (len(windows), OBS, 3, CROP, CROP)
    # A real decode happened: crops vary across time and are never black.
    for i in range(len(windows)):
        assert not bool((crops[i] == 0).all()), f"window {i} is an all-black crop"
    frames_ref = cv2.VideoCapture(str(tmp / "videos" / SCENE / "video0" / "video.avi"))
    assert frames_ref.isOpened()
    frames_ref.release()
    # Each crop is centered on the annotation box: a bright marker sits there.
    for i in range(len(windows)):
        center = crops[i, :, :, CROP // 2, CROP // 2]
        assert float(center.max()) >= 250.0, (
            f"window {i} crop center is not the annotation-box marker"
        )
    # At least one window contained a lost frame (else the test is vacuous).
    ann = parse_annotations(tmp, SCENE, "video0")
    mask = (ann["frame"] >= 12) & (ann["frame"] <= 19)
    assert bool((ann["lost"][mask] > 0).any()), "fixture lost-frame coverage is empty"


def test_drop_lost_strict_opt_in(tmp: Path) -> None:
    """drop_lost=True (strict) invalidates only the lost-frame window."""
    _build_mirror(tmp, "video0", with_video=True, with_reference=False)
    windows = _load_windows(tmp)
    dset = SDDAgentWindowDataset(
        tmp, windows, crop_size=CROP, obs_frames=OBS, padding=0, drop_lost=True
    )
    assert not dset.valid.all()
    # Track 0 carries lost frames 18/19 -> its window must be the invalid one;
    # track 1 has none and must remain valid.
    invalid_tracks = sorted(
        {int(windows.rows["track_id"][i]) for i in range(len(dset)) if not dset.valid[i]}
    )
    assert invalid_tracks == [0]


def test_missing_video_reference_image_fallback(tmp: Path) -> None:
    """A group with annotations but no raw video uses reference.jpg."""
    _build_mirror(tmp, "videoRef", with_video=False, with_reference=True)
    windows = _load_windows(tmp)
    dset = SDDAgentWindowDataset(
        tmp, windows, crop_size=CROP, obs_frames=OBS, padding=0, drop_lost=False
    )
    assert dset.valid.all()
    crops = dset.crops
    # reference.png is solid red (BGR): R channel saturated, no black crop.
    assert bool((crops[:, :, 2, :, :] >= 250).all()), "R channel not from reference.png"
    assert float(crops[:, :, 0, :, :].mean()) < 20.0, "B channel unexpectedly bright"


def test_no_pixel_source_raises(tmp: Path) -> None:
    """Annotations without any pixel source raise instead of black-filling."""
    _build_mirror(tmp, "videoGhost", with_video=False, with_reference=False)
    windows = _load_windows(tmp)
    try:
        SDDAgentWindowDataset(
            tmp, windows, crop_size=CROP, obs_frames=OBS, padding=0, drop_lost=False
        )
    except RuntimeError as exc:
        assert "videoGhost" in str(exc)
        assert "no" in str(exc).lower()
        return
    raise AssertionError("expected RuntimeError for a group with no pixel source")


def test_ordinal_for_frames_lost_gating() -> None:
    """_ordinal_for_frames keeps lost frames by default, gates them strictly."""
    frames = np.arange(4, dtype=np.int32)
    track = AgentTrack(
        track_id=7,
        frames=frames,
        bbox=np.ones((4, 4), dtype=np.float32),
        lost=np.array([0, 1, 0, 1], dtype=np.uint8),
        occluded=np.zeros(4, dtype=np.uint8),
        generated=np.zeros(4, dtype=np.uint8),
        labels=np.full(4, "pedestrian", dtype="U24"),
    )
    want = np.array([1, 2, 3], dtype=np.int32)
    assert _ordinal_for_frames(track, want, drop_lost=False) is not None
    assert _ordinal_for_frames(track, want, drop_lost=True) is None
    assert _ordinal_for_frames(track, np.array([2], dtype=np.int32), drop_lost=True) is not None


# ---------------------------------------------------------------------------
# Runner (pytest optional)
# ---------------------------------------------------------------------------

def main() -> None:
    test_ordinal_for_frames_lost_gating()
    for case in (
        test_lost_frame_windows_keep_crops,
        test_drop_lost_strict_opt_in,
        test_missing_video_reference_image_fallback,
        test_no_pixel_source_raises,
    ):
        with tempfile.TemporaryDirectory() as td:
            case(Path(td))
    print("AGENT_CROP_WINDOW_ALL_OK")


if __name__ == "__main__":
    main()