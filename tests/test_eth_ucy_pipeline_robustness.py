"""Regression tests for the ETH/UCY projection, bounds and video pipeline.

These tests encode the invariants established during the alignment audit:

* one authoritative world->pixel mapping per scene family;
* strict normalization bounds (never a ``[-1, 1]`` fallback);
* raw-metre ground truth that is never unnormalized;
* temporal anchoring at ``start + (P - 1) * annotation_step`` (uniform 10-frame grid);
* agent crops decoded directly from ``video.avi`` (no JPEG frame dumps).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2

import data.dataloader_eth_ucy as dataloader
from data.dataloader_eth_ucy import ETHDataset, seq_collate_eth
from utils.eth_ucy_geometry import (ANNOTATION_STEP, SCENE_RESOLUTIONS,
                                    annotation_step, assert_temporal_alignment,
                                    observed_frame_ids, scene_video_path,
                                    temporal_anchor_frame,
                                    video_index_for_frame, world_to_pixel)
from utils.norm_bounds import (NormBoundsError, bounds_from_cfg,
                               load_norm_bounds_from_run)
from utils.normalization import normalize_min_max, unnormalize_min_max
from utils.video_reader import VideoDecodeError, VideoFrameReader

REPO = Path(__file__).resolve().parents[1]
RAW_DIR = REPO / "data" / "eth_ucy" / "original" / "raw" / "all_data"
H_DIR = REPO / "data" / "eth_ucy" / "homography"
VIDEO_DIR = REPO / "data" / "eth_ucy" / "videos"

RAW_FILES = {
    "eth": "biwi_eth.txt",
    "hotel": "biwi_hotel.txt",
    "univ": "students003.txt",
    "zara1": "crowds_zara01.txt",
    "zara2": "crowds_zara02.txt",
}

SCENE_GROUPS = {
    "eth": ("ETH", "seq_eth"),
    "hotel": ("ETH", "seq_hotel"),
    "univ": ("UCY", "students03"),
    "zara1": ("UCY", "zara01"),
    "zara2": ("UCY", "zara02"),
}

TRAIN_BOUNDS = SimpleNamespace(
    past_traj_min=-10.254,
    past_traj_max=17.144,
    fut_traj_min=-11.598,
    fut_traj_max=10.603,
)


def _require_assets() -> None:
    missing = [p for p in (RAW_DIR, H_DIR, VIDEO_DIR) if not p.is_dir()]
    if missing:
        pytest.skip(f"ETH/UCY assets missing: {missing}")


def _load_raw_world(scene: str) -> np.ndarray:
    return np.loadtxt(RAW_DIR / RAW_FILES[scene])[:, 2:4]


def _load_H(scene: str) -> np.ndarray:
    return np.loadtxt(H_DIR / f"{scene}_H.txt")


def _make_strict_root(base: Path) -> Path:
    root = base / "datasets"
    raw_eth = np.loadtxt(RAW_DIR / RAW_FILES["eth"])
    raw_hotel = np.loadtxt(RAW_DIR / RAW_FILES["hotel"])
    obsmat = {
        "eth": raw_eth,
        "hotel": raw_hotel,
        "univ": raw_eth[:3],
        "zara1": raw_eth[:3],
        "zara2": raw_eth[:3],
    }
    for scene, (group, scene_dir) in SCENE_GROUPS.items():
        scene_path = root / group / scene_dir
        scene_path.mkdir(parents=True, exist_ok=True)
        np.savetxt(scene_path / "H.txt", _load_H(scene))
        data = obsmat[scene]
        table = np.column_stack(
            [
                data[:, 0],
                data[:, 1],
                data[:, 2],
                np.zeros(len(data)),
                data[:, 3],
                np.zeros((len(data), 3)),
            ]
        )
        np.savetxt(scene_path / "obsmat.txt", table, fmt="%.6f")
        video = VIDEO_DIR / f"{scene}.avi"
        if video.is_file():
            os.symlink(video.resolve(), scene_path / "video.avi")
    return root


@pytest.fixture(scope="module")
def strict_root(tmp_path_factory):
    _require_assets()
    return _make_strict_root(tmp_path_factory.mktemp("tu_clausthal"))


def _cfg(**overrides):
    encoder = SimpleNamespace(
        USE_VIDEO=overrides.pop("USE_VIDEO", False),
        USE_AGENT_VIDEO=overrides.pop("USE_AGENT_VIDEO", True),
        AGENT_CROP_SIZE=overrides.pop("AGENT_CROP_SIZE", [8, 8]),
        AGENTS=1,
        VIDEO_MODE="off",
    )
    cfg = SimpleNamespace(
        past_frames=8,
        future_frames=12,
        rotate=False,
        data_norm="min_max",
        agents=1,
        past_traj_min=TRAIN_BOUNDS.past_traj_min,
        past_traj_max=TRAIN_BOUNDS.past_traj_max,
        fut_traj_min=TRAIN_BOUNDS.fut_traj_min,
        fut_traj_max=TRAIN_BOUNDS.fut_traj_max,
        MODEL=SimpleNamespace(CONTEXT_ENCODER=encoder),
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _eth_track(min_len: int = 20) -> np.ndarray:
    data = np.loadtxt(RAW_DIR / RAW_FILES["eth"])
    data = data[np.argsort(data[:, 0])]
    best = None
    for ped_id in np.unique(data[:, 1]):
        track = data[data[:, 1] == ped_id]
        if len(track) >= min_len and (best is None or len(track) > len(best)):
            best = track
    assert best is not None
    return best


def _eth_split():
    track = _eth_track()[:20]
    windows = track[None, :, 2:4].astype(np.float32)
    split = {
        "windows": windows,
        "start_frames": np.asarray([int(track[0, 0])], dtype=np.int64),
        "ped_ids": np.asarray([int(track[0, 1])], dtype=np.int64),
        "scenes": np.asarray(["eth"], dtype=object),
    }
    return {"train": split, "test": split}


def _fake_windows(data_root, held_out, past_frames, future_frames):
    return _eth_split()


def _dataset(strict_root, monkeypatch, **overrides):
    monkeypatch.setattr(dataloader, "load_loso_windows", _fake_windows)
    return ETHDataset(
        _cfg(**overrides),
        training=False,
        data_dir=str(strict_root),
        subset="eth",
    )


@pytest.mark.parametrize("scene", list(SCENE_GROUPS))
def test_world_to_pixel_mapping_stays_inside_image(scene):
    _require_assets()
    world = _load_raw_world(scene)
    H = _load_H(scene)
    px = world_to_pixel(world, H, scene=scene)
    width, height = SCENE_RESOLUTIONS[scene]
    finite = np.isfinite(px).all(axis=1)
    inside = (
        finite
        & (px[:, 0] >= 0)
        & (px[:, 0] < width)
        & (px[:, 1] >= 0)
        & (px[:, 1] < height)
    )
    assert finite.mean() == 1.0
    assert inside.mean() >= 0.95, f"{scene}: only {inside.mean():.3f} in bounds"


@pytest.mark.parametrize("scene", ["eth", "hotel"])
def test_swap_mapping_projects_onto_feet(scene):
    _require_assets()
    world = _load_raw_world(scene)
    H = _load_H(scene)
    width, height = SCENE_RESOLUTIONS[scene]
    hom = (np.linalg.inv(H) @ np.c_[world, np.ones(len(world))].T).T
    expected = np.c_[hom[:, 1] / hom[:, 2], hom[:, 0] / hom[:, 2]]
    px = world_to_pixel(world, H, scene=scene)
    assert np.allclose(px, expected)

    x = np.array([[3.0, 4.0]])
    assert not np.allclose(
        world_to_pixel(x, H, scene=scene),
        world_to_pixel(x, H, scene="univ"),
    )
    assert width > 0 and height > 0


@pytest.mark.parametrize("scene", ["univ", "zara1", "zara2"])
def test_centered_flip_mapping_uses_authoritative_formula(scene):
    _require_assets()
    world = _load_raw_world(scene)
    H = _load_H(scene)
    width, height = SCENE_RESOLUTIONS[scene]
    hom = (np.linalg.inv(H) @ np.c_[world, np.ones(len(world))].T).T
    expected = np.c_[
        hom[:, 0] / hom[:, 2] + width / 2.0, height / 2.0 - hom[:, 1] / hom[:, 2]
    ]
    px = world_to_pixel(world, H, scene=scene)
    assert np.allclose(px, expected)


@pytest.mark.parametrize("scene", ["eth", "hotel", "univ"])
def test_projected_points_land_on_moving_pedestrians(scene):
    _require_assets()
    world_all = np.loadtxt(RAW_DIR / RAW_FILES[scene])
    H = _load_H(scene)
    frames = np.unique(world_all[:, 0].astype(int))
    selected = frames[
        np.linspace(len(frames) // 4, 3 * len(frames) // 4, 3).astype(int)
    ]

    cap = cv2.VideoCapture(str(VIDEO_DIR / f"{scene}.avi"))
    assert cap.isOpened()
    distances = []
    try:
        for frame_id in selected:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_id) - 1))
            ok_prev, prev = cap.read()
            ok_curr, curr = cap.read()
            if not (ok_prev and ok_curr):
                continue
            diff = cv2.absdiff(
                cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY),
            )
            mask = cv2.dilate((diff > 25).astype(np.uint8), np.ones((5, 5), np.uint8))
            ys, xs = np.nonzero(mask)
            if len(xs) < 10:
                continue
            pts = world_all[world_all[:, 0].astype(int) == frame_id][:, 2:4]
            if len(pts) == 0:
                continue
            px = world_to_pixel(pts, H, scene=scene)
            d = np.sqrt(
                (px[:, 0, None] - xs[None, :]) ** 2
                + (px[:, 1, None] - ys[None, :]) ** 2
            ).min(axis=1)
            distances.extend(d.tolist())
    finally:
        cap.release()

    assert distances, f"{scene}: no decodable moving-pixel frames"
    distances = np.asarray(distances)
    assert np.median(distances) < 10.0
    assert (distances < 10.0).mean() >= 0.7


@pytest.mark.parametrize("scene", list(ANNOTATION_STEP))
def test_temporal_anchor_uses_benchmark_10_frame_step(scene):
    step = annotation_step(scene)
    assert step == 10, "SocialGAN benchmark windows advance 10 video frames"
    start = 100
    ids = observed_frame_ids(start, scene, 8)
    assert ids == [start + i * step for i in range(8)]
    assert temporal_anchor_frame(start, scene, 8) == start + 7 * step
    assert assert_temporal_alignment(start, scene, 8, frame_ids=ids) == ids[-1]
    with pytest.raises(AssertionError):
        assert_temporal_alignment(start, scene, 8, frame_ids=ids[:-1] + [ids[-1] + 1])


def test_best_of_k_keeps_ground_truth_raw_and_unnormalizes_predictions():
    from tools.generate_synchronized_comparisons_eth_ucy import best_of_k_eth

    cfg = SimpleNamespace(
        future_frames=12,
        past_traj_min=TRAIN_BOUNDS.past_traj_min,
        past_traj_max=TRAIN_BOUNDS.past_traj_max,
        fut_traj_min=TRAIN_BOUNDS.fut_traj_min,
        fut_traj_max=TRAIN_BOUNDS.fut_traj_max,
    )
    gt = torch.linspace(-2.0, 3.0, 24, dtype=torch.float32).reshape(1, 1, 12, 2)
    pred_norm = normalize_min_max(
        gt, cfg.fut_traj_min, cfg.fut_traj_max, -1, 1
    ).unsqueeze(0)
    ade, best, pred_out, gt_out = best_of_k_eth(pred_norm, gt, cfg)
    expected_pred = unnormalize_min_max(
        pred_norm, cfg.fut_traj_min, cfg.fut_traj_max, -1, 1
    )
    assert ade == pytest.approx(0.0, abs=1e-6)
    assert best == 0
    assert torch.allclose(pred_out, expected_pred[0], atol=1e-6)
    assert torch.allclose(gt_out[0], gt[0], atol=1e-6)
    double_unnorm = unnormalize_min_max(gt, cfg.fut_traj_min, cfg.fut_traj_max, -1, 1)
    assert not torch.allclose(gt_out[0], double_unnorm)


def test_bounds_are_strict_when_missing():
    with pytest.raises(NormBoundsError):
        bounds_from_cfg(SimpleNamespace(past_traj_min=0.0))
    with pytest.raises(NormBoundsError):
        load_norm_bounds_from_run(REPO / "tests" / "__missing_run__")


def test_agent_crops_are_read_from_video_at_observed_frames(strict_root, monkeypatch):
    H = _load_H("eth")
    observed = observed_frame_ids(780, "eth", 8)
    expected_indices = [video_index_for_frame("eth", f) for f in observed]

    class FakeReader:
        size = (640, 480)

        def __init__(self):
            self.indices = []

        def read(self, index, color="rgb"):
            self.indices.append(int(index))
            value = (int(index) % 251) + 1
            return np.full((480, 640, 3), value, dtype=np.uint8)

    reader = FakeReader()
    monkeypatch.setattr(dataloader, "get_video_reader", lambda path: reader)
    monkeypatch.setattr(
        cv2,
        "imread",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("JPEG read")),
    )

    ds = ETHDataset.__new__(ETHDataset)
    ds.use_agent_video = True
    ds.data_dir = str(strict_root)
    ds.subset = "eth"
    ds.current_scene = "eth"
    ds.agent_crop_size = [8, 8]
    ds.scene_H = {"eth": H}
    ds.H = H

    world = _load_raw_world("eth")[0]
    past = torch.zeros(1, 8, 6)
    past[0, :, :2] = torch.tensor(world.astype(np.float32))

    crops = ds._extract_agent_crops(past, start_frame=780, scene="eth")
    assert crops.shape == (1, 8, 3, 8, 8)
    assert reader.indices == expected_indices
    assert crops.abs().sum() > 0
    assert not torch.allclose(crops[0, 0], crops[0, 1])


def test_agent_crops_fail_loudly_without_video_or_start_frame(strict_root):
    H = _load_H("eth")
    ds = ETHDataset.__new__(ETHDataset)
    ds.use_agent_video = True
    ds.data_dir = str(strict_root / "ETH" / "seq_eth")
    ds.subset = "eth"
    ds.current_scene = "eth"
    ds.agent_crop_size = [8, 8]
    ds.scene_H = {"eth": H}
    ds.H = H
    past = torch.zeros(1, 8, 6)

    with pytest.raises(ValueError, match="start_frame is required"):
        ds._extract_agent_crops(past, start_frame=None, scene="eth")

    ds.data_dir = str(strict_root / "ETH")
    with pytest.raises(FileNotFoundError, match="Native video not found"):
        ds._extract_agent_crops(past, start_frame=780, scene="zara1")


def test_video_reader_fails_explicitly(tmp_path):
    with pytest.raises(FileNotFoundError):
        VideoFrameReader(tmp_path / "missing.avi")

    video = VIDEO_DIR / "eth.avi"
    if not video.is_file():
        pytest.skip("eth video missing")
    reader = VideoFrameReader(video)
    with pytest.raises(VideoDecodeError):
        reader.read(10**9, color="rgb")


def test_eval_dataset_crops_without_any_jpeg(strict_root, monkeypatch):
    monkeypatch.setattr(
        cv2,
        "imread",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("JPEG read")),
    )
    dataset = _dataset(strict_root, monkeypatch)
    assert dataset.use_agent_video
    batch = seq_collate_eth([dataset[0]])
    assert "agent_crops" in batch
    crops = batch["agent_crops"]
    assert crops.shape == (1, 1, 8, 3, 8, 8)
    assert torch.isfinite(crops).all()
    assert crops.abs().sum() > 0


def test_eth_ucy_sources_have_no_jpeg_frame_dump_references():
    files = [
        REPO / "data" / "dataloader_eth_ucy.py",
        REPO / "utils" / "eth_ucy_geometry.py",
        REPO / "utils" / "video_reader.py",
        REPO / "utils" / "norm_bounds.py",
        REPO / "tools" / "generate_synchronized_comparisons_eth_ucy.py",
        REPO / "tools" / "gradcam_eth.py",
        REPO / "tools" / "plot_homography_verification.py",
        REPO / "video_encoder" / "scripts" / "build_ethucy_features.py",
        REPO / "video_encoder" / "__init__.py",
        REPO / "video_encoder" / "__main__.py",
    ]
    forbidden = (".jpg", ".jpeg", "visual_data", "frame%06d")
    offenders = {}
    for path in files:
        assert path.is_file(), path
        text = path.read_text(encoding="utf-8")
        hits = [token for token in forbidden if token in text]
        if hits:
            offenders[str(path.relative_to(REPO))] = hits
    assert not offenders, f"Forbidden JPEG/dump references: {offenders}"


def test_scene_video_path_requires_native_avi(strict_root):
    resolved = scene_video_path(strict_root, "eth")
    assert resolved.name == "video.avi"
    assert resolved.is_file()
    with pytest.raises(FileNotFoundError):
        scene_video_path(strict_root / "missing_root", "zara1")
