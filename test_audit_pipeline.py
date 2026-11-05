import numpy as np
import cv2
import json

from audit_pipeline_alignment import (
    homogeneous_transform,
    world_to_pixel_audit,
    motion_lift_score,
    grid_search_projection,
    run_audit,
    synthetic_homography,
)


def test_homogeneous_transform_shape():
    H = synthetic_homography()
    pts = np.array([[0, 0], [1, 2], [3, 4]])
    out = homogeneous_transform(H, pts)
    assert out.shape == (3, 2)


def test_world_to_pixel_audit_matches_direct():
    H = synthetic_homography()
    world = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = world_to_pixel_audit(world, H, 'eth', orig_res=(640, 480), target_res=(640, 480), use_inv=True)
    direct = homogeneous_transform(np.linalg.inv(H), world)
    np.testing.assert_allclose(out, direct, atol=1e-6)


def test_univ_convention_applies_offsets():
    H = synthetic_homography()
    world = np.array([[1.0, 2.0]])
    out = world_to_pixel_audit(world, H, 'univ', orig_res=(720, 576), target_res=(720, 576), use_inv=True)
    base = homogeneous_transform(np.linalg.inv(H), world)
    assert not np.allclose(out, base)


def test_motion_lift_score_range():
    frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)]
    frames[1][40:60, 40:60] = 255
    res = motion_lift_score(frames, np.array([[50, 50]]))
    assert -1.0 <= res['lift'] <= 1.0
    assert 0.0 <= res['hit_rate'] <= 1.0


def test_grid_search_finds_identity_mapping():
    H = synthetic_homography()
    world = np.array([[0, 0], [1, 2], [3, 4]])
    pix = world_to_pixel_audit(world, H, 'eth', (640, 480), (640, 480), True)
    grid = grid_search_projection(world, pix, H, 'eth', (640, 480), (640, 480))
    assert len(grid) >= 1
    assert grid[0]['mean_err_px'] < 1e-6


def test_run_audit_e2e_structure():
    report = run_audit()
    assert 'preprocessing_audit' in report
    assert 'visualization_grid_search' in report
    assert 'motion_lift' in report
    assert isinstance(report['visualization_grid_search'], list)
    json.dumps(report)
