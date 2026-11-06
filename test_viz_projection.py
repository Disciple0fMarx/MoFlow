import numpy as np
from audit_pipeline_alignment import world_to_pixel_audit, synthetic_homography

def test_projection_params_match_audit():
    H = synthetic_homography()
    world = np.array([[0,0],[1,2],[3,4]])
    pix = world_to_pixel_audit(world, H, 'eth', (640,480), (640,480), use_inv=True)
    pix2 = world_to_pixel_audit(world, H, 'eth', (640,480), (320,240), use_inv=True)
    assert np.allclose(pix2[:,0], pix[:,0]*0.5)

def test_scene_convention_ucy_offsets():
    H = synthetic_homography()
    world = np.array([[1.0,2.0]])
    eth = world_to_pixel_audit(world, H, 'eth', (720,576), (720,576), True)
    univ = world_to_pixel_audit(world, H, 'univ', (720,576), (720,576), True)
    assert not np.allclose(eth, univ)
