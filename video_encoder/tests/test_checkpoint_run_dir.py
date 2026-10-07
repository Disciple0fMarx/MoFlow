"""Regression tests for the checkpoint run-dir path rule in
``tools/visualize_trajectory_comparison.py``.

The render tool must locate the same run dirs the research scripts create:
``_SDD_ho<scene>_vmfull`` / ``_SDD_ho<scene>_novid``, with an optional
``_vid<id>`` appended to BOTH variants when a ``--video-id`` filter is active.
"""

from __future__ import annotations

from tools.visualize_trajectory_comparison import _checkpoint_path


def test_novariant_geographic_loso() -> None:
    assert str(_checkpoint_path("bookstore", "novid", None)).endswith(
        "results_sdd/cor_fm/_SDD_hobookstore_novid/models/checkpoint_best.pt"
    )


def test_vmfull_geographic_loso() -> None:
    assert str(_checkpoint_path("bookstore", "vid", None)).endswith(
        "results_sdd/cor_fm/_SDD_hobookstore_vmfull/models/checkpoint_best.pt"
    )


def test_vmfull_video_id_suffix() -> None:
    """Angle 2 (single-video): _vid<id> must mirror vid_out_suffix."""
    assert str(_checkpoint_path("deathCircle", "vid", "video0")).endswith(
        "results_sdd/cor_fm/_SDD_hodeathCircle_vmfull_vidvideo0/models/checkpoint_best.pt"
    )


def test_novid_video_id_suffix() -> None:
    """The trajectory-only arm is also retrained per video in angle 2."""
    assert str(_checkpoint_path("quad", "novid", "video0")).endswith(
        "results_sdd/cor_fm/_SDD_hoquad_novid_vidvideo0/models/checkpoint_best.pt"
    )


def test_video_id_cannot_be_empty_string() -> None:
    # an empty --video-id must behave like "no filter" (never append a bare '_vid')
    assert str(_checkpoint_path("coupa", "vid", "")).endswith(
        "results_sdd/cor_fm/_SDD_hocoupa_vmfull/models/checkpoint_best.pt"
    )