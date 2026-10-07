"""Regression tests for truncated feature-cache detection.

A raw ``.npy`` feature cache that was cut short by a crash must fail fast with
an actionable ``ValueError`` (pointing at re-encoding) instead of surfacing as
a cryptic ``EOFError``/``OSError`` deep inside training or attribution.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from video_encoder.moflow_adapter import FrameFeatureLookup, feature_dim
from video_encoder.sdd_adapter import SDDFrameFeatureLookup

from .sdd_test_utils import write_synthetic_video_cache


@pytest.mark.parametrize("builder", [SDDFrameFeatureLookup.from_root, FrameFeatureLookup.from_root])
def test_valid_cache_loads(builder) -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        feats_dir = write_synthetic_video_cache(
            Path(td) / "feats", "bookstore", ("video0",), frames_per_video=10
        )
        lookup = builder(feats_dir, "bookstore")
        assert lookup.features.shape[1] == 512


@pytest.mark.parametrize("builder", [SDDFrameFeatureLookup.from_root, FrameFeatureLookup.from_root])
def test_truncated_cache_raises_actionable_error(builder) -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        feats_dir = write_synthetic_video_cache(
            Path(td) / "feats", "bookstore", ("video0",), frames_per_video=10
        )
        npy = feats_dir / "bookstore.npy"
        raw = npy.read_bytes()
        npy.write_bytes(raw[: len(raw) // 2])  # truncate after the header
        with pytest.raises(ValueError, match="[Cc]orrupt feature cache"):
            builder(feats_dir, "bookstore")


def test_feature_dim_raises_on_truncated_cache() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        feats_dir = write_synthetic_video_cache(
            Path(td) / "feats", "quad", ("video0",), frames_per_video=10
        )
        npy = feats_dir / "quad.npy"
        raw = npy.read_bytes()
        npy.write_bytes(raw[: len(raw) // 2])
        with pytest.raises(ValueError, match="[Cc]orrupt feature cache"):
            feature_dim(feats_dir, "quad")