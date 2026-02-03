"""Tests for the synthetic vessel phantom generator."""

import numpy as np
import pytest

from src.data.phantom import create_vessel_phantom

# Use small volume and few vessels for speed.
SHAPE = (32, 32, 32)
KWARGS = dict(shape=SHAPE, num_main_vessels=2, num_branches=2, seed=99)


@pytest.fixture(scope="module")
def phantom():
    """Create a single phantom shared by all tests in this module."""
    return create_vessel_phantom(**KWARGS)


def test_phantom_shape(phantom):
    """Output volume has correct shape and dtype (float64)."""
    vol = phantom["volume"]
    assert vol.shape == SHAPE
    assert vol.dtype == np.float64


def test_phantom_range(phantom):
    """Volume values are in [0, 1]."""
    vol = phantom["volume"]
    assert vol.min() >= 0.0
    assert vol.max() <= 1.0


def test_phantom_mask(phantom):
    """Ground truth mask is boolean and has nonzero voxels."""
    mask = phantom["ground_truth_mask"]
    assert mask.dtype == bool
    assert mask.shape == SHAPE
    assert mask.sum() > 0, "Mask should contain some vessel voxels"


def test_phantom_centerlines(phantom):
    """Centerlines are a list of arrays each with shape (N, 3)."""
    centerlines = phantom["ground_truth_centerlines"]
    assert isinstance(centerlines, list)
    assert len(centerlines) > 0
    for cl in centerlines:
        assert isinstance(cl, np.ndarray)
        assert cl.ndim == 2
        assert cl.shape[1] == 3


def test_phantom_metadata(phantom):
    """Metadata dictionary has all required keys."""
    meta = phantom["metadata"]
    required_keys = {"shape", "num_vessels", "vessel_voxel_count", "vessel_volume_fraction"}
    assert required_keys.issubset(meta.keys())
    assert meta["shape"] == SHAPE
    assert meta["num_vessels"] == 4  # 2 main + 2 branches
    assert meta["vessel_voxel_count"] > 0
    assert 0.0 < meta["vessel_volume_fraction"] < 1.0


def test_phantom_reproducible():
    """Same seed produces identical output."""
    p1 = create_vessel_phantom(**KWARGS)
    p2 = create_vessel_phantom(**KWARGS)
    np.testing.assert_array_equal(p1["volume"], p2["volume"])
    np.testing.assert_array_equal(p1["ground_truth_mask"], p2["ground_truth_mask"])
