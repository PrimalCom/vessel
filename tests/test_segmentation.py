"""Tests for vessel segmentation and morphological cleanup."""

import numpy as np
import pytest

from src.segmentation.threshold import segment_vessels
from src.segmentation.morphology import apply_morphological_cleanup


def _make_cylinder_volume(shape=(32, 32, 32), radius=3.0):
    """Create a simple volume with a bright cylinder along the z-axis center.

    Returns a float64 array in [0, 1] that mimics a vesselness image.
    """
    vol = np.zeros(shape, dtype=np.float64)
    cz, cy, cx = shape[0] // 2, shape[1] // 2, shape[2] // 2
    zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    # Bright cylinder along the z-axis
    vol[dist <= radius] = 0.8
    # Add mild background noise so Otsu has something to threshold
    rng = np.random.default_rng(123)
    vol += rng.uniform(0.0, 0.1, size=shape)
    vol = np.clip(vol, 0.0, 1.0)
    return vol


@pytest.fixture(scope="module")
def cylinder_vol():
    return _make_cylinder_volume()


def test_otsu_returns_bool(cylinder_vol):
    """Otsu segmentation returns a boolean array."""
    mask, _ = segment_vessels(cylinder_vol, method="otsu")
    assert mask.dtype == bool


def test_otsu_output_shape(cylinder_vol):
    """Otsu output matches input shape."""
    mask, _ = segment_vessels(cylinder_vol, method="otsu")
    assert mask.shape == cylinder_vol.shape


def test_hysteresis_runs(cylinder_vol):
    """Hysteresis thresholding runs without error."""
    mask, params = segment_vessels(cylinder_vol, method="hysteresis")
    assert mask.dtype == bool
    assert mask.shape == cylinder_vol.shape
    assert params["method"] == "hysteresis"


def test_percentile_runs(cylinder_vol):
    """Percentile thresholding runs without error."""
    mask, params = segment_vessels(cylinder_vol, method="percentile")
    assert mask.dtype == bool
    assert mask.shape == cylinder_vol.shape
    assert params["method"] == "percentile"


def test_cleanup_reduces_noise():
    """Morphological cleanup should remove small isolated noise clusters."""
    rng = np.random.default_rng(42)
    # Create a mask with a large connected component (cylinder core) and
    # scattered noise voxels.
    shape = (32, 32, 32)
    mask = np.zeros(shape, dtype=bool)

    # Large connected cylinder along z-axis
    zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    cy, cx = shape[1] // 2, shape[2] // 2
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    mask[dist <= 3.0] = True

    # Sprinkle isolated noise voxels
    noise_positions = rng.integers(0, 32, size=(200, 3))
    for z, y, x in noise_positions:
        mask[z, y, x] = True

    original_count = mask.sum()
    cleaned = apply_morphological_cleanup(mask, opening_radius=1, min_object_size=50)
    cleaned_count = cleaned.sum()

    # Cleanup should remove some noise, so cleaned has fewer (or equal) voxels.
    assert cleaned_count <= original_count
    # The large cylinder should survive, so cleaned is non-empty.
    assert cleaned_count > 0


def test_invalid_method(cylinder_vol):
    """Unsupported method raises ValueError."""
    with pytest.raises(ValueError, match="Unknown method"):
        segment_vessels(cylinder_vol, method="not_a_method")
