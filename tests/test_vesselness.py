"""Tests for vesselness enhancement filters."""

import numpy as np
import pytest

from src.data.phantom import create_vessel_phantom
from src.vesselness.filters import compute_vesselness

SHAPE = (32, 32, 32)
SIGMAS = [1.0, 2.0]


@pytest.fixture(scope="module")
def phantom():
    return create_vessel_phantom(shape=SHAPE, num_main_vessels=2, num_branches=1, seed=42)


@pytest.fixture(scope="module")
def frangi_result(phantom):
    return compute_vesselness(phantom["volume"], method="frangi", sigmas=SIGMAS)


def test_frangi_output_shape(frangi_result):
    """Frangi output matches input shape."""
    vesselness, params = frangi_result
    assert vesselness.shape == SHAPE
    assert params["output_shape"] == SHAPE


def test_frangi_output_range(frangi_result):
    """Frangi output is in [0, 1]."""
    vesselness, _ = frangi_result
    assert vesselness.min() >= 0.0
    assert vesselness.max() <= 1.0


def test_sato_runs(phantom):
    """Sato filter runs without error and returns correct shape."""
    vesselness, params = compute_vesselness(phantom["volume"], method="sato", sigmas=SIGMAS)
    assert vesselness.shape == SHAPE
    assert params["method"] == "sato"


def test_meijering_runs(phantom):
    """Meijering filter runs without error and returns correct shape."""
    vesselness, params = compute_vesselness(phantom["volume"], method="meijering", sigmas=SIGMAS)
    assert vesselness.shape == SHAPE
    assert params["method"] == "meijering"


def test_vesselness_enhances_vessels(phantom, frangi_result):
    """Mean vesselness inside vessel mask should be greater than outside."""
    vesselness, _ = frangi_result
    mask = phantom["ground_truth_mask"]

    # Only test if there are both vessel and background voxels
    assert mask.sum() > 0
    assert (~mask).sum() > 0

    mean_inside = vesselness[mask].mean()
    mean_outside = vesselness[~mask].mean()
    assert mean_inside > mean_outside, (
        f"Expected vesselness inside vessels ({mean_inside:.4f}) to exceed "
        f"outside ({mean_outside:.4f})"
    )


def test_invalid_method(phantom):
    """Unsupported method raises ValueError."""
    with pytest.raises(ValueError, match="Unknown method"):
        compute_vesselness(phantom["volume"], method="invalid_filter")
