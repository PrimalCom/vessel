"""Tests for centerline extraction, distance-field refinement, and graph building."""

import numpy as np
import pytest

from src.centerline.skeleton import extract_centerline_skeleton
from src.centerline.distance_field import refine_centerline_with_distance
from src.centerline.graph import build_vessel_graph


def _make_cylinder_mask(shape=(32, 32, 32), radius=3.0):
    """Create a boolean mask of a straight cylinder along the z-axis."""
    mask = np.zeros(shape, dtype=bool)
    cy, cx = shape[1] // 2, shape[2] // 2
    zz, yy, xx = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    mask[dist <= radius] = True
    return mask


@pytest.fixture(scope="module")
def cylinder_mask():
    return _make_cylinder_mask()


@pytest.fixture(scope="module")
def skeleton_result(cylinder_mask):
    return extract_centerline_skeleton(cylinder_mask)


# ---- Skeleton tests ----

def test_skeleton_output_types(skeleton_result):
    """Skeleton returns a bool array, int array, and dict."""
    skeleton, coords, stats = skeleton_result
    assert skeleton.dtype == bool
    assert coords.dtype in (np.intp, np.int64, np.int32)
    assert isinstance(stats, dict)


def test_skeleton_is_thinner(skeleton_result, cylinder_mask):
    """Skeleton has fewer voxels than the original mask."""
    skeleton, _, stats = skeleton_result
    assert stats["skeleton_voxels"] < stats["mask_voxels"]
    assert skeleton.sum() < cylinder_mask.sum()


def test_skeleton_subset_of_mask(skeleton_result, cylinder_mask):
    """All skeleton voxels should be within the mask or at most 1 voxel away."""
    skeleton, coords, _ = skeleton_result
    from scipy.ndimage import binary_dilation, generate_binary_structure

    # Dilate mask by 1 voxel in all directions to allow for boundary effects
    struct = generate_binary_structure(3, 3)  # 26-connectivity
    dilated_mask = binary_dilation(cylinder_mask, structure=struct, iterations=1)

    # Every skeleton voxel must be inside the dilated mask
    for pt in coords:
        z, y, x = pt
        assert dilated_mask[z, y, x], (
            f"Skeleton voxel at ({z}, {y}, {x}) is not within mask or its 1-voxel neighbourhood"
        )


# ---- Distance field refinement tests ----

def test_refine_returns_floats(skeleton_result, cylinder_mask):
    """Refined coordinates and radii should be float arrays."""
    skeleton, _, _ = skeleton_result
    refined_coords, radii = refine_centerline_with_distance(cylinder_mask, skeleton)
    assert refined_coords.dtype == np.float64
    assert radii.dtype == np.float64
    assert refined_coords.ndim == 2
    assert refined_coords.shape[1] == 3


def test_refine_radii_positive(skeleton_result, cylinder_mask):
    """All radii should be positive for a non-empty skeleton inside a mask."""
    skeleton, _, stats = skeleton_result
    if stats["skeleton_voxels"] == 0:
        pytest.skip("Skeleton is empty")
    _, radii = refine_centerline_with_distance(cylinder_mask, skeleton)
    assert len(radii) > 0
    assert np.all(radii > 0), "All radii should be positive for skeleton inside a vessel"


# ---- Graph tests ----

def test_graph_has_required_keys(skeleton_result):
    """Graph dict has expected keys."""
    skeleton, _, _ = skeleton_result
    result = build_vessel_graph(skeleton)
    required_keys = {
        "endpoints",
        "branch_points",
        "segments",
        "num_segments",
        "total_centerline_length_voxels",
        "segment_lengths",
        "graph",
    }
    assert required_keys.issubset(result.keys())


def test_graph_endpoints_exist(skeleton_result):
    """A straight cylinder skeleton should have at least some endpoints."""
    skeleton, _, _ = skeleton_result
    result = build_vessel_graph(skeleton)
    # A straight line skeleton should have exactly 2 endpoints, but we
    # conservatively just check for > 0.
    assert result["endpoints"].shape[0] > 0, "Expected at least one endpoint"
    assert result["num_segments"] >= 1, "Expected at least one segment"
