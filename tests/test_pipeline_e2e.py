"""End-to-end pipeline tests for the vessel centerline extraction project."""

import numpy as np
import pytest

from src.data.phantom import create_vessel_phantom
from src.preprocessing.normalize import normalize_intensity
from src.vesselness.filters import compute_vesselness
from src.segmentation.threshold import segment_vessels
from src.centerline.skeleton import extract_centerline_skeleton
from src.centerline.distance_field import refine_centerline_with_distance
from src.centerline.graph import build_vessel_graph

SHAPE = (32, 32, 32)
SIGMAS = [1.0, 2.0]


@pytest.fixture(scope="module")
def pipeline_results():
    """Run the full pipeline once and cache all intermediate results."""
    # 1. Generate phantom
    phantom = create_vessel_phantom(
        shape=SHAPE, num_main_vessels=2, num_branches=2, seed=42
    )
    volume = phantom["volume"]

    # 2. Normalize
    normalized = normalize_intensity(volume, method="minmax")

    # 3. Vesselness
    vesselness, v_params = compute_vesselness(
        normalized, method="frangi", sigmas=SIGMAS
    )

    # 4. Segment
    mask, s_params = segment_vessels(vesselness, method="otsu", min_object_size=20)

    # 5. Skeleton
    skeleton, coords, skel_stats = extract_centerline_skeleton(mask)

    # 6. Refine
    refined_coords, radii = refine_centerline_with_distance(mask, skeleton)

    # 7. Graph
    graph_result = build_vessel_graph(skeleton)

    return {
        "phantom": phantom,
        "normalized": normalized,
        "vesselness": vesselness,
        "mask": mask,
        "skeleton": skeleton,
        "coords": coords,
        "skel_stats": skel_stats,
        "refined_coords": refined_coords,
        "radii": radii,
        "graph": graph_result,
    }


def test_full_pipeline_runs(pipeline_results):
    """The complete pipeline runs on a small phantom without errors."""
    # If we got here, no exception was raised.
    assert pipeline_results["vesselness"].shape == SHAPE
    assert pipeline_results["mask"].dtype == bool
    assert pipeline_results["skeleton"].dtype == bool
    assert pipeline_results["normalized"].dtype == np.float64


def test_pipeline_produces_centerline(pipeline_results):
    """Extracted centerline has at least some voxels."""
    skel_stats = pipeline_results["skel_stats"]
    assert skel_stats["skeleton_voxels"] > 0, "Pipeline should produce a non-empty centerline"
    assert pipeline_results["coords"].shape[0] > 0


def test_pipeline_graph_segments(pipeline_results):
    """Graph has at least 1 segment."""
    graph = pipeline_results["graph"]
    assert graph["num_segments"] >= 1, "Graph should have at least one segment"
    assert len(graph["segments"]) >= 1
    assert graph["total_centerline_length_voxels"] > 0
