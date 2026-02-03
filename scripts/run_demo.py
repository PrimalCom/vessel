#!/usr/bin/env python
"""Quick demo of the vessel centerline extraction pipeline.

Generates a synthetic phantom, runs the full extraction pipeline with
default settings, and prints the path to the summary figure.

Usage:
    python scripts/run_demo.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so that ``from src...`` works when
# the script is invoked directly (e.g. ``python scripts/run_demo.py``).
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.loaders import load_volume
from src.preprocessing.normalize import normalize_intensity
from src.vesselness.filters import compute_vesselness
from src.segmentation.threshold import segment_vessels
from src.centerline.skeleton import extract_centerline_skeleton
from src.centerline.distance_field import refine_centerline_with_distance
from src.centerline.graph import build_vessel_graph
from src.visualization.projections import create_summary_figure
from src.visualization.report import save_centerline_data, generate_report


def main() -> None:
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "figures" / "summary.png"

    t0 = time.perf_counter()

    # 1. Generate synthetic phantom
    print("[demo] Generating synthetic vessel phantom ...")
    volume, metadata = load_volume(None)
    print(f"       Shape: {volume.shape}")

    # 2. Normalise
    print("[demo] Normalising intensity ...")
    volume = normalize_intensity(volume, method="minmax")

    # 3. Vesselness
    sigmas = np.arange(1.0, 5.0, 0.5)
    print(f"[demo] Computing Frangi vesselness (sigmas={sigmas.tolist()}) ...")
    vesselness, vess_params = compute_vesselness(
        volume, method="frangi", sigmas=sigmas, black_ridges=False
    )
    print(f"       Done in {vess_params['elapsed_seconds']:.2f}s")

    # 4. Segmentation
    print("[demo] Segmenting vessels (Otsu) ...")
    vessel_mask, seg_params = segment_vessels(vesselness, method="otsu")
    print(f"       Vessel voxels: {seg_params['voxel_count']}")

    # 5. Skeleton
    print("[demo] Extracting skeleton ...")
    skeleton, skel_coords, skel_stats = extract_centerline_skeleton(vessel_mask)
    print(f"       Skeleton voxels: {skel_stats['skeleton_voxels']}")

    # 6. Distance-field refinement
    print("[demo] Refining with distance transform ...")
    refined_coords, radii = refine_centerline_with_distance(vessel_mask, skeleton)

    # 7. Graph
    print("[demo] Building vessel graph ...")
    graph_data = build_vessel_graph(skeleton)
    print(f"       Segments: {graph_data['num_segments']}  "
          f"Length: {graph_data['total_centerline_length_voxels']:.1f} voxels")

    # 8. Summary figure
    print(f"[demo] Saving summary figure -> {figure_path}")
    fig = create_summary_figure(
        volume=volume,
        vesselness=vesselness,
        vessel_mask=vessel_mask,
        skeleton=skeleton,
        centerline_coords=refined_coords,
        graph_data=graph_data,
        save_path=figure_path,
    )
    import matplotlib
    matplotlib.pyplot.close(fig)

    # 9. Save data and report
    metrics = {
        "total_centerline_length": graph_data["total_centerline_length_voxels"],
    }
    save_centerline_data(
        coords=refined_coords, radii=radii,
        graph_data=graph_data, save_dir=output_dir,
    )
    generate_report(
        metrics=metrics, graph_data=graph_data, save_dir=output_dir,
    )

    elapsed = time.perf_counter() - t0
    print()
    print(f"Demo complete!  ({elapsed:.2f}s)")
    print(f"Summary figure: {figure_path.resolve()}")


if __name__ == "__main__":
    main()
