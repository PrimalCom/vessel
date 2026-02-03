#!/usr/bin/env python
"""Full vessel centerline extraction pipeline.

Usage:
    python scripts/run_pipeline.py                           # synthetic phantom demo
    python scripts/run_pipeline.py --input /path/to/data.nii.gz
    python scripts/run_pipeline.py --config config.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so that ``from src...`` works when
# the script is invoked directly (e.g. ``python scripts/run_pipeline.py``).
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import yaml
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Imports from the project source tree
# ---------------------------------------------------------------------------
from src.data.loaders import load_volume
from src.preprocessing.normalize import normalize_intensity
from src.preprocessing.denoise import denoise_volume
from src.vesselness.filters import compute_vesselness
from src.segmentation.threshold import segment_vessels
from src.centerline.skeleton import extract_centerline_skeleton
from src.centerline.distance_field import refine_centerline_with_distance
from src.centerline.graph import build_vessel_graph
from src.visualization.projections import create_summary_figure
from src.visualization.report import save_centerline_data, generate_report


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------
def evaluate_centerline(
    extracted_coords: np.ndarray,
    ground_truth_coords: list[np.ndarray],
    tolerance_voxels: float = 2.0,
    extracted_mask: np.ndarray | None = None,
    gt_mask: np.ndarray | None = None,
) -> dict:
    """Compare extracted centerline coordinates against ground-truth curves.

    Parameters
    ----------
    extracted_coords : np.ndarray
        (N, 3) array of extracted centerline points (z, y, x).
    ground_truth_coords : list[np.ndarray]
        List of (M_i, 3) arrays, one per ground-truth vessel curve.
    tolerance_voxels : float
        Distance threshold (in voxels) for a GT point to be considered
        "covered" by the extraction.
    extracted_mask : np.ndarray or None
        Binary segmentation mask from the pipeline (for Dice computation).
    gt_mask : np.ndarray or None
        Ground-truth binary mask (for Dice computation).

    Returns
    -------
    dict
        ``mean_centerline_distance``, ``overlap_percent``, and optionally
        ``dice_coefficient``.
    """
    # Concatenate all ground-truth centerline points into a single array.
    gt_all = np.concatenate(ground_truth_coords, axis=0)

    if extracted_coords.shape[0] == 0 or gt_all.shape[0] == 0:
        result: dict = {
            "mean_centerline_distance": float("inf"),
            "overlap_percent": 0.0,
        }
        if extracted_mask is not None and gt_mask is not None:
            result["dice_coefficient"] = 0.0
        return result

    # Build a KD-tree on the extracted centerline for fast nearest-neighbour
    # queries from the ground-truth side.
    tree_extracted = cKDTree(extracted_coords)

    # For each GT point, find the distance to the nearest extracted point.
    gt_distances, _ = tree_extracted.query(gt_all)

    mean_distance = float(np.mean(gt_distances))
    overlap_count = int(np.sum(gt_distances <= tolerance_voxels))
    overlap_percent = 100.0 * overlap_count / gt_all.shape[0]

    result = {
        "mean_centerline_distance": mean_distance,
        "overlap_percent": overlap_percent,
    }

    # Dice coefficient on binary masks (if available).
    if extracted_mask is not None and gt_mask is not None:
        ext_bool = extracted_mask.astype(bool)
        gt_bool = gt_mask.astype(bool)
        intersection = np.count_nonzero(ext_bool & gt_bool)
        total = np.count_nonzero(ext_bool) + np.count_nonzero(gt_bool)
        dice = 2.0 * intersection / total if total > 0 else 0.0
        result["dice_coefficient"] = dice

    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Vessel centerline extraction pipeline"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input volume (NIfTI, DICOM dir, or .npy). "
        "Omit for synthetic phantom.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs",
        help="Root directory for output files (default: outputs).",
    )
    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # Step 0: Load configuration
    # ------------------------------------------------------------------
    config_path = Path(args.config)
    if config_path.exists():
        print(f"[step 0] Loading configuration from {config_path}")
        with open(config_path, "r") as fh:
            config = yaml.safe_load(fh)
    else:
        print(f"[step 0] Config file '{config_path}' not found -- using defaults")
        config = {}

    # Provide sensible fallback sections so downstream look-ups never fail.
    config.setdefault("data", {})
    config.setdefault("preprocessing", {"normalize": "minmax", "denoise": {"enabled": False}})
    config.setdefault("vesselness", {})
    config.setdefault("segmentation", {})
    config.setdefault("centerline", {})
    config.setdefault("evaluation", {})
    config.setdefault("output", {})

    output_dir = Path(args.output if args.output else config["output"].get("save_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Step 1: Load volume
    # ------------------------------------------------------------------
    input_path = args.input
    if input_path is None and config["data"].get("source") not in (None, "phantom"):
        input_path = config["data"]["source"]
    if input_path == "phantom":
        input_path = None

    print(f"[step 1] Loading volume (source={input_path or 'synthetic phantom'}) ...")
    volume, metadata = load_volume(input_path)
    print(f"         Volume shape: {volume.shape}  dtype: {volume.dtype}")

    # ------------------------------------------------------------------
    # Step 2: Normalise intensity
    # ------------------------------------------------------------------
    norm_method = config["preprocessing"].get("normalize", "minmax")
    print(f"[step 2] Normalising intensity (method={norm_method}) ...")
    volume = normalize_intensity(volume, method=norm_method)

    # ------------------------------------------------------------------
    # Step 3: Optional denoising
    # ------------------------------------------------------------------
    denoise_cfg = config["preprocessing"].get("denoise", {})
    if denoise_cfg.get("enabled", False):
        dn_method = denoise_cfg.get("method", "gaussian")
        dn_sigma = denoise_cfg.get("sigma", 0.5)
        print(f"[step 3] Denoising (method={dn_method}, sigma={dn_sigma}) ...")
        volume = denoise_volume(volume, method=dn_method, sigma=dn_sigma)
    else:
        print("[step 3] Denoising skipped (disabled in config)")

    # ------------------------------------------------------------------
    # Step 4: Vesselness enhancement
    # ------------------------------------------------------------------
    vess_cfg = config["vesselness"]
    vess_method = vess_cfg.get("method", "frangi")
    sigmas_start = vess_cfg.get("sigmas_start", 1.0)
    sigmas_stop = vess_cfg.get("sigmas_stop", 5.0)
    sigmas_step = vess_cfg.get("sigmas_step", 0.5)
    sigmas = np.arange(sigmas_start, sigmas_stop, sigmas_step)
    black_ridges = vess_cfg.get("black_ridges", False)

    print(f"[step 4] Computing vesselness (method={vess_method}, "
          f"sigmas={sigmas.tolist()}) ...")
    vesselness, vess_params = compute_vesselness(
        volume, method=vess_method, sigmas=sigmas, black_ridges=black_ridges
    )
    print(f"         Vesselness computed in {vess_params['elapsed_seconds']:.2f}s")

    # ------------------------------------------------------------------
    # Step 5: Segmentation
    # ------------------------------------------------------------------
    seg_cfg = config["segmentation"]
    seg_method = seg_cfg.get("method", "otsu")
    threshold_factor = seg_cfg.get("threshold_factor", 1.0)
    min_object_size = seg_cfg.get("min_object_size", 100)

    print(f"[step 5] Segmenting vessels (method={seg_method}) ...")
    vessel_mask, seg_params = segment_vessels(
        vesselness,
        method=seg_method,
        threshold_factor=threshold_factor,
        min_object_size=min_object_size,
    )
    print(f"         Segmented voxels: {seg_params['voxel_count']} "
          f"({100.0 * seg_params['voxel_count'] / seg_params['total_voxels']:.2f}%)")

    # ------------------------------------------------------------------
    # Step 6: Skeleton extraction
    # ------------------------------------------------------------------
    print("[step 6] Extracting centerline skeleton ...")
    skeleton, skel_coords, skel_stats = extract_centerline_skeleton(vessel_mask)
    print(f"         Skeleton voxels: {skel_stats['skeleton_voxels']}  "
          f"(reduction ratio: {skel_stats['reduction_ratio']:.4f})")

    # ------------------------------------------------------------------
    # Step 7: Distance-field refinement
    # ------------------------------------------------------------------
    centerline_cfg = config["centerline"]
    refine = centerline_cfg.get("refine_with_distance", True)

    if refine:
        print("[step 7] Refining centerline with distance transform ...")
        refined_coords, radii = refine_centerline_with_distance(
            vessel_mask, skeleton
        )
        print(f"         Refined {refined_coords.shape[0]} points  "
              f"(mean radius: {radii.mean():.2f} voxels)")
    else:
        print("[step 7] Distance-field refinement skipped")
        refined_coords = skel_coords.astype(np.float64)
        radii = None

    # ------------------------------------------------------------------
    # Step 8: Build vessel graph
    # ------------------------------------------------------------------
    print("[step 8] Building vessel graph ...")
    graph_data = build_vessel_graph(skeleton)
    print(f"         Segments: {graph_data['num_segments']}  "
          f"Endpoints: {graph_data['endpoints'].shape[0]}  "
          f"Branch points: {graph_data['branch_points'].shape[0]}  "
          f"Total length: {graph_data['total_centerline_length_voxels']:.1f} voxels")

    # ------------------------------------------------------------------
    # Step 9: Evaluation (if ground truth is available)
    # ------------------------------------------------------------------
    metrics: dict = {
        "total_centerline_length": graph_data["total_centerline_length_voxels"],
    }

    gt_centerlines = metadata.get("gt_centerlines")
    gt_mask = metadata.get("gt_mask")
    tolerance = config["evaluation"].get("tolerance_voxels", 2.0)

    if gt_centerlines is not None:
        print(f"[step 9] Evaluating against ground truth (tolerance={tolerance} voxels) ...")
        eval_result = evaluate_centerline(
            extracted_coords=refined_coords,
            ground_truth_coords=gt_centerlines,
            tolerance_voxels=tolerance,
            extracted_mask=vessel_mask,
            gt_mask=gt_mask,
        )
        metrics.update({
            "mean_centerline_distance": eval_result["mean_centerline_distance"],
            "overlap_percentage": eval_result["overlap_percent"],
        })
        if "dice_coefficient" in eval_result:
            metrics["dice"] = eval_result["dice_coefficient"]

        print(f"         Mean centerline distance : {eval_result['mean_centerline_distance']:.4f} voxels")
        print(f"         Overlap (within {tolerance} vx)  : {eval_result['overlap_percent']:.2f}%")
        if "dice_coefficient" in eval_result:
            print(f"         Dice coefficient         : {eval_result['dice_coefficient']:.4f}")
    else:
        print("[step 9] No ground truth available -- skipping evaluation")

    # ------------------------------------------------------------------
    # Step 10: Visualisation
    # ------------------------------------------------------------------
    figure_path = output_dir / "figures" / "summary.png"
    if config["output"].get("save_figures", True):
        print(f"[step 10] Creating summary figure -> {figure_path}")
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
    else:
        print("[step 10] Figure saving disabled in config")

    # ------------------------------------------------------------------
    # Step 11: Save centerline data and report
    # ------------------------------------------------------------------
    if config["output"].get("save_centerline_coords", True):
        print(f"[step 11] Saving centerline data -> {output_dir / 'results'}")
        save_centerline_data(
            coords=refined_coords,
            radii=radii,
            graph_data=graph_data,
            save_dir=output_dir,
        )

    if config["output"].get("save_metrics", True):
        print(f"[step 11] Generating report -> {output_dir / 'results'}")
        generate_report(
            metrics=metrics,
            graph_data=graph_data,
            save_dir=output_dir,
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.perf_counter() - pipeline_start
    print()
    print("=" * 60)
    print("  Pipeline complete!")
    print("=" * 60)
    print(f"  Volume shape          : {volume.shape}")
    print(f"  Segmented voxels      : {seg_params['voxel_count']}")
    print(f"  Skeleton voxels       : {skel_stats['skeleton_voxels']}")
    print(f"  Vessel segments       : {graph_data['num_segments']}")
    print(f"  Total centerline len  : {graph_data['total_centerline_length_voxels']:.1f} voxels")
    if "mean_centerline_distance" in metrics:
        print(f"  Mean CL distance      : {metrics['mean_centerline_distance']:.4f} voxels")
    if "overlap_percentage" in metrics:
        print(f"  Overlap               : {metrics['overlap_percentage']:.2f}%")
    if "dice" in metrics:
        print(f"  Dice                  : {metrics['dice']:.4f}")
    print(f"  Elapsed time          : {elapsed:.2f}s")
    print(f"  Output directory      : {output_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
