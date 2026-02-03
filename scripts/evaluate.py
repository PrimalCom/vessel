#!/usr/bin/env python
"""Standalone evaluation of extracted vessel centerlines against ground truth.

Computes distance-based metrics between extracted and ground-truth centerline
coordinates using a KD-tree for efficient nearest-neighbour queries.

Usage:
    python scripts/evaluate.py \\
        --extracted outputs/results/centerline_coords.npy \\
        --ground-truth ground_truth_centerlines.npy \\
        --tolerance 2.0

    # With optional binary masks for Dice computation:
    python scripts/evaluate.py \\
        --extracted outputs/results/centerline_coords.npy \\
        --ground-truth ground_truth_centerlines.npy \\
        --extracted-mask outputs/results/vessel_mask.npy \\
        --gt-mask ground_truth_mask.npy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Core evaluation function
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
        Distance threshold (in voxels) for a GT point to count as "covered".
    extracted_mask : np.ndarray or None
        Binary segmentation mask from the pipeline (for Dice computation).
    gt_mask : np.ndarray or None
        Ground-truth binary mask (for Dice computation).

    Returns
    -------
    dict
        ``mean_centerline_distance`` -- average distance from each GT point
        to its nearest extracted point (in voxels).

        ``overlap_percent`` -- percentage of GT points that have a nearest
        extracted point within *tolerance_voxels*.

        ``dice_coefficient`` -- (only present when both masks are supplied)
        Dice similarity coefficient between the two binary masks.
    """
    # Concatenate all ground-truth curves into a single point cloud.
    gt_all = np.concatenate(ground_truth_coords, axis=0)

    if extracted_coords.shape[0] == 0 or gt_all.shape[0] == 0:
        result: dict = {
            "mean_centerline_distance": float("inf"),
            "overlap_percent": 0.0,
        }
        if extracted_mask is not None and gt_mask is not None:
            result["dice_coefficient"] = 0.0
        return result

    # Build a KD-tree on the extracted centerline.
    tree_extracted = cKDTree(extracted_coords)

    # For every GT point, find the distance to its nearest extracted point.
    gt_to_ext_dist, _ = tree_extracted.query(gt_all)

    mean_distance = float(np.mean(gt_to_ext_dist))
    overlap_count = int(np.sum(gt_to_ext_dist <= tolerance_voxels))
    overlap_percent = 100.0 * overlap_count / gt_all.shape[0]

    # Also compute extracted-to-GT distances for a symmetric view.
    tree_gt = cKDTree(gt_all)
    ext_to_gt_dist, _ = tree_gt.query(extracted_coords)
    mean_distance_symmetric = float(
        0.5 * (np.mean(gt_to_ext_dist) + np.mean(ext_to_gt_dist))
    )

    result = {
        "mean_centerline_distance": mean_distance,
        "mean_symmetric_distance": mean_distance_symmetric,
        "overlap_percent": overlap_percent,
        "num_gt_points": int(gt_all.shape[0]),
        "num_extracted_points": int(extracted_coords.shape[0]),
        "num_gt_covered": overlap_count,
        "tolerance_voxels": tolerance_voxels,
    }

    # Dice coefficient on binary masks (if both are available).
    if extracted_mask is not None and gt_mask is not None:
        ext_bool = extracted_mask.astype(bool)
        gt_bool = gt_mask.astype(bool)
        intersection = int(np.count_nonzero(ext_bool & gt_bool))
        total = int(np.count_nonzero(ext_bool) + np.count_nonzero(gt_bool))
        dice = 2.0 * intersection / total if total > 0 else 0.0
        result["dice_coefficient"] = dice

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate extracted vessel centerlines against ground truth"
    )
    parser.add_argument(
        "--extracted",
        type=str,
        required=True,
        help="Path to extracted centerline coordinates (.npy, shape N x 3).",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        required=True,
        help="Path to ground-truth centerline coordinates. Accepts a single "
        ".npy file (N x 3) or a directory of .npy files (one per curve).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=2.0,
        help="Distance tolerance in voxels (default: 2.0).",
    )
    parser.add_argument(
        "--extracted-mask",
        type=str,
        default=None,
        help="Path to extracted binary mask (.npy) for Dice computation.",
    )
    parser.add_argument(
        "--gt-mask",
        type=str,
        default=None,
        help="Path to ground-truth binary mask (.npy) for Dice computation.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="If set, write metrics to this JSON file.",
    )
    args = parser.parse_args(argv)

    # -- Load extracted coordinates ----------------------------------------
    ext_path = Path(args.extracted)
    print(f"Loading extracted centerline from {ext_path} ...")
    extracted_coords = np.load(str(ext_path))
    if extracted_coords.ndim == 1:
        extracted_coords = extracted_coords.reshape(-1, 3)
    print(f"  {extracted_coords.shape[0]} points loaded")

    # -- Load ground-truth coordinates -------------------------------------
    gt_path = Path(args.ground_truth)
    gt_curves: list[np.ndarray] = []

    if gt_path.is_dir():
        npy_files = sorted(gt_path.glob("*.npy"))
        if not npy_files:
            print(f"ERROR: No .npy files found in {gt_path}")
            sys.exit(1)
        for f in npy_files:
            arr = np.load(str(f))
            if arr.ndim == 1:
                arr = arr.reshape(-1, 3)
            gt_curves.append(arr)
        print(f"  Loaded {len(gt_curves)} ground-truth curves from {gt_path}")
    else:
        arr = np.load(str(gt_path))
        if arr.ndim == 1:
            arr = arr.reshape(-1, 3)
        gt_curves.append(arr)
        print(f"  {arr.shape[0]} ground-truth points loaded from {gt_path}")

    # -- Optional masks ----------------------------------------------------
    ext_mask = None
    gt_mask = None
    if args.extracted_mask is not None:
        ext_mask = np.load(args.extracted_mask)
        print(f"  Extracted mask loaded: shape={ext_mask.shape}")
    if args.gt_mask is not None:
        gt_mask = np.load(args.gt_mask)
        print(f"  Ground-truth mask loaded: shape={gt_mask.shape}")

    # -- Evaluate ----------------------------------------------------------
    print(f"\nEvaluating (tolerance = {args.tolerance} voxels) ...\n")
    result = evaluate_centerline(
        extracted_coords=extracted_coords,
        ground_truth_coords=gt_curves,
        tolerance_voxels=args.tolerance,
        extracted_mask=ext_mask,
        gt_mask=gt_mask,
    )

    # -- Print results -----------------------------------------------------
    print("=" * 55)
    print("  Centerline Evaluation Results")
    print("=" * 55)
    print(f"  Extracted points          : {result['num_extracted_points']}")
    print(f"  Ground-truth points       : {result['num_gt_points']}")
    print(f"  Tolerance                 : {result['tolerance_voxels']:.1f} voxels")
    print(f"  Mean CL distance (GT->E)  : {result['mean_centerline_distance']:.4f} voxels")
    print(f"  Mean symmetric distance   : {result['mean_symmetric_distance']:.4f} voxels")
    print(f"  Overlap (GT covered)      : {result['overlap_percent']:.2f}%  "
          f"({result['num_gt_covered']}/{result['num_gt_points']})")
    if "dice_coefficient" in result:
        print(f"  Dice coefficient          : {result['dice_coefficient']:.4f}")
    print("=" * 55)

    # -- Optionally save to JSON -------------------------------------------
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"\nMetrics saved to {out_path.resolve()}")


if __name__ == "__main__":
    main()
