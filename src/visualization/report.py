"""Metrics reporting and centerline data persistence utilities."""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path


def generate_report(
    metrics: dict,
    graph_data: dict,
    save_dir: str | Path,
) -> dict:
    """Compile pipeline metrics into JSON and a human-readable text summary.

    Parameters
    ----------
    metrics : dict
        Arbitrary metric dictionary produced by the evaluation step.  Common
        keys include ``"total_centerline_length"``,
        ``"mean_centerline_distance"``, ``"overlap_percentage"``, and
        ``"dice"``.
    graph_data : dict
        Graph-level information.  Expected keys:
        - ``"segments"``      : list of segment arrays
        - ``"endpoints"``     : array-like of endpoint coords
        - ``"branch_points"`` : array-like of branch-point coords
    save_dir : str | Path
        Root output directory.  Files are written into ``save_dir/results/``.

    Returns
    -------
    dict
        A combined report dictionary containing both the raw *metrics* and
        the derived *summary* statistics.
    """
    save_dir = Path(save_dir)
    results_dir = save_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---- Derive summary statistics ------------------------------------
    segments = graph_data.get("segments", [])
    endpoints = np.asarray(graph_data.get("endpoints", [])).reshape(-1, 3) if len(graph_data.get("endpoints", [])) else np.empty((0, 3))
    branch_points = np.asarray(graph_data.get("branch_points", [])).reshape(-1, 3) if len(graph_data.get("branch_points", [])) else np.empty((0, 3))

    num_segments = len(segments)
    num_endpoints = int(endpoints.shape[0])
    num_branch_points = int(branch_points.shape[0])
    total_length = float(metrics.get("total_centerline_length", 0.0))
    mean_distance = metrics.get("mean_centerline_distance", None)
    overlap_pct = metrics.get("overlap_percentage", None)
    dice = metrics.get("dice", None)

    summary = {
        "num_segments": num_segments,
        "num_endpoints": num_endpoints,
        "num_branch_points": num_branch_points,
        "total_centerline_length": total_length,
        "mean_centerline_distance": mean_distance,
        "overlap_percentage": overlap_pct,
        "dice": dice,
    }

    report = {
        "metrics": metrics,
        "summary": summary,
    }

    # ---- Save JSON -----------------------------------------------------
    def _default(obj):
        """JSON serialiser for numpy types."""
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    metrics_path = results_dir / "metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(report, fh, indent=2, default=_default)

    # ---- Save text summary ---------------------------------------------
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("  Vessel Centerline Extraction — Summary Report")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"  Segments        : {num_segments}")
    lines.append(f"  Endpoints       : {num_endpoints}")
    lines.append(f"  Branch points   : {num_branch_points}")
    lines.append(f"  Total length    : {total_length:.2f} voxels")
    lines.append("")

    if mean_distance is not None:
        lines.append(f"  Mean centerline distance : {mean_distance:.4f}")
    if overlap_pct is not None:
        lines.append(f"  Overlap percentage       : {overlap_pct:.2f} %")
    if dice is not None:
        lines.append(f"  Dice coefficient         : {dice:.4f}")

    lines.append("")
    lines.append("=" * 60)
    lines.append("")

    summary_text = "\n".join(lines)
    summary_path = results_dir / "summary.txt"
    summary_path.write_text(summary_text)

    return report


def save_centerline_data(
    coords: np.ndarray,
    radii: np.ndarray | None,
    graph_data: dict,
    save_dir: str | Path,
) -> None:
    """Persist centerline coordinates, radii, and graph summary to disk.

    Parameters
    ----------
    coords : np.ndarray
        (N, 3) centerline coordinates in (z, y, x) order.
    radii : np.ndarray | None
        (N,) estimated vessel radii at each centerline point.
    graph_data : dict
        Graph-level information (segments, endpoints, branch_points).
    save_dir : str | Path
        Root output directory.  Files are written into ``save_dir/results/``.
    """
    save_dir = Path(save_dir)
    results_dir = save_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---- Centerline coordinates ----------------------------------------
    np.save(str(results_dir / "centerline_coords.npy"), coords)

    coords_list = coords.tolist()  # list of [z, y, x] lists
    with open(results_dir / "centerline_coords.json", "w") as fh:
        json.dump(coords_list, fh)

    # ---- Radii ---------------------------------------------------------
    if radii is not None:
        np.save(str(results_dir / "centerline_radii.npy"), radii)

    # ---- Graph data summary --------------------------------------------
    def _serialise(obj):
        """Convert numpy types for JSON encoding."""
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    segments = graph_data.get("segments", [])
    endpoints = graph_data.get("endpoints", [])
    branch_points = graph_data.get("branch_points", [])

    graph_summary = {
        "num_segments": len(segments),
        "num_endpoints": len(np.asarray(endpoints).reshape(-1, 3)) if len(endpoints) else 0,
        "num_branch_points": len(np.asarray(branch_points).reshape(-1, 3)) if len(branch_points) else 0,
        "endpoints": np.asarray(endpoints).tolist() if len(endpoints) else [],
        "branch_points": np.asarray(branch_points).tolist() if len(branch_points) else [],
        "segment_lengths": [
            int(np.asarray(s).shape[0]) for s in segments
        ],
    }

    with open(results_dir / "graph_data.json", "w") as fh:
        json.dump(graph_summary, fh, indent=2, default=_serialise)
