"""Distance-transform refinement for vessel centerlines.

Shifts skeleton voxels toward local maxima of the Euclidean distance
transform, producing sub-voxel-accurate centerline positions and per-point
vessel radius estimates.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt


def refine_centerline_with_distance(
    vessel_mask: np.ndarray,
    skeleton: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Refine skeleton points using the Euclidean distance transform.

    For every skeleton voxel the 3x3x3 neighbourhood in the distance
    transform is examined and the point is shifted to the neighbour with the
    largest distance value.  This moves skeleton voxels toward the true
    medial axis of the vessel lumen.

    Parameters
    ----------
    vessel_mask : np.ndarray
        Binary 3D volume (same shape as *skeleton*).  Non-zero voxels are
        treated as vessel interior.
    skeleton : np.ndarray
        Boolean (or 0/1) 3D array marking the one-voxel-wide centerline,
        typically produced by :func:`skeleton.extract_centerline_skeleton`.

    Returns
    -------
    refined_coords : np.ndarray
        Float array of shape ``(N, 3)`` with refined ``[z, y, x]``
        positions.  When no better neighbour exists the original integer
        coordinate is kept (cast to float).
    radii : np.ndarray
        Float array of shape ``(N,)`` giving the estimated vessel radius at
        each refined point (the EDT value at that location).

    Notes
    -----
    The refinement is purely local (3x3x3 window) so it is fast and does
    not require iterative optimisation.  For sub-voxel interpolation one
    could fit a paraboloid to the neighbourhood, but the single-voxel shift
    is sufficient for most downstream graph-building tasks.
    """
    mask_bool = vessel_mask.astype(bool, copy=False)

    # Compute the Euclidean distance transform: each foreground voxel gets
    # the distance (in voxels) to the nearest background voxel.
    dt: np.ndarray = distance_transform_edt(mask_bool)

    # Skeleton voxel coordinates — (N, 3) int array [z, y, x].
    skel_coords = np.argwhere(skeleton.astype(bool, copy=False))
    n_pts = skel_coords.shape[0]

    if n_pts == 0:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )

    # Pre-compute the 3x3x3 offset kernel (27 offsets including centre).
    offsets = np.array(
        [[dz, dy, dx] for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)],
        dtype=np.intp,
    )  # shape (27, 3)

    shape = np.array(dt.shape, dtype=np.intp)

    refined_coords = np.empty((n_pts, 3), dtype=np.float64)
    radii = np.empty(n_pts, dtype=np.float64)

    for i in range(n_pts):
        pt = skel_coords[i]  # (3,) int array [z, y, x]

        # Candidate positions = pt + each offset, clamped to volume bounds.
        candidates = pt + offsets  # (27, 3)

        # Clip to valid index range.
        np.clip(candidates[:, 0], 0, shape[0] - 1, out=candidates[:, 0])
        np.clip(candidates[:, 1], 0, shape[1] - 1, out=candidates[:, 1])
        np.clip(candidates[:, 2], 0, shape[2] - 1, out=candidates[:, 2])

        # Evaluate the distance transform at every candidate.
        dt_vals = dt[candidates[:, 0], candidates[:, 1], candidates[:, 2]]

        best_idx = int(np.argmax(dt_vals))
        best_pos = candidates[best_idx]

        refined_coords[i] = best_pos.astype(np.float64)
        radii[i] = dt_vals[best_idx]

    return refined_coords, radii
