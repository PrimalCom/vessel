"""3D Skeletonization for vessel centerline extraction.

Extracts a one-voxel-wide centerline from a binary vessel mask using
morphological skeletonization.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from skimage.morphology import skeletonize


def extract_centerline_skeleton(
    vessel_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Extract a 3D skeleton (centerline) from a binary vessel segmentation mask.

    Uses ``skimage.morphology.skeletonize`` which handles 2D and 3D inputs
    natively.  The input is cast to ``bool`` before processing so both boolean
    and uint8 masks are accepted.

    Parameters
    ----------
    vessel_mask : np.ndarray
        Binary 3D volume where non-zero voxels represent the vessel lumen.
        Must be convertible to a boolean array.

    Returns
    -------
    skeleton : np.ndarray
        Boolean array of the same shape as *vessel_mask* with ``True`` at
        voxels that belong to the one-voxel-wide centerline.
    coords : np.ndarray
        Integer array of shape ``(N, 3)`` containing ``[z, y, x]`` indices of
        every skeleton voxel, obtained via ``np.argwhere(skeleton)``.
    stats : dict
        Dictionary with the following keys:

        - ``skeleton_voxels`` – number of skeleton voxels.
        - ``mask_voxels`` – number of foreground voxels in the input mask.
        - ``reduction_ratio`` – ``skeleton_voxels / mask_voxels`` (0.0 if the
          mask is empty).
        - ``elapsed_seconds`` – wall-clock time spent on skeletonization.

    Raises
    ------
    ValueError
        If *vessel_mask* has fewer than 2 dimensions.
    """
    if vessel_mask.ndim < 2:
        raise ValueError(
            f"vessel_mask must be at least 2-D, got {vessel_mask.ndim}-D array"
        )

    # Ensure boolean input (skeletonize expects bool or 0/1 uint8).
    mask = vessel_mask.astype(bool, copy=False)
    mask_voxels = int(np.count_nonzero(mask))

    t0 = time.perf_counter()
    skeleton: np.ndarray = skeletonize(mask)
    elapsed = time.perf_counter() - t0

    # skeletonize returns a boolean array for boolean input.
    skeleton = skeleton.astype(bool, copy=False)

    coords = np.argwhere(skeleton)  # (N, 3) int array, [z, y, x] ordering
    skeleton_voxels = coords.shape[0]

    reduction_ratio = (
        skeleton_voxels / mask_voxels if mask_voxels > 0 else 0.0
    )

    stats: dict[str, Any] = {
        "skeleton_voxels": skeleton_voxels,
        "mask_voxels": mask_voxels,
        "reduction_ratio": reduction_ratio,
        "elapsed_seconds": elapsed,
    }

    return skeleton, coords, stats
