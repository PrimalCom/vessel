"""Vessel segmentation via thresholding of vesselness-enhanced volumes.

Provides multiple thresholding strategies (Otsu, hysteresis, percentile)
followed by morphological post-processing to produce clean binary vessel masks.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from skimage.filters import apply_hysteresis_threshold, threshold_otsu
from skimage.morphology import ball, opening, remove_small_objects


def segment_vessels(
    vesselness: np.ndarray,
    method: str = "otsu",
    threshold_factor: float = 1.0,
    min_object_size: int = 100,
    opening_radius: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Segment vessels from a vesselness-enhanced volume.

    Parameters
    ----------
    vesselness : np.ndarray
        Vesselness image (typically normalized to [0, 1]).
    method : str
        Thresholding strategy. One of:

        - ``"otsu"`` -- Otsu threshold computed on non-zero voxels, scaled by
          *threshold_factor*.
        - ``"hysteresis"`` -- Hysteresis thresholding with low = otsu * 0.5 and
          high = otsu * 1.0.
        - ``"percentile"`` -- Keep the top fraction of voxels. The default
          fraction is 5 %; *threshold_factor* > 1 keeps fewer voxels, < 1 keeps
          more.
    threshold_factor : float
        Multiplier applied to the computed threshold (``"otsu"``), or scaling
        for the percentile cutoff (``"percentile"``).
    min_object_size : int
        Connected components with fewer voxels than this are removed.
    opening_radius : int
        Radius of the ball structuring element for morphological opening.
        Set to 0 to skip opening entirely (useful for preserving thin vessels).

    Returns
    -------
    mask : np.ndarray
        Boolean mask of segmented vessels.
    params : dict
        Dictionary containing ``"method"``, ``"threshold_value"``,
        ``"voxel_count"``, and ``"total_voxels"``.

    Raises
    ------
    ValueError
        If *method* is not a supported thresholding strategy.
    """
    supported = {"otsu", "hysteresis", "percentile"}
    if method not in supported:
        raise ValueError(f"Unknown method '{method}'. Choose from {supported}.")

    # --- Compute threshold ------------------------------------------------
    if method == "otsu":
        threshold_value = _otsu_on_nonzero(vesselness) * threshold_factor
        mask = vesselness >= threshold_value

    elif method == "hysteresis":
        otsu_val = _otsu_on_nonzero(vesselness)
        low = otsu_val * 0.5
        high = otsu_val * 1.0
        threshold_value = otsu_val  # report the base Otsu value
        mask = apply_hysteresis_threshold(vesselness, low, high)

    elif method == "percentile":
        # Default: keep top 5 % of voxels.  threshold_factor > 1 raises the
        # percentile (keeps fewer), < 1 lowers it (keeps more).
        keep_pct = 5.0 / threshold_factor
        keep_pct = np.clip(keep_pct, 0.01, 99.99)
        threshold_value = float(np.percentile(vesselness, 100.0 - keep_pct))
        mask = vesselness >= threshold_value

    # --- Post-processing --------------------------------------------------
    # Morphological opening to remove speckle noise
    if opening_radius > 0:
        mask = opening(mask, footprint=ball(opening_radius))

    # Remove small connected components
    mask = remove_small_objects(mask, max_size=min_object_size - 1)

    # Ensure boolean dtype
    mask = mask.astype(bool)

    params: dict[str, Any] = {
        "method": method,
        "threshold_value": float(threshold_value),
        "threshold_factor": threshold_factor,
        "voxel_count": int(mask.sum()),
        "total_voxels": int(mask.size),
    }

    return mask, params


def _otsu_on_nonzero(image: np.ndarray) -> float:
    """Compute Otsu threshold using only non-zero voxels.

    Background voxels (exact zeros) dominate many vesselness images and
    can skew the Otsu threshold to an unhelpfully low value. Restricting
    the histogram to non-zero values produces a more meaningful split.

    Parameters
    ----------
    image : np.ndarray
        Input image array.

    Returns
    -------
    float
        Otsu threshold computed on the non-zero values. Falls back to the
        global Otsu threshold if all values are zero.
    """
    nonzero = image[image > 0]
    if nonzero.size == 0:
        return float(threshold_otsu(image))
    return float(threshold_otsu(nonzero))
