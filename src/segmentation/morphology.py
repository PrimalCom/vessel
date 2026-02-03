"""Morphological post-processing for binary vessel masks.

Provides cleanup operations (opening, small-object removal) to refine
a raw binary segmentation into a cleaner vessel mask.
"""

from __future__ import annotations

import numpy as np
from skimage.morphology import ball, opening, remove_small_objects


def apply_morphological_cleanup(
    mask: np.ndarray,
    opening_radius: int = 1,
    min_object_size: int = 100,
) -> np.ndarray:
    """Clean a binary vessel mask using morphological operations.

    Applies binary opening to remove small noise speckles, then discards
    connected components that are smaller than *min_object_size*.

    Parameters
    ----------
    mask : np.ndarray
        Input binary mask (boolean or integer). Will be cast to ``bool``
        internally.
    opening_radius : int
        Radius of the ball structuring element used for binary opening.
        Larger values remove more noise but may erode thin vessels. Set to
        0 to skip the opening step.
    min_object_size : int
        Connected components with fewer voxels than this value are removed.
        Set to 0 to skip small-object removal.

    Returns
    -------
    cleaned : np.ndarray
        Cleaned boolean mask with the same shape as *mask*.
    """
    cleaned = mask.astype(bool)

    # Binary opening to remove noise speckle
    if opening_radius > 0:
        structuring_element = ball(opening_radius)
        cleaned = opening(cleaned, footprint=structuring_element)

    # Remove small connected components
    if min_object_size > 0:
        cleaned = remove_small_objects(cleaned, max_size=min_object_size - 1)

    return cleaned.astype(bool)
