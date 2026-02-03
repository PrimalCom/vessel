"""Denoising utilities for 3-D volumes.

Wraps scipy.ndimage filters behind a simple unified interface so that the
rest of the pipeline can switch strategies via a single string parameter.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter


def denoise_volume(
    volume: np.ndarray,
    method: str = "gaussian",
    sigma: float = 0.5,
) -> np.ndarray:
    """Denoise a 3-D volume using the specified filter.

    Parameters
    ----------
    volume : np.ndarray
        Input volume (any numeric dtype; converted to float64 internally).
    method : str
        Denoising strategy:

        * ``"gaussian"`` -- isotropic Gaussian smoothing via
          :func:`scipy.ndimage.gaussian_filter`.
        * ``"median"`` -- median filtering via
          :func:`scipy.ndimage.median_filter` with a cubic kernel whose
          side length is derived from *sigma* (``size = 2 * round(sigma) + 1``,
          minimum 3).

    sigma : float
        Controls the strength of smoothing.  For Gaussian filtering this is
        the standard deviation of the kernel.  For median filtering it
        determines the kernel size.

    Returns
    -------
    np.ndarray
        Denoised volume as float64.

    Raises
    ------
    ValueError
        If *method* is not one of the supported strategies.
    """
    volume = volume.astype(np.float64)

    if method == "gaussian":
        return gaussian_filter(volume, sigma=sigma)

    if method == "median":
        size = max(3, 2 * int(round(sigma)) + 1)
        return median_filter(volume, size=size).astype(np.float64)

    raise ValueError(
        f"Unknown denoising method '{method}'. "
        "Choose from 'gaussian' or 'median'."
    )
