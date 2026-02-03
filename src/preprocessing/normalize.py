"""Intensity normalisation utilities for 3-D volumes.

Provides min-max scaling, z-score standardisation, and CT windowing that are
commonly needed before vessel segmentation or centreline extraction.
"""

from __future__ import annotations

import numpy as np


def normalize_intensity(volume: np.ndarray, method: str = "minmax") -> np.ndarray:
    """Normalise the intensity of a 3-D volume.

    Parameters
    ----------
    volume : np.ndarray
        Input volume (any numeric dtype).
    method : str
        Normalisation strategy:

        * ``"minmax"`` -- scale linearly to [0, 1].
        * ``"zscore"`` -- zero mean, unit variance.
        * ``"window"`` -- apply a default CT angiography window
          (level=300, width=700) then scale to [0, 1].

    Returns
    -------
    np.ndarray
        Normalised volume as float64.

    Raises
    ------
    ValueError
        If *method* is not one of the supported strategies.
    """
    volume = volume.astype(np.float64)

    if method == "minmax":
        vmin, vmax = volume.min(), volume.max()
        if vmax - vmin > 0:
            volume = (volume - vmin) / (vmax - vmin)
        else:
            volume = np.zeros_like(volume)
        return np.clip(volume, 0.0, 1.0)

    if method == "zscore":
        mean = volume.mean()
        std = volume.std()
        if std > 0:
            return (volume - mean) / std
        return volume - mean

    if method == "window":
        # Default CT angiography window
        return apply_ct_window(volume, level=300.0, width=700.0)

    raise ValueError(
        f"Unknown normalisation method '{method}'. "
        "Choose from 'minmax', 'zscore', or 'window'."
    )


def apply_ct_window(
    volume: np.ndarray,
    level: float,
    width: float,
) -> np.ndarray:
    """Apply standard CT intensity windowing then scale to [0, 1].

    Voxels outside the window are clamped to the window boundaries before
    linear rescaling.

    Parameters
    ----------
    volume : np.ndarray
        Input volume in Hounsfield units (or any linear scale).
    level : float
        Centre of the window (window level / WL).
    width : float
        Total width of the window (window width / WW).

    Returns
    -------
    np.ndarray
        float64 array with values in [0, 1].
    """
    volume = volume.astype(np.float64)
    lower = level - width / 2.0
    upper = level + width / 2.0
    volume = np.clip(volume, lower, upper)
    if upper - lower > 0:
        volume = (volume - lower) / (upper - lower)
    else:
        volume = np.zeros_like(volume)
    return volume
