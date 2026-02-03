"""Vesselness enhancement filters for 3D vascular volumes.

Provides multi-scale vesselness computation using Frangi, Sato, and Meijering
filters from scikit-image. These filters enhance tubular structures (vessels)
by analyzing the eigenvalues of the Hessian matrix at multiple scales.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from skimage.filters import frangi, meijering, sato


def compute_vesselness(
    volume: np.ndarray,
    method: str = "frangi",
    sigmas: np.ndarray | None = None,
    black_ridges: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute vesselness enhancement of a 3D volume.

    Applies a multi-scale Hessian-based vesselness filter to enhance tubular
    structures. The output is normalized to [0, 1].

    Parameters
    ----------
    volume : np.ndarray
        Input 3D array (grayscale volume).
    method : str
        Vesselness filter to use. One of ``"frangi"``, ``"sato"``, or
        ``"meijering"``.
    sigmas : np.ndarray or None
        Iterable of scales (standard deviations) for multi-scale analysis.
        If *None*, defaults to ``np.arange(1.0, 5.0, 0.5)``.
    black_ridges : bool
        If *True*, detect dark ridges on a bright background. Set to *False*
        (default) for bright vessels on a dark background.

    Returns
    -------
    vesselness : np.ndarray
        Vesselness-enhanced volume normalized to [0, 1].
    params : dict
        Dictionary with keys ``"method"``, ``"sigmas"``, ``"black_ridges"``,
        ``"elapsed_seconds"``, and ``"output_shape"``.

    Raises
    ------
    ValueError
        If *method* is not one of the supported filter names.
    """
    if sigmas is None:
        sigmas = np.arange(1.0, 5.0, 0.5)

    sigmas_list = list(sigmas)

    supported_methods = {"frangi", "sato", "meijering"}
    if method not in supported_methods:
        raise ValueError(
            f"Unknown method '{method}'. Choose from {supported_methods}."
        )

    t_start = time.perf_counter()

    if method == "frangi":
        vesselness = frangi(
            volume,
            sigmas=sigmas_list,
            black_ridges=black_ridges,
        )
    elif method == "sato":
        vesselness = sato(
            volume,
            sigmas=sigmas_list,
            black_ridges=black_ridges,
        )
    elif method == "meijering":
        vesselness = meijering(
            volume,
            sigmas=sigmas_list,
            black_ridges=black_ridges,
        )

    elapsed = time.perf_counter() - t_start

    # Replace any NaN values that may arise from degenerate Hessian computations
    vesselness = np.nan_to_num(vesselness, nan=0.0)

    # Normalize to [0, 1]
    v_min = vesselness.min()
    v_max = vesselness.max()
    if v_max - v_min > 0:
        vesselness = (vesselness - v_min) / (v_max - v_min)
    else:
        vesselness = np.zeros_like(vesselness)

    params: dict[str, Any] = {
        "method": method,
        "sigmas": sigmas_list,
        "black_ridges": black_ridges,
        "elapsed_seconds": elapsed,
        "output_shape": vesselness.shape,
    }

    return vesselness, params
