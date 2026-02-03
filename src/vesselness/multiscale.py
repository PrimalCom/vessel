"""Multi-scale aggregation utility for custom vesselness pipelines.

Provides manual control over per-sigma filter application and element-wise
maximum aggregation. The built-in scikit-image vesselness filters already
perform multi-scale aggregation internally; this module is useful when a
custom filter function or non-standard aggregation strategy is needed.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def aggregate_multiscale(
    volume: np.ndarray,
    sigmas: np.ndarray,
    filter_func: Callable[..., np.ndarray],
    **kwargs: Any,
) -> np.ndarray:
    """Apply a filter at multiple scales and aggregate via element-wise maximum.

    For each sigma in *sigmas*, ``filter_func(volume, sigma=sigma, **kwargs)``
    is called. The per-scale results are combined by taking the element-wise
    maximum across all scales, which preserves the strongest response at every
    voxel regardless of which scale produced it.

    Parameters
    ----------
    volume : np.ndarray
        Input 3D (or 2D) array to filter.
    sigmas : np.ndarray
        Array of scale values (standard deviations) to iterate over.
    filter_func : callable
        A function with signature ``filter_func(volume, sigma=..., **kwargs)``
        that returns an array of the same shape as *volume*.
    **kwargs
        Additional keyword arguments forwarded to *filter_func* at every scale.

    Returns
    -------
    aggregated : np.ndarray
        Element-wise maximum across all scale responses, same shape as *volume*.

    Raises
    ------
    ValueError
        If *sigmas* is empty.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.ndimage import gaussian_filter
    >>> vol = np.random.default_rng(0).random((32, 32, 32))
    >>> result = aggregate_multiscale(
    ...     vol,
    ...     sigmas=np.array([1.0, 2.0, 3.0]),
    ...     filter_func=lambda v, sigma: gaussian_filter(v, sigma=sigma),
    ... )
    >>> result.shape
    (32, 32, 32)
    """
    sigmas_arr = np.atleast_1d(sigmas)
    if sigmas_arr.size == 0:
        raise ValueError("sigmas must contain at least one value.")

    aggregated: np.ndarray | None = None

    for sigma in sigmas_arr:
        response = filter_func(volume, sigma=float(sigma), **kwargs)

        if aggregated is None:
            aggregated = response.copy()
        else:
            np.maximum(aggregated, response, out=aggregated)

    # aggregated is guaranteed non-None because sigmas is non-empty
    return aggregated  # type: ignore[return-value]
