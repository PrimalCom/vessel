"""Synthetic vessel phantom generation for testing centerline extraction algorithms.

Creates 3D volumes with known vessel trees built from quadratic Bezier curves,
providing ground-truth centerlines and segmentation masks for validation.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def _quadratic_bezier(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, num_points: int = 200
) -> np.ndarray:
    """Evaluate a quadratic Bezier curve at evenly spaced parameter values.

    Parameters
    ----------
    p0, p1, p2 : np.ndarray
        Control points, each of shape (3,) in [z, y, x] order.
    num_points : int
        Number of sample points along the curve.

    Returns
    -------
    np.ndarray
        Curve coordinates of shape (num_points, 3).
    """
    t = np.linspace(0.0, 1.0, num_points)[:, np.newaxis]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2


def _random_point_in_margin(
    shape: tuple[int, int, int],
    margin_frac: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a random 3-D point that lies within the volume but outside the margin band."""
    low = np.array([s * margin_frac for s in shape])
    high = np.array([s * (1.0 - margin_frac) for s in shape])
    return rng.uniform(low, high)


def _radius_profile(num_points: int, base_radius: float, rng: np.random.Generator) -> np.ndarray:
    """Generate a per-point radius that tapers at both ends and has slight random variation.

    The profile is a smooth bell-like curve (sin-based taper) multiplied by
    small random perturbations so vessels don't look perfectly uniform.
    """
    t = np.linspace(0.0, 1.0, num_points)
    # Smooth taper: sin goes 0 -> 1 -> 0 but we keep a floor so vessels don't vanish
    taper = 0.3 + 0.7 * np.sin(np.pi * t)
    noise = 1.0 + rng.normal(0.0, 0.05, size=num_points)
    noise = np.clip(noise, 0.8, 1.2)
    return base_radius * taper * noise


def _draw_vessel(
    volume: np.ndarray,
    mask: np.ndarray,
    centerline: np.ndarray,
    radii: np.ndarray,
    intensity: float = 1.0,
    falloff_width: float = 0.8,
) -> None:
    """Rasterise a single vessel into *volume* and *mask* in-place.

    At each centreline sample a sphere of the local radius is filled.  Voxels
    receive a smooth intensity that falls off near the boundary so that the
    vessel edges are not hard-stepped.

    Parameters
    ----------
    volume : np.ndarray
        The 3-D float64 image being built.
    mask : np.ndarray
        The 3-D bool ground-truth segmentation mask.
    centerline : np.ndarray
        (N, 3) array of centreline coordinates [z, y, x].
    radii : np.ndarray
        (N,) per-point vessel radius.
    intensity : float
        Peak voxel intensity inside the vessel core.
    falloff_width : float
        Width (in voxels) of the smooth Gaussian-like boundary transition.
    """
    shape = np.array(volume.shape)

    for pt, r in zip(centerline, radii):
        # Bounding box around the sphere
        r_ceil = int(np.ceil(r + 2 * falloff_width))
        lo = np.maximum(np.floor(pt - r_ceil).astype(int), 0)
        hi = np.minimum(np.ceil(pt + r_ceil).astype(int) + 1, shape)

        # Build local coordinate grids
        zz, yy, xx = np.mgrid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        dist = np.sqrt(
            (zz - pt[0]) ** 2 + (yy - pt[1]) ** 2 + (xx - pt[2]) ** 2
        )

        # Hard mask: everything within the stated radius
        inside = dist <= r
        mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] |= inside

        # Smooth intensity falloff using a Gaussian-like profile at the boundary
        # Core (dist <= r - falloff_width) gets full intensity
        # Boundary zone transitions smoothly to zero
        relative = np.clip((r - dist) / falloff_width, 0.0, 1.0)
        contrib = intensity * relative
        np.maximum(
            volume[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]],
            contrib,
            out=volume[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]],
        )


def create_vessel_phantom(
    shape: tuple[int, int, int] = (128, 128, 128),
    num_main_vessels: int = 3,
    num_branches: int = 5,
    radius_range: tuple[float, float] = (1.5, 4.5),
    noise_level: float = 0.05,
    seed: int = 42,
) -> dict:
    """Create a synthetic 3-D vessel phantom with ground-truth annotations.

    Vessels are modelled as quadratic Bezier curves with smoothly varying
    radii.  Branches fork from random locations on the main vessels.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Spatial dimensions (Z, Y, X) of the output volume.
    num_main_vessels : int
        Number of independent main vessel segments.
    num_branches : int
        Total number of branch vessels forking from main vessels.
    radius_range : tuple[float, float]
        (min_radius, max_radius) in voxels for base vessel radii.
    noise_level : float
        Standard deviation of additive Gaussian noise.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Keys:
        - ``"volume"`` -- float64 array normalised to [0, 1].
        - ``"ground_truth_mask"`` -- bool array.
        - ``"ground_truth_centerlines"`` -- list of (N, 3) float64 arrays [z, y, x].
        - ``"metadata"`` -- dict with ``shape``, ``num_vessels``,
          ``vessel_voxel_count``, ``vessel_volume_fraction``.
    """
    rng = np.random.default_rng(seed)
    volume = np.zeros(shape, dtype=np.float64)
    mask = np.zeros(shape, dtype=bool)
    centerlines: list[np.ndarray] = []

    margin = 0.12  # keep control points 12 % away from edges

    # --- Main vessels ---
    main_curves: list[np.ndarray] = []
    for _ in range(num_main_vessels):
        p0 = _random_point_in_margin(shape, margin, rng)
        p2 = _random_point_in_margin(shape, margin, rng)
        # Control point: offset from midpoint for curvature
        mid = (p0 + p2) / 2.0
        offset = rng.uniform(-0.25, 0.25, size=3) * np.array(shape)
        p1 = np.clip(mid + offset, [0, 0, 0], np.array(shape) - 1)

        num_pts = max(200, int(np.linalg.norm(p2 - p0) * 2))
        curve = _quadratic_bezier(p0, p1, p2, num_points=num_pts)
        # Clip to volume bounds
        curve = np.clip(curve, 0, np.array(shape) - 1)

        base_r = rng.uniform(*radius_range)
        radii = _radius_profile(num_pts, base_r, rng)

        _draw_vessel(volume, mask, curve, radii)
        centerlines.append(curve.copy())
        main_curves.append(curve)

    # --- Branch vessels ---
    for _ in range(num_branches):
        # Pick a random main vessel to branch from
        parent = main_curves[rng.integers(len(main_curves))]
        # Fork point: somewhere in the middle 60 % of the parent
        fork_idx = rng.integers(int(len(parent) * 0.2), int(len(parent) * 0.8))
        p0 = parent[fork_idx].copy()

        # Direction: roughly tangential + random deviation
        tangent = parent[min(fork_idx + 1, len(parent) - 1)] - parent[max(fork_idx - 1, 0)]
        tangent = tangent / (np.linalg.norm(tangent) + 1e-9)
        perp = rng.standard_normal(3)
        perp -= perp.dot(tangent) * tangent
        perp /= np.linalg.norm(perp) + 1e-9
        branch_dir = 0.4 * tangent + 0.6 * perp
        branch_dir /= np.linalg.norm(branch_dir) + 1e-9

        branch_len = rng.uniform(0.15, 0.35) * min(shape)
        p2 = p0 + branch_dir * branch_len
        p2 = np.clip(p2, [0, 0, 0], np.array(shape) - 1)

        mid = (p0 + p2) / 2.0
        offset = rng.uniform(-0.15, 0.15, size=3) * np.array(shape)
        p1 = np.clip(mid + offset, [0, 0, 0], np.array(shape) - 1)

        num_pts = max(80, int(np.linalg.norm(p2 - p0) * 2))
        curve = _quadratic_bezier(p0, p1, p2, num_points=num_pts)
        curve = np.clip(curve, 0, np.array(shape) - 1)

        # Branches are thinner
        base_r = rng.uniform(radius_range[0], (radius_range[0] + radius_range[1]) / 2.0)
        radii = _radius_profile(num_pts, base_r * 0.6, rng)

        _draw_vessel(volume, mask, curve, radii)
        centerlines.append(curve.copy())

    # --- Background tissue (low intensity) ---
    background = rng.uniform(0.05, 0.15, size=shape)
    background = gaussian_filter(background, sigma=3.0)
    volume = np.where(volume > background, volume, background)

    # --- Additive Gaussian noise ---
    if noise_level > 0:
        volume += rng.normal(0.0, noise_level, size=shape)

    # --- Normalise to [0, 1] ---
    vmin, vmax = volume.min(), volume.max()
    if vmax - vmin > 0:
        volume = (volume - vmin) / (vmax - vmin)
    volume = np.clip(volume, 0.0, 1.0)

    vessel_voxel_count = int(mask.sum())
    total_voxels = int(np.prod(shape))

    metadata = {
        "shape": shape,
        "num_vessels": num_main_vessels + num_branches,
        "vessel_voxel_count": vessel_voxel_count,
        "vessel_volume_fraction": vessel_voxel_count / total_voxels,
    }

    return {
        "volume": volume,
        "ground_truth_mask": mask,
        "ground_truth_centerlines": centerlines,
        "metadata": metadata,
    }
