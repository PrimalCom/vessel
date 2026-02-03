"""3-D visualization utilities for vessel centerline data."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path


def plot_centerline_3d(
    centerline_coords: np.ndarray,
    graph_data: dict | None = None,
    radii: np.ndarray | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Render a 3-D scatter / line plot of the extracted centerline.

    Parameters
    ----------
    centerline_coords : np.ndarray
        (N, 3) array of voxel coordinates in (z, y, x) order.
    graph_data : dict | None
        Optional dictionary that may contain:
        - ``"endpoints"``     : array-like of (z, y, x) endpoint coords
        - ``"branch_points"`` : array-like of (z, y, x) branch-point coords
        - ``"segments"``      : list of (M_i, 3) arrays, each a connected segment
    radii : np.ndarray | None
        (N,) array of estimated radii at each centerline point.  When
        provided the scatter colour encodes radius instead of z-position.
    save_path : str | Path | None
        If given, figure is saved to this path.

    Returns
    -------
    plt.Figure
        The generated matplotlib figure.
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    if centerline_coords.ndim != 2 or centerline_coords.shape[1] != 3:
        raise ValueError(
            f"centerline_coords must be (N, 3), got {centerline_coords.shape}"
        )

    zz = centerline_coords[:, 0]
    yy = centerline_coords[:, 1]
    xx = centerline_coords[:, 2]

    # ---- colour by radius or z-position --------------------------------
    if radii is not None:
        colour_values = radii
        cmap = "plasma"
        clabel = "Radius"
    else:
        colour_values = zz
        cmap = "viridis"
        clabel = "Z position"

    sc = ax.scatter3D(
        xx, yy, zz,
        c=colour_values,
        cmap=cmap,
        s=2,
        alpha=0.5,
        label="centerline",
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.08)
    cbar.set_label(clabel)

    # ---- graph overlays ------------------------------------------------
    if graph_data is not None:
        # Segments as connected lines
        segments = graph_data.get("segments", [])
        for seg in segments:
            seg = np.asarray(seg)
            if seg.ndim == 2 and seg.shape[0] >= 2 and seg.shape[1] == 3:
                ax.plot3D(
                    seg[:, 2], seg[:, 1], seg[:, 0],
                    color="steelblue",
                    linewidth=1.0,
                    alpha=0.7,
                )

        # Endpoints
        end_pts = np.asarray(graph_data.get("endpoints", []))
        if end_pts.ndim == 2 and end_pts.shape[0] > 0:
            ax.scatter3D(
                end_pts[:, 2],
                end_pts[:, 1],
                end_pts[:, 0],
                c="red",
                s=50,
                marker="^",
                edgecolors="k",
                linewidths=0.5,
                label=f"endpoints ({end_pts.shape[0]})",
                zorder=5,
            )

        # Branch points
        branch_pts = np.asarray(graph_data.get("branch_points", []))
        if branch_pts.ndim == 2 and branch_pts.shape[0] > 0:
            ax.scatter3D(
                branch_pts[:, 2],
                branch_pts[:, 1],
                branch_pts[:, 0],
                c="yellow",
                s=50,
                marker="o",
                edgecolors="k",
                linewidths=0.5,
                label=f"branch pts ({branch_pts.shape[0]})",
                zorder=5,
            )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3-D Vessel Centerline")
    ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    return fig
