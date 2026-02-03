"""MIP projections and summary figure generation for vessel centerline extraction."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – needed for 3D projection
from pathlib import Path


def compute_mip(volume: np.ndarray, axis: int = 0) -> np.ndarray:
    """Maximum Intensity Projection along given axis.

    Parameters
    ----------
    volume : np.ndarray
        3-D array (Z, Y, X).
    axis : int
        Axis along which to project. 0 = Z (axial), 1 = Y (coronal), 2 = X (sagittal).

    Returns
    -------
    np.ndarray
        2-D projected image.
    """
    return np.max(volume, axis=axis)


def create_summary_figure(
    volume: np.ndarray,
    vesselness: np.ndarray,
    vessel_mask: np.ndarray,
    skeleton: np.ndarray,
    centerline_coords: np.ndarray,
    graph_data: dict,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Create an 8-panel summary figure of the vessel extraction pipeline.

    Layout (2 rows x 4 cols):
        Row 1: Original MIP-Z | Vesselness MIP-Z | Segmentation MIP-Z | Centerline MIP-Z
        Row 2: Original MIP-Y | Vesselness MIP-Y | Overlay (seg+skel) MIP-Z | 3-D scatter

    Parameters
    ----------
    volume : np.ndarray
        Original 3-D image volume (Z, Y, X).
    vesselness : np.ndarray
        Vesselness filter response, same shape as *volume*.
    vessel_mask : np.ndarray
        Binary vessel segmentation mask.
    skeleton : np.ndarray
        Binary skeleton / centerline voxel mask.
    centerline_coords : np.ndarray
        (N, 3) array of centerline coordinates in (z, y, x) order.
    graph_data : dict
        Must contain at least:
        - ``"branch_points"`` : array-like of (z, y, x) branch-point coords
        - ``"endpoints"``     : array-like of (z, y, x) endpoint coords
        May also contain ``"segments"`` (list of arrays for connected lines).
    save_path : str | Path | None
        If given, figure is saved to this path.

    Returns
    -------
    plt.Figure
        The generated matplotlib figure.
    """
    # ------------------------------------------------------------------
    # Pre-compute all MIPs
    # ------------------------------------------------------------------
    vol_mip_z = compute_mip(volume, axis=0)
    vol_mip_y = compute_mip(volume, axis=1)

    vess_mip_z = compute_mip(vesselness, axis=0)
    vess_mip_y = compute_mip(vesselness, axis=1)

    seg_mip_z = compute_mip(vessel_mask, axis=0)

    skel_mip_z = compute_mip(skeleton, axis=0)

    # ------------------------------------------------------------------
    # Build figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # --- Row 1 --------------------------------------------------------
    # (0,0) Original MIP-Z
    axes[0, 0].imshow(vol_mip_z, cmap="gray")
    axes[0, 0].set_title("Original MIP-Z")
    axes[0, 0].axis("off")

    # (0,1) Vesselness MIP-Z
    axes[0, 1].imshow(vess_mip_z, cmap="gray")
    axes[0, 1].set_title("Vesselness MIP-Z")
    axes[0, 1].axis("off")

    # (0,2) Segmentation MIP-Z (binary white-on-black)
    axes[0, 2].imshow(seg_mip_z > 0, cmap="gray", vmin=0, vmax=1)
    axes[0, 2].set_title("Segmentation MIP-Z")
    axes[0, 2].axis("off")

    # (0,3) Centerline MIP-Z – skeleton overlay on volume
    axes[0, 3].imshow(vol_mip_z, cmap="gray")
    skel_overlay = np.ma.masked_where(skel_mip_z == 0, skel_mip_z)
    axes[0, 3].imshow(skel_overlay, cmap="autumn", alpha=0.8)
    axes[0, 3].set_title("Centerline MIP-Z")
    axes[0, 3].axis("off")

    # --- Row 2 --------------------------------------------------------
    # (1,0) Original MIP-Y
    axes[1, 0].imshow(vol_mip_y, cmap="gray")
    axes[1, 0].set_title("Original MIP-Y")
    axes[1, 0].axis("off")

    # (1,1) Vesselness MIP-Y
    axes[1, 1].imshow(vess_mip_y, cmap="gray")
    axes[1, 1].set_title("Vesselness MIP-Y")
    axes[1, 1].axis("off")

    # (1,2) Overlay: segmentation (red) + skeleton (cyan) MIP-Z
    seg_rgb = np.zeros((*seg_mip_z.shape, 3), dtype=np.float32)
    seg_norm = (seg_mip_z > 0).astype(np.float32)
    skel_norm = (skel_mip_z > 0).astype(np.float32)
    seg_rgb[..., 0] = seg_norm          # red channel  = segmentation
    seg_rgb[..., 1] = skel_norm * 0.9   # green channel = skeleton (cyan-ish)
    seg_rgb[..., 2] = skel_norm * 0.9   # blue channel  = skeleton (cyan-ish)
    axes[1, 2].imshow(seg_rgb)
    axes[1, 2].set_title("Overlay: Seg (R) + Skel (C)")
    axes[1, 2].axis("off")

    # (1,3) 3-D scatter of centerline
    ax3d = fig.add_subplot(2, 4, 8, projection="3d")
    axes[1, 3].set_visible(False)  # hide the flat axes behind it

    if centerline_coords.shape[0] > 0:
        zz = centerline_coords[:, 0]
        yy = centerline_coords[:, 1]
        xx = centerline_coords[:, 2]

        ax3d.scatter3D(
            xx, yy, zz,
            c=zz,
            cmap="viridis",
            s=1,
            alpha=0.4,
            label="centerline",
        )

        # Branch points
        branch_pts = np.asarray(graph_data.get("branch_points", []))
        if branch_pts.ndim == 2 and branch_pts.shape[0] > 0:
            ax3d.scatter3D(
                branch_pts[:, 2],
                branch_pts[:, 1],
                branch_pts[:, 0],
                c="yellow",
                s=30,
                marker="o",
                edgecolors="k",
                linewidths=0.5,
                label="branch pts",
            )

        # Endpoints
        end_pts = np.asarray(graph_data.get("endpoints", []))
        if end_pts.ndim == 2 and end_pts.shape[0] > 0:
            ax3d.scatter3D(
                end_pts[:, 2],
                end_pts[:, 1],
                end_pts[:, 0],
                c="red",
                s=30,
                marker="^",
                edgecolors="k",
                linewidths=0.5,
                label="endpoints",
            )

    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.set_title("3-D Centerline")
    ax3d.legend(loc="upper left", fontsize=7)

    # ------------------------------------------------------------------
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")

    return fig
