"""Data loading utilities for vessel centerline extraction.

Supports synthetic phantom generation, NIfTI, DICOM directories, and NumPy
arrays.  Each loader returns a float64 volume normalised to [0, 1] together
with a metadata dictionary.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _normalize_to_unit(volume: np.ndarray) -> np.ndarray:
    """Scale an arbitrary numeric array to float64 in [0, 1]."""
    volume = volume.astype(np.float64)
    vmin, vmax = volume.min(), volume.max()
    if vmax - vmin > 0:
        volume = (volume - vmin) / (vmax - vmin)
    return np.clip(volume, 0.0, 1.0)


def _load_phantom() -> tuple[np.ndarray, dict]:
    """Generate a synthetic vessel phantom and return volume + metadata."""
    from src.data.phantom import create_vessel_phantom

    result = create_vessel_phantom()
    metadata = dict(result["metadata"])
    metadata["source"] = "synthetic_phantom"
    metadata["gt_centerlines"] = result["ground_truth_centerlines"]
    metadata["gt_mask"] = result["ground_truth_mask"]
    return result["volume"], metadata


def _load_nifti(path: Path) -> tuple[np.ndarray, dict]:
    """Load a NIfTI (.nii / .nii.gz) file via nibabel."""
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError(
            "nibabel is required to load NIfTI files. "
            "Install it with:  pip install nibabel"
        ) from exc

    img = nib.load(str(path))
    volume = np.asarray(img.dataobj)
    volume = _normalize_to_unit(volume)

    header = img.header
    metadata: dict = {
        "source": "nifti",
        "path": str(path),
        "shape": volume.shape,
        "affine": img.affine.tolist(),
    }
    if hasattr(header, "get_zooms"):
        metadata["voxel_size"] = list(header.get_zooms())

    return volume, metadata


def _load_dicom_directory(path: Path) -> tuple[np.ndarray, dict]:
    """Load a directory of DICOM (.dcm) files via pydicom.

    Slices are sorted by ``InstanceNumber`` when available, falling back to
    ``ImagePositionPatient[2]`` (the slice position along the axial axis).
    """
    try:
        import pydicom
    except ImportError as exc:
        raise ImportError(
            "pydicom is required to load DICOM directories. "
            "Install it with:  pip install pydicom"
        ) from exc

    dcm_files = sorted(path.glob("*.dcm"))
    if not dcm_files:
        raise FileNotFoundError(f"No .dcm files found in {path}")

    slices = [pydicom.dcmread(str(f)) for f in dcm_files]

    # Sort by InstanceNumber if present, else by ImagePositionPatient z
    def _sort_key(ds):  # type: ignore[no-untyped-def]
        if hasattr(ds, "InstanceNumber") and ds.InstanceNumber is not None:
            return int(ds.InstanceNumber)
        if hasattr(ds, "ImagePositionPatient") and ds.ImagePositionPatient is not None:
            return float(ds.ImagePositionPatient[2])
        return 0

    slices.sort(key=_sort_key)

    pixel_arrays = [s.pixel_array.astype(np.float64) for s in slices]
    volume = np.stack(pixel_arrays, axis=0)

    # Apply rescale slope / intercept if present
    ds0 = slices[0]
    slope = float(getattr(ds0, "RescaleSlope", 1.0))
    intercept = float(getattr(ds0, "RescaleIntercept", 0.0))
    volume = volume * slope + intercept

    volume = _normalize_to_unit(volume)

    metadata: dict = {
        "source": "dicom",
        "path": str(path),
        "shape": volume.shape,
        "num_slices": len(slices),
    }
    if hasattr(ds0, "PixelSpacing"):
        ps = [float(v) for v in ds0.PixelSpacing]
        st = float(getattr(ds0, "SliceThickness", ps[0]))
        metadata["voxel_size"] = [st, ps[0], ps[1]]

    return volume, metadata


def _load_numpy(path: Path) -> tuple[np.ndarray, dict]:
    """Load a volume from a ``.npy`` file."""
    volume = np.load(str(path))
    volume = _normalize_to_unit(volume)
    metadata: dict = {
        "source": "numpy",
        "path": str(path),
        "shape": volume.shape,
    }
    return volume, metadata


def load_volume(path: str | Path | None = None) -> tuple[np.ndarray, dict]:
    """Load a 3-D volume from disk or generate a synthetic phantom.

    The format is auto-detected from the file extension or path type:

    * ``None`` -- generate a synthetic vessel phantom.
    * ``.nii`` / ``.nii.gz`` -- NIfTI via *nibabel*.
    * Directory of ``.dcm`` files -- DICOM via *pydicom*.
    * ``.npy`` -- NumPy binary file.

    Parameters
    ----------
    path : str | Path | None
        Path to the data source, or ``None`` for a synthetic phantom.

    Returns
    -------
    volume : np.ndarray
        float64 array normalised to [0, 1].
    metadata : dict
        Metadata about the loaded volume.  For synthetic phantoms this
        includes ``"gt_centerlines"`` and ``"gt_mask"``.
    """
    if path is None:
        return _load_phantom()

    path = Path(path)

    if path.is_dir():
        return _load_dicom_directory(path)

    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".nii.gz") or suffixes.endswith(".nii"):
        return _load_nifti(path)

    if suffixes.endswith(".npy"):
        return _load_numpy(path)

    raise ValueError(
        f"Unsupported file format: {path}. "
        "Expected .nii, .nii.gz, a directory of .dcm files, or .npy"
    )
