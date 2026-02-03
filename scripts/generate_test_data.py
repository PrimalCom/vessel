#!/usr/bin/env python3
"""Generate diverse synthetic test datasets for the vessel pipeline.

Creates several .npy volumes with varying parameters (vessel count, noise,
radius, shape) plus their ground-truth masks and centerlines, so the
pipeline can be tested against known geometry.

Usage:
    python -m scripts.generate_test_data
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.phantom import create_vessel_phantom

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
NPY_DIR = DATA_DIR / "numpy"
GT_DIR = DATA_DIR / "ground_truth"


DATASETS = [
    {
        "name": "simple_single_vessel",
        "params": {
            "shape": (64, 64, 64),
            "num_main_vessels": 1,
            "num_branches": 0,
            "radius_range": (2.0, 3.0),
            "noise_level": 0.02,
            "seed": 100,
        },
        "description": "Single straight-ish vessel, low noise, small volume",
    },
    {
        "name": "dense_branching",
        "params": {
            "shape": (96, 96, 96),
            "num_main_vessels": 4,
            "num_branches": 8,
            "radius_range": (1.5, 4.0),
            "noise_level": 0.05,
            "seed": 200,
        },
        "description": "Dense vessel tree with many branches",
    },
    {
        "name": "thin_vessels_noisy",
        "params": {
            "shape": (64, 64, 64),
            "num_main_vessels": 3,
            "num_branches": 3,
            "radius_range": (1.0, 2.0),
            "noise_level": 0.12,
            "seed": 300,
        },
        "description": "Thin vessels with high noise - challenging case",
    },
    {
        "name": "thick_vessels_clean",
        "params": {
            "shape": (80, 80, 80),
            "num_main_vessels": 2,
            "num_branches": 2,
            "radius_range": (3.5, 6.0),
            "noise_level": 0.01,
            "seed": 400,
        },
        "description": "Thick vessels with minimal noise - easy case",
    },
    {
        "name": "production_size",
        "params": {
            "shape": (128, 128, 128),
            "num_main_vessels": 3,
            "num_branches": 5,
            "radius_range": (1.5, 4.5),
            "noise_level": 0.05,
            "seed": 500,
        },
        "description": "Production-size phantom matching default config",
    },
    {
        "name": "anisotropic_volume",
        "params": {
            "shape": (128, 64, 64),
            "num_main_vessels": 2,
            "num_branches": 3,
            "radius_range": (1.5, 3.5),
            "noise_level": 0.04,
            "seed": 600,
        },
        "description": "Non-cubic volume simulating anisotropic voxel spacing",
    },
]


def save_centerlines_as_json(centerlines: list[np.ndarray], path: Path) -> None:
    """Save centerlines as JSON (list of list of [z, y, x] coords)."""
    data = [cl.tolist() for cl in centerlines]
    with open(path, "w") as f:
        json.dump(data, f)


def main() -> None:
    NPY_DIR.mkdir(parents=True, exist_ok=True)
    GT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = []

    for ds in DATASETS:
        name = ds["name"]
        print(f"Generating: {name} ...")

        result = create_vessel_phantom(**ds["params"])

        # Save volume
        vol_path = NPY_DIR / f"{name}.npy"
        np.save(str(vol_path), result["volume"])

        # Save ground-truth mask
        mask_path = GT_DIR / f"{name}_mask.npy"
        np.save(str(mask_path), result["ground_truth_mask"])

        # Save ground-truth centerlines as JSON (safe serialization)
        cl_path = GT_DIR / f"{name}_centerlines.json"
        save_centerlines_as_json(result["ground_truth_centerlines"], cl_path)

        entry = {
            "name": name,
            "description": ds["description"],
            "volume": str(vol_path.relative_to(DATA_DIR.parent)),
            "ground_truth_mask": str(mask_path.relative_to(DATA_DIR.parent)),
            "ground_truth_centerlines": str(cl_path.relative_to(DATA_DIR.parent)),
            "shape": list(result["metadata"]["shape"]),
            "num_vessels": result["metadata"]["num_vessels"],
            "vessel_volume_fraction": round(result["metadata"]["vessel_volume_fraction"], 4),
        }
        manifest.append(entry)

        size_mb = vol_path.stat().st_size / (1024 * 1024)
        print(f"  -> {vol_path.name} ({size_mb:.1f} MB), shape={result['metadata']['shape']}, "
              f"vessels={result['metadata']['num_vessels']}, "
              f"vessel_fraction={result['metadata']['vessel_volume_fraction']:.3f}")

    # Save manifest
    manifest_path = DATA_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {manifest_path}")
    print(f"Total datasets: {len(manifest)}")


if __name__ == "__main__":
    main()
