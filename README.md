# Vessel-Guided Centerline Extraction

A 3D vessel centerline extraction pipeline for volumetric medical images (CTA/MRA). Generates synthetic phantoms for testing or processes real NIfTI/DICOM data through vesselness enhancement, segmentation, skeletonization, and graph-based centerline analysis.

## Requirements

- Python 3.10+

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

Run the demo with a synthetic vessel phantom:

```bash
python scripts/run_demo.py
```

This generates a 128x128x128 phantom, runs the full pipeline, and saves a summary figure to `outputs/figures/summary.png`.

## Full Pipeline

### Synthetic phantom (default)

```bash
python scripts/run_pipeline.py
```

### Real data

```bash
python scripts/run_pipeline.py --input /path/to/volume.nii.gz
```

Supported formats: NIfTI (`.nii`, `.nii.gz`), DICOM (directory of `.dcm` files), NumPy (`.npy`).

### Custom configuration

```bash
python scripts/run_pipeline.py --config config.yaml
```

Two config profiles are included:

- **`config.yaml`** -- Default profile tuned for synthetic phantoms and general use. Uses percentile thresholding, Gaussian denoising, and no morphological opening (preserves thin vessels).
- **`config_mra.yaml`** -- Optimized for real brain MRA data (e.g. `chris_MRA.nii.gz`). Uses wider sigma range, stronger denoising, and morphological opening to clean up noise.

```bash
# Synthetic data (default config)
python scripts/run_pipeline.py

# Real MRA data
python scripts/run_pipeline.py --config config_mra.yaml
```

Edit `config.yaml` to control every pipeline stage. Key options:

| Section | Option | Default | Description |
|---|---|---|---|
| `data.source` | `phantom` / file path | `phantom` | Input data source |
| `data.phantom_shape` | `[128,128,128]` | -- | Phantom volume dimensions |
| `preprocessing.normalize_method` | `minmax` / `zscore` / `window` | `minmax` | Intensity normalization |
| `preprocessing.denoise_method` | `gaussian` / `median` / `null` | `null` | Optional smoothing |
| `vesselness.method` | `frangi` / `sato` / `meijering` | `frangi` | Enhancement filter |
| `vesselness.sigma_start/stop/step` | floats | `1.0/5.0/0.5` | Scale range for multi-scale filtering |
| `segmentation.method` | `otsu` / `hysteresis` / `percentile` | `otsu` | Thresholding strategy |
| `segmentation.min_object_size` | int | `100` | Remove components smaller than this |
| `centerline.method` | `skeleton` / `distance` | `skeleton` | Extraction method |

## Evaluation

Compare extracted centerlines against ground truth:

```bash
python scripts/evaluate.py \
    --extracted outputs/results/centerline_coords.npy \
    --ground-truth outputs/results/gt_centerlines/ \
    --tolerance 2.0 \
    --output-json outputs/results/eval_metrics.json
```

Metrics computed: mean centerline distance, mean symmetric distance, overlap percentage, and Dice coefficient (when masks are provided via `--extracted-mask` and `--gt-mask`).

## Pipeline Stages

1. **Load** -- Synthetic phantom generation or file import
2. **Normalize** -- Intensity normalization (min-max, z-score, or CT windowing)
3. **Denoise** -- Optional Gaussian or median smoothing
4. **Vesselness** -- Multi-scale Hessian-based vessel enhancement (Frangi/Sato/Meijering)
5. **Segment** -- Thresholding (Otsu/hysteresis/percentile) + morphological cleanup
6. **Skeletonize** -- 3D topological thinning to single-voxel-wide skeleton
7. **Refine** -- Distance transform shifts skeleton points to vessel centers
8. **Graph** -- Topology analysis: endpoints, branch points, and traced segments

## Outputs

After a pipeline run, `outputs/` contains:

```
outputs/
  figures/
    summary.png          # 2x4 panel: MIPs, overlays, 3D scatter
  results/
    metrics.json         # Quantitative metrics
    summary.txt          # Human-readable report
    centerline_coords.npy
    centerline_radii.npy
    graph_data.json
```

## Project Structure

```
src/
  data/           phantom.py, loaders.py
  preprocessing/  normalize.py, denoise.py
  vesselness/     filters.py, multiscale.py
  segmentation/   threshold.py, morphology.py
  centerline/     skeleton.py, distance_field.py, graph.py
  visualization/  projections.py, plot_3d.py, report.py
scripts/
  run_pipeline.py   Full configurable pipeline
  run_demo.py       Quick synthetic demo
  evaluate.py       Standalone evaluation
tests/              28 unit + integration tests
```

## Test Datasets

Test data lives in `data/` and includes both real medical images and generated synthetic phantoms.

### Real Medical Images (NIfTI)

Downloaded from [neurolabusc/niivue-images](https://github.com/neurolabusc/niivue-images) -- a collection of low-resolution sample NIfTI images covering multiple modalities.

| File | Modality | Shape | Voxel Size (mm) | Description |
|------|----------|-------|------------------|-------------|
| `chris_MRA.nii.gz` | MR Angiography | 200x256x120 | 0.52 x 0.52 x 0.65 | Brain vessel imaging -- most relevant for this pipeline |
| `CT_AVM.nii.gz` | CT | 256x242x154 | 0.72 x 0.72 x 1.0 | Arteriovenous malformation (vascular pathology) |
| `CT_Abdo.nii.gz` | Abdominal CT | 255x178x256 | 1.49 x 1.49 x 1.49 | Abdominal CT with visible vessels |
| `pcasl.nii.gz` | Arterial Spin Label | 52x68x20x10 | 3.0 x 3.0 x 6.0 | Cerebral perfusion imaging (4D) |
| `zstat1.nii.gz` | fMRI stat map | 64x64x21 | 4.0 x 4.0 x 6.0 | Brain activation statistics (from [NIMH NIfTI test data](https://nifti.nimh.nih.gov/nifti-1/data/)) |

To run the pipeline on real MRA data:

```bash
python scripts/run_pipeline.py --input data/nifti/chris_MRA.nii.gz
```

Or update `config.yaml`:

```yaml
data:
  source: "data/nifti/chris_MRA.nii.gz"
```

### Synthetic Phantoms (NumPy)

Generated with `scripts/generate_test_data.py` using the project's own phantom generator. Each dataset has a matching ground-truth mask and centerline file in `data/ground_truth/`.

| File | Shape | Vessels | Noise | Description |
|------|-------|---------|-------|-------------|
| `simple_single_vessel.npy` | 64x64x64 | 1 | 0.02 | Single vessel, low noise -- baseline |
| `dense_branching.npy` | 96x96x96 | 12 | 0.05 | Dense vessel tree with many branches |
| `thin_vessels_noisy.npy` | 64x64x64 | 6 | 0.12 | Thin vessels with high noise -- challenging |
| `thick_vessels_clean.npy` | 80x80x80 | 4 | 0.01 | Thick vessels, minimal noise -- easy |
| `production_size.npy` | 128x128x128 | 8 | 0.05 | Matches default config parameters |
| `anisotropic_volume.npy` | 128x64x64 | 5 | 0.04 | Non-cubic volume, anisotropic spacing |

To regenerate the synthetic datasets:

```bash
python scripts/generate_test_data.py
```

The manifest at `data/manifest.json` lists all synthetic datasets with their metadata.

### Data Directory Layout

```
data/
  nifti/              Real medical images (.nii.gz)
  numpy/              Synthetic phantom volumes (.npy)
  ground_truth/       Masks and centerlines for synthetic phantoms
  manifest.json       Metadata for all synthetic datasets
```

## Tests

```bash
pytest tests/ -v
```
