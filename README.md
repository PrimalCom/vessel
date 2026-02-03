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

## Tests

```bash
pytest tests/ -v
```
