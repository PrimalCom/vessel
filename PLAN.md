# Vessel-Guided Centerline Extraction — Coding Agent Task Spec

## Context & Objective

Build a **proof-of-concept Python pipeline** that demonstrates **vessel-guided centerline extraction** from 3D medical imaging volumes (CT Angiography / MR Angiography). This is a prototype to validate the approach before investing in a production system.

The pipeline must work **end-to-end**: raw volume in → centerline coordinates + visualization out. It must run standalone with zero external data dependencies (synthetic phantom for demo), but be structured so real DICOM/NIfTI data can be swapped in trivially.

---

## What is Vessel-Guided Centerline Extraction?

Blood vessels are tubular structures in medical images. A **centerline** is the 1-voxel-wide medial axis running through the center of each vessel. Extracting centerlines is critical for:

- Measuring vessel length, tortuosity, and stenosis
- Surgical planning and navigation
- Building vessel tree graphs for analysis
- Generating curved planar reformations (CPR) for radiologists

The "vessel-guided" part means we first enhance/segment vessel structures, then extract the centerline from the segmentation — as opposed to purely intensity-based tracking.

The standard pipeline is:

```
Raw Volume → Vesselness Enhancement → Segmentation → Skeletonization → Graph/Centerline
```

---

## Technical Requirements

### Environment

- **Language**: Python 3.10+
- **Core dependencies** (all pip-installable):
  - `numpy` — array operations
  - `scipy` — distance transforms, ndimage operations
  - `scikit-image` — vesselness filters (Frangi/Sato/Meijering), skeletonization, morphology, thresholding
  - `matplotlib` — 2D visualization and MIP projections
- **Optional dependencies** (for real data and advanced viz):
  - `nibabel` — NIfTI file loading (.nii, .nii.gz)
  - `pydicom` — DICOM file loading
  - `SimpleITK` — alternative loader + preprocessing
  - `vmtk` — advanced centerline computation (marching cubes + Voronoi)
  - `napari` — interactive 3D visualization
  - `networkx` — graph-based centerline analysis
  - `trimesh` or `open3d` — 3D mesh/point cloud viz
- **No GPU required** — CPU-only for the prototype
- **No internet required at runtime** — all processing is local

### Project Structure

```
vessel-centerline-poc/
├── README.md                          # Setup, usage, architecture overview
├── requirements.txt                   # Pinned dependencies
├── pyproject.toml                     # Project metadata (optional)
├── config.yaml                        # Pipeline parameters (thresholds, scales, etc.)
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── phantom.py                 # Synthetic vessel phantom generator
│   │   └── loaders.py                 # DICOM, NIfTI, NumPy loaders
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── normalize.py               # Intensity normalization, windowing
│   │   └── denoise.py                 # Gaussian smoothing, NLM denoising
│   │
│   ├── vesselness/
│   │   ├── __init__.py
│   │   ├── filters.py                 # Frangi, Sato, Meijering wrappers
│   │   └── multiscale.py              # Multi-scale vesselness aggregation
│   │
│   ├── segmentation/
│   │   ├── __init__.py
│   │   ├── threshold.py               # Otsu, hysteresis, percentile methods
│   │   └── morphology.py              # Cleanup: erosion, dilation, small object removal
│   │
│   ├── centerline/
│   │   ├── __init__.py
│   │   ├── skeleton.py                # 3D skeletonization (skimage)
│   │   ├── distance_field.py          # Distance-transform-based centerline refinement
│   │   └── graph.py                   # Skeleton → graph: endpoints, branches, segments
│   │
│   └── visualization/
│       ├── __init__.py
│       ├── projections.py             # MIP, slice views, overlay renderings
│       ├── plot_3d.py                 # 3D scatter/line plots of centerline
│       └── report.py                  # Generate summary figure + metrics JSON
│
├── scripts/
│   ├── run_pipeline.py                # Main entry point — runs full pipeline
│   ├── run_demo.py                    # Quick demo with synthetic phantom
│   └── evaluate.py                    # Compare extracted vs ground-truth centerlines
│
├── tests/
│   ├── test_phantom.py                # Phantom generates valid volumes
│   ├── test_vesselness.py             # Vesselness filters produce expected output
│   ├── test_segmentation.py           # Segmentation produces binary masks
│   ├── test_centerline.py             # Skeleton is topologically valid
│   └── test_pipeline_e2e.py           # Full pipeline runs without errors
│
└── outputs/                           # Generated results (gitignored)
    ├── figures/
    └── results/
```

---

## Module Specifications

### 1. `src/data/phantom.py` — Synthetic Vessel Phantom

**Purpose**: Generate a realistic 3D volume with known vessel geometry so we can validate the pipeline without real medical data.

```python
def create_vessel_phantom(
    shape: tuple[int, int, int] = (128, 128, 128),
    num_main_vessels: int = 3,
    num_branches: int = 5,
    radius_range: tuple[float, float] = (1.5, 4.5),
    noise_level: float = 0.05,
    seed: int = 42,
) -> dict:
    """
    Returns:
        {
            "volume": np.ndarray,           # float64, shape=(Z,Y,X), range [0,1]
            "ground_truth_mask": np.ndarray, # bool, shape=(Z,Y,X)
            "ground_truth_centerlines": list[np.ndarray],  # list of (N,3) arrays [z,y,x]
            "metadata": {
                "shape": tuple,
                "num_vessels": int,
                "vessel_voxel_count": int,
                "vessel_volume_fraction": float,
            }
        }
    """
```

**Implementation details**:
- Use quadratic Bezier curves to create curved vessel paths
- Vary vessel radius along the path (taper at ends, slight random variation)
- Fill spherical cross-sections at each point along the path
- Apply smooth intensity falloff at vessel boundaries (not hard edges)
- Add Gaussian noise to simulate imaging noise
- Optionally add background tissue with lower intensity
- Store ground-truth centerline coordinates for validation

### 2. `src/data/loaders.py` — Data Loading

```python
def load_volume(path: str | Path | None = None) -> tuple[np.ndarray, dict]:
    """
    Auto-detect format and load:
    - None → synthetic phantom
    - .nii / .nii.gz → nibabel
    - directory of .dcm → pydicom, sorted by ImagePositionPatient
    - .npy → numpy
    
    Returns:
        volume: float64, normalized to [0, 1]
        metadata: dict with source info, spacing, affine, etc.
    """
```

### 3. `src/preprocessing/normalize.py`

```python
def normalize_intensity(volume: np.ndarray, method: str = "minmax") -> np.ndarray:
    """Methods: 'minmax', 'zscore', 'window' (CT windowing with level/width)"""

def apply_ct_window(volume: np.ndarray, level: float, width: float) -> np.ndarray:
    """Standard CT intensity windowing. E.g., level=300, width=700 for CTA vessels."""
```

### 4. `src/vesselness/filters.py` — Vesselness Enhancement

**This is the core "vessel-guided" step.**

```python
def compute_vesselness(
    volume: np.ndarray,
    method: str = "frangi",           # "frangi" | "sato" | "meijering"
    sigmas: np.ndarray | None = None, # Multi-scale sigmas, auto if None
    black_ridges: bool = False,       # True for dark vessels on bright background
) -> tuple[np.ndarray, dict]:
    """
    Apply multi-scale Hessian-based vesselness filter.
    
    How it works (conceptual):
    - At each voxel and each scale (sigma), compute the Hessian matrix
    - Eigenvalue analysis: tubes have λ1 ≈ 0, |λ2| ≈ |λ3| >> 0
    - Frangi combines eigenvalue ratios to produce a "vesselness" score
    - Maximum response across scales gives the final vesselness map
    
    Returns:
        vesselness: float64 array, same shape, normalized [0, 1]
        params: dict with method, sigmas, timing
    """
```

**Key implementation notes**:
- Default sigmas should cover expected vessel radii: `np.arange(1.0, 5.0, 0.5)` for typical CTA
- Use `skimage.filters.frangi`, `sato`, or `meijering` — they handle the Hessian computation internally
- Set `black_ridges=False` for bright vessels on dark background (standard CTA)
- The vesselness output should be normalized to [0, 1]

### 5. `src/segmentation/threshold.py`

```python
def segment_vessels(
    vesselness: np.ndarray,
    method: str = "otsu",             # "otsu" | "hysteresis" | "percentile"
    threshold_factor: float = 1.0,    # Multiplier for auto threshold
    min_object_size: int = 100,       # Remove components smaller than this
) -> tuple[np.ndarray, dict]:
    """
    Binarize the vesselness response into a vessel mask.
    
    Methods:
    - otsu: Automatic threshold via Otsu's method on non-zero voxels
    - hysteresis: Dual-threshold with connectivity (better for thin vessels)
    - percentile: Keep top N% of voxels (simple but effective)
    
    Post-processing:
    - Binary erosion → dilation (opening) to remove noise speckle
    - Remove small connected components < min_object_size
    
    Returns:
        mask: bool array, same shape as input
        params: dict with threshold value, voxel counts
    """
```

### 6. `src/centerline/skeleton.py` — Centerline via Skeletonization

```python
def extract_centerline_skeleton(
    vessel_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Extract the medial axis of the vessel segmentation using
    3D morphological thinning (skeletonization).
    
    Uses skimage.morphology.skeletonize_3d which implements
    the Lee et al. (1994) algorithm — iteratively peels surface
    voxels while preserving topology.
    
    Returns:
        skeleton: bool array (1-voxel-wide centerline)
        coords: (N, 3) int array of centerline voxel positions [z, y, x]
        stats: {skeleton_voxels, reduction_ratio, elapsed}
    """
```

### 7. `src/centerline/distance_field.py` — Refinement

```python
def refine_centerline_with_distance(
    vessel_mask: np.ndarray,
    skeleton: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Refine centerline positions using the distance transform.
    
    The skeleton from morphological thinning can be slightly off-center.
    This refines each skeleton voxel by shifting it to the local maximum
    of the distance transform (= true center of the vessel).
    
    Also annotates each centerline point with the local vessel radius
    (= distance transform value at that point).
    
    Returns:
        refined_coords: (N, 3) float array of sub-voxel centerline positions
        radii: (N,) float array of estimated vessel radius at each point
    """
```

### 8. `src/centerline/graph.py` — Topology Analysis

```python
def build_vessel_graph(
    skeleton: np.ndarray,
) -> dict:
    """
    Analyze skeleton topology using 26-connectivity neighborhood analysis.
    
    For each skeleton voxel, count neighbors:
    - 1 neighbor → endpoint
    - 2 neighbors → interior/continuation point
    - 3+ neighbors → branch/junction point
    
    Then trace individual vessel segments between branch points and endpoints.
    
    Returns:
        {
            "endpoints": (M, 3) array,
            "branch_points": (K, 3) array,
            "segments": list of (N_i, 3) arrays,  # ordered point sequences per segment
            "num_segments": int,
            "total_centerline_length_voxels": float,
            "segment_lengths": list[float],
        }
    
    Optionally (if networkx available):
        "graph": nx.Graph with nodes=junction/endpoints, edges=segments
    """
```

### 9. `src/visualization/projections.py`

```python
def create_summary_figure(
    volume: np.ndarray,
    vesselness: np.ndarray,
    vessel_mask: np.ndarray,
    skeleton: np.ndarray,
    centerline_coords: np.ndarray,
    graph_data: dict,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Generate a multi-panel summary figure:
    
    Layout (2 rows × 4 cols):
    Row 1: [Original MIP-Z] [Vesselness MIP-Z] [Segmentation MIP-Z] [Centerline MIP-Z]
    Row 2: [Original MIP-Y] [Vesselness MIP-Y] [Overlay: seg + skeleton] [3D scatter of centerline]
    
    MIP = Maximum Intensity Projection (np.max along axis)
    
    Color coding:
    - Segmentation: semi-transparent red overlay
    - Centerline: bright green/cyan line
    - Branch points: yellow dots
    - Endpoints: red dots
    """
```

### 10. `scripts/run_pipeline.py` — Main Entry Point

```python
"""
Usage:
    python scripts/run_pipeline.py                           # synthetic phantom demo
    python scripts/run_pipeline.py --input /path/to/data.nii.gz
    python scripts/run_pipeline.py --input /path/to/dicoms/
    python scripts/run_pipeline.py --config config.yaml

Full pipeline:
    1. Load volume (or generate phantom)
    2. Preprocess (normalize, optional denoise)
    3. Compute vesselness (Frangi multi-scale)
    4. Segment vessels (Otsu + morphological cleanup)
    5. Extract centerline (3D skeletonization)
    6. Refine centerline (distance transform)
    7. Build vessel graph (endpoints, branches, segments)
    8. Evaluate against ground truth (if available)
    9. Generate visualizations + metrics report
    10. Save all outputs
"""
```

### 11. `scripts/evaluate.py` — Validation

```python
def evaluate_centerline(
    extracted_coords: np.ndarray,
    ground_truth_coords: list[np.ndarray],
    tolerance_voxels: float = 2.0,
) -> dict:
    """
    Quantitative evaluation metrics:
    
    - Mean Centerline Distance (MCD): Average distance from extracted to nearest GT point
    - Overlap (OV): % of GT centerline within tolerance of extracted
    - Overlap until first error (OF): Longest contiguous match
    - Dice coefficient of vessel segmentation vs GT mask
    - Topology metrics: correct number of branches, endpoints
    
    Returns:
        {
            "mean_centerline_distance": float,
            "overlap_percent": float,
            "dice_coefficient": float,
            "num_extracted_segments": int,
            "num_gt_segments": int,
        }
    """
```

---

## `config.yaml` — Default Parameters

```yaml
# Vessel-Guided Centerline Extraction Configuration

data:
  source: "phantom"           # "phantom" | path to NIfTI/DICOM
  phantom:
    shape: [128, 128, 128]
    num_main_vessels: 3
    num_branches: 5
    noise_level: 0.05
    seed: 42

preprocessing:
  normalize: "minmax"         # "minmax" | "zscore" | "window"
  ct_window:                  # only used if normalize == "window"
    level: 300
    width: 700
  denoise:
    enabled: false
    method: "gaussian"        # "gaussian" | "median" | "nlm"
    sigma: 0.5

vesselness:
  method: "frangi"            # "frangi" | "sato" | "meijering"
  sigmas_start: 1.0
  sigmas_stop: 5.0
  sigmas_step: 0.5
  black_ridges: false

segmentation:
  method: "otsu"              # "otsu" | "hysteresis" | "percentile"
  threshold_factor: 1.0
  min_object_size: 100
  morphology:
    opening_radius: 1         # ball radius for binary opening

centerline:
  method: "skeleton"          # "skeleton" | "distance" (future: "vmtk")
  refine_with_distance: true

evaluation:
  tolerance_voxels: 2.0

output:
  save_dir: "outputs"
  save_figures: true
  save_metrics: true
  save_centerline_coords: true  # .npy and .json
  figure_dpi: 150
```

---

## Pipeline Flow (Pseudocode)

```python
def main(config):
    # 1. Load
    volume, meta = load_volume(config["data"]["source"])
    print_volume_stats(volume)
    
    # 2. Preprocess
    volume = normalize_intensity(volume, config["preprocessing"]["normalize"])
    if config["preprocessing"]["denoise"]["enabled"]:
        volume = denoise(volume, **config["preprocessing"]["denoise"])
    
    # 3. Vesselness
    vesselness, v_params = compute_vesselness(
        volume,
        method=config["vesselness"]["method"],
        sigmas=np.arange(
            config["vesselness"]["sigmas_start"],
            config["vesselness"]["sigmas_stop"],
            config["vesselness"]["sigmas_step"],
        ),
    )
    
    # 4. Segment
    vessel_mask, seg_params = segment_vessels(
        vesselness,
        method=config["segmentation"]["method"],
        threshold_factor=config["segmentation"]["threshold_factor"],
        min_object_size=config["segmentation"]["min_object_size"],
    )
    
    # 5. Centerline
    skeleton, coords, skel_stats = extract_centerline_skeleton(vessel_mask)
    
    # 6. Refine
    if config["centerline"]["refine_with_distance"]:
        refined_coords, radii = refine_centerline_with_distance(vessel_mask, skeleton)
    
    # 7. Graph
    graph_data = build_vessel_graph(skeleton)
    
    # 8. Evaluate (if ground truth available)
    if "gt_centerlines" in meta:
        metrics = evaluate_centerline(coords, meta["gt_centerlines"])
        print_metrics(metrics)
    
    # 9. Visualize
    fig = create_summary_figure(volume, vesselness, vessel_mask, skeleton, coords, graph_data)
    fig.savefig(output_dir / "pipeline_summary.png", dpi=config["output"]["figure_dpi"])
    
    # 10. Save
    save_results(coords, radii, graph_data, metrics, output_dir)
```

---

## Acceptance Criteria (Definition of Done)

1. **`python scripts/run_demo.py` runs without errors** and produces a summary figure showing the full pipeline stages
2. **Synthetic phantom test passes**: extracted centerline is within 2 voxels of ground truth for >80% of points
3. **Visualization clearly shows**: original volume → vesselness → segmentation → centerline overlay
4. **Graph analysis correctly identifies**: number of endpoints, branch points, and individual vessel segments
5. **All tests pass**: `pytest tests/ -v` with no failures
6. **Pipeline completes in <60 seconds** on the 128³ synthetic phantom on a standard laptop CPU
7. **Structured outputs**: centerline coordinates saved as `.npy` and `.json`, metrics as `.json`
8. **Clean code**: typed, documented, follows PEP 8, no hardcoded paths

---

## Stretch Goals (Nice-to-Have)

If time permits, in priority order:

1. **Interactive 3D viewer** using `napari` — load volume, overlay segmentation, draw centerline
2. **VMTK integration** — use `vmtk.vmtkCenterlines` for geodesic centerline computation (more robust than skeletonization for complex vessel trees)
3. **Curved Planar Reformation (CPR)** — sample the original volume along the extracted centerline to produce a "straightened" vessel view
4. **Web demo** — simple FastAPI + Three.js or Plotly viewer that shows the 3D centerline
5. **Real data validation** — download a public dataset (VESSEL12, IRCAD, coronary CTA from Zenodo) and run the pipeline on it
6. **GPU acceleration** — use CuPy for distance transforms and scipy operations on larger volumes

---

## Relevant Background for the Agent

### Frangi Vesselness Filter (the key algorithm)

The Frangi filter (Frangi et al., 1998) analyzes the eigenvalues (λ1, λ2, λ3, sorted by magnitude) of the Hessian matrix at each voxel:

- **Tubular structure** (vessel): λ1 ≈ 0, |λ2| ≈ |λ3| >> 0
- **Blob**: |λ1| ≈ |λ2| ≈ |λ3| >> 0
- **Plate/sheet**: λ1 ≈ 0, λ2 ≈ 0, |λ3| >> 0
- **Background**: λ1 ≈ λ2 ≈ λ3 ≈ 0

The vesselness function:

```
V(s) = (1 - exp(-R_A² / 2α²)) * exp(-R_B² / 2β²) * (1 - exp(-S² / 2c²))
```

Where:
- R_A = |λ2| / |λ3| (plate vs line discrimination)
- R_B = |λ1| / sqrt(|λ2 * λ3|) (blob vs line discrimination)  
- S = sqrt(λ1² + λ2² + λ3²) (background suppression / "structureness")

Multi-scale: compute at multiple sigma values (smoothing scales), take max.

### 3D Skeletonization

Lee et al. (1994) algorithm: iteratively removes surface voxels from the binary object while preserving:
- Topology (connectivity, number of holes)
- Endpoints
- The medial position

Result: 1-voxel-wide skeleton that represents the centerline.

### Distance Transform Refinement

The Euclidean distance transform assigns each voxel in the vessel mask its distance to the nearest boundary. The **ridge** of the distance transform (local maxima) corresponds to the true center of the vessel. We can shift skeleton voxels to the nearest local maximum of the distance transform for sub-voxel centerline accuracy.

---

## Common Pitfalls to Avoid

1. **Don't forget `black_ridges=False`** — CTA vessels are bright on dark, which is the non-default for some skimage filters
2. **Normalize the volume to [0,1]** before vesselness — Hessian eigenvalues are scale-dependent
3. **Use `ball(1)` not `np.ones((3,3,3))`** for morphological structuring elements in 3D — `ball` gives proper spherical connectivity
4. **`skeletonize_3d` expects uint8 input** — cast the boolean mask with `.astype(np.uint8)`
5. **Small phantom size matters** — 64³ is too small for meaningful vessel geometry, 128³ is the minimum, 256³ is better for demos
6. **Otsu thresholding on vesselness** — compute on non-zero voxels only, otherwise the huge background skews the threshold
7. **Memory** — 256³ float64 = ~134MB per volume. Keep intermediate results as float32 if memory is a concern
8. **Skeletonization artifacts** — can produce spurious branches. Post-process by pruning short branches (< N voxels from an endpoint)

