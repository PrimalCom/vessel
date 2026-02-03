"""Metrics reporting and centerline data persistence utilities."""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path


def generate_report(
    metrics: dict,
    graph_data: dict,
    save_dir: str | Path,
) -> dict:
    """Compile pipeline metrics into JSON and a human-readable text summary.

    Parameters
    ----------
    metrics : dict
        Arbitrary metric dictionary produced by the evaluation step.  Common
        keys include ``"total_centerline_length"``,
        ``"mean_centerline_distance"``, ``"overlap_percentage"``, and
        ``"dice"``.
    graph_data : dict
        Graph-level information.  Expected keys:
        - ``"segments"``      : list of segment arrays
        - ``"endpoints"``     : array-like of endpoint coords
        - ``"branch_points"`` : array-like of branch-point coords
    save_dir : str | Path
        Root output directory.  Files are written into ``save_dir/results/``.

    Returns
    -------
    dict
        A combined report dictionary containing both the raw *metrics* and
        the derived *summary* statistics.
    """
    save_dir = Path(save_dir)
    results_dir = save_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---- Derive summary statistics ------------------------------------
    segments = graph_data.get("segments", [])
    endpoints = np.asarray(graph_data.get("endpoints", [])).reshape(-1, 3) if len(graph_data.get("endpoints", [])) else np.empty((0, 3))
    branch_points = np.asarray(graph_data.get("branch_points", [])).reshape(-1, 3) if len(graph_data.get("branch_points", [])) else np.empty((0, 3))

    num_segments = len(segments)
    num_endpoints = int(endpoints.shape[0])
    num_branch_points = int(branch_points.shape[0])
    total_length = float(metrics.get("total_centerline_length", 0.0))
    mean_distance = metrics.get("mean_centerline_distance", None)
    overlap_pct = metrics.get("overlap_percentage", None)
    dice = metrics.get("dice", None)

    summary = {
        "num_segments": num_segments,
        "num_endpoints": num_endpoints,
        "num_branch_points": num_branch_points,
        "total_centerline_length": total_length,
        "mean_centerline_distance": mean_distance,
        "overlap_percentage": overlap_pct,
        "dice": dice,
    }

    report = {
        "metrics": metrics,
        "summary": summary,
    }

    # ---- Save JSON -----------------------------------------------------
    def _default(obj):
        """JSON serialiser for numpy types."""
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    metrics_path = results_dir / "metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(report, fh, indent=2, default=_default)

    # ---- Save text summary ---------------------------------------------
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("  Vessel Centerline Extraction — Summary Report")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"  Segments        : {num_segments}")
    lines.append(f"  Endpoints       : {num_endpoints}")
    lines.append(f"  Branch points   : {num_branch_points}")
    lines.append(f"  Total length    : {total_length:.2f} voxels")
    lines.append("")

    if mean_distance is not None:
        lines.append(f"  Mean centerline distance : {mean_distance:.4f}")
    if overlap_pct is not None:
        lines.append(f"  Overlap percentage       : {overlap_pct:.2f} %")
    if dice is not None:
        lines.append(f"  Dice coefficient         : {dice:.4f}")

    lines.append("")
    lines.append("=" * 60)
    lines.append("")

    summary_text = "\n".join(lines)
    summary_path = results_dir / "summary.txt"
    summary_path.write_text(summary_text)

    return report


def save_centerline_data(
    coords: np.ndarray,
    radii: np.ndarray | None,
    graph_data: dict,
    save_dir: str | Path,
) -> None:
    """Persist centerline coordinates, radii, and graph summary to disk.

    Parameters
    ----------
    coords : np.ndarray
        (N, 3) centerline coordinates in (z, y, x) order.
    radii : np.ndarray | None
        (N,) estimated vessel radii at each centerline point.
    graph_data : dict
        Graph-level information (segments, endpoints, branch_points).
    save_dir : str | Path
        Root output directory.  Files are written into ``save_dir/results/``.
    """
    save_dir = Path(save_dir)
    results_dir = save_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---- Centerline coordinates ----------------------------------------
    np.save(str(results_dir / "centerline_coords.npy"), coords)

    coords_list = coords.tolist()  # list of [z, y, x] lists
    with open(results_dir / "centerline_coords.json", "w") as fh:
        json.dump(coords_list, fh)

    # ---- Radii ---------------------------------------------------------
    if radii is not None:
        np.save(str(results_dir / "centerline_radii.npy"), radii)

    # ---- Graph data summary --------------------------------------------
    def _serialise(obj):
        """Convert numpy types for JSON encoding."""
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    segments = graph_data.get("segments", [])
    endpoints = graph_data.get("endpoints", [])
    branch_points = graph_data.get("branch_points", [])

    graph_summary = {
        "num_segments": len(segments),
        "num_endpoints": len(np.asarray(endpoints).reshape(-1, 3)) if len(endpoints) else 0,
        "num_branch_points": len(np.asarray(branch_points).reshape(-1, 3)) if len(branch_points) else 0,
        "endpoints": np.asarray(endpoints).tolist() if len(endpoints) else [],
        "branch_points": np.asarray(branch_points).tolist() if len(branch_points) else [],
        "segment_lengths": [
            int(np.asarray(s).shape[0]) for s in segments
        ],
    }

    with open(results_dir / "graph_data.json", "w") as fh:
        json.dump(graph_summary, fh, indent=2, default=_serialise)


def generate_html_report(
    volume_shape: tuple,
    config: dict,
    seg_params: dict,
    skel_stats: dict,
    graph_data: dict,
    metrics: dict,
    mean_radius: float | None,
    input_source: str,
    config_path: str,
    save_dir: str | Path,
    figure_rel_path: str = "figures/summary.png",
) -> Path:
    """Generate an HTML report summarising the pipeline run.

    Parameters
    ----------
    volume_shape : tuple
        Shape of the input volume.
    config : dict
        Full pipeline configuration dictionary.
    seg_params : dict
        Segmentation parameters returned by ``segment_vessels``.
    skel_stats : dict
        Skeleton extraction statistics.
    graph_data : dict
        Graph-level information (segments, endpoints, branch_points).
    metrics : dict
        Evaluation metrics (centerline length, dice, etc.).
    mean_radius : float or None
        Mean vessel radius from distance-field refinement.
    input_source : str
        Description of the input data source.
    config_path : str
        Path to the config file used.
    save_dir : str or Path
        Root output directory.  The HTML file is written to ``save_dir/report.html``.
    figure_rel_path : str
        Relative path (from *save_dir*) to the summary figure image.

    Returns
    -------
    Path
        Path to the written HTML file.
    """
    import base64
    import statistics

    save_dir = Path(save_dir)

    # --- Gather numbers ---------------------------------------------------
    seg_voxels = seg_params.get("voxel_count", 0)
    total_voxels = seg_params.get("total_voxels", 1)
    vol_frac = 100.0 * seg_voxels / total_voxels if total_voxels else 0
    skel_voxels = skel_stats.get("skeleton_voxels", 0)

    segments = graph_data.get("segments", [])
    seg_lengths = [int(np.asarray(s).shape[0]) for s in segments]
    num_segments = len(segments)
    endpoints = np.asarray(graph_data.get("endpoints", []))
    num_endpoints = int(endpoints.reshape(-1, 3).shape[0]) if endpoints.size else 0
    branch_points = np.asarray(graph_data.get("branch_points", []))
    num_branch_points = int(branch_points.reshape(-1, 3).shape[0]) if branch_points.size else 0
    total_length = metrics.get("total_centerline_length", 0.0)
    mean_rad_str = f"{mean_radius:.2f}" if mean_radius is not None else "N/A"
    dice = metrics.get("dice")
    overlap = metrics.get("overlap_percentage")
    mean_cl_dist = metrics.get("mean_centerline_distance")

    # Segment length stats
    if seg_lengths:
        seg_lengths_sorted = sorted(seg_lengths)
        seg_min = min(seg_lengths)
        seg_max = max(seg_lengths)
        seg_mean = statistics.mean(seg_lengths)
        seg_median = statistics.median(seg_lengths)
    else:
        seg_lengths_sorted = []
        seg_min = seg_max = seg_mean = seg_median = 0

    # Config values for display
    pre_cfg = config.get("preprocessing", {})
    dn_cfg = pre_cfg.get("denoise", {})
    vess_cfg = config.get("vesselness", {})
    seg_cfg = config.get("segmentation", {})
    morph_cfg = seg_cfg.get("morphology", {})

    # Embed figure as base64 so the HTML is self-contained
    figure_path = save_dir / figure_rel_path
    img_tag = ""
    if figure_path.exists():
        img_data = base64.b64encode(figure_path.read_bytes()).decode("ascii")
        img_tag = f'<img src="data:image/png;base64,{img_data}" alt="Pipeline summary figure">'
    else:
        img_tag = '<p style="color:var(--muted);">Summary figure not found.</p>'

    # JS array for bar chart
    js_lengths = json.dumps(seg_lengths_sorted)

    # Shape string
    shape_str = " &times; ".join(str(d) for d in volume_shape)

    # Evaluation rows (only shown if available)
    eval_rows = ""
    if dice is not None:
        eval_rows += f"""
    <div class="metric-card">
      <div class="value">{dice:.4f}</div>
      <div class="label">Dice Coefficient</div>
    </div>"""
    if overlap is not None:
        eval_rows += f"""
    <div class="metric-card">
      <div class="value">{overlap:.2f}%</div>
      <div class="label">Overlap</div>
    </div>"""
    if mean_cl_dist is not None:
        eval_rows += f"""
    <div class="metric-card">
      <div class="value">{mean_cl_dist:.2f}</div>
      <div class="label">Mean CL Distance (vx)</div>
    </div>"""

    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Vessel Centerline Extraction Report</title>
<style>
  :root {{
    --bg: #0f1117; --surface: #1a1d2e; --border: #2a2d3e;
    --text: #e1e4ed; --muted: #8b8fa3; --accent: #6c8cff;
    --green: #34d399; --amber: #fbbf24; --red: #f87171;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg); color: var(--text); line-height:1.6; padding:2rem;
  }}
  .container {{ max-width:1200px; margin:0 auto; }}
  header {{
    text-align:center; margin-bottom:2.5rem;
    padding-bottom:1.5rem; border-bottom:1px solid var(--border);
  }}
  header h1 {{ font-size:1.75rem; font-weight:600; margin-bottom:0.25rem; }}
  header p {{ color:var(--muted); font-size:0.95rem; }}
  .config-section {{
    background:var(--surface); border:1px solid var(--border);
    border-radius:10px; padding:1.5rem; margin-bottom:2rem;
  }}
  .config-section h2 {{ font-size:1.1rem; margin-bottom:1rem; color:var(--accent); }}
  .config-grid {{
    display:grid; grid-template-columns:repeat(auto-fit, minmax(280px,1fr)); gap:1rem;
  }}
  .config-card {{
    background:var(--bg); border:1px solid var(--border);
    border-radius:8px; padding:1rem;
  }}
  .config-card h3 {{
    font-size:0.8rem; text-transform:uppercase; letter-spacing:0.05em;
    color:var(--muted); margin-bottom:0.5rem;
  }}
  .config-card table {{ width:100%; }}
  .config-card td {{ padding:0.2rem 0; font-size:0.85rem; }}
  .config-card td:first-child {{ color:var(--muted); padding-right:1rem; white-space:nowrap; }}
  .config-card td:last-child {{
    font-family:'SF Mono','Fira Code',monospace; color:var(--green); text-align:right;
  }}
  .metrics-row {{
    display:grid; grid-template-columns:repeat(auto-fit, minmax(160px,1fr));
    gap:1rem; margin-bottom:2rem;
  }}
  .metric-card {{
    background:var(--surface); border:1px solid var(--border);
    border-radius:10px; padding:1.25rem; text-align:center;
  }}
  .metric-card .value {{
    font-size:1.8rem; font-weight:700; color:var(--green);
    font-family:'SF Mono','Fira Code',monospace;
  }}
  .metric-card .label {{
    font-size:0.78rem; color:var(--muted); text-transform:uppercase;
    letter-spacing:0.04em; margin-top:0.25rem;
  }}
  .image-section {{ margin-bottom:2rem; }}
  .image-section h2 {{ font-size:1.1rem; margin-bottom:1rem; color:var(--accent); }}
  .summary-img-wrapper {{
    background:var(--surface); border:1px solid var(--border);
    border-radius:10px; padding:1rem; text-align:center;
  }}
  .summary-img-wrapper img {{ max-width:100%; height:auto; border-radius:6px; }}
  .summary-img-wrapper .caption {{
    margin-top:0.75rem; font-size:0.82rem; color:var(--muted);
  }}
  .panel-guide {{
    display:grid; grid-template-columns:repeat(4,1fr); gap:0.5rem; margin-top:1rem;
  }}
  .panel-guide .panel {{
    background:var(--bg); border:1px solid var(--border);
    border-radius:6px; padding:0.6rem; text-align:center;
  }}
  .panel-guide .panel .title {{ font-size:0.78rem; font-weight:600; color:var(--text); }}
  .panel-guide .panel .desc {{ font-size:0.7rem; color:var(--muted); margin-top:0.15rem; }}
  .segment-section {{
    background:var(--surface); border:1px solid var(--border);
    border-radius:10px; padding:1.5rem; margin-bottom:2rem;
  }}
  .segment-section h2 {{ font-size:1.1rem; margin-bottom:1rem; color:var(--accent); }}
  .bar-chart {{
    display:flex; align-items:flex-end; gap:2px; height:100px; padding:0.5rem 0;
  }}
  .bar {{
    flex:1; background:var(--accent); border-radius:2px 2px 0 0;
    min-width:2px; opacity:0.8; transition:opacity 0.2s;
  }}
  .bar:hover {{ opacity:1; }}
  .bar-labels {{
    display:flex; justify-content:space-between;
    font-size:0.7rem; color:var(--muted); margin-top:0.25rem;
  }}
  .dist-stats {{
    display:grid; grid-template-columns:repeat(auto-fit, minmax(120px,1fr));
    gap:0.75rem; margin-top:1rem;
  }}
  .dist-stat {{ text-align:center; }}
  .dist-stat .val {{
    font-family:'SF Mono','Fira Code',monospace;
    font-size:1rem; font-weight:600; color:var(--text);
  }}
  .dist-stat .lbl {{ font-size:0.7rem; color:var(--muted); text-transform:uppercase; }}
  footer {{
    text-align:center; padding-top:1.5rem; border-top:1px solid var(--border);
    color:var(--muted); font-size:0.78rem;
  }}
</style>
</head>
<body>
<div class="container">

  <header>
    <h1>Vessel Centerline Extraction Report</h1>
    <p>Input: {input_source} &mdash; Volume shape: {shape_str}</p>
    <p>Config: {config_path}</p>
  </header>

  <div class="config-section">
    <h2>Pipeline Configuration</h2>
    <div class="config-grid">
      <div class="config-card">
        <h3>Preprocessing</h3>
        <table>
          <tr><td>Normalize</td><td>{pre_cfg.get("normalize", "minmax")}</td></tr>
          <tr><td>Denoise</td><td>{"enabled" if dn_cfg.get("enabled") else "disabled"}</td></tr>
          <tr><td>Denoise method</td><td>{dn_cfg.get("method", "N/A")}</td></tr>
          <tr><td>Denoise sigma</td><td>{dn_cfg.get("sigma", "N/A")}</td></tr>
        </table>
      </div>
      <div class="config-card">
        <h3>Vesselness</h3>
        <table>
          <tr><td>Method</td><td>{vess_cfg.get("method", "frangi")}</td></tr>
          <tr><td>Sigmas</td><td>{vess_cfg.get("sigmas_start", 1.0)} &ndash; {vess_cfg.get("sigmas_stop", 5.0)} (step {vess_cfg.get("sigmas_step", 0.5)})</td></tr>
          <tr><td>Black ridges</td><td>{vess_cfg.get("black_ridges", False)}</td></tr>
        </table>
      </div>
      <div class="config-card">
        <h3>Segmentation</h3>
        <table>
          <tr><td>Method</td><td>{seg_cfg.get("method", "otsu")}</td></tr>
          <tr><td>Threshold factor</td><td>{seg_cfg.get("threshold_factor", 1.0)}</td></tr>
          <tr><td>Min object size</td><td>{seg_cfg.get("min_object_size", 100)}</td></tr>
          <tr><td>Opening radius</td><td>{morph_cfg.get("opening_radius", 1)}</td></tr>
        </table>
      </div>
    </div>
  </div>

  <div class="metrics-row">
    <div class="metric-card">
      <div class="value">{seg_voxels:,}</div>
      <div class="label">Segmented Voxels</div>
    </div>
    <div class="metric-card">
      <div class="value">{vol_frac:.2f}%</div>
      <div class="label">Volume Fraction</div>
    </div>
    <div class="metric-card">
      <div class="value">{skel_voxels:,}</div>
      <div class="label">Skeleton Voxels</div>
    </div>
    <div class="metric-card">
      <div class="value">{num_segments:,}</div>
      <div class="label">Vessel Segments</div>
    </div>
    <div class="metric-card">
      <div class="value">{num_endpoints:,}</div>
      <div class="label">Endpoints</div>
    </div>
    <div class="metric-card">
      <div class="value">{num_branch_points:,}</div>
      <div class="label">Branch Points</div>
    </div>
    <div class="metric-card">
      <div class="value">{total_length:,.0f}</div>
      <div class="label">Centerline Length (vx)</div>
    </div>
    <div class="metric-card">
      <div class="value">{mean_rad_str}</div>
      <div class="label">Mean Radius (vx)</div>
    </div>{eval_rows}
  </div>

  <div class="image-section">
    <h2>Pipeline Output Visualization</h2>
    <div class="summary-img-wrapper">
      {img_tag}
      <div class="caption">Maximum Intensity Projections (Z and Y), vesselness response, segmentation mask, skeleton overlay, and 3D centerline scatter.</div>
    </div>
    <div class="panel-guide">
      <div class="panel">
        <div class="title">Original MIP</div>
        <div class="desc">Raw input, Z &amp; Y projections</div>
      </div>
      <div class="panel">
        <div class="title">Vesselness MIP</div>
        <div class="desc">Multi-scale filter response</div>
      </div>
      <div class="panel">
        <div class="title">Segmentation</div>
        <div class="desc">Binary mask &amp; skeleton overlay</div>
      </div>
      <div class="panel">
        <div class="title">3D Centerline</div>
        <div class="desc">Extracted graph with branch points</div>
      </div>
    </div>
  </div>

  <div class="segment-section">
    <h2>Segment Length Distribution</h2>
    <div class="bar-chart" id="barChart"></div>
    <div class="bar-labels">
      <span>{seg_min} vx</span>
      <span>Segment index (sorted by length)</span>
      <span>{seg_max} vx</span>
    </div>
    <div class="dist-stats">
      <div class="dist-stat"><div class="val">{num_segments}</div><div class="lbl">Total Segments</div></div>
      <div class="dist-stat"><div class="val">{seg_min}</div><div class="lbl">Min Length</div></div>
      <div class="dist-stat"><div class="val">{seg_max}</div><div class="lbl">Max Length</div></div>
      <div class="dist-stat"><div class="val">{seg_mean:.1f}</div><div class="lbl">Mean Length</div></div>
      <div class="dist-stat"><div class="val">{seg_median:.0f}</div><div class="lbl">Median Length</div></div>
    </div>
  </div>

  <footer>
    Generated by Vessel Centerline Extraction Pipeline &mdash; {config_path}
  </footer>

</div>
<script>
  const lengths = {js_lengths};
  const max = Math.max(...lengths, 1);
  const chart = document.getElementById('barChart');
  lengths.forEach(l => {{
    const bar = document.createElement('div');
    bar.className = 'bar';
    bar.style.height = (l / max * 100) + '%';
    bar.title = l + ' voxels';
    chart.appendChild(bar);
  }});
</script>
</body>
</html>
"""

    report_path = save_dir / "report.html"
    report_path.write_text(html)
    return report_path
