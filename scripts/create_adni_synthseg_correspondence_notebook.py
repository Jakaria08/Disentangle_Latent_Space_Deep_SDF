#!/usr/bin/env python3
"""Create an interactive notebook for raw, smooth, and correspondence SynthSeg meshes."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOK_OUTPUT_DIR = (
    REPO_ROOT / "examples" / "ADNI_1_L_No_MCI_large_strict_left" / "task3_longitudinal_prediction" / "notebooks"
)


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


def build_notebook(output_root: Path):
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {"display_name": "inr_sdf", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    }
    notebook["cells"] = [
        markdown(
            """# ADNI SynthSeg mesh inspection

This notebook shows each requested SynthSeg structure in three separate forms:

1. **Raw**: direct marching-cubes mesh from the segmentation label.
2. **Minimal smooth**: the conservative, lightly smoothed correspondence input.
3. **Correspondence**: Deformetrica output with shared vertex order and faces.

No data are modified by this notebook. By default it shows both requested structures for up to three selected IDs; change the display controls below for a different subset."""
        ),
        code(
            f'''from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import trimesh
from IPython.display import display
from plotly.subplots import make_subplots

OUTPUT_ROOT = Path({str(output_root)!r})
CONFIG_PATH = OUTPUT_ROOT / "run_configuration.json"
MANIFEST_PATH = OUTPUT_ROOT / "manifests" / "selected_scans.csv"
if not CONFIG_PATH.is_file() or not MANIFEST_PATH.is_file():
    raise FileNotFoundError("Run configuration or selected manifest is missing.")

CONFIG = json.loads(CONFIG_PATH.read_text())
manifest = pd.read_csv(MANIFEST_PATH, dtype={{"RID": str, "VISCODE": str, "scan_id": str}})
structure_specs = CONFIG["structures"]
structure_names = [item["name"] for item in structure_specs]
print(f"Output root: {{OUTPUT_ROOT}}")
display(pd.DataFrame(structure_specs))
display(manifest[[column for column in ["scan_id", "RID", "VISCODE", "visit_dx_3class", "AGE", "PTGENDER"] if column in manifest.columns]])
'''
        ),
        markdown("## Output quality-control tables"),
        code(
            '''validation_path = OUTPUT_ROOT / "reports" / "validation_summary.json"
if validation_path.is_file():
    validation = json.loads(validation_path.read_text())
    display(pd.DataFrame(validation.get("structures", {})).T)

qc_by_structure = {}
for structure in structure_names:
    path = OUTPUT_ROOT / structure / "mesh_qc.csv"
    qc_by_structure[structure] = pd.read_csv(path, dtype={"RID": str, "VISCODE": str, "scan_id": str}) if path.is_file() else pd.DataFrame()
    print(f"\\n{structure}: {len(qc_by_structure[structure])} records")
    if not qc_by_structure[structure].empty:
        wanted = [
            "scan_id", "diagnosis", "status", "mask_volume_mm3", "raw_mesh_volume_mm3",
            "smooth_mesh_volume_mm3", "smooth_vs_mask_volume_pct", "raw_mesh_components",
            "smooth_mesh_components", "raw_watertight", "smooth_watertight",
            "smooth_watertight_strategy", "smooth_effective_closing_iterations",
            "smooth_effective_fill_holes", "smooth_watertight_attempts",
        ]
        display(qc_by_structure[structure][[column for column in wanted if column in qc_by_structure[structure].columns]])

lineage_path = OUTPUT_ROOT / "reports" / "mesh_volume_lineage.csv"
if lineage_path.is_file():
    lineage = pd.read_csv(lineage_path, dtype={"RID": str, "VISCODE": str, "scan_id": str})
    wanted = [
        "scan_id", "structure", "mask_volume_mm3", "raw_mesh_volume_mm3",
        "smooth_mesh_volume_mm3", "global_scale_factor", "rigid_linear_scale_factor",
        "correspondence_rescale_factor", "final_volume_mm3",
        "final_vs_smooth_volume_error_pct", "range_linear_scale_factor",
    ]
    print("\\nVolume and coordinate lineage")
    display(lineage[[column for column in wanted if column in lineage.columns]])
'''
        ),
        markdown("## Inspect both structures for the selected pilot IDs"),
        code(
            '''# The pilot displays every selected ID (at most three) and every requested structure.
# For a full run, change MAX_SCANS_TO_SHOW or SCAN_IDS deliberately before executing the next cell.
MAX_SCANS_TO_SHOW = 3
SCAN_IDS = manifest["scan_id"].astype(str).head(MAX_SCANS_TO_SHOW).tolist()
STRUCTURES_TO_SHOW = structure_names
PRIMARY_STRUCTURE = structure_names[0]  # Used by the correspondence overlay below.

if not SCAN_IDS:
    raise ValueError("The selected manifest has no scan IDs.")
if not set(STRUCTURES_TO_SHOW).issubset(structure_names):
    raise ValueError("STRUCTURES_TO_SHOW contains an unknown structure.")

def mesh_path(structure, scan_id, stage):
    base = OUTPUT_ROOT / structure
    if stage == "raw":
        return base / "raw_ply" / f"{scan_id}.ply"
    if stage == "minimal smooth":
        return base / "minimal_smooth_ply" / f"{scan_id}.ply"
    if stage == "correspondence mm":
        return base / "minimal_smooth_correspondence" / "final_ply_mm" / f"{scan_id}.ply"
    raise ValueError(stage)

for scan_id in SCAN_IDS:
    print(f"\\n{scan_id}")
    for structure in STRUCTURES_TO_SHOW:
        for stage in ["raw", "minimal smooth", "correspondence mm"]:
            path = mesh_path(structure, scan_id, stage)
            print(f"  {structure:26s} | {stage:18s} | exists={path.is_file()}")
'''
        ),
        code(
            '''def stats(mesh):
    return {
        "vertices": len(mesh.vertices), "faces": len(mesh.faces),
        "components": len(mesh.split(only_watertight=False)), "watertight": bool(mesh.is_watertight),
        "euler_number": int(mesh.euler_number), "surface_area_mm2": float(mesh.area),
        "mesh_volume_mm3": abs(float(mesh.volume)),
    }

def add_mesh(figure, mesh, name, color, row, column, opacity=1.0, index_colors=False):
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    options = dict(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], name=name, opacity=opacity,
        flatshading=False, showscale=False,
        lighting=dict(ambient=0.42, diffuse=0.75, specular=0.16, roughness=0.78),
    )
    if index_colors:
        options.update(intensity=np.arange(len(vertices)), colorscale="Turbo")
    else:
        options.update(color=color)
    figure.add_trace(go.Mesh3d(**options), row=row, col=column)

stages = ["raw", "minimal smooth", "correspondence mm"]
colors = {"raw": "#8E44AD", "minimal smooth": "#2E86DE", "correspondence mm": "#D35400"}
for scan_id in SCAN_IDS:
    rows = []
    figure = make_subplots(
        rows=len(STRUCTURES_TO_SHOW), cols=3,
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}] for _ in STRUCTURES_TO_SHOW],
        subplot_titles=[title for _ in STRUCTURES_TO_SHOW for title in ("Raw", "Minimal smooth", "Correspondence (mm)")],
        row_titles=[item["display_name"] for item in structure_specs if item["name"] in STRUCTURES_TO_SHOW],
    )
    for row_index, structure in enumerate(STRUCTURES_TO_SHOW, start=1):
        for column, stage in enumerate(stages, start=1):
            path = mesh_path(structure, scan_id, stage)
            if not path.is_file():
                continue
            mesh = trimesh.load(path, force="mesh", process=False)
            rows.append({"scan_id": scan_id, "structure": structure, "stage": stage, **stats(mesh)})
            add_mesh(figure, mesh, f"{structure} | {stage}", colors[stage], row_index, column)
    display(pd.DataFrame(rows))
    for index in range(1, len(STRUCTURES_TO_SHOW) * 3 + 1):
        scene = "scene" if index == 1 else f"scene{index}"
        figure.update_layout(**{scene: dict(aspectmode="data", xaxis_title="X", yaxis_title="Y", zaxis_title="Z")})
    figure.update_layout(
        title=f"Raw, minimal-smooth, and correspondence meshes | {scan_id}",
        width=1550, height=480 * len(STRUCTURES_TO_SHOW), template="plotly_white",
        margin=dict(l=0, r=0, t=65, b=0),
    )
    figure.show()
'''
        ),
        markdown("## Correspondence overlay"),
        code(
            '''# Full runs may contain thousands of meshes. Keep this small for interactive viewing.
MAX_SCANS = 3
overlay_rows = manifest[["scan_id", "visit_dx_3class"]].head(MAX_SCANS)
diagnosis_colors = {"CN": "#2E86DE", "AD": "#C0392B", "MCI": "#F39C12"}
overlay = go.Figure()
overlay_stats = []
for _, row in overlay_rows.iterrows():
    scan_id = str(row["scan_id"])
    path = OUTPUT_ROOT / PRIMARY_STRUCTURE / "minimal_smooth_correspondence" / "final_ply" / f"{scan_id}.ply"
    if not path.is_file():
        continue
    mesh = trimesh.load(path, force="mesh", process=False)
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    diagnosis = str(row["visit_dx_3class"])
    overlay.add_trace(go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], name=f"{scan_id} | {diagnosis}",
        color=diagnosis_colors.get(diagnosis, "#6C757D"), opacity=0.45, flatshading=False, showscale=False,
    ))
    overlay_stats.append({"scan_id": scan_id, "diagnosis": diagnosis, **stats(mesh)})
display(pd.DataFrame(overlay_stats))
overlay.update_layout(
    title=f"Correspondence overlay: {PRIMARY_STRUCTURE}", width=900, height=700, template="plotly_white",
    scene=dict(aspectmode="data", xaxis_title="X", yaxis_title="Y", zaxis_title="Z"), margin=dict(l=0, r=0, t=55, b=0),
)
overlay.show()
'''
        ),
        markdown("## Direct point-correspondence check"),
        code(
            '''final_paths = sorted((OUTPUT_ROOT / PRIMARY_STRUCTURE / "minimal_smooth_correspondence" / "final_ply").glob("*.ply"))
if len(final_paths) < 2:
    print("At least two correspondence meshes are needed for this check.")
else:
    reference = trimesh.load(final_paths[0], force="mesh", process=False)
    rows = []
    index_figure = go.Figure()
    for path in final_paths[:min(3, len(final_paths))]:
        mesh = trimesh.load(path, force="mesh", process=False)
        rows.append({
            "scan_id": path.stem, "vertices": len(mesh.vertices), "faces": len(mesh.faces),
            "faces_identical_to_first": bool(np.array_equal(mesh.faces, reference.faces)),
        })
        vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
        index_figure.add_trace(go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], intensity=np.arange(len(vertices)),
            colorscale="Turbo", showscale=False, opacity=0.55, name=path.stem,
        ))
    display(pd.DataFrame(rows))
    index_figure.update_layout(
        title=f"Vertex-index colors: {PRIMARY_STRUCTURE}", width=900, height=700, template="plotly_white",
        scene=dict(aspectmode="data"), margin=dict(l=0, r=0, t=55, b=0),
    )
    index_figure.show()
'''
        ),
    ]
    for index, cell in enumerate(notebook["cells"]):
        cell["id"] = f"cell-{index:02d}"
    return notebook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--notebook-output",
        type=Path,
        default=None,
        help="Destination .ipynb path; default is task3_longitudinal_prediction/notebooks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_path = (
        args.notebook_output.resolve()
        if args.notebook_output is not None
        else DEFAULT_NOTEBOOK_OUTPUT_DIR / f"visualize_synthseg_meshes_{output_root.name}.ipynb"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build_notebook(output_root), output_path)
    print(output_path)


if __name__ == "__main__":
    main()
