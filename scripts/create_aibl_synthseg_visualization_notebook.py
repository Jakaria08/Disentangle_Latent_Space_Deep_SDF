#!/usr/bin/env python3
"""Create an inspection notebook for the AIBL SynthSeg segmentations.

This mirrors the ADNI notebook ``data/1001_bl/visualize_synthseg_segmentation.ipynb``
but targets the AIBL cohort, where two things differ:

*   AIBL has no ``adni_hipp_surfs`` reference meshes, so that section is replaced
    by a cross-check against the FreeSurfer 5.3 ``aseg.mgz`` shipped inside each
    AIBL subject tar.  That is an independent segmentation of the same scan and
    is the strongest QC available for this cohort.
*   The pilot contains repeat visits for one subject, so a longitudinal section
    compares timepoints of the same subject directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "AIBL"
DEFAULT_NOTEBOOK_PATH = DEFAULT_DATA_ROOT / "visualize_aibl_synthseg_segmentation.ipynb"
DEFAULT_PRIMARY_SCAN = "1001_bl"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text)


def code(text: str):
    return nbf.v4.new_code_cell(text)


def build_notebook(data_root: Path, primary_scan: str):
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {"display_name": "inr_sdf", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    }

    cells = []

    cells.append(
        markdown(
            f"""# AIBL SynthSeg Segmentation Inspection

This notebook inspects the AIBL SynthSeg label maps produced by `data/aibl_synthseg.sh`.
It summarizes the labels present, shows orthogonal slice views, and builds raw and
minimally smoothed 3D meshes for the hippocampi (labels 17, 53) and the lateral
ventricles (labels 4, 43).

The smoothing path reuses the same conventions as the ADNI notebook and as
`scripts/adni_synthseg_minimal_correspondence.py`, so mesh statistics here are
directly comparable to the ADNI cohort.

Two sections are specific to AIBL:

*   **FreeSurfer aseg cross-validation** - each AIBL tar carries `mri/aseg.mgz` from
    the original FreeSurfer 5.3 run.  That is an independent segmentation of the same
    scan, which ADNI's pipeline did not have available here.
*   **Longitudinal consistency** - repeat visits of the same subject are compared
    directly, which is the cheapest biological plausibility check available.

Primary detailed scan: `{primary_scan}`.
"""
        )
    )

    cells.append(
        code(
            f'''from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import trimesh
from IPython.display import display
from plotly.subplots import make_subplots
from scipy import ndimage
from skimage import measure

plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["image.cmap"] = "tab20"
np.set_printoptions(precision=3, suppress=True)
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 180)

DATA_ROOT = Path("{data_root}")
SEGMENTATIONS_DIR = DATA_ROOT / "aibl_synthseg" / "segmentations"
TAR_DIR = DATA_ROOT
PRIMARY_SCAN = "{primary_scan}"

DATA_PATH = SEGMENTATIONS_DIR / PRIMARY_SCAN / f"{{PRIMARY_SCAN}}.synthseg.mgz"
OUTPUT_DIR = DATA_ROOT / "synthseg_meshes" / PRIMARY_SCAN
WRITE_MESH_FILES = True

TARGET_LABELS = [17, 53]
VENTRICLE_LABELS = [4, 43]

# These match the ADNI notebook so that mesh statistics stay comparable.
GAUSSIAN_SIGMA = 0.5
CLOSING_ITERATIONS = 1
CROP_MARGIN = 4

LABEL_NAMES = {{
    0: "Background",
    2: "Left-Cerebral-White-Matter",
    3: "Left-Cerebral-Cortex",
    4: "Left-Lateral-Ventricle",
    5: "Left-Inf-Lat-Vent",
    7: "Left-Cerebellum-White-Matter",
    8: "Left-Cerebellum-Cortex",
    10: "Left-Thalamus-Proper",
    11: "Left-Caudate",
    12: "Left-Putamen",
    13: "Left-Pallidum",
    14: "3rd-Ventricle",
    15: "4th-Ventricle",
    16: "Brain-Stem",
    17: "Left-Hippocampus",
    18: "Left-Amygdala",
    24: "CSF",
    26: "Left-Accumbens-area",
    28: "Left-VentralDC",
    41: "Right-Cerebral-White-Matter",
    42: "Right-Cerebral-Cortex",
    43: "Right-Lateral-Ventricle",
    44: "Right-Inf-Lat-Vent",
    46: "Right-Cerebellum-White-Matter",
    47: "Right-Cerebellum-Cortex",
    49: "Right-Thalamus-Proper",
    50: "Right-Caudate",
    51: "Right-Putamen",
    52: "Right-Pallidum",
    53: "Right-Hippocampus",
    54: "Right-Amygdala",
    58: "Right-Accumbens-area",
    60: "Right-VentralDC",
}}

SEGMENTATION_PATHS = sorted(SEGMENTATIONS_DIR.glob("*/*.synthseg.mgz"))
if not SEGMENTATION_PATHS:
    raise FileNotFoundError(f"No SynthSeg volumes found under {{SEGMENTATIONS_DIR}}")
if not DATA_PATH.is_file():
    raise FileNotFoundError(f"Primary scan is missing: {{DATA_PATH}}")

print(f"Found {{len(SEGMENTATION_PATHS)}} AIBL SynthSeg volumes")
for path in SEGMENTATION_PATHS:
    print(f"  {{path.parent.name}}")
'''
        )
    )

    cells.append(markdown("## Load the primary SynthSeg volume and summarize labels"))
    cells.append(
        code(
            '''image = nib.load(str(DATA_PATH))
data = np.asanyarray(image.dataobj).astype(np.int16)
spacing = image.header.get_zooms()[:3]
voxel_volume_mm3 = abs(float(np.linalg.det(image.affine[:3, :3])))

labels, counts = np.unique(data, return_counts=True)
label_summary = pd.DataFrame(
    {
        "label": labels.astype(int),
        "name": [LABEL_NAMES.get(int(label), f"Label-{int(label)}") for label in labels],
        "voxel_count": counts.astype(int),
        "volume_mm3": counts.astype(float) * voxel_volume_mm3,
    }
)
label_summary = label_summary.sort_values(["voxel_count", "label"], ascending=[False, True]).reset_index(drop=True)
present_labels = set(label_summary["label"].tolist())

target_labels = [label for label in TARGET_LABELS if label in present_labels]
missing_targets = [label for label in TARGET_LABELS if label not in present_labels]
if missing_targets:
    print(f"Target labels missing from this volume: {missing_targets}")
if not target_labels:
    raise ValueError("None of the requested target labels are present in the SynthSeg volume.")

summary_rows = pd.DataFrame(
    [
        ("scan_id", PRIMARY_SCAN),
        ("shape", data.shape),
        ("dtype", str(data.dtype)),
        ("spacing_mm", spacing),
        ("voxel_volume_mm3", voxel_volume_mm3),
        ("nonzero_voxels", int((data != 0).sum())),
        ("unique_labels", int(len(labels))),
    ],
    columns=["item", "value"],
)
display(summary_rows)
display(label_summary[label_summary["label"] != 0].reset_index(drop=True))
'''
        )
    )

    cells.append(markdown("## Raw segmentation slice views"))
    cells.append(
        code(
            '''def center_from_mask(mask: np.ndarray) -> tuple[int, int, int]:
    if mask.any():
        center = ndimage.center_of_mass(mask.astype(np.uint8))
        return tuple(int(round(value)) for value in center)
    return tuple(size // 2 for size in mask.shape)


def draw_orthogonal_views(axes, volume: np.ndarray, center: tuple[int, int, int], title_prefix: str, cmap: str, vmin=None, vmax=None):
    x_idx, y_idx, z_idx = center
    planes = [
        (volume[x_idx, :, :].T, f"{title_prefix} | sagittal x={x_idx}"),
        (volume[:, y_idx, :].T, f"{title_prefix} | coronal y={y_idx}"),
        (volume[:, :, z_idx].T, f"{title_prefix} | axial z={z_idx}"),
    ]
    for axis, (plane, title) in zip(axes, planes):
        axis.imshow(plane, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.set_xlabel("voxel")
        axis.set_ylabel("voxel")


hippocampus_union = np.isin(data, target_labels)
global_center = center_from_mask(hippocampus_union)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
draw_orthogonal_views(
    axes,
    data,
    global_center,
    title_prefix=f"{PRIMARY_SCAN} full SynthSeg label map",
    cmap="tab20",
    vmin=0,
    vmax=max(int(data.max()), 1),
)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(len(target_labels), 3, figsize=(16, 5 * len(target_labels)))
if len(target_labels) == 1:
    axes = np.expand_dims(axes, axis=0)

for row_index, label in enumerate(target_labels):
    mask = data == label
    label_center = center_from_mask(mask)
    draw_orthogonal_views(
        axes[row_index],
        mask.astype(np.uint8),
        label_center,
        title_prefix=LABEL_NAMES.get(label, f"Label-{label}"),
        cmap="gray",
        vmin=0,
        vmax=1,
    )
plt.tight_layout()
plt.show()
'''
        )
    )

    cells.append(markdown("## Extract raw and smoothed meshes"))
    cells.append(
        code(
            '''def keep_largest_component(mask: np.ndarray) -> tuple[np.ndarray, int]:
    labeled, component_count = ndimage.label(mask)
    if component_count <= 1:
        return mask.astype(bool), int(component_count)
    sizes = ndimage.sum(mask, labeled, index=np.arange(1, component_count + 1))
    keep_label = int(np.argmax(sizes)) + 1
    return labeled == keep_label, int(component_count)


def crop_with_margin(mask: np.ndarray, margin: int) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.argwhere(mask)
    if coordinates.size == 0:
        raise ValueError("Mask is empty.")
    lower = np.maximum(coordinates.min(axis=0) - margin, 0)
    upper = np.minimum(coordinates.max(axis=0) + margin + 1, mask.shape)
    cropped = mask[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]]
    return cropped, lower


def sanitize_label_name(label: int) -> str:
    return LABEL_NAMES.get(label, f"label_{label}").replace("-", "_").replace(" ", "_").lower()


def mesh_from_mask_for_image(reference_image, mask: np.ndarray, *, smooth: bool, sigma: float, margin: int, closing_iterations: int):
    base_mask, source_component_count = keep_largest_component(mask)
    cropped, lower = crop_with_margin(base_mask, margin=margin)
    cropped = np.pad(cropped, 1, mode="constant", constant_values=False)
    origin = lower - 1

    if smooth:
        working = ndimage.binary_closing(cropped, iterations=closing_iterations)
        working = ndimage.binary_fill_holes(working)
        working, repaired_component_count = keep_largest_component(working)
        field = ndimage.gaussian_filter(working.astype(np.float32), sigma=sigma)
    else:
        repaired_component_count = source_component_count
        field = cropped.astype(np.float32)

    vertices, faces, _, _ = measure.marching_cubes(field, level=0.5)
    world_vertices = nib.affines.apply_affine(reference_image.affine, vertices + origin)
    mesh = trimesh.Trimesh(vertices=world_vertices, faces=faces, process=False)
    parts = sorted(mesh.split(only_watertight=False), key=lambda item: len(item.faces), reverse=True)
    if parts:
        mesh = parts[0].copy()
        mesh.remove_unreferenced_vertices()

    metadata = {
        "source_components": int(source_component_count),
        "post_repair_components": int(repaired_component_count),
        "mesh_parts": int(len(parts)) if parts else 1,
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "euler_number": int(mesh.euler_number),
        "mesh_volume_mm3": abs(float(mesh.volume)),
        "surface_area_mm2": float(mesh.area),
    }
    return mesh, metadata


def mesh_from_mask(mask: np.ndarray, *, smooth: bool, sigma: float, margin: int, closing_iterations: int):
    return mesh_from_mask_for_image(
        image, mask, smooth=smooth, sigma=sigma, margin=margin, closing_iterations=closing_iterations
    )


mesh_store = {}
mesh_rows = []

if WRITE_MESH_FILES:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for label in target_labels:
    name = LABEL_NAMES.get(label, f"Label-{label}")
    mask = data == label
    raw_mesh, raw_meta = mesh_from_mask(
        mask, smooth=False, sigma=GAUSSIAN_SIGMA, margin=CROP_MARGIN, closing_iterations=CLOSING_ITERATIONS
    )
    smooth_mesh, smooth_meta = mesh_from_mask(
        mask, smooth=True, sigma=GAUSSIAN_SIGMA, margin=CROP_MARGIN, closing_iterations=CLOSING_ITERATIONS
    )

    mesh_store[(label, "raw")] = raw_mesh
    mesh_store[(label, "smoothed")] = smooth_mesh

    raw_path = None
    smooth_path = None
    if WRITE_MESH_FILES:
        stem = sanitize_label_name(label)
        raw_path = OUTPUT_DIR / f"{stem}_raw.ply"
        smooth_path = OUTPUT_DIR / f"{stem}_smoothed.ply"
        raw_mesh.export(raw_path)
        smooth_mesh.export(smooth_path)

    mesh_rows.append(
        {
            "label": int(label),
            "name": name,
            "voxel_count": int(mask.sum()),
            "mask_volume_mm3": float(mask.sum() * voxel_volume_mm3),
            "source_components": raw_meta["source_components"],
            "raw_vertices": raw_meta["vertices"],
            "raw_faces": raw_meta["faces"],
            "raw_watertight": raw_meta["watertight"],
            "raw_euler_number": raw_meta["euler_number"],
            "raw_mesh_volume_mm3": raw_meta["mesh_volume_mm3"],
            "smoothed_post_repair_components": smooth_meta["post_repair_components"],
            "smoothed_vertices": smooth_meta["vertices"],
            "smoothed_faces": smooth_meta["faces"],
            "smoothed_watertight": smooth_meta["watertight"],
            "smoothed_euler_number": smooth_meta["euler_number"],
            "smoothed_mesh_volume_mm3": smooth_meta["mesh_volume_mm3"],
            "smoothed_surface_area_mm2": smooth_meta["surface_area_mm2"],
            "raw_ply": str(raw_path) if raw_path else None,
            "smoothed_ply": str(smooth_path) if smooth_path else None,
        }
    )

mesh_summary = pd.DataFrame(mesh_rows)
display(mesh_summary.round(3))
'''
        )
    )

    cells.append(markdown("## Interactive 3D comparison of raw versus smoothed meshes"))
    cells.append(
        code(
            '''def add_mesh_trace(fig, mesh: trimesh.Trimesh, *, name: str, color: str, row: int, col: int):
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    fig.add_trace(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            name=name,
            color=color,
            opacity=1.0,
            flatshading=False,
            lighting=dict(ambient=0.4, diffuse=0.75, specular=0.2, roughness=0.75),
            lightposition=dict(x=100, y=160, z=180),
            hovertemplate=name + "<br>x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<extra></extra>",
            showlegend=True,
        ),
        row=row,
        col=col,
    )


colors = {17: "#2E86DE", 53: "#E67E22"}

figure = make_subplots(
    rows=1,
    cols=2,
    specs=[[{"type": "scene"}, {"type": "scene"}]],
    subplot_titles=("Raw meshes", "Smoothed meshes"),
)

for label in target_labels:
    name = LABEL_NAMES.get(label, f"Label-{label}")
    color = colors.get(label, "#6C757D")
    add_mesh_trace(figure, mesh_store[(label, "raw")], name=f"raw: {name}", color=color, row=1, col=1)
    add_mesh_trace(figure, mesh_store[(label, "smoothed")], name=f"smoothed: {name}", color=color, row=1, col=2)

figure.update_layout(
    title=f"AIBL {PRIMARY_SCAN}: SynthSeg hippocampus meshes, raw versus smoothed",
    template="plotly_white",
    width=1400,
    height=700,
    margin=dict(l=0, r=0, t=60, b=0),
    legend=dict(x=0.01, y=0.99),
    scene=dict(aspectmode="data", xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
    scene2=dict(aspectmode="data", xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
)
figure.show()
'''
        )
    )

    cells.append(markdown("## Lateral ventricles for the primary scan"))
    cells.append(
        code(
            '''ventricle_colors = {4: "#2E86DE", 43: "#E67E22"}
ventricle_store = {}
ventricle_rows = []

for label in VENTRICLE_LABELS:
    if label not in present_labels:
        continue
    mask = data == label
    raw_mesh, raw_meta = mesh_from_mask(
        mask, smooth=False, sigma=GAUSSIAN_SIGMA, margin=CROP_MARGIN, closing_iterations=CLOSING_ITERATIONS
    )
    smooth_mesh, smooth_meta = mesh_from_mask(
        mask, smooth=True, sigma=GAUSSIAN_SIGMA, margin=CROP_MARGIN, closing_iterations=CLOSING_ITERATIONS
    )
    ventricle_store[(label, "raw")] = raw_mesh
    ventricle_store[(label, "smoothed")] = smooth_mesh
    ventricle_rows.append(
        {
            "label": int(label),
            "name": LABEL_NAMES.get(label, f"Label-{label}"),
            "voxel_count": int(mask.sum()),
            "mask_volume_mm3": float(mask.sum() * voxel_volume_mm3),
            "source_components": raw_meta["source_components"],
            "raw_vertices": int(len(raw_mesh.vertices)),
            "raw_faces": int(len(raw_mesh.faces)),
            "raw_watertight": bool(raw_mesh.is_watertight),
            "raw_mesh_volume_mm3": abs(float(raw_mesh.volume)),
            "smoothed_vertices": int(len(smooth_mesh.vertices)),
            "smoothed_faces": int(len(smooth_mesh.faces)),
            "smoothed_watertight": bool(smooth_mesh.is_watertight),
            "smoothed_euler_number": smooth_meta["euler_number"],
            "smoothed_mesh_volume_mm3": abs(float(smooth_mesh.volume)),
        }
    )

if ventricle_rows:
    ventricle_summary = pd.DataFrame(ventricle_rows)
    display(ventricle_summary.round(3))

    ventricle_figure = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Raw lateral ventricle meshes", "Smoothed lateral ventricle meshes"),
    )

    for label in VENTRICLE_LABELS:
        if (label, "raw") not in ventricle_store:
            continue
        name = LABEL_NAMES.get(label, f"Label-{label}")
        color = ventricle_colors.get(label, "#6C757D")
        add_mesh_trace(ventricle_figure, ventricle_store[(label, "raw")], name=f"raw: {name}", color=color, row=1, col=1)
        add_mesh_trace(ventricle_figure, ventricle_store[(label, "smoothed")], name=f"smoothed: {name}", color=color, row=1, col=2)

    ventricle_figure.update_layout(
        title=f"AIBL {PRIMARY_SCAN}: SynthSeg lateral ventricle meshes, raw versus smoothed",
        template="plotly_white",
        width=1400,
        height=700,
        margin=dict(l=0, r=0, t=60, b=0),
        legend=dict(x=0.01, y=0.99),
        scene=dict(aspectmode="data", xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        scene2=dict(aspectmode="data", xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
    )
    ventricle_figure.show()
else:
    print("No lateral ventricle labels found in the SynthSeg volume.")
'''
        )
    )

    cells.append(markdown("## Overview of every AIBL segmentation"))
    cells.append(
        code(
            '''def summarize_segmentation_targets(seg_paths):
    rows = []
    for seg_path in seg_paths:
        seg_image = nib.load(str(seg_path))
        seg_data = np.asanyarray(seg_image.dataobj).astype(np.int16)
        voxel_volume = abs(float(np.linalg.det(seg_image.affine[:3, :3])))
        rows.append(
            {
                "scan_id": seg_path.parent.name,
                "shape": tuple(int(v) for v in seg_image.shape),
                "voxel_volume_mm3": voxel_volume,
                "unique_labels": int(len(np.unique(seg_data))),
                "left_hippocampus_voxels": int((seg_data == 17).sum()),
                "right_hippocampus_voxels": int((seg_data == 53).sum()),
                "left_lateral_ventricle_voxels": int((seg_data == 4).sum()),
                "right_lateral_ventricle_voxels": int((seg_data == 43).sum()),
            }
        )
    frame = pd.DataFrame(rows).sort_values("scan_id").reset_index(drop=True)
    frame["hippocampus_asymmetry_pct"] = (
        100.0
        * (frame["left_hippocampus_voxels"] - frame["right_hippocampus_voxels"]).abs()
        / frame[["left_hippocampus_voxels", "right_hippocampus_voxels"]].mean(axis=1)
    )
    return frame


def build_multiscan_structure_figure(seg_paths, labels, title: str, color_map: dict[int, str]):
    subplot_titles = []
    for seg_path in seg_paths:
        scan_id = seg_path.parent.name
        subplot_titles.extend([f"{scan_id} | raw", f"{scan_id} | smoothed"])

    figure = make_subplots(
        rows=len(seg_paths),
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}] for _ in seg_paths],
        subplot_titles=tuple(subplot_titles),
        vertical_spacing=0.02,
    )

    summary_rows = []
    for row_index, seg_path in enumerate(seg_paths, start=1):
        reference_image = nib.load(str(seg_path))
        seg_data = np.asanyarray(reference_image.dataobj).astype(np.int16)
        scan_id = seg_path.parent.name
        voxel_volume = abs(float(np.linalg.det(reference_image.affine[:3, :3])))

        for label in labels:
            mask = seg_data == label
            if not mask.any():
                continue
            raw_mesh, raw_meta = mesh_from_mask_for_image(
                reference_image, mask, smooth=False, sigma=GAUSSIAN_SIGMA, margin=CROP_MARGIN,
                closing_iterations=CLOSING_ITERATIONS,
            )
            smooth_mesh, smooth_meta = mesh_from_mask_for_image(
                reference_image, mask, smooth=True, sigma=GAUSSIAN_SIGMA, margin=CROP_MARGIN,
                closing_iterations=CLOSING_ITERATIONS,
            )
            name = LABEL_NAMES.get(label, f"Label-{label}")
            color = color_map.get(label, "#6C757D")
            add_mesh_trace(figure, raw_mesh, name=f"{scan_id} | raw: {name}", color=color, row=row_index, col=1)
            add_mesh_trace(figure, smooth_mesh, name=f"{scan_id} | smoothed: {name}", color=color, row=row_index, col=2)
            summary_rows.append(
                {
                    "scan_id": scan_id,
                    "label": int(label),
                    "name": name,
                    "voxel_count": int(mask.sum()),
                    "mask_volume_mm3": float(mask.sum() * voxel_volume),
                    "source_components": raw_meta["source_components"],
                    "raw_vertices": raw_meta["vertices"],
                    "raw_faces": raw_meta["faces"],
                    "raw_watertight": raw_meta["watertight"],
                    "raw_mesh_volume_mm3": raw_meta["mesh_volume_mm3"],
                    "smoothed_vertices": smooth_meta["vertices"],
                    "smoothed_faces": smooth_meta["faces"],
                    "smoothed_watertight": smooth_meta["watertight"],
                    "smoothed_euler_number": smooth_meta["euler_number"],
                    "smoothed_mesh_volume_mm3": smooth_meta["mesh_volume_mm3"],
                }
            )

    scene_layout = {}
    total_scenes = len(seg_paths) * 2
    for scene_index in range(1, total_scenes + 1):
        scene_key = "scene" if scene_index == 1 else f"scene{scene_index}"
        scene_layout[scene_key] = dict(aspectmode="data", xaxis_title="X", yaxis_title="Y", zaxis_title="Z")

    figure.update_layout(
        title=title,
        template="plotly_white",
        width=1400,
        height=max(900, 260 * len(seg_paths)),
        margin=dict(l=0, r=0, t=70, b=0),
        legend=dict(x=0.01, y=0.99),
        **scene_layout,
    )
    return pd.DataFrame(summary_rows), figure


segmentations_overview = summarize_segmentation_targets(SEGMENTATION_PATHS)
display(segmentations_overview.round(2))
print(f"Loaded {len(SEGMENTATION_PATHS)} SynthSeg volumes from {SEGMENTATIONS_DIR}")
'''
        )
    )

    cells.append(markdown("## Hippocampus across all AIBL scans"))
    cells.append(
        code(
            '''segmentations_hipp_summary, segmentations_hipp_figure = build_multiscan_structure_figure(
    SEGMENTATION_PATHS,
    [17, 53],
    title="AIBL SynthSeg hippocampus meshes across segmentations/",
    color_map={17: "#2E86DE", 53: "#E67E22"},
)

display(segmentations_hipp_summary.round(3))
segmentations_hipp_figure.show()
'''
        )
    )

    cells.append(markdown("## Lateral ventricles across all AIBL scans"))
    cells.append(
        code(
            '''segmentations_vent_summary, segmentations_vent_figure = build_multiscan_structure_figure(
    SEGMENTATION_PATHS,
    [4, 43],
    title="AIBL SynthSeg lateral ventricle meshes across segmentations/",
    color_map={4: "#2E86DE", 43: "#E67E22"},
)

display(segmentations_vent_summary.round(3))
segmentations_vent_figure.show()
'''
        )
    )

    cells.append(
        markdown(
            """## FreeSurfer 5.3 `aseg` cross-validation

Each AIBL tar carries `mri/aseg.mgz` from the original FreeSurfer 5.3 run, on the
same conformed 256^3 grid as the SynthSeg output.  Comparing them is an independent
check that SynthSeg placed each structure on the right anatomy.

Expect SynthSeg volumes to run roughly 10-15 percent larger than `aseg` - that is the
known and well documented offset between the two methods.  What matters is that the
offset is **consistent in direction and magnitude across all four structures**, and
that Dice overlap is high.  A structure that disagrees in a different direction from
the others, or that has low Dice, is the failure signature worth chasing.

Tars are only read if present locally; scans without a tar are skipped."""
        )
    )
    cells.append(
        code(
            '''import subprocess
import tempfile

ASEG_LABELS = {4: "Left-Lateral-Ventricle", 17: "Left-Hippocampus", 43: "Right-Lateral-Ventricle", 53: "Right-Hippocampus"}


def dice(first: np.ndarray, second: np.ndarray) -> float:
    denominator = int(first.sum()) + int(second.sum())
    if denominator == 0:
        return float("nan")
    return float(2.0 * int((first & second).sum()) / denominator)


aseg_rows = []
for seg_path in SEGMENTATION_PATHS:
    scan_id = seg_path.parent.name
    tar_path = TAR_DIR / f"{scan_id}.tar"
    if not tar_path.is_file():
        print(f"[skip] no local tar for {scan_id}")
        continue

    with tempfile.TemporaryDirectory() as work_dir:
        member = f"{scan_id}/mri/aseg.mgz"
        try:
            subprocess.run(["tar", "-xf", str(tar_path), "-C", work_dir, member], check=True)
        except subprocess.CalledProcessError:
            print(f"[skip] {member} not found inside {tar_path.name}")
            continue

        aseg_data = np.asanyarray(nib.load(str(Path(work_dir) / member)).dataobj).astype(np.int16)

    synth_data = np.asanyarray(nib.load(str(seg_path)).dataobj).astype(np.int16)
    if aseg_data.shape != synth_data.shape:
        print(f"[skip] shape mismatch for {scan_id}: {aseg_data.shape} vs {synth_data.shape}")
        continue

    for label, name in ASEG_LABELS.items():
        aseg_mask = aseg_data == label
        synth_mask = synth_data == label
        aseg_rows.append(
            {
                "scan_id": scan_id,
                "label": label,
                "name": name,
                "aseg_voxels": int(aseg_mask.sum()),
                "synthseg_voxels": int(synth_mask.sum()),
                "ratio_synthseg_over_aseg": float(synth_mask.sum() / aseg_mask.sum()) if aseg_mask.any() else float("nan"),
                "dice": dice(aseg_mask, synth_mask),
            }
        )

if aseg_rows:
    aseg_comparison = pd.DataFrame(aseg_rows)
    display(aseg_comparison.round(3))
    print("\\nRatio summary (SynthSeg / aseg), expect a consistent 1.0-1.2 across structures:")
    display(aseg_comparison.groupby("name")[["ratio_synthseg_over_aseg", "dice"]].agg(["mean", "min", "max"]).round(3))
else:
    print("No local tars available for aseg cross-validation.")
    print(f"Copy the AIBL tars into {TAR_DIR} to enable this check.")
'''
        )
    )

    cells.append(
        markdown(
            """## Longitudinal consistency

Scan IDs follow the ADNI-style `RID_VISCODE` convention, so repeat visits of the same
subject can be grouped by splitting on the final underscore.  Where a subject has more
than one timepoint, hippocampal volume should decrease and ventricular volume should
increase with time in an ageing or AD cohort, and the two hemispheres should move in
the same direction.

Opposite or wildly asymmetric movement between timepoints is the signature of a
segmentation failure rather than of biology."""
        )
    )
    cells.append(
        code(
            '''def viscode_to_month(value: str) -> float:
    text = str(value).strip()
    if text == "bl":
        return 0.0
    if text.startswith("m") and text[1:].isdigit():
        return float(int(text[1:]))
    return float("nan")


longitudinal_rows = []
for seg_path in SEGMENTATION_PATHS:
    scan_id = seg_path.parent.name
    subject_id, _, viscode = scan_id.rpartition("_")
    seg_image = nib.load(str(seg_path))
    seg_data = np.asanyarray(seg_image.dataobj).astype(np.int16)
    voxel_volume = abs(float(np.linalg.det(seg_image.affine[:3, :3])))
    longitudinal_rows.append(
        {
            "subject_id": subject_id,
            "scan_id": scan_id,
            "viscode": viscode,
            "month": viscode_to_month(viscode),
            "left_hippocampus_mm3": float((seg_data == 17).sum() * voxel_volume),
            "right_hippocampus_mm3": float((seg_data == 53).sum() * voxel_volume),
            "left_lateral_ventricle_mm3": float((seg_data == 4).sum() * voxel_volume),
            "right_lateral_ventricle_mm3": float((seg_data == 43).sum() * voxel_volume),
        }
    )

longitudinal = pd.DataFrame(longitudinal_rows).sort_values(["subject_id", "month"]).reset_index(drop=True)
display(longitudinal.round(1))

repeat_subjects = longitudinal.groupby("subject_id").filter(lambda group: len(group) > 1)
if repeat_subjects.empty:
    print("No subject has more than one timepoint in this set; nothing to compare.")
else:
    measures = [
        "left_hippocampus_mm3",
        "right_hippocampus_mm3",
        "left_lateral_ventricle_mm3",
        "right_lateral_ventricle_mm3",
    ]
    change_rows = []
    for subject_id, group in repeat_subjects.groupby("subject_id"):
        group = group.sort_values("month")
        first = group.iloc[0]
        for _, later in group.iloc[1:].iterrows():
            span_months = later["month"] - first["month"]
            row = {
                "subject_id": subject_id,
                "from": first["scan_id"],
                "to": later["scan_id"],
                "months": span_months,
            }
            for measure_name in measures:
                change_pct = 100.0 * (later[measure_name] - first[measure_name]) / first[measure_name]
                row[f"{measure_name}_change_pct"] = change_pct
                if span_months and not np.isnan(span_months) and span_months > 0:
                    row[f"{measure_name}_pct_per_year"] = change_pct * 12.0 / span_months
            change_rows.append(row)

    change = pd.DataFrame(change_rows)
    display(change.round(2))

    print("Expected direction: hippocampus negative, ventricles positive.")
    for _, row in change.iterrows():
        hippocampus_ok = row["left_hippocampus_mm3_change_pct"] < 0 and row["right_hippocampus_mm3_change_pct"] < 0
        ventricle_ok = row["left_lateral_ventricle_mm3_change_pct"] > 0 and row["right_lateral_ventricle_mm3_change_pct"] > 0
        verdict = "as expected" if hippocampus_ok and ventricle_ok else "REVIEW"
        print(f"  {row['from']} -> {row['to']} ({row['months']:.0f} months): {verdict}")

    plot_frame = repeat_subjects.copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for subject_id, group in plot_frame.groupby("subject_id"):
        group = group.sort_values("month")
        axes[0].plot(group["month"], group["left_hippocampus_mm3"], marker="o", label=f"{subject_id} L")
        axes[0].plot(group["month"], group["right_hippocampus_mm3"], marker="s", linestyle="--", label=f"{subject_id} R")
        axes[1].plot(group["month"], group["left_lateral_ventricle_mm3"], marker="o", label=f"{subject_id} L")
        axes[1].plot(group["month"], group["right_lateral_ventricle_mm3"], marker="s", linestyle="--", label=f"{subject_id} R")
    axes[0].set_title("Hippocampus volume over time")
    axes[1].set_title("Lateral ventricle volume over time")
    for axis in axes:
        axis.set_xlabel("months from baseline")
        axis.set_ylabel("volume (mm^3)")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
'''
        )
    )

    notebook["cells"] = cells
    return notebook


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--primary-scan", default=DEFAULT_PRIMARY_SCAN)
    parser.add_argument("--output", type=Path, default=DEFAULT_NOTEBOOK_PATH)
    args = parser.parse_args()

    notebook = build_notebook(args.data_root.resolve(), args.primary_scan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        nbf.write(notebook, handle)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
