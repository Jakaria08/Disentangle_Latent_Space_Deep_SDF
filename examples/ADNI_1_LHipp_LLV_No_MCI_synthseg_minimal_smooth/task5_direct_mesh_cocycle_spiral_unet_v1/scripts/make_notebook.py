#!/usr/bin/env python3
"""Create the fast, load-only Spiral versus Adaptive results notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


TASK = Path(__file__).resolve().parents[1]
TARGET = TASK / "notebooks" / "direct_mesh_results.ipynb"


def markdown(value: str):
    return nbf.v4.new_markdown_cell(value.strip() + "\n")


def code(value: str):
    return nbf.v4.new_code_cell(value.strip() + "\n")


def main() -> int:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3 (pytorch_geo)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    }
    notebook["cells"] = [
        markdown(r"""
# Direct surface cocycle: Spiral versus Adaptive Spiral

This notebook compares the two direct registered-mesh models. It never displays
a latent dimension because neither model has a global latent bottleneck. All
training, model inference, surface sampling, and bootstrap resampling must be
completed by the scripts first; these cells only load tables/maps and plot them.

## Read this before interpreting instantaneous velocity

A longitudinal scan measures shape at a visit, not the exact time derivative at
that visit. Therefore there is **no directly measured instantaneous ground
truth**. The model quantity is mathematically well-defined:

`v_model(X,a,d) = d Phi(X,a,t,d)/dt at t=a = G(X,a,a,d)`.

It is compared with the derivative of a smooth trajectory fitted to each
subject's observed visits after rigid removal. The figures call that quantity
**Observed fitted longitudinal reference**. Agreement supports a plausible
learned local trend; disagreement can reflect model error, registration error,
limited visits, or uncertainty in the fitted reference.
"""),
        markdown(r"""
## Load completed evaluation artifacts

Edit the two run names only if you used different names. `SPLIT="test"` should
be used only after model and analysis choices were frozen on validation.
"""),
        code(r"""
from pathlib import Path
import json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

BULK = Path("/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task5_direct_mesh_cocycle_spiral_unet_v1")
RUNS = {
    "Spiral": os.environ.get("DIRECT_MESH_SPIRAL_RUN", "direct_mesh_spiral_optuna_main_v1_s42"),
    "Adaptive Spiral": os.environ.get("DIRECT_MESH_ADAPTIVE_RUN", "direct_mesh_adaptive_optuna_main_v1_s42"),
}
SPLIT = os.environ.get("DIRECT_MESH_SPLIT", "test")
COLORS = {"CN": "#2b6cb0", "AD": "#c53030"}
METHOD_COLORS = {"Spiral": "#f58518", "Adaptive Spiral": "#54a24b"}

pairs, velocities, summaries, maps = [], [], {}, {}
for method, run in RUNS.items():
    folder = BULK / "runs" / run / "evaluation" / SPLIT
    required = [folder / "per_pair_metrics.csv", folder / "instantaneous_velocity_metrics.csv", folder / "summary.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Run the evaluation commands first:\n" + "\n".join(missing))
    pair = pd.read_csv(required[0], dtype={"source_scan_id": str, "target_scan_id": str, "subject": str})
    pair["Method"] = method; pairs.append(pair)
    velocity = pd.read_csv(required[1], dtype={"scan_id": str, "subject_id": str})
    velocity["Method"] = method; velocities.append(velocity)
    summaries[method] = json.loads(required[2].read_text())
    map_path = folder / "mean_vertex_maps.npz"
    if map_path.is_file():
        maps[method] = dict(np.load(map_path))

pairs = pd.concat(pairs, ignore_index=True)
velocities = pd.concat(velocities, ignore_index=True)
print(f"Split: {SPLIT}; methods: {', '.join(RUNS)}; pair rows: {len(pairs):,}")
print("Epochs:", {method: value["epoch"] for method, value in summaries.items()})
"""),
        markdown(r"""
## Endpoint deformation accuracy

**Mean vertex error** is the mean Euclidean distance between corresponding
predicted and observed vertices. **Vertex RMSE** emphasizes larger local errors.
**No-change ratio** divides model error by the error from returning the source
mesh unchanged; below 1 is improvement. **Relative volume error** is absolute
target-volume error divided by observed target volume. Lower is better for all
four. Bars first average visits within each subject, then average subjects so a
subject with many visits does not dominate.
"""),
        code(r"""
def subject_table(frame, columns):
    return frame.groupby(["Method", "diagnosis", "subject"], as_index=False)[columns].mean()

endpoint = subject_table(pairs, ["mean_vertex_error_mm", "vertex_rmse_mm", "nochange_mean_vertex_error_mm", "volume_relative_error"])
endpoint["error_to_nochange_ratio"] = endpoint.mean_vertex_error_mm / endpoint.nochange_mean_vertex_error_mm.clip(lower=1e-8)
metrics = [
    ("mean_vertex_error_mm", "Mean vertex error (mm)"),
    ("vertex_rmse_mm", "Vertex RMSE (mm)"),
    ("error_to_nochange_ratio", "Error / no-change error"),
    ("volume_relative_error", "Relative volume error"),
]
fig, axes = plt.subplots(1, 4, figsize=(18, 4.3))
for axis, (metric, label) in zip(axes, metrics):
    table = endpoint.groupby(["Method", "diagnosis"])[metric].mean().unstack()
    table.plot.bar(ax=axis, color=[COLORS.get(c, "gray") for c in table.columns], legend=False)
    axis.set_title(label); axis.set_xlabel(""); axis.tick_params(axis="x", rotation=20)
axes[0].legend(title="Diagnosis")
plt.tight_layout(); plt.show()
display(endpoint.groupby(["Method", "diagnosis"])[[m[0] for m in metrics]].agg(["mean", "sem"]).round(5))
"""),
        markdown(r"""
## AD versus CN volume trend

**Annualized log-volume rate** is
`100 * [log(V_target)-log(V_source)] / elapsed years`, approximately percent
volume change per year. Negative values mean shrinkage. The observed bar comes
from real visit volumes; the method bars come from each predicted target mesh.
The age curves place each interval at its midpoint age. They show
interval-average change, not an exactly observed derivative.
"""),
        code(r"""
base_method = next(iter(RUNS))
observed = pairs[pairs.Method.eq(base_method)].copy()
observed["Series"] = "Observed visits"
observed["rate_percent"] = 100 * observed.observed_log_volume_rate_per_year
predicted = pairs.copy()
predicted["Series"] = predicted.Method
predicted["rate_percent"] = 100 * predicted.predicted_log_volume_rate_per_year
volume = pd.concat([
    observed[["diagnosis", "subject", "source_age_years", "target_age_years", "Series", "rate_percent"]],
    predicted[["diagnosis", "subject", "source_age_years", "target_age_years", "Series", "rate_percent"]],
], ignore_index=True)
volume["mid_age"] = 0.5 * (volume.source_age_years + volume.target_age_years)

subject_volume = volume.groupby(["Series", "diagnosis", "subject"], as_index=False).rate_percent.mean()
table = subject_volume.groupby(["Series", "diagnosis"]).rate_percent.mean().unstack()
ax = table.plot.bar(figsize=(10, 4.8), color=[COLORS.get(c, "gray") for c in table.columns])
ax.axhline(0, color="black", linewidth=.8); ax.set_xlabel("")
ax.set_ylabel("Annualized log-volume change (%/year)")
ax.set_title("Observed and predicted AD/CN volume trends")
ax.tick_params(axis="x", rotation=15); plt.tight_layout(); plt.show()

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
bins = np.arange(55, 101, 5)
for axis, diagnosis in zip(axes, ("CN", "AD")):
    part = volume[volume.diagnosis.eq(diagnosis)].copy()
    part["age_bin"] = pd.cut(part.mid_age, bins)
    trend = part.groupby(["Series", "age_bin"], observed=True).rate_percent.mean().reset_index()
    trend["age"] = trend.age_bin.map(lambda value: value.mid).astype(float)
    for series, line in trend.groupby("Series"):
        axis.plot(line.age, line.rate_percent, marker="o", label=series)
    axis.axhline(0, color="black", linewidth=.7); axis.set_title(diagnosis)
    axis.set_xlabel("Interval midpoint age (years)")
axes[0].set_ylabel("Annualized log-volume change (%/year)")
axes[1].legend(); plt.tight_layout(); plt.show()
display(subject_volume.groupby(["Series", "diagnosis"]).rate_percent.agg(["mean", "sem", "count"]).round(4))
"""),
        markdown(r"""
## Surface-distance and topology metrics

These are computed on the deterministic balanced subset requested during
evaluation. **ASSD** is average symmetric surface distance. **HD95** is its 95th
percentile and reveals localized failures. **Chamfer L2 squared** averages
squared bidirectional distances. **Normal signed cosine** measures orientation
agreement (higher is better). **Flipped-face fraction** detects local topology
inversions (lower is better). **Mesh Dice** measures volumetric overlap (higher
is better). The number of eligible subject-balanced cases is printed below.
"""),
        code(r"""
surface_metrics = [
    ("assd_mm", "ASSD (mm)"), ("hd95_mm", "HD95 (mm)"),
    ("chamfer_l2_squared_mm2", "Chamfer L2² (mm²)"),
    ("normal_signed_cosine", "Normal cosine"),
    ("flipped_face_fraction", "Flipped-face fraction"), ("mesh_dice", "Mesh Dice"),
]
available = [(name, label) for name, label in surface_metrics if name in pairs and pairs[name].notna().any()]
if available:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    surface_rows = []
    for axis, (metric, label) in zip(axes.flat, available):
        current = pairs.dropna(subset=[metric]).groupby(["Method", "diagnosis", "subject"], as_index=False)[metric].mean()
        table = current.groupby(["Method", "diagnosis"])[metric].mean().unstack()
        table.plot.bar(ax=axis, color=[COLORS.get(c, "gray") for c in table.columns], legend=False)
        axis.set_title(label); axis.set_xlabel(""); axis.tick_params(axis="x", rotation=20)
        surface_rows.append({"metric": metric, "subjects": current.subject.nunique(), "pairs": int(pairs[metric].notna().sum())})
    for axis in axes.flat[len(available):]: axis.axis("off")
    axes.flat[0].legend(title="Diagnosis"); plt.tight_layout(); plt.show()
    display(pd.DataFrame(surface_rows))
else:
    print("No exact surface metrics found. Re-run evaluate.py with --surface-metrics.")
"""),
        markdown(r"""
## Instantaneous surface velocity

The reference below is inferred from observed longitudinal trajectories, not
directly measured instantaneous truth. **Vector RMSE** compares all xyz
components. **Normal RMSE** compares only inward/outward motion. **Vector
cosine** measures global directional agreement. **Normal Pearson** measures the
spatial pattern after removing each map's mean. **Normal sign agreement** asks
whether local inward/outward direction agrees. **High-change-region Dice**
compares the top 20% absolute-normal-change vertices. **Speed ratio** is model
mean speed divided by reference mean speed; values far below 1 mean the model is
too static. Errors are lower-is-better; all other agreement metrics are
higher-is-better, while speed ratio is best near 1.
"""),
        code(r"""
velocity_columns = [
    ("vector_rmse_mm_per_year", "Vector RMSE\n(mm/year)"),
    ("normal_rmse_mm_per_year", "Normal RMSE\n(mm/year)"),
    ("vector_cosine", "Vector cosine"),
    ("normal_pearson", "Normal Pearson"),
    ("normal_sign_agreement", "Normal sign agreement"),
    ("hotspot_dice", "High-change-region Dice"),
    ("speed_ratio", "Model / observed speed"),
]
subject_velocity = velocities.groupby(["Method", "diagnosis", "subject_id"], as_index=False)[[x[0] for x in velocity_columns]].mean()
fig, axes = plt.subplots(2, 4, figsize=(17, 8))
for axis, (metric, label) in zip(axes.flat, velocity_columns):
    table = subject_velocity.groupby(["Method", "diagnosis"])[metric].mean().unstack()
    table.plot.bar(ax=axis, color=[COLORS.get(c, "gray") for c in table.columns], legend=False)
    axis.set_title(label); axis.set_xlabel(""); axis.tick_params(axis="x", rotation=20)
axes.flat[-1].axis("off"); axes.flat[0].legend(title="Diagnosis")
plt.tight_layout(); plt.show()
display(subject_velocity.groupby(["Method", "diagnosis"])[[x[0] for x in velocity_columns]].agg(["mean", "sem"]).round(4))
"""),
        markdown(r"""
## Instantaneous velocity across age: AD versus CN

These age-stratified plots compare the model derivative at each visit with the
**Observed velocity** derived from that subject's fitted registered-visit
trajectory. Observed does not mean a directly measured continuous-time
derivative. **Mean surface speed** measures deformation magnitude without
direction. **Inward-normal velocity** is positive for shrinkage. **Vector error
/ zero-velocity error** below 1 means the model improves on predicting no
change. **Vector cosine** tests xyz direction, and **normal-velocity spatial
correlation** tests the vertexwise inward/outward pattern. Error bars resample
subjects, not individual visits. Every figure reports the age intervals and
the CSV table provides visit and subject counts.
"""),
        code(r"""
from IPython.display import Image

age_tables = []
for method, run in RUNS.items():
    folder = BULK / "runs" / run / "evaluation" / SPLIT / "age_velocity"
    table_path = folder / "age_velocity_summary.csv"
    if not table_path.is_file():
        print(f"{method}: run analyze_velocity_by_age.py first: {table_path}")
        continue
    current = pd.read_csv(table_path)
    current["Method"] = method
    age_tables.append(current)
    display(Markdown(f"### {method}"))
    for filename in ("speed_by_age.png", "inward_normal_velocity_by_age.png", "velocity_agreement_by_age.png"):
        path = folder / filename
        if path.is_file():
            display(Image(filename=str(path)))

if age_tables:
    age_velocity = pd.concat(age_tables, ignore_index=True)
    columns = [
        "Method", "diagnosis", "age_bin", "visits", "subjects",
        "observed_speed_mm_per_year", "predicted_speed_mm_per_year", "speed_ratio",
        "vector_error_to_zero_ratio", "vector_cosine", "normal_pearson",
    ]
    display(age_velocity[columns].round(4))
"""),
        markdown(r"""
## One averaged surface per method

Colors are mean surface-normal velocity in mm/year over all evaluated AD pairs.
Blue is inward and red is outward relative to the source surface normal. The
observed surface is the interval-average real visit change; the two model
surfaces are predictions. Correspondence makes vertexwise averaging possible.
This is one observed mesh and one mesh per method, not a gallery of individual
subjects.
"""),
        code(r"""
if len(maps) == len(RUNS):
    diagnosis = "AD"
    first = maps[next(iter(RUNS))]
    panels = [("Observed visits", first[f"{diagnosis}_source_vertices"], first[f"{diagnosis}_observed_normal_rate"])]
    for method in RUNS:
        item = maps[method]
        panels.append((method, item[f"{diagnosis}_source_vertices"], item[f"{diagnosis}_predicted_normal_rate"]))
    faces = first["faces"]
    bound = max(np.quantile(np.abs(values), .98) for _, _, values in panels)
    fig = plt.figure(figsize=(16, 5))
    for index, (title, vertices, values) in enumerate(panels, 1):
        axis = fig.add_subplot(1, len(panels), index, projection="3d")
        mesh = axis.plot_trisurf(vertices[:,0], vertices[:,1], vertices[:,2], triangles=faces,
                                cmap="coolwarm", vmin=-bound, vmax=bound, linewidth=0,
                                antialiased=False, shade=False)
        mesh.set_array(values); axis.set_title(title); axis.set_axis_off(); axis.view_init(20, -70)
    fig.colorbar(mesh, ax=fig.axes, shrink=.65, label="Normal velocity (mm/year)")
    plt.show()
else:
    print("Mean vertex maps are missing. Re-run evaluate.py with --save-vertex-maps for both methods.")
"""),
        markdown(r"""
## Cocycle, inverse, and Adaptive support diagnostics

**Relative cocycle defect** compares direct `s→t` transport with composed
`s→u→t` transport after normalizing by typical observed change. **Relative
inverse defect** measures failure to recover the source after `s→t→s`. Lower is
better. Adaptive support statistics must be spatially variable and remain
within their configured bounds; a zero standard deviation would indicate that
the adaptive operator had collapsed to a fixed support.
"""),
        code(r"""
diagnostics = []
supports = []
for method, summary in summaries.items():
    diagnostics.append({"Method": method, **summary["cocycle"]})
    for item in summary.get("adaptive_support", []):
        supports.append({"Method": method, "module": item["module"], **item["statistics"]})
display(pd.DataFrame(diagnostics).round(7))
if supports:
    support_table = pd.DataFrame(supports)
    display(support_table.round(4))
    ax = support_table.plot.bar(x="module", y="mean", yerr="std", figsize=(9,4), color=METHOD_COLORS["Adaptive Spiral"], legend=False)
    ax.set_ylabel("Learned effective spiral support"); ax.tick_params(axis="x", rotation=20)
    plt.tight_layout(); plt.show()
"""),
        markdown(r"""
## Paired subject-bootstrap comparison

The comparison uses exactly matched longitudinal pairs and averages within each
subject before resampling subjects. `adaptive_minus_spiral_mean` is negative
when Adaptive has a lower error and positive when it has a higher Dice/cosine;
`better_direction` removes that ambiguity. A 95% interval crossing zero means
the observed difference is not resolved by this sample. This is the primary
operator comparison; unpaired scan-level tests are not used.
"""),
        code(r"""
comparison_dir = BULK / "comparisons" / f"{RUNS['Spiral']}_vs_{RUNS['Adaptive Spiral']}_{SPLIT}"
comparison_path = comparison_dir / "paired_bootstrap.json"
if comparison_path.is_file():
    comparison = json.loads(comparison_path.read_text())
    table = pd.DataFrame(comparison["comparison"]).T.reset_index(names="metric")
    display(table.round(6))
else:
    print("Run compare_models.py first:", comparison_path)
"""),
        markdown(r"""
## Interpretation checklist

1. Require endpoint/no-change ratio below 1 in both CN and AD before claiming
   learned temporal prediction.
2. Inspect HD95, flipped faces, and the averaged surface map; a good mean error
   can hide a local failure.
3. Require predicted AD/CN volume-rate ordering to agree with observed visits,
   with subject-level uncertainty.
4. Treat instantaneous-velocity agreement as evidence against an overly static
   or spatially wrong field, not as validation against directly measured truth.
5. Require low cocycle/inverse defects and non-collapsed Adaptive support.
6. Rank Spiral versus Adaptive only with the paired subject bootstrap and
   matched seeds—not the implementation smoke runs.
"""),
    ]
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, TARGET)
    print(TARGET)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
