#!/usr/bin/env python3
"""Create a load-only notebook for the matched velocity-by-age analysis."""

from pathlib import Path

import nbformat as nbf


TASK_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = TASK_DIR / "notebooks" / "all_methods_velocity_by_age.ipynb"
RESULT_DIR = Path(
    "/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/"
    "task4_all_flow_visual_analysis_v1/age_velocity_all_methods_v1"
)


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(text.strip() + "\n")


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"]["kernelspec"] = {
        "display_name": "pytorch_geo",
        "language": "python",
        "name": "python3",
    }
    nb["metadata"]["language_info"] = {"name": "python", "version": "3.10"}
    nb["cells"] = [
        markdown(
            """
# Instantaneous surface velocity by age and diagnosis

This notebook compares **direct-mesh cocycle**, **latent cocycle**, and the
three-member **LAMM ensemble** on the validation cohort. INR is included in a
strict shared subset. Historical latent ODE and BrainODE results are shown in a
separate section because their data and evaluation protocol differ.

The continuous-time derivative is not observed by MRI. The line labelled
**Observed reference** is the derivative of a smooth trajectory fitted through
each subject's registered longitudinal meshes after removing residual rigid
motion. It is therefore the best available longitudinal reference, not a
directly measured physical ground truth.
"""
        ),
        markdown(
            """
## Load completed results

This notebook performs no model inference and does not touch the test split. It
only reads the already validated CSV tables and PNG figures.
"""
        ),
        code(
            f"""
from pathlib import Path
import json
import pandas as pd
from IPython.display import Image, Markdown, display

RESULT_DIR = Path({str(RESULT_DIR)!r})
FIGURE_DIR = RESULT_DIR / "figures"
TABLE_DIR = RESULT_DIR / "tables"

with (RESULT_DIR / "summary.json").open() as f:
    manifest = json.load(f)
full = pd.read_csv(TABLE_DIR / "full_cohort_age_summary.csv")
strict = pd.read_csv(TABLE_DIR / "strict_common_age_summary.csv")
legacy = pd.read_csv(TABLE_DIR / "legacy_age_summary.csv")

assert manifest["status"] == "complete"
assert manifest["split"] == "val"
assert manifest["test_data_loaded"] is False
print("Validated result:", manifest["status"])
print("Full matched cohort: 269 visits from 61 subjects")
print("Strict INR-shared cohort: 100 visits from 20 subjects")
"""
        ),
        markdown(
            """
## What velocity each model supplies

- **Mesh Spiral / Mesh Adaptive:** the network outputs a zero-horizon velocity
  at every corresponding template vertex. This is a direct surface-space field.
- **Latent PCA / Spiral / Adaptive:** the network outputs the instantaneous
  derivative of its latent state. Decoder differentiation maps that derivative
  to the corresponding surface vertices.
- **LAMM N=3:** the three trained latent-flow members are differentiated and
  decoded independently; their three surface velocity vectors are averaged.
- **Latent INR:** the latent derivative is propagated through the implicit SDF
  decoder. An implicit level set determines only normal surface motion, so no
  unsupported tangential INR velocity is invented.

All surface quantities below are in millimetres per year. Latent widths are
intentionally omitted from result labels.
"""
        ),
        code(
            """
counts = pd.DataFrame(manifest["method_counts"]).T
counts.index = [
    full.loc[full.method.eq(name), "method_label"].iloc[0]
    if full.method.eq(name).any()
    else strict.loc[strict.method.eq(name), "method_label"].iloc[0]
    for name in counts.index
]
display(counts.rename(columns={"subjects": "Subjects", "visits": "Visits"}))
"""
        ),
        markdown(
            """
## Metric 1 — velocity magnitude

**Speed** is the mean length of the vertex velocity vectors. It answers “how
much surface motion per year does the model produce?” but not whether that
motion points in the correct direction. A prediction below the observed line
is too static. Confidence bands are 95% intervals from resampling subjects,
which preserves the dependence among repeated visits.
"""
        ),
        code('display(Image(filename=str(FIGURE_DIR / "full_cohort_speed_by_age.png")))'),
        markdown(
            """
## Metric 2 — signed inward normal velocity

Each vertex velocity is projected onto the inward surface normal and then
averaged. Positive values mean inward motion/atrophy under this sign
convention; negative values mean outward motion. Unlike speed, this metric
distinguishes contraction from expansion, although tangential motion does not
contribute.
"""
        ),
        code('display(Image(filename=str(FIGURE_DIR / "full_cohort_inward_normal_by_age.png")))'),
        markdown(
            """
## Metrics 3–5 — agreement with the observed reference

- **Normal error / zero prediction:** RMSE of the predicted versus observed
  normal velocity, divided by the RMSE obtained by predicting no motion.
  Values below 1 improve on a zero-velocity baseline; lower is better.
- **Normal correlation:** spatial Pearson correlation between predicted and
  observed normal velocities after weighted aggregation; higher is better.
- **Vector cosine:** directional agreement of full 3-D vertex velocities.
  1 is aligned, 0 is unrelated, and −1 is opposite.

These metrics complement speed: a method can predict a plausible magnitude
while placing change on the wrong part of the hippocampal surface.
"""
        ),
        code('display(Image(filename=str(FIGURE_DIR / "full_cohort_normal_agreement_by_age.png")))'),
        markdown(
            """
### Full-cohort numerical results

The table exposes the values underlying the figures. `Speed ratio` is predicted
speed divided by observed speed; 1 would match the reference magnitude.
"""
        ),
        code(
            """
cols = [
    "method_label", "diagnosis", "age_bin", "visits", "subjects",
    "predicted_speed_mm_per_year", "observed_speed_mm_per_year", "speed_ratio",
    "normal_error_to_zero_ratio", "normal_pearson", "vector_cosine",
]
display(full[cols].rename(columns={
    "method_label": "Method", "diagnosis": "Group", "age_bin": "Age",
    "visits": "Visits", "subjects": "Subjects",
    "predicted_speed_mm_per_year": "Predicted speed",
    "observed_speed_mm_per_year": "Observed speed", "speed_ratio": "Speed ratio",
    "normal_error_to_zero_ratio": "Normal error / zero prediction",
    "normal_pearson": "Normal correlation", "vector_cosine": "Vector cosine",
}).round(4))
"""
        ),
        markdown(
            """
## Strict shared cohort including INR

INR is available for fewer validation subjects, so adding it requires reducing
every method to exactly the same 20 subjects and 100 visits. These figures are
for an apples-to-apples INR comparison only. The oldest AD bin contains one
subject and must be treated as descriptive, not a stable population estimate.
For INR, agreement should be interpreted as normal-motion agreement because an
implicit surface does not identify tangential correspondence.
"""
        ),
        code(
            """
for filename in [
    "strict_common_speed_by_age.png",
    "strict_common_inward_normal_by_age.png",
    "strict_common_normal_agreement_by_age.png",
]:
    display(Image(filename=str(FIGURE_DIR / filename)))
"""
        ),
        code(
            """
display(strict[cols].rename(columns={
    "method_label": "Method", "diagnosis": "Group", "age_bin": "Age",
    "visits": "Visits", "subjects": "Subjects",
    "predicted_speed_mm_per_year": "Predicted speed",
    "observed_speed_mm_per_year": "Observed speed", "speed_ratio": "Speed ratio",
    "normal_error_to_zero_ratio": "Normal error / zero prediction",
    "normal_pearson": "Normal correlation", "vector_cosine": "Vector cosine",
}).round(4))
"""
        ),
        markdown(
            """
## LAMM N=3 consistency check

The independent implementation is checked against the existing LAMM evaluator.
The maximum absolute discrepancy in surface speed is shown below; a value near
floating-point precision verifies that the correct three checkpoints and
surface-space averaging rule were used.
"""
        ),
        code(
            """
print(
    "Maximum absolute LAMM N=3 speed difference (mm/year):",
    f'{manifest["lamm_n3_baseline_speed_max_abs_difference"]:.3e}',
)
"""
        ),
        markdown(
            """
## Historical ODE and BrainODE comparison — separate protocol

These completed models use a different historical cohort, reference
construction, and evaluation protocol. They are useful as context but must not
be pooled with or ranked directly against the primary matched validation
results. “Observed reference” here therefore refers to that historical
protocol's own fitted longitudinal derivative.
"""
        ),
        code('display(Image(filename=str(FIGURE_DIR / "legacy_ode_brainode_velocity_by_age.png")))'),
        code(
            """
display(legacy[cols].rename(columns={
    "method_label": "Method", "diagnosis": "Group", "age_bin": "Age",
    "visits": "Visits", "subjects": "Subjects",
    "predicted_speed_mm_per_year": "Predicted speed",
    "observed_speed_mm_per_year": "Observed speed", "speed_ratio": "Speed ratio",
    "normal_error_to_zero_ratio": "Normal error / zero prediction",
    "normal_pearson": "Normal correlation", "vector_cosine": "Vector cosine",
}).round(4))
"""
        ),
        markdown(
            """
## Interpretation limits

The observed reference is inferred from discrete longitudinal visits, so its
quality depends on registration, mesh correspondence, scan interval, and the
trajectory smoother. Age-bin confidence intervals reflect subject sampling,
not uncertainty in every preprocessing step. These results measure whether a
model reproduces the fitted population/subject trajectory field; they do not by
themselves prove a biological mechanism or causal disease effect.
"""
        ),
    ]
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NOTEBOOK_PATH)
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
