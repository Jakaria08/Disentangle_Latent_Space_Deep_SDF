#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import textwrap


REPO_ROOT = Path(__file__).resolve().parents[4]
NOTEBOOK_DIR = (
    REPO_ROOT
    / "examples"
    / "ADNI_1_L_No_MCI"
    / "brainode_comparison_task3_core_brainode_original"
    / "analysis"
    / "longitudinal_visual_notebooks"
)
OUTPUT_NOTEBOOK = NOTEBOOK_DIR / "04_adni_dataset_cohort_audit.ipynb"


def md_cell(text: str) -> dict:
    cleaned = textwrap.dedent(text).strip()
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line if line.endswith("\n") else f"{line}\n" for line in cleaned.splitlines()],
    }


def code_cell(code: str) -> dict:
    cleaned = textwrap.dedent(code).strip()
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line if line.endswith("\n") else f"{line}\n" for line in cleaned.splitlines()],
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


SETUP_CODE = """
from pathlib import Path
import importlib
import sys

import pandas as pd
from IPython.display import Markdown, display

root = Path.cwd().resolve()
while not (root / ".git").exists():
    if root.parent == root:
        raise RuntimeError("Could not locate repo root from current working directory.")
    root = root.parent

script_dir = root / "examples" / "ADNI_1_L_No_MCI" / "brainode_comparison_task3_core_brainode_original" / "scripts"
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

import dataset_cohort_visualization_support as dcv
dcv = importlib.reload(dcv)

frame = dcv.load_cohort_frame()
display(Markdown(
    f"Loaded `{len(frame)}` ground-truth scan rows from `{frame['subject_id'].nunique()}` unique subject IDs. "
    "Volumes are displayed in cubic centimeters. Meshes are only loaded for representative examples."
))
"""


COHORT_NOTEBOOK = notebook(
    [
        md_cell(
            """
            # ADNI No-MCI Dataset Cohort And Shape Audit

            This notebook compares three ADNI hippocampus cohorts used in the longitudinal shape experiments:

            - **Large ADNI no MCI:** the large strict-left no-MCI cohort before the later longitudinal QC filter.
            - **QC-filtered large ADNI:** the filtered large cohort used by the current BrainODE/PCA/SIREN comparisons.
            - **Old ADNI subset:** the smaller historical ADNI subset used by the previous BrainODE/PCA experiments.

            The plots focus on cohort balance, longitudinal scan density, observed AD/CN atrophy, volume trends,
            and representative ground-truth shapes. No model inference or mesh generation is performed here.
            """
        ),
        code_cell(SETUP_CODE),
        md_cell(
            """
            ## Data Sources

            The notebook reuses the existing whole-dataset volume audit table. For the old subset, the original audit
            volume is converted back to physical scale using the median conversion factor observed in the large cohorts.
            This keeps all plotted hippocampus volumes on the same `cm^3` scale.
            """
        ),
        code_cell(
            """
            display(dcv.data_sources_table())

            overview = dcv.cohort_overview(frame)
            display(Markdown(
                "### Cohort overview table\\n"
                "This table counts unique scans and unique subjects per dataset and diagnosis. "
                "The atrophy columns summarize start-to-end subject-level volume change."
            ))
            display(overview)
            """
        ),
        md_cell(
            """
            ## AD/CN Counts

            This section answers how many AD and CN subjects and scans are available in each dataset. The first plot
            uses all scans. The second plot splits subjects by train/val/test so model-evaluation imbalance is visible.
            """
        ),
        code_cell(
            """
            fig = dcv.plot_subject_and_scan_counts(frame)
            fig.show()

            display(Markdown(
                "The left panel is subject count and the right panel is scan count. "
                "A large gap between scans and subjects means subjects have repeated longitudinal visits."
            ))

            fig = dcv.plot_split_counts(frame)
            fig.show()

            display(Markdown(
                "Train/val/test counts are shown by subject, with scan counts in hover text. "
                "This is useful before comparing longitudinal models because a split can have enough scans but few subjects."
            ))
            """
        ),
        md_cell(
            """
            ## Longitudinal Density

            These plots show how many scans each subject contributes and how long the subject is followed. Better
            longitudinal modeling needs both repeated visits and enough follow-up time.
            """
        ),
        code_cell(
            """
            fig = dcv.plot_scan_count_and_followup(frame)
            fig.show()

            scan_followup = dcv.subject_summary(frame).groupby(
                ["dataset_label", "diagnosis"],
                sort=False,
            ).agg(
                subjects=("subject_id", "nunique"),
                mean_scans=("scan_count", "mean"),
                median_scans=("scan_count", "median"),
                mean_followup_years=("followup_years", "mean"),
                median_followup_years=("followup_years", "median"),
            ).reset_index()
            display(scan_followup)

            display(Markdown(
                "The box centers and spread show whether the cohort is truly longitudinal. "
                "Subjects with one scan contribute to baseline distributions but not to start-to-end atrophy rates."
            ))
            """
        ),
        md_cell(
            """
            ## Age And Baseline Volume

            These figures check whether AD and CN have comparable baseline age and hippocampus volume. The volume
            unit is `cm^3`, so typical hippocampus values should be a few cubic centimeters rather than thousands.
            """
        ),
        code_cell(
            """
            fig = dcv.plot_age_distribution(frame)
            fig.show()

            fig = dcv.plot_baseline_volume_and_atrophy(frame)
            fig.show()

            display(Markdown(
                "Baseline volume is measured at the first available scan for each subject. "
                "Atrophy rate is annualized from that subject's first to last observed scan."
            ))

            display(dcv.atrophy_summary_table(frame))
            """
        ),
        md_cell(
            """
            ## Observed Volume Trends

            The relative trend plot normalizes each subject by their own baseline volume. It therefore compares
            longitudinal change shape instead of absolute hippocampus size. The shaded band is the interquartile range
            of subject curves at each follow-up time.
            """
        ),
        code_cell(
            """
            fig = dcv.plot_volume_trends(frame)
            fig.show()

            display(Markdown(
                "More negative relative change means stronger hippocampal volume loss after baseline. "
                "The plotted trend is observed data only; no forecast model is used."
            ))

            fig = dcv.plot_volume_vs_age(frame)
            fig.show()

            display(Markdown(
                "The age plot uses each subject's baseline scan. "
                "It shows dataset-specific age/volume spread and whether AD/CN are naturally separated at baseline."
            ))
            """
        ),
        md_cell(
            """
            ## QC Filter Impact

            This section compares the large no-MCI cohort before and after the QC longitudinal filter. It shows how
            many scans and subjects were retained or removed, separated by AD/CN diagnosis.
            """
        ),
        code_cell(
            """
            fig = dcv.plot_qc_filter_impact(frame)
            fig.show()

            display(dcv.qc_filter_impact_table(frame))

            display(Markdown(
                "Removed rows are present in the large no-MCI cohort but not in the QC-filtered cohort. "
                "This is a dataset-processing diagnostic, not a model-quality metric."
            ))
            """
        ),
        md_cell(
            """
            ## Representative Ground-Truth Shapes

            The mesh grid shows one baseline shape per dataset for:

            - CN near the dataset's median baseline volume.
            - AD near the dataset's median baseline volume.
            - CN with the largest baseline volume.
            - CN with the smallest baseline volume.

            Meshes are loaded from the existing ground-truth mesh paths and displayed as solid surfaces. If a mesh load
            fails, that panel is skipped and the figure title reports the skipped count.
            """
        ),
        code_cell(
            """
            examples = dcv.shape_examples_table(frame)
            display(examples)

            fig = dcv.plot_shape_examples(frame)
            fig.show()

            display(Markdown(
                "The table gives the exact subject, scan, split, age, volume, and mesh path used in each panel. "
                "These meshes are visual examples only; the cohort statistics above are computed from all available audit rows."
            ))
            """
        ),
        md_cell(
            """
            ## Dataset-Specific Checks

            This final cell prints compact dataset summaries that are useful when deciding which cohort is suitable for
            longitudinal experiments. Use it to quickly compare scan count, follow-up length, baseline volume scale,
            and observed AD/CN annualized change.
            """
        ),
        code_cell(
            """
            subjects = dcv.subject_summary(frame)
            for dataset in dcv.DATASET_ORDER:
                subset = subjects.loc[subjects["dataset"].eq(dataset)].copy()
                display(Markdown(f"### {dcv.DATASET_LABELS[dataset]}"))
                display(
                    subset.groupby(["split", "diagnosis"], sort=False).agg(
                        subjects=("subject_id", "nunique"),
                        mean_scans=("scan_count", "mean"),
                        median_followup_years=("followup_years", "median"),
                        median_baseline_volume_cm3=("baseline_volume_cm3", "median"),
                        median_annual_percent_change=("annual_percent_change", "median"),
                    ).reset_index()
                )
            """
        ),
    ]
)


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_NOTEBOOK.write_text(json.dumps(COHORT_NOTEBOOK, indent=2), encoding="utf-8")
    print(OUTPUT_NOTEBOOK)


if __name__ == "__main__":
    main()
