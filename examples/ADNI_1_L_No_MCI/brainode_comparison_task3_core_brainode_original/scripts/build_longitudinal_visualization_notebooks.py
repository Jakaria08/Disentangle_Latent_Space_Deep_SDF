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

import longitudinal_visual_notebook_support as lv
lv = importlib.reload(lv)

ctx = lv.create_context(device="auto")
display(Markdown(
    "Using `longitudinal_visual_notebook_support.py`. "
    "Observed-pair plots are light. Anchor forecast cells decode meshes for the SIREN models and run best on CUDA."
))
"""


PLOTS_NOTEBOOK = notebook(
    [
        md_cell(
            """
            # QC-large four-model comparison

            This notebook focuses on the QC-large strict-left dataset because it is the only dataset in this repo
            that contains the exact four-model comparison:

            - BrainODE PCA150
            - PCA150 cocycle flow
            - SIREN cocycle flow
            - SIREN latent ODE

            The figures stay on the current evaluated outputs and only decode new meshes when you explicitly run
            the anchor forecast cells.
            """
        ),
        code_cell(SETUP_CODE),
        md_cell(
            """
            ## Ground-truth CN vs AD aging

            Relative change is measured from each subject's baseline scan. The shaded band is the interquartile range
            across subjects with at least two scans.
            """
        ),
        code_cell(
            """
            display(Markdown(
                "### Ground-truth relative aging curves\\n"
                "- **Input:** all QC-large ground-truth scan volumes in `train`, `val`, and `test`.\\n"
                "- **Output:** mean relative hippocampus volume change from each subject's own baseline, shown separately for `CN` and `AD`.\\n"
                "- **Calculation:** for each subject, `relative change = 100 * (current volume - baseline volume) / baseline volume`."
            ))
            fig = ctx.plot_ground_truth_cn_ad_aging()
            fig.show()
            display(Markdown(
                "After this graph: more negative values indicate stronger volume loss from baseline. "
                "Because the axis is baseline-relative, subjects with different absolute hippocampus sizes are directly comparable."
            ))

            display(Markdown(
                "### Ground-truth annualized start-to-end change\\n"
                "- **Input:** the first and last observed scan for each subject.\\n"
                "- **Output:** subject-level annualized percent volume change, grouped by split and diagnosis.\\n"
                "- **Calculation:** `(last - first) / first / follow-up years`."
            ))
            fig = ctx.plot_ground_truth_start_end_rates()
            fig.show()
            display(Markdown(
                "After this graph: this is a compact endpoint summary. "
                "The previous curve uses all observed visits, while this box plot uses only the first and last scan per subject."
            ))
            """
        ),
        md_cell(
            """
            ## Future-pair reconstruction metrics

            These are mesh metrics on observed future pairs. Distances stay in millimeters.
            Volume quantities are displayed in cubic centimeters and surface-area quantities in square centimeters
            so the magnitudes read like hippocampus-scale anatomy instead of raw millimeter powers.
            """
        ),
        code_cell(
            """
            display(Markdown(
                "### Observed future-pair reconstruction error\\n"
                "- **Input:** observed source-to-future scan pairs shared by the four comparison models.\\n"
                "- **Models:** BrainODE PCA150, PCA150 cocycle flow, SIREN cocycle flow, and SIREN latent ODE.\\n"
                "- **Output:** geometric and anatomical endpoint errors after forecasting the future shape. Distances stay in `mm`; volume and surface-area errors are shown in `cm^3` and `cm^2`."
            ))
            fig = ctx.plot_reconstruction_metric_grid()
            fig.show()

            summary = ctx.model_pair_summary().sort_values(["model_label", "transport_label"]).reset_index(drop=True)
            display(summary)
            display(Markdown(
                "After this graph: lower bars are better. "
                "ASSD and HD95 summarize surface mismatch; the volume and surface-area panels summarize endpoint anatomy error."
            ))
            """
        ),
        md_cell(
            """
            ## Cached train/val/test model volume trends

            This section uses already-computed observed-age trend tables only when they already exist in the repo.
            If a model has no matching cached QC-large split-wide trend table, it is intentionally omitted here.
            """
        ),
        code_cell(
            """
            trend_models = ctx.available_cached_split_trend_models()
            if not trend_models:
                display(Markdown("No cached split-wide QC-large model trend tables were found for notebook 1."))
            else:
                display(Markdown(
                    "The next figures use existing rows from `selected_volume_trends.csv`. "
                    "They are not recomputed inside this notebook."
                ))
                for model in trend_models:
                    summary = ctx.cached_split_volume_trend_summary(model)
                    transport_label = summary["transport"].iloc[0] if not summary.empty else "cached transport"
                    display(Markdown(
                        f"### {lv.MODEL_LABELS[model]} split-wise CN vs AD trend\\n"
                        f"- **Input:** cached QC-large observed-age trend rows for `{lv.MODEL_LABELS[model]}`.\\n"
                        f"- **Transport used:** `{transport_label}`.\\n"
                        "- **Output:** mean relative volume change from baseline across subjects, separated by split and diagnosis.\\n"
                        "- **Calculation:** both the observed and predicted curves are normalized by the subject's observed baseline volume."
                    ))
                    fig = ctx.plot_cached_split_volume_trends(model)
                    fig.show()
                    display(summary)
                    display(Markdown(
                        "After this graph: solid lines are observed ground-truth trends and dashed lines are model-predicted trends on the same baseline-relative scale."
                    ))
            """
        ),
        md_cell(
            """
            ## Instantaneous latent velocity

            This section evaluates the local latent vector field at every observed scan. It does not decode meshes.

            For cocycle flow models, the displayed velocity is the diagonal generator:

            `G(z, t, t, c) / age_range_years`

            For ODE models, the displayed velocity is the ODE vector field:

            `f(z, t, c) / age_range_years`

            The observed reference velocity is estimated from neighboring real scans in the same latent space.
            PCA and SIREN latent speeds are not compared as raw units; the main plots use train-standardized
            latent speed so each model is judged against its own representation scale.
            """
        ),
        code_cell(
            """
            display(Markdown(
                "### Build or load instantaneous velocity cache\\n"
                "- **Input:** all train/val/test observed scan latents for BrainODE PCA150, PCA150 cocycle flow, SIREN cocycle flow, and SIREN latent ODE.\\n"
                "- **Output:** per-scan instantaneous model velocity, observed scan-to-scan latent velocity, model-vs-real error, and CN-vs-AD condition gap.\\n"
                "- **Calculation:** no mesh is generated. The cache is reused on later notebook runs unless the support script version changes."
            ))
            velocity_tables = ctx.build_instantaneous_velocity_tables()
            display(velocity_tables["summary"].head(18))
            display(Markdown(
                f"Loaded `{len(velocity_tables['per_scan'])}` per-scan/condition velocity rows and "
                f"`{len(velocity_tables['condition_gap'])}` CN-vs-AD condition-gap rows."
            ))

            display(Markdown(
                "### Model velocity magnitude vs real local velocity\\n"
                "- **Input:** observed-condition rows only.\\n"
                "- **Output:** bars show the model's median instantaneous latent speed; `x` markers show the median real scan-to-scan latent speed.\\n"
                "- **Calculation:** speeds are divided by the training latent standard deviation component-by-component, then summarized as an L2 norm per year."
            ))
            fig = ctx.plot_instantaneous_velocity_model_vs_real()
            fig.show()
            display(Markdown(
                "After this graph: if a model bar is far below the real marker for AD, the model is underestimating AD progression speed even if its future meshes look plausible."
            ))

            display(Markdown(
                "### Velocity error and direction agreement\\n"
                "- **Input:** observed-condition rows only.\\n"
                "- **Output:** top row = median train-standardized velocity error; bottom row = median cosine similarity between model velocity and observed local velocity.\\n"
                "- **Calculation:** cosine near `1` means the model moves in the same latent direction as the observed scan-to-scan change; near `0` means weak directional agreement."
            ))
            fig = ctx.plot_instantaneous_velocity_alignment()
            fig.show()
            display(Markdown(
                "After this graph: this separates speed magnitude from direction. A model can have plausible speed but still move in the wrong latent direction."
            ))

            display(Markdown(
                "### CN-vs-AD instantaneous condition gap\\n"
                "- **Input:** the same source scan evaluated twice: once with CN condition and once with AD condition.\\n"
                "- **Output:** `AD-conditioned speed - CN-conditioned speed`, grouped by source diagnosis and split.\\n"
                "- **Calculation:** positive values mean the AD condition makes the instantaneous vector field faster than the CN condition for the same scan."
            ))
            fig = ctx.plot_instantaneous_velocity_condition_gap()
            fig.show()
            display(ctx.instantaneous_velocity_condition_gap_summary())
            display(Markdown(
                "After this graph: this is the direct local test of whether the condition label changes the velocity field in the expected disease direction."
            ))

            display(Markdown(
                "### Velocity by age bin\\n"
                "- **Input:** observed-condition rows only.\\n"
                "- **Output:** median instantaneous model speed and median observed local speed by age bin, separated by CN and AD.\\n"
                "- **Calculation:** this checks whether the learned vector field changes with age rather than only with diagnosis."
            ))
            fig = ctx.plot_instantaneous_velocity_age_trend()
            fig.show()
            display(Markdown(
                "After this graph: a clinically useful disease-aging model should show both age-dependent behavior and stronger AD velocity than CN where the real data shows that pattern."
            ))
            """
        ),
        md_cell(
            """
            ## Representative observed-pair cases

            Cases are chosen split-by-split and diagnosis-by-diagnosis from pairs that exist for all four models.
            Preference is given to longer gaps, then lower average future-pair error.
            """
        ),
        code_cell(
            """
            cases = ctx.representative_cases()
            case_table = pd.DataFrame(
                [
                    {
                        "split": case.split,
                        "diagnosis": case.diagnosis,
                        "subject_id": case.subject_id,
                        "source_scan_id": case.source_scan_id,
                        "target_scan_id": case.target_scan_id,
                    }
                    for case in cases
                ]
            )
            display(case_table)
            display(Markdown(
                "Each row defines one source-to-target case used below. "
                "The first case-study section shows only the selected target prediction per model; the next case-study section expands that into predictions at each observed follow-up age between source and target."
            ))
            """
        ),
        md_cell(
            """
            ## Volume history around the selected future-pair cases

            Each panel shows the real subject trajectory and the model-predicted target volume at the selected future age.
            Volumes are displayed in cubic centimeters.
            """
        ),
        code_cell(
            """
            for case in cases:
                display(Markdown(f"### {case.split.upper()} {case.diagnosis} | subject {case.subject_id}"))
                display(ctx.case_overview_table(case))
                display(Markdown(
                    "- **Input:** the full observed subject history plus the selected source-to-target pair for this case.\\n"
                    "- **Output:** black line = all observed scan volumes for the subject; colored diamonds = each model's prediction only at the selected target age.\\n"
                    "- **Important:** intermediate observed ages before the target are shown only on the black ground-truth line in this plot."
                ))
                fig = ctx.plot_case_subject_history(case)
                fig.show()
                display(Markdown(
                    "After this graph: this is a target-only endpoint comparison. "
                    "The next section fills in the missing intermediate predictions before the target."
                ))
            """
        ),
        md_cell(
            """
            ## Per-model predictions at every observed follow-up age between source and target

            This section addresses the gap in the simpler history plots above: if a subject has observed follow-up scans
            before the selected target, each model is now evaluated at those observed ages as well, using the same source scan as input.
            """
        ),
        code_cell(
            """
            for case in cases:
                display(Markdown(f"## {case.split.upper()} {case.diagnosis} | subject {case.subject_id} | source-to-observed follow-up forecasts"))
                for model in ctx.available_models():
                    fig, table = ctx.plot_case_model_followup_forecast(case, model)
                    first = table.iloc[0]
                    transport_label = str(table['transport_label'].dropna().iloc[0]) if table['transport_label'].notna().any() else 'endpoint'
                    display(Markdown(
                        f"### {lv.MODEL_LABELS[model]}\\n"
                        f"- **Input:** source scan `{first['source_scan_id']}` at age `{first['source_age_years']:.2f}` years with diagnosis `{first['diagnosis']}`.\\n"
                        f"- **Output:** predicted hippocampus volume at each observed follow-up age from the source age up to the selected target age `{first['target_age_years']:.2f}`.\\n"
                        f"- **Transport / mode:** `{transport_label}`.\\n"
                        "- **Calculation:** ground truth uses the physical mesh volume when available; the model curve uses the volume of the decoded forecast mesh."
                    ))
                    fig.show()
                    display(table)
                    ok = table.loc[table['status'].astype(str).eq('ok') & table['predicted_volume_cm3'].notna()].copy()
                    mae = float(ok['abs_error_cm3'].mean()) if not ok.empty else float('nan')
                    skipped = int(table['status'].astype(str).ne('ok').sum())
                    display(Markdown(
                        f"After this graph: mean absolute error across the shown observed follow-up ages is `{mae:.3f} cm^3` "
                        f"for the successful predictions. Skipped ages due to failed mesh decode: `{skipped}`."
                    ))
            """
        ),
        md_cell(
            """
            ## Conditioned future volume forecasts from a single shape

            These forecasts start from the first observed scan of the selected anchor subject.
            Solid lines use the CN condition. Dashed lines use the AD condition.
            Larger markers indicate prediction points at the exact observed follow-up ages for that subject.
            The grey region begins after the last observed scan for the chosen anchor subject.
            Volumes are displayed in cubic centimeters.
            """
        ),
        code_cell(
            """
            display(Markdown(
                "### Test CN anchor forecast\\n"
                "- **Input:** one CN source scan from the test split, used as a fixed starting shape.\\n"
                "- **Output:** future predicted volume under both the `CN` and `AD` condition labels.\\n"
                "- **Calculation:** black is observed truth; solid colored lines are CN-conditioned forecasts; dashed colored lines are AD-conditioned forecasts."
            ))
            fig = ctx.plot_anchor_forecasts(split="test", diagnosis="CN", transport="direct")
            fig.show()
            display(Markdown(
                "After this graph: the difference between the solid and dashed curves shows how much the diagnosis condition changes the future rollout while the source anatomy stays fixed."
            ))

            display(Markdown(
                "### Test AD anchor forecast\\n"
                "- **Input:** one AD source scan from the test split, used as a fixed starting shape.\\n"
                "- **Output:** future predicted volume under both the `AD` and `CN` condition labels.\\n"
                "- **Calculation:** the same source shape is rolled forward to observed and out-of-distribution ages, with the grey region starting after the subject's last observed scan."
            ))
            fig = ctx.plot_anchor_forecasts(split="test", diagnosis="AD", transport="direct")
            fig.show()
            display(Markdown(
                "After this graph: this is a conditioning sensitivity test. "
                "It does not prove disentanglement, but it does show the direction and size of the label-dependent forecast change."
            ))
            """
        ),
        md_cell(
            """
            ## Disease-conditioned gap

            This is not disentanglement. It is the forecast separation induced by swapping the conditioning label
            while keeping the source shape fixed.
            Volume gaps are displayed in cubic centimeters.
            """
        ),
        code_cell(
            """
            display(Markdown(
                "### Test CN anchor disease-conditioned gap\\n"
                "- **Input:** the CN anchor forecast from the previous section.\\n"
                "- **Output:** `AD-conditioned predicted volume - CN-conditioned predicted volume` at each future age.\\n"
                "- **Calculation:** each point subtracts two forecasts made from the same source shape."
            ))
            fig = ctx.plot_anchor_condition_gap(split="test", diagnosis="CN", transport="direct")
            fig.show()
            display(Markdown(
                "After this graph: values below zero mean the AD-conditioned rollout predicts a smaller hippocampus than the CN-conditioned rollout."
            ))

            display(Markdown(
                "### Test AD anchor disease-conditioned gap\\n"
                "- **Input:** the AD anchor forecast from the previous section.\\n"
                "- **Output:** the same `AD - CN` conditional gap, now starting from an AD source shape.\\n"
                "- **Calculation:** the source scan is fixed and only the condition label is swapped."
            ))
            fig = ctx.plot_anchor_condition_gap(split="test", diagnosis="AD", transport="direct")
            fig.show()
            display(Markdown(
                "After this graph: this quantifies label sensitivity on a shared source shape, rather than measuring reconstruction against ground truth."
            ))
            """
        ),
    ]
)


MESH_NOTEBOOK_A = notebook(
    [
        md_cell(
            """
            # QC-large mesh visualization, part 1

            This notebook shows the first half of the representative observed-pair cases.

            Each case has two visual blocks:

            - **Solid mesh comparison:** the baseline/source ground-truth mesh, the observed future/target
              ground-truth mesh, and one predicted future mesh from each model. These meshes are rendered as
              opaque solids. A display-only repair note appears when the viewer filled small visual holes or
              fixed mesh bookkeeping; the numerical metrics still come from the saved evaluation outputs.
            - **Surface-change map:** each surface is rigidly aligned to the source mesh. Color shows how far
              each displayed future surface moved away from the source surface in millimeters. This is a
              change-from-baseline map, not a direct error heatmap against the target.
            """
        ),
        code_cell(SETUP_CODE),
        code_cell(
            """
            cases = ctx.representative_cases()
            first_half = cases[:3]
            pd.DataFrame(
                [{"split": c.split, "diagnosis": c.diagnosis, "subject_id": c.subject_id} for c in first_half]
            )
            """
        ),
        code_cell(
            """
            for case in first_half:
                display(Markdown(f"## {case.split.upper()} {case.diagnosis} | subject {case.subject_id}"))
                display(Markdown(
                    "### Case metadata\\n"
                    "This table identifies the selected observed pair. The source scan is the input shape and "
                    "the target scan is the real later scan used as ground truth for this case."
                ))
                display(ctx.case_overview_table(case))
                display(Markdown(
                    "### Solid mesh comparison\\n"
                    "- **Source GT:** real baseline mesh given to the model.\\n"
                    "- **Target GT:** real future mesh at the later scan age.\\n"
                    "- **Model panels:** predicted future meshes from BrainODE PCA, PCA cocycle flow, "
                    "SIREN cocycle flow, and SIREN latent ODE.\\n"
                    "The panel is for visual plausibility and gross anatomical comparison. It is not colored by error."
                ))
                fig = ctx.plot_case_mesh_panel(case)
                fig.show()
                display(Markdown(
                    "### Surface-change maps\\n"
                    "These maps compare each future surface to the same source mesh after rigid alignment. "
                    "Darker/lower values mean little surface movement from baseline; brighter/higher values mean "
                    "larger local displacement from baseline. The Target GT panel shows the real change, and the "
                    "model panels show the predicted change pattern."
                ))
                heat_fig, heat_table = ctx.plot_case_change_heatmaps(case)
                heat_fig.show()
                display(Markdown(
                    "The table below summarizes the displayed change map: mean, p95, and max are surface-shift "
                    "magnitudes in millimeters. Failed mesh decodes are kept in the table instead of stopping the notebook."
                ))
                display(heat_table)
            """
        ),
    ]
)


MESH_NOTEBOOK_B = notebook(
    [
        md_cell(
            """
            # QC-large mesh visualization, part 2

            This notebook shows the remaining observed-pair cases and two long-horizon anchor-shape change-map comparisons.
            The observed-pair figures use the same interpretation as notebook 2: solid future meshes first,
            then change-from-source heatmaps. The anchor cells are heavier because they decode long-horizon
            future meshes for all four models and compare disease-conditioned futures from the same source shape.
            """
        ),
        code_cell(SETUP_CODE),
        code_cell(
            """
            cases = ctx.representative_cases()
            second_half = cases[3:]
            pd.DataFrame(
                [{"split": c.split, "diagnosis": c.diagnosis, "subject_id": c.subject_id} for c in second_half]
            )
            """
        ),
        code_cell(
            """
            for case in second_half:
                display(Markdown(f"## {case.split.upper()} {case.diagnosis} | subject {case.subject_id}"))
                display(Markdown(
                    "### Case metadata\\n"
                    "This table identifies the selected observed pair. The source scan is the input shape and "
                    "the target scan is the real later scan used as ground truth for this case."
                ))
                display(ctx.case_overview_table(case))
                display(Markdown(
                    "### Solid mesh comparison\\n"
                    "- **Source GT:** real baseline mesh given to the model.\\n"
                    "- **Target GT:** real future mesh at the later scan age.\\n"
                    "- **Model panels:** predicted future meshes from BrainODE PCA, PCA cocycle flow, "
                    "SIREN cocycle flow, and SIREN latent ODE.\\n"
                    "The panel is for visual plausibility and gross anatomical comparison. It is not colored by error."
                ))
                fig = ctx.plot_case_mesh_panel(case)
                fig.show()
                display(Markdown(
                    "### Surface-change maps\\n"
                    "These maps compare each future surface to the same source mesh after rigid alignment. "
                    "Darker/lower values mean little surface movement from baseline; brighter/higher values mean "
                    "larger local displacement from baseline. The Target GT panel shows the real change, and the "
                    "model panels show the predicted change pattern."
                ))
                heat_fig, heat_table = ctx.plot_case_change_heatmaps(case)
                heat_fig.show()
                display(Markdown(
                    "The table below summarizes the displayed change map: mean, p95, and max are surface-shift "
                    "magnitudes in millimeters. Failed mesh decodes are kept in the table instead of stopping the notebook."
                ))
                display(heat_table)
            """
        ),
        md_cell(
            """
            ## Long-horizon anchor change maps

            These are source-to-future change maps for the selected anchor subject in the test split.
            The source mesh is fixed. Only the requested future age and disease condition change.
            These are not compared to real 20-year ground truth because that follow-up is outside the observed data.
            Instead, they show whether a model produces a slow CN-like future or a faster AD-like future from
            the same baseline anatomy.
            """
        ),
        code_cell(
            """
            display(Markdown(
                "### CN anchor -> CN future condition\\n"
                "Input is a test-split CN source mesh. Every model forecasts the same long-horizon future age "
                "using the CN condition. The heatmap is future-vs-source displacement, so it shows predicted aging "
                "change from the baseline shape."
            ))
            fig, table = ctx.plot_anchor_change_maps(
                split="test",
                diagnosis="CN",
                condition_label="CN",
                transport="direct",
            )
            fig.show()
            display(table)

            display(Markdown(
                "### CN anchor -> AD counterfactual future condition\\n"
                "The source shape is still the same CN baseline mesh, but the disease condition is switched to AD. "
                "A useful disease-conditioned model should generally show a stronger hippocampal change than the "
                "CN-conditioned rollout from the same source."
            ))
            fig, table = ctx.plot_anchor_change_maps(
                split="test",
                diagnosis="CN",
                condition_label="AD",
                transport="direct",
            )
            fig.show()
            display(table)
            """
        ),
        code_cell(
            """
            display(Markdown(
                "### AD anchor -> AD future condition\\n"
                "Input is a test-split AD source mesh. Every model forecasts the same long-horizon future age "
                "using the AD condition. This shows the model's disease-like aging trajectory from an AD baseline."
            ))
            fig, table = ctx.plot_anchor_change_maps(
                split="test",
                diagnosis="AD",
                condition_label="AD",
                transport="direct",
            )
            fig.show()
            display(table)

            display(Markdown(
                "### AD anchor -> CN counterfactual future condition\\n"
                "The source shape is the same AD baseline mesh, but the condition is switched to CN. "
                "This is a counterfactual test of whether the model can slow the future deformation when the "
                "condition says CN."
            ))
            fig, table = ctx.plot_anchor_change_maps(
                split="test",
                diagnosis="AD",
                condition_label="CN",
                transport="direct",
            )
            fig.show()
            display(table)
            """
        ),
    ]
)


def write_notebook(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    write_notebook(NOTEBOOK_DIR / "01_qc_large_four_model_plots.ipynb", PLOTS_NOTEBOOK)
    write_notebook(NOTEBOOK_DIR / "02_qc_large_four_model_mesh_cases_part1.ipynb", MESH_NOTEBOOK_A)
    write_notebook(NOTEBOOK_DIR / "03_qc_large_four_model_mesh_cases_part2.ipynb", MESH_NOTEBOOK_B)
    print(NOTEBOOK_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
