#!/usr/bin/env python3
"""Generate the interactive strict-no-MCI SynthSeg CN-versus-AD analysis notebook."""

from __future__ import annotations

import argparse
from pathlib import Path


HERE = Path(__file__).resolve().parent


def markdown(source: str):
    import nbformat.v4 as nbf

    return nbf.new_markdown_cell(source)


def code(source: str):
    import nbformat.v4 as nbf

    return nbf.new_code_cell(source)


def build_notebook():
    import nbformat.v4 as nbf

    notebook = nbf.new_notebook()
    notebook.metadata.update(
        {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        }
    )
    notebook.cells = [
        markdown(
            """# SynthSeg strict no-MCI CN-versus-AD: longitudinal volume, disease trajectory, and local shape speed

This notebook visualizes the **left hippocampus** and **left lateral ventricle** produced by the SynthSeg correspondence pipeline.

Run `synthseg_longitudinal_qc.py` first, then `synthseg_longitudinal_analysis.py`. The source meshes are never modified. The default analysis excludes all baseline-MCI, unknown-baseline, and MCI-visit subjects, then compares baseline-CN with baseline-AD."""
        ),
        code(
            """from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display, Markdown

def find_repo_root(start: Path) -> Path:
    for candidate in (start.resolve(), *start.resolve().parents):
        if (candidate / '.git').exists() and (candidate / 'examples').is_dir():
            return candidate
    raise RuntimeError('Could not find Deep3DComp repository root.')

repo_root = find_repo_root(Path.cwd())
work_dir = repo_root / 'examples' / 'ADNI_1_L_With_MCI'
qc_dir = work_dir / 'synthseg_mesh_qc'
analysis_dir = work_dir / 'synthseg_longitudinal_analysis'
for required in [qc_dir / 'qc_summary.json', analysis_dir / 'analysis_summary.json']:
    if not required.is_file():
        raise FileNotFoundError(f'{required} is missing. Run the QC and analysis commands in README_synthseg_longitudinal_analysis.md first.')

qc_summary = json.loads((qc_dir / 'qc_summary.json').read_text())
analysis_summary = json.loads((analysis_dir / 'analysis_summary.json').read_text())
visits = pd.read_csv(analysis_dir / 'visit_metrics.csv', dtype={'scan_id': str, 'subject_id': str})
pairs = pd.read_csv(analysis_dir / 'adjacent_pair_metrics.csv', dtype={'source_scan_id': str, 'target_scan_id': str, 'subject_id': str})
subjects = pd.read_csv(analysis_dir / 'subject_rate_summary.csv', dtype={'subject_id': str})
counts = pd.read_csv(analysis_dir / 'cohort_counts.csv')
trajectory = pd.read_csv(analysis_dir / 'visit_trajectory_summary.csv')
age_rates = pd.read_csv(analysis_dir / 'age_bin_rate_summary.csv')
contrasts = pd.read_csv(analysis_dir / 'diagnosis_rate_contrasts.csv')
transitions = pd.read_csv(analysis_dir / 'clinical_transition_summary.csv')
within_trend_path = analysis_dir / 'within_subject_trend_summary.csv'
within_trends = pd.read_csv(within_trend_path) if within_trend_path.is_file() else pd.DataFrame()
scan_qc = pd.read_csv(qc_dir / 'scan_qc.csv', dtype={'scan_id': str, 'subject_id': str})
pair_qc = pd.read_csv(qc_dir / 'adjacent_pair_qc.csv', dtype={'source_scan_id': str, 'target_scan_id': str, 'subject_id': str})

COLORS = {'CN': '#2E86DE', 'MCI': '#F39C12', 'AD': '#C0392B'}
STRUCTURE_ORDER = ['left_hippocampus', 'left_lateral_ventricle']

def hex_to_rgba(color, alpha):
    color = color.lstrip('#')
    red, green, blue = (int(color[index:index + 2], 16) for index in (0, 2, 4))
    return f'rgba({red}, {green}, {blue}, {alpha})'

display(Markdown(f"**Cohort definition:** {analysis_summary['cohort_definition']['meaning']}"))
display(counts)
"""
        ),
        markdown("## 1. Data integrity and mesh quality control"),
        code(
            """display(pd.DataFrame([{
    'scan records': qc_summary['scan_records'],
    'adjacent pairs': qc_summary['adjacent_pairs'],
    'strong flagged pairs': qc_summary['strong_flagged_pairs'],
    'scan actions': qc_summary['scan_actions'],
    'subject actions': qc_summary['subject_actions'],
}]))

qc_by_structure = (scan_qc.groupby(['structure', 'final_scan_qc_action']).size().rename('count').reset_index())
fig = px.bar(qc_by_structure, x='structure', y='count', color='final_scan_qc_action', barmode='group',
             title='QC action counts by structure', color_discrete_sequence=px.colors.qualitative.Safe)
fig.update_layout(template='plotly_white')
fig.show()

review_pairs = pair_qc.loc[pair_qc['any_pair_qc_flag']].sort_values('qc_score', ascending=False)
display(review_pairs.head(25))
display(Markdown(f"Interactive source/target overlays for top flagged pairs: [{qc_dir / 'index.html'}]({qc_dir / 'index.html'})"))
"""
        ),
        markdown("## 2. Raw, smoothed, and correspondence-volume lineage"),
        code(
            """lineage_columns = ['structure_display', 'cohort_diagnosis', 'mask_volume_mm3', 'raw_mesh_volume_mm3', 'smooth_mesh_volume_mm3']
lineage = visits[[column for column in lineage_columns if column in visits.columns]].dropna()
fig = px.scatter(lineage.sample(min(2500, len(lineage)), random_state=42),
                 x='mask_volume_mm3', y='smooth_mesh_volume_mm3', color='cohort_diagnosis', facet_col='structure_display',
                 opacity=0.55, color_discrete_map=COLORS,
                 title='Minimal-smooth physical volume versus SynthSeg mask volume')
fig.update_layout(template='plotly_white')
fig.show()

fig = px.scatter(lineage.sample(min(2500, len(lineage)), random_state=43),
                 x='raw_mesh_volume_mm3', y='smooth_mesh_volume_mm3', color='cohort_diagnosis', facet_col='structure_display',
                 opacity=0.55, color_discrete_map=COLORS,
                 title='Raw mesh volume versus minimal-smooth/correspondence volume')
fig.update_layout(template='plotly_white')
fig.show()
"""
        ),
        markdown("## 3. Volume trajectories and annualized disease speed"),
        code(
            """MIN_TRAJECTORY_N = 25
display(Markdown(
    f'Visit-wise cohort means below **n={MIN_TRAJECTORY_N}** are hidden. They are descriptive because the participants observed at later visits can differ from the baseline cohort. '
    'Use the within-subject model below for the primary direction and rate of change.'
))
raw_summary = trajectory.loc[
    trajectory.metric.eq('smooth_mesh_volume_mm3') & trajectory['count'].ge(MIN_TRAJECTORY_N)
]
relative_summary = trajectory.loc[
    trajectory.metric.eq('relative_volume_to_baseline') & trajectory['count'].ge(MIN_TRAJECTORY_N)
]
for title, frame, ylabel in [
    ('Observed cohort mean physical volume (adequately supported visits)', raw_summary, 'volume (mm³)'),
    ('Observed cohort mean relative volume (adequately supported visits)', relative_summary, 'volume / baseline volume'),
]:
    figure = go.Figure()
    for (structure, diagnosis), group in frame.groupby(['structure', 'cohort_diagnosis'], sort=True):
        group = group.sort_values('years_from_baseline')
        figure.add_trace(go.Scatter(x=group.years_from_baseline, y=group.ci95_high, mode='lines', line=dict(width=0),
                                    showlegend=False, hoverinfo='skip', legendgroup=f'{structure}-{diagnosis}'))
        figure.add_trace(go.Scatter(x=group.years_from_baseline, y=group.ci95_low, mode='lines', line=dict(width=0),
                                    fill='tonexty', fillcolor=hex_to_rgba(COLORS.get(diagnosis, '#777777'), 0.13), showlegend=False,
                                    hoverinfo='skip', legendgroup=f'{structure}-{diagnosis}'))
        figure.add_trace(go.Scatter(x=group.years_from_baseline, y=group['mean'], mode='lines+markers',
                                    name=f'{structure}: {diagnosis}', line=dict(color=COLORS.get(diagnosis, '#777')), legendgroup=f'{structure}-{diagnosis}'))
    figure.update_layout(title=title, xaxis_title='years from baseline', yaxis_title=ylabel, template='plotly_white')
    figure.show()

if within_trends.empty:
    display(Markdown('**Within-subject trend table is unavailable.** Re-run `synthseg_longitudinal_analysis.py` after updating the scripts.'))
else:
    display(Markdown('### Primary longitudinal estimate: participant-intercept-adjusted trend'))
    display(within_trends)
    figure = go.Figure()
    for _, row in within_trends.iterrows():
        years = np.linspace(0.0, min(2.0, float(row['max_observed_years'])), 81)
        relative_volume = np.exp(float(row['annual_log_volume_change_pct']) * years / 100.0)
        figure.add_trace(go.Scatter(
            x=years, y=relative_volume, mode='lines',
            name=f"{row['structure']}: {row['cohort_diagnosis']}",
            line=dict(color=COLORS.get(row['cohort_diagnosis'], '#777777'), width=3),
        ))
    figure.update_layout(
        title='Within-subject model: relative volume over the common first two years',
        xaxis_title='years from baseline', yaxis_title='modelled volume / baseline volume', template='plotly_white'
    )
    figure.show()

fig = px.box(subjects, x='cohort_diagnosis', y='signed_volume_change_pct_per_year', color='cohort_diagnosis',
             facet_col='structure_display', points='all', color_discrete_map=COLORS,
             title='Subject-mean signed annual volume change')
fig.update_layout(template='plotly_white', showlegend=False)
fig.update_yaxes(title='signed log-volume change (% / year)')
fig.show()

fig = px.scatter(pairs, x='midpoint_age_years', y='signed_volume_change_pct_per_year', color='cohort_diagnosis',
                 facet_col='structure_display', opacity=0.35, trendline='lowess', color_discrete_map=COLORS,
                 title='Adjacent-pair volume-change speed versus age')
fig.update_layout(template='plotly_white')
fig.show()
"""
        ),
        markdown("## 4. Age bins, disease-stage transitions, and CN-versus-AD contrasts"),
        code(
            """fig = px.line(age_rates, x='age_bin_center', y='mean_signed_volume_change_pct_per_year', color='cohort_diagnosis',
              facet_col='structure', markers=True, color_discrete_map=COLORS,
              title='Mean annualized volume change by midpoint-age bin')
fig.update_layout(template='plotly_white')
fig.update_xaxes(title='midpoint age (years)')
fig.update_yaxes(title='signed log-volume change (% / year)')
fig.show()

display(Markdown('### Subject-level diagnosis-rate contrasts'))
display(contrasts.sort_values(['structure', 'metric', 'group_a', 'group_b']))

if not transitions.empty:
    transition_plot = transitions.copy()
    transition_plot['transition'] = transition_plot['source_visit_diagnosis'].fillna('missing') + ' → ' + transition_plot['target_visit_diagnosis'].fillna('missing')
    fig = px.bar(transition_plot, x='transition', y='pair_count', color='cohort_diagnosis', facet_col='structure',
                 barmode='group', color_discrete_map=COLORS,
                 title='Observed visit-diagnosis transitions inside each baseline cohort')
    fig.update_layout(template='plotly_white')
    fig.show()
"""
        ),
        markdown("## 5. Correspondence-based CN-versus-AD local shape-speed maps"),
        code(
            """shape_path = analysis_dir / 'local_shape_speed_maps.npz'
if not shape_path.is_file():
    print('Shape map file is missing. Re-run the analysis without --skip-shape-maps.')
else:
    maps = np.load(shape_path)
    for structure in STRUCTURE_ORDER:
        face_key = f'{structure}__faces'
        vertex_key = f'{structure}__vertices'
        if face_key not in maps or vertex_key not in maps:
            continue
        faces, vertices = maps[face_key], maps[vertex_key]
        figure = make_subplots(rows=1, cols=2, specs=[[{'type': 'scene'}] * 2], subplot_titles=['CN', 'AD'])
        for column, diagnosis in enumerate(['CN', 'AD'], start=1):
            key = f'{structure}__{diagnosis}__signed_speed'
            if key not in maps:
                continue
            values = maps[key]
            maximum = float(np.nanmax(np.abs(values)))
            figure.add_trace(go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                                       i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], intensity=values,
                                       colorscale='RdBu', cmin=-maximum, cmax=maximum, colorbar=dict(title='mm/year'),
                                       showscale=(column == 2), name=diagnosis), row=1, col=column)
        figure.update_layout(title=f'{structure}: signed local normal speed (outward positive)', template='plotly_white', height=560)
        for scene_index in range(1, 3):
            name = 'scene' if scene_index == 1 else f'scene{scene_index}'
            figure.update_layout(**{name: dict(aspectmode='data')})
        figure.show()

        if f'{structure}__AD__signed_speed' in maps and f'{structure}__CN__signed_speed' in maps:
            difference = maps[f'{structure}__AD__signed_speed'] - maps[f'{structure}__CN__signed_speed']
            maximum = float(np.nanmax(np.abs(difference)))
            figure = go.Figure(go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                                         i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], intensity=difference,
                                         colorscale='RdBu', cmin=-maximum, cmax=maximum, colorbar=dict(title='mm/year')))
            figure.update_layout(title=f'{structure}: AD minus CN signed local shape speed', template='plotly_white', scene=dict(aspectmode='data'))
            figure.show()
"""
        ),
        markdown("## 6. Data-driven regional summary (not anatomical subfields)"),
        code(
            """if (analysis_dir / 'local_shape_speed_maps.npz').is_file():
    maps = np.load(analysis_dir / 'local_shape_speed_maps.npz')
    regional_rows = []
    for structure in STRUCTURE_ORDER:
        vertex_key = f'{structure}__vertices'
        if vertex_key not in maps:
            continue
        vertices = maps[vertex_key]
        # Geometry sectors are descriptive axes/tertiles only; they are not hippocampal or ventricular subfields.
        axis = vertices[:, 0]
        lo, hi = np.quantile(axis, [1/3, 2/3])
        sectors = np.where(axis < lo, 'low-x sector', np.where(axis > hi, 'high-x sector', 'middle-x sector'))
        for diagnosis in ['CN', 'AD']:
            key = f'{structure}__{diagnosis}__signed_speed'
            if key not in maps:
                continue
            for sector in np.unique(sectors):
                regional_rows.append({'structure': structure, 'diagnosis': diagnosis, 'geometry_sector': sector,
                                      'mean_signed_speed_mm_per_year': float(np.mean(maps[key][sectors == sector]))})
    regional = pd.DataFrame(regional_rows)
    display(regional)
    fig = px.bar(regional, x='geometry_sector', y='mean_signed_speed_mm_per_year', color='diagnosis', facet_col='structure',
                 barmode='group', color_discrete_map=COLORS,
                 title='Descriptive geometry-sector local speed (not anatomical subfields)')
    fig.update_layout(template='plotly_white')
    fig.show()
"""
        ),
        markdown(
            """## Interpretation safeguards

- Negative signed volume-rate means shrinkage; positive means enlargement. For the lateral ventricle, enlargement is expected to be clinically meaningful and should not be described as atrophy.
- Signed local speed is normal displacement after rigidly aligning each adjacent correspondence pair; positive is outward relative to the target surface normal.
- QC recommendations describe derived tables only. The original raw, minimal-smooth, and correspondence meshes are retained unchanged.
- The regional sectors are geometry-based descriptive partitions, not labelled hippocampal or ventricular subfields."""
        ),
    ]
    return notebook


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--notebook-output",
        type=Path,
        default=HERE / "visualize_synthseg_with_mci_volume_shape_speeds.ipynb",
    )
    args = parser.parse_args()
    import nbformat

    output = args.notebook_output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(build_notebook(), output)
    print(f"Wrote notebook: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
