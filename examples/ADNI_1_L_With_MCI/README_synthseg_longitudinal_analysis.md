# SynthSeg strict-no-MCI longitudinal QC and CN-versus-AD analysis

These tools read the existing SynthSeg raw, minimal-smooth, and final physical-unit correspondence meshes for the left hippocampus and left lateral ventricle. They never repair, overwrite, move, or delete meshes.

The full correspondence run contains 4,155 scan IDs, including 334 baseline-MCI participants. By default, these tools retain only subjects with a baseline `CN` or `AD` diagnosis and no MCI-labelled meshed visit. This strict CN/AD cohort has **2,715 scan IDs** and **5,430 structure–scan records** (two structures per scan). Baseline-MCI converters/reverters and unknown baseline groups are excluded.

Run from the repository root with the INR environment:

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python \
  examples/ADNI_1_L_With_MCI/synthseg_longitudinal_qc.py
```

The command shows `[scan QC]` and `[pair shape QC]` progress in the terminal. It writes review-only reports for the strict no-MCI cohort to `examples/ADNI_1_L_With_MCI/synthseg_mesh_qc/`:

- `scan_qc.csv`, `adjacent_pair_qc.csv`, and `subject_qc.csv` contain all evidence and recommendations.
- `review_scans.csv`, `bad_adjacent_pairs.csv`, and `review_subjects.csv` are prioritized review lists.
- `records_after_drop_review_scans.csv` and `records_after_drop_excluded_subjects.csv` are derived, filtered metadata only; no meshes are removed.
- `index.html` and `top_pair_overlays/` provide visual inspection of the most unusual adjacent pairs.

Then create the complete analysis artifacts. Start with all scans to understand the data and QC effects:

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python \
  examples/ADNI_1_L_With_MCI/synthseg_longitudinal_analysis.py \
  --qc-policy all
```

For a sensitivity analysis that excludes only subjects flagged as severe longitudinal failures, run:

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python \
  examples/ADNI_1_L_With_MCI/synthseg_longitudinal_analysis.py \
  --qc-policy drop-excluded-subjects \
  --output-dir examples/ADNI_1_L_With_MCI/synthseg_longitudinal_analysis_excluded_subjects
```

`--qc-policy drop-review-scans` is a stricter analysis that removes every review-marked scan from derived tables. It does not change any mesh data.

If only the tabular analysis changed and the existing correspondence shape maps are still valid, use `--reuse-existing-shape-maps` to avoid rebuilding them.

The analysis displays `[local shape maps]` terminal progress and writes:

- visit and adjacent-pair volume metrics in physical mm³ and signed annual log-volume percent;
- cohort trajectory, age-bin, subject-rate, and clinical-transition summaries;
- within-subject longitudinal volume trends that remove participant-specific baseline-volume differences and avoid late-visit survivor-composition bias;
- CN-versus-AD subject-level contrasts with bootstrap confidence intervals, Cohen's d, and Welch test p-values;
- `local_shape_speed_maps.npz` plus cohort local signed/absolute correspondence-speed summaries.

Finally, generate the notebook (already generated once in this folder) if you need a fresh copy:

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python \
  examples/ADNI_1_L_With_MCI/create_synthseg_with_mci_analysis_notebook.py
```

Open and run `visualize_synthseg_with_mci_volume_shape_speeds.ipynb`. It shows QC evidence, raw/smooth/correspondence-volume lineage, individual and cohort trajectories, speed-versus-age curves, disease transitions, CN-versus-AD contrasts, correspondence local shape-speed maps, and clearly labelled descriptive geometry sectors.

To deliberately analyze the broader dataset later (including baseline-MCI participants), pass `--cohort-filter all`. That is a separate analysis and should not be mixed with the strict no-MCI CN-versus-AD trend.

For the lateral ventricle, a positive signed volume rate means enlargement; it must not be relabelled as atrophy. Geometry sectors in the notebook are not anatomical subfields.
