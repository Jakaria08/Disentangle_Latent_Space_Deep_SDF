# Task 1: ADNI No-MCI Left Smooth Longitudinal Manifest

This folder is an isolated data-contract layer for the BrainODE comparison.
It reads existing ADNI metadata, smooth No-MCI split JSON files, smooth meshes,
and smooth SDF samples, then writes cleaned CSV/JSON artifacts for later tasks.

No existing dataset files, split files, specs, meshes, or SDF samples are edited.

## Inputs

- `examples/splits/splits_left_hippocampus_ADNI_smooth_No_MCI/*.json`
- `/home/jakaria/ADNI/ADNI_1/adni_processed/adni_metadata_filtered.csv`
- `/home/jakaria/ADNI/ADNI_1/adni_processed/left_hippocampus_correspondence/minimal_smooth_scaled_obj_files`
- `/home/jakaria/ADNI/ADNI_1/adni_processed/left_hippocampus_correspondence/sdf_data_smooth/SdfSamples/minimal_smooth_scaled_obj_files`

## Outputs

- `metadata/adni_no_mci_left_smooth_master.csv`
- `metadata/adni_no_mci_left_smooth_clean.csv`
- `metadata/adni_no_mci_left_smooth_subject_trajectories.json`
- `metadata/age_norm_stats.json`
- `metadata/split_summary.json`
- `metadata/cleaning_report.json`
- `metadata/validation_report.json`
- `splits/train_clean.json`
- `splits/val_clean.json`
- `splits/test_clean.json`
- `pairs/pairs_train.csv`
- `pairs/pairs_val.csv`
- `pairs/pairs_test.csv`
- `pairs/pairs_all.csv`
- `triplets/triplets_train.csv`
- `triplets/triplets_val.csv`
- `triplets/triplets_test.csv`
- `triplets/triplets_all.csv`

## Cleaning Rule

Visits are grouped by visit month derived from the ADNI visit label:

- `sc`, `bl`: month 0
- `m06`: month 6
- `m12`: month 12
- other `mXX`: month XX

Because the available ADNI age field is an integer year, scans with the same
age but different visit labels are not treated as duplicate visits. This avoids
dropping valid six-month follow-up scans that share the same integer age.

When multiple scans map to the same subject and visit month, the kept scan is
chosen by:

1. existing mesh and SDF sample;
2. `bl` over `sc` for month 0;
3. lower numeric image ID.

Subjects with fewer than two cleaned visits are excluded from forecasting
splits, pairs, and triplets.

## Commands

```bash
python examples/ADNI_1_L_No_MCI/brainode_comparison_task1_manifest/scripts/build_task1_manifest.py
python examples/ADNI_1_L_No_MCI/brainode_comparison_task1_manifest/scripts/validate_task1_manifest.py
```
