# Large Strict-Left PCA-150 BrainODE

This experiment runs BrainODE on the large strict no-MCI left hippocampus cohort using the same QC-clean stable-subject metadata as the SIREN flow comparison.

Key choices:

- Shape representation: mesh PCA coefficients from Task 2, using PCA-150 for BrainODE and retaining PCA-256 for validation.
- Time coordinate: `continuous_age_norm` copied from the QC-clean SIREN flow metadata.
- Condition: fixed scalar CN/AD label, `CN -> 0.0`, `AD -> 1.0`.
- Conversion subjects: excluded by the QC/stable metadata, matching the current SIREN flow contract.

## Run Order

From repo root:

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python \
  examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/brainode_pca150_qc_stable/scripts/prepare_large_qc_inputs.py

/home/jakaria/anaconda3/envs/inr_sdf/bin/python \
  examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/brainode_pca150_qc_stable/scripts/audit_inputs.py

/home/jakaria/anaconda3/envs/inr_sdf/bin/python \
  examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/brainode_pca150_qc_stable/scripts/build_brainode_dataset.py

/home/jakaria/anaconda3/envs/inr_sdf/bin/python \
  examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/brainode_pca150_qc_stable/scripts/validate_core_brainode.py \
  --stage dataset
```

Smoke test:

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python \
  examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/brainode_pca150_qc_stable/scripts/train_core_brainode.py \
  --device cpu \
  --epochs 1 \
  --batch-size 4 \
  --run-name smoke_test \
  --max-train-trajectories 16 \
  --max-val-trajectories 8 \
  --disable-tensorboard
```

Full training:

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python \
  examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/brainode_pca150_qc_stable/scripts/train_core_brainode.py \
  --device cuda:0
```

Evaluation and comparison:

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python \
  examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/brainode_pca150_qc_stable/scripts/evaluate_brainode.py \
  --checkpoint best \
  --split all \
  --pairwise \
  --device cuda:0

/home/jakaria/anaconda3/envs/inr_sdf/bin/python \
  examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/brainode_pca150_qc_stable/scripts/evaluate_brainode.py \
  --checkpoint best \
  --split all \
  --first-last-only \
  --output-dir examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/brainode_pca150_qc_stable/training/core_attention_pca150_qc_stable/evaluation/best_first_last \
  --device cuda:0

/home/jakaria/anaconda3/envs/inr_sdf/bin/python \
  examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/brainode_pca150_qc_stable/scripts/analyze_brainode_comparison.py
```

Main outputs:

- `metadata/dataset_summary.json`
- `training/core_attention_pca150_qc_stable/history.json`
- `training/core_attention_pca150_qc_stable/evaluation/best/all_trajectory_metrics.csv`
- `analysis/brainode_comparison/brainode_summary.csv`
- `analysis/brainode_comparison/brainode_vs_siren_flow.csv`

The comparison CSV intentionally keeps metrics side by side rather than pretending they are identical: BrainODE reports PCA inverse-transform vertex MAE, while the SIREN flow analysis reports decoder SDF L1.
