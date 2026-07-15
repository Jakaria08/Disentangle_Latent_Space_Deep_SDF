# Core BrainODE Dataset

This folder holds the dataset packaging step for the direct BrainODE comparison on the cleaned original ADNI left hippocampus CN/AD cohort.

Part 1 creates a new Task 3 dataset without modifying Task 1 or Task 2:

- `audit/input_audit.json`: verifies Task 1 clean trajectories and Task 2 PCA exports are complete and internally consistent.
- `metadata/core_brainode_scan_manifest.csv`: one scan per row with continuous age, fixed CN/AD condition value, and PCA coefficient path.
- `metadata/core_brainode_subject_sequences.json`: human-readable subject trajectories with ordered visits.
- `dataset/*_subject_sequences.npz`: packed arrays for all/train/val/test subject sequences, using PCA-150 as the primary BrainODE representation and retaining PCA-256 for checks.
- `metadata/validation_dataset.json`: validation report for the generated dataset files.

Part 2 adds the core BrainODE attention ODE model and trainer:

- `scripts/brainode_model.py`: BrainODE-style `ODEFuncWithAttention` plus a local differentiable RK4 integrator.
- `scripts/train_core_brainode.py`: trains the PCA-150 ODE on all combinatorial forward and backward subject trajectories, with the same shared noise and scale augmentation pattern used in the public BrainODE training code.
- `training/<run_name>/checkpoints/best.pth`: validation-selected checkpoint.
- `training/<run_name>/history.jsonl`: one JSON row per epoch.
- `training/<run_name>/training_status.json`: run status, best epoch, and latest metrics summary.

Run order:

```bash
python examples/ADNI_1_L_No_MCI/brainode_comparison_task3_core_brainode_original/scripts/audit_inputs.py
python examples/ADNI_1_L_No_MCI/brainode_comparison_task3_core_brainode_original/scripts/build_brainode_dataset.py
python examples/ADNI_1_L_No_MCI/brainode_comparison_task3_core_brainode_original/scripts/validate_core_brainode.py --stage dataset
```

Core BrainODE training:

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python \
  examples/ADNI_1_L_No_MCI/brainode_comparison_task3_core_brainode_original/scripts/train_core_brainode.py \
  --device cuda:0
```

Fast smoke test:

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python \
  examples/ADNI_1_L_No_MCI/brainode_comparison_task3_core_brainode_original/scripts/train_core_brainode.py \
  --device cpu \
  --epochs 1 \
  --batch-size 4 \
  --run-name smoke_test \
  --max-train-trajectories 16 \
  --max-val-trajectories 8 \
  --disable-tensorboard
```

The continuous time variable follows the planned BrainODE adaptation:

- `continuous_age_years = baseline_age_years + months_from_baseline / 12`
- `continuous_age_norm = (continuous_age_years - age_min_train) / (age_max_train - age_min_train)`

The cognition condition for this core version is fixed directly from diagnosis:

- `CN -> 0.0`
- `AD -> 1.0`

This trainer does not depend on `torchdiffeq`. The environment available in this repo did not include that package, so the RK4 integration is implemented locally in `brainode_model.py` while keeping the same method choice as the public BrainODE code.
