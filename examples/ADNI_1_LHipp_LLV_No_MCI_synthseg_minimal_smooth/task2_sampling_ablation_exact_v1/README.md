# Exact-SDF sampling ablation

This isolated experiment reuses the epoch-2500 `hippo_exact_z256_eikonal_warm`
multiresolution model and its 401 training latents. It deliberately does not
reuse source Adam state, learning-rate history, epoch number, best metrics, or
RNG state.

All runtime configurations, temporary files, caches, checkpoints, logs,
evaluations, meshes, comparison tables, and latent exports are restricted to:

```text
/mnt/bulk10tb/Deep3DComp/synthseg_qc_v1/left_hippocampus/sampling_ablation_exact_v1
```

The repository folder contains source code, tests, and the immutable experiment
matrix only.

## Experiments

```text
control_u030_n100   training bands 0.030 / 0.100
medium_u005_n030    training bands 0.005 / 0.030
tight_u002_n010     training bands 0.002 / 0.010
shell_stratified    explicit distance shells; run only after Phase 1
```

Validation and held-out latent fitting always retain the control sampling
distribution. The test split is rejected unless `--confirm-test` is explicitly
provided.

## Short checks

From the Deep3DComp repository root:

```bash
TASK=examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task2_sampling_ablation_exact_v1

bash "$TASK/scripts/run_fast_tests.sh"

/home/jakaria/anaconda3/envs/inr_sdf/bin/python \
  "$TASK/scripts/launch_sampling_ablation.py" smoke \
  --experiment control_u030_n100 --device cuda:1 --execute
```

The smoke command prepares an epoch-0 initialization checkpoint on the bulk
disk and performs only three optimizer steps. The trainer smoke path writes no
training history, trained checkpoint, mesh, or evaluation.

## Phase 1 training

Run one arm per available GPU/screen if desired. GPU 1 commands are:

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python "$TASK/scripts/launch_sampling_ablation.py" train --experiment control_u030_n100 --device cuda:1 --execute
/home/jakaria/anaconda3/envs/inr_sdf/bin/python "$TASK/scripts/launch_sampling_ablation.py" train --experiment medium_u005_n030 --device cuda:1 --execute
/home/jakaria/anaconda3/envs/inr_sdf/bin/python "$TASK/scripts/launch_sampling_ablation.py" train --experiment tight_u002_n010 --device cuda:1 --execute
```

Each run performs 300 fine-tuning epochs. A full fixed 100-scan validation/PCA
evaluation is launched at epoch 300. For an interrupted target run:

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python "$TASK/scripts/launch_sampling_ablation.py" train --experiment medium_u005_n030 --device cuda:1 --resume latest --execute
```

## Manual validation and comparison

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python "$TASK/scripts/launch_sampling_ablation.py" evaluate --experiment medium_u005_n030 --checkpoint best_mesh --splits val --device cuda:1 --execute

bash "$TASK/scripts/run_on_bulk.sh" "$TASK/scripts/compare_sampling_runs.py" \
  --baseline /mnt/bulk10tb/Deep3DComp/synthseg_qc_v1/left_hippocampus/sampling_ablation_exact_v1/runs/control_u030_n100/periodic_evaluation/epoch_0300/per_scan_metrics.csv \
  --candidate medium=/mnt/bulk10tb/Deep3DComp/synthseg_qc_v1/left_hippocampus/sampling_ablation_exact_v1/runs/medium_u005_n030/periodic_evaluation/epoch_0300/per_scan_metrics.csv \
  --candidate tight=/mnt/bulk10tb/Deep3DComp/synthseg_qc_v1/left_hippocampus/sampling_ablation_exact_v1/runs/tight_u002_n010/periodic_evaluation/epoch_0300/per_scan_metrics.csv \
  --output-dir /mnt/bulk10tb/Deep3DComp/synthseg_qc_v1/left_hippocampus/sampling_ablation_exact_v1/comparisons/phase1_val
```

Measure how much a candidate's 401 training codes moved from the shared source:

```bash
bash "$TASK/scripts/run_on_bulk.sh" "$TASK/scripts/analyze_latent_drift.py" \
  --source /mnt/bulk10tb/Deep3DComp/synthseg_qc_v1/left_hippocampus/inr_multires_single_field_v1/runs/hippo_exact_z256_eikonal_warm/checkpoints/best_mesh.pth \
  --candidate /mnt/bulk10tb/Deep3DComp/synthseg_qc_v1/left_hippocampus/sampling_ablation_exact_v1/runs/medium_u005_n030/checkpoints/best_mesh.pth \
  --output-dir /mnt/bulk10tb/Deep3DComp/synthseg_qc_v1/left_hippocampus/sampling_ablation_exact_v1/comparisons/medium_latent_drift
```

## Evaluation-ceiling pilot

This is intentionally a user-launched, longer evaluation. It evaluates the
source checkpoint on the same first 20 fixed validation scans at resolutions
256, 384, and 512 and at 500, 1000, and 2000 latent-fit steps:

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python "$TASK/scripts/launch_sampling_ablation.py" ceiling --device cuda:1 --execute

bash "$TASK/scripts/run_on_bulk.sh" "$TASK/scripts/summarize_ceiling_pilot.py" \
  --root /mnt/bulk10tb/Deep3DComp/synthseg_qc_v1/left_hippocampus/sampling_ablation_exact_v1/ceiling_pilot
```

## Final test and latent export

Only after selecting a model using validation results:

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python "$TASK/scripts/launch_sampling_ablation.py" evaluate --experiment medium_u005_n030 --checkpoint best_mesh --splits test --device cuda:1 --confirm-test --execute

/home/jakaria/anaconda3/envs/inr_sdf/bin/python "$TASK/scripts/launch_sampling_ablation.py" export --experiment medium_u005_n030 --checkpoint best_mesh --splits train val test --steps 500 --device cuda:1 --confirm-test --execute

/home/jakaria/anaconda3/envs/inr_sdf/bin/python "$TASK/scripts/launch_sampling_ablation.py" audit-export --experiment medium_u005_n030 --checkpoint best_mesh --require-all-splits --execute
```
