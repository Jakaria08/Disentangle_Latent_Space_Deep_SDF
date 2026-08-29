# Runbook

Run commands from the task's `scripts` directory:

```bash
cd /home/jakaria/INR/Deep3DComp/examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task5_direct_mesh_cocycle_spiral_unet_v1/scripts
```

Use this tested environment:

```bash
PYTHON=/home/jakaria/anaconda3/envs/pytorch_geo/bin/python
```

All persistent outputs go to
`/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task5_direct_mesh_cocycle_spiral_unet_v1`.

## 1. Verify prepared data

The cache is already prepared. This is read-only and checks split isolation,
counts, hierarchy sizes, and training-only normalization statistics.

```bash
$PYTHON prepare_data.py --verify-only
```

Do not add `--overwrite` unless intentionally rebuilding the isolated cache.

## 2. Tests and contract validation

```bash
$PYTHON run_tests.py
$PYTHON validate_experiment.py --device cpu
$PYTHON validate_experiment.py --device cuda:1
```

Optional one-epoch GPU 1 training smokes:

```bash
$PYTHON train.py --config ../configs/spiral_direct_c4_velocity_v2_s42.json --device cuda:1 --run-name smoke_spiral_velocity_v2_manual --smoke
$PYTHON train.py --config ../configs/adaptive_direct_c4_velocity_v2_s42.json --device cuda:1 --run-name smoke_adaptive_velocity_v2_manual --smoke
```

The Adaptive smoke fails deliberately if any of its three support predictors
does not update.

## 3. Recommended long job: focused velocity-v2 Optuna searches on GPU 1

Run Spiral first, inspect it, and then run Adaptive. The first trial is the
completed v1 trial-13 setting, providing an explicit endpoint reference inside
each new study. The remaining trials explore only the focused ranges justified
by v1: base width 32/48, velocity weight 0.05/0.1/0.2/0.3, dropout, learning
rate, small weight decay, smoothness, and Adaptive initial support.

```bash
./run_optuna_velocity_v2_gpu1.sh spiral main_v2 16 80 384
```

```bash
./run_optuna_velocity_v2_gpu1.sh adaptive main_v2 16 80 384
```

Equivalent explicit commands are:

```bash
$PYTHON optuna_search_v2.py --operator spiral --device cuda:1 --study-tag main_v2 --n-trials 16 --trial-epochs 80 --trial-samples-per-epoch 384
```

```bash
$PYTHON optuna_search_v2.py --operator adaptive --device cuda:1 --study-tag main_v2 --n-trials 16 --trial-epochs 80 --trial-samples-per-epoch 384
```

The studies are resumable. Repeating a command adds the requested number of
trials. Spiral and Adaptive outputs never share a database:

```text
/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task5_direct_mesh_cocycle_spiral_unet_v1/optuna/spiral_direct_c4_velocity_main_v2/
/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task5_direct_mesh_cocycle_spiral_unet_v1/optuna/adaptive_direct_c4_velocity_main_v2/
```

Each directory contains `study.db`, `trials.csv`, `best_trial.json`, and a
200-epoch `best_config.json`. Pruning waits until epoch 15, after the velocity
and consistency ramp, so it does not select only for fast endpoint convergence.

For a wiring-only check, add `--smoke`. The `_smoke` suffix is automatic.

## 4. Optional legacy endpoint-v1 search

The completed Spiral `main_v1` study should be retained as the endpoint-only
reference. These commands are provided for reproducibility; do not mix their
SQLite databases or scores with velocity-v2.

Run these sequentially. Each command creates a resumable SQLite database and
adds 30 trials. Repeating the same command adds 30 more trials to that study.

```bash
$PYTHON optuna_search.py --operator spiral --device cuda:1 --study-tag main_v1 --n-trials 30 --trial-epochs 40 --trial-samples-per-epoch 256
```

```bash
$PYTHON optuna_search.py --operator adaptive --device cuda:1 --study-tag main_v1 --n-trials 30 --trial-epochs 40 --trial-samples-per-epoch 256
```

The best full-training configurations will be:

```text
/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task5_direct_mesh_cocycle_spiral_unet_v1/optuna/spiral_direct_c4_main_v1/best_config.json
/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task5_direct_mesh_cocycle_spiral_unet_v1/optuna/adaptive_direct_c4_main_v1/best_config.json
```

## 5. Long job: train the selected velocity-v2 configurations

Seed 42 is the primary matched comparison:

```bash
$PYTHON train.py --config /mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task5_direct_mesh_cocycle_spiral_unet_v1/optuna/spiral_direct_c4_velocity_main_v2/best_config.json --device cuda:1 --seed 42 --run-name direct_mesh_spiral_velocity_main_v2_s42
```

```bash
$PYTHON train.py --config /mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task5_direct_mesh_cocycle_spiral_unet_v1/optuna/adaptive_direct_c4_velocity_main_v2/best_config.json --device cuda:1 --seed 42 --run-name direct_mesh_adaptive_velocity_main_v2_s42
```

If interrupted, repeat the corresponding command with `--resume`. `--epochs`
is the desired total epoch number, not the number of additional epochs.

For uncertainty across initialization, repeat both commands with matched seeds
314 and 1701 and use run names ending `_s314` and `_s1701`.

To train the endpoint-v1 Spiral reference under the same 200-epoch/512-sample
budget:

```bash
$PYTHON train.py --config /mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task5_direct_mesh_cocycle_spiral_unet_v1/optuna/spiral_direct_c4_main_v1/best_config.json --device cuda:1 --seed 42 --run-name direct_mesh_spiral_endpoint_main_v1_s42
```

## 6. Freeze on validation, then evaluate test once

First evaluate both selected checkpoints on validation. Fast metrics use all
pairs; expensive surface metrics use the same balanced 128-pair subset. Vertex
maps are averaged over all evaluated pairs.

```bash
$PYTHON evaluate.py --checkpoint /mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task5_direct_mesh_cocycle_spiral_unet_v1/runs/direct_mesh_spiral_velocity_main_v2_s42/checkpoints/best.pt --split val --device cuda:1 --surface-metrics --surface-max-pairs 128 --surface-points 3000 --save-vertex-maps
```

```bash
$PYTHON evaluate.py --checkpoint /mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task5_direct_mesh_cocycle_spiral_unet_v1/runs/direct_mesh_adaptive_velocity_main_v2_s42/checkpoints/best.pt --split val --device cuda:1 --surface-metrics --surface-max-pairs 128 --surface-points 3000 --save-vertex-maps
```

After the models and analysis settings are locked, replace `--split val` with
`--split test` in both commands. Omit `--surface-max-pairs 128` only if exact
surface metrics are required for every test pair and the longer runtime is
acceptable.

## 7. Instantaneous velocity by diagnosis and age

Run this on validation first. The default intervals are 70--<75, 75--<80, and
80--95 years, where both CN and AD have useful subject support. The plots call
the fitted longitudinal-visit derivative **Observed velocity**; it is not
presented as a directly measured continuous-time derivative.

```bash
$PYTHON analyze_velocity_by_age.py --checkpoint /mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task5_direct_mesh_cocycle_spiral_unet_v1/runs/direct_mesh_spiral_velocity_main_v2_s42/checkpoints/best.pt --split val --device cuda:1 --age-bins 70 75 80 95 --bootstrap-samples 5000
```

```bash
$PYTHON analyze_velocity_by_age.py --checkpoint /mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task5_direct_mesh_cocycle_spiral_unet_v1/runs/direct_mesh_adaptive_velocity_main_v2_s42/checkpoints/best.pt --split val --device cuda:1 --age-bins 70 75 80 95 --bootstrap-samples 5000
```

Each command writes `per_visit_velocity.csv`, `age_velocity_summary.csv`,
`summary.json`, and publication-ready PNG/PDF figures under
`evaluation/val/age_velocity/`. Use `--max-scans 8 --bootstrap-samples 20` only
for a quick wiring smoke. After choices are frozen, replace `--split val` with
`--split test`; do not use test results to revise the bins or model.

## 8. Paired subject-bootstrap comparison

```bash
$PYTHON compare_models.py --spiral-run direct_mesh_spiral_velocity_main_v2_s42 --adaptive-run direct_mesh_adaptive_velocity_main_v2_s42 --split test --bootstrap-samples 10000
```

The report uses the subject as the independent unit, requires identical pairs,
and applies the correct direction for each metric: lower for errors/distances
and higher for Dice/normal cosine.

## 9. Load-only results notebook

Generate the notebook after any code update:

```bash
$PYTHON make_notebook.py
```

Open `../notebooks/direct_mesh_results.ipynb`, edit only the two run names if
needed, and run all cells. It performs no training, model inference, surface
sampling, or bootstrap resampling.

Optional headless cell check:

```bash
MPLCONFIGDIR=/tmp/mpl-task5-direct $PYTHON check_notebook_cells.py --spiral-run direct_mesh_spiral_velocity_main_v2_s42 --adaptive-run direct_mesh_adaptive_velocity_main_v2_s42 --split test
```
