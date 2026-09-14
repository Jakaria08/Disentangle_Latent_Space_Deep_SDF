# Runbook

Run from `/home/jakaria/INR/Deep3DComp`:

```bash
TASK=examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task7_direct_mesh_cocycle_lamm_v1
PY=/home/jakaria/anaconda3/envs/inr_sdf/bin/python
```

## Verify

```bash
$PY "$TASK/scripts/run_tests.py"
$PY "$TASK/scripts/validate_experiment.py" --device cuda:1
```

## Three training commands

Global 256D:

```bash
$PY "$TASK/scripts/train.py" \
  --config "$TASK/configs/lamm_global_c4_z256_s42.json" \
  --device cuda:1
```

Global 384D:

```bash
$PY "$TASK/scripts/train.py" \
  --config "$TASK/configs/lamm_global_c4_z384_s42.json" \
  --device cuda:1
```

Regional tokens, no global bottleneck:

```bash
$PY "$TASK/scripts/train.py" \
  --config "$TASK/configs/lamm_token_c4_s42.json" \
  --device cuda:1
```

They run sequentially with `bash "$TASK/scripts/run_final_gpu1.sh"`. Add `--resume` to an
individual command after interruption. Training reads train/validation only and writes to:

```text
/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task7_direct_mesh_cocycle_lamm_v1/runs/
```

Use `checkpoints/best.pt`, never `latest.pt`, in a reported comparison.

## Optional Optuna architecture/hyperparameter search

The search includes all three state designs and tunes token width, encoder/decoder depth,
condition width, global-flow or token-flow depth, dropout, optimization, fitted-velocity
weight, and spatial smoothness. Its first three queued trials are the three fixed anchors.

```bash
$PY "$TASK/scripts/optuna_search.py" \
  --device cuda:1 \
  --n-trials 18 \
  --trial-epochs 80 \
  --trial-samples-per-epoch 384 \
  --study-tag architecture_main_v2
```

The SQLite study is resumable by running exactly the same command. Train its exported
`best_config.json` with `scripts/train.py` only after inspecting the completed trials.

## One centralized validation comparison

After all three new runs finish:

```bash
$PY "$TASK/scripts/evaluate_all.py" \
  --manifest "$TASK/configs/central_evaluation.json" \
  --split val \
  --device cuda:1 \
  --surface-points 10000 \
  --bootstrap-samples 2000 \
  --output-dir /mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task7_direct_mesh_cocycle_lamm_v1/comparisons/all_methods_val_v1
```

This command evaluates:

- LAMM global 256D, global 384D, and regional-token direct mesh flows;
- selected Spiral and Adaptive-Spiral direct mesh flows;
- current 128D LAMM latent flow, including decoder-JVP surface velocity.

It writes native per-method results plus `comparison.csv` and `comparison.json`. The primary
ranking is overall first-to-last mean vertex error; no-change-normalized, surface, volume,
cocycle, and velocity metrics remain separate columns and should all be inspected.

Test is a separate, explicitly authorized final evaluation:

```bash
$PY "$TASK/scripts/evaluate_all.py" \
  --manifest "$TASK/configs/central_evaluation.json" \
  --split test \
  --allow-test \
  --device cuda:1 \
  --surface-points 10000 \
  --bootstrap-samples 2000 \
  --output-dir /mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task7_direct_mesh_cocycle_lamm_v1/comparisons/all_methods_test_v1
```

For a command-only check before training, add `--dry-run`. For a quick partial check, use
`--only METHOD`, `--max-subjects N`, `--max-velocity-scans N`, and fewer surface points.
