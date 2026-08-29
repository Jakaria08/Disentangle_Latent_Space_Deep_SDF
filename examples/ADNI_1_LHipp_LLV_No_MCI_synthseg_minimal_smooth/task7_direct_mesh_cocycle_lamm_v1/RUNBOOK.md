# Runbook

Run commands from `/home/jakaria/INR/Deep3DComp`.

The `inr_sdf` environment is used for Optuna because it contains Optuna 4.7. The model has
no torch-scatter dependency.

```bash
TASK=examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task7_direct_mesh_cocycle_lamm_v1
PY=/home/jakaria/anaconda3/envs/inr_sdf/bin/python
```

## Fast verification

```bash
$PY "$TASK/scripts/run_tests.py"
$PY "$TASK/scripts/validate_experiment.py" --device cuda:1
```

If CUDA is temporarily unavailable, use `--device cpu` for validation.

End-to-end smoke training on GPU 1:

```bash
$PY "$TASK/scripts/train.py" \
  --config "$TASK/configs/lamm_direct_c4_z128_s42.json" \
  --device cuda:1 \
  --run-name direct_mesh_lamm_z128_gpu1_smoke \
  --smoke
```

The smoke run fails if any of the conditioning, tokenizer, encoder, down projection, latent
flow, up projection, decoder tokens, decoder, or velocity heads receives no gradient.

## Optuna search on GPU 1

The search jointly compares 128D/256D allocation, token width, encoder/decoder depth,
conditioning width, latent-flow width/depth, dropout, learning rate, weight decay, velocity
loss, and smoothness loss.

```bash
$PY "$TASK/scripts/optuna_search.py" \
  --device cuda:1 \
  --n-trials 18 \
  --trial-epochs 80 \
  --trial-samples-per-epoch 384 \
  --study-tag main_v1
```

The resumable SQLite study and exports are written under:

```text
/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task7_direct_mesh_cocycle_lamm_v1/optuna/
```

The selected configuration is `best_config.json` in the study directory. Re-running the
same command continues the existing study.

## Matched final-width runs

Run 128D:

```bash
$PY "$TASK/scripts/train.py" \
  --config "$TASK/configs/lamm_direct_c4_z128_s42.json" \
  --device cuda:1
```

Run the controlled 256D equal split:

```bash
$PY "$TASK/scripts/train.py" \
  --config "$TASK/configs/lamm_direct_c4_z256_equal_s42.json" \
  --device cuda:1
```

Run the 256D fine-heavy allocation:

```bash
$PY "$TASK/scripts/train.py" \
  --config "$TASK/configs/lamm_direct_c4_z256_fine_s42.json" \
  --device cuda:1
```

Resume any interrupted run by adding `--resume` with the same configuration and run name.

To train the Optuna-selected configuration:

```bash
$PY "$TASK/scripts/train.py" \
  --config /mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task7_direct_mesh_cocycle_lamm_v1/optuna/lamm_direct_c4_main_v1/best_config.json \
  --device cuda:1
```

## Validation and test evaluation

Use the selected `best.pt`, never `latest.pt`, for reported results.

```bash
CKPT=/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task7_direct_mesh_cocycle_lamm_v1/runs/direct_mesh_lamm_c4_z128_s42/checkpoints/best.pt

$PY "$TASK/scripts/evaluate.py" \
  --checkpoint "$CKPT" \
  --split val \
  --device cuda:1 \
  --batch-size 8

$PY "$TASK/scripts/evaluate.py" \
  --checkpoint "$CKPT" \
  --split test \
  --device cuda:1 \
  --batch-size 8 \
  --surface-metrics \
  --surface-max-pairs 100 \
  --save-vertex-maps
```

Expensive ASSD/HD95/Dice metrics use a deterministic balanced subset when
`--surface-max-pairs` is supplied. Correspondence endpoint, volume, instantaneous velocity,
and cocycle metrics are still reported by the evaluator.

## Instantaneous velocity by age and diagnosis

Inspect available options:

```bash
$PY "$TASK/scripts/analyze_velocity_by_age.py" --help
```

Then run it with the selected checkpoint using the same arguments as the direct Spiral task.
The quantity is a predicted surface velocity in mm/year:

```text
V(X,a,a,d)
```

The comparison target is the reliability-weighted velocity estimate fitted from repeated
observed visits; it is not direct physical ground truth.

## Custom data/output roots

The default data root is the validated task-5 cache. Override it without copying data:

```bash
export DEEP3DCOMP_DIRECT_LAMM_DATA_ROOT=/absolute/path/to/prepared/direct_mesh_root
export DEEP3DCOMP_DIRECT_LAMM_ROOT=/absolute/path/to/task7_outputs
```

