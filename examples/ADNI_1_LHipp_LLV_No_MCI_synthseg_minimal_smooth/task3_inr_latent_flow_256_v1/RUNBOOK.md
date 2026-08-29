# ADNI left-hippocampus INR-256 latent-flow comparison

This is an isolated 256-D follow-up to `task3_latent_flow_128_v1`. It uses the
frozen `hippo_exact_z256_eikonal_warm` multiresolution SDF decoder and its
601 exported scan latents. The PCA-128, SpiralNet++-128, and Adaptive-Spiral-128
tasks and outputs are read-only.

The three matched transports are direct C4 cocycle flow, plain neural ODE, and
singleton-attention BrainODE. Only direct C4 places the frozen decoder in the
optimizer graph. Its former fixed-topology vertex loss is replaced by exact-SDF
supervision at deterministic per-scan query points. Plain ODE and BrainODE keep
the latent-trajectory-MSE-only contract from the 128-D task.

Persistent outputs live below:

`/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task3_inr_latent_flow_256_v1`

All commands are run from the repository root with:

```bash
TASK=examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task3_inr_latent_flow_256_v1
PY=/home/jakaria/anaconda3/envs/inr_sdf/bin/python
```

## Preparation and contract checks

```bash
$PY $TASK/scripts/prepare_inr256.py --dry-run --device cpu
$PY $TASK/scripts/prepare_inr256.py
$PY $TASK/tests/verify_contracts.py
```

## Smoke tests

```bash
$PY $TASK/scripts/train_c4.py --config $TASK/configs/inr256_direct_c4_s42.json --device cuda:0 --dry-run
$PY $TASK/scripts/train_latent_ode.py --config $TASK/configs/inr256_plain_ode_s42.json --device cuda:0 --dry-run
$PY $TASK/scripts/train_latent_ode.py --config $TASK/configs/inr256_brainode_s42.json --device cuda:0 --dry-run
```

## Full runs

```bash
$PY $TASK/scripts/train_c4.py --config $TASK/configs/inr256_direct_c4_s42.json --device cuda:0
$PY $TASK/scripts/train_latent_ode.py --config $TASK/configs/inr256_plain_ode_s42.json --device cuda:0
$PY $TASK/scripts/train_latent_ode.py --config $TASK/configs/inr256_brainode_s42.json --device cuda:0
```

Training refuses to overwrite an existing run unless `--resume` is supplied.
Only train and validation archives are opened by the trainers.

## Evaluation and comparison

```bash
$PY $TASK/scripts/evaluate.py \
  --run-dir /mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task3_inr_latent_flow_256_v1/training/inr256/direct_c4/inr256_direct_c4_s42 \
  --split test --device cuda:0 \
  --evaluation-name evaluation_mc256_exact \
  --surface-resolution 256 --surface-samples 30000

$PY $TASK/scripts/compare_previous.py \
  --inr-run /mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task3_inr_latent_flow_256_v1/training/inr256/direct_c4/inr256_direct_c4_s42 \
  --inr-evaluation-name evaluation_mc256_exact \
  --output-name comparison_previous_128d_exact

$PY $TASK/scripts/analyze_volume_trends.py \
  --run-dir /mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task3_inr_latent_flow_256_v1/training/inr256/direct_c4/inr256_direct_c4_s42 \
  --device cuda:0 --surface-resolution 256 --bootstrap-samples 2000 \
  --exact-evaluation-name evaluation_mc256_exact \
  --output-name volume_trend_analysis_mc256
```

The evaluator reports the no-change baseline, sampled exact-SDF transport
ratios, latent errors, soft-volume/rate errors, cocycle/inverse defects, and
first-to-last marching-cubes surface metrics. Test is opened only after the
validation-selected checkpoint has been fixed.

The final command adds first-to-last AD-versus-CN signed volume trends, annual
percent change, rate MAE, atrophy-direction agreement, bootstrap intervals, the
AD-minus-CN rate gap, and a trend comparison with the earlier PCA, SpiralNet++,
and Adaptive-Spiral cocycle runs.
