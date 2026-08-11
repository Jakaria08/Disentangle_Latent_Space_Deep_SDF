# Matched 256-D SIREN v5 flow, ODE, and BrainODE experiment

This isolated package implements points 1--8 of the SIREN-256 comparison
protocol.  It never retrains the INR or refits scan latents.  The three
transports share the same frozen no-skip SIREN decoder, QC-clean scan manifest,
subject splits, 256-D latent archives, registered mesh cache, pair tables,
losses, validation rule, and test evaluator.

The package deliberately references the pre-existing data rather than copying
large SDFs, meshes, latents, or decoder checkpoints.  `prepare_siren256_v5_inputs.py`
only creates small manifests, pair/sequence tables, input hashes, train-only
basis files, and train-only loss scales inside this directory.

## Setup

From the repository root, prepare the immutable input contract once:

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python \
  examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren256_v5_flow_vs_ode_brainode_qc_v1/scripts/prepare_siren256_v5_inputs.py \
  --config examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren256_v5_flow_vs_ode_brainode_qc_v1/configs/common_qc_siren256.json
```

This step refuses to overwrite a prepared contract unless `--force` is given.

## Training

Each run reads only train/validation pair tables and latent archives.  It
reconstructs registered train/validation meshes directly from their OBJ paths;
it does not open the all-split mesh cache, test pair table, or test latents.

```bash
ROOT=examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren256_v5_flow_vs_ode_brainode_qc_v1
PY=/home/jakaria/anaconda3/envs/inr_sdf/bin/python

$PY $ROOT/scripts/train_siren256_transport.py --config $ROOT/configs/v5_flow_c3.json --device cuda:0
$PY $ROOT/scripts/train_siren256_transport.py --config $ROOT/configs/plain_ode_c3_matched.json --device cuda:0
$PY $ROOT/scripts/train_siren256_transport.py --config $ROOT/configs/brainode_attention_c3_matched.json --device cuda:0
```

Use `--smoke-steps 2` for a short differentiability/data-contract check.  Runs
are created under `runs/<run_name>` and cannot be overwritten accidentally.

## Evaluation and comparison

```bash
$PY $ROOT/scripts/evaluate_siren256_transport.py --run $ROOT/runs/v5_flow_c3 --split test --device cuda:0
$PY $ROOT/scripts/evaluate_siren256_transport.py --run $ROOT/runs/plain_ode_c3_matched --split test --device cuda:0
$PY $ROOT/scripts/evaluate_siren256_transport.py --run $ROOT/runs/brainode_attention_c3_matched --split test --device cuda:0
$PY $ROOT/scripts/compare_siren256_transports.py --root $ROOT
$PY $ROOT/scripts/audit_brainode_batch_context.py --run $ROOT/runs/brainode_attention_c3_matched --device cuda:0
```

The evaluator produces subject-macro summaries, CN/AD, gap-bin, and
first-to-last summaries. It reports registered-normal error, decoded-surface
Chamfer L2 squared, ASSD, HD95, volume/rate/slope error, semigroup/inverse
defects, direct-vs-rollout error, no-change improvement, observed CN/AD
rate-order agreement, future trends, and change-hotspot overlap. Surface
metrics use the frozen SIREN zero-level surface extracted by marching cubes;
the normal-displacement proxy remains only a training diagnostic.

## BrainODE attention contract

BrainODE follows the paper-core one-case/one-trajectory semantics: each Q/K/V
attention operation has exactly one state token.  It therefore cannot make a
prediction depend on unrelated scans that share a dataloader batch.  The
attention is mathematically a singleton weight of one; this architectural
limitation is recorded by `audit_brainode_batch_context.py` and in every
checkpoint.

`v5_flow_c4_optional.json` is intentionally only a continuation configuration
for the direct flow after C3.  It is not trained or evaluated as a primary
result by this package.

## PCA-parity full-dimensional follow-up

`pca_parity_full_flow_v1.json` is the isolated corrective follow-up motivated
by the completed C3/C4 analysis.  It leaves every completed run unchanged and
copies the successful PCA flow's essential transport rather than the rank-32
v5 restriction:

```text
u = (z - train_mean) / train_std
u_target = u + (target_time - source_time) * G(u, source_time, target_time,
                                               delta_time, diagnosis)
z_target = train_mean + train_std * u_target
```

`G` is one zero-initialized `[128,128]` SiLU MLP with a full 256-D output.
Statistics come only from the frozen training latent archive.  The primary run
uses forward pairs, the original PCA-style unbalanced shuffle, and gradient
accumulation to reproduce an effective pair batch of 128 without placing 128
SIREN decodes in memory at once.

For SIREN-specific geometry, volume is trained from differentiable occupancy
of the frozen decoder rather than the registered-normal volume approximation.
Normal, volume, rate, and slope residuals are normalized before applying the
Huber loss.  Epoch zero is saved and evaluated as the exact no-change fallback;
a trained checkpoint replaces it only when it passes semigroup/inverse gates,
stays within the registered-geometry tolerance, improves volume over no-change,
and improves the combined validation score.

Run from the repository root:

```bash
ROOT=examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren256_v5_flow_vs_ode_brainode_qc_v1
PY=/home/jakaria/anaconda3/envs/inr_sdf/bin/python

$PY $ROOT/scripts/train_siren256_transport.py \
  --config $ROOT/configs/pca_parity_full_flow_v1.json --device cuda:0

$PY $ROOT/scripts/evaluate_siren256_transport.py \
  --run $ROOT/runs/pca_parity_full_flow_v1 --split test --device cuda:0
```

The test evaluator remains the same frozen-SIREN marching-cubes evaluator used
by C3, C4, the plain ODE, and BrainODE.
