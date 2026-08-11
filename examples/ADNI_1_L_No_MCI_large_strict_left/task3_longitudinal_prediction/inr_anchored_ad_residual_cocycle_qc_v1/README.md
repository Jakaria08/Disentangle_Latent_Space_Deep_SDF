# Large SIREN anchored AD residual cocycle (QC v1)

This isolated experiment improves the **existing QC-clean large SIREN direct
cocycle flow** without retraining or changing its decoder or checkpoint.  It
uses the existing direct flow as a frozen CN anchor.  For a source latent
`z`, normalized ages `s,t`, and disease label `c in {0,1}`, the new transport
is

```text
Phi(z,s,t,c) = Phi_CN_base(z,s,t)
             + c [ a(z,s,t) (Phi_AD_base - Phi_CN_base)
                   + (t-s) R(z,s,t) ],
```

where `a` is a bounded, initialized-at-one disease speed and `R` is a small
coefficient vector in a train-only PCA basis of AD residual velocities.
Therefore, at initialization, both CN and AD predictions exactly equal the
frozen base flow.  CN remains exactly anchored throughout training; only the
AD excess is calibrated.

## Why these files are separate

The source SIREN decoder, QC metadata, latent archives, and direct-flow
checkpoint are inputs only.  This directory owns every cache, checkpoint,
metric, and log produced by the new experiment.  It is safe to prepare or run
while another experiment is using the source directory.

## Pipeline

1. `build_large_siren_cache.py` validates all inputs, reads the already
   registered meshes once, and writes scan/pair caches.  It fits the feature
   PCA and AD residual basis **from train scans/pairs only**.
2. `validate_large_siren_anchored_cocycle.py` checks cache integrity and the
   initialization invariant (`new CN == base CN`, `new AD == base AD`).
3. `train_large_siren_anchored_cocycle.py` optimizes only the small AD
   calibrator.  Decoder and base flow are frozen.
4. `evaluate_large_siren_anchored_cocycle.py` writes per-pair metrics and
   aggregate summaries for the base and calibrated model.

## Key output locations

- `metadata/`: input contract, scan manifest, cached registered meshes, and
  pair tables.
- `basis/`: train-only feature PCA and AD residual basis.
- `runs/<run-name>/checkpoints/`: `latest.pth`, `best_primary.pth`,
  `best_feasible_surface.pth`, `best_feasible_ad_volume.pth`.
- `runs/<run-name>/logs/`: epoch CSV and JSON summary.
- `runs/<run-name>/evaluation/<checkpoint-name>/`: pair CSV plus summary CSV
  and JSON.  These artifacts can be plotted later without recomputation.

`best_primary.pth` is selected by **AD-only validation volume-rate error**.
Training uses an AD-weighted pair sampler, while CN remains an exact frozen
anchor and therefore cannot be degraded by the calibrator.

## Validation commands

From the repository root:

```bash
V1=examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/inr_anchored_ad_residual_cocycle_qc_v1
PY=/home/jakaria/anaconda3/envs/inr_sdf/bin/python

$PY $V1/scripts/build_large_siren_cache.py --config $V1/configs/large_siren_primary.json
$PY $V1/scripts/validate_large_siren_anchored_cocycle.py --config $V1/configs/large_siren_primary.json --device cuda:0
```

The cache builder refuses to overwrite existing cache artifacts.  Use
`--force` only when deliberately regenerating this experiment's own cache.

## Training and evaluation commands

```bash
$PY $V1/scripts/train_large_siren_anchored_cocycle.py \
  --config $V1/configs/large_siren_primary.json --run-name primary --device cuda:0

$PY $V1/scripts/evaluate_large_siren_anchored_cocycle.py \
  --config $V1/configs/large_siren_primary.json \
  --checkpoint $V1/runs/primary/checkpoints/best_primary.pth --device cuda:0
```

For a harmless smoke run after the cache is present, add
`--smoke-batches 2 --epochs 1 --run-name smoke` to training and
`--max-pairs 4` to evaluation.  Smoke output is written below
`runs/smoke/`; it does not affect the primary run.
