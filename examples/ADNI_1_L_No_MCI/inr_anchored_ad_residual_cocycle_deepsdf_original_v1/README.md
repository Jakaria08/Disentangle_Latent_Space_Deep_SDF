# Original-cohort DeepSDF anchored AD residual cocycle (v1)

This is the third INR experiment: the smaller original no-MCI cohort using
frozen DeepSDF latents and decoder. It is entirely separate from the original
DeepSDF direct-flow experiment and the two SIREN calibrations.

It uses the shared anchored calibration

```text
Phi(z,s,t,c) = Phi_CN_base(z,s,t)
             + c [ a(z,s,t) (Phi_AD_base - Phi_CN_base)
                   + (t-s) R(z,s,t) ],
```

where `a` is bounded and initialized at one and `R` uses a train-only AD
residual basis (one mean direction plus 12 PCA directions). CN stays exactly
equal to the frozen base prediction. The original metadata supplies no mm3
volume column; percentage and rate trends are computed in registered mesh
coordinates and are scale-invariant.

## Terminal progress and outputs

Training prints batch 1 and every 10 batches with total loss, target SDF,
AD-volume-rate loss, and mean speed, then prints an AD-only validation line
per epoch. Change `ProgressEveryBatches` in the config to adjust this.

All generated artifacts belong to this directory: `metadata/`, `basis/`, and
`runs/<run-name>/`. Evaluation writes reusable pair CSVs and summaries below
`runs/<run-name>/evaluation/`.

## Commands

```bash
V3=examples/ADNI_1_L_No_MCI/inr_anchored_ad_residual_cocycle_deepsdf_original_v1
PY=/home/jakaria/anaconda3/envs/inr_sdf/bin/python

$PY $V3/scripts/build_original_deepsdf_cache.py --config $V3/configs/original_deepsdf_primary.json
$PY $V3/scripts/validate_original_deepsdf_anchored_cocycle.py --config $V3/configs/original_deepsdf_primary.json --device cuda:0

$PY $V3/scripts/train_original_deepsdf_anchored_cocycle.py \
  --config $V3/configs/original_deepsdf_primary.json --run-name primary_full --device cuda:0
```
