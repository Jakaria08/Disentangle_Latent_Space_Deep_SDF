# Original-cohort SIREN anchored AD residual cocycle (v1)

This is the second INR experiment: the smaller original no-MCI ADNI cohort
using the enhanced frozen SIREN decoder and its strongest existing direct
sequence-cocycle checkpoint.  It is isolated from both the original source
experiment and the large QC-clean SIREN experiment.

The exact same anchored calibrator used by the large SIREN experiment is used
here:

```text
Phi(z,s,t,c) = Phi_CN_base(z,s,t)
             + c [ a(z,s,t) (Phi_AD_base - Phi_CN_base)
                   + (t-s) R(z,s,t) ].
```

`a` is a bounded speed initialized to one. `R` is an AD residual velocity in
a train-only basis (mean direction plus 12 PCA directions). CN is exactly the
frozen base prediction throughout training. The old metadata has no explicit
mm3 mesh-volume column, so volume losses and reports use registered mesh
coordinates; relative-volume trends and percentage/year rates are invariant
to the fixed mesh scale.

## Terminal progress

The training command prints a line every 10 training batches, including total
loss, target SDF loss, AD volume-rate loss, and mean AD speed. It also prints
an epoch validation line with AD-only SDF and AD volume-rate metrics. Change
`ProgressEveryBatches` in the config if you want a different frequency.

## Outputs

- `metadata/`: cache, manifest, pair tables, and validation report.
- `basis/`: PCA and residual basis fitted only from original-cohort train data.
- `runs/<run-name>/checkpoints/`: primary and feasible checkpoints.
- `runs/<run-name>/logs/epochs.csv`: every epoch’s terminal metrics.
- `runs/<run-name>/evaluation/<checkpoint>/`: reusable pair-level and summary metrics.

## Commands

```bash
V2=examples/ADNI_1_L_No_MCI/inr_anchored_ad_residual_cocycle_siren_original_v1
PY=/home/jakaria/anaconda3/envs/inr_sdf/bin/python

$PY $V2/scripts/build_original_siren_cache.py --config $V2/configs/original_siren_primary.json
$PY $V2/scripts/validate_original_siren_anchored_cocycle.py --config $V2/configs/original_siren_primary.json --device cuda:0

$PY $V2/scripts/train_original_siren_anchored_cocycle.py \
  --config $V2/configs/original_siren_primary.json --run-name primary_full --device cuda:0
```
