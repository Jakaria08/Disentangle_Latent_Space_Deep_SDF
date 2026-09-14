# Runbook

Environments: `inr_sdf` for phases 1–3 and 6 (CPU), `pytorch_geo` for phase 4/5 (GPU).
All commands are run from this directory.

```bash
PY=/home/jakaria/anaconda3/envs/inr_sdf/bin/python
```

## 0. Tests first

```bash
$PY tests/test_xcohort_common.py
```

Includes an end-to-end check that the stored ADNI PCA basis reproduces ADNI's published
PCA-128 validation error (0.033668). If that fails, stop — something upstream has moved.

## 1. Build cohorts (CPU, ~20–30 min per cohort)

Requires the cohort's mesh run to be finished (`<mesh_root>/manifests/selected_scans.csv`).

```bash
$PY scripts/phase1_build_cohorts.py --cohorts aibl calsnic oasis
$PY scripts/phase1_build_cohorts.py --cohorts oasis --dry-run   # inspect commands only
```

Runs the corrected QC (component tolerance 1.0 %, interval-stratified pair rule k=8,
culprit attribution) and then the cohort/split builder, per cohort, with the filters from
`configs/cohorts.json`. Writes `cohorts/<name>/qc/` and `cohorts/<name>/cohort/`, and
validates the resulting keep manifest against the shared topology.

Re-running requires clearing `cohorts/<name>/` first — the cohort builder refuses to
overwrite an existing output tree on purpose.

## 2. Shared-space gate (CPU, ~15 min)

```bash
$PY scripts/phase2_shared_space_audit.py --split test --k 128
```

Checks face connectivity against the reference, reconstructs each cohort with the ADNI
basis, and reports the gap against that cohort's own PCA plus a latent-shift measure.
**Exit code 1 means the shared space does not hold — do not continue.**

## 3. PCA protocol matrix (CPU, <1 h)

```bash
$PY scripts/phase3_pca_protocols.py                      # all four protocols
$PY scripts/phase3_pca_protocols.py --protocols internal external --components 128
```

This alone answers the whole experimental question. Run it before any GPU time.

## 4/5. Learned models (GPU)

```bash
PGEO=/home/jakaria/anaconda3/envs/pytorch_geo/bin/python

# internal: retrain per cohort with ADNI's frozen hyperparameters
$PGEO scripts/phase4_learned_protocols.py --protocol internal --cohorts aibl --gpu 0

# external: ADNI checkpoint, ADNI normalisation, target vertices
$PGEO scripts/phase4_learned_protocols.py --protocol external --cohorts aibl oasis calsnic \
      --models spiralnet_z128 --reference-checkpoint <path-to-adni-checkpoint> --gpu 0

# pooled and leave-one-cohort-out
$PGEO scripts/phase4_learned_protocols.py --protocol pooled --gpu 1
$PGEO scripts/phase4_learned_protocols.py --protocol loco --held-out aibl --gpu 2
```

Use `--dry-run` first for any protocol: it prints the manifests and commands without
training. One GPU per cohort; three fit comfortably.

LAMM's external evaluation is not wired: it needs an encode/decode entry point its CLI does
not expose. Its `internal`, `pooled` and `loco` paths work. The spiral and adaptive models
support all four.

## 6. Report

```bash
$PY scripts/phase6_report.py --split test --k 128
```

Produces `reports/phase6_summary_test_k128.md` with internal / external / gap / pooled /
loco per model and cohort, in both metric conventions.

## Order of operations

Phase 1 → 2 (gate) → 3 (PCA, cheap, complete answer) → 4/5 (learned models) → 6.

Phase 3's result decides how much the GPU phases are worth: if the ADNI PCA basis transfers
well, phases 4/5 are a robustness check; if it does not, they are the investigation.
