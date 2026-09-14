# Cross-cohort left hippocampus reconstruction

PCA, SpiralNet++, Adaptive-Spiral and LAMM on the left hippocampus across four cohorts —
ADNI (reference), AIBL, OASIS-3 and CALSNIC — under three evaluation protocols.

This is possible because every cohort's meshes were registered to the **ADNI template**, so
they share vertex order and face connectivity. The topology hash is identical everywhere
(`a8485554b4637ef9…`), which is what lets an ADNI-fitted model be applied to another cohort
without retraining. Every script here refuses to run on a manifest that violates that.

## The three protocols, and why the distinction matters

The protocols differ in exactly one respect: **where the fitted parameters come from.**

| protocol | fitted on | evaluated on | answers |
|---|---|---|---|
| `internal` | the cohort's own train split | its own val/test | each cohort's own ceiling |
| `external` | ADNI only, applied unchanged | each target cohort | does the ADNI shape space generalize? |
| `pooled` | ADNI + AIBL + OASIS train splits | each cohort's test | best achievable with all data |
| `loco` | every eligible cohort except one | the held-out cohort | generalization to an unseen cohort |

`external` is the one that is easy to get wrong. `spiral_common` derives the template, the
decimation hierarchy and the per-vertex normalisation from whichever manifest it is pointed
at. Re-deriving any of those on the target cohort would feed the network a different input
distribution than its weights were trained on, quietly turning an external validation into
a partial refit. The external path therefore keeps the reference environment in place and
streams target vertices through it.

Names follow the literature: `external` is *external validation* / zero-shot cross-cohort
transfer; `internal` is per-cohort *internal validation*; `loco` is leave-one-cohort-out.
BrainODE (NeurIPS 2025) used the pooled setting for its headline results and leave-one-out
for its cross-benchmark table.

## Metric

`vertex_rmse_mm` — per-scan RMSE over **coordinates**, imported from the ADNI spiral task
rather than reimplemented, so numbers are directly comparable with existing ADNI results.
Per-vertex **Euclidean** distance is √3 larger and is reported alongside, because that is
the convention BrainODE uses. Never compare one against the other.

## Reference numbers (ADNI, validation)

| model | val `vertex_rmse_mm` |
|---|---|
| **PCA-128** | **0.033668** |
| SpiralNet++-128 | 0.036784 |
| Adaptive-Spiral-128 | 0.037237 |
| LAMM-128 | 0.037685 |
| MeshMAE (tuned) | 0.037867 |

On ADNI, PCA-128 beats every learned model at matched latent size. Treat that as the
baseline to beat on each new cohort — not as a foregone conclusion in either direction.

## Layout

```
configs/      cohorts.json (registry, filters, labels), hyperparameters_z128.json (frozen ADNI configs)
scripts/      xcohort_common.py + phase1..phase6
cohorts/      per-cohort QC and cohort manifests (written by phase 1)
reports/      protocol results (CSV) and the phase 6 summary
tests/        python tests/test_xcohort_common.py
logs/         phase run logs
```

Heavy artifacts (vertex caches, checkpoints) live under
`/mnt/bulk10tb/Deep3DComp/CrossCohort_LHipp_v1/`, never in the repo.

## Cohort notes

- **AIBL / OASIS** use the ADNI `strict_no_mci` filter: baseline CN/AD, no MCI-labelled
  visit, no direct CN↔AD change, at least two visits.
- **CALSNIC is ALS, not Alzheimer's.** It uses `--cohort-filter all`, is restricted to
  Control vs ALS, and is **excluded from pooled AD training**. Use it as a held-out cohort
  and as an out-of-distribution reconstruction test. Its median visit gap is 0.36 y, below
  the measurement noise floor, so it supports no longitudinal claim.

## Hyperparameters

Frozen at ADNI's Optuna-searched values and reused verbatim; only weights are retrained.
Re-searching per cohort would cost ~20 h per study and would make cohorts incomparable.

See `RUNBOOK.md` for the commands.
