# LHipp_LatentDynamics_CrossCohort_v1

Left-hippocampus longitudinal shape dynamics compared BrainODE-style across ADNI, AIBL, OASIS-3
and CALSNIC. The comparison crosses four frozen 128-D representations (PCA, SpiralNet, Adaptive,
LAMM) with five dynamics models: latest cocycle direct_c4, plain ODE, BrainODE, faithful Latent
ODE, and a residual Latent ODE that predicts change from the last observation.
The full design is in [PLAN.md](PLAN.md). It is implemented in five stages; each stage is
implemented, validated, then run.

## Where things live

| what | location |
|---|---|
| code, configs, tests, docs | `examples/LHipp_LatentDynamics_CrossCohort_v1/` (this folder) |
| everything generated | `/mnt/bulk10tb/Deep3DComp/LHipp_LatentDynamics_CrossCohort_v1/` |

Every script writes through `benchmark_common.require_bulk`, which refuses any path outside the
bulk root. GPU work is restricted to `cuda:0` and `cuda:2`; GPU 1 is reserved.

## Stage status

| stage | content | status |
|---|---|---|
| 1 | data foundation: inclusive cohorts, frozen codes, protocol views, tasks, baselines, audit | implemented; see RUNBOOK |
| 2 | model zoo (5 dynamics), unified trainer/evaluator, anchor reproduction, Latent ODE searches | complete; all gates pass |
| 3 | ADNI matrix (20 cells x 3 seeds) | complete 2026-09-13; all gates pass; report `stage3_adni/reports/stage3_adni_report.md` |
| 4 | cross-cohort protocols | complete 2026-09-13: 480 external evaluations, 266 trainings, 140 transfer evaluations; gates G4.1–G4.4 pass; 14 OASIS-only cocycle runs not estimable (left out) |
| 5 | ablations A1–A4 + sensitivity arms (step 1), converter line (step 2), consolidated BrainODE-style report (step 3) | complete 2026-09-14: 227 jobs; report `stage5_brainode_style/reports/brainode_style_report.md` (byte-identical rebuild); gates G5.1–G5.6 pass (G5.6 only trivially), G5.7 fails for fine-tuned C1 (C1-frozen passes) |

## Results notebook

`notebooks/LHipp_LatentDynamics_Results.ipynb` presents every stage 1–5 result as a paper-style report: study
design, key findings and all quality gates, 17 tables, 27 figures including hippocampus mesh renders (cohort examples,
observed change, reconstruction error, condition-effect deformation, one-shot error maps, a converter example). It reads
only stored results through `notebooks/lhipp_results.py`; the mesh figures run CPU forward passes of already-evaluated
checkpoints. Rebuild and execute it with
`/home/jakaria/anaconda3/envs/pytorch_geo/bin/python notebooks/build_results_notebook.py --execute` (about one minute).

## Stage 1 layout

```
configs/
  cohort_sources.json          strict manifests, inclusive QC roots, label axes per cohort
  representation_registry.json frozen R1 representations with checkpoint hashes
  protocol_views.json          P0-P4c views + 5-fold cross-fit definition
  evaluation_tasks.json        one_shot_first, one_shot_prev, four_shot, all_prior_k
scripts/
  benchmark_common.py              paths, config, torch-free data logic
  stage1_build_inclusive_cohorts.py  converter-keeping AIBL/OASIS cohorts
  stage1_encode_cohort_latents.py    R1 codes for every cohort scan
  stage1_build_protocol_views.py     August-format archives, pairs, task tables per view
  stage1_compute_baselines.py        no-change, linear, population-drift baselines
  stage1_audit_data_foundation.py    gates G1.1-G1.8
  run_stage1_detached.sh             all of the above as one detached, resumable job
tests/
  test_stage1_data_foundation.py
```

Bulk outputs under `stage1_data_foundation/`: `cohorts/`, `vertices/`, `latents/`, `folds/`,
`views/<view>/{dataset,pairs,representations,tasks}`, `baselines/`, `reports/`, `logs/`.

## Stage 2 layout

```
configs/recipes/{direct_c4,plain_ode,brainode,latent_ode,latent_ode_residual}.json   one recipe per method, all representations
configs/recipes/<method>_selected_overrides.json                                   written by the Latent ODE searches
scripts/
  dynamics_core.py              view registry, real meshes per view split, leakage guard, recipes, run dirs
  rubanova_latent_ode.py        Latent ODE model (ODE-RNN encoder, latent ODE, decoder, ELBO)
  train_cocycle.py              direct C4 cocycle (August train_c4 semantics) on any view
  train_ode_transport.py        plain ODE / BrainODE (task3 trajectory trainer) on any view
  train_rubanova_latent_ode.py  Latent ODE trainer
  evaluate_dynamics.py          August pair metrics + BrainODE-style n-shot tasks, bootstrap CIs
  stage2_verify_anchors.py      G2.7a (stored anchors) and G2.7b (--retrain)
  stage2_build_jobs.py          job files: smoke, anchor_retrain, latent_ode_search
  stage2_select_latent_ode.py   picks the Latent ODE search winner (validation only)
  orchestrate_dynamics.py       detached multi-GPU queue (GPUs 0 and 2), resumable
tests/test_stage2_dynamics.py
```

Training runs go to `/mnt/bulk10tb/.../runs/<view>/<representation>/<method>/<run>/`; stage 2
validation outputs go to `.../stage2_validation/`.

## Decisions that matter downstream

- **One frozen representation space.** AIBL, OASIS and CALSNIC are encoded with ADNI-trained
  weights, the ADNI PCA basis and ADNI-train mesh normalization (registry R1). A difference
  between dynamics models is therefore never a representation difference.
- **ADNI codes are copied, not recomputed.** They come from the archives the existing
  August_Version and task3 LAMM results were trained on. Re-encoding them is the correctness
  check for the external cohorts (gate G1.2).
- **Pinned LAMM builder.** The working-tree `task_lamm_ae_v1/scripts/train_lamm.py` gained
  training-only options after the LAMM archives were built. Old checkpoints lack those
  arguments, so the committed blob `17015113…` (sha256 `a3095083…`) is executed in memory. The
  repository file is not modified.
- **Views are August-format.** Every view is a train/val/test triple in the archive schema the
  August and task3 trainers read. Standardization and age scaling are fitted on the view's
  train rows only.
- **Canonical labels.** Label 1 is the cohort's disease class (AD, or ALS for CALSNIC). It is
  stored as `CN`/`AD`, with source names in `*_diagnoses_source`.
- **Qualified ids.** Ids take the form `cohort:id`, because bare ADNI and AIBL RIDs collide.
- **Inclusive cohorts keep strict splits.** A strict subject is never moved to a different
  split, so the stage 5 converter models cannot leak strict test subjects.
