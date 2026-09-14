# Runbook

All commands run from this folder: `examples/LHipp_LatentDynamics_CrossCohort_v1/`.

## Environments

| use | python |
|---|---|
| GPU steps (encoding, decoding) | `/home/jakaria/anaconda3/envs/pytorch_geo/bin/python` |
| tests for stages 1, 3, 4 (pytest available, no torch needed) | `/home/jakaria/anaconda3/envs/inr_sdf/bin/python -m pytest -q tests/test_stage1_data_foundation.py tests/test_stage3_adni.py tests/test_stage4_crosscohort.py` |
| stage 2 tests (need torch + torch_scatter; this env has no pytest, so call each `test_*` function directly) | `/home/jakaria/anaconda3/envs/pytorch_geo/bin/python` |

## Stage 1 — data foundation

### Run everything as one detached job (survives SSH disconnects)

```bash
setsid nohup bash scripts/run_stage1_detached.sh > /dev/null 2>&1 < /dev/null &
```

Progress:

```bash
cat /mnt/bulk10tb/Deep3DComp/LHipp_LatentDynamics_CrossCohort_v1/stage1_data_foundation/logs/stage1_status.txt
```

Each step logs to `.../logs/stage1_<step>.log`, and the job stops at the first failure.

- **Resuming:** finished outputs are skipped, so rerunning after a failure continues from where it stopped.
- **Rebuilding:** `STAGE1_OVERWRITE=--overwrite bash scripts/run_stage1_detached.sh`.
- **GPU:** `STAGE1_DEVICE=cuda:0` selects GPU 0 instead of the default `cuda:2`. `cuda:1` is refused.

### Steps

| step | command (in `scripts/`) | device | time |
|---|---|---|---|
| tests | `inr_sdf python ../tests/test_stage1_data_foundation.py` | CPU | ~30 s |
| inclusive cohorts | `python stage1_build_inclusive_cohorts.py` | CPU | ~1 min |
| encode latents | `python stage1_encode_cohort_latents.py --device cuda:2` | GPU | ~10–20 min |
| protocol views | `python stage1_build_protocol_views.py` | CPU | a few min |
| baselines | `python stage1_compute_baselines.py --device cuda:2 --splits val test` | GPU | ~10–20 min |
| audit | `python stage1_audit_data_foundation.py` | CPU | ~2 min |

Encoder smoke check without writing anything:

```bash
python stage1_encode_cohort_latents.py --cohorts adni calsnic --dry-run --device cuda:2
```

### Gates (`reports/stage1_audit.md`)

| gate | check |
|---|---|
| G1.1 | strict and inclusive manifests, PLY faces and PCA faces all match the ADNI topology |
| G1.2 | ADNI codes are exact copies of the anchor archives; re-encoding reproduces them within 1e-3 of each code dimension's std (LAMM codes are far larger in scale than SpiralNet's, so an absolute bound is not comparable) |
| G1.3 | ADNI val reconstruction within 0.5% of published values; PCA external test RMSE matches CrossCohort phase 3 (±1e-4) |
| G1.4 | no subject or scan overlap across view splits; LOCO held-out cohort absent from train/val; cross-fit tests every subject exactly once; standardization fitted on train; inclusive manifests keep strict splits |
| G1.5 | evaluable task counts reproduce PLAN §0.3 |
| G1.6 | ADNI test no-change metrics equal the August/task3 anchor summaries (±1e-5) |
| G1.7 | inclusive converter cohorts validated, with AD converters present |
| G1.8 | unit and regression tests pass |

## Stage 2 — models, trainers, evaluator

Commands run from `scripts/`. `PY=/home/jakaria/anaconda3/envs/pytorch_geo/bin/python`.
`BULK=/mnt/bulk10tb/Deep3DComp/LHipp_LatentDynamics_CrossCohort_v1`.

```bash
# tests (CPU)
CUDA_VISIBLE_DEVICES="" $PY ../tests/test_stage2_dynamics.py

# dry runs: gradients + validation, nothing written
$PY train_cocycle.py --view p0_internal_adni --representation pca128 --device cuda:2 --dry-run
$PY train_ode_transport.py --view p0_internal_adni --representation spiralnet128 --method brainode --device cuda:2 --dry-run
$PY train_rubanova_latent_ode.py --view p0_internal_adni --representation pca128 --device cuda:2 --dry-run

# G2.7a: evaluator reproduces the 10 stored anchor summaries
$PY stage2_verify_anchors.py --device cuda:2

# job suites, each run detached on GPUs 0 and 2
$PY stage2_build_jobs.py smoke anchor_retrain latent_ode_search latent_ode_residual_search
J=$BULK/stage2_validation/jobs/smoke.json
setsid nohup $PY orchestrate_dynamics.py --jobs $J --gpus 0,2 > ${J%.json}.orchestrator.log 2>&1 < /dev/null &
cat ${J%.json}.state.json   # progress; per-job logs in $BULK/stage2_validation/jobs/logs/<suite>/
```

Rerunning the orchestrator on the same job file resumes: jobs already complete are skipped.

### Gates

| gate | check |
|---|---|
| G2.1–G2.6 | `tests/test_stage2_dynamics.py`: RK4 vs matrix exponential, cocycle identity and disease-head structure, ODE semigroup/inverse, Latent ODE loss/KL/masking/order/conditioning/overfit, recipes, leakage guard, determinism |
| G2.7a | `reports/anchor_verification.md`: every stored anchor pair metric reproduced within 1e-5 |
| G2.7b | `reports/anchor_retrain_comparison.json`: PCA cocycle and plain-ODE seed-42 retrains within ±0.002 mm first-to-last MAE and ±0.05 AD capture of the anchors |
| G2.8 | BrainODE vectorized field equals August's per-subject loop (outputs and gradients) |
| G2.9 | `jobs/smoke.state.json`: all 20 cells (4 representations x 5 dynamics) train one epoch and evaluate on validation |
| G2.10 | `reports/{latent_ode,latent_ode_residual}_search.md` + `configs/recipes/<method>_selected_overrides.json` (16 paired trials each, validation only) |

## Stage 3 — ADNI internal matrix (P0)

60 seed-runs = 4 representations x 5 dynamics x seeds 42/43/44. 54 are trained; 6 are reused August seed-42
plain ODE / BrainODE anchors. Seed-42 cocycles are retrained so every cocycle has `best_min_epoch.pt`.

```bash
S3=$BULK/stage3_adni
$PY stage3_build_jobs.py training            # results_index.json + jobs/adni_training.json (54 train + 144 evaluations)
setsid nohup $PY orchestrate_dynamics.py --jobs $S3/jobs/adni_training.json --gpus 0,2 > $S3/jobs/adni_training.orchestrator.log 2>&1 < /dev/null &

# after training finishes: condition sweeps for all 60 seed-runs, then the report
$PY stage3_build_jobs.py analysis
setsid nohup $PY orchestrate_dynamics.py --jobs $S3/jobs/adni_analysis.json --gpus 0,2 > $S3/jobs/adni_analysis.orchestrator.log 2>&1 < /dev/null &

# or launch the finisher at any time: it waits for training, refuses to continue past an unrepaired
# failure, reconciles repaired jobs, then runs the analysis suite (log: $S3/jobs/finisher.log)
setsid nohup bash run_stage3_analysis_after_training.sh > /dev/null 2>&1 < /dev/null &

# a draft from whatever has finished, at any time
$PY stage3_report_adni.py --allow-partial      # writes reports/stage3_adni_report_partial.md
```

Tests: `$PY ../tests/test_stage3_adni.py` (statistics helpers, index/job contract).

Outputs under `$S3/reports/`: `stage3_adni_report.md`, `stage3_gates.json`, `tables/p3a..p3g_*.csv`, `figures/*.png`.

| gate | check |
|---|---|
| G3.1 | all 60 seed-runs have val and test evaluations (trained runs: status complete) |
| G3.2 | no run loaded test data during training or selection |
| G3.3 | reused anchors reproduce their stored summaries (stage 2 G2.7a) |
| G3.4 | models not beating no-change on validation are flagged, never dropped |
| G3.5 | cocycle relative defects <= 0.25; plain ODE / BrainODE semigroup defect <= 1e-5 |
| G3.6 | one sealed test summary per seed-run, SHA-256 recorded |

## Stage 4 — cross-cohort protocols (P1–P4c)

```bash
S4=$BULK/stage4_crosscohort
$PY stage4_build_jobs.py external training     # results_index.json + jobs/crosscohort_{external,training}.json
setsid nohup $PY orchestrate_dynamics.py --jobs $S4/jobs/crosscohort_external.json --gpus 0,2 --capacity-per-gpu 2 > $S4/jobs/crosscohort_external.orchestrator.log 2>&1 < /dev/null &
setsid nohup $PY orchestrate_dynamics.py --jobs $S4/jobs/crosscohort_training.json --gpus 0,2 --capacity-per-gpu 3 > $S4/jobs/crosscohort_training.orchestrator.log 2>&1 < /dev/null &
```

Tests: `/home/jakaria/anaconda3/envs/inr_sdf/bin/python ../tests/test_stage4_crosscohort.py`.

| suite | content |
|---|---|
| external (480 jobs) | P1 zero-shot: 60 ADNI seed-runs evaluated on AIBL, OASIS, CALSNIC (test split and whole cohort); CALSNIC twice, with labels and with `--condition-override 0` |
| training (848 jobs) | 266 trainings (P2 internal x 3 cohorts, P2 cross-fit AIBL/OASIS for PCA + Adaptive, P3 pooled x 3 seeds, P4 LOCO) with val/test evaluations, plus P4b (AIBL -> ADNI+OASIS) and P4c (pooled -> CALSNIC) transfer evaluations |

- **Not estimable, left out (14 runs):** the cocycle on OASIS-only training views (`p2_internal_oasis`, `p2_crossfit_oasis_*`). Their training data contain no AD subject with three or more visits, so there are no non-adjacent AD pairs for the cocycle's balanced pair sampler.
- **Report** (after both suites finish): `$PY stage4_report_crosscohort.py` writes `$S4/reports/stage4_crosscohort_report.md`, `stage4_gates.json`, `tables/s4a..s4h_*.csv` and `figures/s4f_interval_shift.png`. `--allow-partial` writes `stage4_crosscohort_report_partial.md` while jobs are still running.
- **Dry-run caveat:** `--dry-run` on views whose validation split has no AD subject with 3+ visits fails inside `balanced_subset`. That strata check applies only when a smoke/dry-run record limit is set, and full training does not hit it (verified on AIBL internal, LOCO without ADNI, and the OASIS views). Gate launches on exit codes, never on grep output.

## Stage 5 — ablations, sensitivity arms, converter line, consolidated report

### Step 1: ablations A1–A4 and sensitivity arms

```bash
S5=$BULK/stage5_brainode_style
$PY stage5_pooled_pca.py                          # CPU: gate S5.1, pooled basis, latents, p3_pooled archives, reconstruction report
$PY stage5_build_jobs.py ablations sensitivity    # results_index.json + jobs/stage5_{ablations,sensitivity}.json
setsid nohup $PY orchestrate_dynamics.py --jobs $S5/jobs/stage5_ablations.json --gpus 0,2 --capacity-per-gpu 3 > $S5/jobs/stage5_ablations.orchestrator.log 2>&1 < /dev/null &
setsid nohup $PY orchestrate_dynamics.py --jobs $S5/jobs/stage5_sensitivity.json --gpus 0,2 --capacity-per-gpu 3 > $S5/jobs/stage5_sensitivity.orchestrator.log 2>&1 < /dev/null &
```

Tests (torch; no pytest in that environment, the file runs itself):
`/home/jakaria/anaconda3/envs/pytorch_geo/bin/python ../tests/test_stage5_ablations_sensitivity.py`.
Smoke before launching: `bash $S5/smoke/run_step1_smoke.sh` (1-epoch run of every new training path, then val evaluation and a val condition sweep; outputs under `$S5/smoke/`).

| suite | content |
|---|---|
| ablations (64 jobs, 16 trainings) | A1 exact / volume-v2 coboundary (PCA, SpiralNet, s42, no early stopping) via `train_coboundary_ablation.py`; A2 `train_cocycle.py --method brainode_v`; A3 `--method direct_c4_no_disease`; A4 `train_rubanova_latent_ode.py` with `model.condition_in_encoder/dynamics=false`; each with val/test evaluation and a test condition sweep |
| sensitivity | `pooled_pca_basis` (skipped when its report is complete), 5 dynamics x s42–s44 on `pca128_pooled` in `p3_pooled`, and test evaluations of `best_min_epoch.pt` for the 28 stage-4 cocycle runs that have one |

- **A1 runs** land in `runs/p0_internal_adni/<rep>/<method>/<rep>_<method>_complete_s42` through a registry shim under `$S5/ablations/august_registry_shim/`; the derived August configs are in `<method>/_configs/`.
- **Min-epoch-15:** 14 stage-4 cocycle runs have no `best_min_epoch.pt` because no epoch ≥ 15 was feasible; `results_index.json` lists them.
- **Age subset 65–95** has no jobs; the stage 5 report rescores stored `task_rows.csv`.

### Step 2: converter line

```bash
$PY stage5_build_converter_view.py            # CPU: views/p5_converter_pooled (P3 normalization, converter keys)
$PY stage5_build_converter_jobs.py            # converter_index.json + jobs/stage5_converter.json
setsid nohup $PY orchestrate_dynamics.py --jobs $S5/jobs/stage5_converter.json --gpus 0,2 --capacity-per-gpu 2 > $S5/jobs/stage5_converter.orchestrator.log 2>&1 < /dev/null &
```

Tests: `/home/jakaria/anaconda3/envs/pytorch_geo/bin/python ../tests/test_stage5_converter.py` (11 tests: dose additivity ≤ 1e-12, C1 = direct_c4 on stable subjects, onset rules, objective = task3 with fixed labels, evaluation adapter, onset posterior, converter view, BrainODE-full pseudo pairs and loss, pooled feedback = task3 feedback, job dependencies).

| job group | scripts | outputs |
|---|---|---|
| gate G5.6 | `stage5_synthetic_onset.py --representation {pca128,adaptive128} --seed 42` | `$S5/converter/synthetic_onset/<rep>_s42.{json,csv}`; test evaluations wait for it |
| C1 variants | `train_converter_cocycle.py --variant {c1,c1_frozen,c1_mci,c1_w050}` then `evaluate_converter.py` (prefix-only and oracle onsets) | `runs/p5_converter_pooled/<rep>/<variant>/...`, `learned_onsets.csv` |
| comparators | `evaluate_dynamics.py --view p5_converter_pooled --trained-view p3_pooled --skip-pairs` on P3 direct_c4 (C0) and brainode | `$S5/converter/comparators/<rep>__<method>__s<seed>/<split>` |
| BrainODE-full | `train_brainode_full.py --seed` then `evaluate_brainode_full.py` (voxel feedback, spawn process pool) | `runs/p5_converter_pooled/pca128/brainode_full/...` |

- Validation selection of C1 uses val converters' oracle-window onsets (validation labels only); test uses prefix-only (primary) and oracle (upper bound) rules and never a learned onset (gate G5.5 hashes the onset parameters).
- G5.6 must be read against its window-midpoint reference: at the observed noise, recovery equals the midpoint, so learned onsets are prior-dominated.

### Step 3: consolidated BrainODE-style report

```bash
$PY report_brainode_style.py --allow-partial            # draft while step 1/2 jobs run (gates over missing runs read PENDING)
$PY stage5_error_maps.py                                # CPU: figures/f6_error_maps.png + error_maps.npz (needs model inference)
$PY report_brainode_style.py --verify                   # final: all results present; rebuilds twice and writes G5.1 to verification.json
```

Outputs under `$S5/reports/`: `brainode_style_report.md`, `gates.json` (G5.2-G5.7), `tables/` (R-T1..R-T8, endpoints E1-E4, fidelity, consistency and cost, three sensitivity tables, published-number context, traceability), `figures/f1..f5_*.png`. Tests: `/home/jakaria/anaconda3/envs/pytorch_geo/bin/python ../tests/test_stage5_report.py`.

- **E2 improvement** = AD capture closer to 100% (|1 - capture| lower); **E1 non-inferiority** = the upper 95% limit of the paired error difference to the cocycle is at most 2% of the no-change error. A method is preferred over the cocycle only when both hold; otherwise methods rank by E1.

## Results notebook

```bash
cd notebooks
/home/jakaria/anaconda3/envs/pytorch_geo/bin/python build_results_notebook.py --execute   # writes and runs LHipp_LatentDynamics_Results.ipynb (~1 min, CPU)
```

- The cells are defined in `build_results_notebook.py`; data loading, the report palette, tables and mesh rendering live in `lhipp_results.py`. Edit those files and rebuild rather than editing the `.ipynb` by hand.
- Needs the finished stage 1–5 outputs (tables, summaries, checkpoints, `verification.json`); it evaluates no test set anew.
