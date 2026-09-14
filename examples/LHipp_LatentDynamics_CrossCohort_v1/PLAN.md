# LHipp_LatentDynamics_CrossCohort_v1 — plan

Left hippocampus, latent 128. Four frozen representations × four longitudinal dynamics, evaluated
BrainODE-style on ADNI, AIBL, OASIS-3 and CALSNIC. Written 2026-09-12. Nothing here is implemented
yet: each part is implemented when asked, then run when asked.

## 0. What the latest results say, and how they shape this plan

### 0.1 ADNI longitudinal test, seed 42 (latest runs)

Cells: first→last (FL) coordinate MAE / all-pair MAE / FL per-vertex Euclidean (mm) / predicted AD
log-volume rate, with % of the observed −0.0528. No-change FL MAE ≈ 0.170.

| representation | cocycle (direct_c4) | plain ODE | BrainODE-core |
|---|---|---|---|
| PCA-128 | 0.1540 / 0.1501 / 0.3067 / −0.0492 (93%) | 0.1606 / 0.1522 / 0.3196 / −0.0158 (30%) | 0.1633 / 0.1539 / 0.3250 / −0.0146 (28%) |
| SpiralNet-128 | 0.1536 / 0.1499 / 0.3053 / −0.0483 (91%) | 0.1548 / 0.1489 / 0.3079 / −0.0207 (39%) | 0.1591 / 0.1517 / 0.3161 / −0.0189 (36%) |
| Adaptive-128 | 0.1543 / 0.1502 / 0.3068 / −0.0472 (89%) | 0.1555 / 0.1494 / 0.3093 / −0.0218 (41%) | 0.1576 / 0.1513 / 0.3133 / −0.0280 (53%) |
| LAMM-128 | 0.1541 / 0.1496 / 0.3069 / −0.0445 (84%) | not run | not run |

- No Latent ODE (Rubanova et al., NeurIPS 2019) exists for any representation.
  `task3_latent_flow_128_v1/scripts/train_latent_ode.py` only trains plain-ODE and BrainODE transport.
- Sources:
  - `/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/August_Version/training/<rep>/<method>/<run>/evaluation/test/summary.json`
  - LAMM: `.../task3_latent_flow_128_v3_lamm_latest/training/lamm128/direct_c4/lamm128_direct_c4_s42`

What follows from this:

1. **The cocycle is the method to beat, on disease fidelity more than on error.**
   - Shape error is only 1–6% below the ODEs, and the gap is smallest for SpiralNet.
   - The cocycle keeps 84–93% of the observed AD atrophy rate. The ODEs keep 28–53%.
   - Error alone would call this a near-tie, so every table reports disease fidelity beside Euclidean error.
2. **"Latest cocycle" means `direct_c4`.** Coboundary variants are an optional ablation, not the headline.
   - Trainers: the August v2 trainer, or the `task3_latent_flow_128_v2_lamm` code for LAMM.
   - The exact-coboundary and volume-coboundary-v2 runs ended after 12–43 of 160–180 epochs and were never test-evaluated.
   - Exact-coboundary validation scores are worse: PCA 1.151, SpiralNet 1.136, Adaptive 1.124. direct_c4 scores 1.094, 1.087, 1.087. Volume-v2 uses a different score.
3. **direct_c4 selected its checkpoint very early** (PCA epoch 5, SpiralNet 1, Adaptive 2, LAMM 1).
   - That is before its 10-epoch consistency ramp and 15-epoch anatomy ramp finish.
   - The published selection rule stays, so anchors remain valid. Seeds 43/44 are added, plus a min-epoch-15 selection as a sensitivity check.
4. **BrainODE-core takes ~205 min per run; plain ODE takes 8–10 min.**
   - The trainer settings are identical (100 epochs, batch 64, AdamW 5e-4, RK4 with 4 substeps); only the vector field differs.
   - At that cost BrainODE would be ~75% of all GPU time, so Part 2 profiles and vectorizes it under an equivalence test.
5. **Three different representation checkpoint sets are in use today.**
   - August: SpiralNet v6 trial 2 and Adaptive v2 trial 2.
   - LAMM cocycle and task10: LAMM E1_N3_ml_s1.
   - CrossCohort_LHipp_v1: SpiralNet v7, Adaptive v7 and LAMM E1_N1_ml_s1.
   - One frozen registry is required (§1.2).
6. **ADNI-trained encoders transfer better than local refits** (SpiralNet test coordinate RMSE, external vs internal):

   | cohort | external | internal |
   |---|---|---|
   | AIBL | 0.0904 | 0.139 |
   | OASIS | 0.0969 | 0.109 |
   | CALSNIC | 0.0921 | 0.107 |

   All dynamics therefore run in the frozen ADNI latent space, so a dynamics difference is never a representation difference.
7. **BrainODE's exact 4-shot needs ≥5 visits.**
   - Qualifying subjects: ADNI 182 (1 AD), OASIS 25 (0 AD), AIBL 0, CALSNIC 0.
   - AD subjects outside ADNI are scarce: AIBL 20, OASIS 6, and only 2 and 1 in their test splits.
   - Tasks and evaluation sets are built around these counts (§0.3, §1.4).
8. **Existing ADNI runs are valid regression anchors.** August and CrossCohort use the same 2583 ADNI scans with identical splits (verified).

### 0.2 BrainODE (NeurIPS 2025): the protocol we mirror

- **Data**
  - ≥2 visits per subject; NC, AD and NC→AD converters; MCI excluded.
  - Ages 65–95, with t = (age − 65)/30.
- **Training**
  - Every pair of a subject's time points, integrated forward and backward.
  - L2 loss on PCA coefficients; RK4; AdamW, lr 5e-4, 100 epochs; random-scaling augmentation.
- **Tasks:** predict each subject's latest shape.
  - *4-shot:* subjects with ≥5 observations. One prediction from each of the 4 earlier observations; the 4 predicted shapes are averaged.
  - *1-shot:* from a single observation.
- **Metric:** mean per-vertex Euclidean distance (mm), reported as mean ± SD over subjects.
- **Tables**
  - T1: regular intervals (LBC1936, AIBL).
  - T2: irregular intervals (AIBL + ADNI + OASIS unified).
  - T3: ablation. Shape accuracy NC&AD 0.606, CONV 0.216; τθ accuracy 0.891.
  - T5/T6: composition and evaluable subjects. Hippocampus 4-shot/1-shot: LBC 20/20, AIBL 18/19, irregular 65/79.
  - T7: per dataset, 4-shot / 1-shot.

    | dataset | 4-shot | 1-shot |
    |---|---|---|
    | AIBL | 0.52±0.07 | 0.46±0.05 |
    | ADNI | 0.59±0.14 | 0.54±0.15 |
    | OASIS | 0.52±0.07 | 0.48±0.08 |

  - T8: cross-benchmark. Exp1 AIBL → ADNI+OASIS 0.518; Exp2 AIBL+ADNI+OASIS → LBC1936 0.522.
  - Fig 3 / C.3: condition sweeps (volume under c = 0 vs c = 1).
- **Baselines**
  - Linear extrapolation, RNN, LSTM, RNN-Decay, ShapeFlow, BrLP.
  - Latent ODE scored 1.126/1.079 on AIBL-regular and 1.215/1.119 on irregular data, roughly 2× BrainODE's error. The paper blames its autoencoder and notes near-identical outputs across conditions.
- **Absolute mm values are not comparable.**
  - BrainODE differs in template, Point2Mesh-style fitting, PCA-150 and subjects.
  - Our FL Euclidean ≈ 0.31 mm is therefore not "better than 0.54".
  - Only rankings inside this benchmark, and ratios to no-change and linear baselines, can be claimed. BrainODE's numbers appear only in a separate context table.

Deliberate deviations:

- Main line uses BrainODE-core: fixed labels, no pseudo-cognitive sampling.
  - This matches the paper's ablation "Baseline" row, because strict cohorts contain no converters.
  - BrainODE-full appears in the converter line (Part 5).
- No LBC1936. CALSNIC controls stand in for the unseen-cohort Exp2 analog.
- No 65–95 age filter in the main analysis. A BrainODE-matched age subset is a sensitivity analysis.

### 0.3 Evaluable subjects in the strict cohorts

Subjects with ≥k visits; AD (ALS for CALSNIC) in parentheses.

| cohort | split | ≥2 | ≥3 | ≥4 | ≥5 | median interval | median first→last span |
|---|---|---|---|---|---|---|---|
| ADNI | all | 597 (271) | 539 (233) | 395 (122) | 182 (1) | 0.50 y | 2.00 y |
| ADNI | test | 61 (28) | 58 (25) | 42 (12) | 21 (1) | | |
| AIBL | all | 148 (20) | 114 (8) | 72 (1) | 0 | 1.50 y | 4.08 y |
| AIBL | test | 15 (2) | 11 (0) | 7 (0) | 0 | | |
| OASIS | all | 385 (6) | 184 (0) | 76 (0) | 25 (0) | 2.83 y | 4.50 y |
| OASIS | test | 39 (1) | 20 (0) | 10 (0) | 5 (0) | | |
| CALSNIC | all | 303 (144) | 200 (89) | 0 | 0 | 0.36 y | 0.68 y |
| CALSNIC | test | 30 (14) | 23 (12) | 0 | 0 | | |

Consequences:

- **Interval type.** AIBL (fixed 1.5 y) is the "regular" cohort, as in BrainODE T1. ADNI and OASIS are "irregular" (T2).
- **Task choice.**
  - Exact 4-shot is reported only where ≥10 subjects qualify: ADNI (CN), and OASIS across the whole cohort.
  - Elsewhere the task is all-prior-k, with k = min(n − 1, 4).
- **Where AD fidelity can be estimated.**
  - Estimable: ADNI, pooled runs, and AIBL across the whole cohort (20 AD).
  - OASIS AD (6 subjects) is descriptive only.
  - CALSNIC ALS is an out-of-distribution condition.
- **Zero-shot evaluation set.** It scores every subject of the target cohort, not only the 15–39-subject test split, because nothing from that cohort was fitted.

## 1. Global design (applies to all five parts)

### 1.1 Model matrix: 16 cells plus baselines

Every dynamics model runs on the 128-D code of a frozen representation R. Predictions are decoded by
R's frozen decoder.

- R ∈ {PCA-128, SpiralNet-128, Adaptive-128, LAMM-128}
- D ∈ {cocycle, plain ODE, BrainODE-core, Latent ODE}

| D | transport / generative form | training signal | condition | code source |
|---|---|---|---|---|
| Cocycle (direct_c4) | Φ(z,s,t,d) = z + (t−s)·[v_CN(z,s,t) + d·v_AD(z,s,t)] | latent + decoded-vertex objective (13 leaves) | d at source visit | August `DirectC4Flow` + objective; LAMM via task3_v2_lamm |
| Plain ODE | dz/dτ = f_θ(z, τ, d), residual MLP, RK4 | latent L2 over all forward/backward pair trajectories | fixed d | August `PlainODEFunc` |
| BrainODE-core | same, f_θ = singleton Q/K/V attention field | same as plain ODE (paper recipe) | fixed d | August `BrainODEAttentionFunc` |
| Latent ODE | ODE-RNN encoder → q(z₀) → latent ODE → decoder to the 128-D code; ELBO | codes, reconstruct prefix + extrapolate suffix | d into encoder and dynamics | new (Part 2) |

Baselines need no training and use the same evaluator:

- **No-change:** predict the most recent prefix observation.
- **Linear extrapolation:** least-squares line through the prefix codes. Needs ≥2 prefix visits; this is BrainODE's baseline.
- **Population drift:** source code plus the mean train velocity for its diagnosis. Shows how much of any model is just the group average.

### 1.2 Frozen representation registry R1

| name | checkpoint or basis | reason |
|---|---|---|
| pca128 | first 128 components of ADNI-train PCA-150 (`hippocampus_pca_cocycle_v4/pca/model`) | every PCA anchor uses it |
| spiralnet128 | `task_spiral_ae_v1/studies/spiralnet_z128_v6/trials/trial_0002_val0.037215.pt` | August anchors |
| adaptive128 | `task_spiral_ae_v1/studies/adaptive_z128_v2/trials/trial_0002_val0.036765.pt` | August anchors |
| lamm128 | `task_lamm_ae_v1/studies/E1_N3_ml_s1/best.pt` | LAMM cocycle anchor, task10 |

- **Why not the newer checkpoints.** SpiralNet v7 (0.036784), Adaptive v7 (0.037237) and LAMM E1_N1_ml_s1 (0.037685) reconstruct within 2% of R1. Switching would orphan every longitudinal anchor, so R1 keeps the anchored checkpoints.
- **Provenance.** The registry stores each file's SHA-256; every archive, run and report records it.

### 1.3 Protocols

| id | dynamics fitted on | selected on | tested on | BrainODE analog |
|---|---|---|---|---|
| P0 internal-ADNI | ADNI train | ADNI val | ADNI test | T7 ADNI row |
| P1 external zero-shot | ADNI train (P0 checkpoints) | ADNI val | AIBL/OASIS/CALSNIC: test split and whole cohort | T8 direction |
| P2 internal-X | X train, X ∈ {AIBL, OASIS, CALSNIC} | X val | X test; plus 5-fold subject cross-fit (AIBL, OASIS) | T1 AIBL-regular, T7 rows |
| P3 pooled | ADNI+AIBL+OASIS train | pooled val (macro over cohorts) | each cohort's test, and the unified test | T2 unified irregular |
| P4 LOCO | two of {ADNI, AIBL, OASIS} | their val | held-out cohort, whole | T8 |
| P4b Exp1 replica | AIBL train (P2 checkpoints) | AIBL val | ADNI+OASIS, whole | T8 Exp1 exactly |
| P4c Exp2 analog | pooled (P3 checkpoints) | pooled val | CALSNIC Control, whole; ALS as OOD condition | T8 Exp2 (no LBC) |

- **Representation.** Every protocol uses R1; only the dynamics are refit.
- **PCA sensitivity arm.** Part 4 adds a pooled PCA basis, for PCA only.
- **CALSNIC condition.**
  - Under P1, controls use d = 0, and ALS is scored with both d = 0 and d = 1.
  - Under P2, d is Control = 0 / ALS = 1, which trains a new disease head.

### 1.4 Evaluation tasks, metrics, statistics

Tasks score one target per subject: the latest visit.

| task | prefix | eligibility | aggregation |
|---|---|---|---|
| 1-shot-first (primary) | first visit | ≥2 visits | — |
| 1-shot-prev | penultimate visit | ≥2 visits | — |
| 4-shot (BrainODE-exact) | the 4 visits before the latest | ≥5 visits | mean of 4 predicted meshes |
| all-prior-k | all earlier visits, k ≤ 4 | ≥3 visits | mean of k predictions; Latent ODE also scored with native k-observation encoding |
| all pairs, horizons | existing August pair metrics | — | ≤1 y, 1–2 y, >2 y; forward and backward |

Metrics:

- **Shape error.**
  - Primary: per-vertex Euclidean (mm) against the ground-truth correspondence mesh, end-to-end.
  - Floor-corrected: against the decoded ground-truth code.
  - Coordinate MAE/RMSE, for continuity with earlier tables.
  - Relative volume error.
- **Skill:** 1 − error / no-change error, and the ratio to linear extrapolation.
- **Disease fidelity.**
  - Predicted vs observed log-volume rate per group, giving AD capture % and the AD/CN rate ratio (observed on ADNI ≈ 4.6×).
  - Subject-level annualized slope Pearson and Spearman.
- **Condition sweep** (BrainODE Fig 3): from identical baselines, d = 0 vs d = 1 over 0–10 years. Reports the volume trajectories and the fraction of the observed group gap reproduced.
- **Consistency:** relative semigroup and inverse defects; direct vs chained composition gap.

Statistics:

- **Unit of analysis:** the subject.
  - Mean ± SD (BrainODE style).
  - 95% subject-bootstrap CI: 2000 resamples, stratified by diagnosis.
- **Paired comparisons:** per-subject difference vs the cocycle of the same R.
  - Bootstrap CI and Wilcoxon signed-rank.
  - Holm-corrected within each representation × task.
- **Seeds:** 42/43/44. Report the seed mean ± SD of each aggregate.

Pre-registered primary endpoints:

- **E1:** P0 1-shot-first Euclidean.
- **E2:** P0 AD capture %.
- **E3:** P3 AIBL all-prior-k Euclidean.
- **E4:** P1 AIBL whole-cohort 1-shot-first Euclidean.

Decision rule:

- A dynamics D is preferred over the cocycle when both hold:
  - it improves E2, with a CI that excludes 0;
  - it is non-inferior on E1, within a margin of 2% of the no-change error.
- Otherwise the ranking follows E1.

### 1.5 Validity rules (enforced in code, tested in Part 2)

- **Sealed test.**
  - Training and selection never open test rows.
  - Each test report is written once and records its hash. The evaluator refuses to overwrite.
- **Fitted statistics.**
  - Normalization, time scaling and population drift come from the protocol's training rows only.
  - R1 is fixed and fitted on ADNI train only.
- **Subject separation.**
  - A subject never crosses splits or folds.
  - Subject IDs carry a cohort prefix.
  - Inclusive manifests keep each strict subject's split.
- **LOCO.** The held-out cohort is never used for selection or early stopping.
- **Condition labels.** The condition is the source-visit diagnosis; no future label is ever used.
- **Hyperparameters.**
  - ADNI hyperparameters are reused everywhere.
  - Latent ODE has no ADNI setting, so it gets one bounded, validation-only search on ADNI.
  - Training recipes are used verbatim: the cocycle is step-based, the ODE arms are epoch-based.
  - A run whose selected epoch falls in the last 10% of training is extended 2×. This is a validation-only decision.
  - Steps and epochs are logged.
- **Compute.**
  - GPUs 0 and 2 only; GPU 1 stays free.
  - Every job is detached (`setsid nohup`) under a resumable state file, so an SSH drop is harmless.
- **Locations.**
  - Code: `examples/LHipp_LatentDynamics_CrossCohort_v1/`.
  - Outputs: `/mnt/bulk10tb/Deep3DComp/LHipp_LatentDynamics_CrossCohort_v1/`.

---

## Part 1 — Data foundation: manifests, R1 code archives, task tables, baselines

**Goal:** encode every cohort once in R1. Every protocol, task and fold then exists as audited data
before any model is trained.

### 1A. Implement

1. **Configs**
   - `configs/registry.json`: R1 paths, SHA-256, decoder loader, published validation RMSE.
   - `configs/protocols.json`: P0–P4c, each listing training/selection cohorts, test sets and condition mapping.
   - `configs/tasks.json`: the task definitions from §1.4.
2. **`scripts/p1_inclusive_manifests.py`** builds converter-keeping manifests for AIBL (625 scans / 196 subjects) and OASIS (1521 / 536). It uses the already-built `CrossCohort_LHipp_v1/cohorts_inclusive/*/qc`.
   - Add an `--allow-diagnosis-change` flag to `prepare_adni_synthseg_separate_structure_cohorts.py`. The default stays byte-identical, so ADNI is regression-checked again.
   - New columns:
     - `visit_label` (CN/MCI/AD);
     - `trajectory_group` ∈ {CN-stable, AD-stable, CN→AD, MCI→AD, CN→MCI, MCI-stable, other};
     - `conv_window_a_years` / `conv_window_b_years` (last pre-label visit / first post-label visit);
     - `n_label_changes`.
   - A strict subject keeps its strict split. New subjects are split by the same seed, stratified by `trajectory_group`.
   - Subjects with an unresolvable label (e.g. OASIS `nan` baselines) are kept but flagged `trajectory_group = other`.
   - Part 1 only builds these manifests. They are used in Part 5.
3. **`scripts/p1_encode_archives.py`** covers every cohort × R1 entry: ADNI 2583, AIBL 625, OASIS 1521, CALSNIC 806 scans.
   - One `.npz` per cohort × representation:
     - codes `z` (float32, N×128);
     - `recon_rmse_mm` per scan;
     - `scan_id`, `subject_id`, `cohort`, `split`, visit time (years from baseline), `age`, `visit_label`, `trajectory_group`;
     - checkpoint SHA-256.
   - Encoding: SpiralNet/Adaptive via `spiral_common` with `SPIRAL_COHORT_TAG`; LAMM via its own loader; PCA via `xcohort_common.PCAModel`.
   - GPU 2. GPU 0 is still busy with CrossCohort LOCO. Batch-only inference.
4. **`scripts/p1_build_tasks.py`** writes, per protocol:
   - **Training tables:**
     - ordered pairs (i<j and j<i) for the cocycle and ODE arms;
     - padded sequences with masks for Latent ODE;
     - sequence tables for the cocycle's sequence leaves.
   - **Evaluation tables** for every §1.4 task, with prefix scan IDs, target scan ID, horizon, diagnosis and eligibility.
   - **Cross-fit folds** for P2 on AIBL and OASIS: 5 folds, subject-level, stratified by diagnosis × visit count, seed 42.
   - **Normalization statistics** (code mean/std, time scale), fitted on each protocol's training rows only.
5. **`scripts/p1_baselines.py`** computes no-change, linear extrapolation (code space and vertex space) and population drift on every protocol × task. Rows use the final result schema, so Parts 3–4 tables include baselines without any GPU.
6. **`scripts/p1_audit.py`** and **`tests/test_part1.py`** run the gates below and write `reports/part1_audit.md`.

### 1B. Fit / train / validate / test

- **Fitting:** protocol normalization statistics and population-drift velocities, from training rows only.
  No learned model is trained in Part 1.
- **Validation:** gates G1.x only.
- **Testing:** baseline test metrics are computed now; they involve no selection. Written once to sealed baseline reports.

### 1C. Gates (all must pass before Part 2 runs)

| gate | check | pass criterion |
|---|---|---|
| G1.1 | topology hash and faces for all cohorts | identical to ADNI (`a8485554…`) |
| G1.2 | ADNI archives vs August / v3-LAMM archives | max abs code diff ≤ 1e-5, same scan order |
| G1.3 | reconstruction from archives | PCA-128 per-cohort RMSE equals CrossCohort phase 3 (±1e-6); learned ADNI val within ±0.5% of published |
| G1.4 | split and fold hygiene | no subject in two splits or folds; inclusive keeps strict splits; LOCO training tables exclude held-out IDs |
| G1.5 | evaluable counts | reproduce §0.3 exactly |
| G1.6 | baseline regression | ADNI test no-change FL MAE equals August `nochange_*` fields (±1e-6) |
| G1.7 | inclusive manifests | ADNI rebuild byte-identical; converter counts match the QC inventory (AIBL CN→AD 5, MCI→AD 11; OASIS 7, 17) |
| G1.8 | unit tests | `pytest tests/test_part1.py` passes |

### 1D. Outputs and runtime

- **Outputs**
  - `archives/<cohort>/<rep>.npz`
  - `tasks/<protocol>/{train_pairs,val_pairs,sequences,eval_<task>}.csv`
  - `folds/<cohort>_cv5.csv`
  - `results/baselines.csv`
  - `reports/part1_audit.md`
- **Runtime**
  - Encoding: ~20–30 min on one GPU (≈5.5k scans × 4 representations).
  - Tasks, baselines and audit: CPU, ≤20 min.
  - Runs detached; safe to launch alongside the running CrossCohort LOCO job.

---

## Part 2 — Model zoo, unified trainer and evaluator, anchor reproduction

**Goal:** one harness that can train and evaluate any of the 16 cells under any protocol. It must
reproduce the existing ADNI anchors before a single new run is trusted.

### 2A. Implement

1. **`scripts/dyn/registry.py`** builds cell (R, D) from a config.
   - It loads R1 decoders and the Part 1 tables.
   - It exposes `transport(z, s, t, d)` (cocycle/ODE) or `predict(prefix, t_target, d)` (Latent ODE) behind one interface.
2. **Cocycle adapter**
   - Imports August `DirectC4Flow` and its objective read-only for PCA, SpiralNet and Adaptive, and the `task3_latent_flow_128_v2_lamm` code for LAMM. Nothing is forked, so anchors stay reproducible.
   - Adds `selection.min_epoch`. The default 0 keeps the published rule; 15 is the sensitivity rule.
   - Saves the best checkpoint under both rules during the same run, so no retraining is needed.
3. **Plain ODE and BrainODE-core**
   - August funcs and trainer, plus new `lamm128_plain_ode` and `lamm128_brainode` configs with identical hyperparameters.
   - **BrainODE speed-up.**
     - Profile 20 batches with `torch.profiler`.
     - Vectorize the all-combination forward/backward trajectories into one batched RK4 solve per minibatch.
     - Equivalence test on a fixed init and batch: |Δloss| ≤ 1e-6, max |Δgrad| ≤ 1e-5, and a 3-epoch loss curve identical within 1e-5.
     - Target: ≤ 40 min per ADNI run. If equivalence fails, keep the original code and budget for it.
4. **Latent ODE (Rubanova et al. 2019).** Implemented in-house with the shared RK4, since `torchdiffeq` is not installed in either env.
   - **Data**
     - Observations x_i ∈ ℝ¹²⁸: codes standardized with protocol training statistics.
     - Times τ_i: the same encoding the August ODE arms use.
     - Condition d ∈ {0, 1}.
   - **Encoder (ODE-RNN), run backward in time from τ_k to τ_1**
     - Between observations: dh/dτ = f_enc(h), RK4 with 4 substeps.
     - At each observation: h ← GRU(h, [x_i, d, Δτ_i]).
     - Output: (μ, log σ) of q(z₀ ∣ x_1..k, d) at τ_1.
   - **Generative model**
     - Prior: z₀ ~ N(0, I).
     - Dynamics: dz/dτ = f_θ(z, τ, d) (residual MLP, width 256, 2 blocks, tanh).
     - Decoder: x̂ = g(z(τ)) (2-layer MLP).
     - Likelihood: N(x ∣ x̂, diag σ_obs²), with learned per-dimension σ_obs and a floor.
   - **Objective (extrapolation ELBO)**
     - For a subject with n visits, sample a prefix size k ~ U{1..n−1}; the encoder sees visits 1..k.
     - Loss = −Σ_{i=1..n} log p(x_i ∣ z(τ_i)) + β·KL(q ‖ p).
     - β is annealed from 0 to 1 over the first 10 epochs, with 0.1 nats/dim free bits against posterior collapse.
     - Forward time only, as in the original.
   - **Prediction**
     - Integrate from τ_1 to τ_target with the posterior mean.
     - 20 posterior samples give prediction spread for calibration.
     - 1-shot is the encoder given one observation. all-prior-k is native k-observation encoding, plus BrainODE-style averaging for parity.
   - **Bounded search:** ADNI, PCA-128, seed 42, validation only, 16 trials.
     - z₀ dim ∈ {32, 64, 128}
     - GRU hidden ∈ {128, 256}
     - lr ∈ {3e-4, 1e-3}
     - σ_obs floor ∈ {0.01, 0.1}
     - KL anneal ∈ {5, 20} epochs
     - Selection uses the macro CN/AD first→last decoded-shape ratio shared by all arms. The winner is frozen and reused for every representation.
   - **Ablation:** unconditional Latent ODE (d removed), for the condition-sensitivity analysis.
5. **`scripts/evaluate_tasks.py`** implements every §1.4 task and metric for all 16 cells and the baselines.
   - Outputs: `summary.json`, `per_subject.csv`, bootstrap CIs, and paired differences vs the cocycle of the same R.
   - Keeps the August pair-metric families, so old and new tables line up.
6. **`scripts/train_dynamics.py`** takes `--cell --protocol --seed [--fold]` and writes resolved config, run contract, history, status and the selected checkpoint(s).
   - A loader-level guard raises if a test scan ID enters any training or selection loader.
7. **`scripts/orchestrate.py`** follows the `run_sequence.py` pattern.
   - Resumable JSON state; job classes with per-GPU slots (latent-only: 3 per GPU; decoder-in-loss: 2 per GPU).
   - GPU allowlist 0,2. Runs detached with `setsid nohup`.
   - Dry-run mode, a precondition wait for busy GPUs, and a log per job.
8. **`tests/`**: algebra, leakage, determinism, overfit and equivalence tests (below).

### 2B. Fit / train / validate / test in this part

- **Fitting:** none beyond Part 1 statistics.
- **Training:**
  - smoke runs;
  - two anchor retrains: PCA cocycle s42 and PCA plain ODE s42;
  - the Latent ODE search.
- **Validation:** Latent ODE search selection (ADNI val only), plus all gates.
- **Testing:** re-evaluate existing s42 anchor checkpoints on ADNI test with the new evaluator. This is allowed: these checkpoints were selected long ago and nothing is being selected now.

### 2C. Gates

| gate | check | pass criterion |
|---|---|---|
| G2.1 | RK4 on dz/dτ = Az vs matrix exponential | error ≤ 1e-6 |
| G2.2 | cocycle identity Φ(z,s,s,d) = z; ODE semigroup and inverse | exact; ODE defects ≤ 1e-5 |
| G2.3 | Latent ODE correctness | KL ≥ 0; ELBO finite; padding/mask invariance; reversed-time encoder test; gradient check |
| G2.4 | overfit 8 subjects, every D | train loss drops ≥ 95% |
| G2.5 | determinism | same seed → identical first-epoch losses |
| G2.6 | leakage guards | a planted test ID raises; LOCO held-out IDs rejected |
| G2.7a | new evaluator on existing s42 checkpoints (10 cells) | FL MAE, all-pair MAE and AD rate equal stored summaries (±1e-6) |
| G2.7b | retrain PCA cocycle and PCA plain ODE, s42 | FL MAE within ±0.002 of anchors; AD capture within ±5 points |
| G2.8 | BrainODE vectorization | equivalence test passes; measured minutes per run recorded |
| G2.9 | smoke: 16 cells × P0, 2 batches + validation | all finish on GPU 2 in < 15 min total |
| G2.10 | Latent ODE search | 16 trials complete; winner frozen in `configs/dynamics/latent_ode.json` |

### 2D. Runtime

- Tests: CPU, minutes.
- Smoke: ~15 min.
- Anchor evaluation: ~30 min GPU.
- Two anchor retrains: ~15 min.
- Latent ODE search: ~16 × 15–25 min ≈ 5 GPU-h, ~1.5 h wall with 3 slots on each of GPUs 0 and 2.
- LAMM cocycle: measured 9.5 min per ADNI run (`task3_latent_flow_128_v3_lamm_latest`, 31 epochs).

---

## Part 3 — ADNI internal matrix (P0) and BrainODE-style ADNI results

**Goal:** the headline comparison. It covers 16 cells × 3 seeds on ADNI, selected on validation and
tested once.

### 3A. Fit

- Normalization statistics from ADNI train codes (Part 1).
- R1 stays frozen.
- Latent ODE hyperparameters are frozen from G2.10.

### 3B. Train

| arm | PCA | SpiralNet | Adaptive | LAMM | runs |
|---|---|---|---|---|---|
| cocycle | s43, s44 (s42 anchor reused) | s43, s44 | s43, s44 | s42, s43, s44 | 9 |
| plain ODE | s43, s44 | s43, s44 | s43, s44 | s42, s43, s44 | 9 |
| BrainODE-core | s43, s44 | s43, s44 | s43, s44 | s42, s43, s44 | 9 |
| Latent ODE (faithful) | s42–s44 | s42–s44 | s42–s44 | s42–s44 | 12 |
| Latent ODE (residual) | s42–s44 | s42–s44 | s42–s44 | s42–s44 | 12 |

- **Totals:** 51 new training runs, plus 9 reused seed-42 anchors (August PCA/SpiralNet/Adaptive x cocycle/plain ODE/BrainODE). Reuse requires G2.7a to pass; otherwise those cells are retrained too.
- **LAMM cocycle s42 is trained, not reused.** The earlier LAMM cocycle (task3_v3_lamm_latest) used the older task3 recipe (lr 3e-4, no dropout, tolerance 0.01) and trainer, not the August recipe every other cell uses. It still serves as an evaluator regression check in G2.7a.
- **Optional ablations** (flagged; run only on request):
  - **A1:** exact coboundary and volume coboundary v2, PCA and SpiralNet, trained to completion. The previous runs stopped early.
  - **A2:** BrainODE-V, i.e. BrainODE-core plus the cocycle's decoded-vertex and volume/rate leaves (PCA). It separates "loss" from "structure" as the source of AD capture.
  - **A3:** cocycle without the disease head (v_AD removed, PCA). Shows how much AD capture the d·v_AD term carries.
  - **A4:** unconditional Latent ODE (PCA).

### 3C. Validate (selection)

- Every run selects on ADNI val with its arm's rule.
  - **Cocycle:** macro CN/AD FL decoded-shape ratio + 0.2 × all-pair ratio + volume tiebreak, gated at relative semigroup/inverse defect ≤ 0.25.
  - **ODE arms and Latent ODE:** the same primary ratio.
- The cocycle also stores its min-epoch-15 selection (sensitivity, reported separately).
- Selection logs record every validation score. G3.2 verifies no test access.

### 3D. Test (sealed, once, after every selection is frozen)

- **Tasks:** 1-shot-first, 1-shot-prev, all-prior-k, all pairs, horizons.
  - 4-shot is scored on the 21 ADNI test subjects with ≥5 visits. Reported as CN, because the AD group has n = 1.
- **Metrics and statistics:** as in §1.4. Seed mean ± SD; subject bootstrap; paired differences vs the cocycle, Holm-corrected.

### 3E. Gates

| gate | criterion |
|---|---|
| G3.1 | 48 cells × seeds complete, each with status, selected checkpoint(s) and a val summary |
| G3.2 | no test scan ID in any training or selection log |
| G3.3 | reused s42 anchors reproduce stored test metrics (from G2.7a) |
| G3.4 | every trained model beats no-change on ADNI val FL; failures are flagged, never silently dropped |
| G3.5 | cocycle defects within gate; ODE defects ≤ 1e-5 |
| G3.6 | each test report written once, hash recorded |

### 3F. Deliverables

`reports/part3_adni_internal.md` and CSVs:

- **P3-A:** 4×4 + baselines table. Euclidean mean ± SD for 1-shot-first, 1-shot-prev, all-prior-k and 4-shot (CN), plus FL/all-pair MAE for continuity.
- **P3-B:** disease fidelity. AD capture %, AD/CN rate ratio, subject slope correlation, disease gap.
- **P3-C:** consistency and cost. Defects, parameters, GPU-minutes.
- **P3-D:** error vs horizon (≤1, 1–2, >2 y), forward and backward.
- **P3-E:** condition sweep (BrainODE Fig 3). Volume vs years for d = 0 / 1 from matched baselines, every cell.
- **P3-F:** forest plot of paired differences vs the cocycle, per R, with 95% CI.
- **P3-G:** seed-sensitivity and min-epoch-15 selection table.

### 3G. Budget

| arm | min per ADNI run | new runs | GPU-min |
|---|---|---|---|
| cocycle | PCA 3.5, SpiralNet 21, Adaptive 36, LAMM 9.5 | 8 | ~140 |
| plain ODE | 8–10 | 9 | ~90 |
| BrainODE-core | 205 now; ≤40 if vectorized | 9 | 1845 now; ~360 vectorized |
| Latent ODE | ~15–25 (measured in Part 2) | 12 | ~300 |

- **GPU time:** ~40 GPU-h with the current BrainODE; ~15 GPU-h vectorized.
- **Wall time:** ~4–8 h on GPUs 0 and 2 with the slot rule, vectorized. Evaluation adds ~2 GPU-h.

### 3H. Expectations (from the anchors; hypotheses, not results)

- **FL MAE:** cocycle 0.1536–0.1543, plain ODE 0.1548–0.1606, BrainODE-core 0.1576–0.1633. Seed SD is expected ≈ 0.001 and will be measured.
- **AD capture:** cocycle 84–93%; plain ODE and BrainODE-core 28–53%.
- **Latent ODE:**
  - 1-shot: expected close to plain ODE (±3%), since one observation gives its encoder little to use.
  - all-prior-k: may beat averaging.
  - AD capture: likely ≤ 50%. The condition enters only through d, with no disease-specific velocity head.
  - BrainODE saw ~2× worse Latent ODE error. The gap should be smaller here, because every arm shares one frozen decoder.
- **LAMM ODE arms:** close to the SpiralNet ODE arms. LAMM's cocycle already matches the others at 0.1541.
- **Pre-registered decision:** E1 and E2 (§1.4) decide which dynamics is preferred on ADNI.

---

## Part 4 — Cross-cohort dynamics (P1–P4c) and the BrainODE-style per-dataset tables

**Goal:** show whether each dynamics model transfers, benefits from pooling, and survives
leave-one-cohort-out. Every dataset gets its own BrainODE-style table.

### 4A. Fit

- **P1:** nothing is fitted. ADNI normalization and the P0 checkpoints (3 seeds), verified by hash.
- **P2, P3, P4:** normalization statistics and population drift from each protocol's training rows (Part 1).
- **CALSNIC P2:** d = 1 means ALS. The disease head is learned from CALSNIC train only.
- **Sensitivity arm:** a pooled PCA basis (ADNI+AIBL+OASIS train), for PCA cells under P3 only. It needs one extra registry entry, `pca128_pooled`, and one re-encode.

### 4B. Train

| protocol | cells | seeds / folds | runs |
|---|---|---|---|
| P1 external zero-shot | — | reuse P0 (s42–s44) | 0 |
| P2 internal AIBL, OASIS, CALSNIC | 16 | s42 | 48 |
| P2 cross-fit AIBL, OASIS (core) | PCA + best learned R from Part 3, × 4 D | 5 folds × 2 cohorts | 80 |
| P2 cross-fit (full, optional) | 16 | 5 folds × 2 cohorts | 160 |
| P3 pooled | 16 | s42–s44 | 48 |
| P4 LOCO | 16 | 3 held-out cohorts × s42 | 48 |
| P4b Exp1, P4c Exp2 | — | reuse P2 AIBL / P3 | 0 |

- **Core total:** 224 runs.
- **Training recipes are used verbatim.**
  - The cocycle is step-based (4096 samples per epoch).
  - The ODE arms and Latent ODE are epoch-based (100 epochs).
- **Extension rule:** if the selected epoch falls in the last 10% of training, that run is extended 2×. This is a validation-only decision, logged in the run contract.
- **Why not match optimizer steps to ADNI:** it would make every small-cohort run cost as much as an ADNI run. ADNI's ODE arms mostly selected early (plain ODE epochs 3–14; BrainODE 11–64), so the extension rule covers the undertraining risk far more cheaply.

### 4C. Validate (selection)

- **In-protocol validation only.**
  - **P2:** X val.
  - **P3:** macro over cohorts of each cohort's selection score. A cohort without AD in val contributes only its CN term; OASIS val has 1 AD subject, so it counts as CN only.
  - **P4:** val of the two training cohorts only.
  - **Cross-fit:** nested. One of the 4 training folds is the inner validation, so the outer fold stays test-only.
- **No retuning:** ADNI hyperparameters, and Latent ODE's frozen setting from G2.10.

### 4D. Test (sealed, once per protocol)

| protocol | test sets | tasks and notes |
|---|---|---|
| P1 | AIBL, OASIS, CALSNIC: test split and whole cohort | all tasks. AIBL = regular analysis; OASIS 4-shot on its 25 whole-cohort CN subjects; CALSNIC ALS scored with d = 0 and d = 1 |
| P2 | X test, plus out-of-fold predictions | whole-cohort metrics from cross-fit: AIBL 148 subjects / 20 AD; OASIS 385 / 6 |
| P3 | each cohort's test split and unified ADNI+AIBL+OASIS test | T2 analog: unified 4-shot and 1-shot |
| P4 | held-out cohort, whole | T8 analog |
| P4b | ADNI+OASIS, whole | exact Exp1 replica |
| P4c | CALSNIC Control, whole; ALS as OOD | Exp2 analog |

### 4E. Analyses

1. **Transfer and pooling gaps** per cohort × cell, with paired subject bootstrap.
   - Transfer gap = P1 − P2.
   - Pooling gain = P2 − P3.
2. **Interval shift.** Error and skill vs interval bins (≤1, 1–2, 2–3, >3 y), pooled over cohorts. This tests BrainODE's explanation for its T8 degradation.
3. **AD fidelity outside ADNI.**
   - AIBL whole-cohort AD capture under P1, P3 and P4 (held-out AIBL).
   - OASIS CN aging-rate fidelity vs observed (SynthSeg CN-stable hippocampus −0.62%/y).
4. **Data dependence.** P4 without ADNI (~20 AD training subjects) vs P3. In BrainODE's data, 77% of AD subjects came from ADNI; this quantifies what that dependence costs.
5. **CALSNIC as OOD.**
   - Skill vs no-change over a 0.68 y median span.
   - P2 Control vs ALS predicted-rate gap.
   - d-sweep: d = 1 means AD-like loss. Observed hippocampal decline is ALS −1.72%/y vs AD −4.31%/y, so d = 1 should overshoot if conditioning is disease-specific.
6. **Sensitivity checks.**
   - BrainODE-matched age subset (65–95).
   - Pooled PCA basis (PCA cells, P3).
   - Min-epoch-15 cocycle selection.
7. **Representation effect.** Same D across R within each protocol: does a learned encoder transfer better longitudinally than PCA?

### 4F. Gates

| gate | criterion |
|---|---|
| G4.1 | per-protocol task counts equal the Part 1 tables |
| G4.2 | LOCO held-out IDs absent from all training and selection logs |
| G4.3 | P1 uses ADNI normalization and P0 checkpoints (hash equality) |
| G4.4 | each eligible cross-fit subject has exactly one out-of-fold prediction; inner val never overlaps outer test |
| G4.5 | every run selected on in-protocol val; every test report written once |
| G4.6 | CALSNIC ALS head trained on CALSNIC train only |

### 4G. Deliverables

`reports/part4_cross_cohort.md` and CSVs:

- **P4-A (T7 analog):** per-dataset table for every protocol.
  - 16 cells + baselines.
  - Euclidean mean ± SD for 1-shot-first, all-prior-k and 4-shot where eligible.
  - Skill; AD capture where estimable.
- **P4-B (T1 analog):** AIBL-regular table under P1, P2 (+ cross-fit) and P3.
- **P4-C (T2 analog):** unified irregular test under P3.
- **P4-D (T8 analog):** P4 × 3, the P4b Exp1 replica, the P4c Exp2 analog.
- **P4-E:** transfer and pooling gaps.
- **P4-F:** interval-shift curves.
- **P4-G:** AD fidelity outside ADNI.
- **P4-H:** CALSNIC OOD panel.

### 4H. Budget

Scale factors: ODE-type arm cost scales with training rows s relative to ADNI:

| training set | s |
|---|---|
| AIBL | 0.19 |
| OASIS | 0.42 |
| CALSNIC | 0.32 |
| pooled | 1.6 |
| LOCO without ADNI | 0.6 |
| LOCO without AIBL | 1.42 |
| LOCO without OASIS | 1.19 |

Cost per seed across all 4 representations, at ADNI scale:

- **Cocycle:** ~70 min (PCA 3.5 + SpiralNet 21 + Adaptive 36 + LAMM 9.5), independent of cohort size.
- **ODE-type arms:** ~280 min with vectorized BrainODE (plain 40 + BrainODE 160 + Latent ODE ~80). About 940 min with the current BrainODE.

| block | GPU-h (vectorized BrainODE) |
|---|---|
| P2 internal, s42 | ~8 |
| P2 cross-fit, core | ~11 |
| P3 pooled, 3 seeds | ~26 |
| P4 LOCO, s42 | ~18 |
| evaluation, all protocols | ~4 |
| **core total** | **~67 GPU-h, ≈ 12–20 h wall on GPUs 0 + 2** |

- **Prerequisite:** with the current BrainODE the core grows to ~150+ GPU-h, so vectorization (G2.8) gates Part 4.
- **If G2.8 fails:** BrainODE runs s42 only in P3, and core cells only in cross-fit.

### 4I. Expectations (hypotheses)

- **P1 AIBL:** skill vs no-change largely retained, since AIBL resembles ADNI in ages and intervals. Absolute error rises with the representation floor: PCA external reconstruction RMSE is AIBL 0.080 vs ADNI 0.034.
- **OASIS:** longer intervals give a larger no-change error. Models keep more relative skill if their dynamics extrapolate, with some degradation from the time shift, as in BrainODE T8.
- **CALSNIC:** no-change is nearly optimal over 0.68 y (model skill ≤ a few %). d = 1 over-predicts ALS hippocampal loss.
- **P2 small cohorts:** the AD head is poorly identified (AIBL 16 and OASIS 4 AD training subjects). Expect P1 ≥ P2, mirroring the autoencoder results.
- **P3 pooled:** best or tied for AIBL and OASIS; ADNI unchanged within CI.
- **P4 without ADNI:** AD capture collapses for every arm.
- **Cocycle advantage:** its AD-fidelity lead should hold wherever AD training data is ample (P0, P1, P3, and LOCO without AIBL or OASIS). Where AD data is scarce, all methods regress toward CN aging.

---

## Part 5 — Consolidated BrainODE-style report and the converter line

### 5A. Consolidated report (CPU)

`scripts/report_brainode_style.py` builds everything from result CSVs only. Output is deterministic.

- **Tables**
  - **R-T5/T6:** composition per cohort and evaluable subjects per task.
    - Composition: subjects; NC / AD / ALS; age at first visit, observations and intervals (each mean ± SD).
  - **R-T1:** AIBL-regular, under P1, P2 and P3.
  - **R-T2:** unified irregular, P3.
  - **R-T7:** per dataset, under P0–P3.
  - **R-T8:** LOCO, the Exp1 replica, the Exp2 analog.
  - **R-T3:** ablations A1–A4 and converter-line results.
  - Disease-fidelity table; consistency and cost table.
  - **Context table:** BrainODE's published T1/T2/T7/T8 numbers beside ours, explicitly marked not head-to-head.
- **Figures**
  - Condition sweeps (Fig 3 / C.3).
  - Error vs number of shots; error vs horizon and interval.
  - Paired-difference forest plots.
  - AD subject slope scatter.
  - Per-vertex error maps for representative CN, AD and AIBL subjects (Fig 2 analog).
- **Traceability:** every cell records run ID, config hash, checkpoint SHA-256, git commit and report hash.

Gates:

- **G5.1:** the report regenerates byte-identically.
- **G5.2:** 10 random cells match their raw summaries.
- **G5.3:** E1–E4 are reported with CI and the decision-rule outcome.

### 5B. Converter line (separate experiment; inclusive AIBL + OASIS)

**Data:** Part 1 inclusive manifests. Converters with ≥2 meshed visits:

| cohort | CN→AD | MCI→AD | CN→MCI |
|---|---|---|---|
| AIBL | 5 | 11 | 14 |
| OASIS | 7 | 17 | 43 |

**Model C1 — converter cocycle**

- Condition path and onset:
  - c_i(τ) = σ((τ − τ_i)/w)
  - τ_i = a_i + (b_i − a_i)·σ(θ_i), where [a_i, b_i] is the conversion window
  - w = 0.25 y (sensitivity: 0.5 y)
- Per-leg average dose:
  - c̄_i(s,t) = w·[softplus((t − τ_i)/w) − softplus((s − τ_i)/w)] / (t − s)
  - c̄_i(s,s) = c_i(s)
- Transport: Φ_i(z,s,t) = z + (t − s)·[v_CN(z,s,t) + c̄_i(s,t)·v_AD(z,s,t)]
- **Exact dose additivity:** (u − s)·c̄(s,u) + (t − u)·c̄(u,t) = (t − s)·c̄(s,t).
  - Verified numerically at 2.2e-16.
  - Using the source-visit label instead breaks composition by 0.35 in the same test; the per-leg c̄ gives 4.5e-16.
  - C1's composition defect therefore equals direct_c4's; the time-varying condition adds none.
- **Stable subjects:** c ≡ 0 or 1, so C1 reduces exactly to direct_c4.
- **Exploratory MCI variant:** MCI→AD uses the same window. CN→MCI gets a partial dose, κ_MCI·c_i(τ), with a learned global κ_MCI ∈ (0, 1).

**Fit**

- v_CN and v_AD are initialized from the P3 pooled cocycle, which was trained on stable subjects only. Population priors come from stable subjects only.
- Converter stage (train subjects only):
  - Learn θ_i with penalty λ‖θ_i‖², which pulls onset toward the window midpoint.
  - Heads fine-tuned at 0.1× lr (C1), or frozen (C1-frozen).

**Train:** PCA + the best learned R from Part 3; seeds 42–44.

**Validate:** macro first→last decoded-shape ratio over {CN-stable, AD-stable, converters} on val. Stable-subject error must stay non-inferior to P3 within 1%.

**Test (no leakage)**

- A val/test subject's onset is never optimized on its own future visits.
- **Prefix-only onset:**
  - If the observed prefix already contains the post-conversion label, the window comes from prefix labels and τ_i sits at its midpoint.
  - Otherwise the subject keeps its current label; this is true forecasting.
- **Oracle-window variant:** window from all visits. Reported separately as an upper bound.

**Comparators**

- **C0:** direct_c4 with the source-visit label.
- **BrainODE-core:** source-visit label.
- **BrainODE-full, PCA only:** BrainODE-core plus pseudo-cognitive sampling and a cognition estimator τθ (shape → c).
  - Pseudo samples are interpolated intermediate shapes with c̃ = relative position between endpoints.
  - First check whether the older PCA-150 BrainODE folders have reusable code (`ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/brainode_pca150_qc_stable`, `ADNI_1_L_No_MCI/brainode_comparison_task3_core_brainode_original`). The task3 cognition code has pseudo sampling explicitly off.
- **Optional:** plain ODE driven by the c_i(τ) path, the ODE analog of C1.

**Evaluation**

- **E-C1:** latest-shape Euclidean per group (CN-stable, AD-stable, CN→AD, MCI→AD), for 1-shot-first and all-prior-k. Analog of BrainODE T3 (NC&AD vs CONV).
- **E-C2: synthetic onset recovery.**
  - Generate synthetic converters with C0 from held-out stable CN baselines, switching d at a known τ inside a sampled window, with noise matched to observed residuals.
  - Recover τ̂ from the visits; report median |τ̂ − τ| and 80% interval coverage.
- **E-C3:** composition defects along converter trajectories should match stable subjects.
- **E-C4:** τθ accuracy for BrainODE-full (paper: 0.883 → 0.891).
- **E-C5:** AD capture within the MCI→AD group.

**Gates**

- **G5.4:** additivity ≤ 1e-12, and C1 ≡ direct_c4 on stable subjects ≤ 1e-6.
- **G5.5:** test converters' θ receive no gradient (parameter hash unchanged).
- **G5.6:** synthetic recovery median |τ̂ − τ| ≤ 0.5 y before any real-data test.
- **G5.7:** stable non-inferiority vs P3.

**Expectations**

- Few converters (12 CN→AD and 28 MCI→AD in total), so results are descriptive with wide CIs.
- C1 should beat C0 mainly when conversion falls inside the prediction horizon and the prefix reveals it.
- Stable subjects are unchanged by construction.
- BrainODE-full's CONV gain (T3: 0.282 → 0.216) shows the direction to compare, not the magnitude.

**Budget:** ≤ 10 GPU-h.

### 5C. LV extension (after the hippocampus line is complete)

- Same Parts 1–4 for the left lateral ventricle, in a sibling folder. BrainODE reports LV too.
- All four cohorts are already meshed on the ADNI LV template (V = 8346).
- LV needs its own frozen representation registry; checkpoint availability is checked then.

---

## Execution summary

| part | implement | run (only when asked) | GPU-h | wall |
|---|---|---|---|---|
| 1 | manifests, archives, task tables, baselines, audit | encode, build, audit | < 1 | ~1 h |
| 2 | harness, Latent ODE, BrainODE speed-up, evaluator, orchestrator, tests | tests, smoke, anchor checks, Latent ODE search | ~6 | ~3 h |
| 3 | P0 configs (38 new runs) + report | ADNI matrix + sealed test | ~15 (~40 if BrainODE not vectorized) | 4–8 h |
| 4 | protocol runner, cross-fit, report | 224 core runs + sealed tests | ~67 (~150+) | 12–20 h |
| 5 | report builder, converter cocycle, BrainODE-full | converter runs + final report | ≤ 10 | ~4 h |

Operating rules:

- GPUs 0 and 2 only.
- Detached, resumable jobs.
- One part at a time. Each part ends with its gate report, and nothing advances past a failed gate.
- CrossCohort_LHipp_v1's LOCO-without-OASIS job is still on GPU 0, so Part 1 encoding uses GPU 2 until it finishes.

## Risks and mitigations

| risk | mitigation |
|---|---|
| BrainODE runtime (205 min/run) | vectorize under an equivalence test (G2.8); fallback: fewer BrainODE seeds and cells |
| Latent ODE posterior collapse | KL annealing and free bits; monitor per-dimension KL |
| Early checkpoint selection (epochs 1–5) | three seeds; min-epoch-15 sensitivity selection |
| AD scarcity outside ADNI | whole-cohort zero-shot, cross-fit, pooling; OASIS AD descriptive only |
| mm not comparable with BrainODE | context table only; claims rest on within-benchmark ranks and skill |
| CALSNIC short span, ALS ≠ AD | OOD framing; no AD claims from CALSNIC |
| many comparisons | pre-registered E1–E4; Holm correction |
| few converters | synthetic recovery gate; descriptive reporting |

## Decisions recorded during implementation

- **Stage 1 (2026-09-12).** Population drift stays in the stage-1 baseline outputs but is not a required comparator (user decision).
- **Stage 2 recipes.** One recipe per method for all four representations: the August recipes behind the latest good results (`configs/recipes/*.json`).
  - Only the cocycle's memory-driven batch sizes differ (PCA 96, SpiralNet 24, Adaptive 8, LAMM 96).
  - The August ODE config key `early_stopping_patience` was never read by that trainer, so it is omitted.
- **BrainODE speed-up.** Already present in the task3 core: the singleton Q/K/V block is evaluated vectorized, which is mathematically identical to August's per-subject loop. The ~205-minute August runs came from that loop. G2.8 is therefore an equivalence test (outputs and gradients), not new code.
- **Trainers.**
  - `train_cocycle.py` keeps August train_c4 semantics: patience counts genuine improvement; best.pt is the best feasible epoch.
  - `train_ode_transport.py` imports task3's trajectory trainer functions unchanged.
  - Both read stage-1 views and add a test-leakage guard.
  - The cocycle also saves best_min_epoch.pt (min epoch 15) for the sensitivity analysis.
- **Latent ODE** (`rubanova_latent_ode.py`, `train_rubanova_latent_ode.py`):
  - Extrapolation ELBO with free bits, using the shared RK4.
  - Selection uses the same first-to-last validation score as the ODE arms.
  - The 16-trial validation-only search writes `configs/recipes/latent_ode_selected_overrides.json`, which every later Latent ODE run applies.
- **Latent ODE diagnostics and the residual variant (2026-09-12, user decision: faithful plus residual).**
  - The first search attempt had a learned observation sigma and only 8 optimizer steps per epoch; every trial stayed at code MSE ~1 (validation ~2x no-change). It was stopped and archived under `stage2_validation/discarded/`.
  - After fixing sigma (searched) and drawing 4096 subjects per epoch, the faithful model learns but overfits: validation 1.66–1.89x the no-change error. It cannot rebuild a new subject's full 128-D code more accurately than two years of atrophy. This matches BrainODE's report of Latent ODE errors roughly 2x BrainODE's.
  - The matrix is therefore 4 representations x 5 dynamics = 20 cells. `latent_ode` stays faithful to the paper. `latent_ode_residual` predicts change from the last observation, x_hat(t) = x_ref + g(z(t)) - g(z(t_ref)), and is a documented departure that predicts exactly no-change before training.
  - Both variants use early stopping (patience 25) and the same 16 search trials (same grid and sampler seed), each selected on validation independently.
- **Latent ODE search outcome (2026-09-13).** Validation score on ADNI PCA-128: 1.0 = no-change, lower is better; the trained plain ODE scores 0.930.
  - Faithful Latent ODE: best 1.553 (encoder 256, z0 128, sigma 0.3, lr 3e-4, KL anneal 5); range 1.55–1.94. Every setting is worse than no-change.
  - Residual Latent ODE: best 0.903 (encoder 256, z0 64, sigma 0.3, lr 3e-4, KL anneal 5); range 0.903–0.925. Every setting beats no-change, and it is insensitive to settings.
  - Selected settings are in `configs/recipes/{latent_ode,latent_ode_residual}_selected_overrides.json`; reports are in `stage2_validation/reports/`.
- **Stage 3 complete (2026-09-13).** 60 seed-runs on ADNI test (61 subjects); gates G3.1–G3.6 pass.
  - One-shot first-to-last Euclidean error: cocycle 0.312–0.316 mm and residual Latent ODE 0.310–0.314 mm, statistically tied; plain ODE 0.316–0.326; BrainODE 0.319–0.329; faithful Latent ODE 0.41–0.56; no-change 0.344.
  - Cocycle vs plain ODE / BrainODE on error is significant only for PCA (Holm p 0.005 / 0.002); with learned encoders the gaps are 0.002–0.009 mm and not significant.
  - AD atrophy capture: cocycle 0.81–0.84, residual Latent ODE 0.72–0.91, plain ODE 0.30–0.54, BrainODE 0.27–0.52. The condition sweep attributes 42–52% (cocycle) and 28–51% (residual) of the observed AD–CN rate gap to the disease condition, versus 2–8% for plain ODE and BrainODE.
  - No model predicts individual AD atrophy rates: subject-level slope correlations run from −0.47 to +0.29.
  - The epoch ≥ 15 cocycle checkpoints are slightly worse (error +0.002 to +0.012 mm, AD capture lower), so best.pt stays primary.
  - Operational: one Adaptive test evaluation ran out of GPU memory with 4 evaluations sharing a GPU with training. It was rerun, and evaluate_dynamics.py now uses representation-specific batch sizes.
- **Stage 4 complete (2026-09-13).** 480 zero-shot evaluations, 266 trainings and 140 transfer evaluations; gates G4.1–G4.4 pass; tests 38/38 across stages.
  - The 14 cocycle runs on OASIS-only training views are not estimable and are left out (no AD subject with ≥ 3 training visits).
  - One CALSNIC subject with no age at its scanned visits produced NaN predictions; subjects with missing age, visit time or volume are now excluded (CALSNIC 804 scans / 302 subjects) and the affected outputs were rerun.
  - The pooled-PCA sensitivity arm listed under Part 4 was not built in this stage.
  - Frozen ADNI representations reconstruct the external cohorts 2.3–2.6× worse than ADNI test (coordinate RMSE 0.078–0.099 vs 0.034–0.038 mm). This adds about 0.025–0.034 mm to every dynamics model's external error, versus about 0.005 mm on ADNI.
- **Stage 5 scope (user decision, 2026-09-13).** All ablations A1–A4, all three sensitivity checks (age subset 65–95, min-epoch-15 across protocols, pooled PCA basis), and BrainODE-full for PCA only. Step 1 = ablations + sensitivity; step 2 = converter line; step 3 = consolidated report.
  - **A1** runs the pinned August coboundary trainers unchanged (`configs/ablation_sources.json`) on the ADNI view, PCA and SpiralNet, s42, with early stopping disabled. Their transports need each subject's first visit as context; the evaluator and condition sweep pass it.
  - **A2** BrainODE-V = BrainODE's field trained by `train_cocycle.py` with the cocycle objective. **A3** = the cocycle with the condition removed from its velocity. Both use recipes composed from the main recipes, PCA, s42–s44. **A4** = both Latent ODE variants with `condition_in_encoder/dynamics = false`, PCA, s42–s44.
  - **Pooled PCA** lives in a separate sensitivity registry so R1 is unchanged; it is added to `p3_pooled` only. Gate S5.1 (refit on ADNI train reproduces R1) is checked on the 128 code components: the first run failed only on explained-variance ratios of components 129–150 (2.2% relative), which are never used and whose directions are numerically unstable (cosine 0.978 at component 149); over the 128 code components the ratios agree within 6e-5, the subspace cosine is 0.9999 and test reconstruction agrees within 6e-7 mm.
  - **Min-epoch-15:** 14 of 42 stage-4 cocycle runs (AIBL internal and cross-fit, CALSNIC Adaptive) have no `best_min_epoch.pt`: no epoch ≥ 15 was feasible (validation shape worse than 1.05 × no-change after early overfitting on the small cohorts). They are reported as not estimable.
- **Stage 5 step 2, converter line design (2026-09-13).** Settings not given by the plan are fixed in `configs/converter_line.json` before any converter result was seen.
  - **View `p5_converter_pooled`:** ADNI strict stable subjects plus AIBL/OASIS inclusive CN-stable, AD-stable, CN→AD, MCI→AD and CN→MCI subjects (MCI-stable and reverters excluded), splits preserved. Ages and codes use the P3 train normalization (not refitted), so P3 checkpoints run on it unchanged. AD converters: train 28 (CN→AD 8, MCI→AD 20), val 5, test 5; results are descriptive.
  - **Coding:** the binary archive label is the source-visit label with MCI = 0; fixed-label comparators (C0, BrainODE-core) receive it. They are scored with `evaluate_dynamics.py --skip-pairs`, because task3's pair loader requires one label per pair.
  - **C1** = task3's DirectC4Flow fed the per-leg average dose; the objective re-implements task3's leaves with a leg condition (tested equal to task3 with fixed labels), with group-rate and disease-gap leaves on stable subjects and the P3 checkpoint's statistics. Variants: c1 (heads at 0.1× lr), c1_frozen, c1_mci (CN→MCI partial dose, PCA), c1_w050 (w = 0.5 y, PCA); PCA + Adaptive, s42–s44, initialized from the P3 cocycle of the same seed.
  - **BrainODE-full** reuses the P3 BrainODE-core field and task3's voxel cognition CNN and feedback transport; the estimator is trained on voxelized decoded PCA shapes (the domain it sees in feedback) of stable train subjects plus pseudo-cognitive mixtures of age-matched CN and AD codes with soft targets.
  - **Gate G5.6** passes (median onset error 0.47 y PCA, 0.44 y Adaptive), but the window midpoint alone gives 0.46 y: at the observed noise the visits do not localize the onset inside the observed window. Onset estimates must be reported as prior-dominated.
- **Stage 5 complete (2026-09-14).** 227 jobs (ablations 64, sensitivity 74, converter 89) without failures; stage 5 tests 24/24; the consolidated report `stage5_brainode_style/reports/brainode_style_report.md` regenerates byte-identically (G5.1, 26 files).
  - **Gates:** G5.1–G5.6 pass, G5.6 only trivially (onset recovery equals the window midpoint). **G5.7 fails** for the fine-tuned C1 variants: stable-subject test error +2.2% (PCA) and +1.1% (Adaptive) against C0, beyond the 1% margin; C1-frozen passes.
  - **Endpoints:** no method is preferred over the cocycle in any representation (AD-capture CIs are too wide with 28 AD test subjects), so methods rank by E1: residual Latent ODE and cocycle are within 0.006 mm everywhere.
  - **Ablations (PCA, ADNI test):** exact and volume coboundary cocycles capture 17–21% of AD atrophy (cocycle 84%) even when trained to completion; BrainODE's field with the cocycle loss captures 69% (BrainODE 27%); the cocycle without its disease head 65%; the unconditional residual Latent ODE 52% (conditioned 72%). The objective, not the ODE-vs-cocycle structure, carries most of the AD capture; the disease condition adds about 0.2.
  - **Sensitivity:** the pooled PCA basis lowers AIBL/OASIS error by at most 0.009 mm and leaves ADNI unchanged; min-epoch-15 selection is worse in every protocol; the 65–95 age subset leaves ADNI and AIBL skill unchanged.
  - **Converter line (5 test AD converters, descriptive):** C1 with oracle-window onsets lowers AD-converter error (PCA one-shot 0.571 → 0.547 mm; all-prior-k 0.525 → 0.496 mm) and raises MCI→AD capture; prefix-only gains are small; learned onsets stay at the window midpoint; BrainODE-full's estimator separates CN and AD shapes (test subject AUROC 0.96–0.97) but its feedback does not improve forecasts over BrainODE-core.
- **Stage 2 complete.** Gates G2.1–G2.10 pass:
  - tests 14/14;
  - the new evaluator reproduces all 10 stored anchor summaries within 2.1e-7;
  - PCA cocycle and plain-ODE retrains are identical to their anchors;
  - smoke run 40/40 jobs for all 20 cells;
  - both searches complete.
