# CALSNIC left-pial cortex: Instant-NGP and Compact-SDF (`task_exact_instant_ngp_cortex_v1`)

Four controlled arms testing whether a multiresolution **hash** encoding, and a
Compact-SDF style **global/local narrow-band fusion**, close the gap to the
rank-172 oracle PCA that MR128 currently loses.

Every arm keeps **one 256-D code per subject**. Hash tables and decoders are
population-shared, so the per-subject representation size is identical to
MR64/MR128 and to a PCA coefficient vector of the same length.

## Why this task exists

MR128 (dense multiresolution grids, 13.62 M shared parameters) on the locked test split:

| | MR128 @ MC256 | MR128 @ MC512 | PCA-172 oracle |
|---|---|---|---|
| ASSD mm | 1.663 | 1.570 | **1.335** |
| HD95 mm | 7.153 | 6.527 | **4.014** |
| normal abs cos | 0.616 | 0.605 | **0.685** |

`fraction_inr_better = 0.0` on ASSD. Two hypotheses: the decoder cannot resolve
sulcal/gyral folds, and the field fragments into spurious components off-surface.
Arms A test the first; arm B tests the second.

## The measurement that set the design

Every scan shares one scaled-OBJ similarity scale, so **1 normalized unit = 117.431971 mm**
(verified across all 203 manifest rows; cohort spread 2.1e-8). On the shared
`grid_aabb`, finest feature spacing (geometric mean over axes):

| | finest level | cell size |
|---|---|---|
| MR128 dense | 128 | 1.192 mm |
| A1 Compact-SDF's published NGP scale (1.174) | 177 | 0.860 mm |
| A2/A3/B cortex-tuned (1.2944) | 767 | **0.198 mm** |

Marching cubes voxel: 256 -> 0.921 mm, 512 -> 0.460 mm. Pial sulcal gaps are
often below 1 mm, so **any claim about fold sharpness must be made at MC 512**;
MC 256 cannot resolve what these models are built to add. This is consistent with
MR128's own r256 -> r512 improvement (ASSD 1.663 -> 1.570).

`per_level_scale=1.174` is the only Instant-NGP setting Compact-SDF states; level
count, features per level and hash size are "default configuration", so 16 x 2 x 2^19
is inferred from tiny-cuda-nn defaults and is documented as inference, not fact.

## The four arms

| Run | config | scale / log2T | finest cell | grid params | model params | question |
|---|---|---|---|---|---|---|
| **A1** | `calsnic_ngp_paper_z256_exact` | 1.1740 / 19 | 0.860 mm | 7,786,162 | 7,905,971 | the published NGP baseline, as-is |
| **A2** | `calsnic_ngp_cortex_z256_exact` | 1.2944 / 19 | 0.198 mm | 10,979,478 | 11,099,287 | resolution, at MR128's capacity |
| **A3** | `calsnic_ngp_cortex_big_z256_exact` | 1.2944 / 22 | 0.198 mm | 66,088,648 | 66,208,457 | capacity, once collisions are gone |
| **B** | `calsnic_compact_ngp_z256_exact` | 1.2944 / 19 | 0.198 mm | 10,979,478 | 13,202,584 | does global/local fusion fix fragmentation |

MR128 reference: 13,623,296 grid / 13,743,105 model parameters.

A2 and A3 differ **only** in `log2_hashmap_size`. Collision load at the finest
level (estimated surface cells / table entries) is 4.88 for A2 and 0.61 for A3,
so the pair separates "needs more table" from "collisions were never binding".

Measured on one TITAN RTX (steady state, 5 warmup + 10 timed steps):

| Run | s/epoch | 2200 epochs | peak CUDA |
|---|---|---|---|
| A1 | 3.3 | 2.0 h | 0.92 GiB |
| A2 | 3.1 | 1.9 h | 0.95 GiB |
| A3 | 3.7 | 2.3 h | 1.78 GiB |
| B | 7.1 | 4.3 h | 2.48 GiB |

Plus periodic evaluation every 500 epochs (15 validation scans, latent fit + mesh decode).

## Geometry follow-up arms (C1-C3)

The first sweep answered its questions and produced a clear negative: hash
resolution and hash capacity are **not** the binding constraint. A2 was flat
across 2200 epochs (ASSD 1.770 -> 1.775) while its finest active level went from
1.58 mm to 0.198 mm, and A3 with six times the parameters matched it. What the
first sweep did establish is where the geometry actually goes wrong.

**Measured on 6 validation scans (display-independent, full-resolution meshes):**

| | area mm2 | vs target | dihedral median | roughness deg/mm | components |
|---|---|---|---|---|---|
| Target | 99,101 | 100% | 14.4 | 13.1 | 1 |
| PCA-172 oracle | 77,481 | 78% | 12.9 | 12.7 | 1 |
| MR128 dense | 56,796 | 58% | 13.0 | 20.8 | 356 |
| A2 NGP | 56,248 | 57% | 15.6 | 23.7 | 404 |
| B global_only | 53,984 | 55% | 3.3 | 7.2 | 32 |
| B fused_smooth_gate | 58,584 | 59% | 17.1 | 26.4 | 740 |

Two facts drive the C arms. First, **every model recovers only 55-59% of true
pial surface area** (PCA reaches 78%), so the folds are missing everywhere.
Second, `fused_smooth_gate` is *rougher* than real cortex (17.1 deg vs 14.4 deg)
while still carrying less area — its extra angle is noise, not folding. Roughness
and folding are different things and must not be conflated.

The finite-difference Laplacian at the surface separates them cleanly:

| branch | h = 0.5 mm | h = 1 mm | h = 2 mm | ratio |
|---|---|---|---|---|
| global | 0.393 /mm | 0.314 | 0.230 | 1.7x |
| fused | 3.539 /mm | 1.194 | 0.398 | **8.9x** |
| local | 5.083 /mm | 1.657 | 0.509 | **10x** |

Scale-stable curvature at ~0.4 /mm is a 5 mm radius, i.e. real cortex. Curvature
that collapses ~9x when the measurement window widens is sub-millimetre noise.

### What changed in the code

- **Band-limited Fourier features** (`networks/fourier_features.py`). A ReLU MLP
  on raw coordinates cannot represent folding at all, which is why the grid-free
  branch is both the cleanest surface and the flattest. Wavelengths are log
  spaced in **millimetres** with a hard floor, so the decoder gains exactly the
  scales requested and nothing finer — unlike a hash grid, whose local capacity
  is unbounded. The frequency matrix is a buffer, never trained.
- **Curvature hinge, not L2** (`second_order` config block). The 2025
  finite-difference second-order literature (arXiv:2511.08980) targets CAD, where
  driving curvature to zero yields developable patches; cortex is curved
  everywhere, so an L2 penalty would flatten the folds we are trying to recover.
  Only curvature above `curvature_threshold_per_mm` is charged. At the measured
  threshold of 1.0 /mm this charges 8% of grid-free points and 77% of fused
  points. The Laplacian reuses the six offsets the Eikonal term already computes,
  so it costs **one** extra network evaluation, not the nine a Hessian needs.
- **Ladder capped at R=273** (0.56 mm, collision load 0.62). Levels finer than
  the MC-512 voxel (0.46 mm) cannot contribute renderable detail and only alias;
  A2's top three levels ran at loads of 1.74 / 2.92 / 4.88.
- **Gate tightened** from 2.35 mm to 0.94 mm, and `far_field_agreement` 0.1 -> 0.3.
- **`sampling_balance_resolution` 128 -> 256**, so a thin sulcal gap no longer
  gets the same sample budget as a flat gyral crown.
- **Grid-free architecture** (`networks/fourier_global_sdf.py`) exposing the same
  empty-grid contract the shared trainer expects.

### The three arms

| Run | config | latent | grid params | model params | s/epoch | total | peak |
|---|---|---|---|---|---|---|---|
| **C1** Fourier-Compact | `calsnic_compact_fourier_z256_exact` | 256 | 6,785,174 | 9,072,792 | 6.8 | 4.2 h | 2.45 GiB |
| **C2** Fourier-Compact | `calsnic_compact_fourier_z512_exact` | **512** | 6,785,174 | 9,400,472 | 7.5 | 4.6 h | 2.83 GiB |
| **C3** grid-free Fourier | `calsnic_fourier_global_z256_exact` | 256 | **0** | 2,226,177 | 4.2 | 2.6 h | 1.67 GiB |

- **C1** applies every fix at unchanged latent size, so it is directly comparable to B.
- **C2** is C1 with a 512-D code. Since PCA-172 reaches only 78% of pial area,
  this tests whether the code, not the decoder, is what caps fold fidelity.
- **C3** removes the spatial grid entirely. The grid-free branch already gives the
  cleanest topology (32 components, best volume error) and cannot manufacture
  off-surface zero crossings; this asks whether Fourier features alone give it
  folds. If C3 works it is the ideal outcome — clean *and* folded, 2.2 M parameters.

Watch `laplacian_abs_mean_per_mm` in `training_history.csv`: the target is to
bring the fused field from 3.5 /mm down toward the ~0.4 /mm of real cortex
**without** `near_sdf_l1` degrading, and to raise recovered surface area.
Evaluate at MC 512 — MC 256 has a 0.92 mm voxel and cannot show sub-millimetre
folds.

### Running C1-C3 (three GPUs, ~4.6 h wall clock)

```bash
CUDA_VISIBLE_DEVICES=0 nohup $RUN $TASK/scripts/train_hashgrid_sdf.py \
  --config $TASK/configs/calsnic_compact_fourier_z512_exact.json --device cuda:0 \
  > /mnt/bulk10tb/c2_fourier_z512.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup $RUN $TASK/scripts/train_hashgrid_sdf.py \
  --config $TASK/configs/calsnic_compact_fourier_z256_exact.json --device cuda:0 \
  > /mnt/bulk10tb/c1_fourier_z256.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup $RUN $TASK/scripts/train_hashgrid_sdf.py \
  --config $TASK/configs/calsnic_fourier_global_z256_exact.json --device cuda:0 \
  > /mnt/bulk10tb/c3_fourier_global.log 2>&1 &
```

---

## Deliberate deviations from Compact-SDF

| Paper (arXiv:2511.14539) | Here | Reason |
|---|---|---|
| NGP `per_level_scale` 1.174 | kept as A1; A2/A3/B use 1.2944 | 1.174 tops out at 0.86 mm, still coarser than sulcal gaps |
| dense 128^3 x 128-dim grid (~1 GiB) | hash tables, <= 252 MiB | 128^3 resolves only ~1.5 mm here; dense at 767^3 is impossible |
| dense 512^3 near-surface resampling | existing cell-balanced sampler over the audited archives | ~393 k exact samples/scan already on disk |
| MSE reconstruction loss | clamped L1 (clamp 0.1) | matches MR64/MR128 so the comparison is controlled |
| no Eikonal | Eikonal on, millimetre-pinned epsilon | MR128 already fragments; hash grids worsen off-surface zero crossings |
| hard narrow-band replacement only | also a continuous `smooth_gate` | the paper notes bandwidth can leave a discontinuity |

## What is reused unchanged

Nothing about the data or the metrics is re-implemented:

- exact SDF archives, manifest, audit and the locked 173/15/15 split, read in place
  from `control_L_exact_multires_v1/`;
- the pinned sampler `shared_grid_common.py` (sha256 checked at config load);
- `calsnic_common.py` for the manifest and the SDF-to-millimetre transform;
- `periodic_evaluate_multires.py` for `surface_metrics`, the 12-metric `METRICS`
  tuple, `MatchedPCA` (rank 172, reused, never refitted) and `cluster_bootstrap`;
- the MR128 optimizer groups, gradient accumulation, latent norm-ball projection,
  checkpoint payload and selection policy.

`check_pipeline.py` asserts every shared hyperparameter still matches MR128, so the
encoder stays the only difference.

## Output schema

`per_scan_metrics.csv` carries the MR128 columns plus a `variant` column. `method`
is `inr` for the configured selection variant and `inr_<variant>` for the others,
so `compare_evaluations.py` and the existing notebook work unmodified while every
variant stays in the CSV. A two-branch checkpoint is decoded four ways from the
**same** fitted latent -- `global_only`, `local_only`, `fused_hard_band`,
`fused_smooth_gate` -- so the fusion ablation costs no extra training.

## Commands

All commands run from the repository root, through `run_on_bulk.sh`.

```bash
REPO=/home/jakaria/INR/Deep3DComp
TASK=$REPO/examples/CALSNIC_control_L/task_exact_instant_ngp_cortex_v1
RUN="bash $TASK/scripts/run_on_bulk.sh"
cd $REPO
```

### 0. Static and CPU checks (no GPU, writes nothing but one report)

```bash
$RUN $TASK/scripts/check_pipeline.py --require-generated-data
/home/jakaria/anaconda3/envs/inr_sdf/bin/python -m pytest $TASK/tests/ -q
for c in calsnic_ngp_paper calsnic_ngp_cortex calsnic_ngp_cortex_big calsnic_compact_ngp; do
  $RUN $TASK/scripts/validate_implementation.py --config $TASK/configs/${c}_z256_exact.json --device cpu
done
```

### 1. Train (three GPUs; B is the long pole, start it first)

```bash
CUDA_VISIBLE_DEVICES=0 nohup $RUN $TASK/scripts/train_hashgrid_sdf.py \
  --config $TASK/configs/calsnic_compact_ngp_z256_exact.json --device cuda:0 \
  > $TASK/../compact_ngp.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup $RUN $TASK/scripts/train_hashgrid_sdf.py \
  --config $TASK/configs/calsnic_ngp_cortex_z256_exact.json --device cuda:0 \
  > $TASK/../ngp_cortex.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup $RUN $TASK/scripts/train_hashgrid_sdf.py \
  --config $TASK/configs/calsnic_ngp_cortex_big_z256_exact.json --device cuda:0 \
  > $TASK/../ngp_cortex_big.log 2>&1 &
```

Then A1 on whichever GPU frees first:

```bash
CUDA_VISIBLE_DEVICES=1 $RUN $TASK/scripts/train_hashgrid_sdf.py \
  --config $TASK/configs/calsnic_ngp_paper_z256_exact.json --device cuda:0
```

Add `--resume latest` to continue an interrupted run. Periodic validation
evaluation runs automatically at epochs 500/1000/1500/2000/2200 and selects
`best_mesh` on validation ASSD.

**Check at epoch 500 before committing to the full run:**

```bash
BULK=/mnt/bulk10tb/Deep3DComp/CALSNIC/control_L_exact_instant_ngp_v1/runs
column -s, -t $BULK/calsnic_ngp_cortex_z256_exact/logs/training_history.csv | tail -5
python3 -c "import json;d=json.load(open('$BULK/calsnic_ngp_cortex_z256_exact/periodic_evaluation/epoch_0500/summary.json'));print(d['selection_metric'])"
```

Watch `gradient_norm_p95` (Eikonal health, should approach 1.0) and
`near_sdf_l1`. If `gradient_norm_p95` is far from 1 or `near_sdf_l1` stalls,
lower `eikonal.weight` before running the remaining arms.

### 2. Compare on validation (never test)

```bash
BULK=/mnt/bulk10tb/Deep3DComp/CALSNIC/control_L_exact_instant_ngp_v1
MR=/mnt/bulk10tb/Deep3DComp/CALSNIC/control_L_exact_multires_v1
$RUN $TASK/scripts/../../task_exact_multires_cortex_v1/scripts/compare_evaluations.py \
  --evaluation mr128=$MR/runs/calsnic_control_L_multires128_z256_exact/periodic_evaluation/epoch_2200 \
  --evaluation ngp_paper=$BULK/runs/calsnic_ngp_paper_z256_exact/periodic_evaluation/epoch_2200 \
  --evaluation ngp_cortex=$BULK/runs/calsnic_ngp_cortex_z256_exact/periodic_evaluation/epoch_2200 \
  --evaluation ngp_cortex_big=$BULK/runs/calsnic_ngp_cortex_big_z256_exact/periodic_evaluation/epoch_2200 \
  --evaluation compact_ngp=$BULK/runs/calsnic_compact_ngp_z256_exact/periodic_evaluation/epoch_2200 \
  --split val --output-dir $BULK/comparisons/hashgrid_validation
```

### 3. Full validation evaluation at MC 512, all variants

MC 256 cannot resolve the folds these models add; run the winner at 512.

```bash
CUDA_VISIBLE_DEVICES=0 $RUN $TASK/scripts/evaluate_hashgrid_variants.py \
  --config $TASK/configs/calsnic_compact_ngp_z256_exact.json \
  --checkpoint best_mesh --device cuda:0 --splits val --resolution 512 \
  --variants global_only local_only fused_hard_band fused_smooth_gate \
  --output-dir $BULK/manual_evaluation/compact_ngp_best_mesh_r512
```

### 4. Export latents and meshes (validation)

```bash
CUDA_VISIBLE_DEVICES=0 $RUN $TASK/scripts/fit_export_hashgrid_latents.py \
  --config $TASK/configs/calsnic_ngp_cortex_z256_exact.json \
  --checkpoint best_mesh --device cuda:0 --splits train val

CUDA_VISIBLE_DEVICES=0 $RUN $TASK/scripts/reconstruct_hashgrid_meshes.py \
  --config $TASK/configs/calsnic_ngp_cortex_z256_exact.json \
  --checkpoint best_mesh --device cuda:0 --splits val --resolution 512 \
  --latents $BULK/runs/calsnic_ngp_cortex_z256_exact/latent_exports/best_mesh/latents.pth
```

### 5. Locked test -- only after the architecture and checkpoint are chosen on validation

```bash
CUDA_VISIBLE_DEVICES=0 $RUN $TASK/scripts/evaluate_hashgrid_variants.py \
  --config $TASK/configs/<selected>.json --checkpoint best_mesh --device cuda:0 \
  --splits test --resolution 512 --confirm-test \
  --output-dir $BULK/final_test/<selected>_best_mesh_r512
```

`--confirm-test` is required by `evaluate_hashgrid_variants.py`,
`fit_export_hashgrid_latents.py` and `reconstruct_hashgrid_meshes.py`.
`periodic_evaluation.splits` is pinned to `["val"]` and `check_pipeline.py`
asserts it, so training can never touch the test split.

## Storage

Everything persistent lives under
`/mnt/bulk10tb/Deep3DComp/CALSNIC/control_L_exact_instant_ngp_v1/`
(9.1 T free). The manifest, `sdf_exact/`, audits and the PCA basis are read in
place from `control_L_exact_multires_v1/`; nothing is copied. Checkpoints are
roughly 32 MiB (A1), 45 MiB (A2), 265 MiB (A3) and 54 MiB (B); no dense SDF
volume is ever written to disk.
