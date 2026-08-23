# CALSNIC left-cortex SPHARM vs PCA reconstruction-error comparison

This task answers one question for the CALSNIC left-pial cohort: how does a spherical-harmonic
(SPHARM) shape encoding compare to the existing matched PCA basis as a compact, invertible
reconstruction of the surface? It is scoped to that comparison only — it does not feed SPHARM into
the cocycle-flow longitudinal-prediction network. It reuses the locked manifest, exact SDF archives,
and PCA basis already built by `task_exact_multires_cortex_v1` rather than rebuilding any of them.

All generated embedding/fit/evaluation artifacts are restricted to the same root the multires task
uses:

`/mnt/bulk10tb/Deep3DComp/CALSNIC/control_L_exact_multires_v1`

(new subdirectories `spharm/` and `comparisons/spharm_*` under it). Nothing here writes to the
locked manifest, exact archives, or PCA basis.

## Why SPHARM needed almost no new machinery here

SPHARM's hard part is normally per-subject: parameterizing each subject's own folded, concave
surface onto a sphere without self-intersection. That step does not apply here, because the 203
CALSNIC left-pial meshes are already vertex-corresponded — same 40962 vertices, same 81920 faces,
for every subject (verified cohort-wide by `check_shared_topology.py`, not just assumed; the sibling
`fit_matched_pca.py` already depends on this). Vertex index `i` always names the same anatomical
location, so the mesh-to-sphere map only has to be solved **once**, on the shared connectivity, and
reused unchanged for all 203 subjects. Per-subject "encoding" then reduces to an ordinary linear
least-squares fit against a fixed real-spherical-harmonic basis; reconstruction is evaluating the
truncated series back at the same fixed angles. See `scripts/spharm_embedding.py`'s module docstring
for the full derivation, including why the initially-planned Tutte+stereographic recipe was replaced:
that produced an 8-orders-of-magnitude area-distortion ratio (a well-known "small boundary, huge
interior" pathology), while pinning the mesh's 12 valence-5 vertices — which the connectivity itself
proves are combinatorially the icosahedron's 12 vertices — at exact regular-icosahedron coordinates
and solving one direct linear system for everyone else gives a valid (zero flipped triangles), well
conditioned (37x area-distortion ratio) embedding with no iterative relaxation needed.

No new heavy dependency: the embedding uses `scipy.sparse`/`scipy.sparse.csgraph`, and the harmonic
basis uses `scipy.special.sph_harm`, both already in the `inr_sdf` environment. No SPHARM-PDM,
Slicer, or pyshtools.

Unlike PCA, the SPHARM basis is geometric (derived from connectivity, not from any subject's data),
so **every split can be fit/reconstructed with no train/test leakage concern** — there is nothing to
leak. `fit_spharm_basis.py` therefore fits train, val, and test alike. The full 12-metric
`evaluate_spharm_vs_pca.py` still gates any `test`-split run behind `--confirm-test`, matching this
task family's discipline, even though the underlying reason differs from PCA's.

## Commands

Run from `/home/jakaria/INR/Deep3DComp`.

```bash
TASK=examples/CALSNIC_control_L/task_exact_spharm_cortex_v1
RUN="$TASK/scripts/run_on_bulk.sh"
ROOT=/mnt/bulk10tb/Deep3DComp/CALSNIC/control_L_exact_multires_v1
```

### 0. Confirm the shared-topology assumption over the full cohort

```bash
bash "$RUN" "$TASK/scripts/check_shared_topology.py"
```

### 1. Validate the embedding and the fit/reconstruct round trip

```bash
bash "$RUN" "$TASK/scripts/validate_spharm_embedding.py"
```

No-write numerical gate: 12 anchors found and structurally verified, zero flipped triangles, bounded
area distortion, a synthetic round trip (exact recovery at and above the true degree, real residual
below it), and a real-data check (PCA mean-shape reconstruction RMSE decreases monotonically with
degree).

### 2. Fit the SPHARM degree sweep

```bash
bash "$RUN" "$TASK/scripts/fit_spharm_basis.py"
```

Builds the spherical embedding once (cached under `$ROOT/spharm/anchored_harmonic_all203/`), then
fits/reconstructs every one of the 203 scans (all splits) at each swept degree (default `4 8 12 16
20 24 32`, i.e. 75 to 3267 total scalar coefficients per scan — bracketing PCA's rank-172 ceiling
from both sides). Writes `reconstruction_summary.csv` / `reconstruction_per_scan.csv` in the same
schema as `fit_matched_pca.py`'s, so the two curves overlay directly.

### 3. Full 12-metric evaluation against PCA on validation

```bash
bash "$RUN" "$TASK/scripts/evaluate_spharm_vs_pca.py" --degree 16 --splits val
bash "$RUN" "$TASK/scripts/evaluate_spharm_vs_pca.py" --degree 32 --splits val
```

Reuses the metric stack unmodified from the sibling multires task's `periodic_evaluate_multires.py`
(`METRICS`, `surface_metrics`, `MatchedPCA`, `cluster_bootstrap`). Add `--inr-evaluation-dir
"$ROOT/runs/<config>/manual_evaluation/<checkpoint>"` to include an already-exported INR mesh leg.

### 4. One locked test evaluation

Only after picking one degree from validation:

```bash
bash "$RUN" "$TASK/scripts/evaluate_spharm_vs_pca.py" --degree <chosen> --splits test --confirm-test \
  --output-dir "$ROOT/final_test/spharm_degree<chosen>_vs_pca172"
```

## Results so far (this repo, this cohort, validation split, n=15)

`fit_spharm_basis.py`'s corresponded-vertex RMSE (all splits nearly identical — confirming SPHARM
has no train/val gap):

| degree | coefficients | train RMSE (mm) | val RMSE (mm) | test RMSE (mm) |
|---|---|---|---|---|
| 4  | 75   | 8.50 | 8.39 | 8.47 |
| 8  | 243  | 6.02 | 5.97 | 5.98 |
| 16 | 867  | 3.67 | 3.63 | 3.65 |
| 32 | 3267 | 1.89 | 1.86 | 1.88 |

PCA (`task_exact_multires_cortex_v1`'s matched rank-172 basis), for reference: val RMSE plateaus at
**3.86 mm** at rank 172 (its hard ceiling — rank cannot exceed `train_scans - 1 = 172`; requesting
256/512 components gives the identical rank-172 reconstruction).

So by raw corresponded-vertex RMSE, SPHARM only needs ~degree 16 (867 coefficients) to match PCA's
172-coefficient ceiling, and keeps improving well past it since it is not capped by training-cohort
size. But the full 12-metric surface evaluation (`evaluate_spharm_vs_pca.py`) tells a more specific
story — **the two rankings disagree with each other**:

| metric (val mean) | SPHARM deg 16 (867 coef) | SPHARM deg 32 (3267 coef) | PCA rank 172 |
|---|---|---|---|
| ASSD (mm) | 1.88 (worse) | **0.96** (better) | 1.37 |
| Chamfer L1 (mm) | 3.76 (worse) | **1.92** (better) | 2.74 |
| HD95 (mm) | 5.80 (worse) | **2.98** (better) | 4.10 |
| F-score@1mm | 0.34 (worse) | **0.62** (better) | 0.48 |
| high-curvature ASSD (mm) | 3.10 (worse) | **1.68** (better) | 1.84 |
| fraction of scans SPHARM wins | 0/15 | 14/15 | — |

At matched raw-RMSE (degree 16), SPHARM loses on *every* surface-based metric despite the RMSE tie —
consistent with spherical-harmonic truncation ringing (a Gibbs-like artifact) concentrated at high
curvature, exactly where `high_curvature_gt_to_prediction_mm` shows the largest gap. Only once degree
is pushed well past PCA's ceiling (degree 32, ~19x PCA's coefficient budget) does SPHARM decisively
overtake PCA on every metric. **Report both a matched-coefficient-count comparison and a
best-achievable one — they give opposite answers**, and corresponded-vertex RMSE alone is not a
reliable proxy for surface-metric quality at a given degree.

## SPHARM-then-PCA hybrid: tested as a regularizer, did not help

Plain PCA overfits badly at its rank-172 ceiling: train RMSE is 0.00003mm (essentially exact — rank
172 = 173 train scans − 1, a complete basis for the training set) vs val RMSE 3.86mm, a five-order-
of-magnitude gap, from only 173 training scans against 122,886 raw dimensions. The hypothesis: fit a
*moderately* truncated SPHARM basis first (a spectral low-pass filter, applied identically and
independently to every scan — no leakage) and run train-only PCA on the resulting coefficients
instead of on raw vertices, so PCA never sees the fine per-subject detail it has no ability to
generalize from anyway.

```bash
bash "$RUN" "$TASK/scripts/fit_spharm_pca_hybrid.py"
```

Sweeps smoothing degree `{16, 24, 32, 48, 64}` × PCA rank `{16, 32, 64, 100, 128, 172}` (30 cells),
train-only PCA fit on coefficients mirroring `fit_matched_pca.py`'s eigh/dual-PCA math exactly, val/
test projected onto the train basis. **Result: it does not help, at any cell in the grid.**
Val RMSE improves monotonically toward — but never past — plain PCA's 3.86mm as smoothing weakens
(higher degree) and rank increases: the best cell (degree 64, rank 172) gives 3.89mm, i.e. still
slightly worse than plain PCA, and every more-aggressive (lower degree or lower rank) cell is
substantially worse (degree 16/rank 16: 5.45mm). The full 12-metric evaluation at that best cell
confirms the same ordering on real surface metrics, not just RMSE:

```bash
bash "$RUN" "$TASK/scripts/evaluate_spharm_vs_pca.py" --degree 32 --splits val --hybrid-degree 64 --hybrid-rank 172
```

| metric (val mean) | hybrid (deg 64, rank 172) | PCA rank 172 | plain SPHARM deg 32 |
|---|---|---|---|
| ASSD (mm) | 1.41 | 1.37 | **0.96** |
| Chamfer L1 (mm) | 2.81 | 2.74 | **1.92** |
| HD95 (mm) | 4.24 | 4.10 | **2.98** |
| F-score@1mm | 0.46 | 0.48 | **0.62** |
| high-curvature ASSD (mm) | 1.95 | 1.84 | **1.68** |

This is the expected outcome in hindsight, not a surprise once measured: SPHARM-encode and PCA are
both linear, so pre-filtering can only remove information before PCA ever sees it — it cannot improve
PCA's fit on the data it's given, and the val-side generalization gap the hybrid was meant to close
turned out not to be closable this way. Plain higher-degree SPHARM alone (no PCA at all, hence no
rank-172/173-scan ceiling) remains the best of the three on every metric measured. **Do not use the
hybrid basis for anything downstream** — this section exists so the negative result doesn't get
re-tested from scratch later.

## Storage and test safety

Meshes are exported as PLY only (no dense SDF volumes retained). `run_on_bulk.sh` moves temporary
files and caches off the NVMe. Test-split evaluation requires `--confirm-test`; `fit_spharm_basis.py`
itself has no such gate because it performs no model/degree selection, only a diagnostic per-degree
RMSE sweep, matching how `fit_matched_pca.py`'s own `reconstruction_summary.csv` is already produced
without one.
