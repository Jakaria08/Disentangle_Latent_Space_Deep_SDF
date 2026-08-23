# task_spiral_ae_v1 — runbook

SpiralNet++ vs ICCV'23 Adaptive Spiral autoencoders on the ADNI left hippocampus, reconstruction
only, Optuna-searched, benchmarked against PCA at matched latent dimension. Latents are exported
in the PCA coefficient schema so the cocycle / Brain-ODE flow network can consume them directly.

Environment: `conda activate pytorch_geo` (nothing extra to install).
All outputs: `/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task_spiral_ae_v1/` — the root filesystem is
full, and `spiral_common.require_bulk_path` refuses to write anywhere else.

## Targets

Metric is `vertex_rmse_mm`: per-scan RMSE over **coordinates** in mm. This is the convention used
by `hippocampus_pca_cocycle_v4`; per-vertex Euclidean distance is √3 larger and is *not* what the
published table reports.

| baseline | val | test |
|---|---|---|
| PCA-128 | 0.033668 | 0.034380 |
| PCA-256 | 0.010338 | 0.010787 |

`fit_pca_baseline.py` refits PCA and **refuses to continue** unless PCA-128 val reproduces the
published 0.033667 within 1e-4. Run it first after any change to splits, caching, or the metric.

## Pipeline

```bash
cd examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task_spiral_ae_v1/scripts
PY=/home/jakaria/anaconda3/envs/pytorch_geo/bin/python

# 1. PCA baseline + reproduction gate (also builds the mesh cache). ~2 min.
$PY fit_pca_baseline.py

# 2. Wiring check for any config before spending GPU hours (2 trials x 4 epochs).
$PY optuna_search.py --exp spiralnet_z128 --gpu 0 --smoke

# 3. The four studies. Each is resumable: re-running the same command continues the
#    existing SQLite study rather than starting over.
$PY optuna_search.py --exp spiralnet_z128 --gpu 0 --n-trials 80 --trial-time-budget 900
$PY optuna_search.py --exp spiralnet_z256 --gpu 0 --n-trials 80 --trial-time-budget 900
$PY optuna_search.py --exp adaptive_z128  --gpu 1 --n-trials 60 --trial-time-budget 1200
$PY optuna_search.py --exp adaptive_z256  --gpu 2 --n-trials 60 --trial-time-budget 1200

# 4. Latents for the flow network (after the studies finish).
$PY export_latents.py --gpu 0

# 5. Final comparison table.
$PY summarize_experiments.py
```

The two SpiralNet++ studies share GPU 0 (they peak under ~3.5 GB); each adaptive study gets its
own GPU because the ICCV operator can peak near 11 GB.

## Monitoring

```bash
B=/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task_spiral_ae_v1
tail -f $B/logs/spiralnet_z128/*.log                 # live log (stdout+stderr are tee'd)
column -s, -t $B/studies/spiralnet_z128/trial_metrics.csv | less -S   # per-trial results
```

Per experiment you get: `logs/<exp>/*.log`, `studies/<exp>/study.db` (resumable),
`studies/<exp>/trial_metrics.csv`, `studies/<exp>/trials/*.pt` (one checkpoint per trial),
`best/<exp>/best_model.pt` + `best_summary.json`, `latents/<exp>/{train,val,test}_coefficients.npz`.

## Things that will bite you

- **Trial time budget.** `--trial-time-budget` caps wall-clock per trial; a trial that hits the cap
  keeps its best-so-far score and is flagged `stopped_on_time_budget` in the CSV. At 900 s these
  configs reach roughly 430–450 epochs, so a trial that sampled `epochs=500` or `600` is effectively
  truncated. Raise the budget if you want the long-epoch end of the space explored honestly.
- **Pre-latent funnel.** The stock SpiralNet++ hierarchy (`ds=[4,4,4,4]`, `oc=[32,32,32,64]`) pushes
  everything through 11×64 = 704 features — fewer than the 8238 input dimensions. Trials with
  pre-latent width below 2× the latent are pruned untrained. This is why `ds_factors` and
  `base_channels` are searched rather than fixed.
- **Adaptive operator guards.** It materialises `[B, N, max_seq, C]`, so it is only placed on levels
  with ≤ 400 nodes, with dynamic spiral length ≤ 320 (`MAX_ADAPTIVE_NODES` / `MAX_DYNAMIC_SEQ` in
  `optuna_search.py`). This mirrors the paper, which uses it on the two coarsest CoMA levels. OOM
  during a trial is caught and pruned, not fatal.
- **Spiral length clamping.** `seq_length × dilation` cannot exceed a level's vertex count (the
  coarsest level can be 11 vertices) or spiral extraction raises inside the KD-tree fallback.
  `build_spiral_stack` clamps per level; each conv reads its own length off its index tensor.
- **Adaptive port deviation.** Upstream `Dynamic_spiral_pool.__init__` ends with a bare
  `self.reset_parameters` (no call), leaving the spiral-length predictor un-zeroed. This port calls
  it, and records `adaptive_reset_parameters_fixed: true` in every checkpoint.
- **Test split** is evaluated exactly once per study, on the selected model, after the search ends.

## Expectation

PCA is unusually strong here because these meshes come from a template-deformation correspondence
pipeline, so they lie on a nearly linear manifold — 99.72% of variance at 128 components, 99.98% at
256. An untuned probe landed ~27% above PCA-128 and ~4× above PCA-256. Closing the 128 gap is
plausible; the 256 gap probably is not. `summarize_experiments.py` reports the matched-latent
verdict either way, alongside parameter counts.
