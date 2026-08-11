# v4: v1-anchored AD speed calibration

## Purpose

This is an isolated follow-up to the completed v1 direct PCA cocycle flow and
the registered-mesh v3 experiment. It does **not** use an ODE. The v1 flow is
frozen and supplies a base transport

`z_v1 = Phi_v1(z_source, age_source, age_target, diagnosis)`.

The new model predicts

`z_pred = z_source + s * (z_v1 - z_source)`.

For CN, `s = 1` exactly. For AD, `s` is a positive bounded speed made from a
global AD speed and a small feature-conditioned residual. This retains the
strong v1 spatial correspondence while targeting the observed AD
under-progression in volume trajectories.

The primary configuration starts at `s=2.5`, supported by the earlier
validation scaling diagnosis. The original v1 state is always saved separately
as `epoch_0000_v1_identity.pth`; it is never overwritten.

## Inputs

- Frozen v1 flow checkpoint: `pca150_direct_cocycle_flow_qc_v1/checkpoints/best_val_endpoint_vertex_mae.pth`
- QC-stable BrainODE split archives used only for the shared train/validation/test partition
- Existing PCA-150 representation and registered mesh faces/normals from v3
- BrainODE best checkpoint only for fair, post-training comparison

`prepare_v1_speed_metadata.py` creates all normalization and loss scaling from
the **training split only**. It also records the v1 checkpoint hash and v1
validation reference errors. Neither validation nor test targets enter those
statistics.

## Training safeguards

- Frozen v1 parameters are checked after every epoch.
- CN is a hard gate to the unmodified v1 output, not a soft penalty.
- `best_feasible_volume.pth` must remain within the configured v1 tolerance on
  all validation pairs **and** first-to-last validation pairs.
- `best_endpoint.pth` emphasizes endpoint accuracy; `best_ad_volume.pth`
  emphasizes AD volume/slope fit.
- `epoch_0000_configured_speed.pth` is the no-training, fixed-2.5 baseline;
  `epoch_0000_v1_identity.pth` is the exact v1 fallback.

## Full-run commands

Run from the repository root. `cuda:0` can be replaced with the GPU you want
to use.

```bash
PY=/home/jakaria/anaconda3/envs/inr_sdf/bin/python
V4=examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/pca150_v1_anchored_ad_speed_cocycle_qc_v4

$PY $V4/scripts/prepare_v1_speed_metadata.py \
  --config $V4/configs/v1_anchored_ad_speed_primary.json \
  --device cuda:0

$PY $V4/scripts/sweep_fixed_ad_speed.py \
  --config $V4/configs/v1_anchored_ad_speed_primary.json \
  --metadata-dir $V4/metadata \
  --device cuda:0

$PY $V4/scripts/train_v1_anchored_ad_speed.py \
  --config $V4/configs/v1_anchored_ad_speed_primary.json \
  --metadata-dir $V4/metadata \
  --run-name v1_anchor_ad_speed_seed42 \
  --seed 42 --device cuda:0
```

Run seeds 43 and 44 with only `--run-name` and `--seed` changed. Do not use the
test split to pick a seed or a checkpoint.

## Validation selection

Evaluate these four validation checkpoints for each seed:

`epoch_0000_v1_identity`, `epoch_0000_configured_speed`, `best_endpoint`, and
`best_feasible_volume`.

```bash
$PY $V4/scripts/evaluate_v1_anchored_ad_speed.py \
  --config $V4/configs/v1_anchored_ad_speed_primary.json \
  --metadata-dir $V4/metadata \
  --run-name v1_anchor_ad_speed_seed42 \
  --checkpoint best_feasible_volume \
  --splits val --device cuda:0 \
  --transport-methods direct composed_observed \
  --include-backward-eval --include-cycle-eval --include-sequence-eval
```

Use the validation reports to select one checkpoint with: (1) no material
endpoint regression versus v1, (2) a better AD volume-relative/rate/slope
result, and (3) stable recursive first-visit trajectories. `composed_observed`
is a cocycle diagnostic that deliberately uses observed intermediate shapes;
`sequence_recursive_from_first` is the no-future-shape trajectory result to
report as prospective forecasting.

After choosing the checkpoint on validation, run it once on test:

```bash
$PY $V4/scripts/evaluate_v1_anchored_ad_speed.py \
  --config $V4/configs/v1_anchored_ad_speed_primary.json \
  --metadata-dir $V4/metadata \
  --run-name v1_anchor_ad_speed_seed42 \
  --checkpoint best_feasible_volume \
  --splits test --device cuda:0 \
  --transport-methods direct composed_observed \
  --include-backward-eval --include-cycle-eval --include-sequence-eval

$PY $V4/scripts/evaluate_unified_baselines.py \
  --config $V4/configs/v1_anchored_ad_speed_primary.json \
  --metadata-dir $V4/metadata \
  --run-name v1_anchor_ad_speed_seed42 \
  --checkpoint best_feasible_volume \
  --splits val test --device cuda:0
```

## Main outputs

For a run named `v1_anchor_ad_speed_seed42`, results are under
`runs/v1_anchor_ad_speed_seed42/`.

- `checkpoints/`: exact v1 fallback, fixed-speed initial state, and selected training checkpoints
- `history.json`: every training/validation loss and constrained-checkpoint status
- `analysis/checkpoint_*/registered_flow_per_pair.csv`: matched registered-mesh endpoint, volume, and local-change metrics
- `volume_trends.csv` and `volume_slope_summary.csv`: direct first-visit volume trend
- `sequence_trajectory_*.csv`: direct and recursive first-visit forecasts; the recursive branch does not use observed intermediate shapes
- `cycle_consistency*.csv`: forward/backward consistency
- `unified_baselines/`: v1, v4, and BrainODE predicted meshes scored with the same mesh/volume/local-change evaluator on identical pairs

The paper-facing comparison should use `unified_baselines/unified_comparison_summary.csv`: it avoids comparing v4 mesh-volume metrics against BrainODE metrics computed by a different evaluator.
