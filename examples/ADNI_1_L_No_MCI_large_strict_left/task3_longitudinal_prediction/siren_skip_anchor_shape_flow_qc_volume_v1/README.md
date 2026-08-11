# Skip-SIREN Anchor Shape Flow

This experiment predicts future left-hippocampus shape from one optimized latent per subject: the first available scan latent only.

## Protocol

- Decoder: frozen skip SIREN `task2_representations/inr/siren_naisr_5x512_latent_skip_warmstart/checkpoints/best.pth`.
- Source input: first scan latent for each subject.
- Targets: every later scan from the same stable CN/AD subject.
- Main loss: decoded target SDF L1.
- Regularization: observed/virtual cocycle consistency, relative soft-volume change, and CN-vs-AD counterfactual volume ordering.
- Target latents are not used as a supervised loss. They are loaded for diagnostics and compatibility with the existing direct-flow code.

## Run Order

From the repository root:

```bash
EXP=examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_skip_anchor_shape_flow_qc_volume_v1
PY=/home/jakaria/anaconda3/envs/inr_sdf/bin/python
```

Build anchor-pair metadata:

```bash
$PY $EXP/scripts/build_anchor_pair_tables.py -e $EXP
```

Export skip-SIREN latents for the QC longitudinal scans. Train latents are copied from the skip-SIREN checkpoint; val/test latents are optimized with the frozen decoder:

```bash
$PY examples/ADNI_1_L_No_MCI_large_strict_left/task2_representations/scripts/fit_export_inr_latents.py \
  --config examples/ADNI_1_L_No_MCI_large_strict_left/task2_representations/configs/inr_siren_eikonal_warmstart_latent_skip.json \
  --checkpoint best \
  --splits train,val,test \
  --output-dir $EXP \
  --metadata-filter $EXP/../siren_no_skip_flow_real_sdf_observed_virtual_cocycle_qc_drop_bad_scans_min2/metadata/adni_large_strict_no_mci_left_direct_flow_records_qc_drop_bad_scans_min2.csv \
  --use-checkpoint-train-latents \
  --skip-existing \
  --device cuda:0
```

Validate the experiment contract:

```bash
$PY train_deep_sdf_longitudinal_direct_flow.py -e $EXP --validate-only
```

Train:

```bash
$PY train_deep_sdf_longitudinal_direct_flow.py -e $EXP --gpu 0
```

Evaluate SDF, long-horizon direct/composed consistency, and proxy volume trend:

```bash
$PY evaluate_deep_sdf_longitudinal_direct_flow.py -e $EXP --checkpoint best --split all --gpu 0 --composed-step-years 0.5
$PY $EXP/scripts/summarize_anchor_protocols.py --analysis $EXP/analysis/checkpoint_best
$PY $EXP/scripts/export_anchor_long_horizon_diagnostics.py -e $EXP --checkpoint best --split both --gpu 0
```

## Optional Horizon-OOD Training Variant

To train only on short horizons, set this in `specs.json` before training:

```json
"PairMaxGapYearsBySplit": {
  "train": 3.0,
  "val": 3.0
}
```

Then evaluate normally on `test`; the summary script reports `in_horizon_le_3y` vs `ood_gt_3y`.
