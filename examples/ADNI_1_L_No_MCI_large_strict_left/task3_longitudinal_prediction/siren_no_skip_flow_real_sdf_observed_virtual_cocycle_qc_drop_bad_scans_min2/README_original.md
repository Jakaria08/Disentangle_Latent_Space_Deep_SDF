# Large Strict Left SIREN No-Skip Direct Flow

This experiment trains only the direct longitudinal flow. The no-skip SIREN
decoder and per-scan latents from Task 2 are frozen.

The metadata was filtered to remove subjects with diagnosis progression/change.
Removed subjects are listed in `metadata/filter_summary.json`.

## Data Contract

- metadata: `metadata/adni_large_strict_no_mci_left_direct_flow_records.csv`
- latents: `latents/train_latents.npz`, `latents/val_latents.npz`,
  `latents/test_latents.npz`
- decoder checkpoint:
  `../../task2_representations/inr/siren_naisr_5x512_warmstart_no_skip/checkpoints/best.pth`

Filtered counts:

- train: 2161 scans, 537 subjects, 4362 forward pairs
- val: 267 scans, 68 subjects, 571 forward pairs
- test: 265 scans, 68 subjects, 555 forward pairs

## Commands

From repo root:

```bash
cd /home/jakaria/INR/Deep3DComp
PY=/home/jakaria/anaconda3/envs/inr_sdf/bin/python
EXP=examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_flow_real_sdf_observed_virtual_cocycle
```

Validate inputs and one CPU forward pass:

```bash
$PY train_deep_sdf_longitudinal_direct_flow.py -e "$EXP" --validate-only
```

Run a short GPU smoke test:

```bash
$PY train_deep_sdf_longitudinal_direct_flow.py -e "$EXP" --gpu 0 --smoke-batches 2
```

Train:

```bash
$PY train_deep_sdf_longitudinal_direct_flow.py -e "$EXP" --gpu 0
```

Resume:

```bash
$PY train_deep_sdf_longitudinal_direct_flow.py -e "$EXP" --gpu 0 --continue-from latest
```

Evaluate all splits after training:

```bash
$PY evaluate_deep_sdf_longitudinal_direct_flow.py -e "$EXP" --checkpoint best --split all --gpu 0
```

If `ModelParameters/best.pth` does not exist because the model never beat
no-change on validation, use `best_candidate`:

```bash
$PY evaluate_deep_sdf_longitudinal_direct_flow.py -e "$EXP" --checkpoint best_candidate --split all --gpu 0
```

Create dashboards:

```bash
$PY visualize_deep_sdf_longitudinal_direct_flow.py -e "$EXP" --checkpoint best --split val
$PY visualize_deep_sdf_longitudinal_direct_flow.py -e "$EXP" --checkpoint best --split test
```

Create the gap-bin table after evaluation:

```bash
$PY "$EXP/scripts/summarize_gap_bins.py" --analysis "$EXP/analysis/checkpoint_best"
```

