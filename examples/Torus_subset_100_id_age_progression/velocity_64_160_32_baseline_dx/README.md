# Baseline-Aligned 64/160/32 Experiment

Run on GPU 0:

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python \
  train_deep_sdf_longitudinal_flow_64_160_32_baseline_dx.py \
  -e examples/Torus_subset_100_id_age_progression/velocity_64_160_32_baseline_dx \
  --gpu 0
```

This experiment reuses `networks/longitudinal_flow_64_160_32_pred_dx.py`.

The controlled changes from Experiment 1 are:

- subject anchors are initialized from the earliest pretrained scan latent;
- only baseline reconstruction and code regularization update subject anchors;
- future reconstruction and all temporal/disentanglement losses use detached anchors;
- disease gating uses oracle labels through epoch 100, linearly transitions to
  predicted soft probabilities by epoch 300, and then remains label-free;
- diagnosis targets remain the true labels even when the transition gate is soft.

The current training-time evaluation uses the primary one-shot, composed,
predicted-soft protocol. The visualization notebook reads
`EvalObservedTimepointsList`, `EvalRolloutModes`, and
`EvalDiseaseConditionModes` to run the complete one-/two-/three-shot analysis.

Backward-compatible settings are:

```json
"SubjectAnchorTrainingMode": "all_scans",
"PretrainedSubjectAnchorInitMode": "mean_all_scans",
"TrainDiseaseConditionMode": "oracle"
```

Those settings restore Experiment 1's anchor and training-gate behavior.

Useful controlled ablations require only spec changes:

| Purpose | `SubjectAnchorTrainingMode` | `PretrainedSubjectAnchorInitMode` | `TrainDiseaseConditionMode` |
| --- | --- | --- | --- |
| Experiment 1 behavior | `all_scans` | `mean_all_scans` | `oracle` |
| Isolate baseline-aligned anchors | `baseline_only` | `baseline_scan` | `oracle` |
| Isolate predicted training gate | `all_scans` | `mean_all_scans` | `scheduled_predicted_soft` |
| Full Experiment 2 | `baseline_only` | `baseline_scan` | `scheduled_predicted_soft` |

Do not set `EvalDiseaseConditionMode` to `oracle` for the primary test result.
Oracle evaluation is only an upper-bound ablation because it uses the held-out
diagnosis label.
