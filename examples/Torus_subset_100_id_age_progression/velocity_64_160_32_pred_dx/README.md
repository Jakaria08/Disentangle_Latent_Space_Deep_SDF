# 64/160/32 Predicted-Diagnosis Experiment

Train with:

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python \
  train_deep_sdf_longitudinal_flow_64_160_32_pred_dx.py \
  -e examples/Torus_subset_100_id_age_progression/velocity_64_160_32_pred_dx
```

The experiment keeps the previous reconstruction, pair, and cocycle objectives. Its
changes are configured in `specs.json`:

- velocity blocks: `64/160/32`;
- stronger existing residual magnitude, covariance, and adversarial losses;
- anchor diagnosis classifier trained from detached all-visit training anchors;
- label-free test gating with `EvalDiseaseConditionMode=predicted_soft`.

Alternative test modes are `predicted_hard` and `oracle`. Test diagnosis labels are
used for metrics in predicted modes, not for inference.

Diagnosis artifacts are written to `DiagnosisPredictions/`. Analyze the checkpoint
with:

```text
visualize_torus_velocity_64_160_32_pred_dx.ipynb
```

The training and network scripts remain modular. Setting the block dimensions back
to `64/128/64`, disabling `UseAnchorDiseaseClassification`, and selecting
`EvalDiseaseConditionMode=oracle` reproduces the previous experiment behavior.
