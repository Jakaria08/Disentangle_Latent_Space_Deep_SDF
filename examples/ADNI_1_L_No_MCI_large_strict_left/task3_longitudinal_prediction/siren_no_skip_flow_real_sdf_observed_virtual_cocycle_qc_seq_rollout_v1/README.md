# QC Sequence-Rollout Direct Flow v1

This experiment starts from the QC-clean no-progression dataset in
`siren_no_skip_flow_real_sdf_observed_virtual_cocycle_qc_drop_bad_scans_min2`.
The SIREN shape decoder is still frozen. Only the longitudinal direct-flow
model is trained.

The purpose is to move closer to BrainODE-style longitudinal behavior while
keeping the cocycle/direct-flow formulation instead of using an ODE. The
important change is that training now includes real multi-scan sequence rollout
supervision and sequence consistency, so composed prediction is part of the
training objective rather than only a visualization/evaluation mode.

## Data

The metadata and latent files are copied from the QC-clean experiment:

- no MCI scans in this training experiment
- no subjects with diagnosis progression/change
- bad longitudinal mesh scans removed by the previous QC pass
- subjects kept only if at least two scans remain after QC

Counts copied from the QC-clean source:

```json
{
  "train": {
    "scans": 1791,
    "subjects": 421,
    "pairs": 3680,
    "pairs_with_observed_intermediate": 2310,
    "diagnosis_subjects": {
      "CN": 261,
      "AD": 160
    }
  },
  "val": {
    "scans": 224,
    "subjects": 52,
    "pairs": 522,
    "pairs_with_observed_intermediate": 350,
    "diagnosis_subjects": {
      "CN": 28,
      "AD": 24
    }
  },
  "test": {
    "scans": 214,
    "subjects": 50,
    "pairs": 483,
    "pairs_with_observed_intermediate": 319,
    "diagnosis_subjects": {
      "CN": 29,
      "AD": 21
    }
  }
}
```

## Objective Changes

Enabled beyond the previous pair SDF plus observed/virtual cocycle losses:

- sequence rollout SDF loss
- sequence direct-vs-rollout cocycle loss
- latent displacement direction loss
- latent displacement magnitude loss
- sequence displacement magnitude loss
- latent manifold guard
- latent speed guard
- gap-weighted auxiliary/sequence losses
- sequence-aware validation checkpoint selection

The first acceptance target is not just lower SDF error. The selected model
should reproduce the AD > CN atrophy trend in train, val, and test, and composed
long-horizon prediction should be smoother and at least as stable as direct
prediction.

## Commands

From repo root, validate the contract first:

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python train_deep_sdf_longitudinal_direct_flow.py -e /home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_flow_real_sdf_observed_virtual_cocycle_qc_seq_rollout_v1 --validate-only
```

Optional small GPU smoke run:

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python train_deep_sdf_longitudinal_direct_flow.py -e /home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_flow_real_sdf_observed_virtual_cocycle_qc_seq_rollout_v1 --gpu 0 --smoke-batches 2
```

Full training:

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python train_deep_sdf_longitudinal_direct_flow.py -e /home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_flow_real_sdf_observed_virtual_cocycle_qc_seq_rollout_v1 --gpu 0
```

After training, evaluate the best checkpoint:

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python evaluate_deep_sdf_longitudinal_direct_flow.py -e /home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_flow_real_sdf_observed_virtual_cocycle_qc_seq_rollout_v1 --checkpoint best --split all --gpu 0 --composed-step-years 0.5
```

Summarize gap bins:

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python /home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_flow_real_sdf_observed_virtual_cocycle_qc_seq_rollout_v1/scripts/summarize_gap_bins.py --analysis /home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_flow_real_sdf_observed_virtual_cocycle_qc_seq_rollout_v1/analysis/checkpoint_best
```

Generate rich HTML analysis:

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python /home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_flow_real_sdf_observed_virtual_cocycle_qc_seq_rollout_v1/scripts/run_rich_visualization_analysis.py --checkpoint best --gpu 0 --composed-step-years 0.5
```

If `best.pth` is not created, that means the model did not beat both pair
no-change and sequence no-change. In that case, evaluate `best_candidate`
instead for diagnostics, but do not treat it as a successful model.
