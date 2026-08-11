# SIREN Latent ODE QC Baseline

This experiment tests whether a continuous ODE transport model beats the existing
SIREN latent flow models when representation and data are fixed.

Only the transport model changes:

```text
dz / dt = f(z, t, diagnosis)
```

The frozen decoder, 256D SIREN latents, QC-clean metadata, train/val/test split,
SDF target samples, and CN/AD condition are shared with:

`../siren_no_skip_flow_real_sdf_observed_virtual_cocycle_qc_drop_bad_scans_min2`

## Model

- `FlowType`: `latent_ode`
- Latent dimension: `256`
- Vector field: MLP `[256, 256]`
- Condition: fixed `label_ad`, where `0 = CN` and `1 = AD`
- Integration: RK4 with `ODEIntegrationSubsteps = 4`
- No cognition/disease-conversion module is used.

## Objectives

- Real target SDF L1
- Observed-intermediate latent cocycle consistency
- Virtual-intermediate latent cocycle consistency

The objective family matches the direct SIREN flow baseline so the comparison is
transport model versus transport model, not representation versus representation.

## Commands

Validate the data/model contract:

```bash
python train_deep_sdf_longitudinal_direct_flow.py -e examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_latent_ode_qc_drop_bad_scans_min2_v1 --validate-only
```

Short smoke train:

```bash
python train_deep_sdf_longitudinal_direct_flow.py -e examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_latent_ode_qc_drop_bad_scans_min2_v1 --gpu 0 --smoke-batches 2
```

Full training:

```bash
python train_deep_sdf_longitudinal_direct_flow.py -e examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_latent_ode_qc_drop_bad_scans_min2_v1 --gpu 0
```

Numerical evaluation:

```bash
python evaluate_deep_sdf_longitudinal_direct_flow.py -e examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_latent_ode_qc_drop_bad_scans_min2_v1 --checkpoint best --split all --gpu 0
```

If `best.pth` is not created, evaluate `best_candidate`:

```bash
python evaluate_deep_sdf_longitudinal_direct_flow.py -e examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_latent_ode_qc_drop_bad_scans_min2_v1 --checkpoint best_candidate --split all --gpu 0
```

Gap-bin summary:

```bash
python examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_flow_real_sdf_observed_virtual_cocycle_qc_drop_bad_scans_min2/scripts/summarize_gap_bins.py --analysis examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_latent_ode_qc_drop_bad_scans_min2_v1/analysis/checkpoint_best
```

Rich visualization:

```bash
python examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_flow_real_sdf_observed_virtual_cocycle_qc_drop_bad_scans_min2/scripts/run_rich_visualization_analysis.py --experiment examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_latent_ode_qc_drop_bad_scans_min2_v1 --checkpoint best --gpu 0
```

Fair real-mesh forecast comparison against BrainODE and existing SIREN flows:

```bash
python examples/ADNI_1_L_No_MCI/brainode_comparison_task3_core_brainode_original/scripts/evaluate_future_mesh_forecasts.py --models qc_siren_latent_ode qc_siren_drop_bad_min2 qc_siren_local_decomp_volume qc_brainode_pca150 --splits train val test --device cuda:0 --surface-samples 30000 --mesh-resolution 80
```

## Output Folders

Training checkpoints:

`ModelParameters/`

Numerical split summaries:

`analysis/checkpoint_best/`

Rich HTML:

`analysis/notebook_best/index.html`

Real-mesh forecast comparison:

`examples/ADNI_1_L_No_MCI/brainode_comparison_task3_core_brainode_original/analysis/future_mesh_forecast_comparison`
