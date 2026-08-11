# Delta-SDF Local Disease-Decomposed Flow, QC v1

This experiment uses the existing QC-filtered large ADNI left-hippocampus data
and the frozen no-skip SIREN decoder. It does not create new scan latents.

Main settings:

- `FlowType=local_disease_decomposed`
- transport is composed internally in 0.5-year steps
- velocity is `CN velocity + label_ad * AD residual velocity`
- source and target states are existing scan-specific SIREN latents
- latent updates are low-rank, with `FlowRank=64`
- flow MLP hidden dimensions are `[192, 192]`
- training includes target SDF reconstruction
- training includes change-weighted SDF, delta-SDF direction, delta-SDF RMAE,
  and no-change margin losses
- training keeps weaker differentiable relative-volume and CN-vs-AD
  counterfactual volume losses
- latent direction, magnitude, manifold, speed, sequence rollout, and
  sequence-magnitude losses are disabled
- training pairs use subject/diagnosis-balanced sampling

The same data contract still rejects subjects with diagnosis changes.

## Commands

Validate inputs and run one CPU forward pass:

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python train_deep_sdf_longitudinal_direct_flow.py -e /home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_delta_sdf_local_decomposed_flow_qc_v1 --validate-only
```

Optional small smoke run:

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python train_deep_sdf_longitudinal_direct_flow.py -e /home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_delta_sdf_local_decomposed_flow_qc_v1 --gpu 0 --smoke-batches 2
```

Full training:

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python train_deep_sdf_longitudinal_direct_flow.py -e /home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_delta_sdf_local_decomposed_flow_qc_v1 --gpu 0
```

Evaluate all splits after training:

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python evaluate_deep_sdf_longitudinal_direct_flow.py -e /home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_delta_sdf_local_decomposed_flow_qc_v1 --checkpoint best --split all --gpu 0 --composed-step-years 0.5
```

Summarize gap bins:

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python /home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_delta_sdf_local_decomposed_flow_qc_v1/scripts/summarize_gap_bins.py --analysis /home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_delta_sdf_local_decomposed_flow_qc_v1/analysis/checkpoint_best
```

Generate rich HTML visualizations:

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python /home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_delta_sdf_local_decomposed_flow_qc_v1/scripts/run_rich_visualization_analysis.py --checkpoint best --gpu 0 --composed-step-years 0.5 --subjects-per-split 30 --min-scans 3
```

The visualization output is written to:

```text
/home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_delta_sdf_local_decomposed_flow_qc_v1/analysis/notebook_best
```
