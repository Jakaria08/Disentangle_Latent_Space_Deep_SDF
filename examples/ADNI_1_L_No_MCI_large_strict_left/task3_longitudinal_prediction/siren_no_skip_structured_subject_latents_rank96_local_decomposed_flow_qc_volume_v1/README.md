# Structured Subject Latents Rank 96 + Local Disease-Decomposed Flow, QC Volume v1

This is the rank-96 revision of experiment 2.

The previous structured-latent run improved disease-conditioned volume behavior,
but it damaged SIREN reconstruction too much.  This version keeps the same
frozen SIREN decoder and same flow architecture, but changes the Stage 1
structured-latent fit:

```text
StructuredRank = 96
StructuredLatentRawLossLambda = 2.0
StructuredLatentVelocitySmoothnessLambda = 0.001
StructuredLatentAccelerationLambda = 0.002
StructuredLatentRelativeVolumeLambda = 0.005
```

The goal is to preserve individual scan reconstruction much better while still
reducing longitudinal latent noise.

## Stage 1: Validate Inputs

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python /home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_structured_subject_latents_rank96_local_decomposed_flow_qc_volume_v1/scripts/fit_structured_siren_latents.py --experiment /home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_structured_subject_latents_rank96_local_decomposed_flow_qc_volume_v1 --validate-only
```

## Optional Smoke Check

This writes only to `latents_smoke/` and `structured_latent_fit_smoke/`.
It does not create the production latent files used by flow training.

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python /home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_structured_subject_latents_rank96_local_decomposed_flow_qc_volume_v1/scripts/fit_structured_siren_latents.py --experiment /home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_structured_subject_latents_rank96_local_decomposed_flow_qc_volume_v1 --gpu 0 --limit-subjects-per-split 2 --epochs 1 --inference-epochs 1 --batch-scans 2 --samples-per-scan 256 --pair-batch-size 4 --triple-batch-size 4 --smoke-batches 1
```

## Stage 1: Fit Structured Latents

Single GPU:

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python /home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_structured_subject_latents_rank96_local_decomposed_flow_qc_volume_v1/scripts/fit_structured_siren_latents.py --experiment /home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_structured_subject_latents_rank96_local_decomposed_flow_qc_volume_v1 --gpu 0
```

Three GPUs for decoder evaluation:

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python /home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_structured_subject_latents_rank96_local_decomposed_flow_qc_volume_v1/scripts/fit_structured_siren_latents.py --experiment /home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_structured_subject_latents_rank96_local_decomposed_flow_qc_volume_v1 --gpus 0,1,2
```

Stage 1 must create:

```text
/home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_structured_subject_latents_rank96_local_decomposed_flow_qc_volume_v1/latents/train_latents.npz
/home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_structured_subject_latents_rank96_local_decomposed_flow_qc_volume_v1/latents/val_latents.npz
/home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_structured_subject_latents_rank96_local_decomposed_flow_qc_volume_v1/latents/test_latents.npz
```

## Stage 1 Audit

Run this before flow training:

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python /home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_structured_subject_latents_rank96_local_decomposed_flow_qc_volume_v1/scripts/audit_structured_latents.py --experiment /home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_structured_subject_latents_rank96_local_decomposed_flow_qc_volume_v1 --gpu 0 --sdf-samples-per-scan 2048
```

Look first at:

```text
/home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_structured_subject_latents_rank96_local_decomposed_flow_qc_volume_v1/structured_latent_fit/audit/sdf_summary.csv
/home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_structured_subject_latents_rank96_local_decomposed_flow_qc_volume_v1/structured_latent_fit/audit/summary.csv
```

Continue to flow training only if `mean_structured_minus_raw_sdf_l1` is much
smaller than the previous rank-96 run.  A good target is around `0.0015` or
less for train, validation, and test.

## Stage 2: Validate Flow Inputs

Run this after Stage 1 creates the production latent archives:

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python /home/jakaria/INR/Deep3DComp/train_deep_sdf_longitudinal_direct_flow.py -e /home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_structured_subject_latents_rank96_local_decomposed_flow_qc_volume_v1 --validate-only
```

## Stage 2: Train Flow

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python /home/jakaria/INR/Deep3DComp/train_deep_sdf_longitudinal_direct_flow.py -e /home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_structured_subject_latents_rank96_local_decomposed_flow_qc_volume_v1 --gpu 0
```

## Evaluate

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python /home/jakaria/INR/Deep3DComp/evaluate_deep_sdf_longitudinal_direct_flow.py -e /home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_structured_subject_latents_rank96_local_decomposed_flow_qc_volume_v1 --checkpoint best --split all --gpu 0 --composed-step-years 0.5
```

## Gap Bins

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python /home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_structured_subject_latents_rank96_local_decomposed_flow_qc_volume_v1/scripts/summarize_gap_bins.py --analysis /home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_structured_subject_latents_rank96_local_decomposed_flow_qc_volume_v1/analysis/checkpoint_best
```

## Rich HTML Visualization

Use 30 subjects per split with at least 3 scans because the test split has only
15 AD subjects with at least 3 scans.

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python /home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_structured_subject_latents_rank96_local_decomposed_flow_qc_volume_v1/scripts/run_rich_visualization_analysis.py --checkpoint best --gpu 0 --composed-step-years 0.5 --subjects-per-split 30 --min-scans 3
```

HTML output:

```text
/home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_structured_subject_latents_rank96_local_decomposed_flow_qc_volume_v1/analysis/notebook_best
```
