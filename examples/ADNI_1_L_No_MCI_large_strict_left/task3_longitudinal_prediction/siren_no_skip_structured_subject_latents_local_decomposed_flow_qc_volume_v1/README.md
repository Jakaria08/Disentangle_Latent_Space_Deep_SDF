# Structured Subject Latents + Local Disease-Decomposed Flow, QC Volume v1

This is experiment 2.

The purpose is to fix the latent-space problem before training the flow.  The
frozen no-skip SIREN decoder is kept fixed, but the per-scan latents are
re-estimated with a longitudinal structure:

```text
z_struct(i,k) = a_i + B q_i,k
```

where `a_i` is one subject anchor, `B` is a shared low-rank basis, and `q_i,k`
is a visit coordinate.  The shared basis is learned only on train subjects.
Validation and test subjects infer anchors and visit coordinates with that
basis frozen.

The exported structured latent files keep the existing direct-flow contract:

```text
latents/train_latents.npz
latents/val_latents.npz
latents/test_latents.npz
```

Each archive contains `scan_ids` and `latents`, so the existing flow trainer can
be reused.

## Stage 1: Validate Inputs

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python /home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_structured_subject_latents_local_decomposed_flow_qc_volume_v1/scripts/fit_structured_siren_latents.py --experiment /home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_structured_subject_latents_local_decomposed_flow_qc_volume_v1 --validate-only
```

## Optional Smoke Check

This writes small structured latent outputs for a tiny subject subset under
`latents_smoke/` and `structured_latent_fit_smoke/`.  It does not create the
production latent files used by flow training.

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python /home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_structured_subject_latents_local_decomposed_flow_qc_volume_v1/scripts/fit_structured_siren_latents.py --experiment /home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_structured_subject_latents_local_decomposed_flow_qc_volume_v1 --gpu 0 --limit-subjects-per-split 2 --epochs 1 --inference-epochs 1 --batch-scans 2 --samples-per-scan 256 --pair-batch-size 4 --triple-batch-size 4 --smoke-batches 1
```

After the smoke check, run the full Stage 1 command to create the production
`latents/` archives.

## Stage 1: Fit Structured Latents

Single GPU:

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python /home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_structured_subject_latents_local_decomposed_flow_qc_volume_v1/scripts/fit_structured_siren_latents.py --experiment /home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_structured_subject_latents_local_decomposed_flow_qc_volume_v1 --gpu 0
```

Multiple GPUs for decoder evaluation:

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python /home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_structured_subject_latents_local_decomposed_flow_qc_volume_v1/scripts/fit_structured_siren_latents.py --experiment /home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_structured_subject_latents_local_decomposed_flow_qc_volume_v1 --gpus 0,1,2
```

Stage 1 outputs:

```text
/home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_structured_subject_latents_local_decomposed_flow_qc_volume_v1/latents
/home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_structured_subject_latents_local_decomposed_flow_qc_volume_v1/structured_latent_fit
```

## Stage 1 Audit

Fast latent-trajectory audit:

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python /home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_structured_subject_latents_local_decomposed_flow_qc_volume_v1/scripts/audit_structured_latents.py --experiment /home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_structured_subject_latents_local_decomposed_flow_qc_volume_v1
```

Audit with SDF reconstruction comparison:

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python /home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_structured_subject_latents_local_decomposed_flow_qc_volume_v1/scripts/audit_structured_latents.py --experiment /home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_structured_subject_latents_local_decomposed_flow_qc_volume_v1 --gpu 0 --sdf-samples-per-scan 2048
```

Audit outputs:

```text
/home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_structured_subject_latents_local_decomposed_flow_qc_volume_v1/structured_latent_fit/audit
```

Check `summary.csv` first.  Continue to flow training only if structured latents
do not damage SDF reconstruction too much and improve trajectory smoothness.

## Stage 2: Validate Flow Inputs

Run this after Stage 1 creates the structured latent archives.

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python /home/jakaria/INR/Deep3DComp/train_deep_sdf_longitudinal_direct_flow.py -e /home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_structured_subject_latents_local_decomposed_flow_qc_volume_v1 --validate-only
```

## Stage 2: Train Flow

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python /home/jakaria/INR/Deep3DComp/train_deep_sdf_longitudinal_direct_flow.py -e /home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_structured_subject_latents_local_decomposed_flow_qc_volume_v1 --gpu 0
```

## Evaluate

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python /home/jakaria/INR/Deep3DComp/evaluate_deep_sdf_longitudinal_direct_flow.py -e /home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_structured_subject_latents_local_decomposed_flow_qc_volume_v1 --checkpoint best --split all --gpu 0 --composed-step-years 0.5
```

## Gap Bins

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python /home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_structured_subject_latents_local_decomposed_flow_qc_volume_v1/scripts/summarize_gap_bins.py --analysis /home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_structured_subject_latents_local_decomposed_flow_qc_volume_v1/analysis/checkpoint_best
```

## Rich HTML Visualization

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python /home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_structured_subject_latents_local_decomposed_flow_qc_volume_v1/scripts/run_rich_visualization_analysis.py --checkpoint best --gpu 0 --composed-step-years 0.5 --subjects-per-split 34 --min-scans 3
```

HTML output:

```text
/home/jakaria/INR/Deep3DComp/examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/siren_no_skip_structured_subject_latents_local_decomposed_flow_qc_volume_v1/analysis/notebook_best
```
