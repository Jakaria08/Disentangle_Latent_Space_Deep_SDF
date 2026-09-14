# Reproducible commands

Run from `/home/jakaria/INR/Deep3DComp`. GPU 1 is used below.

## Matched PCA plain ODE and PCA BrainODE surface evaluation

The two saved baselines use the same fixed PCA archive as the PCA cocycle. This
command evaluates their first-to-last predictions on the exact strict test
subjects used by the primary endpoint table. The evaluator intentionally
refuses to overwrite an existing output directory; choose a new versioned
directory and update `configs/model_registry.json` when recomputing.

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task3_pca_corrective_cocycle_128_v1/scripts/evaluate_surface_forecasts.py \
  --run pca_plain_ode=/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/August_Version/training/pca128/plain_ode/pca128_plain_ode_s42 \
  --run pca_brainode=/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/August_Version/training/pca128/brainode/pca128_brainode_s42 \
  --split test --allow-test --device cuda:1 --surface-points 10000 --bootstrap-samples 2000 \
  --subject-ids-npz /mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task3_inr_latent_flow_256_v1/representations/inr256/test_subject_sequences_256.npz \
  --output-dir /mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/All_Visualization_v1/pca_ode_surface_test
```

## Matched PCA ODE instantaneous surface velocity

This evaluates the exact ODE vector field at each validation scan and maps it
through the frozen PCA decoder Jacobian. It uses the same observed fitted
surface-velocity reference as the cocycle methods.

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/All_Visualization/scripts/extract_pca_ode_velocity.py \
  --device cuda:1 --batch-size 16 --force
```

## Central validation comparison

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task7_direct_mesh_cocycle_lamm_v1/scripts/evaluate_all.py \
  --manifest examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/All_Visualization/configs/central_direct_evaluation.json \
  --split val --device cuda:1 --surface-points 10000 --bootstrap-samples 2000 \
  --output-dir /mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/All_Visualization_v1/central_val
```

## Central final test comparison

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task7_direct_mesh_cocycle_lamm_v1/scripts/evaluate_all.py \
  --manifest examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/All_Visualization/configs/central_direct_evaluation.json \
  --split test --allow-test --device cuda:1 --surface-points 10000 --bootstrap-samples 2000 \
  --output-dir /mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/All_Visualization_v1/central_test
```

## Direct LAMM per-scan diagonal velocity

Repeat the following command for `lamm_global_384` and `lamm_regional_tokens` by changing the name, checkpoint, and final output directory.

```bash
/home/jakaria/anaconda3/envs/pytorch_geo/bin/python examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/All_Visualization/scripts/extract_direct_velocity.py \
  --family lamm --name lamm_global_256 \
  --checkpoint /mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task7_direct_mesh_cocycle_lamm_v1/runs/direct_mesh_lamm_global_c4_z256_s42/checkpoints/best.pt \
  --split test --allow-test --device cuda:1 \
  --output-dir /mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/All_Visualization_v1/direct_velocity_test/lamm_global_256
```

## Build and validate the report

See `README.md` for the audit, analysis-cache, OOD HTML, notebook execution, and final validation commands.

## Matched surface-progression and diagonal-velocity analysis

This command loads the selected frozen checkpoints, uses validation scans only,
and writes the regional, interval-integrated, tangent-consistency, paired-condition,
surface-area, radial-distance, and spatial-map results used by notebook section 6.
It does not train any model.

```bash
/home/jakaria/anaconda3/envs/pytorch_geo/bin/python examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/All_Visualization/scripts/build_surface_progression.py \
  --device cuda:1 --batch-size 16 --bootstrap-samples 2000 --force
```
