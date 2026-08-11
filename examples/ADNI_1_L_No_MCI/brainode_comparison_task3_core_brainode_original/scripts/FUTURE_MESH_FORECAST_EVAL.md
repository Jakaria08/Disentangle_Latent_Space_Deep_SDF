# Future Mesh Forecast Evaluation

This is the fair future-shape test for comparing BrainODE PCA forecasts against
SIREN/DeepSDF longitudinal flow forecasts on real target meshes.

The script decodes predicted future endpoint meshes and compares each prediction
to the true future mesh using the same sampled-surface metrics:

- symmetric Chamfer L2 squared
- Chamfer L1
- ASSD
- HD95
- volume absolute/relative error
- triangle surface-area absolute/relative error

It also records two no-change references:

- `source_gt_no_change_*`: the real source mesh held fixed
- `model_no_change_*`: the model's own source reconstruction held fixed

## Smoke Run

Use this first to confirm paths, checkpoint loading, and CUDA mesh decoding.

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python \
  examples/ADNI_1_L_No_MCI/brainode_comparison_task3_core_brainode_original/scripts/evaluate_future_mesh_forecasts.py \
  --models old_brainode_pca150 qc_brainode_pca150 qc_siren_drop_bad_min2 qc_siren_local_decomp_volume \
  --splits test \
  --max-pairs-per-split 2 \
  --device cuda:0 \
  --surface-samples 30000 \
  --mesh-resolution 80
```

## Full Run

This decodes all configured BrainODE and SIREN/DeepSDF future endpoint meshes for
train/val/test. It can take a long time because SDF marching-cubes decoding is
the expensive part.

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python \
  examples/ADNI_1_L_No_MCI/brainode_comparison_task3_core_brainode_original/scripts/evaluate_future_mesh_forecasts.py \
  --models all \
  --splits train val test \
  --device cuda:0 \
  --surface-samples 30000 \
  --mesh-resolution 80
```

To save predicted meshes as `.ply` files, add:

```bash
--save-meshes
```

## Refresh The HTML Report

After the future-mesh script finishes, rebuild the existing comparison report:

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python \
  examples/ADNI_1_L_No_MCI/brainode_comparison_task3_core_brainode_original/scripts/build_best_model_brainode_report.py \
  --surface-samples 30000
```

The report builder imports:

- `analysis/future_mesh_forecast_comparison/future_mesh_per_pair.csv`
- `analysis/future_mesh_forecast_comparison/future_mesh_summary.csv`

and writes copies into:

- `analysis/best_model_brainode_comparison/future_mesh_forecast_per_pair.csv`
- `analysis/best_model_brainode_comparison/future_mesh_forecast_summary.csv`
