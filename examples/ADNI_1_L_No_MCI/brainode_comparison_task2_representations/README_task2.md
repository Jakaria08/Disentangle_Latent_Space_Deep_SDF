# Task 2: INR/PCA Representations and Reconstruction Checks

This folder is an isolated representation pipeline for the smooth left
hippocampus ADNI No-MCI cohort produced by Task 1.

It creates and compares:

1. train-only PCA with a primary 150D BrainODE-compatible representation;
2. a 256D ReLU DeepSDF decoder with eikonal regularization;
3. a 256D SIREN decoder with eikonal regularization;
4. the legacy `minimal_eikonal` DeepSDF checkpoint refit on the current smooth cohort.

No Task 1 file, source dataset, existing experiment, or repository training
script is modified.

The initial memory-efficient pilot outputs are preserved under
`inr/deepsdf_eikonal/` and `inr/siren_eikonal/`. The first corrected attempts
are preserved under `inr/deepsdf_eikonal_spec/` and
`inr/siren_eikonal_spec/`. The failed all-point SIREN attempt is preserved
under `inr/siren_eikonal_spec_fast/`, and the failed L1-only warm-up is
preserved under `inr/siren_eikonal_warmup/`. The current outputs are
`inr/deepsdf_eikonal_spec_fast/` for ReLU and
`inr/siren_naisr_enhanced/` for SIREN. The legacy checkpoint is loaded from
`examples/ADNI/minimal_eikonal/ModelParameters/latest.pth` and its Task 2
artifacts are written under `inr/minimal_eikonal/`.

## Fixed Data Contract

| Split | Scans |
| --- | ---: |
| train | 617 |
| validation | 39 |
| test | 71 |
| total | 727 |

The decoder and PCA model are fitted using training scans only. Validation is
used for checkpoint/model selection. Test data is reporting-only.

The existing smooth scaled coordinates are used without additional
normalization because the mesh and SDF files already share this coordinate
system.

## Environment

Run every command from the repository root:

```bash
cd /home/jakaria/INR/Deep3DComp
conda activate inr_sdf
```

The long INR stages require a CUDA GPU. Confirm it before training:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

## Step 1: Audit Inputs

This reads all 727 meshes and SDF archives but does not alter them:

```bash
python examples/ADNI_1_L_No_MCI/brainode_comparison_task2_representations/scripts/audit_inputs.py
python examples/ADNI_1_L_No_MCI/brainode_comparison_task2_representations/scripts/validate_task2.py --stage inputs
```

Expected result: both commands report `status: pass`.

## Step 2: Fit and Export PCA

PCA is fitted on the 617 training scans. Coefficients are exported for all
scans. Reconstruction metrics are calculated for 32, 64, 128, 150, and 256
components. Full PLY meshes are written for 150D and 256D.

```bash
python examples/ADNI_1_L_No_MCI/brainode_comparison_task2_representations/scripts/fit_export_pca.py
python examples/ADNI_1_L_No_MCI/brainode_comparison_task2_representations/scripts/validate_task2.py --stage pca
```

The canonical BrainODE input is the first 150 columns of each exported
coefficient vector.

## Step 3: Train ReLU DeepSDF + Eikonal

This is a long GPU job:

```bash
python examples/ADNI_1_L_No_MCI/brainode_comparison_task2_representations/scripts/train_inr_decoder.py \
  --config examples/ADNI_1_L_No_MCI/brainode_comparison_task2_representations/configs/inr_deepsdf_eikonal.json \
  --device cuda:0
```

Resume the latest periodic checkpoint:

```bash
python examples/ADNI_1_L_No_MCI/brainode_comparison_task2_representations/scripts/train_inr_decoder.py \
  --config examples/ADNI_1_L_No_MCI/brainode_comparison_task2_representations/configs/inr_deepsdf_eikonal.json \
  --device cuda:0 \
  --resume latest
```

If stopped with `Ctrl+C`, resume with `--resume interrupted`.

## Step 4: Train SIREN + Hybrid Reconstruction

This must be a fresh run. Do not resume a checkpoint from any earlier SIREN
folder:

```bash
python examples/ADNI_1_L_No_MCI/brainode_comparison_task2_representations/scripts/train_inr_decoder.py \
  --config examples/ADNI_1_L_No_MCI/brainode_comparison_task2_representations/configs/inr_siren_eikonal.json \
  --device cuda:0
```

The corrected SIREN schedule is:

```text
all epochs: L1 + 0.1 normalized SDF-MSE + 0.1 sign BCE
architecture: four 256-wide sine layers, latent concatenated at input only
epochs 1-200: hybrid reconstruction and latent regularization only
epochs 201-400: linearly ramp raw-field eikonal from 0 to 0.002
epochs 401-2001: raw-field eikonal fixed at 0.002
latent learning rate: 0.001
sign-logit temperature: 0.02
network learning rate: 0.0001
validation frequency: 25 epochs
```

The SIREN log also records `sign_accuracy`, `raw_output_abs_mean`, and
`raw_output_std`. During warm-up, sign accuracy must rise clearly above 0.5 and
validation SDF L1 must fall below the zero-field baseline of approximately
0.0346.

After both training runs:

```bash
python examples/ADNI_1_L_No_MCI/brainode_comparison_task2_representations/scripts/validate_task2.py --stage trained
```

Each run writes `best.pth`, selected only by held-out validation SDF L1.

Both corrected configurations use:

```text
effective scenes per batch: 32
ReLU scenes per gradient chunk: 16
SIREN scenes per gradient chunk: 32
SDF samples per scene: 16,384
eikonal points per scene: 16,384
code regularization: mean L2 latent norm
code regularization warm-up: 100 epochs
epochs: 2,001
data-loader workers: 16
training SDF data: preloaded into shared host RAM
persistent workers: enabled
prefetch factor: 4
```

Gradient chunking changes peak memory, not the effective batch objective.
Every eikonal point is retained. Each epoch log reports
`cuda_peak_allocated_gb` and `cuda_peak_reserved_gb`.

For SIREN, all eikonal points are retained after the 200-epoch reconstruction
warm-up, and gradients are computed from the raw, unclamped field.

The defaults are based on measured peak memory from the first corrected
attempts: 2.61 GiB for ReLU and 4.97 GiB for SIREN at chunk 4. If the GPU still
has substantial free memory, increase occupancy without changing the effective
batch:

```bash
python examples/ADNI_1_L_No_MCI/brainode_comparison_task2_representations/scripts/train_inr_decoder.py \
  --config examples/ADNI_1_L_No_MCI/brainode_comparison_task2_representations/configs/inr_deepsdf_eikonal.json \
  --device cuda:0 \
  --scene-chunk-size 24
```

SIREN already processes the full effective batch in one chunk. If this causes
CUDA out-of-memory after eikonal begins, reduce its chunk to 16 or 8. The ReLU
default remains 16.
Do not reduce `eikonal_points_per_scene`; the corrected configuration evaluates
the eikonal term on all 16,384 sampled points.

## Step 5: Fit Frozen-Decoder Latents

These are long GPU jobs. Every train/validation/test scan is fitted with the
same latent inference procedure.

ReLU:

```bash
python examples/ADNI_1_L_No_MCI/brainode_comparison_task2_representations/scripts/fit_export_inr_latents.py \
  --config examples/ADNI_1_L_No_MCI/brainode_comparison_task2_representations/configs/inr_deepsdf_eikonal.json \
  --checkpoint best \
  --device cuda:0 \
  --skip-existing
```

SIREN:

```bash
python examples/ADNI_1_L_No_MCI/brainode_comparison_task2_representations/scripts/fit_export_inr_latents.py \
  --config examples/ADNI_1_L_No_MCI/brainode_comparison_task2_representations/configs/inr_siren_eikonal.json \
  --checkpoint best \
  --device cuda:0 \
  --skip-existing
```

Legacy `minimal_eikonal`:

```bash
python examples/ADNI_1_L_No_MCI/brainode_comparison_task2_representations/scripts/fit_export_inr_latents.py \
  --config examples/ADNI_1_L_No_MCI/brainode_comparison_task2_representations/configs/inr_deepsdf_eikonal.json \
  --checkpoint examples/ADNI/minimal_eikonal/ModelParameters/latest.pth \
  --output-dir examples/ADNI_1_L_No_MCI/brainode_comparison_task2_representations/inr/minimal_eikonal \
  --name minimal_eikonal \
  --device cuda:0 \
  --skip-existing
```

Validate:

```bash
python examples/ADNI_1_L_No_MCI/brainode_comparison_task2_representations/scripts/validate_task2.py --stage latents
```

## Step 6: Reconstruct INR Meshes

These are long GPU jobs because each scan uses a 256-cubed SDF grid.

ReLU:

```bash
python examples/ADNI_1_L_No_MCI/brainode_comparison_task2_representations/scripts/reconstruct_inr_meshes.py \
  --config examples/ADNI_1_L_No_MCI/brainode_comparison_task2_representations/configs/inr_deepsdf_eikonal.json \
  --checkpoint best \
  --device cuda:0 \
  --skip-existing
```

SIREN:

```bash
python examples/ADNI_1_L_No_MCI/brainode_comparison_task2_representations/scripts/reconstruct_inr_meshes.py \
  --config examples/ADNI_1_L_No_MCI/brainode_comparison_task2_representations/configs/inr_siren_eikonal.json \
  --checkpoint best \
  --device cuda:0 \
  --skip-existing
```

Legacy `minimal_eikonal`:

```bash
python examples/ADNI_1_L_No_MCI/brainode_comparison_task2_representations/scripts/reconstruct_inr_meshes.py \
  --config examples/ADNI_1_L_No_MCI/brainode_comparison_task2_representations/configs/inr_deepsdf_eikonal.json \
  --checkpoint examples/ADNI/minimal_eikonal/ModelParameters/latest.pth \
  --output-dir examples/ADNI_1_L_No_MCI/brainode_comparison_task2_representations/inr/minimal_eikonal \
  --name minimal_eikonal \
  --device cuda:0 \
  --skip-existing
```

Build the combined scan-level representation manifest:

```bash
python examples/ADNI_1_L_No_MCI/brainode_comparison_task2_representations/scripts/build_representation_manifest.py \
  --require-complete
```

## Step 7: Evaluate All Representations

This compares PCA-150, PCA-256, ReLU INR, SIREN INR, and `minimal_eikonal`
without ICP or other post-reconstruction alignment:

```bash
python examples/ADNI_1_L_No_MCI/brainode_comparison_task2_representations/scripts/evaluate_reconstructions.py
```

Metrics:

- squared symmetric Chamfer distance;
- ASSD;
- HD95;
- absolute and relative volume error;
- volume correlation;
- watertightness and winding consistency.

## Step 8: Compare the INR Models

```bash
python examples/ADNI_1_L_No_MCI/brainode_comparison_task2_representations/scripts/compare_inr_models.py
```

This uses the validation-scan intersection across all configured INR models and
reports all pairwise Wilcoxon tests with Holm-adjusted p-values.
Select using validation metrics, primarily median ASSD, then Chamfer, HD95,
and relative volume error. Do not select using test results.

Final validation:

```bash
python examples/ADNI_1_L_No_MCI/brainode_comparison_task2_representations/scripts/validate_task2.py --stage complete
```

## Important Outputs

```text
pca/model/
pca/model/pca_brainode.pkl
pca/coefficients/
pca/reconstructed_meshes/
inr/deepsdf_eikonal_spec_fast/checkpoints/best.pth
inr/deepsdf_eikonal_spec_fast/latents/
inr/siren_naisr_enhanced/checkpoints/best.pth
inr/siren_naisr_enhanced/latents/
inr/minimal_eikonal/latents/
evaluation/metrics/reconstruction_per_scan.csv
evaluation/metrics/reconstruction_summary.json
evaluation/comparison/inr_model_comparison.json
metadata/representation_manifest.csv
metadata/validation_complete.json
```

## Small Tests

The following do not save checkpoints:

```bash
python examples/ADNI_1_L_No_MCI/brainode_comparison_task2_representations/scripts/train_inr_decoder.py \
  --config examples/ADNI_1_L_No_MCI/brainode_comparison_task2_representations/configs/inr_deepsdf_eikonal.json \
  --device cpu \
  --smoke-test

python examples/ADNI_1_L_No_MCI/brainode_comparison_task2_representations/scripts/train_inr_decoder.py \
  --config examples/ADNI_1_L_No_MCI/brainode_comparison_task2_representations/configs/inr_siren_eikonal.json \
  --device cpu \
  --smoke-test
```
