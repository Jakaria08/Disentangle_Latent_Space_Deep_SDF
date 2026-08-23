# CALSNIC left-cortex exact SDF and multiresolution reconstruction

This task is the CALSNIC-control left-pial counterpart of the recent ADNI exact-label and
dense-multiresolution single-field pipelines. It keeps the tested training/sampling engine and adds
the CALSNIC-specific transform that `PreprocessMesh.cpp` applied but did not save: the per-mesh AABB
midpoint is subtracted before SDF sampling.

All generated manifests, exact archives, PCA arrays, checkpoints, meshes, metrics, caches, and
temporary files are restricted to:

`/mnt/bulk10tb/Deep3DComp/CALSNIC/control_L_exact_multires_v1`

The source OBJ and NPZ files are read-only. The original 203 SDF archives are never overwritten.

## Fixed experiment contract

- Cohort: 203 CALSNIC healthy-control left pial surfaces.
- Split: 173 train, 15 validation, 15 locked test.
- Split balance: study, site, sex, age quartile, and eTIV quartile.
- Exact labels: original XYZ bit-for-bit; exact point-to-triangle magnitude and fast-winding sign.
- Geometry transform: subtract the scaled OBJ AABB midpoint in SDF space; invert the fitted
  scaled-OBJ-to-original-OBJ similarity transform for millimetre evaluation.
- Latent size: 256.
- Capacity control: shared dense 8/16/24/32/48/64 grids.
- Primary model: shared dense 8/16/24/32/48/64/96/128 grids.
- Primary loss: broad and cell-balanced near-surface clamped L1; no Eikonal.
- Periodic geometry evaluation: validation only at 256 cubed.
- Final test: explicit confirmation, selected model only, 256 and 512 cubed.
- PCA: train-only matched basis; effective rank is at most 172. Requests for 256/512 PCA
  components therefore equal the rank-172 reconstruction.

## Commands

Run from `/home/jakaria/INR/Deep3DComp`.

```bash
TASK=examples/CALSNIC_control_L/task_exact_multires_cortex_v1
RUN="$TASK/scripts/run_on_bulk.sh"
C64="$TASK/configs/calsnic_control_L_multires64_z256_exact.json"
C128="$TASK/configs/calsnic_control_L_multires128_z256_exact.json"
ROOT=/mnt/bulk10tb/Deep3DComp/CALSNIC/control_L_exact_multires_v1
```

### 1. No-write code/configuration check

```bash
bash "$RUN" "$TASK/scripts/check_pipeline.py"
```

### 2. Build and inspect the locked 173/15/15 manifest

```bash
bash "$RUN" "$TASK/scripts/build_manifest.py"
```

Inspect `$ROOT/selections/calsnic_control_L_split.json` before continuing. Rebuilding an existing
manifest requires `--overwrite` and invalidates every later audit/model tied to the old manifest.

### 3. Recompute exact labels

```bash
bash "$RUN" "$TASK/scripts/relabel_exact_triangle_sdf.py" --workers 4
```

Resume an interrupted run without replacing completed archives:

```bash
bash "$RUN" "$TASK/scripts/relabel_exact_triangle_sdf.py" --workers 4 --resume
```

### 4. Audit every exact archive

```bash
bash "$RUN" "$TASK/scripts/audit_exact_triangle_sdf.py" --points-per-scan 2048 --independent-points-per-scan 64
```

Training wrappers refuse to proceed unless the full audit exists, passed, and still matches the
exact-manifest SHA-256.

### 5. Fit matched train-only PCA

```bash
bash "$RUN" "$TASK/scripts/fit_matched_pca.py"
```

### 6. Validate data and implementation

```bash
bash "$RUN" "$TASK/scripts/check_pipeline.py" --require-generated-data
bash "$RUN" "$TASK/scripts/validate_inputs.py" --config "$C64" --scans 8 --points-per-scan 2048
bash "$RUN" "$TASK/scripts/validate_inputs.py" --config "$C128" --scans 8 --points-per-scan 2048
bash "$RUN" "$TASK/scripts/validate_implementation.py" --config "$C64" --device cpu
bash "$RUN" "$TASK/scripts/validate_implementation.py" --config "$C128" --device cpu
```

Optional no-write GPU benchmark using the full configured sampling counts:

```bash
CUDA_VISIBLE_DEVICES=0 bash "$RUN" "$TASK/scripts/train_multires_sdf.py" --config "$C64" --device cuda:0 --benchmark-step --benchmark-scenes 4
CUDA_VISIBLE_DEVICES=0 bash "$RUN" "$TASK/scripts/train_multires_sdf.py" --config "$C128" --device cuda:0 --benchmark-step --benchmark-scenes 4
```

### 7. Train

```bash
CUDA_VISIBLE_DEVICES=0 bash "$RUN" "$TASK/scripts/train_multires_sdf.py" --config "$C64" --device cuda:0
CUDA_VISIBLE_DEVICES=0 bash "$RUN" "$TASK/scripts/train_multires_sdf.py" --config "$C128" --device cuda:0
```

Resume example:

```bash
CUDA_VISIBLE_DEVICES=0 bash "$RUN" "$TASK/scripts/train_multires_sdf.py" --config "$C128" --device cuda:0 --resume latest
```

Periodic mesh evaluation is validation-only. Validation ASSD selects `best_mesh`.

### 8. Re-evaluate selected checkpoints on validation

```bash
CUDA_VISIBLE_DEVICES=0 bash "$RUN" "$TASK/scripts/periodic_evaluate_multires.py" --config "$C64" --checkpoint best_mesh --device cuda:0 --splits val --per-split 15
CUDA_VISIBLE_DEVICES=0 bash "$RUN" "$TASK/scripts/periodic_evaluate_multires.py" --config "$C128" --checkpoint best_mesh --device cuda:0 --splits val --per-split 15
```

Use `compare_evaluations.py` with explicit `LABEL=directory` arguments after locating the two
manual-evaluation directories:

```bash
E64="$ROOT/runs/calsnic_control_L_multires64_z256_exact/manual_evaluation/best_mesh"
E128="$ROOT/runs/calsnic_control_L_multires128_z256_exact/manual_evaluation/best_mesh"
bash "$RUN" "$TASK/scripts/compare_evaluations.py" \
  --evaluation "MR64=$E64" \
  --evaluation "MR128=$E128" \
  --split val \
  --output-dir "$ROOT/comparisons/mr64_vs_mr128_validation"
```

Each evaluation's `summary.json` already contains the paired INR-minus-PCA result for the matched
rank-172 oracle PCA baseline.

### 9. One locked test evaluation

After choosing one model and checkpoint from validation, run both extraction resolutions. Example
for the 128-grid model:

```bash
CUDA_VISIBLE_DEVICES=0 bash "$RUN" "$TASK/scripts/periodic_evaluate_multires.py" --config "$C128" --checkpoint best_mesh --device cuda:0 --splits test --per-split 15 --resolution 256 --confirm-test --output-dir "$ROOT/final_test/multires128_best_mesh_r256"
CUDA_VISIBLE_DEVICES=0 bash "$RUN" "$TASK/scripts/periodic_evaluate_multires.py" --config "$C128" --checkpoint best_mesh --device cuda:0 --splits test --per-split 15 --resolution 512 --confirm-test --output-dir "$ROOT/final_test/multires128_best_mesh_r512"
```

Do not run test evaluation for every candidate. Test results never select the model.

### 10. Optional latent export and standalone reconstruction

```bash
CUDA_VISIBLE_DEVICES=0 bash "$RUN" "$TASK/scripts/fit_export_multires_latents.py" --config "$C128" --checkpoint best_mesh --device cuda:0 --splits train val
```

Test export requires the wrapper-only `--confirm-test` flag. Standalone reconstruction similarly
defaults to validation and requires confirmation for test.

## Notebook

`notebooks/calsnic_exact_multires_vs_pca.ipynb` loads the saved PLY triangle meshes and metrics. It
uses Plotly `Mesh3d` with faces, not a scatter/point-cloud rendering, and includes target overlays
and surface-error colouring. The notebook stores no copied meshes or dense SDF grids.

## Storage and test safety

The reconstruction code streams grid queries and saves only PLY meshes; no 256/512 cubed SDF volume
is retained. `run_on_bulk.sh` moves temporary files and caches off the nearly full NVMe. All test
commands require `--confirm-test`, and automatic periodic evaluation contains only `val`.
