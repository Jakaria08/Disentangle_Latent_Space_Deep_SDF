# PCA-conditioned corrective deformation — ADNI left hippocampus

This task keeps all PCA coefficients and learns only a small graph-based correction to the
corresponding PCA mesh:

```text
PCA coefficients -> exact PCA mesh -> orthogonal graph correction -> final mesh
```

The decoder receives no ground-truth mesh at inference. Its displacement is hard-projected
outside the retained PCA subspace, so re-encoding a corrected mesh returns the same PCA latent.
The final correction layer is zero-initialized; epoch zero is exact PCA and is the initial best
checkpoint. Model selection uses validation coordinate RMSE. Test evaluation is a separate,
explicitly authorized command.

Source and configs live in this folder. Every runtime artifact is forced below:

```text
/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task_pca_corrective_deformation_v1
```

## Environment

```bash
cd /home/jakaria/INR/Deep3DComp
export PYTHONDONTWRITEBYTECODE=1
PY=/home/jakaria/anaconda3/envs/pytorch_geo/bin/python
PYTEST=/home/jakaria/anaconda3/envs/inr_sdf/bin/python
TASK=examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task_pca_corrective_deformation_v1
CFG=$TASK/configs/pca128_corrective.json
```

## Validation and tests

```bash
$PYTEST -m pytest -p no:cacheprovider $TASK/tests -q
$PY $TASK/scripts/validate_inputs.py --config $CFG --device cpu
```

## GPU-0 smoke test

Choose an unused bulk output directory:

```bash
SMOKE=/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task_pca_corrective_deformation_v1/smoke/manual_gpu0
$PY $TASK/scripts/train_corrective.py \
  --config $CFG --device cuda:0 --smoke --output-dir $SMOKE

$PY $TASK/scripts/evaluate_corrective.py \
  --config $CFG --checkpoint $SMOKE/best.pt --device cuda:0 \
  --splits val --max-shapes 5 --surface-points 1000 \
  --write-meshes --output-dir $SMOKE/evaluation_smoke
```

The evaluator uses the same exact point-to-triangle `trimesh` proximity metric as the INR
experiments. If `pytorch_geo` does not contain `rtree`, the evaluator imports only `rtree` from
the sibling `inr_sdf` environment; it does not replace the active torch/numpy environment.

## Full training on GPU 0

```bash
RUN=/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task_pca_corrective_deformation_v1/runs/pca128_corrective_s42
$PY $TASK/scripts/train_corrective.py \
  --config $CFG --device cuda:0 --output-dir $RUN
```

Resume the same run after interruption:

```bash
$PY $TASK/scripts/train_corrective.py \
  --config $CFG --device cuda:0 --output-dir $RUN --resume
```

## Validation evaluation

```bash
$PY $TASK/scripts/evaluate_corrective.py \
  --config $CFG --checkpoint $RUN/best.pt --device cuda:0 \
  --splits val --write-meshes --output-dir $RUN/evaluation/best_val
```

## Final test evaluation

Run only after selecting the model from validation:

```bash
$PY $TASK/scripts/evaluate_corrective.py \
  --config $CFG --checkpoint $RUN/best.pt --device cuda:0 \
  --splits test --allow-test --write-meshes --output-dir $RUN/evaluation/best_test
```

The evaluator writes `per_scan_metrics.csv`, `evaluation_summary.json`, and optionally PCA and
corrected PLY meshes. All reported differences are corrected minus PCA, so negative values favor
the corrective decoder for error and distance metrics.
