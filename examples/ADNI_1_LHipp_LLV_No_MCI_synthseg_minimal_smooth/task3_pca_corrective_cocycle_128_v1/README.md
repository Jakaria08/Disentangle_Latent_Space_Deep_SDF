# PCA-corrective 128-D Direct-C4 cocycle

This isolated task applies the completed August-regularized Direct-C4 transport to the
validation-selected PCA-conditioned corrective Spiral decoder. The transport architecture,
loss weights, subject splits, sampling, and validation selection are unchanged. Runtime
artifacts are restricted to:

`/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task3_pca_corrective_cocycle_128_v1`

The corrective checkpoint uses its own PCA-128 basis. Its longitudinal coefficients are
therefore reprojected from the correspondence meshes and standardized using the training
split only; the older PCA archive is never relabelled or reused as corrective coefficients.

## Commands

```bash
TASK=examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task3_pca_corrective_cocycle_128_v1
PY=/home/jakaria/anaconda3/envs/pytorch_geo/bin/python
export PYTHONDONTWRITEBYTECODE=1

$PY $TASK/tests/verify_contracts.py
$PY $TASK/tests/verify_comparison.py
$PY $TASK/scripts/prepare_representation.py --device cuda:0
$PY $TASK/scripts/train_c4.py --config $TASK/configs/pca_corrective128_direct_c4_s42.json --device cuda:0 --dry-run
$PY $TASK/scripts/train_c4.py --config $TASK/configs/pca_corrective128_direct_c4_s42.json --device cuda:0
```

Validation and test are separate. Test requires explicit authorization:

```bash
RUN=/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task3_pca_corrective_cocycle_128_v1/training/pca_corrective128/direct_c4/pca_corrective128_direct_c4_s42
$PY $TASK/scripts/evaluate.py --run-dir $RUN --split val --device cuda:0
$PY $TASK/scripts/evaluate.py --run-dir $RUN --split test --device cuda:0 --allow-test
```

## Reuse the completed latent models and compare all models

PCA, SpiralNet++, Adaptive-Spiral, and INR are not retrained. Their immutable completed
Direct-C4 runs are read directly. Only the new PCA-corrective transport is trained because
its latent coefficients come from the corrective checkpoint's own PCA basis.

First produce the same validation/test tables used by the completed 128-D experiment:

```bash
PCA=/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/August_Version/training/pca128/direct_c4/pca128_direct_c4_s42_v2
SPIRAL=/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/August_Version/training/spiralnet128/direct_c4/spiralnet128_direct_c4_s42_v2
ADAPTIVE=/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/August_Version/training/adaptive128/direct_c4/adaptive128_direct_c4_s42_v2
COMPARE=/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task3_pca_corrective_cocycle_128_v1/comparisons

$PY $TASK/scripts/compare_results.py --split val --run $PCA --run $SPIRAL --run $ADAPTIVE --run $RUN --output $COMPARE/direct_c4_val.csv
$PY $TASK/scripts/compare_results.py --split test --run $PCA --run $SPIRAL --run $ADAPTIVE --run $RUN --output $COMPARE/direct_c4_test.csv
```

Then make the exact 30,000-point surface, topology, mesh-volume, and first-to-last trend
comparison. `--include-corrective-pca-branch` measures the effect of the corrective decoder
while holding the transported latent exactly fixed. All tables and meshes go to the 10 TB
disk:

```bash
SURFACE=$COMPARE/exact_surface_test_30k
$PY $TASK/scripts/evaluate_surface_forecasts.py \
  --run pca128=$PCA \
  --run spiralnet128=$SPIRAL \
  --run adaptive128=$ADAPTIVE \
  --run pca_corrective128=$RUN \
  --split test --allow-test --device cuda:0 \
  --surface-points 30000 --bootstrap-samples 2000 \
  --include-corrective-pca-branch --write-meshes \
  --output-dir $SURFACE
```

Finally build both the full 61-subject fixed-topology table and the scientifically fairer
20-subject table matched to the completed INR-256 test cohort:

```bash
INR=/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task3_inr_latent_flow_256_v1/training/inr256/direct_c4/inr256_direct_c4_s42
$PY $TASK/scripts/compare_five_models.py \
  --fixed-surface-dir $SURFACE \
  --inr-run $INR \
  --bootstrap-samples 2000 \
  --output-dir $COMPARE/five_model_test_30k
```

The five-model output includes the earlier latent-flow reconstruction, volume, rate,
semigroup, inverse, and validation-selection fields, plus exact ASSD, HD95, Chamfer,
normal/topology diagnostics, absolute volume error in mm3, CN/AD trends, a matched
per-subject table, and paired bootstrap differences against PCA.
