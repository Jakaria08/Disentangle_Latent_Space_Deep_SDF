# PCA128 voxel-cognition BrainODE results

## Status and selected models

The seed-42 experiment is complete. Training used 2,037 scans from 475 subjects;
validation used 269 scans from 61 disjoint subjects. The held-out test set has
277 scans from 61 subjects and was opened only after both components were
selected.

- Cognition CNN: selected epoch 7 of an early-stopped 40-epoch schedule.
- BrainODE: trained for all 100 epochs; selected epoch 2 by the prespecified
  decoded first-to-last validation shape ratio.
- PCA cocycle comparator: its existing feasible `best.pt`, epoch 7; never
  `latest.pt`.
- Anatomy and representation: left hippocampus, PCA128, physical-mm decoder.
- Converter/MCI/pseudo samples: 0 / 0 / 0.

The cognition output is an AD-like anatomical probability. It is not a measured
cognitive score, clinical conversion probability, or forecast that a CN subject
will become AD.

## Cognition-estimator performance

The estimator consumes only a 32 x 32 x 32 solid shape mask. It does not receive
age, diagnosis metadata, subject identity, or latent coordinates.

| Held-out test unit/domain | AUROC | AUPRC | Balanced accuracy | Brier |
|---|---:|---:|---:|---:|
| Scan, raw observed mesh | 0.9308 | 0.8384 | 0.8605 | 0.0987 |
| Subject, raw observed mesh | 0.9307 | 0.8906 | 0.8653 | 0.1030 |
| Scan, PCA-decoded observed latent | 0.9320 | 0.8397 | 0.8631 | 0.0972 |
| Subject, PCA-decoded observed latent | 0.9307 | 0.8815 | 0.8653 | 0.1017 |

The mean absolute raw-versus-PCA-decoded probability difference is 0.0156 on
test, so PCA reconstruction introduces little estimator-domain shift.

## Held-out first-to-last prediction

All surface values below are computed from decoded correspondence meshes in mm;
volume and rate are not computed from voxel counts.

| Method | Coordinate MAE mm | Vertex Euclidean mm | End-to-end RMSE mm | Volume relative error | Log-volume-rate error |
|---|---:|---:|---:|---:|---:|
| No change | 0.17106 | 0.33990 | — | 0.06306 | 0.03197 |
| PCA128 direct cocycle | **0.15547** | **0.30966** | **0.20888** | **0.03414** | **0.02015** |
| BrainODE, fixed observed label | 0.15864 | 0.31589 | 0.21252 | 0.04344 | 0.02503 |
| BrainODE, voxel cognition feedback | 0.15975 | 0.31804 | 0.21378 | 0.04523 | 0.02554 |

Relative to no-change coordinate MAE, PCA cocycle improves 9.11%, fixed-label
BrainODE improves 7.26%, and voxel-feedback BrainODE improves 6.61%.

On all forward pairs with subjects as the bootstrap unit, fixed-label BrainODE
minus PCA cocycle coordinate MAE is +0.00127 mm (95% CI +0.00019 to +0.00243).
Voxel-feedback BrainODE minus PCA cocycle is +0.00155 mm (95% CI +0.00041 to
+0.00273). Positive is worse, so PCA cocycle wins both paired comparisons.

## One-, two-, and four-shot BrainODE

The n-shot result averages mesh predictions from the first n observed visits to
the final visit. Cohort sizes differ because a subject needs at least n+1 scans.

| Mode | Shots | Subjects | Coordinate MAE mm | End-to-end RMSE mm |
|---|---:|---:|---:|---:|
| Fixed label | 1 | 61 | 0.15864 | 0.21252 |
| Fixed label | 2 | 58 | 0.14321 | 0.19183 |
| Fixed label | 4 | 21 | 0.13177 | 0.17652 |
| Voxel feedback | 1 | 61 | 0.15975 | 0.21378 |
| Voxel feedback | 2 | 58 | 0.14411 | 0.19289 |
| Voxel feedback | 4 | 21 | 0.13307 | 0.17794 |

These rows are not a matched n-shot comparison with PCA cocycle; the primary
cocycle comparison above uses the common all-pair and first-to-last contracts.

## Condition behavior and consistency

Using the same test source and ages but setting `c=1` instead of `c=0` changes
the predicted surface by 0.01446 mm coordinate MAE and reduces predicted volume
by 1.04% on average. This demonstrates condition injectivity only; it is not a
CN-to-AD conversion experiment.

For feedback trajectories, the mean decoded-shape condition changes from 0.163
to 0.265 in CN and from 0.672 to 0.713 in AD. Both groups remain group-conditional;
the CN increase must be described as an aging/anatomical-score trend, not
conversion.

| Test consistency | Semigroup mean | Inverse mean |
|---|---:|---:|
| Fixed-label BrainODE | 1.20e-7 | 1.93e-8 |
| Voxel-feedback BrainODE | 7.57e-5 | 3.06e-4 |
| PCA direct cocycle | 8.75e-4 | 3.20e-3 |

The feedback state-dependence introduces a small numerical consistency defect,
but it remains lower than the learned direct cocycle defect.

## Scientific conclusion

The shape-only CNN clearly distinguishes CN-like from AD-like hippocampal
anatomy on unseen subjects, and the BrainODE trajectory model improves over
no-change. However, replacing the known stable diagnosis with CNN feedback does
not improve held-out trajectory prediction and slightly hurts it. The PCA
direct cocycle is the best tested correspondence-based PCA128 method here.

With no converter subjects, this is the defensible BrainODE
"cognition estimator without pseudo sampling" ablation. It cannot reproduce or
validate the paper's converter-specific pseudo-cognition mechanism.

## Result artifacts

- Combined selected checkpoint:
  `/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task3_latent_flow_128_v1/training/pca128/brainode_cognition/pca128_brainode_cognition_s42/checkpoints/combined_best.pt`
- Full validation/test reports:
  `.../brainode_cognition_evaluation/{val,test}/summary.json`
- Matched PCA cocycle reports:
  `.../pca128/direct_c4/pca128_direct_c4_s42/brainode_cognition_comparator/{val,test}/summary.json`
- Final CSV/JSON comparisons:
  `.../pca128/brainode_cognition/pca128_brainode_cognition_s42/comparison/`
