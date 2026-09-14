# Cross-cohort left hippocampus - test split, latent 128

`gap` = external - internal. Positive means the reference model is worse than the cohort's own; that difference is the generalization cost.

| model | cohort | internal | external | gap | pooled | loco | external (Euclidean) |
|---|---|---|---|---|---|---|---|
| adaptive_z128 | adni | 0.038143 | - | - | - | - | - |
| adaptive_z128 | aibl | 0.155392 | 0.090651 | -0.064740 | - | - | 0.157013 |
| adaptive_z128 | calsnic | 0.116812 | 0.092941 | -0.023871 | - | - | 0.160978 |
| adaptive_z128 | oasis | 0.120348 | 0.097847 | -0.022500 | - | - | 0.169477 |
| pca | adni | 0.034380 | 0.034381 | 0.000001 | 0.036151 | 0.054067 | 0.059549 |
| pca | aibl | 0.088284 | 0.080022 | -0.008263 | 0.069858 | 0.071214 | 0.138601 |
| pca | calsnic | 0.077071 | 0.080994 | 0.003923 | 0.070989 | 0.070989 | 0.140286 |
| pca | oasis | 0.080127 | 0.084873 | 0.004747 | 0.075128 | 0.078035 | 0.147005 |
| spiralnet_z128 | adni | 0.037868 | - | - | - | - | - |
| spiralnet_z128 | aibl | 0.138811 | 0.090435 | -0.048376 | - | - | 0.156637 |
| spiralnet_z128 | calsnic | 0.106573 | 0.092110 | -0.014463 | - | - | 0.159539 |
| spiralnet_z128 | oasis | 0.108589 | 0.096935 | -0.011654 | - | - | 0.167896 |

## ADNI reference (validation, published)

| model | val vertex_rmse_mm |
|---|---|
| pca_128 | 0.033668 |
| spiralnet_z128 | 0.036784 |
| adaptive_z128 | 0.037237 |
| lamm_z128 | 0.037685 |
| meshmae_tuned | 0.037867 |

Note: on ADNI, PCA-128 beats every learned model at matched latent size. Any claim that a learned model wins on a new cohort should be checked against its own PCA baseline on that cohort, not against ADNI's.
