# BrainODE paper-core evaluation

This is the strict CN/AD, all-current-QC-subject, paper-core BrainODE evaluation. It uses raw PCA-150 scores and one-subject RK4 inference. It does not include pseudo-cognitive status embedding or a cognition estimator.

## Contract status

- Overall: **pass**
- Training checkpoint reports no test access: passed.
- One-subject inference: passed.
- Raw PCA-150 and all current QC-passed subjects: passed.

## Endpoint Euclidean distance

| Split | Protocol | Diagnosis | Subjects | BrainODE (mm) | No-change (mm) | Improvement (mm) |
|---|---|---|---:|---:|---:|---:|
| test | all_observed_sources | AD | 28 | 0.6459 | 0.7186 | 0.0727 |
| test | all_observed_sources | ALL | 61 | 0.6318 | 0.7281 | 0.0963 |
| test | all_observed_sources | CN | 33 | 0.6198 | 0.7362 | 0.1164 |
| test | four_shot | AD | 1 | 0.4811 | 0.5786 | 0.0974 |
| test | four_shot | ALL | 22 | 0.6842 | 0.9238 | 0.2395 |
| test | four_shot | CN | 21 | 0.6939 | 0.9402 | 0.2463 |
| test | one_shot | AD | 28 | 0.8182 | 0.9200 | 0.1018 |
| test | one_shot | ALL | 61 | 0.8102 | 0.9624 | 0.1522 |
| test | one_shot | CN | 33 | 0.8034 | 0.9984 | 0.1951 |
| val | all_observed_sources | AD | 28 | 0.7329 | 0.8606 | 0.1277 |
| val | all_observed_sources | ALL | 61 | 0.7197 | 0.8418 | 0.1221 |
| val | all_observed_sources | CN | 33 | 0.7085 | 0.8258 | 0.1172 |
| val | four_shot | ALL | 19 | 0.7181 | 0.9552 | 0.2371 |
| val | four_shot | CN | 19 | 0.7181 | 0.9552 | 0.2371 |
| val | one_shot | AD | 28 | 0.8386 | 1.0049 | 0.1663 |
| val | one_shot | ALL | 61 | 0.8432 | 1.0242 | 0.1810 |
| val | one_shot | CN | 33 | 0.8470 | 1.0405 | 0.1935 |

## Protocol definitions

- `one_shot`: first observed shape predicts the final observed shape.
- `four_shot`: first four observed shapes independently predict the final shape; predictions are averaged. Only subjects with at least five scans contribute.
- `all_observed_sources`: supplementary mean of every available earlier source predicting the final shape.

