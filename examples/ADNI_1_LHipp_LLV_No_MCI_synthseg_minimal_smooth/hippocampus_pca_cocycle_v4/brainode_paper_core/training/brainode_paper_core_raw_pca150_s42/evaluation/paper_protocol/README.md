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
| test | all_observed_sources | AD | 28 | 0.2954 | 0.3155 | 0.0201 |
| test | all_observed_sources | ALL | 61 | 0.2671 | 0.2768 | 0.0097 |
| test | all_observed_sources | CN | 33 | 0.2430 | 0.2439 | 0.0009 |
| test | four_shot | AD | 1 | 0.2905 | 0.3272 | 0.0367 |
| test | four_shot | ALL | 21 | 0.2859 | 0.2932 | 0.0074 |
| test | four_shot | CN | 20 | 0.2857 | 0.2915 | 0.0059 |
| test | one_shot | AD | 28 | 0.3425 | 0.3731 | 0.0306 |
| test | one_shot | ALL | 61 | 0.3261 | 0.3413 | 0.0152 |
| test | one_shot | CN | 33 | 0.3121 | 0.3142 | 0.0021 |
| val | all_observed_sources | AD | 28 | 0.2996 | 0.3253 | 0.0256 |
| val | all_observed_sources | ALL | 61 | 0.2692 | 0.2887 | 0.0195 |
| val | all_observed_sources | CN | 33 | 0.2434 | 0.2576 | 0.0142 |
| val | four_shot | ALL | 19 | 0.2642 | 0.2973 | 0.0331 |
| val | four_shot | CN | 19 | 0.2642 | 0.2973 | 0.0331 |
| val | one_shot | AD | 28 | 0.3554 | 0.3901 | 0.0346 |
| val | one_shot | ALL | 61 | 0.3300 | 0.3572 | 0.0273 |
| val | one_shot | CN | 33 | 0.3084 | 0.3294 | 0.0210 |

## Protocol definitions

- `one_shot`: first observed shape predicts the final observed shape.
- `four_shot`: first four observed shapes independently predict the final shape; predictions are averaged. Only subjects with at least five scans contribute.
- `all_observed_sources`: supplementary mean of every available earlier source predicting the final shape.

