# BrainODE independent comparison

This report evaluates BrainODE, no-change, V1, anchored V4, and selected E3 on the exact same locked strict CN/AD test pairs with one structure-specific PCA-150 decoder and one metric implementation. E3 latest remains exploratory.

## Contract status

- Overall: **pass**
- Strict no-MCI and subject/scan-disjoint splits: passed.
- BrainODE training did not load test data: passed.
- BrainODE archives and PCA model exactly match the comparison models: passed.
- BrainODE primary inference is one case at a time, matching its previous evaluator.
- The inherited attention operates across cases in a batch; see `brainode_batch_context_audit.json` for the measured sensitivity.

## Locked test overview

| Structure | Method | Dx | Vertex MAE (mm) | Volume rel. error | Log-rate error | Predicted mm³/year | Observed mm³/year |
|---|---|---:|---:|---:|---:|---:|---:|
| Left hippocampus | No change | CN | 0.1572 | 0.0357 | 0.0153 | 0.0 | -44.6 |
| Left hippocampus | No change | AD | 0.1679 | 0.0552 | 0.0575 | 0.0 | -156.1 |
| Left hippocampus | V1 | CN | 0.1552 | 0.0326 | 0.0146 | -4.9 | -44.6 |
| Left hippocampus | V1 | AD | 0.1675 | 0.0545 | 0.0568 | -3.0 | -156.1 |
| Left hippocampus | Anchored V4 | CN | 0.1552 | 0.0326 | 0.0146 | -4.9 | -44.6 |
| Left hippocampus | Anchored V4 | AD | 0.1663 | 0.0523 | 0.0548 | -11.9 | -156.1 |
| Left hippocampus | E3 selected | CN | 0.1534 | 0.0253 | 0.0127 | -19.0 | -44.6 |
| Left hippocampus | E3 selected | AD | 0.1658 | 0.0500 | 0.0529 | -18.1 | -156.1 |
| Left hippocampus | BrainODE | CN | 0.1487 | 0.0209 | 0.0118 | -44.4 | -44.6 |
| Left hippocampus | BrainODE | AD | 0.1630 | 0.0446 | 0.0485 | -40.6 | -156.1 |
| Left hippocampus | E3 latest (exploratory) | CN | 0.1570 | 0.0344 | 0.0163 | -61.5 | -44.6 |
| Left hippocampus | E3 latest (exploratory) | AD | 0.1586 | 0.0339 | 0.0388 | -114.8 | -156.1 |
| Left lateral ventricle | No change | CN | 0.4405 | 0.1045 | 0.0435 | 0.0 | 809.6 |
| Left lateral ventricle | No change | AD | 0.3784 | 0.0811 | 0.0950 | 0.0 | 2233.8 |
| Left lateral ventricle | V1 | CN | 0.3832 | 0.0518 | 0.0256 | 640.4 | 809.6 |
| Left lateral ventricle | V1 | AD | 0.3582 | 0.0639 | 0.0759 | 604.7 | 2233.8 |
| Left lateral ventricle | Anchored V4 | CN | 0.3832 | 0.0518 | 0.0256 | 640.4 | 809.6 |
| Left lateral ventricle | Anchored V4 | AD | 0.3508 | 0.0510 | 0.0592 | 1831.3 | 2233.8 |
| Left lateral ventricle | E3 selected | CN | 0.4148 | 0.0843 | 0.0360 | 214.1 | 809.6 |
| Left lateral ventricle | E3 selected | AD | 0.3669 | 0.0710 | 0.0840 | 301.4 | 2233.8 |
| Left lateral ventricle | BrainODE | CN | 0.4046 | 0.0673 | 0.0304 | 746.2 | 809.6 |
| Left lateral ventricle | BrainODE | AD | 0.3543 | 0.0581 | 0.0700 | 859.3 | 2233.8 |
| Left lateral ventricle | E3 latest (exploratory) | CN | 0.4290 | 0.0980 | 0.0390 | 1339.9 | 809.6 |
| Left lateral ventricle | E3 latest (exploratory) | AD | 0.3428 | 0.0437 | 0.0499 | 2140.2 | 2233.8 |

## Primary winners on all test pairs

- Left hippocampus: vertex_coordinate_mae_mm = BrainODE (0.15145); volume_relative_error = BrainODE (0.025389); log_volume_rate_abs_error = BrainODE (0.018829); local_normal_rate_mae_mm_per_year = BrainODE (0.15904)
- Left lateral ventricle: vertex_coordinate_mae_mm = Anchored V4 (0.37727); volume_relative_error = Anchored V4 (0.051659); log_volume_rate_abs_error = Anchored V4 (0.031772); local_normal_rate_mae_mm_per_year = Anchored V4 (0.34892)

## Interpretation constraints

- Test results compare already selected models; they must not be used to choose a new checkpoint.
- Primary claims use `method_role=primary`; E3 latest is exploratory only.
- Positive `improvement_mean` in the bootstrap table favors `current_method`.
- Raw-mesh volume is an observed target only. Every model prediction is decoded through the same fixed PCA-150 representation.
- BrainODE batch-context dependence is an architectural limitation and should be reported with its performance.

