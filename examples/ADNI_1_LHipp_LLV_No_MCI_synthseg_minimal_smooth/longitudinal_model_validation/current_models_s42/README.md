# Independent longitudinal model validation

This report uses one evaluator for no-change, V1, anchored V4, E3 selected, and E3 latest sensitivity. BrainODE is not yet trained.

## Contract status

- Overall: **pass**
- Strict CN/AD and disjoint subject/scan splits: passed for both structures.
- Anchored V4 contains an exact frozen copy of its selected V1: passed.
- E3 selected is primary; E3 latest is exploratory and must not replace it based on test results.

## Locked test overview

| Structure | Method | Diagnosis | Vertex MAE (mm) | Volume rel. error | Log-rate error | Predicted mm³/year | Observed mm³/year |
|---|---|---:|---:|---:|---:|---:|---:|
| Left hippocampus | No change | CN | 0.1572 | 0.0357 | 0.0153 | 0.0 | -44.6 |
| Left hippocampus | No change | AD | 0.1679 | 0.0552 | 0.0575 | 0.0 | -156.1 |
| Left hippocampus | V1 | CN | 0.1552 | 0.0326 | 0.0146 | -4.9 | -44.6 |
| Left hippocampus | V1 | AD | 0.1675 | 0.0545 | 0.0568 | -3.0 | -156.1 |
| Left hippocampus | Anchored V4 | CN | 0.1552 | 0.0326 | 0.0146 | -4.9 | -44.6 |
| Left hippocampus | Anchored V4 | AD | 0.1663 | 0.0523 | 0.0548 | -11.9 | -156.1 |
| Left hippocampus | E3 selected | CN | 0.1534 | 0.0253 | 0.0127 | -19.0 | -44.6 |
| Left hippocampus | E3 selected | AD | 0.1658 | 0.0500 | 0.0529 | -18.1 | -156.1 |
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
| Left lateral ventricle | E3 latest (exploratory) | CN | 0.4290 | 0.0980 | 0.0390 | 1339.9 | 809.6 |
| Left lateral ventricle | E3 latest (exploratory) | AD | 0.3428 | 0.0437 | 0.0499 | 2140.2 | 2233.8 |

## Interpretation constraints

- Primary claims must use `method_role=primary` rows only.
- The E3 latest checkpoint is a validation sensitivity analysis; its test metrics are exploratory.
- All predictions and target vertices use the same structure-specific PCA-150 decoder. Raw-mesh volume is also reported separately.
- Subject-bootstrap intervals are in `paired_subject_bootstrap.csv`.

