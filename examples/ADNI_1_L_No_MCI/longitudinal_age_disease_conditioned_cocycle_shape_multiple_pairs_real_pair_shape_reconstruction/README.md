# ADNI No-MCI Longitudinal DeepSDF Experiment

This folder adapts immutable ADNI No-MCI Task 1 and Task 2 outputs into a longitudinal DeepSDF experiment. MCI is excluded everywhere in this experiment.

## Immutable sources

- Task 1 manifest: `examples/ADNI_1_L_No_MCI/brainode_comparison_task1_manifest_original/`
- Task 2 representations: `examples/ADNI_1_L_No_MCI/brainode_comparison_task2_representations_original/`
- Original meshes: `/home/jakaria/ADNI/ADNI_1/adni_processed/left_hippocampus_correspondence/minimal_scaled_obj_files`
- Original SDF archives: `/home/jakaria/ADNI/ADNI_1/adni_processed/left_hippocampus_correspondence/sdf_data/SdfSamples/minimal_scaled_obj_files`

These inputs are read only. Preparation and validation in this folder do not overwrite them.

## Variables

- `baseline_age_years`: age at the first retained visit for a subject.
- `elapsed_years`: `months_from_baseline / 12`.
- `continuous_age_years`: `baseline_age_years + months_from_baseline / 12`.
- `continuous_age_norm`: `(continuous_age_years - 57) / (91 - 57)`.
- `label_ad`: diagnosis conditioning variable used by the trainer. `0 = CN`, `1 = AD`.
- `diagnosis`: human-readable diagnosis label. Valid values are exactly `CN` and `AD`.

## Preparation

Run from the repository root:

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python \
  examples/ADNI_1_L_No_MCI/longitudinal_age_disease_conditioned_cocycle_shape_multiple_pairs_real_pair_shape_reconstruction/scripts/prepare_adni_no_mci_longitudinal_inputs.py
```

This generates:

- `metadata/adni_no_mci_longitudinal_records.csv`
- `metadata/adni_no_mci_longitudinal_labels.pt`
- `metadata/preparation_report.json`
- `pretrained_task2_deepsdf/specs.json`
- `pretrained_task2_deepsdf/ModelParameters/best.pth`
- `pretrained_task2_deepsdf/LatentCodes/best.pth`

## Validation

Do not start training unless `metadata/input_validation_report.json` has `status: "pass"`.

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python \
  examples/ADNI_1_L_No_MCI/longitudinal_age_disease_conditioned_cocycle_shape_multiple_pairs_real_pair_shape_reconstruction/scripts/validate_adni_no_mci_longitudinal_inputs.py
```

This validates:

- 727 scans and 244 subjects
- split counts 617 train, 39 validation, 71 test
- diagnoses exactly `CN` and `AD`, with no `MCI`
- mesh path, SDF path, metadata record, and Task 2 per-scan representation for every scan
- one 256D training latent per training scan
- strictly increasing continuous visit time within subjects
- no subject crossing splits
- pair and triplet scan IDs against the metadata
- bridge decoder loadability and bridge/longitudinal decoder architecture match

## Training

Run training only after validation passes:

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python train_deep_sdf_longitudinal.py \
  -e examples/ADNI_1_L_No_MCI/longitudinal_age_disease_conditioned_cocycle_shape_multiple_pairs_real_pair_shape_reconstruction \
  --gpu 0
```

This experiment uses:

- the original ADNI SDF data as `DataSource`
- Task 1 train split for optimization
- Task 1 validation split as `TestSplit` for model selection
- Task 1 test split as `FinalTestSplit` for later locked offline reporting
- age+disease-conditioned flow with `label_ad`
- pretrained Task 2 DeepSDF decoder and pretrained Task 2 training latents for subject-anchor initialization
- two forward cocycle pairs and two backward cocycle pairs per subject
- two real scan pairs per represented subject, both forward and backward
- latent-only real scan pair consistency in the current spec (`RealScanPairReconstructionLambda = 0.0`)
- cocycle and backward cocycle latent consistency
- cocycle shape loss disabled in the current spec (`UseCocycleShapeLoss = false`)
- mixed adjacent/far pair sampling
- zero-displacement regularization
- aligned Chamfer evaluation
- checkpoints every 250 epochs

## Resume

Resume the latest checkpoint:

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python train_deep_sdf_longitudinal.py \
  -e examples/ADNI_1_L_No_MCI/longitudinal_age_disease_conditioned_cocycle_shape_multiple_pairs_real_pair_shape_reconstruction \
  -c latest \
  --gpu 0
```

Resume a specific checkpoint, for example epoch 1000:

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python train_deep_sdf_longitudinal.py \
  -e examples/ADNI_1_L_No_MCI/longitudinal_age_disease_conditioned_cocycle_shape_multiple_pairs_real_pair_shape_reconstruction \
  -c 1000 \
  --gpu 0
```

## Checkpoint selection

- Select the model by validation metrics from checkpoint epochs only.
- The validation split is the Task 1 validation split configured as `TestSplit`.
- Do not select the last training epoch unless it is also the best validation checkpoint.
- Keep the chosen checkpoint fixed before any locked-test reporting.

## Final locked test

- `FinalTestSplit` points to the Task 1 test split.
- Do not use this split for model selection, hyperparameter changes, or debugging.
- After the validation checkpoint is chosen, run the later offline evaluation step once on `FinalTestSplit` and report it as locked test performance.
- If the locked test reveals a problem, return to validation-only development and rerun training; do not keep tuning against the same test result.

## Visualization order

The intended notebook order for later steps is:

1. `visualize_adni_no_mci_original_and_model_speeds.ipynb`
2. `visualize_adni_no_mci_one_shot_forecast.ipynb`

Run the speed notebook first to verify original-data trajectories and model velocity fields before inspecting one-shot forecast examples.

## Expected generated files

Current preparation and validation outputs:

- `metadata/adni_no_mci_longitudinal_records.csv`
- `metadata/adni_no_mci_longitudinal_labels.pt`
- `metadata/preparation_report.json`
- `metadata/input_validation_report.json`
- `pretrained_task2_deepsdf/specs.json`
- `pretrained_task2_deepsdf/train_split.json`
- `pretrained_task2_deepsdf/ModelParameters/best.pth`
- `pretrained_task2_deepsdf/LatentCodes/best.pth`

Training outputs:

- `ModelParameters/*.pth`
- `LatentCodes/*.pth`
- `OptimizerParameters/*.pth`
- `Logs.pth`
- `train.log`
- `terminal.log`
- `TensorBoard/`

MCI is excluded from preparation, validation, training, evaluation, and visualization for this experiment.
