# PCA128 voxel-cognition BrainODE experiment

This experiment implements the closest defensible BrainODE cognition-estimator
ablation for the current ADNI left-hippocampus cohort. It is deliberately
limited to PCA128; SpiralNet and Adaptive Spiral are not used.

## What is BrainODE-like

- A fixed train-only PCA representation is evolved by a diagnosis-conditioned
  neural ODE with RK4.
- The vector field has the released BrainODE Q/K/V form, evaluated with one-case
  semantics so unrelated subjects cannot attend to one another.
- A 3-D CNN receives only a solid voxelized anatomical shape and emits a
  continuous condition in `[0,1]`.
- During voxel-feedback inference, each predicted PCA mesh is decoded in mm,
  voxelized on a shared train-bounded grid, scored by the CNN, and that score is
  used for the next RK4 substep.
- Fixed-condition and cognition-estimator-without-pseudo-sampling ablations,
  forward/backward prediction, irregular horizon bins, one/two/four-shot
  prediction, condition injectivity, volume/surface metrics, calibration, and
  subject bootstrap intervals are reported.

## What cannot match the converter version of BrainODE

Every subject is diagnosis-stable CN or AD; MCI is absent and there are zero
CN-to-AD converters. The estimator is therefore trained with observed
cross-sectional CN/AD anatomy labels. No intermediate pseudo-cognition targets
are generated. Its output means **AD-like anatomical probability**, not measured
cognition, time to conversion, or probability that a CN subject will convert.

The ODE is optimized with the fixed observed subject condition and latent
trajectory MSE. CNN feedback is introduced only at inference. This corresponds
to the paper's cognition-estimator-without-pseudo-sampling ablation. Calling it
the paper's full converter model would be incorrect.

Under one-case semantics, the attention matrix is 1 x 1 and its softmax is
exactly one. The implementation therefore vectorizes the mathematically
equivalent value path for speed. Q/K parameters remain checkpoint-compatible
with the released architecture but cannot affect a singleton output. This also
means the model avoids the released-code behavior in which unrelated subjects
placed in one batch can attend to each other.

## Data and isolation

- Anatomy: left hippocampus only.
- Representation: the first 128 dimensions of the frozen train-only PCA-150
  basis; no PCA refit.
- CNN input: 32 x 32 x 32 solid masks from raw correspondence meshes in physical
  mm. Bounds and pitch are fitted from training meshes only and reused unchanged
  for validation and test.
- CNN loss: subject- and class-balanced BCE, preventing subjects with more visits
  from dominating.
- Calibration: one temperature fitted to validation logits only.
- Neither training stage opens the test archive or test voxel cache.

Voxel counts are not used as the reported anatomical volume. All volume and
surface outcomes are computed from frozen PCA-decoded meshes in physical units.

## Run

First run the two contract suites and a no-write voxel diagnostic:

```bash
/home/jakaria/anaconda3/envs/pytorch_geo/bin/python \
  examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task3_latent_flow_128_v1/tests/verify_contracts.py

/home/jakaria/anaconda3/envs/pytorch_geo/bin/python \
  examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task3_latent_flow_128_v1/tests/verify_cognition_contracts.py

/home/jakaria/anaconda3/envs/pytorch_geo/bin/python \
  examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task3_latent_flow_128_v1/scripts/prepare_cognition_voxels.py \
  --resolution 32 --padding-voxels 2 --workers 2 --dry-run
```

The resumable end-to-end GPU command is:

```bash
examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task3_latent_flow_128_v1/scripts/run_brainode_cognition_pipeline.sh 1 42
```

It prepares masks if absent, trains/selects the CNN and ODE without test access,
evaluates validation and test once, evaluates the existing PCA direct cocycle
with the same evaluator contract, and writes paired comparison tables.

For a bounded CPU-safe training gate after masks exist:

```bash
/home/jakaria/anaconda3/envs/pytorch_geo/bin/python \
  examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task3_latent_flow_128_v1/scripts/train_brainode_cognition.py \
  --config examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task3_latent_flow_128_v1/configs/pca128_brainode_cognition_s42.json \
  --device cpu --dry-run
```

## Primary comparison

The primary comparator is the selected PCA128 direct C4 run:

`/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task3_latent_flow_128_v1/training/pca128/direct_c4/pca128_direct_c4_s42`

The comparison includes no-change, fixed-label BrainODE, voxel-feedback
BrainODE, and PCA cocycle across decoded surface error, raw-mesh end-to-end
error, volume/rate error, horizons, forward/backward sequences, consistency
defects, and paired subject-bootstrap differences. Negative paired error
differences mean BrainODE is better than PCA cocycle.

The PCA cocycle run is marked complete but stopped at epoch 37 of 180 after its
early-stopping logic selected the feasible epoch-7 checkpoint. The comparison
always uses its explicit `best.pt`; it does not use `latest.pt` or infer a
checkpoint by filename.
