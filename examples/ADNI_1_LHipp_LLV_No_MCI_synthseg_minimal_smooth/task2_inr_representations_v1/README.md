# Phase 2/3 shared-grid SDF

This experiment is a medical-shape adaptation of Compact-SDF with one flow-ready
256-D vector per scan. It preserves the established pretrained global
skip-SIREN and adds a smaller directly supervised local skip-SIREN conditioned
on one population-shared feature grid.

## Default representation

- Per-scan state: one 256-D latent vector.
- Global branch: the existing `[512,512,768,512,512]` skip-SIREN, loaded
  strictly from the configured checkpoint.
- Local branch: three 128-wide SIREN hidden layers with an input skip at hidden
  layer 2.
- Shared grid: `32 x 32 x 32 x 16`, or 524,288 trainable values shared across
  every scan.
- Grid ROI: `[-0.65,-0.95,-0.68]` through `[0.65,0.95,0.68]`.
- Final SDF:

  ```text
  surface_weight = exp(-(global_sdf / 0.05)^2)
  weight = surface_weight * smooth_roi_taper
  final_sdf = (1-weight)*global_sdf + weight*local_sdf
  ```

The smooth ROI taper is 0.04 normalized units wide. The final field is
continuous at the ROI boundary. Trilinear grid interpolation is continuous but
has piecewise, rather than globally continuous, first derivatives.

Only global decoder weights are transferred. The QC SynthSeg meshes receive
new randomly initialized latent codes; old scan codes can be tested later by
setting `reuse_matching_source_latents`, but that is disabled in the primary
experiment.

## Continuous SDF samples

No occupancy conversion or synthetic normal-offset target is used by default.
The signed magnitudes in each QC NPZ archive are trained as continuous values.

Per scene and epoch:

| Branch | Sampling | Count |
|---|---|---:|
| Global | balanced samples with `abs(SDF) <= 0.1` | 4,096 |
| Global | positive, complete archive | 2,048 |
| Global | negative, complete archive | 2,048 |
| Local | grid-cell-balanced `abs(SDF) <= 0.03` | 4,096 |
| Local | grid-cell-balanced positive `abs(SDF) <= 0.1` | 2,048 |
| Local | grid-cell-balanced negative `abs(SDF) <= 0.1` | 2,048 |

That is 8,192 points per branch and 16,384 points per scene. With the default
two-scene GPU chunk, each forward/backward chunk processes 32,768 SDF targets.
The effective scene batch is 16, accumulated with exact scene-count weighting.

## Training schedule

| Epochs | Stage | Trainable state |
|---:|---|---|
| 1-100 | global adaptation | global SIREN and new train codes |
| 101-200 | local warm-up | local SIREN and shared grid |
| 201-1700 | joint fine-tuning | both branches, grid, and train codes |

The global network and latent table are both frozen during local warm-up. The
fusion coefficient ramps from 0 to 1 during joint epochs 201-225.

The primary loss uses directly supervised global/local continuous-SDF L1,
fused SDF L1 during joint training, latent L2 regularization, and sampled grid
feature L2 regularization. Extra normal, eikonal, sign, and TV objectives are
not part of the primary run.

## Validation and checkpoint rules

Quick validation runs every 25 epochs on a fixed, seeded, diagnosis-stratified
set of 24 eligible validation scans. The decoder/grid are frozen and only a new
validation code is fitted using disjoint fitting and held-out NPZ samples.

- Global stage reports/selects global SDF error only.
- Local warm-up reports local branch error but cannot select the final model.
- Joint training selects `best_sdf.pth` using fused validation SDF error.
- Test metrics never select a checkpoint.

Combined resume checkpoints are stored in `checkpoints/`. Compatible model,
optimizer, and latent files are also stored under `ModelParameters/`,
`OptimizerParameters/`, and `LatentCodes/`. Named snapshots are written at
epochs 100, 200, 500, 1000, 1500, and 1700. `latest.pth` is atomically replaced
each epoch and includes model/grid state, latents, optimizer, RNG state,
configuration, scan ordering, and transfer provenance.

## Periodic mesh/PCA evaluation

At epochs 500, 1000, 1500, and the final epoch 1700, training invokes
`periodic_evaluate_shared_grid.py` on fixed random subsets of 100 train, 100
validation, and 100 test scans. The selection is saved once in
`periodic_evaluation/evaluation_subset.json` and reused across epochs.

- Train scans use their learned codes.
- Validation/test scans fit a new code for 500 steps with the decoder frozen.
- Fused INR meshes are generated at 256 vertices per full-domain axis.
- INR meshes are converted to millimetres with the fixed inverse global
  normalization; no test-set alignment is fitted.
- The existing train-only PCA-150 model reconstructs the same scans.
- Metrics include ASSD, bidirectional Chamfer-L1, squared Chamfer-L2, HD95,
  F-score at 0.5/1.0 mm, volume errors, components, watertightness, winding,
  Euler number, boundary/non-manifold edges, bootstrap intervals, and paired
  INR-minus-PCA differences.
- PCA vertex RMSE is reported when correspondence is available; it is not used
  for INR meshes without vertex correspondence.

`best_mesh.pth` is selected only by validation INR ASSD. Periodic test results
are explicitly monitor-only.

## Exact commands

Configuration:

```bash
CONFIG=examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task2_inr_representations_v1/configs/hippocampus_shared_grid_32x16_z256.json
```

Read-only CPU smoke test:

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python \
  examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task2_inr_representations_v1/scripts/train_shared_grid_sdf.py \
  --config "$CONFIG" --device cpu --smoke-test
```

Start training on GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 /home/jakaria/anaconda3/envs/inr_sdf/bin/python \
  examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task2_inr_representations_v1/scripts/train_shared_grid_sdf.py \
  --config "$CONFIG" --device cuda:0
```

Resume:

```bash
CUDA_VISIBLE_DEVICES=0 /home/jakaria/anaconda3/envs/inr_sdf/bin/python \
  examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task2_inr_representations_v1/scripts/train_shared_grid_sdf.py \
  --config "$CONFIG" --device cuda:0 --resume latest
```

Standalone validation of the best SDF checkpoint, including PCA comparison:

```bash
CUDA_VISIBLE_DEVICES=0 /home/jakaria/anaconda3/envs/inr_sdf/bin/python \
  examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task2_inr_representations_v1/scripts/periodic_evaluate_shared_grid.py \
  --config "$CONFIG" --checkpoint best_sdf --splits val --per-split 100 \
  --resolution 256 --latent-steps 500 --compare-pca --device cuda:0
```

Final held-out test of the validation-selected mesh checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 /home/jakaria/anaconda3/envs/inr_sdf/bin/python \
  examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task2_inr_representations_v1/scripts/periodic_evaluate_shared_grid.py \
  --config "$CONFIG" --checkpoint best_mesh --splits test --per-split 100 \
  --resolution 256 --latent-steps 500 --compare-pca --device cuda:0
```

Export all 256-D codes for the later flow model:

```bash
CUDA_VISIBLE_DEVICES=0 /home/jakaria/anaconda3/envs/inr_sdf/bin/python \
  examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task2_inr_representations_v1/scripts/fit_export_shared_grid_latents.py \
  --config "$CONFIG" --checkpoint best_mesh --splits train val test --device cuda:0
```

Training can be run with `--skip-periodic-evaluation` for debugging, but that
does not satisfy the primary experiment protocol.

## Grid sizes and runtime

- Feature grid: `32^3 x 16`, approximately 2 MiB in float32 before optimizer
  state.
- Reconstruction grid: `256^3`, 16,777,216 full-domain SDF queries per shape.
- Normalized spacing: `2/255 = 0.007843`.
- Approximate physical spacing for this dataset: 0.193 mm.
- DataLoader workers: 12.
- Effective scene batch: 16; GPU scene chunk: 2.

The expected single-GPU training time is roughly 35-50 hours, plus several
hours for each 300-shape periodic evaluation. Actual time should be estimated
from the first complete epoch and the first reconstruction.

## Lateral-ventricle reuse

The network, sampler, trainer, latent fitter, and evaluator contain no
hippocampus-specific architecture logic. A lateral-ventricle configuration must
provide its own manifest, pretrained global checkpoint, grid AABB, output
directory, PCA paths, and normalization CSV. The same latent/grid/network
defaults can then be tested without copying implementation code.
