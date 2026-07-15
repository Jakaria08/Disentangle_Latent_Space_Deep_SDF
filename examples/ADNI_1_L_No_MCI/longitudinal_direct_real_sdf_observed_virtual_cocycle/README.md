# No-MCI direct chronological-age flow

This experiment is isolated from the previous longitudinal trainer. It keeps
the validated chronological-age coordinate
`(continuous_age_years - 57) / 34`, the frozen Task-2 DeepSDF decoder, and the
frozen 256-dimensional latent attached to every real scan.

## What is trained

Only `DirectAgeFlow` is optimized:

`Phi(z_s, s, t, c) = z_s + (t - s) G(z_s, s, t, c)`

Here `s` and `t` are normalized chronological ages and `c` is `0` for CN or
`1` for AD. There is no numerical ODE solver.

For visits at ages 70, 70.5, and 71, the training set contains all forward
pairs: 70 to 70.5, 70.5 to 71, and 70 to 71. The observed-intermediate
cocycle applies only to 70 to 71 because 70.5 is a real interior age.

Exactly three objectives are active:

1. Decode the predicted target latent and match real target-scan SDF samples.
2. Match direct and composed target latents through a real interior age.
3. Match direct and composed target latents through a random unobserved age.

Observed and virtual intermediate latents are constructed during training.
The frozen decoder can turn either latent into a shape for visualization, but
they have no target-SDF loss because no ground truth exists at those ages.

## Intentionally disabled in this phase

Real-target latent regression, shape-level cocycle loss, backward/closure
losses, generator and velocity regularizers, zero-displacement loss, eikonal
loss, latent-code regularization, extrapolation loss, and disentanglement are
all false in `specs.json`.

## Commands

Run the complete input-contract and one-batch CPU forward check:

```bash
python train_deep_sdf_longitudinal_direct_flow.py \
  -e examples/ADNI_1_L_No_MCI/longitudinal_direct_real_sdf_observed_virtual_cocycle \
  --validate-only
```

Train:

```bash
python train_deep_sdf_longitudinal_direct_flow.py \
  -e examples/ADNI_1_L_No_MCI/longitudinal_direct_real_sdf_observed_virtual_cocycle \
  --gpu 0
```

Run a one-epoch GPU smoke test before the full run:

```bash
python train_deep_sdf_longitudinal_direct_flow.py \
  -e examples/ADNI_1_L_No_MCI/longitudinal_direct_real_sdf_observed_virtual_cocycle \
  --gpu 0 --smoke-batches 2
```

Resume:

```bash
python train_deep_sdf_longitudinal_direct_flow.py \
  -e examples/ADNI_1_L_No_MCI/longitudinal_direct_real_sdf_observed_virtual_cocycle \
  --gpu 0 --continue-from latest
```

Evaluate after training:

```bash
python evaluate_deep_sdf_longitudinal_direct_flow.py \
  -e examples/ADNI_1_L_No_MCI/longitudinal_direct_real_sdf_observed_virtual_cocycle \
  --checkpoint best --split val
```

Create an interactive HTML dashboard:

```bash
python visualize_deep_sdf_longitudinal_direct_flow.py \
  -e examples/ADNI_1_L_No_MCI/longitudinal_direct_real_sdf_observed_virtual_cocycle \
  --checkpoint best --split val
```

For decoded 3D examples, evaluate with
`--export-example-meshes --gpu 0`; mesh extraction is intentionally optional
because it is substantially slower than metric evaluation.

Open `visualize_direct_flow.ipynb` after evaluation to inspect training curves,
prediction-versus-no-change error, cocycle error, and dense chronological-age
latent trajectories.
