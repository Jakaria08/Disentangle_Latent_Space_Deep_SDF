# Dense multiresolution single-field SDF

This is an isolated hippocampus experiment. It does not import checkpoints or write into the currently running Compact-style experiment. The only scan-specific representation is a 256-D latent. The decoder and all dense grids are population-shared:

`s(x, z) = D(x, Gamma(x), z)`

`Gamma` concatenates trilinearly interpolated 4-D features from dense grids at `8, 16, 24, 32, 48, 64`. The decoder is a four-layer width-128 Softplus MLP, with the full 256-D latent injected at the input and again at hidden layer 3. It predicts one SDF; there are no global/local branches, gate, threshold, or fusion term.

## Run policy

Use `hippocampus_multires_z256_c2f_noeik.json` as the primary run. The existing NPZs are valid signed DeepSDF samples, but repository `src/PreprocessMesh.cpp` estimates magnitude from the nearest rendered surface point (and a close-range point-plane distance), not exact point-to-triangle distance. The `eik001` configuration is therefore a controlled regularization ablation, selected only if validation ASSD/HD95, topology, normals, and gradient diagnostics improve.

Every persistent writer calls a hard path guard and refuses targets outside `/mnt/bulk10tb`. Use `run_on_bulk.sh` for every command; it places temporary and library cache files on the 10-TB mount and disables Python bytecode on the SSD. Its short `/mnt/bulk10tb/.inr_tmp` path is intentional: longer experiment paths exceed Linux's AF_UNIX socket limit when PyTorch DataLoader workers exchange tensors.

## Exact commands

From the repository root:

```bash
TASK=examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task2_inr_multires_single_field_v1
RUNNER="$TASK/scripts/run_on_bulk.sh"
PRIMARY="$TASK/configs/hippocampus_multires_z256_c2f_noeik.json"
EIK="$TASK/configs/hippocampus_multires_z256_c2f_eik001.json"
```

Validate all manifest paths, subject-disjoint splits, mesh AABB, SDF signs, and an exact-distance sample:

```bash
"$RUNNER" "$TASK/scripts/validate_multires_inputs.py" --config "$PRIMARY" --scans 8 --points-per-scan 2048
```

Run the implementation checker on CPU. It exercises real NPZ sampling, both loss paths and backward passes, exact checkpoint reload, and Marching Cubes. Its validation-only artifacts go to the 10-TB disk:

```bash
"$RUNNER" "$TASK/scripts/validate_implementation.py" --config "$PRIMARY" --eik-config "$EIK" --device cpu
```

Check visible GPUs, then start the primary training. `CUDA_VISIBLE_DEVICES=0` makes the selected physical GPU become logical `cuda:0`, avoiding invalid-device-ordinal errors:

```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=0 "$RUNNER" "$TASK/scripts/train_multires_sdf.py" --config "$PRIMARY" --device cuda:0
```

Resume after interruption:

```bash
CUDA_VISIBLE_DEVICES=0 "$RUNNER" "$TASK/scripts/train_multires_sdf.py" --config "$PRIMARY" --device cuda:0 --resume latest
```

Run the Eikonal ablation separately; never point it at the primary output directory:

```bash
CUDA_VISIBLE_DEVICES=0 "$RUNNER" "$TASK/scripts/train_multires_sdf.py" --config "$EIK" --device cuda:0
```

Validation code fitting runs automatically every 25 epochs. Checkpoints are saved as `latest`, `best_sdf`, and at epochs 500, 1000, 1500, and 1700. The 100/100/100 train/val/test mesh evaluation runs at those same milestone epochs and writes INR and PCA meshes to the corresponding bulk evaluation directory. The validation INR ASSD selects `best_mesh`; test results never select a checkpoint.

Manual full validation and test evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 "$RUNNER" "$TASK/scripts/periodic_evaluate_multires.py" --config "$PRIMARY" --checkpoint best_mesh --device cuda:0 --splits val test --per-split 100
```

Export all 256-D scan latents, fitting validation/test codes with the frozen decoder:

```bash
CUDA_VISIBLE_DEVICES=0 "$RUNNER" "$TASK/scripts/fit_export_multires_latents.py" --config "$PRIMARY" --checkpoint best_mesh --device cuda:0
```

Decode an exported latent archive at the configured `256^3` grid resolution:

```bash
LATENTS=/mnt/bulk10tb/Deep3DComp/synthseg_qc_v1/left_hippocampus/inr_multires_single_field_v1/runs/hippocampus_multires_z256_c2f_noeik/latent_exports/best_mesh/latents.pth
CUDA_VISIBLE_DEVICES=0 "$RUNNER" "$TASK/scripts/reconstruct_multires_meshes.py" --config "$PRIMARY" --checkpoint best_mesh --latents "$LATENTS" --device cuda:0
```

## Fixed training values

- Latent: 256-D, one vector per training scan.
- Shared grid features: 1,695,744 parameters (about 6.47 MiB in FP32).
- Whole decoder including grids: about 1.81 million parameters.
- Sampling per scan per epoch: 8,192 broad samples and 8,192 cell-balanced near-surface samples.
- Batch: 16 scans; loss/backward chunk: all 16 scans. On the available TITAN RTX this measured about 0.47 seconds and 1.99 GiB for one full-sampling chunk; the old two-scan chunk was removed because it severely underused the GPU.
- Schedule: grids 8/16/24 active initially, 32 ramps over epochs 101-200, 48 over 301-400, and 64 over 501-600; total 1,700 epochs.
- Loss: clamped broad L1 + clamped near L1 + latent L2 + small grid L2. Primary Eikonal weight is zero; ablation weight is 0.01 from epoch 201 after a 25-epoch warm-up.
- Reconstruction: `256^3`, chunks of 131,072 queries.
- Periodic surface metrics: 30,000 sampled source points to exact target triangles, ASSD, directional distances, Chamfer-L1/L2, HD95, F-score at 0.5/1.0 mm, volume, normals, adjacency roughness, connected components, Euler number, boundary/nonmanifold edges, and watertightness. At least 95% of each requested split/method must succeed, and confidence intervals resample subjects as clusters so repeated visits are not treated as independent people.
- PCA comparator: train-only PCA basis, but each evaluated target has its own projected PCA coefficient. It is explicitly reported as an oracle reconstruction baseline, not a prediction baseline.

For a lateral-ventricle experiment, create a new manifest/config with its own bulk output directory, structure-specific grid AABB and scaling CSV. The network and scripts are otherwise structure-agnostic.
