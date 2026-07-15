# Large Strict No-MCI Left Task2 SIREN Pipeline

This folder defines two SIREN auto-decoder experiments for the large strict no-MCI left hippocampus dataset:

- `siren_naisr_5x512_warmstart_no_skip`: same architecture as the previous successful SIREN/eikonal run, decoder-only warm start from the old `best.pth`.
- `siren_naisr_5x512_latent_skip_warmstart`: one latent skip at layer 3, partial decoder-only warm start from the same old checkpoint.

The dataset stays outside the repo:

```text
/home/jakaria/ADNI/ADNI_1_GO_Large/left_hippocampus_strict_no_mci
```

The repo only stores configs, manifests, labels/metadata, scripts, logs, and checkpoints.

## Build And Validate Inputs

Build the Task2 manifest:

```bash
examples/ADNI_1_L_No_MCI_large_strict_left/task2_representations/bin/00_build_manifest.sh
```

After all SDF `.npz` files are generated, require complete SDF coverage:

```bash
examples/ADNI_1_L_No_MCI_large_strict_left/task2_representations/bin/00_build_manifest.sh --require-sdf
```

Run a quick input audit:

```bash
examples/ADNI_1_L_No_MCI_large_strict_left/task2_representations/bin/01_audit_inputs_quick.sh
```

Run the full input audit:

```bash
examples/ADNI_1_L_No_MCI_large_strict_left/task2_representations/bin/02_audit_inputs_full.sh
```

The quick/full audits should be run only after the manifest exists. The full audit should be run only after SDF generation is complete.

## Train

Train the no-skip baseline:

```bash
DEVICE=cuda:0 examples/ADNI_1_L_No_MCI_large_strict_left/task2_representations/bin/10_train_no_skip.sh
```

Train the latent-skip model:

```bash
DEVICE=cuda:1 examples/ADNI_1_L_No_MCI_large_strict_left/task2_representations/bin/11_train_latent_skip.sh
```

Resume either run from its latest checkpoint:

```bash
DEVICE=cuda:0 examples/ADNI_1_L_No_MCI_large_strict_left/task2_representations/bin/10_train_no_skip.sh --resume latest
DEVICE=cuda:1 examples/ADNI_1_L_No_MCI_large_strict_left/task2_representations/bin/11_train_latent_skip.sh --resume latest
```

Important: `--resume` restores the experiment checkpoint, including the latent table and optimizer. The initial warm-start checkpoint is used only when starting a fresh run.

## Fit Latents

After training completes, export optimized latents for all train/val/test scans:

```bash
DEVICE=cuda:0 examples/ADNI_1_L_No_MCI_large_strict_left/task2_representations/bin/20_fit_latents_no_skip.sh
DEVICE=cuda:1 examples/ADNI_1_L_No_MCI_large_strict_left/task2_representations/bin/21_fit_latents_latent_skip.sh
```

Outputs go under each experiment folder:

```text
inr/<experiment>/latents/per_scan/*.npy
inr/<experiment>/latents/fit_metrics/*.json
inr/<experiment>/latents/inr_latents.csv
inr/<experiment>/latents/train_latents.npz
inr/<experiment>/latents/val_latents.npz
inr/<experiment>/latents/test_latents.npz
```

## Memory Notes

The full SDF dataset is expected to be about 20 GB compressed on disk and larger when loaded as arrays. The configs set:

```json
"load_dataset_into_ram": true
```

With 128 GB RAM, one run should fit comfortably. Running both experiments at once is intended to be feasible, but monitor host RAM because each run preloads its own SDF arrays and surface-triangle data. If memory pressure appears, set `load_dataset_into_ram` to `false` in one or both configs, or run the two experiments sequentially.

## Architecture Notes

The no-skip model keeps:

```json
"dims": [512, 512, 512, 512, 512],
"latent_in": []
```

The skip model uses:

```json
"dims": [512, 512, 768, 512, 512],
"latent_in": [3]
```

In the existing `siren_decoder.py`, `latent_in: [3]` appends the 256D latent vector before layer 3. Setting the third hidden width to 768 keeps the hidden activation width effectively at 512 around that skip. The warm-start loader copies the old weights exactly where shapes match and copies the old layer-3 weights into the first 512 columns while zeroing the new latent-skip columns.
