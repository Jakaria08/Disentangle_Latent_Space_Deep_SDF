# End-to-end direct LAMM surface cocycle

This is an isolated, non-ODE longitudinal surface-flow experiment. It does **not** train a
LAMM autoencoder, export latents, freeze a decoder, or fit a second temporal model.

The transported state is the mesh itself:

```text
Phi(X,s,t,d) = X + (t-s) [V_CN(X,s,t) + d V_AD-residual(X,s,t)]
```

The velocity network is one jointly trained Conditional LAMM:

```text
source mesh [B,2746,3]
  -> template-centred coordinates
  -> 43 coarse + 86 fine non-shared LAMM region tokens
  -> time/disease-conditioned MLPMixer encoder
  -> internal 128D or 256D multiscale bottleneck
  -> conditioned residual MLP
  -> time/disease-conditioned MLPMixer decoder
  -> coarse velocity + fine residual velocity
  -> CN field + diagnosis * AD residual field
  -> source mesh + elapsed years * field
```

The source mesh bypass gives exact identity when source and target age are equal. The
cocycle/composition property is learned with observed, virtual, and local composition losses,
matching the direct Spiral and Adaptive-Spiral experiments.

## Controlled variants

- `lamm_direct_c4_z128_s42.json`: 128D, split `[64,64]`.
- `lamm_direct_c4_z256_equal_s42.json`: 256D, controlled equal split `[128,128]`.
- `lamm_direct_c4_z256_fine_s42.json`: 256D, fine-heavy split `[96,160]`.

All variants use the selected latest-LAMM 43+86 partition. The selected checkpoint is read
only for immutable region membership; none of its learned weights are loaded or frozen.

## Data and scientific isolation

The already validated fixed-topology direct-mesh cache is read in place. No surface data is
copied or changed. Training loads only train and validation splits; test is owned by the
separate evaluator. Outputs default to:

```text
/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task7_direct_mesh_cocycle_lamm_v1
```

See `RUNBOOK.md` for commands and `VALIDATION.md` for the completed verification record.

