# End-to-end direct LAMM mesh cocycle

This isolated experiment trains one non-ODE surface-flow network end to end. The mesh is
the transported state; there is no separately trained autoencoder, frozen decoder, or
required low-dimensional representation:

```text
Phi(X,s,t,d) = X + (t-s) [V_CN(X,s,t) + d V_AD-residual(X,s,t)]
```

The elapsed-time residual gives bit-exact identity at `s=t`. Cocycle behavior is learned by
applying the same embedded network to direct and composed intervals, with observed,
virtual, and local composition losses plus inverse, fitted-velocity, anatomy, topology,
volume, and disease-trend terms.

## Shared network

```text
aligned source mesh [B,2746,3]
  -> subtract train-only template; divide by train-only coordinate scale
  -> non-shared latest-LAMM tokenizers: 43 coarse + 86 fine tokens
  -> eight conditioned MLPMixer encoder blocks
  -> one of the three internal-state variants below
  -> six conditioned MLPMixer decoder blocks
  -> non-shared coarse and fine six-channel regional heads
  -> sum scales; split channels into CN field + AD-residual field
  -> source mesh + elapsed years * diagnosis-specific field
```

The condition embedding matches the direct Spiral setup. It contains normalized source and
target age, signed and absolute age difference, midpoint, `log1p` elapsed time, two Fourier
age frequencies, and a learned diagnosis embedding. A 32D condition vector is injected by
FiLM into both the token and channel mixers of every encoder/decoder block. The global
variants also concatenate it into the latent flow and apply it in each latent residual
block; the regional-token variant applies it in two additional token-flow mixer blocks.

## Three final experiments

1. `lamm_global_c4_z256_s42.json`: 256D internal global state, coarse/fine allocation
   `[96,160]`, 256-wide two-block conditioned residual flow; 33,431,224 parameters.
2. `lamm_global_c4_z384_s42.json`: 384D internal global state, allocation `[128,256]`,
   384-wide two-block conditioned residual flow; 36,771,640 parameters.
3. `lamm_token_c4_s42.json`: no flattened global vector. It retains all `129 x 256`
   regional values through a two-block conditioned token flow; 28,554,706 parameters.

The 256D/384D values are internal architectural widths, not exported or constrained
latents. The third experiment explicitly tests removing that global bottleneck.

All variants use the exact 43+86 partition stored in the selected latest-LAMM checkpoint.
Only immutable region membership is read; no learned checkpoint weight is loaded or frozen.

## Central comparison

`scripts/evaluate_all.py` evaluates the three new runs, selected Spiral and Adaptive-Spiral
direct mesh runs, and the current 128D LAMM latent flow. It uses the same first-to-last
held-out protocol and reports correspondence error, ASSD, HD95, Chamfer, normals, flips,
volume/rate trends, cocycle/inverse defect, and no-change ratios.

For instantaneous velocity, direct methods use `V(X,a,a,d)`. The latent model uses
`V(z,a,a,d)` converted to years and mapped through the frozen decoder Jacobian-vector
product. All are then compared in surface `mm/year` to the reliability-weighted velocity
fit from repeated observed visits. That reference is an observed longitudinal estimate,
not direct physical ground truth.

See `RUNBOOK.md` for the three training commands, Optuna search, and centralized evaluation.
