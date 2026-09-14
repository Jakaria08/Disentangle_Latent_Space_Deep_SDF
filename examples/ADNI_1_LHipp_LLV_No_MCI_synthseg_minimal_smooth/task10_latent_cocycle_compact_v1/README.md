# Latent-preserving compact cocycle objective

Task10 is a new, isolated follow-up to task9. It tests whether the direct latent
cocycle can use five or six meaningful optimization groups while retaining the latent
supervision that task9 showed was necessary for LAMM. It does not modify task3, task8,
task9, their checkpoints, or their results.

The network is unchanged:

```text
Phi(z,s,t,d) = z + (t-s) [v_CN(z,s,t) + d v_AD(z,s,t)]
```

It always transports a 128-dimensional latent. The decoder remains frozen but
differentiable when decoded mesh losses are evaluated. No ODE, attention, mesh-space
transport, autoencoder fitting, or test-set selection is introduced.

## Why these objectives

Task9 established three useful facts:

1. Removing endpoint and sequence latent supervision damaged LAMM semigroup and inverse
   consistency by roughly 40 percent.
2. Removing the sequence-slope term was not identified as the cause of that damage.
3. Removing the population-trend terms improved the independently evaluated group-rate
   error in every representation, although other terms had been removed at the same time.

Task10 therefore restores both latent prediction losses, retains volume and individual-rate
supervision, and isolates the slope and population-trend removals.

## Objective definitions

Let each symbol below denote the same raw loss and weight used in task9. The six compact
groups are

```text
L_endpoint = 1.00 L_real_vertex + 0.25 L_real_latent

L_sequence = 0.25 L_sequence_vertex + 0.0625 L_sequence_latent

L_composition = 0.10 L_observed_semigroup
              + 0.10 L_virtual_semigroup
              + 0.05 L_inverse
              + 0.10 L_sequence_semigroup

L_volume = 0.10 L_volume_raw
L_rate   = 0.10 L_rate_raw

L_population = 0.02 L_group_rate + 0.02 L_disease_gap
```

The tested objectives are:

```text
L_compact6 = L_endpoint + L_sequence + L_composition
           + L_volume + L_rate + L_population

L_compact5 = L_endpoint + L_sequence + L_composition
           + L_volume + L_rate
```

`compact6` is exactly `full13 - 0.05 L_slope`. It has six top-level groups and 12 active
raw constraints. `compact5` additionally removes group-rate and disease-gap supervision;
it has five groups and 10 active raw constraints. All 13 raw values are still computed and
logged in every arm so matched stochastic execution and omitted-term diagnostics remain
possible. This experiment reduces the optimization objective; it is not intended to reduce
per-step wall time.

## Experiment matrix

There are 33 new jobs:

| arm | representations | seeds | jobs | purpose |
|---|---|---:|---:|---|
| `compact6_standard` | all four | 42, 43, 44 | 12 | isolate slope removal |
| `compact5_standard` | all four | 42, 43, 44 | 12 | additionally isolate population trend |
| `full13_pca_low` | PCA | 42, 43, 44 | 3 | matched PCA LR control |
| `compact6_pca_low` | PCA | 42, 43, 44 | 3 | compact6 with corrected PCA LR |
| `compact5_pca_low` | PCA | 42, 43, 44 | 3 | compact5 with corrected PCA LR |

The standard profile exactly matches task9: warm up from `1e-5` to `3e-4` over 128 steps,
then cosine decay to `3e-5`. The PCA-low profile warms from `1e-5` to `1e-4` and decays to
`1e-5`. The lower peak is based on the successful task8 PCA low-LR arm. Its full13 control
prevents learning-rate effects from being attributed to loss reduction.

The completed task9 `full13` jobs are read-only standard-schedule controls. Reusing them
avoids 12 redundant jobs. The comparison requires all task9 controls to exist and pairs
them with task10 jobs by representation and seed.

See [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) for hypotheses and adoption rules and
[RUNBOOK.md](RUNBOOK.md) for commands.
