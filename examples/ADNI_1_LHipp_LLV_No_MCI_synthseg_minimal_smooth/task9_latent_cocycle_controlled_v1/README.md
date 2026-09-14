# Controlled latent cocycle loss reduction

This task tests how far the direct latent cocycle objective can be simplified without
changing the network, data, optimizer batches, validation protocol, or random seed. It is
isolated from task3 and task8. It reads their frozen representation archives and writes only
under `/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task9_latent_cocycle_controlled_v1`.

The transported state is always a 128-D latent vector:

```text
Phi(z,s,t,d) = z + (t-s) [v_CN(z,s,t) + d v_AD(z,s,t)]
```

There is no ODE, attention model, direct-mesh network, trainable decoder, or autoencoder
training. The original task3 post-activation network is fixed for every run.

## Objective ladder

Every arm computes the same 13 raw quantities, which keeps stochastic sampling identical.
Only the weighted optimization sum changes.

| set | top-level terms | omitted relative to `full13` |
|---|---|---|
| `full13` | the original 13 separately weighted terms | none |
| `lean6` | vertex, sequence vertex, composition, volume, rate, group trend | latent endpoint, latent sequence, sequence slope |
| `lean5` | vertex, sequence vertex, composition, rate, group trend | `lean6` minus endpoint volume |
| `lean4` | vertex, sequence vertex, composition, rate | `lean5` minus group trend |

`composition` is the original weighted sum of observed, virtual, inverse, and sequence
composition constraints. `group trend` is the original weighted sum of group-rate and
AD-minus-CN-gap constraints. Grouping preserves their original relative weights.

## Matched protocol

The experiment contains 48 jobs: four representations, four objectives, and seeds 42/43/44.
Within each representation and seed, all four jobs have identical initialization, balanced
pair batches, virtual split-point draws, sequence order, model, optimizer, and evaluation
times. Adaptive128 retains its historical batch size of 8; the other historical batch sizes
are also retained.

An epoch is 32 optimizer updates for every representation. Training lasts 60 epochs (1,920
updates) with no early stopping. Learning rate warms from `1e-5` to `3e-4` over the first four
epochs and then cosine-decays to `3e-5`. This slows the initial jump that previously placed
many optima inside the first evaluation interval.

All checkpoints are evaluated and logged from epoch 1. Checkpoints become eligible for the
primary selection after warmup at step 128, but `best_any.pt` separately preserves the true
best checkpoint from the entire trajectory. A candidate fails the later-best requirement if
any seed's true best remains at epoch 1. Thus the protocol does not manufacture a later best
by hiding early results.

The test split is never loaded. The comparison reports mean and standard deviation over seeds
for the legacy score, decoded-target transport error, observed-mesh forecast error, CN/AD
group-rate error, and cocycle/inverse defects. It recommends the smallest objective that
passes every preregistered tolerance in all four representations.

See [RUNBOOK.md](RUNBOOK.md) for commands.

