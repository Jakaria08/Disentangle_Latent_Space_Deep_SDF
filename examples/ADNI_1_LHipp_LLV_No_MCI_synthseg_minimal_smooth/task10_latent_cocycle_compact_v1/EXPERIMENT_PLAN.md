# Preregistered experiment plan

## Questions

Task10 answers three questions without changing the network:

1. Does restoring endpoint and sequence latent supervision repair the LAMM algebraic
   defects observed with task9 `lean6`?
2. Can sequence-slope supervision be removed across all four representations?
3. Can the population-trend pair also be removed, and does a lower PCA learning rate
   recover historical PCA performance?

## Fixed factors

The following remain fixed from task9 for every matched job:

- direct, non-ODE, 128-D latent-only network;
- width 256, two post-activation residual blocks, zero dropout;
- frozen PCA/SpiralNet/Adaptive/LAMM encoders and decoders;
- train/validation subjects and prepared latent archives;
- pair sampler, sequence sampler, virtual split draws, and seed;
- batch size for each representation, 32 updates per epoch, and 60 epochs;
- AdamW, weight decay, gradient clipping, validation score, and feasibility tests;
- raw loss definitions and all individual weights;
- selection eligibility beginning at step 128;
- validation-only selection, with no test archive loaded.

Early stopping remains disabled. Task9 contained real improvements after gaps of more than
15 evaluations, including an Adaptive best at epoch 47. A patience rule would therefore
change the scientific comparison. `best.pt` stores the best eligible checkpoint, while
`best_any.pt` records the literal best checkpoint from the full trajectory.

## Controlled contrasts

For the standard schedule:

```text
compact6_standard - task9_full13_standard
    = effect of removing sequence slope

compact5_standard - task9_full13_standard
    = joint effect of removing sequence slope and population trend

compact5_standard - compact6_standard
    = incremental effect of removing population trend
```

For PCA-low:

```text
full13_pca_low - task9_full13_standard
    = effect of the lower PCA learning-rate profile

compact6_pca_low - full13_pca_low
    = slope removal at the lower learning rate

compact5_pca_low - full13_pca_low
    = slope plus population-trend removal at the lower learning rate
```

All contrasts are paired by seed. The standard compact arms preserve the same task9
initialization, data order, and raw-term computation path, so the saved task9 controls are
valid paired controls. The comparison script checks the task9 model, seeds, representations,
loss weights, validation settings, batch sizes, schedule, and optimizer-related settings
before accepting those controls. It also writes the direct seed-paired `compact5 - compact6`
contrast for each representation and both PCA learning-rate profiles.

## Primary and safety outcomes

The primary outcome is the task3 decoded-transport validation score. Safety outcomes are:

- observed end-to-end mesh error in millimetres;
- CN plus AD signed-rate error;
- relative semigroup defect;
- relative inverse defect;
- whether the literal best validation epoch is later than epoch 1 for every seed.

The test split is reserved for one final evaluation after an objective is adopted. It must
not be used to choose an arm.

## Adoption thresholds

An arm passes for a representation only when all three seed-matched runs satisfy the
aggregate rules below:

| check | threshold |
|---|---:|
| mean validation-score increase versus matched control | at most 0.0015 |
| mean validation-score increase versus historical run | at most 0.0015 |
| mean observed-mesh increase versus matched control | at most 0.002 mm |
| mean group-rate-error increase versus matched control | at most 0.002 |
| mean semigroup defect divided by control | at most 1.25 |
| mean inverse defect divided by control | at most 1.25 |
| literal best checkpoint | after epoch 1 in every seed |

The universal recipe uses `pca_low` for PCA and `standard` for SpiralNet, Adaptive, and
LAMM. The selector first considers `compact5`, then `compact6`, and adopts the first one
that passes every check for all four representations. If neither passes, the decision is
null and the per-representation results remain available.

## Interpretation order

1. Check completeness, failed jobs, and later-best behavior.
2. Check the PCA `full13_pca_low` schedule control before interpreting PCA loss changes.
3. Compare each compact arm with its declared matched control by seed.
4. Examine LAMM semigroup and inverse ratios before considering mesh-score gains.
5. Prefer `compact5` only if its rate and algebraic checks pass; otherwise retain
   `compact6`.
6. Run the held-out test evaluation only after freezing the selected recipe.

## Expected artifacts

Training will write only below:

```text
/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task10_latent_cocycle_compact_v1
```

Each run directory is keyed by representation, arm, and seed. It contains resumable and
selected checkpoints, step and evaluation logs, the resolved job configuration, and a
summary. Comparison writes `runs.csv`, `aggregate.json`, and `decision.json` below the
task10 runtime `comparison` directory.
