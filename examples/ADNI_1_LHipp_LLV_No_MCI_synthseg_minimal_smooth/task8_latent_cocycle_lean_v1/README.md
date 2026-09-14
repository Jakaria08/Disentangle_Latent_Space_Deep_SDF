# Lean latent cocycle flow — reduced objective and architecture sweep

Isolated experiment on the **latent** cocycle flow only (`Phi(z,s,t,d) = z + (t-s)*v`).
No direct-mesh flow, no LAMM/Spiral mesh integration. Four frozen 128-D representations:
`pca128`, `spiralnet128`, `adaptive128`, `lamm128`.

Runtime artifacts: `/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task8_latent_cocycle_lean_v1`

## What is being tested

The published `direct_c4` objective carries 13 weighted terms, of which three groups are
algebraically redundant rather than merely small (measured shares at a saturated-ramp epoch
of `spiralnet128_direct_c4_s42`):

| group | terms | why redundant |
|---|---|---|
| latent duplicates | `real_latent`, `sequence_latent` (19.4%) | for `pca128` the decoder is affine, so latent MSE is decoded-coordinate error in another metric |
| one composition law | `observed_semigroup`, `virtual_semigroup`, `sequence_semigroup`, `inverse` (0.05% combined) | all are `Phi(Phi(z,s,r),r,t) = Phi(z,s,t)` under four samplers of `r` |
| one volume functional | `volume`, `rate`, `slope`, `group_rate`, `disease_gap` (2.1%) | level, first difference, LS slope, group mean, difference of group means of one decoded log-volume trajectory |

The reduced objective keeps one representative per group and merges the four composition
penalties into a single `comp` term with a configurable family mixture:

```
L = w_vtx * L_vtx + w_seq * L_seq_vtx + w_comp * L_comp + w_rate * L_rate  [+ w_lat, w_trend]
```

## Fixes to the training protocol

The baseline held `samples_per_epoch = 4096` while `batch_size` varied by decoder cost
(96 / 24 / 8), so one "epoch" was 43, 171 or 512 optimizer steps depending on the
representation. All four baselines peak at 170-500 steps; for three of them that lands
inside epoch 1, which is why they report `best_epoch = 1` and stop 30 epochs later.

This experiment therefore:

* defines an epoch as a **fixed number of optimizer steps** (64), independent of batch size;
* **validates on a step interval** (every 32) and selects the best checkpoint per step, so a
  mid-epoch optimum can actually be captured;
* expresses both warm-up ramps in **steps**, not epochs — in the baseline the consistency and
  anatomy ramps were still at 0.10 and 0.067 during epoch 1, i.e. essentially off during the
  exact window that produced the best model;
* **logs every optimizer step** to `steps.jsonl` (per-term raw and weighted losses, total,
  learning rate, gradient norm) and every evaluation to `evals.jsonl`.

Budget: 30 epochs x 64 steps = **1920 steps**, ~60 validations per run.

## Architecture variants

| arch | trunk | note |
|---|---|---|
| `postact` | `SiLU(x + f(LN(x)))` | exact baseline replica, 364,288 params — the control |
| `preact` | `x + f(LN(x))` | clean identity path |
| `film` | `x + f(LN(x)*(1+g)+b)` | zero-init FiLM from an interval embedding injected at **every** block; the baseline injects time only once, at the input layer |

All variants keep both structural guarantees: exact identity at `s=t` and exact no-change at
initialization (zero-initialized CN and AD heads).

## Comparison protocol

Data loading, frozen-decoder geometry, `evaluate_pairs`, `cocycle_defects` and
`validation_score` are imported **unchanged** from `task3_latent_flow_128_v1`, and the
selection score and gates are identical, so every number is directly comparable to the
published baselines:

| representation | baseline score | baseline best epoch |
|---|---|---|
| `pca128` | 1.09044 | 7 |
| `spiralnet128` | 1.08686 | 1 |
| `adaptive128` | 1.08848 | 1 |
| `lamm128` | 1.08646 | 1 |

Reported columns: selection score, feasibility, macro first-to-last shape ratio, all-pair
shape ratio, CN/AD volume-trend absolute rate error and the AD-CN gap error, semigroup and
inverse defects, decoded reconstruction error, and the step at which the best model occurred.

## Provenance and integrity

* Representation archives are **symlinked** from the existing prepared task outputs, so
  latents are bit-identical to the published baselines. Nothing is re-encoded.
* `configs/representations.json` merges the `v1` (pca/spiral/adaptive) and `v2_lamm` (lamm)
  registries. The LAMM builder-source hash is re-pinned with a documented note: `train_lamm.py`
  gained `--out-root` and the semi-amortized training options, but `build()` constructs the
  same architecture and `load_state_dict(strict=True)` remains the binding guarantee.
* `scripts/_bootstrap.py` defaults the three new training-only `train_lamm` fields so frozen
  pre-change LAMM checkpoints still rebuild. Shared task sources are left untouched.
* Test split is never opened.

## Commands

```bash
TASK=examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task8_latent_cocycle_lean_v1
PY=/home/jakaria/anaconda3/envs/pytorch_geo/bin/python   # torch_scatter is required by LAMM

$PY $TASK/scripts/sweep.py --gpus 0,1,2 --per-gpu 2            # 48 jobs
$PY $TASK/scripts/sweep.py --dry-run                            # list jobs
$PY $TASK/scripts/sweep.py --only-variant B1_film_lean4         # subset
$PY $TASK/scripts/compare.py                                    # table + comparison.csv
```

Single run:

```bash
$PY $TASK/scripts/train_lean.py --config <job.json> --device cuda:0 --output-dir <dir>
```
