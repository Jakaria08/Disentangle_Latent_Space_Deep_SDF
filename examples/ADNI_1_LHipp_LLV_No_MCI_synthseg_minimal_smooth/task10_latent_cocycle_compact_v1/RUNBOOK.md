# Runbook

Set the paths once:

```bash
TASK=/home/jakaria/INR/Deep3DComp/examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task10_latent_cocycle_compact_v1
PY=/home/jakaria/anaconda3/envs/pytorch_geo/bin/python
```

## Verify without training

```bash
$PY "$TASK/tests/test_contracts.py"
$PY "$TASK/scripts/sweep.py" --dry-run
```

The dry run must report 33 jobs. These commands do not train a model or create runtime
result directories.

## Optional smoke test

```bash
$PY "$TASK/scripts/make_job.py" \
  --representation pca128 \
  --arm compact6_pca_low \
  --seed 42 \
  --output /tmp/task10_pca_compact6_low_s42.json

$PY "$TASK/scripts/train.py" \
  --config /tmp/task10_pca_compact6_low_s42.json \
  --device cuda:0 \
  --smoke \
  --output-dir /tmp/task10_pca_compact6_low_smoke
```

Use a new output directory for another smoke test. The trainer refuses to overwrite an
existing resolved run.

## Recommended targeted pilot

This seven-job pilot exercises all five PCA arms and the two LAMM standard arms:

```bash
$PY "$TASK/scripts/sweep.py" \
  --representation pca128,lamm128 \
  --seed 42 \
  --gpus 0,1 \
  --per-gpu 1
```

To inspect the selection first:

```bash
$PY "$TASK/scripts/sweep.py" \
  --representation pca128,lamm128 \
  --seed 42 \
  --dry-run
```

## Full 33-job experiment

```bash
$PY "$TASK/scripts/sweep.py" --gpus 0,1,2 --per-gpu 1
```

The dispatcher skips completed jobs and resumes an incomplete job when `latest.pt` exists.
Progress is stored at:

```text
/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task10_latent_cocycle_compact_v1/sweep_status.json
```

Useful filters are:

```bash
# Both standard compact objectives for all representations
$PY "$TASK/scripts/sweep.py" \
  --arm compact6_standard,compact5_standard \
  --gpus 0,1,2 --per-gpu 1

# All 15 PCA jobs, across both learning-rate profiles
$PY "$TASK/scripts/sweep.py" \
  --representation pca128 \
  --objective full13,compact6,compact5 \
  --gpus 0,1,2 --per-gpu 1
```

The second command includes the six standard-schedule PCA compact jobs because
`--objective` filters by objective rather than schedule. To select exactly the nine
lower-LR jobs, use:

```bash
$PY "$TASK/scripts/sweep.py" \
  --arm full13_pca_low,compact6_pca_low,compact5_pca_low \
  --gpus 0,1,2 --per-gpu 1
```

## Compare after completion

```bash
$PY "$TASK/scripts/compare.py"
```

Comparison requires the 12 completed task9 full13 summaries. It pairs each arm with the
control declared in `configs/experiment.json`, reports all preregistered checks, and writes:

```text
/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task10_latent_cocycle_compact_v1/comparison/runs.csv
/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task10_latent_cocycle_compact_v1/comparison/aggregate.json
/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task10_latent_cocycle_compact_v1/comparison/decision.json
```

Use `best.pt` from an adopted arm. `latest.pt` is the resumable final state and is not the
selected model.
