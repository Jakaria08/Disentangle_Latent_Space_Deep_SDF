# Runbook

Use the environment containing PyTorch Geometric and the existing frozen LAMM decoder:

```bash
TASK=/home/jakaria/INR/Deep3DComp/examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task9_latent_cocycle_controlled_v1
PY=/home/jakaria/anaconda3/envs/pytorch_geo/bin/python
```

## Verify the implementation

```bash
$PY "$TASK/tests/test_contracts.py"
$PY "$TASK/scripts/sweep.py" --dry-run
```

The first command checks the exact original `full13` formula, the nested reduced objectives,
the fixed task3 model implementation, identity at equal times, and all 48 resolved jobs.

## Smoke test one job

```bash
$PY "$TASK/scripts/make_job.py" \
  --representation pca128 --loss-set lean5 --seed 42 \
  --output /tmp/task9_pca_lean5_s42.json

$PY "$TASK/scripts/train.py" \
  --config /tmp/task9_pca_lean5_s42.json \
  --device cuda:0 --smoke --output-dir /tmp/task9_pca_lean5_smoke
```

Use a new `/tmp` output name if that smoke directory already exists; the trainer refuses to
overwrite an existing resolved run.

## Recommended pilot

Run the matched control and the middle five-loss candidate on PCA and LAMM first:

```bash
$PY "$TASK/scripts/sweep.py" \
  --representation pca128,lamm128 \
  --loss-set full13,lean5 \
  --seed 42 --gpus 0,1 --per-gpu 1
```

Rerunning the same command skips completed jobs and resumes a job that has `latest.pt`.

## Full 48-job experiment

```bash
$PY "$TASK/scripts/sweep.py" --gpus 0,1,2 --per-gpu 1
```

Two jobs per GPU can be requested with `--per-gpu 2` after the pilot confirms memory headroom.
The dispatcher writes progress to:

```text
/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task9_latent_cocycle_controlled_v1/sweep_status.json
```

Each run contains `steps.jsonl`, `evaluations.jsonl`, resumable `latest.pt`, unconstrained
`best_any.pt`, selection-eligible `best.pt`, `summary.json`, and `train.log`.

## Compare and select the objective

```bash
$PY "$TASK/scripts/compare.py"
```

Results are written to the runtime `comparison` directory as per-run CSV, aggregate JSON,
and a decision JSON. `recommended_loss_set` remains null unless one reduced objective has all
three seeds and passes every shape, observed-mesh, rate, defect, and later-best check for all
four representations.

