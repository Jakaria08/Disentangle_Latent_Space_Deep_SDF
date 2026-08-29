# Runbook

From the repository root:

```bash
TASK=examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task6_ood_mesh_rollout_v1
PY=/home/jakaria/anaconda3/envs/pytorch_geo/bin/python

# Optional no-write wiring check on GPU 1. This stops at age 80.
$PY $TASK/scripts/run_ood_rollout.py --device cuda:1 --smoke

# Full annual composition from age 75 through age 105.
$PY $TASK/scripts/run_ood_rollout.py --device cuda:1 --force
```

The full script saves PLY meshes and latent states under:

```text
/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task6_ood_mesh_rollout_v1/age75_to105_v1
```

Interactive HTML reports are written under
`reports/age75_to105_v1/`. After the script finishes, open
`notebooks/ood_age75_to105.ipynb` and run its cells. The notebook only loads
the completed tables, PNG, and HTML; it performs no inference.

`--force` replaces files with the same names but does not delete the output
directory. To keep a separate run, pass a different `--output-root` and update
the notebook's `RESULT_DIR` accordingly.
