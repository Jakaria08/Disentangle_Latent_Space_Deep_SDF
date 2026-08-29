# Runbook

Run commands from `/home/jakaria/INR/Deep3DComp`.

## 1. Unit tests

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python -m unittest discover \
  -s examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task4_velocity_reference_audit_v2/tests -v
```

## 2. Optional bounded smoke audit

This writes only under `/tmp` and is not accepted as a final result.

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python \
  examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task4_velocity_reference_audit_v2/scripts/build_velocity_reference_cache.py \
  --output-root /tmp/task4_velocity_reference_audit_smoke \
  --max-subjects 4 --bootstrap 100 --skip-surface-distances --force
```

## 3. Full CPU audit

This normally takes under one minute on this workstation. Add `--force` only
when intentionally replacing the isolated generated cache.

```bash
MPLCONFIGDIR=/tmp/mpl-task4-velocity \
/home/jakaria/anaconda3/envs/inr_sdf/bin/python \
  examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task4_velocity_reference_audit_v2/scripts/build_velocity_reference_cache.py
```

## 4. Compare trained generators

This step reuses completed current and legacy caches; it does not need a GPU.

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python \
  examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task4_velocity_reference_audit_v2/scripts/compare_model_generators.py
```

## 5. Validate

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python \
  examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task4_velocity_reference_audit_v2/scripts/validate_velocity_reference_audit.py
```

## 6. Open or check the load-only notebook

It is safe to run all cells; it loads CSV/NPZ results and creates plots only.
Open the notebook in VS Code or another Jupyter frontend and choose the
`inr_sdf` Python kernel:

```bash
examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task4_velocity_reference_audit_v2/notebooks/velocity_reference_audit.ipynb
```

This environment does not currently provide `jupyter-lab` or `nbconvert`. Use
the following tested command for a headless dependency and code-cell check. It
does not save plot outputs:

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python \
  examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task4_velocity_reference_audit_v2/scripts/check_notebook_cells.py
```
