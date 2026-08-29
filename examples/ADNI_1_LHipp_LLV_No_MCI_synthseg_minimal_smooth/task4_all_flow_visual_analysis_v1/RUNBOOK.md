# All-flow longitudinal visual analysis

This isolated analysis reads completed checkpoints and existing evaluation
artifacts without modifying them. Current PCA, Spiral, Adaptive and INR
cocycle models are compared on their exact shared test subset. Completed legacy
Cocycle, latent ODE and BrainODE results are shown separately. Epoch-1 smoke
runs are excluded.

From the repository root:

```bash
TASK=examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task4_all_flow_visual_analysis_v1
PY=/home/jakaria/anaconda3/envs/pytorch_geo/bin/python

$PY $TASK/scripts/build_analysis_cache.py \
  --device cuda:1 --output-name cache_v1 \
  --surface-resolution 256 --surface-samples 5000 \
  --voxel-pitch-mm 0.5 --bootstrap-samples 2000

$PY $TASK/scripts/validate_analysis_cache.py \
  --cache-dir /mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task4_all_flow_visual_analysis_v1/cache_v1

# Velocity-only supplement: current matched cocycle cohort.
$PY $TASK/scripts/build_instantaneous_surface_velocity.py \
  --stage current --device cuda:1 --force

# Velocity-only supplement: completed Cocycle/ODE/BrainODE reference cohort.
LEGACY_PY=/home/jakaria/anaconda3/envs/inr_sdf/bin/python
$LEGACY_PY $TASK/scripts/build_instantaneous_surface_velocity.py \
  --stage legacy --device cuda:1 --force

$PY $TASK/scripts/validate_instantaneous_surface_velocity.py

mkdir -p /tmp/task4_jupyter_config
JUPYTER_CONFIG_DIR=/tmp/task4_jupyter_config $PY -m jupyter nbconvert --to notebook --execute \
  $TASK/notebooks/all_flow_results.ipynb \
  --output all_flow_results.executed.ipynb \
  --output-dir $TASK/notebooks \
  --ExecutePreprocessor.timeout=600
```

The notebook is load-only. The original cache stores endpoint analysis and
latent-velocity results. The two velocity-only commands add decoded
instantaneous surface velocity without changing the original cache or any
checkpoint. Existing completed outputs can be viewed without rebuilding either
cache.

## Matched instantaneous velocity by age

This analysis compares direct-mesh Spiral and Adaptive, latent PCA/Spiral/
Adaptive, and LAMM N=3 on their 61-subject validation intersection. INR is added
on the strict 20-subject intersection. The observed reference is the derivative
of a smooth longitudinal mesh trajectory after rigid-motion removal; it is not
a directly measured continuous-time derivative. Confidence intervals resample
subjects, not individual visits. The test split is never loaded.

From the repository root:

```bash
TASK=examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task4_all_flow_visual_analysis_v1
PY=/home/jakaria/anaconda3/envs/pytorch_geo/bin/python

# Quick all-seven-method, no-write inference check on GPU 1.
$PY $TASK/scripts/analyze_velocity_by_age_all.py \
  --config $TASK/configs/age_velocity_all_methods.json \
  --device cuda:1 --batch-size 8 --smoke

# Full validation computation (long-running INR surface decoding is included).
$PY $TASK/scripts/analyze_velocity_by_age_all.py \
  --config $TASK/configs/age_velocity_all_methods.json \
  --device cuda:1 --batch-size 16 --force

# Validate schemas, matched cohorts, numerical values, figures and LAMM N=3.
$PY $TASK/scripts/validate_velocity_by_age_all.py \
  --root /mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task4_all_flow_visual_analysis_v1/age_velocity_all_methods_v1

# Recreate the load-only explanatory notebook after successful validation.
$PY $TASK/scripts/make_velocity_by_age_notebook.py

# Optional: save an executed copy (loads tables/figures only; no inference).
mkdir -p /tmp/task4_velocity_jupyter
JUPYTER_CONFIG_DIR=/tmp/task4_velocity_jupyter $PY -m jupyter nbconvert \
  --to notebook --execute $TASK/notebooks/all_methods_velocity_by_age.ipynb \
  --output all_methods_velocity_by_age.executed.ipynb \
  --output-dir $TASK/notebooks --ExecutePreprocessor.timeout=120
```

Completed output is under
`/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task4_all_flow_visual_analysis_v1/age_velocity_all_methods_v1`.
The strict common cohort's oldest AD age bin contains only one subject, so that
single bin is descriptive rather than a stable group estimate.
