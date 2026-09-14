# All cocycle-method longitudinal visualization

This experiment consolidates latent and direct-mesh cocycle results, matched PCA plain-ODE and PCA BrainODE baselines, the two INR reconstruction representations, surface-space instantaneous velocity, AD/CN volume trends, and a validation-selected cocycle versus paper-core BrainODE age-105 stress test.

The notebook is load-only. Run the commands below from `/home/jakaria/INR/Deep3DComp` after the centralized validation/test jobs have completed.

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/All_Visualization/scripts/audit_inputs.py --strict

/home/jakaria/anaconda3/envs/inr_sdf/bin/python examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/All_Visualization/scripts/extract_pca_ode_velocity.py --device cuda:1 --force

/home/jakaria/anaconda3/envs/inr_sdf/bin/python examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/All_Visualization/scripts/build_analysis.py --require-central-test

/home/jakaria/anaconda3/envs/pytorch_geo/bin/python examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/All_Visualization/scripts/build_ood_comparison.py --device cuda:1

/home/jakaria/anaconda3/envs/inr_sdf/bin/python examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/All_Visualization/scripts/make_notebook.py

/home/jakaria/anaconda3/envs/inr_sdf/bin/python examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/All_Visualization/scripts/execute_notebook.py

/home/jakaria/anaconda3/envs/inr_sdf/bin/python examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/All_Visualization/scripts/validate_outputs.py
```

The centralized evaluator and direct-LAMM per-scan velocity extraction commands are recorded in `RUNBOOK.md`.
