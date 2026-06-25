# Experiment 3: Personalized Age

This is the single Experiment 3 training run. Its control is the existing
`velocity_64_160_32_baseline_dx` Experiment 2.

The only intended model change is the age formulation:

```text
Experiment 2: population age velocity
Experiment 3: population age velocity + baseline-derived individual deviation
```

Run on GPU 1:

```bash
/home/jakaria/anaconda3/envs/inr_sdf/bin/python \
  train_deep_sdf_longitudinal_flow_64_160_32_personal_age.py \
  -e examples/Torus_subset_100_id_age_progression/velocity_64_160_32_e3_personal \
  --gpu 1
```

The network uses:

```text
age:       64D
disease:  160D
residual:  32D
```

The baseline anchor produces a fixed 16D modulation:

```text
m = 1 + 0.5 * tanh(H_m(stop_gradient(baseline_anchor)))
```

The final layer of `H_m` is initialized to zero, so the initial modulation is
exactly one and the initial age formulation equals Experiment 2.

## Modular Modes

Use personalized aging:

```json
"AgePersonalizationMode": "baseline_modulation"
```

Restore Experiment 2 population aging while retaining the same trainer:

```json
"AgePersonalizationMode": "population",
"UseAgeModulationRegularization": false,
"AgeModulationMagnitudeLambda": 0.0,
"AgeModulationCenterLambda": 0.0
```

Do not enable new real-pair or general-cocycle losses for the primary
Experiment 2 versus Experiment 3 comparison. Those would change more than the
age branch and confound the personalization result.

## Primary Comparison

Compare the existing Experiment 2 and this Experiment 3 with:

- identical train/test splits;
- one-shot predicted-diagnosis forecasting as the primary result;
- direct and composed rollout;
- future SDF and Chamfer error;
- diagnosis accuracy;
- latent and decoded shape cocycle error;
- external diagnosis probes for individual age, disease, and residual;
- neutral modulation (`m=1`) and matched shuffled-modulation evaluation.
