# ADNI left-hippocampus latest-LAMM/SpiralNet-v7 direct-C4 comparison

This task freezes two current 128-D representation snapshots and applies the
same direct, non-ODE, non-coboundary C4 transport to both:

- `lamm128`: latest LAMM `E1_N3_ml_s1`, validation coordinate RMSE
  `0.0338338295` mm and test coordinate RMSE `0.0345900660` mm.
- `spiralnet128`: pure SpiralNet++ v7 trial 36, validation coordinate RMSE
  `0.0367844103` mm.

The LAMM `D=256` value is its internal mixer width. Its transported latent is
128-D, split into the saved 64-D 43-region and 64-D 86-region scale blocks.

The source checkpoints, meshes, hierarchy caches, and longitudinal pair tables
are read-only.  The shared latent-flow core is extended for LAMM, while new
runtime artifacts go only below:

`/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task3_latent_flow_128_v3_lamm_latest`

The N3 ensemble adds `lamm128_n3_s2` and `lamm128_n3_s3` beside the existing
seed-1 `lamm128`.  Latents from independent seeds are not coordinate-aligned:
each receives its own matched direct-C4, and only decoded corresponding mesh
vertices are averaged.

## Scientific contract

For both representations:

1. Normalize meshes with train-split per-vertex statistics.
2. Encode raw 128-D latents with the frozen snapshot.
3. Fit latent mean/std on training visits only and apply it to every split.
4. Train the identical direct C4:

   `Phi(z,s,t,d) = z + (t-s) [v_CN(z,s,t) + d v_AD(z,s,t)]`.

5. Keep the decoder frozen but differentiable with respect to the predicted
   latent, so decoded-vertex and anatomy losses train only the flow.
6. Select checkpoints using train/validation only.  Test evaluation is a
   separate explicit command after the run/seed is selected.

The direct-C4 weights, sampler, validation selection, and consistency guards
are copied exactly from the SpiralNet++ C4 baseline.  Coboundary is explicitly
zero.

## 1. Static contract and checkpoint test

From the repository root:

```bash
/home/jakaria/anaconda3/envs/pytorch_geo/bin/python \
  examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task3_latent_flow_128_v2_lamm/tests/verify_contracts.py
```

Add `--load-models` to strictly reconstruct both frozen models and run a finite
one-mesh encode/decode check on CPU.  This is read-only.

## 2. Representation dry run and export

Dry run (no files written):

```bash
/home/jakaria/anaconda3/envs/pytorch_geo/bin/python \
  examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task3_latent_flow_128_v2_lamm/scripts/prepare_representations.py \
  --device cpu --batch-size 1 --dry-run
```

Persistent export on physical GPU 1:

```bash
bash examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task3_latent_flow_128_v2_lamm/scripts/prepare_gpu.sh 1
```

Preparation refuses to overwrite an existing representation directory.

After export, inspect full-latent and LAMM coarse/fine-block flowability on
train or validation without touching test:

```bash
/home/jakaria/anaconda3/envs/pytorch_geo/bin/python \
  examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task3_latent_flow_128_v2_lamm/scripts/latent_diagnostics.py \
  --split val \
  --output /mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task3_latent_flow_128_v3_lamm_latest/diagnostics/latent_val.json
```

## 3. C4 dry runs

```bash
TASK=examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task3_latent_flow_128_v2_lamm
PY=/home/jakaria/anaconda3/envs/pytorch_geo/bin/python

$PY $TASK/scripts/train_c4.py --config $TASK/configs/spiralnet128_direct_c4_s42.json --device cuda:1 --dry-run
$PY $TASK/scripts/train_c4.py --config $TASK/configs/lamm128_direct_c4_s42.json --device cuda:1 --dry-run
```

Each dry run audits finite losses, validation, decoder-to-flow gradients, and
the absence of decoder parameter gradients.

## 4. Full matched training

Train and evaluate validation only:

```bash
bash examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task3_latent_flow_128_v2_lamm/scripts/run_one_gpu.sh \
  spiralnet128 1 42

bash examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task3_latent_flow_128_v2_lamm/scripts/run_one_gpu.sh \
  lamm128 1 42

bash examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task3_latent_flow_128_v2_lamm/scripts/run_one_gpu.sh \
  lamm128_n3_s2 1 42

bash examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task3_latent_flow_128_v2_lamm/scripts/run_one_gpu.sh \
  lamm128_n3_s3 1 42
```

Repeat with seeds 43 and 44 by changing the final argument.  The wrapper does
not evaluate test, preventing seed selection from observing it.

After selecting a run from validation, evaluate its sealed test split:

```bash
bash examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task3_latent_flow_128_v2_lamm/scripts/evaluate_selected_test.sh \
  /mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task3_latent_flow_128_v3_lamm_latest/training/lamm128/direct_c4/lamm128_direct_c4_s42 \
  1
```

## 5. Standard and paired comparisons

The core evaluator reports representation floor, no-change, transport-only,
end-to-end raw-mesh error, forward/backward horizons, rollout, volume/rate,
semigroup/inverse defects, and subject bootstrap intervals.

```bash
TASK=examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task3_latent_flow_128_v2_lamm
ROOT=/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task3_latent_flow_128_v3_lamm_latest/training

/home/jakaria/anaconda3/envs/pytorch_geo/bin/python $TASK/scripts/paired_compare.py \
  --baseline-run $ROOT/spiralnet128/direct_c4/spiralnet128_direct_c4_s42 \
  --candidate-run $ROOT/lamm128/direct_c4/lamm128_direct_c4_s42 \
  --split test \
  --output /mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task3_latent_flow_128_v3_lamm_latest/comparisons/lamm_vs_spiral_s42_test.json
```

The paired comparison resamples subjects, not individual visits.  Negative
candidate-minus-baseline error favors LAMM.

## 5b. N3 shape ensemble and instantaneous velocity

After all three N3 runs have sealed test evaluations, run
`evaluate_n3_ensemble.py` with the three run directories.  The evaluator
reports the zero-horizon direct-C4 velocity, converts normalized age to years,
and applies a decoder Jacobian-vector product to obtain mm/year mesh velocity.
Its PNG shows CN/AD surface speed and velocity directions; latent speed is
diagnostic only because seed-specific latent bases are not aligned.

## 6. Exact surface/topology metrics

The surface wrapper reuses the established deterministic 10,000-point
point-to-triangle evaluator and reports ASSD, HD95, Chamfer-L1, squared
Chamfer-L2, normals, topology, area, curvature, volume, and first-to-last rate.

```bash
TASK=examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task3_latent_flow_128_v2_lamm
ROOT=/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task3_latent_flow_128_v3_lamm_latest/training

/home/jakaria/anaconda3/envs/pytorch_geo/bin/python $TASK/scripts/evaluate_surface_forecasts.py \
  --run SpiralNet=$ROOT/spiralnet128/direct_c4/spiralnet128_direct_c4_s42 \
  --run LAMM=$ROOT/lamm128/direct_c4/lamm128_direct_c4_s42 \
  --split test --allow-test --device cuda:1 \
  --output-dir /mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task3_latent_flow_128_v3_lamm_latest/surface/s42_test

/home/jakaria/anaconda3/envs/pytorch_geo/bin/python $TASK/scripts/compare_surface_csv.py \
  --csv /mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task3_latent_flow_128_v3_lamm_latest/surface/s42_test/per_subject.csv \
  --baseline SpiralNet --candidate LAMM \
  --output /mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task3_latent_flow_128_v3_lamm_latest/surface/s42_test/paired_lamm_vs_spiral.json
```
