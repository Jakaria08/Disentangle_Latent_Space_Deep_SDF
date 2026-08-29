# Verification record

Date: 2026-08-29

## Completed

- Python compilation: passed.
- JSON parsing and configuration contracts: passed.
- Assertion tests: 10 passed, 0 failed.
- Exact latest-LAMM layout: 43+86 regions, each independently partitions all 2,746 vertices.
- Prepared data: 2,037 train scans / 475 subjects; 269 validation scans / 61 subjects;
  277 test scans / 61 subjects; subject isolation passed.
- Real-mesh 128D, 256D equal, and 256D fine-heavy forward contracts: passed.
- Identity error at `source_age == target_age`: exactly `0.0 mm` for every width.
- Diagonal finite-difference velocity check: maximum error
  `0.0002514 mm/year`, below the `0.0005` gate.
- Diagnosis branch changes velocity: passed.
- End-to-end CPU smoke training: passed, one epoch/four batches.
- Every major component accumulated non-zero gradient in the smoke run: passed.
- Training checkpoint declared `test_data_loaded=false`: passed.
- Separate validation evaluator on the smoke checkpoint: passed.
- Optuna SQLite creation, queued anchor trial, training, reporting, and best-config export:
  passed with Optuna 4.7 in the `inr_sdf` environment.

## Measured model sizes

- 128D `[64,64]`: 30,866,872 trainable parameters.
- 256D `[128,128]`: 33,078,968 trainable parameters.
- 256D `[96,160]`: 33,431,224 trainable parameters.

The six-channel regional velocity heads account for 16,510,194 parameters and produce CN
base velocity plus AD residual velocity. The larger bottlenecks add capacity mainly in the
multiscale down projections and conditioned latent block.

## GPU status during verification

GPU 1 could not be tested in this session because both NVML and PyTorch reported no visible
CUDA device (`cuda_available=false`, `device_count=0`). CPU exercised the full train and
evaluation paths. Run the GPU-1 smoke command in `RUNBOOK.md` when the driver/device is
visible; the smoke gate is already implemented and will fail on missing gradient flow.

## Scope of smoke results

One smoke epoch verifies execution and contracts, not scientific convergence. Its velocity
feasibility gate was expectedly false and its selected checkpoint remained the epoch-0
no-change checkpoint. Do not report the smoke metrics as model performance.
