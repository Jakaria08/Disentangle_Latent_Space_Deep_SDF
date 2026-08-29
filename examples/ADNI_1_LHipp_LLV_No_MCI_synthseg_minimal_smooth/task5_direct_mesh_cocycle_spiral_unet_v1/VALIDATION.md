# Implementation validation

Status: velocity-aware v2 implementation complete; long v2 Optuna searches and
scientific training remain user-run jobs.

## Completed checks

- Prepared cache: 2,037 train, 269 validation, and 277 test scans; 475/61/61
  subject-disjoint subjects.
- Hierarchy: 2,746 -> 1,373 -> 344 -> 86 vertices.
- Dependency-free suite: 21/21 tests passed.
- Exact identity: maximum error 0.0 mm for both operators.
- Diagonal finite-difference check: maximum error 0.0002514 mm/year.
- Disease condition changes the initialized field for both operators.
- Trainable parameters: 346,566 Spiral and 441,285 Adaptive.
- Adaptive placement: exactly three coarse modules and none at the two finest
  resolutions.
- Adaptive isolated gradient/update gate passed.
- GPU 1 one-epoch training passed for both operators.
- GPU 1 Adaptive training changed every support predictor; total L1 change was
  0.08325 across the three modules.
- GPU 1 endpoint, instantaneous-velocity, cocycle, partial exact-surface,
  averaged-map, and direction-aware paired-comparison paths passed.
- Resumable Optuna SQLite/search/export passed for two isolated smoke trials per
  operator for both the original and velocity-v2 paths. Smoke studies cannot
  share the real study path.
- Training checkpoints record `ode_used=false`,
  `global_latent_bottleneck=false`, and `test_data_loaded=false`.
- Velocity-v2 keeps the completed endpoint-v1 study unchanged and writes to
  separate `*_direct_c4_velocity_*` studies.
- Reliability weighting, zero-velocity normalization, speed-collapse,
  AD/CN-volume-order, flipped-face, cocycle, and inverse gates passed unit tests.
- Frozen evaluation preserves the historical unweighted velocity summary and
  additionally exports the reliability-weighted, zero-normalized summary used
  for v2 selection.
- The saved v1 trial-13 checkpoint passes every v2 feasibility gate on all 269
  validation scans: score 0.95224, endpoint macro ratio 0.90849, velocity ratio
  0.87472, robust speed ratio 0.45859, zero flipped faces, relative cocycle
  0.00107, and relative inverse 0.00284.
- GPU 1 velocity-v2 one-epoch training passed for both operators. Adaptive v2
  changed all three support predictors, with total L1 change 0.12361.
- Age-stratified velocity analysis passed deterministic boundary, reliability-
  weighting, diagnosis separation, and subject-cluster bootstrap tests. A real
  checkpoint/data CPU smoke created every CSV/JSON/PNG/PDF artifact.
- Full validation age analysis completed for the current Spiral winner and the
  current best completed Adaptive trial: 241 positive-reliability visits in the
  common-support 70--95 year range, six diagnosis/age groups, and 5,000
  subject-bootstrap samples per group. Test data were not loaded.

The one-epoch/four-pair smoke numbers are not scientific results. In particular,
their endpoint error remains essentially equal to the no-change prediction and
their predicted speed is much smaller than the observed fitted longitudinal
reference. This is expected from a wiring smoke and must not be used to rank the
operators. Use the full commands in `RUNBOOK.md` before interpreting accuracy,
AD/CN volume rates, or instantaneous surface velocity.
