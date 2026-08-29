# Volume-aware exact coboundary V2

This isolated experiment keeps the exact group-coboundary transport law while giving decoded
hippocampal volume an explicit train-only coordinate. It does not modify the earlier ODE,
BrainODE, direct-C4, or exact-coboundary runs.

## Model

For diagnosis `c`, fixed baseline context `d`, source age `s`, and target age `t`,

```text
Phi(z, s, t; c, d) = F_t,c,d(F_s,c,d^-1(z)).
```

The frozen representation decoder supplies train-split log-volumes. A subject-balanced ridge
fit estimates `log V(D(z)) = b + w^T z`; a fixed Householder rotation maps `w/||w||` to the
first chart coordinate. A diagnosis-specific scalar potential shifts only this volume
coordinate. Four affine coupling layers transform only the orthogonal 127-D shape coordinates.
All transforms have analytic inverses, so identity, inverse, and composition are exact up to
floating-point error.

## Loss

The optimizer combines real-pair latent and decoded-vertex errors; sequence latent and vertex
errors; decoded volume, log-volume rate, sequence slope, group-rate, and AD-minus-CN gap terms;
and matching losses applied directly to the explicit volume potential. The frozen PCA,
SpiralNet++, or Adaptive decoder stays differentiable with respect to the predicted latent but
its parameters are excluded from optimization. Anatomy losses ramp over 5 epochs and sequence
losses over 10 epochs. Thirty-five percent of samples are diagnosis-balanced first-to-last
pairs.

Checkpoint selection uses validation subjects only. It combines macro CN/AD first-to-last shape,
all-pair shape, volume, rate, signed trend, and diagnosis-gap error, while requiring shape to stay
within the configured no-change tolerance and exact-coboundary defects below `1e-4`. Test data is
loaded only by the post-training evaluator.

## Runs

The three independent 128-D runs share physical GPU 2 and execute concurrently:

```bash
bash examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/August_Version/coboundary_volume_v2/scripts/start_detached_gpu2.sh
```

Status:

```bash
bash examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/August_Version/coboundary_volume_v2/scripts/status_gpu2.sh
```

Each run writes to a new `volume_exact_coboundary_c4_v2` method directory. After all training
processes finish, validation and test evaluations run sequentially and a subject-paired bootstrap
comparison is generated against matched plain ODE, BrainODE, and direct-C4 results.
