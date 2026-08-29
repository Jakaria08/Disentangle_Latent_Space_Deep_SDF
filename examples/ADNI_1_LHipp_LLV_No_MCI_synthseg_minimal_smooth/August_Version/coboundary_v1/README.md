# Exact coupling-coboundary C4 (128-D)

This isolated experiment compares the same exact coboundary architecture on
fixed PCA-128, SpiralNet++-128, and Adaptive-Spiral-128 representations.
Existing experiments and representation snapshots are read only.

The transport is

`Phi(z,s,t;c,d) = F(t;c,d)(F(s;c,d)^-1(z))`,

where `c` is the subject's fixed first-visit latent and age, and each `F` is a
stack of alternating time-conditioned affine coupling layers. This makes
identity, inverse, and composition architectural equalities up to floating-point
roundoff. The frozen PCA inverse transform or frozen neural decoder maps every
predicted latent back to vertices inside the training loss.

Run the non-mutating algebra/config test:

```bash
/home/jakaria/anaconda3/envs/pytorch3d/bin/python \
  examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/August_Version/coboundary_v1/tests/verify_exact_coboundary.py
```

Standalone physical-GPU-2 launch:

```bash
bash \
  examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/August_Version/coboundary_v1/scripts/start_detached_gpu2.sh
```

The wrapper creates a detached `screen` session named `exact_cob_s42_gpu2`.
Inside it, the launcher starts three separate training processes concurrently
on physical GPU 2, waits for all of them, evaluates validation and sealed test
sets, then writes JSON/CSV/Markdown comparisons against matched plain ODE,
BrainODE, and direct-C4 baselines.

Read-only status command:

```bash
bash \
  examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/August_Version/coboundary_v1/scripts/status_gpu2.sh
```
