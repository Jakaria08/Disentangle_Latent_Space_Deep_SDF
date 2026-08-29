# Velocity reference audit v2

This isolated experiment replaces a noisy visit-difference comparison with a
validated subject-trajectory reference. It does **not** modify the source meshes,
trained checkpoints, existing task-4 notebook, or existing task-4 caches.

There is no directly measured instantaneous ground truth in these longitudinal
ADNI scans. The notebook therefore keeps three distinct observed quantities:

- **Observed — adjacent interval:** change between scans divided by elapsed years.
- **Observed — first-to-last:** the long-interval average rate.
- **Observed — fitted trajectory:** the derivative of the validation-selected
  smoother; this is the primary reference, but it is still an estimate.

Estimator selection uses leave-one-interior-visit-out prediction on validation
subjects only. Test subjects are used once after the estimator is frozen. Rigid
alignment is analysis-only, subject-specific, rotation/translation only, and has
neither scale nor reflection.

The load-only notebook is `notebooks/velocity_reference_audit.ipynb`. Numerical
work is performed by scripts first, so rerunning notebook cells is quick.
