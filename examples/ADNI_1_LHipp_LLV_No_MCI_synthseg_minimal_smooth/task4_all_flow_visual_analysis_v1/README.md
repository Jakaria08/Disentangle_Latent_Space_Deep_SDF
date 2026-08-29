# Scope

The velocity-by-age supplement uses validation data only. Primary comparisons
use the 61-subject, 269-visit intersection shared by direct-mesh Spiral,
direct-mesh Adaptive, latent PCA, latent Spiral, latent Adaptive and LAMM N=3.
INR is compared on the strict 20-subject, 100-visit intersection shared by all
seven methods. Ordinary result labels omit latent width.

Completed older ODE and BrainODE results use a different cohort and protocol;
they are displayed only in the separate existing-architecture section. Smoke
runs never enter the cache. The load-only
`notebooks/all_methods_velocity_by_age.ipynb` explains every velocity metric
before displaying it; an error-free executed copy is stored beside it. See
`RUNBOOK.md` for reproducible commands.
