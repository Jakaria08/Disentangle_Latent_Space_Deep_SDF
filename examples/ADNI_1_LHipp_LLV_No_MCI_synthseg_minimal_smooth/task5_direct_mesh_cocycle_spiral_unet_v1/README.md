# Direct conditional surface cocycle: Spiral versus Adaptive Spiral

This isolated experiment learns longitudinal hippocampal deformation directly
on the registered 2,746-vertex surfaces created after SynthSeg. It does not
modify source meshes, existing latent experiments, or their checkpoints.

The scientific contract is deliberately narrow:

- no ODE or numerical integration;
- no global latent vector or bottleneck;
- fixed-topology mesh input and mesh output;
- actual source/target ages in years and a CN/AD condition;
- subject-disjoint train, validation, and test sets;
- identical data, objective, tuning space, and evaluation for the two operators.

## Model

For a source mesh `X`, source age `s`, target age `t`, and diagnosis `d`,

`Phi(X,s,t,d) = X + (t-s) G(X,s,t,d)`.

`G` is a conditional mesh U-Net and has physical units of mm/year. This form
makes `Phi(X,s,s,d) = X` exact, including before training. The hierarchy is
2,746 -> 1,373 -> 344 -> 86 vertices with skip connections. Age Fourier
features and diagnosis features modulate every resolution using FiLM. The
disease branch is a shared CN field plus an AD correction, rather than two
unrelated networks.

The Spiral model uses fixed ordered spiral neighborhoods everywhere. The
Adaptive model changes only three coarse operators: the 344-vertex encoder,
86-vertex bottleneck, and 344-vertex decoder. Each predicts a bounded,
vertex-specific effective neighborhood support. Fine levels stay fixed to
preserve local detail and keep memory bounded. The initial support is 8.5, not
an integer: integer initialization makes the interpolation derivative vanish.
The support predictor runs in FP32 under mixed precision to prevent gradient
underflow.

## Instantaneous velocity

The model-defined instantaneous surface velocity is

`v_model(X,a,d) = d Phi(X,a,t,d)/dt at t=a = G(X,a,a,d)`.

It is available from one mesh, its age, and its condition. It is not directly
measured ground truth. The comparison reference is the derivative of a smooth
trajectory fitted to each subject's observed registered visits after rigid
removal. Reliability weights reflect visit count and time span. The report
therefore calls it the **observed fitted longitudinal reference** and reports
vector RMSE/cosine, surface-normal RMSE/correlation/sign, high-change-region
overlap, and speed ratio.

## Optimization and evaluation

The objective combines endpoint correspondence and surface-normal error;
observed, virtual-time, and local cocycle consistency; inverse consistency;
the fitted longitudinal velocity comparison; volume and annualized volume-rate
agreement; CN/AD group-rate constraints; edge, Laplacian, tangent, and face-flip
regularization. Sampling is balanced over diagnosis, interval type, and subject.

The completed `main_v1` study is preserved as an endpoint-focused reference.
The recommended `velocity_v2` study starts from its trial-13 architecture and
uses a smaller search over width, dropout, learning rate, weight decay,
instantaneous-velocity weight, smoothness, and Adaptive initial support. Spiral
and Adaptive have independent, resumable SQLite studies. Smoke studies are
automatically suffixed `_smoke` and cannot contaminate a real search.

Velocity-v2 checkpoint selection uses

`endpoint_macro + 0.05 * 0.5 * (vector_RMSE/zero_vector_RMSE + normal_RMSE/zero_normal_RMSE)`.

Velocity quantities are pooled using the fitted-reference reliability weights.
The fitted derivative is not treated as directly measured ground truth. It is a
secondary selection signal and cannot overwhelm endpoint prediction. A
checkpoint must also beat the zero-velocity field, avoid speed collapse,
preserve stronger AD volume decline, have at most 0.1% mean flipped faces, and
keep mean relative cocycle and inverse defects below 0.02. The trainer never
loads test data. The frozen evaluator
reports correspondence error, ASSD, HD95, Chamfer, normal consistency, flipped
faces, mesh Dice, volume error/rate, instantaneous velocity, cocycle defect,
inverse defect, and averaged vertex maps. Expensive metrics can use a fixed
diagnosis/interval/subject-balanced subset while fast metrics still use every
pair. Final `summary.json` keeps the historical unweighted instantaneous-
velocity block and adds `instantaneous_velocity_reliability_weighted`, including
the vector/normal error ratios against predicting zero velocity.

`analyze_velocity_by_age.py` provides the complementary age-stratified view.
For CN and AD separately, it compares model instantaneous velocity with the
observed fitted longitudinal velocity in common-support age intervals. It
reports mean surface speed, inward-normal velocity, vector/normal agreement,
reference-reliability weighting, subject-bootstrap confidence intervals, and
the visit/subject count in every interval. The default 70--95 year range avoids
presenting the validation cohort's younger ages as a balanced disease
comparison: below age 70 there are only two CN subjects.

Prepared counts are 2,037/269/277 scans and 475/61/61 subjects for
train/validation/test. See [RUNBOOK.md](RUNBOOK.md) for every command and
[VALIDATION.md](VALIDATION.md) for completed implementation checks.
