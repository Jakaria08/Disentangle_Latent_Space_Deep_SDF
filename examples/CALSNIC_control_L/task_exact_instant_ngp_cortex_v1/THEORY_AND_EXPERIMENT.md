# Theory and experimental design

## 1. Conditional Instant-NGP

For subject `i`, one global latent `z_i` in R^256. At hash level `l` the grid has
`R_l = floor(16 * s^l)` vertices per axis over the shared `grid_aabb`, so the cell
size is `extent / (R_l - 1)`. A query point takes the trilinear blend of its eight
surrounding vertices,

    h_l(x) = sum_c w_c(x) * T_l[index(v_c)]

where `index` is bijective stride indexing when `R_l^3 <= 2^log2T` and the
Instant-NGP spatial hash `(x*1 XOR y*2654435761 XOR z*805459861) & (2^log2T - 1)`
otherwise. Hashing a level that fits its table only manufactures collisions;
tiny-cuda-nn makes the same split. Levels are concatenated,

    H(x) = [h_0(x), ..., h_15(x)]  in R^32

and the shape code is appended **after** encoding, constant over every point of
subject `i`:

    sdf(x, i) = MLP( xyz | H(x) | z_i )

This is what Compact-SDF describes for its Instant-NGP baseline: a shape-specific
latent code per object, plus `per_level_scale` reduced from 2.0 to 1.174.

Features are multiplied by the same C1 smoothstep ROI taper MR128 uses, so they
vanish exactly at the AABB and the far field is carried by `xyz` and `z` through
the MLP alone. Level weights implement coarse-to-fine unlocking.

## 2. Why the published scale is not enough here

The cohort shares one similarity scale: 1 normalized unit = 117.431971 mm. On the
shared AABB the finest level gives (geometric mean over axes)

| ladder | finest R | cell |
|---|---|---|
| MR128 dense | 128 | 1.192 mm |
| scale 1.174 (A1) | 177 | 0.860 mm |
| scale 1.2944 (A2/A3/B) | 767 | 0.198 mm |

A1 is about 1.4x finer per axis than MR128 -- a real but modest step, and still
coarser than the sub-millimetre gaps between opposing pial banks. A2 is 6x finer
than A1. Since a hash table's size is decoupled from `R^3`, buying that resolution
costs nothing structurally; only the collision load changes, which is exactly what
A3 controls for.

The evaluation lattice is a separate ceiling: marching cubes at 256 has a 0.921 mm
voxel and cannot represent a 0.5 mm feature at all. Fold claims are made at 512
(0.460 mm).

## 3. Compact-SDF two-branch model

Two fields over one shared code:

    global(x, z) = MLP_gen( xyz | z )                 8 x 512 ReLU, no grid input
    local(x, z)  = MLP_ovf( xyz | H(x) | z )          4 x 128 Softplus + hash

The global branch has no spatial feature input, so it cannot fragment; the local
branch carries detail. Training loss:

    L = w_gb * L1(global, broad) + w_gn * L1(global, near)
      + w_lb * L1(local,  broad) + w_ln * L1(local,  near)
      + w_f  * L1(fused,  near)
      + w_far * mean(relu(|local - global| - margin))     over |sdf| > near_band
      + w_eik * mean((|grad s| - 1)^2)                    over near-surface points
      + w_z * mean(z^2) + w_grid * mean(T^2)

All L1 terms are clamped at 0.1, matching MR64/MR128.

`far_field_agreement` is the direct answer to the failure mode: hash features are
active everywhere inside the ROI, so the overfitting branch can invent zero
crossings far from the surface. Outside the near band the branches must agree to
within a margin; inside it the local branch is free.

### Fusion

Two readouts, both exported from one checkpoint and one fitted latent.

**`fused_hard_band`** -- the paper's stated inference, applied on the
reconstruction lattice, not pointwise:

1. evaluate the global branch on the full `R^3` lattice;
2. mark cells where the sign changes between 6-neighbours;
3. dilate by `n = 3` layers (the paper's bandwidth);
4. evaluate the local branch only inside that band and overwrite in place;
5. marching cubes on the fused volume.

At R = 512 the band is a few million points, about 2 s of decode. The released
Compact-SDF `gl_sdf` returns the overfitting prediction regardless of its
`low_tag`, so the paper's "local inside band, global elsewhere" behaviour has to be
written explicitly; this is that implementation.

**`fused_smooth_gate`** -- continuous alternative:

    g(x)   = exp(-(global(x)/tau)^2) * roi_taper(x)
    fused  = global + g * (local - global)

Detail is admitted only where the generalization branch already places a surface,
and there is no band-boundary discontinuity -- the residual noise the paper flags
as a bandwidth-choice risk. `tau = 0.02` normalized = 2.3 mm.

`global_only` and `local_only` complete the ablation, so one training run answers
"is the gain from the hash encoding, from the fusion, or from neither".

## 4. Eikonal epsilon, pinned in millimetres

`multires_common.finite_difference_epsilon` sets `eps = extent / (R_active - 1)`.
At R = 767 that is 0.26 mm, where a central difference on a hash field is dominated
by interpolation noise rather than the field. Because the cohort shares one uniform
scale, an isotropic epsilon in normalized units is an isotropic epsilon in
millimetres for every subject, so this task pins `eikonal.epsilon_mm = 0.5`
(0.35 mm after epoch 1800) and keeps the single batched six-offset evaluation --
no spatial double-backward.

`eikonal.branches` selects which fields are constrained: `["fused"]` for the
single-field arms, `["fused", "global"]` for the two-branch model, where keeping
the global branch a proper distance function is what makes the hard-band surface
search reliable.

## 5. Hash capacity and collisions

`estimated_surface_cells = reference_surface_area_mm2 / cell_mm^2` approximates how
many cells one cortical surface crosses at a level; divided by table entries it
gives the collision load Instant-NGP has to disambiguate through the MLP.

| level | A1 (2^19) | A2 (2^19) | A3 (2^22) |
|---|---|---|---|
| finest cell | 0.860 mm | 0.198 mm | 0.198 mm |
| finest load | 0.26 | 4.88 | 0.61 |
| grid params | 7.79 M | 10.98 M | 66.09 M |

A2 sits at MR128's parameter budget with heavy collisions at the top; A3 removes
the collisions at 6x the parameters. If A3 >> A2 the answer is "buy table
entries"; if A3 ~ A2 collisions were never the binding constraint.

## 6. What each arm can and cannot show

- **A1** places Compact-SDF's published NGP baseline on this cohort. Expected to
  land between MR128 and A2. If it already beats MR128 substantially, the gain is
  from the hash encoding itself rather than from resolution.
- **A2 vs MR128** is the resolution test at matched shared capacity and identical
  sampling, split, latent size, optimizer and schedule.
- **A3 vs A2** is the capacity/collision test; these two differ only in
  `log2_hashmap_size`.
- **B vs A2** is the fusion test; the local branch of B uses A2's exact hash
  configuration, so the difference is the global branch, the fusion and the
  far-field term.
- **PCA-172 remains an oracle**, projecting each target onto a train-only basis
  with exact vertex correspondence and fixed topology. It is a strong upper
  reference, not a like-for-like generative baseline; beating it on ASSD is not
  required for the INR to be the more useful representation, but the gap is the
  honest measure of how much geometry the 256-D code still loses.

Report distances **and** topology together: `predicted_connected_components`,
`predicted_euler_number`, and boundary/non-manifold edge counts. A fragmented mesh
can score better on sampled surface distance while being worse as a shape.
