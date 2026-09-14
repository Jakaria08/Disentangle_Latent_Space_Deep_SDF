# Verification record

Date: 2026-08-29

## Completed checks

- Python compilation: passed.
- Assertion suite: 13 passed, 0 failed.
- Exact latest-LAMM layout: 43+86 regions; each scale independently partitions all 2,746
  vertices.
- Prepared data: 2,037 train scans / 475 subjects; 269 validation scans / 61 subjects;
  277 test scans / 61 subjects; subject isolation passed.
- Real-mesh forward validation for global 256D, global 384D, and regional tokens: passed.
- Exact identity at `source_age == target_age`: `0.0 mm` for all three.
- Diagonal finite-difference error: `0.0002514 mm/year`, below the `0.0005` gate.
- Diagnosis branch changes velocity: passed.
- GPU1 end-to-end smoke training: passed for all three models.
- Every required condition, tokenizer, encoder, flow, decoder, and head group accumulated
  nonzero gradient in every applicable model.
- Training smoke checkpoints recorded `test_data_loaded=false`.
- Direct evaluator adapter: passed for new LAMM and selected Spiral checkpoints on GPU1.
- Latent decoder-JVP velocity adapter: passed on the current 128D LAMM flow on GPU1.
- Central aggregation: passed separately for direct and latent branches.
- Revised Optuna study: three queued architecture anchors ran successfully on GPU1 and
  exported the study/best configuration.

## Measured trainable parameters

- Global 256D `[96,160]`: 33,431,224.
- Global 384D `[128,256]`: 36,771,640.
- Regional tokens `[129,256]`, no global vector: 28,554,706.

The six-channel multiscale regional heads account for 16,510,194 parameters and produce the
CN base plus AD-residual velocity fields.

## Smoke-result scope

One smoke epoch proves execution, data isolation, CUDA operation, and gradient reach; it does
not demonstrate convergence. As expected, the feasibility gates were false and the smoke
`best.pt` remained the epoch-0 no-change checkpoint. Do not report smoke metrics as model
performance.
