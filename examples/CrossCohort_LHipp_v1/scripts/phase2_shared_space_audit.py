#!/usr/bin/env python3
"""Phase 2: does the shared vertex space actually hold across cohorts?

This is the gate before any GPU work.  Three questions, cheapest first:

1. **Topology.**  Do the meshes really share the reference space?  Face connectivity and the
   topology hash are compared against the reference cohort, not merely against each other.
2. **Coverage.**  Does the reference (ADNI) PCA basis reconstruct each cohort?  The number
   that matters is not the absolute error but the gap against the cohort's own PCA at the
   same rank - that gap is the part of the anatomy the reference basis does not span.
3. **Latent shift.**  Where do the cohort's coefficients sit relative to the reference
   training distribution?  A cohort can reconstruct well while living in a region the
   reference never saw, and that distinction changes how you read every later result.

If step 1 fails, nothing downstream is meaningful and the script says so loudly.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

import xcohort_common as xc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cohorts", nargs="+", default=None)
    parser.add_argument("--k", type=int, default=128, help="Rank used for the coverage comparison.")
    parser.add_argument("--split", default="test", choices=[*xc.SPLITS, "all"])
    return parser.parse_args()


def cohort_rows(spec: xc.CohortSpec, config: xc.Config, split: str):
    rows = xc.read_manifest(spec, config)
    return rows, (rows if split == "all" else xc.split_rows(rows, split))


def stack_for(spec: xc.CohortSpec, rows, split: str) -> np.ndarray:
    if split != "all":
        return xc.load_vertices(spec, rows, split)
    return np.concatenate([xc.load_vertices(spec, rows, s) for s in xc.SPLITS], axis=0)


def main() -> int:
    args = parse_args()
    config = xc.load_config()
    reference = config.reference()
    ref_rows = xc.read_manifest(reference, config)
    ref_faces = xc.load_faces(ref_rows)
    ref_pca_full = xc.load_reference_pca(reference)
    ref_pca = ref_pca_full.truncated(args.k)

    # Reference coefficient distribution, for the latent-shift comparison.
    ref_train = xc.load_vertices(reference, ref_rows, "train")
    ref_coeff = ref_pca.encode(ref_train)
    ref_mu, ref_sigma = ref_coeff.mean(axis=0), ref_coeff.std(axis=0) + 1e-12

    names = args.cohorts or list(config.cohorts)
    rows_out, report = [], {}
    topology_ok = True

    for name in names:
        spec = config.cohorts[name]
        try:
            rows, _selected = cohort_rows(spec, config, args.split)
        except FileNotFoundError as exc:
            print(f"[{name}] SKIPPED: {exc}", flush=True)
            continue

        faces = xc.load_faces(rows)
        faces_match = bool(np.array_equal(faces, ref_faces))
        topology_ok &= faces_match
        vertices = stack_for(spec, rows, args.split)

        internal_note = ""
        if spec.name == reference.name:
            internal = ref_pca  # the reference basis is its own internal model
        else:
            train = xc.load_vertices(spec, rows, "train")
            internal = xc.fit_pca(train, args.k)
            if internal.k < args.k:
                internal_note = f"internal rank capped at {internal.k} by train size {len(train)}"

        external_rec = ref_pca.reconstruct(vertices)
        internal_rec = internal.reconstruct(vertices)
        external = xc.reconstruction_metrics(external_rec, vertices, faces=faces)
        internal_metrics = xc.reconstruction_metrics(internal_rec, vertices, faces=faces)

        coeff = ref_pca.encode(vertices)
        z = (coeff - ref_mu) / ref_sigma
        report[name] = {
            "faces_identical_to_reference": faces_match,
            "scans_evaluated": int(len(vertices)),
            "external_vertex_rmse_mm": external["vertex_rmse_mm_mean"],
            "internal_vertex_rmse_mm": internal_metrics["vertex_rmse_mm_mean"],
            "external_minus_internal_mm": external["vertex_rmse_mm_mean"] - internal_metrics["vertex_rmse_mm_mean"],
            "mean_abs_z_vs_reference_train": float(np.abs(z).mean()),
            "p95_abs_z_vs_reference_train": float(np.percentile(np.abs(z), 95)),
            "note": internal_note,
        }
        for label, metrics, source in (
            ("external", external, f"reference:{reference.name}"),
            ("internal", internal_metrics, f"self:{name}"),
        ):
            rows_out.append(
                xc.result_row(
                    protocol=f"phase2_{label}", model="pca", latent_k=args.k,
                    fit_cohorts=reference.name if label == "external" else name,
                    eval_cohort=name, split=args.split, n_scans=len(vertices),
                    vertex_rmse_mm_mean=metrics["vertex_rmse_mm_mean"],
                    vertex_rmse_mm_median=metrics.get("vertex_rmse_mm_median", ""),
                    vertex_rmse_mm_p95=metrics.get("vertex_rmse_mm_p95", ""),
                    vertex_euclidean_mm_mean=metrics["vertex_euclidean_mm_mean"],
                    volume_abs_relative_error_pct_mean=metrics.get("volume_abs_relative_error_pct_mean", ""),
                    normalization_source=source, seed=config.seed, notes=internal_note,
                )
            )
        print(
            f"[{name}] faces_match={faces_match} n={len(vertices)} "
            f"external={external['vertex_rmse_mm_mean']:.6f} internal={internal_metrics['vertex_rmse_mm_mean']:.6f} "
            f"gap={report[name]['external_minus_internal_mm']:+.6f} "
            f"mean|z|={report[name]['mean_abs_z_vs_reference_train']:.2f}",
            flush=True,
        )

    xc.write_results(xc.TASK_ROOT / "reports" / f"phase2_audit_{args.split}_k{args.k}.csv", rows_out)
    xc.write_json(
        xc.TASK_ROOT / "reports" / f"phase2_audit_{args.split}_k{args.k}.json",
        {"k": args.k, "split": args.split, "reference": reference.name,
         "topology_ok": topology_ok, "cohorts": report},
    )
    if not topology_ok:
        print("\nGATE FAILED: at least one cohort does not share the reference topology.", flush=True)
        return 1
    print("\nGate passed: every evaluated cohort shares the reference vertex space.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
