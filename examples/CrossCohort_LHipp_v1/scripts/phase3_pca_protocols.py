#!/usr/bin/env python3
"""Phase 3: the complete protocol matrix for PCA.

PCA answers the whole experimental question cheaply, before any GPU time is spent.  It runs
all four settings:

``internal``  fit on each cohort's own train split; evaluate its own val/test.
``external``  the stored reference (ADNI) basis applied unchanged.  Nothing is fitted on the
              target cohort, so this is a true external validation rather than a partial refit.
``pooled``    fit on the pooled train splits of the pooled-eligible cohorts; evaluate each
              cohort's test split separately.
``loco``      leave one cohort out: fit on the others' train splits, evaluate the held-out
              cohort.  This is the protocol BrainODE used for its cross-benchmark experiment.

Because the reference basis is stored with a fixed number of components, ``external`` is
only available up to that rank; higher ranks are reported as unavailable rather than
silently substituted with a refit.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

import xcohort_common as xc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocols", nargs="+", default=["internal", "external", "pooled", "loco"])
    parser.add_argument("--components", nargs="+", type=int, default=None)
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--cohorts", nargs="+", default=None)
    return parser.parse_args()


class CohortData:
    """Vertices and faces for one cohort, loaded once and reused by every protocol."""

    def __init__(self, spec: xc.CohortSpec, config: xc.Config):
        self.spec = spec
        self.rows = xc.read_manifest(spec, config)
        self.faces = xc.load_faces(self.rows)
        self.splits = {split: xc.load_vertices(spec, self.rows, split) for split in xc.SPLITS}

    @property
    def name(self) -> str:
        return self.spec.name


def evaluate(model: xc.PCAModel, data: CohortData, split: str) -> dict[str, float]:
    vertices = data.splits[split]
    return xc.reconstruction_metrics(model.reconstruct(vertices), vertices, faces=data.faces)


def emit(rows_out: list[dict], *, protocol: str, k: int, fit: str, data: CohortData, split: str,
         metrics: dict[str, float], source: str, seed: int, notes: str = "") -> None:
    rows_out.append(
        xc.result_row(
            protocol=protocol, model="pca", latent_k=k, fit_cohorts=fit, eval_cohort=data.name,
            split=split, n_scans=len(data.splits[split]),
            vertex_rmse_mm_mean=metrics["vertex_rmse_mm_mean"],
            vertex_rmse_mm_median=metrics.get("vertex_rmse_mm_median", ""),
            vertex_rmse_mm_p95=metrics.get("vertex_rmse_mm_p95", ""),
            vertex_euclidean_mm_mean=metrics["vertex_euclidean_mm_mean"],
            volume_abs_relative_error_pct_mean=metrics.get("volume_abs_relative_error_pct_mean", ""),
            normalization_source=source, seed=seed, notes=notes,
        )
    )


def main() -> int:
    args = parse_args()
    config = xc.load_config()
    hparams = xc.load_hyperparameters()
    components = args.components or hparams["pca_components"]
    reference = config.reference()

    names = args.cohorts or list(config.cohorts)
    data: dict[str, CohortData] = {}
    for name in names:
        try:
            data[name] = CohortData(config.cohorts[name], config)
        except FileNotFoundError as exc:
            print(f"[{name}] SKIPPED: {exc}", flush=True)
    if not data:
        raise SystemExit("No cohort has a keep manifest yet; run phase 1 first.")

    rows_out: list[dict] = []
    reference_pca = xc.load_reference_pca(reference)

    max_k = max(components)

    if "internal" in args.protocols:
        print("\n--- internal: fit on each cohort's own train split ---", flush=True)
        for name, cohort in data.items():
            train = cohort.splits["train"]
            # One SVD per fit-set, then truncate: re-fitting per rank would repeat the same
            # decomposition and could return sign-flipped components between ranks.
            full = xc.fit_pca(train, max_k)
            for k in components:
                model = full.truncated(min(k, full.k))
                note = "" if model.k == k else f"rank capped at {model.k}"
                for split in args.splits:
                    metrics = evaluate(model, cohort, split)
                    emit(rows_out, protocol="internal", k=k, fit=name, data=cohort, split=split,
                         metrics=metrics, source=f"self:{name}", seed=config.seed, notes=note)
                print(f"  {name:8} k={k:3d} " + " ".join(
                    f"{s}={evaluate(model, cohort, s)['vertex_rmse_mm_mean']:.6f}" for s in args.splits), flush=True)

    if "external" in args.protocols:
        print(f"\n--- external: stored {reference.name} basis applied unchanged ---", flush=True)
        for name, cohort in data.items():
            for k in components:
                if k > reference_pca.k:
                    print(f"  {name:8} k={k:3d} unavailable (stored basis holds {reference_pca.k})", flush=True)
                    continue
                model = reference_pca.truncated(k)
                for split in args.splits:
                    metrics = evaluate(model, cohort, split)
                    emit(rows_out, protocol="external", k=k, fit=reference.name, data=cohort, split=split,
                         metrics=metrics, source=f"reference:{reference.name}", seed=config.seed)
                print(f"  {name:8} k={k:3d} " + " ".join(
                    f"{s}={evaluate(model, cohort, s)['vertex_rmse_mm_mean']:.6f}" for s in args.splits), flush=True)

    pooled_names = [c.name for c in config.pooled_members() if c.name in data]
    if "pooled" in args.protocols and len(pooled_names) > 1:
        print(f"\n--- pooled: fit on {'+'.join(pooled_names)} train splits ---", flush=True)
        pooled_train = np.concatenate([data[n].splits["train"] for n in pooled_names], axis=0)
        pooled_full = xc.fit_pca(pooled_train, max_k)
        for k in components:
            model = pooled_full.truncated(min(k, pooled_full.k))
            for name, cohort in data.items():
                for split in args.splits:
                    metrics = evaluate(model, cohort, split)
                    emit(rows_out, protocol="pooled", k=k, fit="+".join(pooled_names), data=cohort,
                         split=split, metrics=metrics, source="pooled", seed=config.seed,
                         notes="" if cohort.spec.pooled_eligible else "cohort not in the pooled fit")
            print(f"  k={k:3d} " + " ".join(
                f"{n}={evaluate(model, data[n], 'test')['vertex_rmse_mm_mean']:.6f}" for n in data), flush=True)

    if "loco" in args.protocols and len(pooled_names) > 1:
        print("\n--- loco: fit on every pooled-eligible cohort except the held-out one ---", flush=True)
        for held_out, cohort in data.items():
            fit_names = [n for n in pooled_names if n != held_out]
            if not fit_names:
                continue
            train = np.concatenate([data[n].splits["train"] for n in fit_names], axis=0)
            loco_full = xc.fit_pca(train, max_k)
            for k in components:
                model = loco_full.truncated(min(k, loco_full.k))
                for split in args.splits:
                    metrics = evaluate(model, cohort, split)
                    emit(rows_out, protocol="loco", k=k, fit="+".join(fit_names), data=cohort,
                         split=split, metrics=metrics, source="loco", seed=config.seed)
            primary = loco_full.truncated(min(int(hparams["primary_k"]), loco_full.k))
            print(f"  held out {held_out:8} fit on {'+'.join(fit_names):20} "
                  f"test={evaluate(primary, cohort, 'test')['vertex_rmse_mm_mean']:.6f}", flush=True)

    xc.write_results(xc.TASK_ROOT / "reports" / "phase3_pca_protocols.csv", rows_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
