#!/usr/bin/env python3
"""Ablation A1: the August exact-coboundary and volume-coboundary-v2 cocycles, trained to completion.

The August trainers (configs/ablation_sources.json, sha256-verified on import) run unchanged in this
process. Three things are redirected, by data rather than by editing the trainers:

* the registry is this experiment's view registry. Its ``output_root`` is a small shim folder whose
  ``representations`` link points at the view's code archives and whose ``training`` link points at
  runs/<view>, so the run lands in runs/<view>/<representation>/<method>/<run_name> like every other
  run. The shim lives outside runs/ so no directory links back into itself;
* ``cached_vertices`` returns the view's real meshes in archive order (task3's cache is ADNI-only);
* the config is the August config for the representation with the requested seed and
  ``early_stopping_patience`` above the epoch count, so every epoch runs. The August runs stopped at
  epochs 42-43 of 160-180. best.pt is still the best feasible validation epoch.

The trainers open train and val only. Their checkpoints do not record the view, so evaluation jobs
pass --trained-view; ablation_provenance.json is written next to the checkpoints.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import benchmark_common as bc
import dynamics_core as D

VARIANTS = {"exact": ("exact_coboundary_c4", "config_exact_{rep}"),
            "volume_v2": ("volume_exact_coboundary_c4_v2", "config_volume_v2_{rep}")}
SHIM_ROOT = bc.BULK_ROOT / "stage5_brainode_style" / "ablations" / "august_registry_shim"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--view", default="p0_internal_adni")
    parser.add_argument("--representation", required=True, choices=("pca128", "spiralnet128"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--runs-root", type=Path, default=D.RUNS_ROOT)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def shim_root(runs_root: Path, view: str) -> Path:
    root = bc.require_bulk(SHIM_ROOT / f"{runs_root.name}__{view}", "coboundary registry shim")
    root.mkdir(parents=True, exist_ok=True)
    training = bc.require_bulk(runs_root / view, "run root")
    training.mkdir(parents=True, exist_ok=True)
    for name, target in (("representations", D.view_root(view) / "representations"), ("training", training)):
        link = root / name
        if link.is_symlink():
            if link.resolve() != target.resolve():
                raise RuntimeError(f"{link} points to {link.resolve()}, expected {target.resolve()}")
        elif link.exists():
            raise RuntimeError(f"{link} exists and is not a link")
        else:
            link.symlink_to(target.resolve(), target_is_directory=True)
    return root


def main() -> int:
    args = parse_args()
    bc.require_allowed_gpu(args.device)
    method, source_key = VARIANTS[args.variant]
    C = D.core()["C"]
    trainer = D.coboundary_modules(method)["trainer"]
    source = bc.read_json(bc.CONFIG_DIR / "ablation_sources.json")["files"][source_key.format(rep=args.representation)]
    august_config = bc.read_json(bc.resolve(source["path"]))
    if august_config["representation"] != args.representation or august_config["method"] != method:
        raise ValueError(f"{source['path']} is not {method} on {args.representation}")

    epochs = int(args.epochs if args.epochs is not None else august_config["training"]["epochs"])
    run_name = args.run_name or f"{args.representation}_{method}_complete_s{args.seed}"
    config = copy.deepcopy(august_config)
    config["name"] = run_name
    config["training"].update({"seed": int(args.seed), "run_name": run_name, "early_stopping_patience": epochs + 1})
    config["stage5_ablation"] = {
        "id": "A1", "source_config": source, "view": args.view,
        "change": "seed and run name set; early_stopping_patience = epochs + 1 (trained to completion)",
    }
    method_root = bc.require_bulk(args.runs_root / args.view / args.representation / method, "run root")
    config_path = bc.atomic_json(method_root / "_configs" / f"{run_name}.json", config)

    registry = D.view_registry(args.view)
    registry["output_root"] = str(shim_root(args.runs_root, args.view))
    train_archive = C.load_archive(args.representation, "train", registry)
    val_archive = C.load_archive(args.representation, "val", registry)
    D.assert_no_test_leakage(args.view, [train_archive, val_archive])
    meshes = {"train": D.view_vertices(train_archive), "val": D.view_vertices(val_archive)}

    def view_cached_vertices(split: str, _registry=None, mmap: bool = True):
        if split not in meshes:
            raise PermissionError(f"the {split} split is sealed during training")
        return meshes[split]

    C.load_registry = lambda *_args, **_kwargs: registry
    C.cached_vertices = view_cached_vertices

    argv = ["--config", str(config_path), "--device", args.device, "--run-name", run_name, "--seed", str(args.seed), "--epochs", str(epochs)]
    argv += [flag for flag, on in (("--resume", args.resume), ("--dry-run", args.dry_run), ("--smoke", args.smoke)) if on]
    print(f"A1 {method} | view={args.view} | {args.representation} | seed={args.seed} | epochs={epochs} (no early stopping)", flush=True)
    sys.argv = [str(trainer.__file__), *argv]
    code = int(trainer.main())
    if code == 0 and not args.dry_run:
        run_dir = method_root / run_name
        bc.atomic_json(run_dir / "ablation_provenance.json", {
            "ablation": "A1", "method": method, "view": args.view, "representation": args.representation, "seed": args.seed,
            "epochs_requested": epochs, "config": str(config_path), "sources": bc.read_json(bc.CONFIG_DIR / "ablation_sources.json"),
            "redirections": ["registry -> view registry with shim output_root", "cached_vertices -> view meshes (train/val only)"],
            "view_provenance": D.provenance(args.view, args.representation, config, args.seed),
        })
    return code


if __name__ == "__main__":
    raise SystemExit(main())
