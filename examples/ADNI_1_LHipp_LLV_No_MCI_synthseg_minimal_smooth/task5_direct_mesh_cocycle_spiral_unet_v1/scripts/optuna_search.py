#!/usr/bin/env python3
"""Resumable Optuna search for Spiral or Adaptive direct surface cocycles."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import re
from pathlib import Path

import optuna

import common as C
from train import train_experiment, validate_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", choices=("spiral", "adaptive"), required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--trial-epochs", type=int, default=40)
    parser.add_argument("--trial-samples-per-epoch", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--study-tag",
        default="main_v1",
        help="Version tag used in the SQLite study, output directory, and trial run names.",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def base_config(operator: str) -> dict:
    name = "adaptive_direct_c4_s42.json" if operator == "adaptive" else "spiral_direct_c4_s42.json"
    return C.read_json(C.TASK_ROOT / "configs" / name)


def sampled_config(
    trial: optuna.Trial,
    operator: str,
    args: argparse.Namespace,
    effective_tag: str,
) -> dict:
    config = copy.deepcopy(base_config(operator))
    base = trial.suggest_categorical("base_channels", [16, 24, 32])
    config["model"]["channels"] = [base, 2 * base, 3 * base, 4 * base]
    config["model"]["condition_dim"] = trial.suggest_categorical("condition_dim", [32, 64, 96])
    config["model"]["time_frequencies"] = trial.suggest_int("time_frequencies", 1, 4)
    config["model"]["dropout"] = trial.suggest_float("dropout", 0.0, 0.25, step=0.05)
    if operator == "adaptive":
        config["model"]["adaptive_initial_support"] = trial.suggest_categorical(
            "adaptive_initial_support", [4.5, 8.5, 12.5]
        )
    training = config["training"]
    training["learning_rate"] = trial.suggest_float("learning_rate", 5.0e-5, 5.0e-4, log=True)
    training["weight_decay"] = trial.suggest_float("weight_decay", 1.0e-6, 5.0e-3, log=True)
    training["minimum_learning_rate"] = training["learning_rate"] * 0.1
    training["epochs"] = int(args.trial_epochs)
    training["samples_per_epoch"] = int(args.trial_samples_per_epoch)
    training["early_stopping_patience"] = min(12, max(4, int(args.trial_epochs) // 3))
    training["seed"] = int(args.seed)
    config["loss"]["virtual_cocycle_weight"] = trial.suggest_float(
        "virtual_cocycle_weight", 0.03, 0.2, log=True
    )
    config["loss"]["inverse_weight"] = trial.suggest_float("inverse_weight", 0.01, 0.1, log=True)
    config["loss"]["diagonal_velocity_weight"] = trial.suggest_categorical(
        "diagonal_velocity_weight", [0.0, 0.02, 0.05, 0.1]
    )
    config["loss"]["smoothness_weight"] = trial.suggest_float(
        "smoothness_weight", 0.001, 0.02, log=True
    )
    config["training"]["run_name"] = f"optuna_{operator}_{effective_tag}_trial_{trial.number:04d}"
    config["name"] = config["training"]["run_name"]
    validate_config(config)
    return config


def export_study(study: optuna.Study, directory: Path, operator: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for trial in study.trials:
        rows.append(
            {
                "number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                "run_name": trial.user_attrs.get("run_name", ""),
                **{f"param_{name}": value for name, value in trial.params.items()},
            }
        )
    fields = sorted({key for row in rows for key in row}) if rows else []
    with (directory / "trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    complete = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if complete:
        best = study.best_trial
        config = copy.deepcopy(base_config(operator))
        base = int(best.params["base_channels"])
        config["model"]["channels"] = [base, 2 * base, 3 * base, 4 * base]
        for key in ("condition_dim", "time_frequencies", "dropout", "adaptive_initial_support"):
            if key in best.params:
                config["model"][key] = best.params[key]
        config["training"]["learning_rate"] = best.params["learning_rate"]
        config["training"]["minimum_learning_rate"] = best.params["learning_rate"] * 0.1
        config["training"]["weight_decay"] = best.params["weight_decay"]
        for key in ("virtual_cocycle_weight", "inverse_weight", "diagonal_velocity_weight", "smoothness_weight"):
            config["loss"][key] = best.params[key]
        config["name"] = f"direct_mesh_{operator}_optuna_best"
        config["training"]["run_name"] = config["name"]
        C.atomic_json(directory / "best_config.json", config)
        C.atomic_json(
            directory / "best_trial.json",
            {"number": best.number, "value": best.value, "params": best.params, "user_attrs": best.user_attrs},
        )


def main() -> int:
    args = parse_args()
    root = C.output_root(args.output_root)
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(args.study_tag)).strip("_.-")
    if not tag:
        raise ValueError("--study-tag must contain at least one letter or number")
    # A smoke invocation can never contaminate a real study, even when the caller
    # forgets to provide a separate tag.
    effective_tag = f"{tag}_smoke" if args.smoke else tag
    study_dir = root / "optuna" / f"{args.operator}_direct_c4_{effective_tag}"
    study_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{study_dir / 'study.db'}"
    study = optuna.create_study(
        study_name=f"direct_mesh_{args.operator}_c4_{effective_tag}",
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=int(args.seed)),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=5),
    )

    def objective(trial: optuna.Trial) -> float:
        config = sampled_config(trial, args.operator, args, effective_tag)
        run_name = config["training"]["run_name"]
        trial.set_user_attr("run_name", run_name)
        status = train_experiment(
            config,
            args.device,
            root,
            run_name=run_name,
            seed_override=int(args.seed) + trial.number,
            epochs_override=2 if args.smoke else int(args.trial_epochs),
            resume=False,
            smoke=bool(args.smoke),
            trial=trial,
        )
        trial.set_user_attr("best_epoch", int(status["best_epoch"]))
        trial.set_user_attr("checkpoint", str(status["selected_checkpoint"]))
        return float(status["best_score"])

    trials = 2 if args.smoke else int(args.n_trials)
    try:
        study.optimize(objective, n_trials=trials, gc_after_trial=True)
    finally:
        export_study(study, study_dir, args.operator)
    print(json.dumps({"study": study.study_name, "trials": len(study.trials), "directory": str(study_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
