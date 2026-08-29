#!/usr/bin/env python3
"""Focused velocity-aware Optuna search for direct Spiral surface cocycles."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import re
from pathlib import Path

import optuna
import torch

import common as C
from train import train_experiment, validate_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", choices=("spiral", "adaptive"), required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--n-trials", type=int, default=16)
    parser.add_argument("--trial-epochs", type=int, default=80)
    parser.add_argument("--trial-samples-per-epoch", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--study-tag",
        default="main_v2",
        help="Version tag used in the isolated velocity-aware study and trial run names.",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def base_config(operator: str) -> dict:
    name = (
        "adaptive_direct_c4_velocity_v2_s42.json"
        if operator == "adaptive"
        else "spiral_direct_c4_velocity_v2_s42.json"
    )
    return C.read_json(C.TASK_ROOT / "configs" / name)


def anchor_params(operator: str) -> dict:
    params = {
        "base_channels": 32,
        "dropout": 0.15,
        "learning_rate": 0.00036302570115823044,
        "weight_decay": 0.000008977558447767681,
        "diagonal_velocity_weight": 0.1,
        "smoothness_weight": 0.003,
    }
    if operator == "adaptive":
        params["adaptive_initial_support"] = 8.5
    return params


def sampled_config(
    trial: optuna.Trial,
    operator: str,
    args: argparse.Namespace,
    effective_tag: str,
) -> dict:
    """Search only the parameters still uncertain after the completed v1 study."""
    config = copy.deepcopy(base_config(operator))
    base = trial.suggest_categorical("base_channels", [32, 48])
    config["model"]["channels"] = [base, 2 * base, 3 * base, 4 * base]
    config["model"]["dropout"] = trial.suggest_categorical("dropout", [0.1, 0.15, 0.2])
    if operator == "adaptive":
        config["model"]["adaptive_initial_support"] = trial.suggest_categorical(
            "adaptive_initial_support", [4.5, 8.5, 12.5]
        )
    training = config["training"]
    training["learning_rate"] = trial.suggest_float("learning_rate", 2.5e-4, 7.0e-4, log=True)
    training["weight_decay"] = trial.suggest_float("weight_decay", 1.0e-6, 1.0e-4, log=True)
    training["minimum_learning_rate"] = training["learning_rate"] * 0.1
    training["epochs"] = int(args.trial_epochs)
    training["samples_per_epoch"] = int(args.trial_samples_per_epoch)
    training["early_stopping_patience"] = min(24, max(16, int(args.trial_epochs) // 4))
    training["seed"] = int(args.seed)
    config["loss"]["diagonal_velocity_weight"] = trial.suggest_categorical(
        "diagonal_velocity_weight", [0.05, 0.1, 0.2, 0.3]
    )
    config["loss"]["smoothness_weight"] = trial.suggest_categorical(
        "smoothness_weight", [0.0015, 0.003, 0.006]
    )
    run_name = f"optuna_{operator}_velocity_{effective_tag}_trial_{trial.number:04d}"
    training["run_name"] = run_name
    config["name"] = run_name
    validate_config(config)
    return config


def apply_best_params(config: dict, params: dict) -> dict:
    output = copy.deepcopy(config)
    base = int(params["base_channels"])
    output["model"]["channels"] = [base, 2 * base, 3 * base, 4 * base]
    output["model"]["dropout"] = params["dropout"]
    if "adaptive_initial_support" in params:
        output["model"]["adaptive_initial_support"] = params["adaptive_initial_support"]
    output["training"]["learning_rate"] = params["learning_rate"]
    output["training"]["minimum_learning_rate"] = params["learning_rate"] * 0.1
    output["training"]["weight_decay"] = params["weight_decay"]
    output["loss"]["diagonal_velocity_weight"] = params["diagonal_velocity_weight"]
    output["loss"]["smoothness_weight"] = params["smoothness_weight"]
    return output


def export_study(study: optuna.Study, directory: Path, operator: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for trial in study.trials:
        rows.append(
            {
                "number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                **{f"param_{name}": value for name, value in trial.params.items()},
                **{f"result_{name}": value for name, value in trial.user_attrs.items()},
            }
        )
    fields = sorted({key for row in rows for key in row}) if rows else []
    with (directory / "trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    complete = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        return
    best = study.best_trial
    config = apply_best_params(base_config(operator), best.params)
    config["name"] = f"direct_mesh_{operator}_velocity_v2_optuna_best"
    config["training"]["run_name"] = config["name"]
    validate_config(config)
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
    effective_tag = f"{tag}_smoke" if args.smoke else tag
    study_dir = root / "optuna" / f"{args.operator}_direct_c4_velocity_{effective_tag}"
    study_dir.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=f"direct_mesh_{args.operator}_c4_velocity_{effective_tag}",
        storage=f"sqlite:///{study_dir / 'study.db'}",
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=int(args.seed)),
        # Velocity and consistency losses ramp for 15 epochs. Pruning before
        # then would select merely for fast endpoint convergence.
        pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=15, interval_steps=3),
    )
    if not study.trials:
        study.enqueue_trial(anchor_params(args.operator))

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
        checkpoint = Path(status["selected_checkpoint"])
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        validation = payload["validation"]
        selection = validation["selection_metrics"]
        trial.set_user_attr("best_epoch", int(status["best_epoch"]))
        trial.set_user_attr("checkpoint", str(checkpoint))
        trial.set_user_attr("feasible", bool(validation["feasible"]))
        for key in (
            "macro_endpoint_ratio",
            "velocity_normalized_error_ratio",
            "velocity_speed_ratio",
            "predicted_cn_log_volume_rate_per_year",
            "predicted_ad_log_volume_rate_per_year",
            "first_last_flipped_face_fraction",
        ):
            if key in selection:
                trial.set_user_attr(key, float(selection[key]))
        if not args.smoke and not bool(validation["feasible"]):
            raise optuna.TrialPruned("Trial ended without a feasible velocity-aware checkpoint")
        return float(status["best_score"])

    trials = 2 if args.smoke else int(args.n_trials)
    try:
        study.optimize(objective, n_trials=trials, gc_after_trial=True)
    finally:
        export_study(study, study_dir, args.operator)
    print(
        json.dumps(
            {"study": study.study_name, "trials": len(study.trials), "directory": str(study_dir)},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
