#!/usr/bin/env python3
"""Optuna search for the end-to-end direct LAMM surface cocycle."""

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


LATENT_LAYOUTS = {
    "z128_equal": (128, [64, 64], 256),
    "z256_equal": (256, [128, 128], 384),
    "z256_fine": (256, [96, 160], 384),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--n-trials", type=int, default=18)
    parser.add_argument("--trial-epochs", type=int, default=80)
    parser.add_argument("--trial-samples-per-epoch", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--study-tag", default="main_v1")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def base_config() -> dict:
    return C.read_json(C.TASK_ROOT / "configs" / "lamm_direct_c4_z128_s42.json")


def sampled_config(
    trial: optuna.Trial, args: argparse.Namespace, effective_tag: str
) -> dict:
    config = copy.deepcopy(base_config())
    layout_name = trial.suggest_categorical("latent_layout", list(LATENT_LAYOUTS))
    latent, split, default_width = LATENT_LAYOUTS[layout_name]
    model = config["model"]
    model["latent_dim"] = latent
    model["latent_split"] = split
    model["token_dim"] = trial.suggest_categorical("token_dim", [192, 256])
    model["encoder_depth"] = trial.suggest_categorical("encoder_depth", [4, 6, 8])
    model["decoder_depth"] = trial.suggest_categorical("decoder_depth", [3, 4, 6])
    model["condition_dim"] = trial.suggest_categorical("condition_dim", [32, 64])
    model["latent_width"] = trial.suggest_categorical(
        "latent_width", sorted({256, default_width, 512})
    )
    model["latent_residual_blocks"] = trial.suggest_int("latent_residual_blocks", 1, 3)
    model["dropout"] = trial.suggest_categorical("dropout", [0.05, 0.1, 0.15])
    training = config["training"]
    training["learning_rate"] = trial.suggest_float(
        "learning_rate", 1.5e-4, 6.0e-4, log=True
    )
    training["minimum_learning_rate"] = training["learning_rate"] * 0.1
    training["weight_decay"] = trial.suggest_float(
        "weight_decay", 1.0e-6, 1.0e-4, log=True
    )
    training["epochs"] = int(args.trial_epochs)
    training["samples_per_epoch"] = int(args.trial_samples_per_epoch)
    training["early_stopping_patience"] = min(
        24, max(16, int(args.trial_epochs) // 4)
    )
    training["seed"] = int(args.seed)
    config["loss"]["diagonal_velocity_weight"] = trial.suggest_categorical(
        "diagonal_velocity_weight", [0.05, 0.1, 0.2]
    )
    config["loss"]["smoothness_weight"] = trial.suggest_categorical(
        "smoothness_weight", [0.0015, 0.003, 0.006]
    )
    run_name = f"optuna_lamm_{effective_tag}_trial_{trial.number:04d}"
    training["run_name"] = run_name
    config["name"] = run_name
    validate_config(config)
    return config


def apply_best_params(config: dict, params: dict) -> dict:
    output = copy.deepcopy(config)
    latent, split, _ = LATENT_LAYOUTS[str(params["latent_layout"])]
    output["model"]["latent_dim"] = latent
    output["model"]["latent_split"] = split
    for name in (
        "token_dim",
        "encoder_depth",
        "decoder_depth",
        "condition_dim",
        "latent_width",
        "latent_residual_blocks",
        "dropout",
    ):
        output["model"][name] = params[name]
    output["training"]["learning_rate"] = params["learning_rate"]
    output["training"]["minimum_learning_rate"] = params["learning_rate"] * 0.1
    output["training"]["weight_decay"] = params["weight_decay"]
    output["loss"]["diagonal_velocity_weight"] = params["diagonal_velocity_weight"]
    output["loss"]["smoothness_weight"] = params["smoothness_weight"]
    return output


def export_study(study: optuna.Study, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "number": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            **{f"param_{name}": value for name, value in trial.params.items()},
            **{f"result_{name}": value for name, value in trial.user_attrs.items()},
        }
        for trial in study.trials
    ]
    fields = sorted({key for row in rows for key in row}) if rows else []
    with (directory / "trials.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    complete = [
        trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not complete:
        return
    best = study.best_trial
    config = apply_best_params(base_config(), best.params)
    config["name"] = "direct_mesh_lamm_c4_optuna_best"
    config["training"]["run_name"] = config["name"]
    validate_config(config)
    C.atomic_json(directory / "best_config.json", config)
    C.atomic_json(
        directory / "best_trial.json",
        {
            "number": best.number,
            "value": best.value,
            "params": best.params,
            "user_attrs": best.user_attrs,
        },
    )


def main() -> int:
    args = parse_args()
    root = C.output_root(args.output_root)
    C.configure_data_root(args.data_root)
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(args.study_tag)).strip("_.-")
    if not tag:
        raise ValueError("--study-tag must contain at least one letter or number")
    effective_tag = f"{tag}_smoke" if args.smoke else tag
    study_dir = root / "optuna" / f"lamm_direct_c4_{effective_tag}"
    study_dir.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=f"direct_mesh_lamm_c4_{effective_tag}",
        storage=f"sqlite:///{study_dir / 'study.db'}",
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=int(args.seed)),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=4, n_warmup_steps=15, interval_steps=3
        ),
    )
    if not study.trials:
        study.enqueue_trial(
            {
                "latent_layout": "z128_equal",
                "token_dim": 256,
                "encoder_depth": 8,
                "decoder_depth": 6,
                "condition_dim": 32,
                "latent_width": 256,
                "latent_residual_blocks": 2,
                "dropout": 0.05,
                "learning_rate": 3.0e-4,
                "weight_decay": 1.0e-5,
                "diagonal_velocity_weight": 0.1,
                "smoothness_weight": 0.003,
            }
        )

    def objective(trial: optuna.Trial) -> float:
        config = sampled_config(trial, args, effective_tag)
        status = train_experiment(
            config,
            args.device,
            root,
            data_root=args.data_root,
            run_name=config["training"]["run_name"],
            seed_override=int(args.seed) + trial.number,
            epochs_override=2 if args.smoke else int(args.trial_epochs),
            resume=False,
            smoke=bool(args.smoke),
            trial=trial,
        )
        trial.set_user_attr("run_name", config["training"]["run_name"])
        trial.set_user_attr("best_epoch", status["best_epoch"])
        trial.set_user_attr("latent_dim", status["latent_dim"])
        trial.set_user_attr("all_components_received_gradient", status["all_components_received_gradient"])
        return float(status["best_score"])

    study.optimize(objective, n_trials=int(args.n_trials), gc_after_trial=True)
    export_study(study, study_dir)
    print(
        json.dumps(
            {
                "study": study.study_name,
                "trials": len(study.trials),
                "best_value": study.best_value,
                "best_params": study.best_params,
                "directory": str(study_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

