#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_TASKS = {
    "old_adni": "examples/ADNI_1_L_No_MCI/brainode_comparison_task3_core_brainode_original",
    "qc_large": (
        "examples/ADNI_1_L_No_MCI_large_strict_left/task3_longitudinal_prediction/"
        "brainode_pca150_qc_stable"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Project BrainODE PCA-150 shapes into future ages under fixed CN and AD "
            "conditions and summarize PCA inverse-mesh volume."
        )
    )
    parser.add_argument(
        "--task",
        action="append",
        default=None,
        help=(
            "Task spec as name:path. Can be repeated. Defaults to old_adni and qc_large "
            "BrainODE tasks."
        ),
    )
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument(
        "--future-years",
        nargs="+",
        type=float,
        default=[0.0, 1.0, 2.0, 5.0, 10.0, 15.0],
    )
    parser.add_argument("--max-subjects-per-diagnosis", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        default=(
            "examples/ADNI_1_L_No_MCI/brainode_comparison_task3_core_brainode_original/"
            "analysis/old_vs_qc_longitudinal_comparison"
        ),
    )
    return parser.parse_args()


def repo_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def load_task_specs(args: argparse.Namespace) -> dict[str, Path]:
    if not args.task:
        return {name: repo_path(path) for name, path in DEFAULT_TASKS.items()}
    specs: dict[str, Path] = {}
    for item in args.task:
        if ":" not in item:
            raise ValueError(f"Task must be name:path, got {item!r}")
        name, path = item.split(":", 1)
        specs[name] = repo_path(path)
    return specs


def import_task_modules(task_dir: Path) -> dict[str, Any]:
    scripts_dir = task_dir / "scripts"
    if not scripts_dir.is_dir():
        raise FileNotFoundError(f"Missing BrainODE scripts dir: {scripts_dir}")
    original_path = list(sys.path)
    for module_name in ("brainode_model", "core_brainode_common", "train_core_brainode"):
        sys.modules.pop(module_name, None)
    sys.path.insert(0, str(scripts_dir))
    try:
        return {
            "brainode_model": importlib.import_module("brainode_model"),
            "common": importlib.import_module("core_brainode_common"),
            "train": importlib.import_module("train_core_brainode"),
        }
    finally:
        sys.path[:] = original_path


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def load_age_normalization(
    task_dir: Path,
    config: dict[str, Any],
    common: Any,
    train_archive: dict[str, np.ndarray],
) -> tuple[float, float, str]:
    candidates = [task_dir / "metadata" / "age_norm_stats.json"]
    task1_stats = config.get("task1", {}).get("age_norm_stats")
    if task1_stats:
        candidates.append(common.resolve_repo_path(task1_stats))
    for path in candidates:
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "age_min_train" in payload and "age_max_train" in payload:
            return (
                float(payload["age_min_train"]),
                float(payload["age_max_train"]),
                str(path),
            )
    return (
        float(np.min(train_archive["visit_continuous_age_years"])),
        float(np.max(train_archive["visit_continuous_age_years"])),
        "train_subject_sequences.npz",
    )


def resolve_device(value: str | None) -> torch.device:
    if value:
        device = torch.device(value)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def resolve_checkpoint(
    task_dir: Path,
    config: dict[str, Any],
    common: Any,
    checkpoint: str,
    run_name: str,
) -> Path:
    path = Path(checkpoint).expanduser()
    if path.is_file():
        return path
    if path.suffix == ".pth":
        resolved = common.resolve_repo_path(path)
        if resolved.is_file():
            return resolved
    name = checkpoint if checkpoint.endswith(".pth") else f"{checkpoint}.pth"
    output_root = common.resolve_repo_path(config["training"]["output_root"])
    candidate = output_root / run_name / "checkpoints" / name
    if candidate.is_file():
        return candidate
    fallback = task_dir / "training" / run_name / "checkpoints" / name
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"Missing checkpoint {checkpoint!r} for {task_dir}")


def mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    triangles = vertices[np.asarray(faces, dtype=np.int64)]
    signed = np.einsum(
        "ij,ij->i",
        triangles[:, 0, :],
        np.cross(triangles[:, 1, :], triangles[:, 2, :]),
    )
    return float(abs(signed.sum()) / 6.0)


def inverse_pca_volume(
    coefficients: np.ndarray,
    mean_flat: np.ndarray,
    components: np.ndarray,
    faces: np.ndarray,
) -> float:
    flat = coefficients.astype(np.float32) @ components.astype(np.float32) + mean_flat
    vertices = flat.reshape(-1, 3)
    return mesh_volume(vertices, faces)


def select_subject_indices(
    archive: dict[str, np.ndarray],
    max_per_diagnosis: int,
) -> list[int]:
    selected: list[int] = []
    counts = {"AD": 0, "CN": 0}
    diagnoses = [str(x) for x in archive["subject_diagnoses"].tolist()]
    for index, diagnosis in enumerate(diagnoses):
        if diagnosis not in counts:
            continue
        if counts[diagnosis] >= max_per_diagnosis:
            continue
        selected.append(index)
        counts[diagnosis] += 1
        if all(value >= max_per_diagnosis for value in counts.values()):
            break
    return selected


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["dataset"]),
            str(row["source_diagnosis"]),
            str(row["condition_label"]),
            float(row["years_from_source"]),
        )
        grouped.setdefault(key, []).append(row)

    summaries: list[dict[str, Any]] = []
    for (dataset, source_diagnosis, condition_label, years), group in sorted(grouped.items()):
        volumes = [float(row["predicted_volume"]) for row in group]
        deltas = [float(row["delta_pct_from_source"]) for row in group]
        summaries.append(
            {
                "dataset": dataset,
                "source_diagnosis": source_diagnosis,
                "condition_label": condition_label,
                "years_from_source": years,
                "rows": len(group),
                "mean_predicted_volume": sum(volumes) / len(volumes),
                "mean_delta_pct_from_source": sum(deltas) / len(deltas),
                "ood_train_age_fraction": sum(
                    1.0 for row in group if row["is_train_age_ood"]
                )
                / len(group),
                "future_beyond_observed_fraction": sum(
                    1.0 for row in group if row["is_future_beyond_observed"]
                )
                / len(group),
            }
        )
    return summaries


@torch.no_grad()
def run_task(
    dataset_name: str,
    task_dir: Path,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    modules = import_task_modules(task_dir)
    common = modules["common"]
    train = modules["train"]
    brainode_model = modules["brainode_model"]

    config = common.load_config(task_dir / "configs" / "core_brainode.json")
    training_config = dict(config["training"])
    model_config = dict(config["model"])
    full_config = train.full_brainode_config(config)
    run_name = str(args.run_name or training_config["run_name"])
    device = resolve_device(args.device or training_config.get("device"))
    checkpoint_path = resolve_checkpoint(task_dir, config, common, args.checkpoint, run_name)

    split_archive = load_npz(task_dir / "dataset" / f"{args.split}_subject_sequences.npz")
    train_archive = load_npz(task_dir / "dataset" / "train_subject_sequences.npz")
    age_min, age_max, age_norm_source = load_age_normalization(
        task_dir=task_dir,
        config=config,
        common=common,
        train_archive=train_archive,
    )
    age_range = age_max - age_min
    if age_range <= 0:
        raise ValueError(f"Invalid train age range for {dataset_name}: {age_min}, {age_max}")

    latent_dim = int(split_archive["visit_pca_150"].shape[1])
    model = train.build_model(
        latent_dim=latent_dim,
        model_config=model_config,
        full_config=full_config,
    ).to(device)
    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    pca_model_dir = common.resolve_repo_path(config["task2"]["pca_model_dir"])
    mean_flat = np.load(pca_model_dir / "mean.npy").astype(np.float32)
    components = np.load(pca_model_dir / "components_256.npy").astype(np.float32)[:latent_dim]
    faces = np.load(pca_model_dir / "faces.npy").astype(np.int64)

    subject_indices = select_subject_indices(split_archive, args.max_subjects_per_diagnosis)
    offsets = split_archive["subject_visit_offsets"]
    rows: list[dict[str, Any]] = []
    for subject_index in subject_indices:
        start = int(offsets[subject_index])
        end = int(offsets[subject_index + 1])
        subject_id = str(split_archive["subject_ids"][subject_index])
        diagnosis = str(split_archive["subject_diagnoses"][subject_index])
        base_code_np = split_archive["visit_pca_150"][start].astype(np.float32)
        base_age = float(split_archive["visit_continuous_age_years"][start])
        base_time = float(split_archive["visit_continuous_age_norm"][start])
        final_observed_age = float(np.max(split_archive["visit_continuous_age_years"][start:end]))
        source_volume = inverse_pca_volume(base_code_np, mean_flat, components, faces)
        final_observed_code = split_archive["visit_pca_150"][end - 1].astype(np.float32)
        final_observed_pca_volume = inverse_pca_volume(
            final_observed_code, mean_flat, components, faces
        )

        future_years = sorted(set(float(value) for value in args.future_years))
        target_ages = [base_age + years for years in future_years]
        target_times = [(age - age_min) / age_range for age in target_ages]
        times_tensor = torch.tensor([target_times], dtype=torch.float32, device=device)
        initial_state = torch.from_numpy(base_code_np).float().view(1, -1).to(device)

        for condition_label, condition_value in (("CN", 0.0), ("AD", 1.0)):
            condition = torch.tensor([condition_value], dtype=torch.float32, device=device)
            predictions = brainode_model.integrate_sequence_rk4(
                func=model,
                initial_state=initial_state,
                times=times_tensor,
                condition=condition,
                substeps=max(1, int(args.substeps)),
            )[0].detach().cpu().numpy()
            for years, age, time_value, coefficients in zip(
                future_years, target_ages, target_times, predictions
            ):
                predicted_volume = inverse_pca_volume(coefficients, mean_flat, components, faces)
                rows.append(
                    {
                        "dataset": dataset_name,
                        "split": args.split,
                        "subject_id": subject_id,
                        "source_diagnosis": diagnosis,
                        "condition_label": condition_label,
                        "condition_value": condition_value,
                        "source_age_years": base_age,
                        "target_age_years": age,
                        "years_from_source": years,
                        "target_age_norm": time_value,
                        "train_age_min": age_min,
                        "train_age_max": age_max,
                        "final_observed_age_years": final_observed_age,
                        "is_train_age_ood": bool(time_value < 0.0 or time_value > 1.0),
                        "is_future_beyond_observed": bool(age > final_observed_age + 1e-6),
                        "source_pca_volume": source_volume,
                        "last_observed_pca_volume": final_observed_pca_volume,
                        "predicted_volume": predicted_volume,
                        "delta_volume_from_source": predicted_volume - source_volume,
                        "delta_pct_from_source": (
                            100.0 * (predicted_volume - source_volume) / source_volume
                            if source_volume
                            else float("nan")
                        ),
                        "checkpoint": str(checkpoint_path),
                    }
                )

    summary = {
        "dataset": dataset_name,
        "task_dir": str(task_dir),
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "selected_subjects": len(subject_indices),
        "rows": len(rows),
        "age_min_train": age_min,
        "age_max_train": age_max,
        "age_norm_source": age_norm_source,
        "future_years": sorted(set(float(value) for value in args.future_years)),
        "substeps": max(1, int(args.substeps)),
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    output_dir = repo_path(args.output_dir)
    all_rows: list[dict[str, Any]] = []
    task_summaries: list[dict[str, Any]] = []
    for dataset_name, task_dir in load_task_specs(args).items():
        rows, summary = run_task(dataset_name, task_dir, args)
        all_rows.extend(rows)
        task_summaries.append(summary)

    prediction_fields = [
        "dataset",
        "split",
        "subject_id",
        "source_diagnosis",
        "condition_label",
        "condition_value",
        "source_age_years",
        "target_age_years",
        "years_from_source",
        "target_age_norm",
        "train_age_min",
        "train_age_max",
        "final_observed_age_years",
        "is_train_age_ood",
        "is_future_beyond_observed",
        "source_pca_volume",
        "last_observed_pca_volume",
        "predicted_volume",
        "delta_volume_from_source",
        "delta_pct_from_source",
        "checkpoint",
    ]
    write_csv(output_dir / "brainode_conditional_future_volume.csv", all_rows, prediction_fields)

    summary_rows = summarize_rows(all_rows)
    summary_fields = [
        "dataset",
        "source_diagnosis",
        "condition_label",
        "years_from_source",
        "rows",
        "mean_predicted_volume",
        "mean_delta_pct_from_source",
        "ood_train_age_fraction",
        "future_beyond_observed_fraction",
    ]
    write_csv(output_dir / "brainode_conditional_future_volume_summary.csv", summary_rows, summary_fields)

    payload = {
        "output_dir": str(output_dir),
        "prediction_rows": len(all_rows),
        "summary_rows": len(summary_rows),
        "tasks": task_summaries,
        "note": "Volumes are computed from PCA inverse-transform meshes, not SDF decoder meshes.",
    }
    (output_dir / "brainode_conditional_future_volume_run.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
