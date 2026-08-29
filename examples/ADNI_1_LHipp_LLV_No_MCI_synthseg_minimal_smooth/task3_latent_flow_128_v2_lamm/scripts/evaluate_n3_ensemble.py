#!/usr/bin/env python3
"""Evaluate three N3 LAMM latent flows as a shape-space ensemble.

Each independently trained autoencoder owns a different latent coordinate system.  This
script therefore never averages latent vectors.  It transports each member in its own
standardized 128-D space, decodes the result, and averages corresponding mesh vertices.

For the direct-C4 model, the instantaneous velocity is the zero-horizon limit

    lim[h->0] (Phi(z,t,t+h,d)-z)/h = phi(z,t,t,d).

The learned time coordinate is train-normalized age.  Velocities are converted to per-year
units and pushed through each frozen decoder with an exact Jacobian-vector product.  Only
the resulting shape-space velocity (mm/year) is comparable across latent bases.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from _bootstrap import activate

activate()
import c4_objective as O
import common as C
from models import DirectC4Flow


PAIR_METRICS = (
    "latent",
    "coordinate",
    "euclidean",
    "end_to_end_coordinate_rmse",
    "end_to_end_coordinate_mae",
    "end_to_end_euclidean",
    "volume_relative",
    "rate",
    "predicted_signed_rate",
    "observed_signed_rate",
    "nochange_latent",
    "nochange_coordinate",
    "nochange_euclidean",
    "nochange_end_to_end_coordinate_rmse",
    "nochange_end_to_end_coordinate_mae",
    "nochange_end_to_end_euclidean",
    "nochange_volume_relative",
    "nochange_rate",
)
COMPARISON_METRICS = (
    "coordinate",
    "euclidean",
    "end_to_end_coordinate_rmse",
    "end_to_end_coordinate_mae",
    "end_to_end_euclidean",
    "volume_relative",
    "rate",
)


@dataclass
class Member:
    name: str
    run_dir: Path
    representation: str
    config: dict[str, Any]
    archive: dict[str, np.ndarray]
    train_archive: dict[str, np.ndarray]
    values: dict[str, torch.Tensor]
    geometry: C.FrozenGeometry
    flow: DirectC4Flow


def parse_assignment(text: str, flag: str) -> tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"{flag} must be LABEL=/absolute/path")
    label, raw_path = text.split("=", 1)
    C.validate_run_name(label)
    path = Path(raw_path).expanduser().resolve()
    return label, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--member",
        action="append",
        required=True,
        help="LABEL=RUN_DIR; pass the three N3 runs in seed order",
    )
    parser.add_argument(
        "--baseline-report",
        action="append",
        default=[],
        help="LABEL=.../evaluation/SPLIT/summary.json for paired comparison",
    )
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_member(label: str, run_dir: Path, split: str, device: torch.device) -> Member:
    resolved = C.read_json(run_dir / "resolved_config.json")
    config = resolved["config"]
    representation = str(config["representation"])
    if config.get("method") != "direct_c4":
        raise ValueError(f"{run_dir} is not a direct-C4 run")
    checkpoint_path = run_dir / "checkpoints" / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if bool(checkpoint.get("test_data_loaded", False)):
        raise ValueError(f"Training/test leakage recorded in {checkpoint_path}")
    model = config["model"]
    flow = DirectC4Flow(
        C.LATENT_DIM,
        int(model["width"]),
        int(model["residual_blocks"]),
        float(model.get("dropout", 0.0)),
    ).to(device)
    flow.load_state_dict(checkpoint["flow_state_dict"], strict=True)
    flow.eval()
    train_archive = C.load_archive(representation, "train")
    archive = C.load_archive(representation, split)
    geometry = C.build_geometry(representation, train_archive, device)
    values = C.values_on_device(archive, device)
    O.attach_reference_geometry(values, geometry, int(config["training"].get("decoder_batch_size", 64)))
    return Member(
        name=label,
        run_dir=run_dir,
        representation=representation,
        config=config,
        archive=archive,
        train_archive=train_archive,
        values=values,
        geometry=geometry,
        flow=flow,
    )


def assert_aligned(members: list[Member], split: str) -> list[C.PairRow]:
    if len(members) < 2:
        raise ValueError("An ensemble needs at least two members")
    reference = members[0].archive
    exact_keys = (
        "subject_ids",
        "subject_diagnoses",
        "subject_visit_offsets",
        "visit_scan_ids",
        "visit_subject_ids",
        "visit_diagnoses",
        "visit_label_ad",
        "visit_orders",
    )
    numeric_keys = (
        "visit_age_years",
        "visit_age_norm_train",
        "visit_time_years_from_baseline",
    )
    for member in members[1:]:
        for key in exact_keys:
            if not np.array_equal(reference[key], member.archive[key]):
                raise ValueError(f"Member archives are not aligned at {key}: {member.name}")
        for key in numeric_keys:
            if not np.allclose(reference[key], member.archive[key], rtol=0.0, atol=1.0e-7):
                raise ValueError(f"Member archives drift at {key}: {member.name}")
    pair_sets = [C.load_pairs(split, member.archive) for member in members]
    key = lambda row: (row.subject, row.source, row.target, row.pair_type)
    reference_keys = [key(row) for row in pair_sets[0]]
    for member, rows in zip(members[1:], pair_sets[1:]):
        if [key(row) for row in rows] != reference_keys:
            raise ValueError(f"Pair ordering is not aligned for {member.name}")
    return pair_sets[0]


def tensor_metrics(
    members: list[Member],
    rows: list[C.PairRow],
    raw_vertices: np.ndarray,
    batch_size: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    volume_geometry = members[0].geometry
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            raw = C.collate_pairs(chunk)
            predictions = []
            sources = []
            targets = []
            latent_errors = []
            latent_nochange = []
            member_batches = []
            for member in members:
                batch = O.indexed(member.values, raw)
                member_batches.append(batch)
                prediction = member.flow.transport(
                    batch["source"], batch["source_age"], batch["target_age"], batch["label"]
                )
                predictions.append(member.geometry.vertices(prediction))
                sources.append(batch["source_vertices"])
                targets.append(batch["target_vertices"])
                latent_errors.append(torch.mean((prediction - batch["target"]).square(), dim=1))
                latent_nochange.append(torch.mean((batch["source"] - batch["target"]).square(), dim=1))
            predicted_vertices = torch.stack(predictions).mean(0)
            source_vertices = torch.stack(sources).mean(0)
            target_vertices = torch.stack(targets).mean(0)
            predicted_volume = volume_geometry.volume_from_vertices(predicted_vertices)
            source_volume = volume_geometry.volume_from_vertices(source_vertices)
            target_volume = volume_geometry.volume_from_vertices(target_vertices)
            target_index = raw["target"].cpu().numpy()
            raw_target = torch.from_numpy(
                np.asarray(raw_vertices[target_index], dtype=np.float32).copy()
            ).to(predicted_vertices.device)
            first = member_batches[0]
            years = torch.abs(first["target_years"] - first["source_years"]).clamp_min(1.0e-6)
            transport_delta = predicted_vertices - target_vertices
            end_delta = predicted_vertices - raw_target
            nochange_transport = source_vertices - target_vertices
            nochange_end = source_vertices - raw_target
            tensors = {
                "latent": torch.stack(latent_errors).mean(0),
                "coordinate": torch.mean(torch.abs(transport_delta), dim=(1, 2)),
                "euclidean": torch.linalg.vector_norm(transport_delta, dim=2).mean(dim=1),
                "end_to_end_coordinate_rmse": torch.sqrt(torch.mean(end_delta.square(), dim=(1, 2))),
                "end_to_end_coordinate_mae": torch.mean(torch.abs(end_delta), dim=(1, 2)),
                "end_to_end_euclidean": torch.linalg.vector_norm(end_delta, dim=2).mean(dim=1),
                "volume_relative": torch.abs(predicted_volume - target_volume) / target_volume,
                "rate": torch.abs((torch.log(predicted_volume) - torch.log(target_volume)) / years),
                "predicted_signed_rate": (torch.log(predicted_volume) - torch.log(source_volume)) / years,
                "observed_signed_rate": (torch.log(target_volume) - torch.log(source_volume)) / years,
                "nochange_latent": torch.stack(latent_nochange).mean(0),
                "nochange_coordinate": torch.mean(torch.abs(nochange_transport), dim=(1, 2)),
                "nochange_euclidean": torch.linalg.vector_norm(nochange_transport, dim=2).mean(dim=1),
                "nochange_end_to_end_coordinate_rmse": torch.sqrt(torch.mean(nochange_end.square(), dim=(1, 2))),
                "nochange_end_to_end_coordinate_mae": torch.mean(torch.abs(nochange_end), dim=(1, 2)),
                "nochange_end_to_end_euclidean": torch.linalg.vector_norm(nochange_end, dim=2).mean(dim=1),
                "nochange_volume_relative": torch.abs(source_volume - target_volume) / target_volume,
                "nochange_rate": torch.abs((torch.log(source_volume) - torch.log(target_volume)) / years),
            }
            for index, row in enumerate(chunk):
                output.append({
                    "subject": row.subject,
                    "diagnosis": row.diagnosis,
                    "pair_type": row.pair_type,
                    "source_index": int(row.source),
                    "target_index": int(row.target),
                    "delta_years": float(row.delta_years),
                    **{name: float(value[index].cpu()) for name, value in tensors.items()},
                })
    return output


def grouped_pair_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for diagnosis in ("CN", "AD", "overall"):
        selected = rows if diagnosis == "overall" else [row for row in rows if row["diagnosis"] == diagnosis]
        result[diagnosis] = {
            "rows": len(selected),
            **{
                f"{metric}_mean": float(np.mean([row[metric] for row in selected]))
                for metric in PAIR_METRICS
            },
        }
    return result


def representation_floor(
    members: list[Member], raw_vertices: np.ndarray, batch_size: int
) -> tuple[dict[str, Any], np.ndarray]:
    rmse: list[float] = []
    mae: list[float] = []
    euclidean: list[float] = []
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        count = len(members[0].values["z"])
        for start in range(0, count, batch_size):
            decoded = torch.stack([
                member.values["reference_vertices"][start : start + batch_size]
                for member in members
            ]).mean(0)
            raw = torch.from_numpy(
                np.asarray(raw_vertices[start : start + batch_size], dtype=np.float32).copy()
            ).to(decoded.device)
            delta = decoded - raw
            rmse.extend(torch.sqrt(torch.mean(delta.square(), dim=(1, 2))).cpu().tolist())
            mae.extend(torch.mean(torch.abs(delta), dim=(1, 2)).cpu().tolist())
            euclidean.extend(torch.linalg.vector_norm(delta, dim=2).mean(dim=1).cpu().tolist())
            predictions.append(decoded.cpu().numpy())
    return {
        "scans": len(rmse),
        "coordinate_rmse_mm_mean": float(np.mean(rmse)),
        "coordinate_mae_mm_mean": float(np.mean(mae)),
        "vertex_euclidean_mm_mean": float(np.mean(euclidean)),
    }, np.concatenate(predictions)


def age_norm_per_year(train_archive: dict[str, np.ndarray]) -> tuple[float, float]:
    age = train_archive["visit_age_years"].astype(np.float64)
    normalized = train_archive["visit_age_norm_train"].astype(np.float64)
    centered = age - age.mean()
    slope = float(np.sum(centered * (normalized - normalized.mean())) / np.sum(centered ** 2))
    fitted = normalized.mean() + slope * centered
    return slope, float(np.max(np.abs(fitted - normalized)))


def instantaneous_velocity(
    members: list[Member], batch_size: int
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    archive = members[0].archive
    offsets = archive["subject_visit_offsets"].astype(np.int64)
    baseline = np.asarray(offsets[:-1], dtype=np.int64)
    slope, residual = age_norm_per_year(members[0].train_archive)
    member_vertices: list[torch.Tensor] = []
    member_velocity: list[torch.Tensor] = []
    member_latent_speed: list[np.ndarray] = []
    for member in members:
        vertices_chunks: list[torch.Tensor] = []
        velocity_chunks: list[torch.Tensor] = []
        latent_speed_chunks: list[np.ndarray] = []
        for start in range(0, len(baseline), batch_size):
            index = torch.from_numpy(baseline[start : start + batch_size]).to(member.values["z"].device)
            z = member.values["z"][index]
            age = member.values["age"][index]
            label = member.values["label"][index]
            with torch.no_grad():
                velocity_age_norm = member.flow.average_velocity(z, age, age, label)
                velocity_year = velocity_age_norm * slope
                latent_speed = torch.sqrt(torch.mean(velocity_year.square(), dim=1))
            z_input = z.detach().requires_grad_(True)
            _, vertex_velocity = torch.autograd.functional.jvp(
                member.geometry.vertices,
                z_input,
                velocity_year.detach(),
                create_graph=False,
                strict=False,
            )
            with torch.no_grad():
                vertices_chunks.append(member.geometry.vertices(z).detach().cpu())
                velocity_chunks.append(vertex_velocity.detach().cpu())
                latent_speed_chunks.append(latent_speed.cpu().numpy())
        member_vertices.append(torch.cat(vertices_chunks))
        member_velocity.append(torch.cat(velocity_chunks))
        member_latent_speed.append(np.concatenate(latent_speed_chunks))
    vertices = torch.stack(member_vertices).mean(0)
    velocity = torch.stack(member_velocity).mean(0)
    volume_geometry = members[0].geometry.cpu()
    _, volume_velocity = torch.autograd.functional.jvp(
        volume_geometry.volume_from_vertices,
        vertices.detach().requires_grad_(True),
        velocity,
        create_graph=False,
        strict=False,
    )
    with torch.no_grad():
        volume = volume_geometry.volume_from_vertices(vertices)
        coordinate_rms = torch.sqrt(torch.mean(velocity.square(), dim=(1, 2))).numpy()
        vertex_speed = torch.linalg.vector_norm(velocity, dim=2)
        vertex_speed_mean = vertex_speed.mean(dim=1).numpy()
        vertex_speed_p95 = torch.quantile(vertex_speed, 0.95, dim=1).numpy()
        volume_rate_pct = (100.0 * volume_velocity / volume).numpy()
    diagnoses = archive["subject_diagnoses"].astype(str)
    subjects = archive["subject_ids"].astype(str)
    rows = []
    for index, subject in enumerate(subjects):
        rows.append({
            "subject": str(subject),
            "diagnosis": str(diagnoses[index]),
            "coordinate_rms_speed_mm_per_year": float(coordinate_rms[index]),
            "mean_vertex_speed_mm_per_year": float(vertex_speed_mean[index]),
            "p95_vertex_speed_mm_per_year": float(vertex_speed_p95[index]),
            "signed_volume_rate_pct_per_year": float(volume_rate_pct[index]),
            **{
                f"{member.name}_standardized_latent_rms_speed_per_year": float(member_latent_speed[m][index])
                for m, member in enumerate(members)
            },
        })
    groups: dict[str, Any] = {}
    for diagnosis in ("CN", "AD", "overall"):
        chosen = np.ones(len(rows), dtype=bool) if diagnosis == "overall" else diagnoses == diagnosis
        groups[diagnosis] = {
            "subjects": int(chosen.sum()),
            "coordinate_rms_speed_mm_per_year_mean": float(coordinate_rms[chosen].mean()),
            "coordinate_rms_speed_mm_per_year_median": float(np.median(coordinate_rms[chosen])),
            "mean_vertex_speed_mm_per_year_mean": float(vertex_speed_mean[chosen].mean()),
            "mean_vertex_speed_mm_per_year_median": float(np.median(vertex_speed_mean[chosen])),
            "p95_vertex_speed_mm_per_year_mean": float(vertex_speed_p95[chosen].mean()),
            "signed_volume_rate_pct_per_year_mean": float(volume_rate_pct[chosen].mean()),
            "signed_volume_rate_pct_per_year_median": float(np.median(volume_rate_pct[chosen])),
        }
    maps = {
        "faces": members[0].geometry.faces.detach().cpu().numpy(),
        "subject_ids": subjects,
        "diagnoses": diagnoses,
        "coordinate_rms_speed_mm_per_year": coordinate_rms,
        "mean_vertex_speed_mm_per_year": vertex_speed_mean,
        "signed_volume_rate_pct_per_year": volume_rate_pct,
    }
    for diagnosis in ("CN", "AD"):
        chosen = diagnoses == diagnosis
        maps[f"{diagnosis.lower()}_mean_vertices_mm"] = vertices[chosen].mean(0).numpy()
        maps[f"{diagnosis.lower()}_mean_velocity_mm_per_year"] = velocity[chosen].mean(0).numpy()
        maps[f"{diagnosis.lower()}_mean_vertex_speed_mm_per_year"] = vertex_speed[chosen].mean(0).numpy()
    summary = {
        "definition": "zero-horizon direct-C4 velocity phi(z,t,t,d), decoder JVP, ensemble mean in shared mesh coordinates",
        "time_conversion": {
            "age_norm_per_year": slope,
            "max_linear_fit_residual": residual,
        },
        "sampling": "one baseline visit per held-out subject",
        "latent_warning": "standardized latent speeds are member diagnostics only; latent bases are not aligned",
        "groups": groups,
    }
    return summary, rows, maps


def row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["subject"]),
        int(row["source_index"]),
        int(row["target_index"]),
        str(row["pair_type"]),
    )


def paired_comparisons(
    ensemble_rows: list[dict[str, Any]],
    baselines: list[tuple[str, Path]],
    split: str,
    samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    candidate = {row_key(row): row for row in ensemble_rows}
    by_subject: dict[str, list[tuple[Any, ...]]] = defaultdict(list)
    for key in sorted(candidate):
        by_subject[str(key[0])].append(key)
    subjects = sorted(by_subject)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(subjects), size=(samples, len(subjects)))
    output = []
    for label, path in baselines:
        report = C.read_json(path)
        if report.get("split") != split:
            raise ValueError(f"Baseline split mismatch: {path}")
        rows = report["pair_metrics"]["all_forward"].get("row_metrics", [])
        baseline = {row_key(row): row for row in rows}
        if set(baseline) != set(candidate):
            raise ValueError(
                f"Pair mismatch for {label}: baseline_only={len(set(baseline)-set(candidate))}, "
                f"ensemble_only={len(set(candidate)-set(baseline))}"
            )
        for metric in COMPARISON_METRICS:
            subject_delta = np.asarray([
                np.mean([
                    float(candidate[key][metric]) - float(baseline[key][metric])
                    for key in by_subject[subject]
                ])
                for subject in subjects
            ])
            draws = subject_delta[indices].mean(axis=1)
            output.append({
                "baseline": label,
                "candidate": "LAMM-N3-shape-ensemble",
                "metric": metric,
                "baseline_pair_mean": float(np.mean([baseline[key][metric] for key in baseline])),
                "candidate_pair_mean": float(np.mean([candidate[key][metric] for key in candidate])),
                "candidate_minus_baseline_subject_mean": float(subject_delta.mean()),
                "ci95_low": float(np.quantile(draws, 0.025)),
                "ci95_high": float(np.quantile(draws, 0.975)),
                "fraction_subjects_candidate_better": float(np.mean(subject_delta < 0.0)),
                "subjects": len(subjects),
                "pairs": len(candidate),
                "bootstrap_samples": samples,
            })
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def equal_axes(ax, vertices: np.ndarray) -> None:
    center = 0.5 * (vertices.min(0) + vertices.max(0))
    radius = 0.52 * float((vertices.max(0) - vertices.min(0)).max())
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_axis_off()


def velocity_plot(path: Path, rows: list[dict[str, Any]], maps: dict[str, np.ndarray]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors

    faces = maps["faces"]
    cn_speed = maps["cn_mean_vertex_speed_mm_per_year"]
    ad_speed = maps["ad_mean_vertex_speed_mm_per_year"]
    vmax = float(max(np.quantile(cn_speed, 0.99), np.quantile(ad_speed, 0.99), 1.0e-6))
    norm = colors.Normalize(0.0, vmax)
    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    for panel, diagnosis in enumerate(("cn", "ad"), start=1):
        vertices = maps[f"{diagnosis}_mean_vertices_mm"]
        velocity = maps[f"{diagnosis}_mean_velocity_mm_per_year"]
        speed = maps[f"{diagnosis}_mean_vertex_speed_mm_per_year"]
        ax = fig.add_subplot(2, 2, panel, projection="3d")
        surface = ax.plot_trisurf(
            vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=faces,
            linewidth=0.0, antialiased=True, shade=False,
        )
        surface.set_facecolors(cm.viridis(norm(speed[faces].mean(axis=1))))
        sample = np.linspace(0, len(vertices) - 1, 90, dtype=int)
        ax.quiver(
            vertices[sample, 0], vertices[sample, 1], vertices[sample, 2],
            velocity[sample, 0], velocity[sample, 1], velocity[sample, 2],
            color="black", length=10.0, normalize=False, linewidth=0.45,
        )
        equal_axes(ax, vertices)
        ax.view_init(elev=22, azim=-68)
        ax.set_title(f"{diagnosis.upper()} instantaneous velocity (arrows x10)")
    scalar = cm.ScalarMappable(norm=norm, cmap="viridis")
    scalar.set_array([])
    fig.colorbar(scalar, ax=fig.axes[:2], shrink=0.65, label="Mean vertex speed (mm/year)")

    ax = fig.add_subplot(2, 2, 3)
    diagnoses = [row["diagnosis"] for row in rows]
    cn = [row["mean_vertex_speed_mm_per_year"] for row in rows if row["diagnosis"] == "CN"]
    ad = [row["mean_vertex_speed_mm_per_year"] for row in rows if row["diagnosis"] == "AD"]
    ax.boxplot([cn, ad], labels=["CN", "AD"], showmeans=True)
    ax.set_ylabel("Subject mean vertex speed (mm/year)")
    ax.set_title("Baseline-subject speed distribution")
    ax.grid(axis="y", alpha=0.25)

    ax = fig.add_subplot(2, 2, 4)
    cn_rate = [row["signed_volume_rate_pct_per_year"] for row in rows if row["diagnosis"] == "CN"]
    ad_rate = [row["signed_volume_rate_pct_per_year"] for row in rows if row["diagnosis"] == "AD"]
    means = [np.mean(cn_rate), np.mean(ad_rate)]
    sem = [np.std(cn_rate, ddof=1) / np.sqrt(len(cn_rate)), np.std(ad_rate, ddof=1) / np.sqrt(len(ad_rate))]
    ax.bar(["CN", "AD"], means, yerr=sem, color=["#4C78A8", "#E45756"], capsize=5)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("Instantaneous signed volume rate (%/year)")
    ax.set_title("Mean +/- subject-level SEM")
    ax.grid(axis="y", alpha=0.25)
    fig.suptitle("LAMM N3 three-flow shape-space instantaneous velocity", fontsize=15)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.split == "test" and not args.allow_test:
        raise ValueError("Test evaluation requires explicit --allow-test")
    if args.bootstrap_samples < 100:
        raise ValueError("Use at least 100 bootstrap samples")
    member_specs = [parse_assignment(value, "--member") for value in args.member]
    baseline_specs = [parse_assignment(value, "--baseline-report") for value in args.baseline_report]
    if len({label for label, _ in member_specs}) != len(member_specs):
        raise ValueError("Member labels must be unique")
    for label, path in member_specs:
        if not (path / "checkpoints" / "best.pt").is_file():
            raise FileNotFoundError(f"Missing checkpoint for {label}: {path}")
    device = C.choose_device(args.device)
    members = [load_member(label, path, args.split, device) for label, path in member_specs]
    pairs = assert_aligned(members, args.split)
    raw_vertices = C.cached_vertices(args.split)
    pair_rows = tensor_metrics(members, pairs, raw_vertices, args.batch_size)
    floor, floor_vertices = representation_floor(members, raw_vertices, args.batch_size)
    velocity_summary, velocity_rows, velocity_maps = instantaneous_velocity(members, args.batch_size)
    comparisons = paired_comparisons(
        pair_rows, baseline_specs, args.split, args.bootstrap_samples, args.seed
    )
    summary = {
        "schema_version": 1,
        "split": args.split,
        "method": "direct_c4_shape_ensemble",
        "members": [
            {"label": member.name, "representation": member.representation, "run_dir": str(member.run_dir)}
            for member in members
        ],
        "ensemble_contract": {
            "latent_vectors_averaged": False,
            "transport": "one direct-C4 per member latent basis",
            "aggregation": "equal mean of decoded corresponding vertices",
            "effective_latent_values": len(members) * C.LATENT_DIM,
            "fair_128d_single_representation": False,
        },
        "representation_floor": floor,
        "pair_metrics": grouped_pair_summary(pair_rows),
        "instantaneous_velocity": velocity_summary,
        "paired_comparisons": comparisons,
        "bootstrap_unit": "subject",
        "source_meshes_modified": False,
    }
    C.assert_finite_mapping(summary)
    compact = {
        "split": args.split,
        "floor_rmse_mm": floor["coordinate_rmse_mm_mean"],
        "forecast_rmse_mm": summary["pair_metrics"]["overall"]["end_to_end_coordinate_rmse_mean"],
        "nochange_rmse_mm": summary["pair_metrics"]["overall"]["nochange_end_to_end_coordinate_rmse_mean"],
        "instantaneous_velocity": velocity_summary["groups"],
        "primary_comparisons": [
            row for row in comparisons if row["metric"] == "end_to_end_coordinate_rmse"
        ],
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    if args.dry_run:
        print("DRY RUN PASSED — no files written")
        return 0
    output = C.require_bulk_path(args.output_dir, "N3 ensemble output")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    C.atomic_json(output / "summary.json", summary)
    write_csv(output / "pair_metrics.csv", pair_rows)
    write_csv(output / "paired_comparisons.csv", comparisons)
    write_csv(output / "instantaneous_velocity_subjects.csv", velocity_rows)
    np.savez_compressed(
        output / "instantaneous_velocity_maps.npz",
        **velocity_maps,
    )
    np.save(output / "ensemble_reconstruction_vertices_mm.npy", floor_vertices)
    velocity_plot(output / "instantaneous_velocity.png", velocity_rows, velocity_maps)
    print(f"WROTE {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
