#!/usr/bin/env python3
"""Build decoded-mesh payloads for the SIREN Cocycle Flow comparison notebook.

This is a visualization-only, post-test step.  It reads the three completed,
fixed checkpoints and the identical test pairs, then writes only a new cache
under ``visualization_cache``.  It never changes a checkpoint, mesh, latent,
or evaluation result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from evaluate_siren256_transport import decode_zero_level_mesh
from siren256_common import ensure_prepared, load_basis, load_cache, load_frozen_decoder, load_pairs, read_json, root_dir
from siren256_transport_models import build_transport


SCHEMA_VERSION = 1
RUNS = {
    "plain_ode": {"label": "Plain ODE", "run": "plain_ode_c3_matched"},
    "brainode": {"label": "BrainODE", "run": "brainode_attention_c3_matched"},
    "cocycle_flow": {"label": "Cocycle Flow", "run": "pca_parity_full_flow_v2_geometry_curriculum"},
}
PAIR_KEY = ("subject_id", "source_scan_id", "target_scan_id")


def identifier(value: Any) -> str:
    """Render numeric IDs consistently across pandas' CSV type inference."""
    text = str(value).strip()
    try:
        numeric = float(text)
    except ValueError:
        return text
    return str(int(numeric)) if np.isfinite(numeric) and numeric.is_integer() else text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--grid-resolution", type=int, default=None, help="Defaults to the selected Cocycle Flow evaluation resolution.")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Check completed-run contracts and selected cases without decoding or writing.")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def signature(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    stat = path.stat()
    return {"path": str(path), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns), "sha256": sha256(path)}


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def selected_cases(root: Path) -> dict[str, dict[str, Any]]:
    """Select one long, baseline-anchored CN and AD case deterministically."""
    flow_pairs = pd.read_csv(root / "runs" / RUNS["cocycle_flow"]["run"] / "evaluation" / "test" / "per_pair_metrics.csv")
    candidates = flow_pairs.loc[flow_pairs.source_visit_order == 0].copy()
    output: dict[str, dict[str, Any]] = {}
    for diagnosis in ("CN", "AD"):
        score = (
            candidates.loc[candidates.diagnosis == diagnosis]
            .groupby("subject_id", as_index=False)
            .agg(future_visits=("target_visit_order", "nunique"), maximum_followup_years=("gap_years", "max"))
            .sort_values(["future_visits", "maximum_followup_years", "subject_id"], ascending=[False, False, True])
        )
        if score.empty:
            raise RuntimeError(f"No baseline-anchored {diagnosis} test trajectory is available.")
        winner = score.iloc[0]
        pair = (
            candidates.loc[(candidates.diagnosis == diagnosis) & (candidates.subject_id == winner.subject_id)]
            .sort_values(["gap_years", "target_visit_order"], ascending=[False, False])
            .iloc[0]
        )
        output[diagnosis] = {
            "diagnosis": diagnosis,
            "subject_id": identifier(winner.subject_id),
            "source_scan_id": str(pair.source_scan_id),
            "target_scan_id": str(pair.target_scan_id),
            "future_visits": int(winner.future_visits),
            "maximum_followup_years": float(winner.maximum_followup_years),
            "selection_rule": "most baseline-anchored follow-up visits, then longest follow-up, then subject ID; mesh uses that subject's longest baseline-anchored pair",
        }
    return output


def load_completed_runs(root: Path, device: torch.device) -> tuple[dict[str, dict[str, Any]], dict[str, torch.nn.Module], dict[str, Any]]:
    configs = {model: read_json(root / "runs" / spec["run"] / "config.json") for model, spec in RUNS.items()}
    required = ("RegisteredMeshCache", "DecoderCheckpoint", "SourceExperimentDir", "LatentSize")
    for field in required:
        values = {str(config[field]) for config in configs.values()}
        if len(values) != 1:
            raise RuntimeError(f"Completed runs do not share {field}: {values}")
    basis = load_basis(root)
    models: dict[str, torch.nn.Module] = {}
    provenance: dict[str, Any] = {"configs": {}, "checkpoints": {}}
    for model, spec in RUNS.items():
        run = root / "runs" / spec["run"]
        checkpoint = run / "checkpoints" / "best.pt"
        summary = run / "evaluation" / "test" / "summary.json"
        if not checkpoint.is_file() or not summary.is_file():
            raise FileNotFoundError(f"Expected completed checkpoint and test summary for {spec['label']}: {run}")
        payload = torch.load(checkpoint, map_location="cpu")
        network = build_transport(configs[model], basis).to(device)
        network.load_state_dict(payload["flow_state_dict"], strict=True)
        network.eval()
        models[model] = network
        provenance["configs"][model] = signature(run / "config.json")
        provenance["checkpoints"][model] = signature(checkpoint)
        provenance.setdefault("test_summaries", {})[model] = signature(summary)
    return configs, models, provenance


def metadata_pair(root: Path, case: dict[str, Any]) -> Any:
    pairs = load_pairs("test", root=root)
    matches = pairs.loc[
        (pairs.subject_id.map(identifier) == case["subject_id"])
        & (pairs.source_scan_id.astype(str) == case["source_scan_id"])
        & (pairs.target_scan_id.astype(str) == case["target_scan_id"])
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one metadata pair for {case}; found {len(matches)}.")
    return matches.iloc[0]


def nearest_vertex_error(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree
    except ImportError as error:
        raise RuntimeError("SciPy is required to colour decoded surfaces by nearest target-surface error.") from error
    return np.asarray(cKDTree(target).query(predicted, k=1)[0], dtype=np.float32)


@torch.no_grad()
def decode_case(
    case: dict[str, Any], *, root: Path, cache: dict[str, np.ndarray], configs: dict[str, dict[str, Any]], models: dict[str, torch.nn.Module], decoder: torch.nn.Module, device: torch.device, resolution: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    pair = metadata_pair(root, case)
    source_index, target_index = int(pair.source_cache_index), int(pair.target_cache_index)
    source = torch.from_numpy(np.array(cache["latents"][source_index : source_index + 1], copy=True)).to(device)
    source_time = torch.tensor([[float(pair.source_time)]], dtype=torch.float32, device=device)
    target_time = torch.tensor([[float(pair.target_time)]], dtype=torch.float32, device=device)
    condition = torch.tensor([[float(pair.label_ad)]], dtype=torch.float32, device=device)
    lower, upper = (float(value) for value in configs["cocycle_flow"]["EvaluationBounds"])
    arrays: dict[str, np.ndarray] = {
        "source_vertices": np.asarray(cache["vertices"][source_index], dtype=np.float32),
        "source_faces": np.asarray(cache["faces"], dtype=np.int64),
        "target_vertices": np.asarray(cache["vertices"][target_index], dtype=np.float32),
        "target_faces": np.asarray(cache["faces"], dtype=np.int64),
    }
    mesh_summary: dict[str, Any] = {
        **case,
        "source_age_years": float(pair.source_age_years),
        "target_age_years": float(pair.target_age_years),
        "gap_years": float(pair.gap_years),
        "source_cache_index": source_index,
        "target_cache_index": target_index,
    }
    for model, network in models.items():
        latent = network.transport(source, source_time, target_time, condition)
        vertices, faces = decode_zero_level_mesh(decoder, latent, resolution, lower, upper)
        arrays[f"{model}_vertices"] = vertices
        arrays[f"{model}_faces"] = faces
        arrays[f"{model}_nearest_target_error"] = nearest_vertex_error(vertices, arrays["target_vertices"])
        mesh_summary[f"{model}_vertices"] = int(len(vertices))
        mesh_summary[f"{model}_faces"] = int(len(faces))
        mesh_summary[f"{model}_nearest_target_error_mean"] = float(arrays[f"{model}_nearest_target_error"].mean())
    return arrays, mesh_summary


def main() -> int:
    args = parse_args()
    root = root_dir()
    ensure_prepared(root)
    if args.grid_resolution is not None and args.grid_resolution < 20:
        raise ValueError("--grid-resolution must be at least 20.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {args.device}")
    cases = selected_cases(root)
    configs, models, provenance = load_completed_runs(root, device)
    resolution = int(args.grid_resolution or configs["cocycle_flow"]["EvaluationGridResolution"])
    output = args.output or root / "visualization_cache"
    output = output if output.is_absolute() else (Path.cwd() / output).resolve()
    provenance |= {
        "schema_version": SCHEMA_VERSION,
        "selected_cases": cases,
        "grid_resolution": resolution,
        "split": "test",
        "source_meshes_modified": False,
        "created_by": Path(__file__).name,
    }
    print(json.dumps({"device": str(device), "grid_resolution": resolution, "selected_cases": cases}, indent=2), flush=True)
    if args.dry_run:
        print("Dry-run passed: completed checkpoints, matching contracts, and deterministic test cases are ready.")
        return 0
    if output.exists():
        manifest = output / "manifest.json"
        provenance_path = output / "provenance.json"
        if manifest.is_file() and provenance_path.is_file() and read_json(manifest).get("status") == "complete" and read_json(provenance_path) == provenance:
            print(f"Matching visualization cache already exists: {output}")
            return 0
        raise FileExistsError(f"Refusing to overwrite a partial or non-matching visualization cache: {output}")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "provenance.json", provenance)
    write_json(output / "manifest.json", {"schema_version": SCHEMA_VERSION, "status": "building", "created_at": datetime.now(timezone.utc).isoformat(), "source_meshes_modified": False})
    cache = load_cache(configs["cocycle_flow"])
    decoder = load_frozen_decoder(configs["cocycle_flow"], device)
    arrays: dict[str, np.ndarray] = {}
    mesh_cases: dict[str, Any] = {}
    for diagnosis, case in cases.items():
        print(f"Decoding selected {diagnosis} mesh case on {device}...", flush=True)
        current, info = decode_case(case, root=root, cache=cache, configs=configs, models=models, decoder=decoder, device=device, resolution=resolution)
        arrays |= {f"{diagnosis.lower()}_{name}": value for name, value in current.items()}
        mesh_cases[diagnosis] = info
    np.savez_compressed(output / "selected_meshes.npz", **arrays)
    write_json(output / "selected_cases.json", mesh_cases)
    write_json(output / "manifest.json", {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_meshes_modified": False,
        "selected_cases": mesh_cases,
        "files": ["provenance.json", "selected_cases.json", "selected_meshes.npz"],
    })
    print(f"Prepared visualization cache: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
