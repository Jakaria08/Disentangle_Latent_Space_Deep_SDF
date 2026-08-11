#!/usr/bin/env python3
"""Decode all existing large-ADNI no-skip SIREN latents on multiple GPUs.

This is Stage 2 of the representation-volume audit. It only reads existing
latent archives and the frozen decoder checkpoint; it never fits or modifies a
latent. Outputs are resumable at scan granularity.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import multiprocessing as mp
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any, Iterable

import numpy as np


def find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (
            (parent / "README.md").is_file()
            and (parent / "examples").is_dir()
            and (parent / "networks").is_dir()
        ):
            return parent
    raise RuntimeError("Could not locate the Deep3DComp repository root")


REPO_ROOT = find_repo_root()
TASK2 = (
    REPO_ROOT
    / "examples/ADNI_1_L_No_MCI_large_strict_left/task2_representations"
)
TASK2_SCRIPTS = TASK2 / "scripts"
SIREN_DIR = TASK2 / "inr/siren_naisr_5x512_warmstart_no_skip"
CONFIG_PATH = SIREN_DIR / "run_config.json"
MANIFEST_PATH = TASK2 / "metadata/adni_large_strict_no_mci_left_manifest.csv"
LATENT_DIR = SIREN_DIR / "latents"
DEFAULT_MESH_DIR = SIREN_DIR / "reconstructed_meshes"
DEFAULT_REPORT_DIR = SIREN_DIR / "reconstruction_reports"
DEFAULT_PROGRESS_DIR = SIREN_DIR / "reconstruction_progress"
ANALYSIS_SCRIPT = Path(__file__).resolve().parent / "analyze_available_mesh_volume_trends.py"
AUDIT_ROOT = Path(__file__).resolve().parent / "analysis/available_representation_volume_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decode existing large no-skip SIREN scan latents with one worker "
            "per GPU, then optionally run the CPU volume-trend audit."
        )
    )
    parser.add_argument(
        "--gpus",
        default="0,1,2",
        help="Comma-separated CUDA device indices (default: 0,1,2).",
    )
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--max-batch", type=int, default=262144)
    parser.add_argument("--output-dir", default=str(DEFAULT_MESH_DIR))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--progress-dir", default=str(DEFAULT_PROGRESS_DIR))
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Decode only the first N sorted scans; zero means all scans.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Decode again even when a matching successful report and mesh exist.",
    )
    parser.add_argument(
        "--run-final-analysis",
        action="store_true",
        help="Run Stage 1 against the decoded mesh directory after all workers finish.",
    )
    parser.add_argument(
        "--analysis-output-dir",
        default=str(AUDIT_ROOT / "full_large_siren"),
    )
    parser.add_argument(
        "--analysis-mesh-cache",
        default=str(AUDIT_ROOT / "mesh_volume_cache.csv"),
    )
    parser.add_argument(
        "--cpu-smoke-workers",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_csv_atomic(rows: Iterable[dict[str, Any]], path: Path) -> None:
    records = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = sorted({key for row in records for key in row})
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def read_manifest() -> list[dict[str, str]]:
    with MANIFEST_PATH.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    split_order = {"train": 0, "val": 1, "test": 2}

    def sort_key(row: dict[str, str]) -> tuple[Any, ...]:
        try:
            age = float(row.get("age_years", "nan"))
        except ValueError:
            age = float("inf")
        if not np.isfinite(age):
            age = float("inf")
        return (
            split_order.get(row["split"], 99),
            row["subject_id"],
            age,
            row["scan_id"],
        )

    return sorted(rows, key=sort_key)


def load_latents() -> tuple[dict[str, np.ndarray], dict[str, str]]:
    latents: dict[str, np.ndarray] = {}
    archives: dict[str, str] = {}
    for split in ("train", "val", "test"):
        path = LATENT_DIR / f"{split}_latents.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing latent archive: {path}")
        with np.load(path, allow_pickle=False) as payload:
            for scan_id, latent in zip(payload["scan_ids"], payload["latents"]):
                key = Path(str(scan_id)).stem
                if key in latents:
                    raise ValueError(f"Duplicate latent scan ID: {key}")
                latents[key] = np.asarray(latent, dtype=np.float32)
                archives[key] = str(path)
    return latents, archives


def parse_gpu_ids(value: str) -> list[int]:
    try:
        gpu_ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid --gpus value: {value!r}") from exc
    if not gpu_ids:
        raise ValueError("--gpus must contain at least one device index")
    if len(set(gpu_ids)) != len(gpu_ids) or min(gpu_ids) < 0:
        raise ValueError("--gpus must contain unique non-negative device indices")
    return gpu_ids


def load_report(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def existing_result_is_valid(
    mesh_path: Path,
    report_path: Path,
    *,
    checkpoint: str,
    resolution: int,
) -> bool:
    if not mesh_path.is_file() or mesh_path.stat().st_size == 0:
        return False
    report = load_report(report_path)
    if report is None:
        return False
    return bool(
        report.get("status") == "ok"
        and report.get("checkpoint_requested") == checkpoint
        and int(report.get("resolution", -1)) == resolution
        and Path(str(report.get("mesh_path", ""))).resolve() == mesh_path.resolve()
    )


def worker_main(
    worker_index: int,
    device_index: int | None,
    assignments: list[dict[str, Any]],
    settings: dict[str, Any],
    result_queue,
) -> None:
    worker_name = (
        f"gpu_{device_index}" if device_index is not None else f"cpu_{worker_index}"
    )
    progress_path = Path(settings["progress_dir"]) / f"{worker_name}.csv"
    progress: list[dict[str, Any]] = []
    try:
        import torch
        import trimesh

        if str(TASK2_SCRIPTS) not in sys.path:
            sys.path.insert(0, str(TASK2_SCRIPTS))
        from task2_common import (  # pylint: disable=import-error
            decode_latent_to_mesh,
            load_config,
            load_decoder_checkpoint,
        )

        if device_index is None:
            device = torch.device("cpu")
        else:
            torch.cuda.set_device(device_index)
            device = torch.device(f"cuda:{device_index}")
        config = load_config(CONFIG_PATH)
        decoder, checkpoint_payload, checkpoint_path = load_decoder_checkpoint(
            config, settings["checkpoint"], device
        )
        checkpoint_epoch = checkpoint_payload.get("epoch")

        for position, item in enumerate(assignments, start=1):
            scan_id = item["scan_id"]
            split = item["split"]
            mesh_path = Path(settings["output_dir"]) / split / f"{scan_id}.ply"
            report_path = Path(settings["report_dir"]) / split / f"{scan_id}.json"
            started = time.monotonic()
            base = {
                "scan_id": scan_id,
                "split": split,
                "subject_id": item["subject_id"],
                "worker": worker_name,
                "device": str(device),
                "mesh_path": str(mesh_path),
                "report_path": str(report_path),
                "resolution": int(settings["resolution"]),
                "max_batch": int(settings["max_batch"]),
                "checkpoint_requested": settings["checkpoint"],
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_epoch": checkpoint_epoch,
                "latent_archive": item["latent_archive"],
            }
            if not settings["overwrite"] and existing_result_is_valid(
                mesh_path,
                report_path,
                checkpoint=settings["checkpoint"],
                resolution=settings["resolution"],
            ):
                record = {**base, "status": "skipped_valid", "error": ""}
                progress.append(record)
                write_csv_atomic(progress, progress_path)
                print(
                    f"[{worker_name}] {position}/{len(assignments)} {scan_id}: skipped",
                    flush=True,
                )
                continue

            mesh_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_mesh = mesh_path.with_name(f".{scan_id}.tmp.ply")
            temporary_mesh.unlink(missing_ok=True)
            try:
                mesh_stats = decode_latent_to_mesh(
                    decoder,
                    item["latent"],
                    temporary_mesh,
                    resolution=settings["resolution"],
                    max_batch=settings["max_batch"],
                    device=device,
                )
                mesh = trimesh.load(temporary_mesh, process=False, force="mesh")
                if isinstance(mesh, trimesh.Scene):
                    mesh = mesh.dump(concatenate=True)
                if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                    raise RuntimeError("Decoded mesh has no vertices or faces")
                temporary_mesh.replace(mesh_path)
                finished = utc_now()
                report = {
                    **base,
                    **mesh_stats,
                    "mesh_path": str(mesh_path),
                    "status": "ok",
                    "error": "",
                    "volume": abs(float(mesh.volume)),
                    "watertight": bool(mesh.is_watertight),
                    "winding_consistent": bool(mesh.is_winding_consistent),
                    "started_at": item["run_started_at"],
                    "finished_at": finished,
                    "duration_seconds": time.monotonic() - started,
                }
                write_json_atomic(report, report_path)
                record = report
            except Exception as exc:  # keep the shard running after one bad scan
                temporary_mesh.unlink(missing_ok=True)
                record = {
                    **base,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "duration_seconds": time.monotonic() - started,
                    "finished_at": utc_now(),
                }
                write_json_atomic(record, report_path)
            progress.append(record)
            write_csv_atomic(progress, progress_path)
            print(
                f"[{worker_name}] {position}/{len(assignments)} {scan_id}: "
                f"{record['status']}",
                flush=True,
            )

        result_queue.put(
            {
                "worker": worker_name,
                "assigned": len(assignments),
                "ok": sum(row["status"] == "ok" for row in progress),
                "skipped_valid": sum(
                    row["status"] == "skipped_valid" for row in progress
                ),
                "failed": sum(row["status"] == "failed" for row in progress),
            }
        )
    except Exception as exc:
        result_queue.put(
            {
                "worker": worker_name,
                "fatal_error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        raise


def collect_reconstruction_manifest(
    assignments: list[dict[str, Any]], settings: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    counts = {"ok": 0, "missing": 0, "failed": 0}
    for item in assignments:
        scan_id = item["scan_id"]
        split = item["split"]
        report_path = Path(settings["report_dir"]) / split / f"{scan_id}.json"
        mesh_path = Path(settings["output_dir"]) / split / f"{scan_id}.ply"
        report = load_report(report_path)
        if report is None:
            status = "missing"
            error = "missing or invalid report"
        elif existing_result_is_valid(
            mesh_path,
            report_path,
            checkpoint=settings["checkpoint"],
            resolution=settings["resolution"],
        ):
            status = "ok"
            error = ""
        else:
            status = "failed"
            error = str(report.get("error", "report does not match requested run"))
        counts[status] += 1
        rows.append(
            {
                "scan_id": scan_id,
                "subject_id": item["subject_id"],
                "split": split,
                "diagnosis": item.get("diagnosis", ""),
                "status": status,
                "error": error,
                "mesh_path": str(mesh_path),
                "report_path": str(report_path),
                "latent_archive": item["latent_archive"],
            }
        )
    return rows, counts


def main() -> int:
    args = parse_args()
    if args.resolution < 32:
        raise ValueError("--resolution must be at least 32")
    if args.max_batch <= 0:
        raise ValueError("--max-batch must be positive")
    if args.limit < 0:
        raise ValueError("--limit cannot be negative")
    if args.cpu_smoke_workers < 0:
        raise ValueError("--cpu-smoke-workers cannot be negative")
    for required in (CONFIG_PATH, MANIFEST_PATH, ANALYSIS_SCRIPT):
        if not required.is_file():
            raise FileNotFoundError(required)

    gpu_ids = parse_gpu_ids(args.gpus)
    if args.cpu_smoke_workers:
        devices: list[int | None] = [None] * args.cpu_smoke_workers
    else:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available to PyTorch")
        device_count = torch.cuda.device_count()
        unavailable = [gpu for gpu in gpu_ids if gpu >= device_count]
        if unavailable:
            raise RuntimeError(
                f"Requested CUDA devices {unavailable}, but PyTorch sees {device_count} devices"
            )
        devices = list(gpu_ids)
        print(
            "CUDA devices: "
            + ", ".join(
                f"{gpu}={torch.cuda.get_device_name(gpu)}" for gpu in gpu_ids
            )
        )

    output_dir = Path(args.output_dir).expanduser().resolve()
    report_dir = Path(args.report_dir).expanduser().resolve()
    progress_dir = Path(args.progress_dir).expanduser().resolve()
    for directory in (output_dir, report_dir, progress_dir):
        directory.mkdir(parents=True, exist_ok=True)

    manifest = read_manifest()
    if args.limit:
        manifest = manifest[: args.limit]
    latents, archives = load_latents()
    missing_latents = [row["scan_id"] for row in manifest if row["scan_id"] not in latents]
    if missing_latents:
        raise RuntimeError(
            f"Missing {len(missing_latents)} existing latents; first: {missing_latents[:5]}"
        )
    run_started_at = utc_now()
    assignments = [
        {
            **row,
            "latent": latents[row["scan_id"]],
            "latent_archive": archives[row["scan_id"]],
            "run_started_at": run_started_at,
        }
        for row in manifest
    ]
    shards = [assignments[index:: len(devices)] for index in range(len(devices))]
    estimated_bytes = len(assignments) * 3_000_000
    free_bytes = shutil.disk_usage(output_dir).free
    if free_bytes < estimated_bytes + 2_000_000_000:
        raise RuntimeError(
            f"Insufficient free space: {free_bytes / 1e9:.1f} GB free, "
            f"approximately {estimated_bytes / 1e9:.1f} GB of meshes expected"
        )

    settings = {
        "checkpoint": args.checkpoint,
        "resolution": args.resolution,
        "max_batch": args.max_batch,
        "output_dir": str(output_dir),
        "report_dir": str(report_dir),
        "progress_dir": str(progress_dir),
        "overwrite": args.overwrite,
    }
    run_config = {
        **settings,
        "manifest": str(MANIFEST_PATH),
        "config": str(CONFIG_PATH),
        "worker_devices": devices,
        "selected_scans": len(assignments),
        "run_started_at": run_started_at,
        "new_latents_created": False,
    }
    write_json_atomic(run_config, progress_dir / "run_config.json")
    print(
        f"Selected {len(assignments)} scans across {len(devices)} workers: "
        + ", ".join(str(len(shard)) for shard in shards),
        flush=True,
    )

    context = mp.get_context("spawn")
    result_queue = context.Queue()
    processes: list[mp.Process] = []
    for worker_index, (device_index, shard) in enumerate(zip(devices, shards)):
        process = context.Process(
            target=worker_main,
            args=(worker_index, device_index, shard, settings, result_queue),
            name=(
                f"decode-gpu-{device_index}"
                if device_index is not None
                else f"decode-cpu-{worker_index}"
            ),
        )
        process.start()
        processes.append(process)
    for process in processes:
        process.join()

    worker_results: list[dict[str, Any]] = []
    while True:
        try:
            worker_results.append(result_queue.get_nowait())
        except queue.Empty:
            break
    exit_codes = {process.name: process.exitcode for process in processes}
    reconstruction_rows, counts = collect_reconstruction_manifest(assignments, settings)
    write_csv_atomic(
        reconstruction_rows, progress_dir / "reconstructed_meshes.csv"
    )
    summary = {
        **run_config,
        "run_finished_at": utc_now(),
        "counts": counts,
        "worker_results": worker_results,
        "worker_exit_codes": exit_codes,
        "reconstruction_manifest": str(progress_dir / "reconstructed_meshes.csv"),
    }
    write_json_atomic(summary, progress_dir / "summary.json")
    print(json.dumps(summary, indent=2, sort_keys=True))

    worker_failed = any(code != 0 for code in exit_codes.values())
    if worker_failed or counts["failed"] or counts["missing"]:
        print(
            "Decoding was incomplete. Re-run the same command after inspecting "
            f"{progress_dir / 'summary.json'}; valid scans will be skipped.",
            file=sys.stderr,
        )
        return 2

    if args.run_final_analysis:
        command = [
            sys.executable,
            str(ANALYSIS_SCRIPT),
            "--large-siren-mesh-dir",
            str(output_dir),
            "--output-dir",
            str(Path(args.analysis_output_dir).expanduser().resolve()),
            "--mesh-cache",
            str(Path(args.analysis_mesh_cache).expanduser().resolve()),
        ]
        print("Running Stage 1 analysis: " + " ".join(command), flush=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
