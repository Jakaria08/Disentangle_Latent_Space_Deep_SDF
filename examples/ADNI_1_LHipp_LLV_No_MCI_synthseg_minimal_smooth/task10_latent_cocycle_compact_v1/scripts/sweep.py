#!/usr/bin/env python3
"""Dispatch the preregistered compact latent-objective experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bootstrap import core  # noqa: E402
from configuration import all_jobs, read_experiment  # noqa: E402

C, _ = core()
TASK_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=None)
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--per-gpu", type=int, default=1)
    parser.add_argument("--arm", default=None, help="comma-separated experiment arms")
    parser.add_argument("--objective", default=None, help="comma-separated objectives")
    parser.add_argument("--representation", default=None, help="comma-separated representations")
    parser.add_argument("--seed", default=None, help="comma-separated integer seeds")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def selected_jobs(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    experiment = read_experiment(args.experiment) if args.experiment else read_experiment()
    jobs = all_jobs(experiment)
    filters = {
        "arm": set(args.arm.split(",")) if args.arm else None,
        "objective": set(args.objective.split(",")) if args.objective else None,
        "representation": set(args.representation.split(",")) if args.representation else None,
        "seed": set(int(value) for value in args.seed.split(",")) if args.seed else None,
    }
    for key, values in filters.items():
        if values is not None:
            jobs = [
                job
                for job in jobs
                if (
                    int(job["training"]["seed"])
                    if key == "seed"
                    else job[key]
                )
                in values
            ]
    if not jobs:
        raise ValueError("No jobs match the requested filters")
    return experiment, jobs


def main() -> int:
    args = parse_args()
    _, jobs = selected_jobs(args)
    registry = C.load_registry()
    output_root = C.output_root(registry)
    runs_root = output_root / "runs"
    configs_root = output_root / "job_configs"
    completed: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for job in jobs:
        output = (
            runs_root
            / job["representation"]
            / job["arm"]
            / f"seed_{job['training']['seed']}"
        )
        summary = output / "summary.json"
        if summary.is_file() and json.loads(summary.read_text()).get("status") == "complete":
            completed.append(
                {"id": job["id"], "returncode": 0, "state": "already_complete"}
            )
        else:
            pending.append(job)

    print(
        f"selected={len(jobs)} pending={len(pending)} already_complete={len(completed)}",
        flush=True,
    )
    if args.dry_run:
        for job in jobs:
            print(
                f"{job['id']:50s} objective={job['objective']:8s} "
                f"lr={job['training']['learning_rate_profile']:8s} "
                f"peak={job['training']['peak_learning_rate']:.1e} "
                f"batch={job['training']['batch_size']:3d} "
                f"epochs={job['training']['epochs']} "
                f"steps/epoch={job['training']['steps_per_epoch']}"
            )
        return 0

    gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpu_ids or int(args.per_gpu) < 1:
        raise ValueError("At least one GPU and one slot per GPU are required")
    slots = [f"cuda:{gpu}" for gpu in gpu_ids for _ in range(int(args.per_gpu))]
    runs_root.mkdir(parents=True, exist_ok=True)
    configs_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "sweep_status.json"
    running: list[dict[str, Any]] = []
    started = time.time()

    def write_status(state: str) -> None:
        C.atomic_json(
            status_path,
            {
                "state": state,
                "selected": len(jobs),
                "pending": len(pending),
                "running": [record["id"] for record in running],
                "completed": len(completed),
                "failed": [
                    record["id"] for record in completed if record.get("returncode") != 0
                ],
                "elapsed_minutes": (time.time() - started) / 60.0,
                "results": completed,
            },
        )

    write_status("running")
    while pending or running:
        while pending and len(running) < len(slots):
            used = [record["device"] for record in running]
            device = next(slot for slot in slots if used.count(slot) < slots.count(slot))
            job = pending.pop(0)
            config_path = configs_root / f"{job['id']}.json"
            C.atomic_json(config_path, job)
            output = (
                runs_root
                / job["representation"]
                / job["arm"]
                / f"seed_{job['training']['seed']}"
            )
            output.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(TASK_ROOT / "scripts" / "train.py"),
                "--config",
                str(config_path),
                "--device",
                device,
                "--output-dir",
                str(output),
            ]
            if (output / "latest.pt").is_file():
                command.append("--resume")
            log = (output / "train.log").open("a", encoding="utf-8")
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=str(TASK_ROOT / "scripts"),
            )
            running.append(
                {
                    "id": job["id"],
                    "device": device,
                    "process": process,
                    "log": log,
                    "output": output,
                    "started": time.time(),
                }
            )
            print(f"start {job['id']} on {device}", flush=True)
            write_status("running")

        time.sleep(float(args.poll_seconds))
        for record in list(running):
            code = record["process"].poll()
            if code is None:
                continue
            record["log"].close()
            running.remove(record)
            result = {
                "id": record["id"],
                "returncode": int(code),
                "state": "complete" if code == 0 else "failed",
                "minutes": (time.time() - record["started"]) / 60.0,
            }
            summary_path = record["output"] / "summary.json"
            if summary_path.is_file():
                summary = json.loads(summary_path.read_text())
                result["best_score"] = summary["best_mature"]["score"]
                result["best_epoch"] = summary["best_mature"]["epoch"]
            completed.append(result)
            print(
                f"{result['state']} {record['id']} ({result['minutes']:.1f} min)",
                flush=True,
            )
            write_status("running")

    failed = [record for record in completed if record.get("returncode") != 0]
    write_status("complete_with_failures" if failed else "complete")
    print(
        f"finished {len(completed) - len(failed)}/{len(completed)} jobs successfully",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
