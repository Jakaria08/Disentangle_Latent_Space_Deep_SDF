#!/usr/bin/env python3
"""Run the lean latent-cocycle sweep across all free GPUs.

Each job is one (variant, representation) pair trained by ``train_lean.py`` in its own
process. Jobs are dispatched to GPU slots as slots free up; progress is written to
``sweep_status.json`` after every state change so the sweep can be watched from outside.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import core  # noqa: E402

C, _ = core()

PYTHON = "/home/jakaria/anaconda3/envs/pytorch_geo/bin/python"
TASK_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, default=TASK_ROOT / "configs" / "sweep.json")
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--per-gpu", type=int, default=2)
    parser.add_argument("--only-variant", default=None, help="comma-separated variant ids")
    parser.add_argument("--only-representation", default=None, help="comma-separated names")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sweep = json.loads(Path(args.sweep).read_text())
    jobs = sweep["jobs"]
    if args.only_variant:
        keep = set(args.only_variant.split(","))
        jobs = [job for job in jobs if job["id"] in keep]
    if args.only_representation:
        keep = set(args.only_representation.split(","))
        jobs = [job for job in jobs if job["representation"] in keep]
    if not jobs:
        raise SystemExit("No jobs selected")

    root = C.output_root() / "runs"
    config_dir = C.output_root() / "job_configs"
    root.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    status_path = C.output_root() / "sweep_status.json"

    slots = [f"cuda:{index}" for index in args.gpus.split(",") for _ in range(int(args.per_gpu))]
    print(f"{len(jobs)} jobs over {len(slots)} slots ({args.gpus}, {args.per_gpu}/gpu)", flush=True)
    if args.dry_run:
        for job in jobs:
            print(f"  {job['id']:24s} {job['representation']}")
        return 0

    pending = list(jobs)
    running: list[dict] = []
    done: list[dict] = []
    started = time.time()

    def write_status(state: str) -> None:
        status_path.write_text(
            json.dumps(
                {
                    "state": state,
                    "elapsed_minutes": (time.time() - started) / 60.0,
                    "total": len(jobs),
                    "pending": len(pending),
                    "running": [f"{r['job']['id']}/{r['job']['representation']}@{r['device']}" for r in running],
                    "completed": len(done),
                    "failed": [d["name"] for d in done if d["returncode"] != 0],
                    "results": [
                        {k: d[k] for k in ("name", "returncode", "minutes", "best_score", "best_step")}
                        for d in done
                    ],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    write_status("running")
    while pending or running:
        while pending and len(running) < len(slots):
            used = [r["device"] for r in running]
            device = next(s for s in slots if used.count(s) < slots.count(s))
            job = pending.pop(0)
            name = f"{job['id']}__{job['representation']}"
            config_path = config_dir / f"{name}.json"
            config_path.write_text(json.dumps(job, indent=2) + "\n")
            output_dir = root / job["id"] / job["representation"]
            output_dir.mkdir(parents=True, exist_ok=True)
            log = (output_dir / "train.log").open("w", encoding="utf-8")
            process = subprocess.Popen(
                [PYTHON, str(TASK_ROOT / "scripts" / "train_lean.py"),
                 "--config", str(config_path), "--device", device, "--output-dir", str(output_dir)],
                stdout=log, stderr=subprocess.STDOUT, cwd=str(TASK_ROOT / "scripts"),
            )
            running.append({"job": job, "name": name, "device": device, "process": process,
                            "log": log, "output": output_dir, "started": time.time()})
            print(f"  start {name} on {device}", flush=True)
            write_status("running")

        time.sleep(float(args.poll_seconds))
        for record in list(running):
            code = record["process"].poll()
            if code is None:
                continue
            record["log"].close()
            running.remove(record)
            summary_path = record["output"] / "summary.json"
            best_score = best_step = None
            if summary_path.is_file():
                summary = json.loads(summary_path.read_text())
                best_score, best_step = summary.get("best_score"), summary.get("best_step")
            done.append({"name": record["name"], "returncode": int(code),
                         "minutes": (time.time() - record["started"]) / 60.0,
                         "best_score": best_score, "best_step": best_step})
            flag = "ok " if code == 0 else "FAIL"
            print(f"  {flag} {record['name']} ({(time.time()-record['started'])/60.0:.1f} min) "
                  f"best={best_score}@{best_step}", flush=True)
            write_status("running")

    write_status("complete")
    failed = [d["name"] for d in done if d["returncode"] != 0]
    print(f"\nsweep complete in {(time.time()-started)/60.0:.1f} min; "
          f"{len(done)-len(failed)}/{len(done)} succeeded", flush=True)
    if failed:
        print("failed: " + ", ".join(failed), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
