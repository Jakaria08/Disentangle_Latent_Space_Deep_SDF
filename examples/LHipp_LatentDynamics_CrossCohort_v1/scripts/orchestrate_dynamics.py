#!/usr/bin/env python3
"""Run a job file on GPUs 0 and 2 as a detached, resumable queue.

Launch detached so an SSH disconnect cannot stop it:

    setsid nohup python orchestrate_dynamics.py --jobs <jobs.json> > /dev/null 2>&1 < /dev/null &

Job file (written by stage2_build_jobs.py): {"name": ..., "jobs": [{"id", "argv", "weight",
"after", "done_marker"}]}. ``argv`` is a script plus its arguments, without ``--device``; the
orchestrator appends ``--device cuda:<gpu>``. ``weight`` is the capacity a job uses on its GPU
(a decoder-in-loss cocycle counts 2). A job starts once every id in ``after`` is ``ok``.

State lives in ``<jobs>.state.json``; each job logs to ``<jobs dir>/logs/<id>.log``. On restart:
jobs already ``ok`` are skipped; a job whose ``done_marker`` exists is marked ``ok``; a job left
``running`` by a dead orchestrator is re-queued. A failure never stops unrelated jobs, but its
dependents are marked ``blocked``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import benchmark_common as bc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--gpus", default="0,2")
    parser.add_argument("--capacity-per-gpu", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def marker_done(marker: Any) -> bool:
    """A done marker is a path that must exist, or {"path", "key", "equals"} for a JSON field."""
    if not marker:
        return False
    if isinstance(marker, str):
        return Path(marker).exists()
    path = Path(marker["path"])
    if not path.is_file():
        return False
    try:
        return bc.read_json(path).get(marker["key"]) == marker["equals"]
    except (ValueError, OSError):
        return False


def alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class Queue:
    def __init__(self, jobs_path: Path, gpus: list[int], capacity: int) -> None:
        self.jobs_path = jobs_path.resolve()
        bc.require_bulk(self.jobs_path, "job file")
        spec = bc.read_json(self.jobs_path)
        self.name = spec["name"]
        self.jobs = {job["id"]: job for job in spec["jobs"]}
        self.gpus = gpus
        self.capacity = capacity
        self.state_path = self.jobs_path.with_suffix(".state.json")
        self.log_dir = self.jobs_path.parent / "logs" / self.name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.state: dict[str, dict[str, Any]] = bc.read_json(self.state_path)["jobs"] if self.state_path.is_file() else {}
        self.processes: dict[str, subprocess.Popen] = {}
        for job_id, job in self.jobs.items():
            unknown = set(job.get("after", [])).difference(self.jobs)
            if unknown:
                raise KeyError(f"{job_id}: unknown dependencies {sorted(unknown)}")
            entry = self.state.setdefault(job_id, {"status": "pending"})
            if entry["status"] == "running" and not alive(entry.get("pid")):
                entry.update(status="pending", note="re-queued after orchestrator restart")
            if entry["status"] in ("failed", "blocked"):
                entry.update(status="pending", note=f"retry after previous {entry['status']}")
            if entry["status"] != "ok" and marker_done(job.get("done_marker")):
                entry.update(status="ok", note="done marker present")

    def save(self) -> None:
        counts: dict[str, int] = {}
        for entry in self.state.values():
            counts[entry["status"]] = counts.get(entry["status"], 0) + 1
        bc.atomic_json(self.state_path, {"name": self.name, "updated": time.strftime("%F %T"), "counts": counts, "jobs": self.state})

    def load_on(self, gpu: int) -> int:
        return sum(int(self.jobs[j].get("weight", 1)) for j, e in self.state.items() if e["status"] == "running" and e.get("gpu") == gpu)

    def launch(self, job_id: str, gpu: int) -> None:
        job = self.jobs[job_id]
        argv = [bc.GPU_PYTHON, *job["argv"], "--device", f"cuda:{gpu}"]
        log = self.log_dir / f"{job_id}.log"
        handle = open(log, "a", encoding="utf-8")
        handle.write(f"\n$ {' '.join(argv)}\n")
        handle.flush()
        process = subprocess.Popen(argv, cwd=str(bc.SCRIPT_DIR), stdout=handle, stderr=subprocess.STDOUT)
        self.processes[job_id] = process
        self.state[job_id].update(status="running", gpu=gpu, pid=process.pid, started=time.strftime("%F %T"), log=str(log))

    def step(self) -> bool:
        for job_id, process in list(self.processes.items()):
            code = process.poll()
            if code is None:
                continue
            del self.processes[job_id]
            marker = self.jobs[job_id].get("done_marker")
            ok = code == 0 and (not marker or marker_done(marker))
            self.state[job_id].update(status="ok" if ok else "failed", returncode=code, finished=time.strftime("%F %T"))
        for job_id, job in self.jobs.items():
            entry = self.state[job_id]
            if entry["status"] == "pending" and any(self.state[d]["status"] in ("failed", "blocked") for d in job.get("after", [])):
                entry.update(status="blocked", note="a dependency failed")
        for job_id, job in self.jobs.items():
            entry = self.state[job_id]
            if entry["status"] != "pending" or any(self.state[d]["status"] != "ok" for d in job.get("after", [])):
                continue
            weight = int(job.get("weight", 1))
            gpu = min(self.gpus, key=self.load_on)
            if self.load_on(gpu) + weight <= self.capacity:
                self.launch(job_id, gpu)
        self.save()
        return any(e["status"] in ("pending", "running") for e in self.state.values())


def main() -> int:
    args = parse_args()
    gpus = [int(g) for g in args.gpus.split(",")]
    for gpu in gpus:
        bc.require_allowed_gpu(f"cuda:{gpu}")
    queue = Queue(args.jobs, gpus, args.capacity_per_gpu)
    if args.dry_run:
        for job_id, job in queue.jobs.items():
            print(f"{queue.state[job_id]['status']:8} w={job.get('weight', 1)} after={job.get('after', [])} {job_id}: {' '.join(job['argv'][:6])} ...")
        return 0
    queue.save()
    print(f"[{queue.name}] {len(queue.jobs)} jobs on GPUs {gpus}, capacity {args.capacity_per_gpu} each", flush=True)
    while queue.step():
        time.sleep(args.poll_seconds)
    counts = bc.read_json(queue.state_path)["counts"]
    print(f"[{queue.name}] finished: {json.dumps(counts)}", flush=True)
    return 0 if set(counts) == {"ok"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
