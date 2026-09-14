#!/usr/bin/env python3
"""Run the whole cross-cohort sequence unattended, in dependency order.

The phases are not independent: the gate in phase 2 decides whether phases 3-5 mean
anything, and OASIS cannot enter phase 1 until its mesh run has written *both* structures.
This driver encodes that order, waits for what is not ready yet, and records what it did.

Three properties make it safe to leave alone:

* **Resumable.**  Every step declares how to tell whether it is already done, so re-running
  after an interruption skips completed work instead of repeating or duplicating it.
* **Gated.**  A step marked ``gate`` stops the sequence when it fails - the shared-space
  check is the obvious one, because nothing downstream is interpretable without it.  A
  non-gate failure (one cohort's training) is recorded and the sequence continues.
* **Patient.**  Steps may declare a precondition to wait for, polled rather than assumed.

State lands in ``reports/sequence_state.json`` and per-step logs in ``logs/``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import xcohort_common as xc

INR_PYTHON = "/home/jakaria/anaconda3/envs/inr_sdf/bin/python"
PGEO_PYTHON = "/home/jakaria/anaconda3/envs/pytorch_geo/bin/python"
SCRIPTS = xc.TASK_ROOT / "scripts"
LOGS = xc.TASK_ROOT / "logs"
STATE_FP = xc.TASK_ROOT / "reports" / "sequence_state.json"
SPIRAL_MODELS = ["spiralnet_z128", "adaptive_z128"]
LAMM_MODEL = "lamm_z128"
# GPU 1 is deliberately left free.
DEFAULT_GPUS = "0,2"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Step:
    name: str
    command: list[str]
    done: Callable[[], bool]
    gate: bool = False
    wait_for: Callable[[], bool] | None = None
    wait_description: str = ""
    env: dict[str, str] = field(default_factory=dict)
    timeout_s: int = 6 * 3600


# --------------------------------------------------------------------------------------
# predicates
# --------------------------------------------------------------------------------------


def cohort_ready(name: str) -> bool:
    """True when a cohort has a usable keep manifest (phase 1 done)."""
    config = xc.load_config()
    try:
        xc.read_manifest(config.cohorts[name], config)
        return True
    except (FileNotFoundError, ValueError):
        return False


def meshes_complete(name: str) -> bool:
    """True when the mesh run wrote both structures and its validation passed."""
    config = xc.load_config()
    root = config.cohorts[name].mesh_root
    summary = root / "reports" / "validation_summary.json"
    if not summary.is_file():
        return False
    try:
        payload = json.loads(summary.read_text())
    except json.JSONDecodeError:
        return False
    if not payload.get("passed"):
        return False
    return all(
        (root / structure / "minimal_smooth_correspondence" / "final_ply_mm").is_dir()
        and any((root / structure / "minimal_smooth_correspondence" / "final_ply_mm").glob("*.ply"))
        for structure in ("left_hippocampus", "left_lateral_ventricle")
    )


def report_exists(pattern: str) -> Callable[[], bool]:
    return lambda: any((xc.TASK_ROOT / "reports").glob(pattern))


def no_other_training() -> bool:
    """True when no phase-4 job launched outside this sequence is still running."""
    result = subprocess.run(
        ["pgrep", "-f", "phase4_learned_protocols.py"], capture_output=True, text=True
    )
    mine = str(os.getpid())
    pids = [p for p in result.stdout.split() if p != mine]
    return not pids


def results_contain(protocol: str, model: str, cohort: str) -> bool:
    path = xc.TASK_ROOT / "reports" / f"phase4_{protocol}.csv"
    if not path.is_file():
        return False
    import csv

    with open(path, newline="") as handle:
        return any(
            row.get("protocol") == protocol and row.get("model") == model and row.get("eval_cohort") == cohort
            for row in csv.DictReader(handle)
        )


# --------------------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------------------


def build_plan(args: argparse.Namespace) -> list[Step]:
    config = xc.load_config()
    targets = [c.name for c in config.targets()]
    gpus = [int(g) for g in args.gpus.split(",")]
    steps: list[Step] = []

    # 1. Phase 1 for each target cohort, once its meshes are complete.
    for index, name in enumerate(targets):
        steps.append(
            Step(
                name=f"phase1_{name}",
                command=[INR_PYTHON, str(SCRIPTS / "phase1_build_cohorts.py"), "--cohorts", name],
                done=lambda n=name: cohort_ready(n),
                wait_for=lambda n=name: meshes_complete(n),
                wait_description=f"{name} mesh run to finish both structures",
                timeout_s=3 * 3600,
            )
        )

    ready = lambda: [n for n in config.cohorts if cohort_ready(n)]  # noqa: E731

    # 2. The gate. Nothing downstream is meaningful if the vertex spaces differ.
    steps.append(
        Step(
            name="phase2_gate",
            command=[INR_PYTHON, str(SCRIPTS / "phase2_shared_space_audit.py"), "--split", "test", "--k", "128"],
            done=lambda: False,  # cheap and worth re-running once every cohort is in
            gate=True,
            timeout_s=3600,
        )
    )

    # 3. PCA answers the whole question cheaply, before any GPU time.
    steps.append(
        Step(
            name="phase3_pca",
            command=[INR_PYTHON, str(SCRIPTS / "phase3_pca_protocols.py"), "--splits", "val", "test"],
            done=lambda: False,
            timeout_s=2 * 3600,
        )
    )

    # 4. Learned models. One cohort per GPU; a failure here is recorded, not fatal.
    for index, name in enumerate(sorted(config.cohorts)):
        gpu = gpus[index % len(gpus)]
        steps.append(
            Step(
                name=f"phase4_internal_{name}",
                command=[
                    PGEO_PYTHON, str(SCRIPTS / "phase4_learned_protocols.py"),
                    "--protocol", "internal", "--cohorts", name,
                    "--models", *SPIRAL_MODELS, "--gpu", "0",
                ],
                done=lambda n=name: all(results_contain("internal", m, n) for m in SPIRAL_MODELS),
                wait_for=no_other_training,
                wait_description="other phase-4 jobs to finish",
                env={"CUDA_VISIBLE_DEVICES": str(gpu)},
            )
        )
        # LAMM is a separate step so that a transformer failure cannot block the spiral results.
        steps.append(
            Step(
                name=f"phase4_lamm_{name}",
                command=[
                    PGEO_PYTHON, str(SCRIPTS / "phase4_learned_protocols.py"),
                    "--protocol", "internal", "--cohorts", name,
                    "--models", LAMM_MODEL, "--gpu", "0",
                ],
                done=lambda n=name: results_contain("internal", LAMM_MODEL, n),
                wait_for=no_other_training,
                wait_description="other phase-4 jobs to finish",
                env={"CUDA_VISIBLE_DEVICES": str(gpu)},
            )
        )

    # 5. External validation: inference only, so it is quick.
    steps.append(
        Step(
            name="phase4_external",
            command=[
                PGEO_PYTHON, str(SCRIPTS / "phase4_learned_protocols.py"),
                "--protocol", "external", "--models", *SPIRAL_MODELS, "--gpu", "0",
            ],
            done=lambda: all(results_contain("external", m, n) for m in SPIRAL_MODELS for n in targets),
            wait_for=no_other_training,
            wait_description="other phase-4 jobs to finish",
            env={"CUDA_VISIBLE_DEVICES": str(gpus[0])},
            timeout_s=2 * 3600,
        )
    )

    # 6. Pooled and leave-one-cohort-out.
    steps.append(
        Step(
            name="phase5_pooled",
            command=[
                PGEO_PYTHON, str(SCRIPTS / "phase4_learned_protocols.py"),
                "--protocol", "pooled", "--models", *SPIRAL_MODELS, "--gpu", "0",
            ],
            done=lambda: False,
            wait_for=no_other_training,
            wait_description="other phase-4 jobs to finish",
            env={"CUDA_VISIBLE_DEVICES": str(gpus[0])},
        )
    )
    for index, name in enumerate(c.name for c in config.pooled_members()):
        gpu = gpus[index % len(gpus)]
        steps.append(
            Step(
                name=f"phase5_loco_{name}",
                command=[
                    PGEO_PYTHON, str(SCRIPTS / "phase4_learned_protocols.py"),
                    "--protocol", "loco", "--held-out", name,
                    "--models", *SPIRAL_MODELS, "--gpu", "0",
                ],
                done=lambda: False,
                wait_for=no_other_training,
                wait_description="other phase-4 jobs to finish",
                env={"CUDA_VISIBLE_DEVICES": str(gpu)},
            )
        )

    # 7. Report, always last and always attempted.
    steps.append(
        Step(
            name="phase6_report",
            command=[INR_PYTHON, str(SCRIPTS / "phase6_report.py"), "--split", "test", "--k", "128"],
            done=lambda: False,
            timeout_s=1800,
        )
    )
    return steps


# --------------------------------------------------------------------------------------


def load_state() -> dict:
    if STATE_FP.is_file():
        try:
            return json.loads(STATE_FP.read_text())
        except json.JSONDecodeError:
            pass
    return {"started": now(), "steps": {}}


def save_state(state: dict) -> None:
    STATE_FP.parent.mkdir(parents=True, exist_ok=True)
    state["updated"] = now()
    STATE_FP.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def wait_until(step: Step, poll_s: int, state: dict) -> bool:
    if step.wait_for is None or step.wait_for():
        return True
    deadline = time.time() + step.timeout_s
    print(f"    waiting for {step.wait_description} (poll {poll_s}s, timeout {step.timeout_s // 60} min)", flush=True)
    while time.time() < deadline:
        time.sleep(poll_s)
        if step.wait_for():
            print(f"    precondition met at {now()}", flush=True)
            return True
    print(f"    TIMEOUT waiting for {step.wait_description}", flush=True)
    state["steps"][step.name] = {"status": "timeout_waiting", "at": now()}
    return False


def run_step(step: Step, state: dict, poll_s: int, dry_run: bool) -> str:
    print(f"\n=== {step.name} ===", flush=True)
    if step.done():
        print("    already done, skipping", flush=True)
        return "skipped"
    log_fp = LOGS / f"seq_{step.name}.log"
    pinned = step.env.get("CUDA_VISIBLE_DEVICES")
    print(f"    {'GPU ' + pinned + '  ' if pinned else ''}log: {log_fp}", flush=True)
    print("    $ " + " ".join(step.command), flush=True)
    if dry_run:
        # A dry run must never block: report what it would wait for and move on.
        if step.wait_for is not None and not step.wait_for():
            print(f"    would wait for {step.wait_description}", flush=True)
        return "dry_run"
    if not wait_until(step, poll_s, state):
        return "timeout_waiting"
    LOGS.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **step.env}
    started = time.time()
    with open(log_fp, "w") as handle:
        result = subprocess.run(step.command, stdout=handle, stderr=subprocess.STDOUT,
                                env=env, cwd=str(xc.TASK_ROOT), timeout=step.timeout_s)
    minutes = (time.time() - started) / 60
    status = "ok" if result.returncode == 0 else f"failed(rc={result.returncode})"
    print(f"    {status} in {minutes:.1f} min", flush=True)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpus", default=DEFAULT_GPUS, help="Comma-separated GPU indices to use.")
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", nargs="+", default=None, help="Run only these step names.")
    parser.add_argument("--skip", nargs="+", default=[], help="Step names to skip.")
    args = parser.parse_args()

    state = load_state()
    steps = build_plan(args)
    if args.only:
        steps = [s for s in steps if s.name in set(args.only)]
    steps = [s for s in steps if s.name not in set(args.skip)]

    print("=" * 88)
    print(f"Cross-cohort sequence: {len(steps)} step(s), GPUs {args.gpus}, started {now()}")
    print("=" * 88, flush=True)

    for step in steps:
        try:
            status = run_step(step, state, args.poll_seconds, args.dry_run)
        except subprocess.TimeoutExpired:
            status = "timeout_running"
        except Exception as exc:  # keep the sequence alive for non-gate failures
            status = f"error:{type(exc).__name__}:{exc}"
        state["steps"][step.name] = {"status": status, "at": now()}
        save_state(state)
        if step.gate and not status.startswith(("ok", "skipped", "dry_run")):
            print(f"\nGATE FAILED at {step.name} ({status}); stopping.", flush=True)
            return 1

    print("\n" + "=" * 88)
    print("Sequence complete. Per-step status:")
    for name, entry in state["steps"].items():
        print(f"  {name:28} {entry['status']}")
    print(f"State: {STATE_FP}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
