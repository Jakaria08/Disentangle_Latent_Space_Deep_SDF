#!/usr/bin/env python3
"""Evaluate a Pareto shortlist on validation only, then select one checkpoint."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from siren256_common import read_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-pairs", type=int, default=None)
    args = parser.parse_args()
    run = args.run.resolve()
    manifest = read_json(run / "validation_candidates.json")
    candidates = [candidate for candidate in manifest.get("candidates", []) if candidate.get("shortlisted")]
    if not candidates:
        raise RuntimeError("No shortlisted candidates. Train with UseParetoValidationSelection=true.")
    scripts = Path(__file__).resolve().parent
    evaluator = scripts / "evaluate_siren256_transport.py"
    for candidate in candidates:
        command = [sys.executable, str(evaluator), "--run", str(run), "--split", "val", "--checkpoint", str(candidate["checkpoint"]), "--tag", str(candidate["tag"]), "--device", str(args.device)]
        if args.max_pairs is not None:
            command.extend(("--max-pairs", str(args.max_pairs)))
        subprocess.run(command, check=True)
    subprocess.run([sys.executable, str(scripts / "select_siren256_transport_checkpoint.py"), "--run", str(run)], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
