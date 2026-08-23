#!/usr/bin/env python3
"""Run the tested multires trainer after enforcing the CALSNIC exact-audit gate."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from delegate_multires import delegate


def argument(name: str) -> str:
    if name not in sys.argv:
        raise ValueError(f"Required argument is missing: {name}")
    index = sys.argv.index(name)
    if index + 1 >= len(sys.argv):
        raise ValueError(f"Argument has no value: {name}")
    return sys.argv[index + 1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def enforce_audit(config_path: Path) -> None:
    config = json.loads(config_path.read_text())
    audit_value = config.get("required_exact_audit")
    if not audit_value:
        return
    audit_path = Path(audit_value)
    if not audit_path.is_file():
        raise FileNotFoundError(f"Required full exact-SDF audit does not exist: {audit_path}")
    report = json.loads(audit_path.read_text())
    if not report.get("passed") or not report.get("full_cohort_audit") or not report.get("training_allowed"):
        raise RuntimeError(f"Exact-SDF audit does not authorize training: {audit_path}")
    manifest = Path(config["manifest"])
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if report.get("exact_manifest_sha256") != sha256(manifest):
        raise RuntimeError("Exact manifest changed after the successful audit; rerun the audit.")


if __name__ == "__main__":
    if "--help" not in sys.argv and "-h" not in sys.argv:
        enforce_audit(Path(argument("--config")).resolve())
    delegate("train_multires_sdf.py")
