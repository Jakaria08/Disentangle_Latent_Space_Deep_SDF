#!/usr/bin/env python3
"""Shared paths, configuration and torch-free data logic for LHipp_LatentDynamics_CrossCohort_v1.

Code and configuration live in this repository; everything the experiment generates lives under
``BULK_ROOT`` on the 10 TB disk. Nothing here imports torch, so the data logic - trajectory
groups, split and fold assignment, evaluation tasks, baselines - is unit-testable in any env.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
TASK_ROOT = SCRIPT_DIR.parent
REPO_ROOT = TASK_ROOT.parents[1]
CONFIG_DIR = TASK_ROOT / "configs"
BULK_ROOT = Path("/mnt/bulk10tb/Deep3DComp/LHipp_LatentDynamics_CrossCohort_v1")
STAGE1_ROOT = BULK_ROOT / "stage1_data_foundation"

SPLITS = ("train", "val", "test")
LATENT_DIM = 128
REPRESENTATIONS = ("pca128", "spiralnet128", "adaptive128", "lamm128")
# Sensitivity-only codes (stage 5), never part of R1: configs/representation_registry_sensitivity.json.
SENSITIVITY_REPRESENTATIONS = ("pca128_pooled",)
ALL_REPRESENTATIONS = REPRESENTATIONS + SENSITIVITY_REPRESENTATIONS
ID_SEPARATOR = ":"
LABEL_ORDER = {"CN": 0, "MCI": 1, "AD": 2}
GPU_PYTHON = "/home/jakaria/anaconda3/envs/pytorch_geo/bin/python"
ALLOWED_GPUS = (0, 2)


# --------------------------------------------------------------------------------------
# paths and io
# --------------------------------------------------------------------------------------


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def require_bulk(path: str | Path, what: str = "generated output") -> Path:
    """Refuse to write generated data anywhere but the experiment's bulk root."""
    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(BULK_ROOT)
    except ValueError as error:
        raise ValueError(f"{what} must live under {BULK_ROOT}; refusing {resolved}") from error
    return resolved


def require_allowed_gpu(device: str) -> None:
    """GPU 1 is reserved for other work; only cuda:0 and cuda:2 (or cpu) may be used."""
    if device.startswith("cuda"):
        index = int(device.split(":", 1)[1]) if ":" in device else 0
        if index not in ALLOWED_GPUS:
            raise ValueError(f"GPU {index} is not allowed; use one of {ALLOWED_GPUS}")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_write_text(path: str | Path, text: str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, destination)
    return destination


def atomic_json(path: str | Path, value: Any) -> Path:
    return atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n")


def atomic_npz(path: str | Path, arrays: dict[str, np.ndarray]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, destination)
    return destination


def atomic_csv(path: str | Path, frame: pd.DataFrame) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, destination)
    return destination


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------


def load_cohort_sources() -> dict[str, Any]:
    return read_json(CONFIG_DIR / "cohort_sources.json")


def load_registry() -> dict[str, Any]:
    registry = read_json(CONFIG_DIR / "representation_registry.json")
    if int(registry.get("latent_dim", -1)) != LATENT_DIM:
        raise ValueError(f"registry must declare latent_dim={LATENT_DIM}")
    if tuple(registry["representations"]) != REPRESENTATIONS:
        raise ValueError(f"registry must list exactly {REPRESENTATIONS} in that order")
    return registry


def load_sensitivity_registry() -> dict[str, Any]:
    registry = read_json(CONFIG_DIR / "representation_registry_sensitivity.json")
    if int(registry.get("latent_dim", -1)) != LATENT_DIM or tuple(registry["representations"]) != SENSITIVITY_REPRESENTATIONS:
        raise ValueError(f"sensitivity registry must list exactly {SENSITIVITY_REPRESENTATIONS} at latent_dim={LATENT_DIM}")
    return registry


def load_tasks() -> dict[str, Any]:
    return read_json(CONFIG_DIR / "evaluation_tasks.json")


def load_protocol_config() -> dict[str, Any]:
    return read_json(CONFIG_DIR / "protocol_views.json")


def expand_views(config: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Fixed views plus one generated view per cross-fit cohort and fold."""
    config = load_protocol_config() if config is None else config
    views = {name: dict(spec) for name, spec in config["views"].items()}
    crossfit = config.get("crossfit")
    if crossfit:
        folds = int(crossfit["folds"])
        for cohort in crossfit["cohorts"]:
            for fold in range(folds):
                views[f"p2_crossfit_{cohort}_fold{fold}"] = {
                    "protocol": crossfit["protocol"],
                    "description": f"{cohort} cross-fit: fold {fold} test, fold {(fold + 1) % folds} inner validation.",
                    "crossfit": {
                        "cohort": cohort,
                        "fold": fold,
                        "val_fold": (fold + 1) % folds,
                        "folds": folds,
                        "seed": int(crossfit["seed"]),
                    },
                }
    return views


# --------------------------------------------------------------------------------------
# identifiers and labels
# --------------------------------------------------------------------------------------


def qualify(cohort: str, value: Any) -> str:
    """Cohort-qualified id: ADNI and AIBL both use bare numeric RIDs, so bare ids can collide."""
    return f"{cohort}{ID_SEPARATOR}{value}"


def unqualify(value: str) -> str:
    return value.split(ID_SEPARATOR, 1)[1] if ID_SEPARATOR in value else value


def collapse_labels(labels: Iterable[str]) -> list[str]:
    collapsed: list[str] = []
    for label in labels:
        if not collapsed or collapsed[-1] != label:
            collapsed.append(label)
    return collapsed


def trajectory_group(labels: Sequence[str]) -> str:
    """CN-stable, AD-stable, MCI-stable, CN->AD, MCI->AD, CN->MCI, or reverter.

    Labels must be in time order. A monotone path through MCI (CN->MCI->AD) is a CN->AD
    converter; any step back down the CN<MCI<AD order makes the subject a reverter.
    """
    sequence = collapse_labels(labels)
    unknown = set(sequence).difference(LABEL_ORDER)
    if unknown:
        raise ValueError(f"unknown labels {sorted(unknown)}")
    if len(sequence) == 1:
        return f"{sequence[0]}-stable"
    ranks = [LABEL_ORDER[label] for label in sequence]
    if any(later < earlier for earlier, later in zip(ranks, ranks[1:])):
        return "reverter"
    return f"{sequence[0]}->{sequence[-1]}"


def conversion_window(times: Sequence[float], labels: Sequence[str], target: str) -> tuple[float, float]:
    """(last visit before the first label at or beyond ``target``, that first visit).

    NaNs when the conversion was not observed: never reached, or already present at baseline.
    """
    threshold = LABEL_ORDER[target]
    first = next((index for index, label in enumerate(labels) if LABEL_ORDER[label] >= threshold), None)
    if first is None or first == 0:
        return (math.nan, math.nan)
    return (float(times[first - 1]), float(times[first]))


# --------------------------------------------------------------------------------------
# strict manifests
# --------------------------------------------------------------------------------------


def read_strict_manifest(cohort: str, sources: dict[str, Any] | None = None) -> pd.DataFrame:
    """A cohort's strict keep manifest, validated and given canonical columns.

    Added columns: ``cohort``, ``label`` (0 control / 1 disease), ``diagnosis_source``,
    ``subject_key`` and ``scan_key`` (cohort-qualified).
    """
    sources = load_cohort_sources() if sources is None else sources
    spec = sources["cohorts"][cohort]
    topology = sources["topology"]
    frame = pd.read_csv(resolve(spec["strict_manifest"]), dtype={"scan_id": str, "subject_id": str, "VISCODE": str})
    if set(frame["correspondence_topology_hash"].astype(str)) != {topology["correspondence_topology_hash"]}:
        raise ValueError(f"{cohort}: manifest is not in the shared ADNI vertex space")
    if set(frame["vertex_count"].astype(int)) != {int(topology["vertex_count"])}:
        raise ValueError(f"{cohort}: unexpected vertex count")
    if spec.get("restrict_diagnoses"):
        frame = frame.loc[frame["diagnosis"].isin(spec["restrict_diagnoses"])].copy()
    incomplete = frame[["age_years", "visit_month", "correspondence_volume_mm3"]].isna().any(axis=1)
    if incomplete.any():
        # A subject with any visit lacking age, visit time or volume cannot be placed on the models' time
        # axis: a NaN age becomes a NaN transport time and NaN predictions (stage 4, 2026-09-13: CALSNIC
        # CALSNIC2_EDM_C038 has an age only on its unscanned visit 1). The whole subject is excluded.
        dropped = sorted(frame.loc[incomplete, "subject_id"].astype(str).unique())
        frame = frame.loc[~frame["subject_id"].astype(str).isin(dropped)].copy()
        print(f"[{cohort}] excluded {len(dropped)} subject(s) with missing age/visit time/volume: {dropped}", flush=True)
    labels = frame["diagnosis"].map({spec["negative_label"]: 0, spec["positive_label"]: 1})
    if labels.isna().any():
        raise ValueError(f"{cohort}: diagnoses outside {spec['negative_label']}/{spec['positive_label']}: "
                         f"{sorted(frame.loc[labels.isna(), 'diagnosis'].astype(str).unique())}")
    frame["label"] = labels.astype(int)
    frame["cohort"] = cohort
    frame["diagnosis_source"] = frame["diagnosis"].astype(str)
    frame["subject_key"] = [qualify(cohort, value) for value in frame["subject_id"]]
    frame["scan_key"] = [qualify(cohort, value) for value in frame["scan_id"]]
    frame = frame.sort_values(["subject_id", "visit_month", "scan_id"], kind="stable").reset_index(drop=True)
    validate_longitudinal_frame(frame, cohort)
    return frame


def validate_longitudinal_frame(frame: pd.DataFrame, what: str, require_stable_label: bool = True) -> None:
    grouped = frame.groupby("subject_key", sort=False)
    problems = []
    if frame["scan_key"].duplicated().any():
        problems.append("duplicate scans")
    if grouped["split"].nunique().gt(1).any():
        problems.append("subject in more than one split")
    if grouped["scan_key"].nunique().lt(2).any():
        problems.append("subject with fewer than two visits")
    if require_stable_label and grouped["label"].nunique().gt(1).any():
        problems.append("label changes within a subject")
    if any((group["visit_month"].diff().dropna() <= 0).any() for _, group in grouped):
        problems.append("non-increasing visit times")
    if problems:
        raise ValueError(f"{what}: {', '.join(problems)}")


# --------------------------------------------------------------------------------------
# split and fold assignment
# --------------------------------------------------------------------------------------


def split_sizes(total: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    """Same rounding as scripts/prepare_adni_synthseg_separate_structure_cohorts.split_sizes."""
    _train, val_ratio, test_ratio = ratios
    n_val = max(1, int(round(total * val_ratio)))
    n_test = max(1, int(round(total * test_ratio)))
    n_train = total - n_val - n_test
    if total < 3 or n_train < 1:
        raise ValueError(f"cannot split {total} subjects")
    return n_train, n_val, n_test


def stratified_split(strata: dict[str, str], seed: int, ratios: tuple[float, float, float], salt: str) -> dict[str, str]:
    """Seeded, stratified subject split. Strata under three subjects are pooled; a pool still
    under three goes to train (it cannot give every split a subject)."""
    by_stratum: dict[str, list[str]] = {}
    for subject, stratum in strata.items():
        by_stratum.setdefault(stratum, []).append(subject)
    rare = sorted(subject for stratum, ids in by_stratum.items() if len(ids) < 3 for subject in ids)
    groups = {stratum: sorted(ids) for stratum, ids in by_stratum.items() if len(ids) >= 3}
    if rare:
        groups["__rare__"] = rare
    assignment: dict[str, str] = {}
    for stratum in sorted(groups):
        ids = list(groups[stratum])
        random.Random(f"{seed}:{salt}:{stratum}").shuffle(ids)
        if len(ids) < 3:
            assignment.update({subject: "train" for subject in ids})
            continue
        n_train, n_val, _n_test = split_sizes(len(ids), ratios)
        assignment.update({subject: "train" for subject in ids[:n_train]})
        assignment.update({subject: "val" for subject in ids[n_train : n_train + n_val]})
        assignment.update({subject: "test" for subject in ids[n_train + n_val :]})
    return assignment


def visit_count_bin(count: int) -> str:
    return str(count) if count < 5 else "5+"


def crossfit_folds(strata: dict[str, str], folds: int, seed: int) -> dict[str, int]:
    """Stratified round-robin folds; the running offset keeps fold sizes within one subject."""
    assignment: dict[str, int] = {}
    offset = 0
    for stratum in sorted(set(strata.values())):
        ids = sorted(subject for subject, value in strata.items() if value == stratum)
        random.Random(f"{seed}:crossfit:{stratum}").shuffle(ids)
        for index, subject in enumerate(ids):
            assignment[subject] = (offset + index) % folds
        offset += len(ids)
    return assignment


# --------------------------------------------------------------------------------------
# evaluation tasks and baselines
# --------------------------------------------------------------------------------------


def task_prefix(visit_count: int, spec: dict[str, Any]) -> list[int] | None:
    """Local (time-ordered) prefix indices for a task; the target is always the last visit."""
    if visit_count < int(spec["min_visits"]):
        return None
    target = visit_count - 1
    kind = spec["prefix"]
    if kind == "first":
        return [0]
    if kind == "previous":
        return [target - 1]
    if kind == "last_k_before_target":
        k = int(spec["k"]) if "k" in spec else min(target, int(spec["k_max"]))
        if k > target:
            return None
        return list(range(target - k, target))
    raise ValueError(f"unknown prefix rule {kind!r}")


def linear_extrapolation(times: Sequence[float], values: np.ndarray, target_time: float) -> np.ndarray:
    """Least-squares line through (time, value) rows, evaluated at ``target_time``.

    One observation (or identical times) has no slope, so it returns the latest value.
    """
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if len(times) == 1 or np.ptp(times) <= 1.0e-9:
        return values[-1]
    centred = times - times.mean()
    slope = (centred[:, None] * (values - values.mean(axis=0))).sum(axis=0) / np.square(centred).sum()
    return values.mean(axis=0) + slope * (float(target_time) - times.mean())


def population_drift(codes: np.ndarray, subjects: Sequence[str], times: Sequence[float], labels: Sequence[int]) -> dict[int, np.ndarray]:
    """Mean adjacent-visit velocity per label, from training rows ordered by subject then time."""
    subjects = np.asarray(subjects)
    times = np.asarray(times, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    velocities: dict[int, list[np.ndarray]] = {}
    for index in range(len(subjects) - 1):
        if subjects[index] != subjects[index + 1]:
            continue
        delta = times[index + 1] - times[index]
        if delta <= 0:
            raise ValueError("training rows must be time-ordered within each subject")
        velocities.setdefault(int(labels[index]), []).append((codes[index + 1] - codes[index]) / delta)
    return {label: np.mean(np.stack(rows), axis=0) for label, rows in sorted(velocities.items())}


# --------------------------------------------------------------------------------------
# pinned source modules
# --------------------------------------------------------------------------------------


def load_pinned_git_module(
    module_name: str,
    git_blob: str,
    expected_sha256: str,
    search_paths: Sequence[Path],
    nominal_file: Path,
) -> types.ModuleType:
    """Import a module from an exact git blob, verified by content hash.

    Used when a working-tree file has moved on since an archive was built from it: the pinned
    blob is executed in memory, so nothing in the repository is modified or copied.
    ``nominal_file`` is where the file lives in the repository, so module-relative paths
    (``Path(__file__).parent``) resolve exactly as they did when the archive was built.
    """
    source = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "cat-file", "-p", git_blob], capture_output=True, check=True
    ).stdout
    actual = sha256_bytes(source)
    if actual != expected_sha256:
        raise ValueError(f"pinned {module_name} blob {git_blob} hash {actual} != {expected_sha256}")
    for path in search_paths:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    module = types.ModuleType(module_name)
    module.__file__ = str(nominal_file)
    module.__pinned_git_blob__ = git_blob
    sys.modules[module_name] = module
    exec(compile(source, f"{nominal_file}@{git_blob[:12]}", "exec"), module.__dict__)
    return module
