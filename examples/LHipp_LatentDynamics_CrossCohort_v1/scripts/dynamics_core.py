#!/usr/bin/env python3
"""Shared runtime for training and evaluating dynamics models on stage-1 protocol views.

The tested task3 core (``task3_latent_flow_128_v1/scripts``) supplies the archive contracts,
frozen decoders, the direct C4 objective and the ODE vector fields; this module is the only
place that connects that code to the views built in stage 1:

* a *view registry* - the R1 registry with ``output_root``/``source_sequence_root`` pointed at
  one view, so task3's ``load_archive``/``load_pairs`` read that view's archives;
* the real meshes of any view split, in archive order (task3's ``cached_vertices`` only knows
  ADNI);
* the leakage guard, recipe loading, and run directories under the bulk root.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import benchmark_common as bc

T3_SCRIPTS = bc.REPO_ROOT / "examples" / "ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth" / "task3_latent_flow_128_v1" / "scripts"
AUGUST_SCRIPTS = bc.REPO_ROOT / "examples" / "ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth" / "August_Version" / "scripts"
RECIPE_DIR = bc.CONFIG_DIR / "recipes"
RUNS_ROOT = bc.BULK_ROOT / "runs"
VALIDATION_ROOT = bc.BULK_ROOT / "stage2_validation"
METHODS = ("direct_c4", "plain_ode", "brainode", "latent_ode", "latent_ode_residual")
LATENT_ODE_METHODS = ("latent_ode", "latent_ode_residual")
# Stage 5 ablations (PLAN Part 3). A4, the unconditional Latent ODE, is a config override, not a method.
COBOUNDARY_METHODS = ("exact_coboundary_c4", "volume_exact_coboundary_c4_v2")
ABLATION_METHODS = ("direct_c4_no_disease", "brainode_v", *COBOUNDARY_METHODS)
ALL_METHODS = METHODS + ABLATION_METHODS
FLOW_STATE_METHODS = ("direct_c4", "direct_c4_no_disease", "brainode_v", *COBOUNDARY_METHODS)
_AUGUST = bc.REPO_ROOT / "examples" / "ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth" / "August_Version"
COBOUNDARY_SCRIPTS = {"exact_coboundary_c4": _AUGUST / "coboundary_v1" / "scripts",
                      "volume_exact_coboundary_c4_v2": _AUGUST / "coboundary_volume_v2" / "scripts"}
COBOUNDARY_MODULES = {"exact_coboundary_c4": ("coboundary_model", "coboundary_objective", "train_coboundary"),
                      "volume_exact_coboundary_c4_v2": ("volume_coboundary_model", "volume_objective", "train_volume_coboundary")}

_CORE: dict[str, Any] = {}


def core() -> dict[str, Any]:
    """task3 ``common``, ``models``, ``c4_objective`` and ``evaluate``, imported once.

    The pinned LAMM builder and source hashes are verified first (stage 1's prepare_imports).
    Imports are by module name because c4_objective and evaluate import ``common`` by name;
    the resolved files are then checked to really be task3's.
    """
    if _CORE:
        return _CORE
    import stage1_encode_cohort_latents as encoding

    registry = bc.load_registry()
    encoding.prepare_imports(registry)
    if str(T3_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(T3_SCRIPTS))
    import c4_objective
    import common
    import evaluate
    import models

    for module in (common, models, c4_objective, evaluate):
        if Path(module.__file__).resolve().parent != T3_SCRIPTS.resolve():
            raise ImportError(f"{module.__name__} resolved to {module.__file__}, not task3_latent_flow_128_v1")
    _CORE.update({"C": common, "M": models, "O": c4_objective, "E": evaluate, "registry": registry})
    return _CORE


# --------------------------------------------------------------------------------------
# views
# --------------------------------------------------------------------------------------


def view_root(view: str) -> Path:
    root = bc.STAGE1_ROOT / "views" / view
    if not (root / "view_manifest.json").is_file():
        raise FileNotFoundError(f"view {view!r} has not been built: {root}")
    return root


def view_registry(view: str) -> dict[str, Any]:
    registry = copy.deepcopy(bc.load_registry())
    root = str(view_root(view))
    registry.pop("source_integrity", None)  # verified by core(); task3 would re-check the working-tree LAMM builder
    for name, spec in bc.load_sensitivity_registry()["representations"].items():
        if view in spec.get("views", ()):
            registry["representations"][name] = copy.deepcopy(spec)
    registry.update({"output_root": root, "source_sequence_root": root, "view": view})
    return registry


def load_split(view: str, representation: str, split: str) -> dict[str, np.ndarray]:
    C = core()["C"]
    return C.load_archive(representation, split, view_registry(view))


def split_scan_ids(view: str, split: str) -> set[str]:
    archive = bc.load_npz(view_root(view) / "dataset" / f"{split}_subject_sequences.npz")
    return set(archive["visit_scan_ids"].astype(str))


def assert_no_test_leakage(view: str, archives: Iterable[dict[str, np.ndarray]]) -> None:
    """Raise if any scan of the view's test split is present in archives a trainer loaded."""
    test = split_scan_ids(view, "test")
    for archive in archives:
        leaked = test & set(archive["visit_scan_ids"].astype(str))
        if leaked:
            raise RuntimeError(f"test leakage: {len(leaked)} test scans in a training/selection archive, e.g. {sorted(leaked)[:3]}")


class VertexStore:
    """Real correspondence meshes by cohort-qualified scan key, from stage-1 caches."""

    def __init__(self) -> None:
        self._arrays: dict[str, np.ndarray] = {}
        self._index: dict[str, dict[str, int]] = {}

    def _cohort(self, cohort: str) -> None:
        if cohort not in self._arrays:
            self._arrays[cohort] = np.load(bc.STAGE1_ROOT / "vertices" / f"{cohort}_vertices_mm.npy", mmap_mode="r")
            keys = bc.read_json(bc.STAGE1_ROOT / "vertices" / f"{cohort}_vertices_scans.json")["scan_keys"]
            self._index[cohort] = {key: index for index, key in enumerate(keys)}

    def get(self, scan_keys: Iterable[str]) -> np.ndarray:
        rows = []
        for key in scan_keys:
            cohort = str(key).split(bc.ID_SEPARATOR, 1)[0]
            self._cohort(cohort)
            rows.append(self._arrays[cohort][self._index[cohort][str(key)]])
        return np.stack(rows).astype(np.float32)


_STORE = VertexStore()


def view_vertices(archive: dict[str, np.ndarray]) -> np.ndarray:
    """(N, V, 3) real meshes aligned with an archive's visit order."""
    return _STORE.get(archive["visit_scan_ids"].astype(str))


# --------------------------------------------------------------------------------------
# recipes, runs, provenance
# --------------------------------------------------------------------------------------


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(output.get(key), dict):
            output[key] = deep_merge(output[key], value)
        else:
            output[key] = copy.deepcopy(value)
    return output


def set_dotted(config: dict[str, Any], dotted: str, value: Any) -> None:
    node = config
    *parents, leaf = dotted.split(".")
    for key in parents:
        node = node.setdefault(key, {})
    node[leaf] = value


def _recipe_config(method: str, representation: str) -> dict[str, Any]:
    recipe = bc.read_json(RECIPE_DIR / f"{method}.json")
    return deep_merge(recipe["config"], recipe["representation_overrides"].get(representation, {}))


def load_recipe(method: str, representation: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    if method not in ALL_METHODS:
        raise ValueError(f"unknown method {method!r}")
    if representation not in bc.ALL_REPRESENTATIONS:
        raise ValueError(f"unknown representation {representation!r}")
    recipe_representation = representation
    if representation in bc.SENSITIVITY_REPRESENTATIONS:
        recipe_representation = bc.load_sensitivity_registry()["representations"][representation]["recipe_representation"]
    recipe = bc.read_json(RECIPE_DIR / f"{method}.json")
    if "compose" in recipe:
        # Ablation recipes are assembled from the main recipes, so they cannot drift from them.
        compose = recipe["compose"]
        config = _recipe_config(compose["base"], recipe_representation)
        if "model_from" in compose:
            config["model"] = _recipe_config(compose["model_from"], recipe_representation)["model"]
        for donor, keys in compose.get("training_keys_from", {}).items():
            donor_training = _recipe_config(donor, recipe_representation)["training"]
            for key in keys:
                config["training"][key] = donor_training[key]
        config = deep_merge(config, recipe.get("config_patch", {}))
    else:
        config = deep_merge(recipe["config"], recipe["representation_overrides"].get(recipe_representation, {}))
    selected = RECIPE_DIR / f"{method}_selected_overrides.json"
    if selected.is_file():
        # Written by a validation-only search (stage2_select_latent_ode.py); explicit overrides still win.
        for dotted, value in bc.read_json(selected)["overrides"].items():
            set_dotted(config, dotted, value)
    for dotted, value in (overrides or {}).items():
        set_dotted(config, dotted, value)
    config["method"] = method
    config["representation"] = representation
    return config


def run_directory(view: str, representation: str, method: str, run_name: str, root: Path = RUNS_ROOT) -> Path:
    C = core()["C"]
    C.validate_run_name(run_name)
    return bc.require_bulk(root / view / representation / method / run_name, "run directory")


def git_commit() -> str:
    result = subprocess.run(["git", "-C", str(bc.REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(bc.REPO_ROOT), "status", "--porcelain", "--", str(bc.TASK_ROOT)],
                           capture_output=True, text=True).stdout.strip()
    return result.stdout.strip() + ("+uncommitted" if dirty else "")


def device(requested: str):
    import torch

    bc.require_allowed_gpu(requested)
    return core()["C"].choose_device(requested)


def provenance(view: str, representation: str, config: dict[str, Any], seed: int, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    manifest = bc.read_json(view_root(view) / "view_manifest.json")
    rep_manifest = bc.read_json(view_root(view) / "representations" / representation / "manifest.json")
    return {
        "view": view,
        "view_protocol": manifest["spec"].get("protocol"),
        "view_files_sha256": {k: v for k, v in manifest["files"].items() if k.startswith(("dataset/train", "dataset/val", f"representations/{representation}/train", f"representations/{representation}/val"))},
        "representation": representation,
        "representation_manifest": rep_manifest,
        "config": config,
        "seed": int(seed),
        "git_commit": git_commit(),
        "test_data_loaded": False,
        **(extra or {}),
    }


# --------------------------------------------------------------------------------------
# trained transports
# --------------------------------------------------------------------------------------


_NO_DISEASE_FLOW: dict[str, type] = {}


def no_disease_flow_class() -> type:
    """DirectC4Flow whose velocity never sees the condition: Phi = z + (t-s) v_CN (ablation A3)."""
    if "class" not in _NO_DISEASE_FLOW:
        import torch

        class DirectC4NoDiseaseFlow(core()["M"].DirectC4Flow):
            variant = "direct_no_disease"

            def average_velocity(self, latent, source_time, target_time, condition):
                return super().average_velocity(latent, source_time, target_time, torch.zeros_like(condition))

        _NO_DISEASE_FLOW["class"] = DirectC4NoDiseaseFlow
    return _NO_DISEASE_FLOW["class"]


_COBOUNDARY: dict[str, dict[str, Any]] = {}


def coboundary_modules(method: str) -> dict[str, Any]:
    """The pinned August coboundary model, objective and trainer modules for ``method`` (ablation A1).

    They import ``common`` and ``c4_objective`` by name; core() registers task3's (a strict superset of
    August's) first, and the August scripts folder that the modules push onto sys.path is removed again
    so later name-based imports in this process still resolve to task3.
    """
    if method in _COBOUNDARY:
        return _COBOUNDARY[method]
    import importlib

    core()
    for entry in bc.read_json(bc.CONFIG_DIR / "ablation_sources.json")["files"].values():
        if bc.sha256_file(bc.resolve(entry["path"])) != entry["sha256"]:
            raise ValueError(f"pinned ablation source changed: {entry['path']}")
    saved_path = list(sys.path)
    for scripts in COBOUNDARY_SCRIPTS.values():
        if str(scripts) not in sys.path:
            sys.path.append(str(scripts))
    try:
        modules = [importlib.import_module(name) for name in COBOUNDARY_MODULES[method]]
    finally:
        sys.path[:] = saved_path
    for module in modules:
        if Path(module.__file__).resolve().parent != COBOUNDARY_SCRIPTS[method].resolve():
            raise ImportError(f"{module.__name__} resolved to {module.__file__}")
    for shared in ("common", "c4_objective"):
        if Path(sys.modules[shared].__file__).resolve().parent != T3_SCRIPTS.resolve():
            raise ImportError(f"{shared} must be task3's module, got {sys.modules[shared].__file__}")
    _COBOUNDARY[method] = dict(zip(("model", "objective", "trainer"), modules))
    return _COBOUNDARY[method]


def transport_requires_context(method: str) -> bool:
    """Coboundary transports are conditioned on the subject's first visit (code and age)."""
    return method in COBOUNDARY_METHODS


def transport_call(transport, method: str, values: dict[str, Any], source, target):
    """Transport the codes of visits ``source`` to the ages of visits ``target`` (index tensors into ``values``)."""
    arguments = [values["z"][source], values["age"][source], values["age"][target], values["label"][source]]
    if transport_requires_context(method):
        arguments += [values["context"][source], values["context_age"][source]]
    return transport.transport(*arguments)


def build_transport(config: dict[str, Any], torch_device, statistics: dict[str, Any] | None = None):
    """Untrained model for a config, exposing ``transport(z, s, t, d[, context, context_time])``.

    ``statistics`` (a checkpoint's training statistics) is needed only by the volume coboundary,
    whose volume axis is fitted on train codes.
    """
    method = config["method"]
    parts = core()
    model = config["model"]
    if method in ("direct_c4", "direct_c4_no_disease"):
        flow = parts["M"].DirectC4Flow if method == "direct_c4" else no_disease_flow_class()
        return flow(128, int(model["width"]), int(model["residual_blocks"]), float(model.get("dropout", 0.0))).to(torch_device)
    if method in ("plain_ode", "brainode"):
        return parts["M"].build_ode(config).to(torch_device)
    if method == "brainode_v":
        field = parts["M"].build_ode({**config, "method": "brainode"})
        return parts["E"].ODETransport(field, int(config["training"]["integration_substeps"])).to(torch_device)
    if method in LATENT_ODE_METHODS:
        import rubanova_latent_ode as R

        return R.LatentODE.from_config(config).to(torch_device)
    if method == "exact_coboundary_c4":
        return coboundary_modules(method)["model"].build_flow(config).to(torch_device)
    if method == "volume_exact_coboundary_c4_v2":
        if statistics is None or "volume_axis" not in statistics:
            raise ValueError("the volume coboundary needs its train-only volume axis (checkpoint statistics)")
        return coboundary_modules(method)["model"].build_flow(config, statistics["volume_axis"]["coefficient"]).to(torch_device)
    raise ValueError(method)


def load_trained_transport(checkpoint: Path, torch_device):
    """(transport, config) from any run's checkpoint, in eval mode."""
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    method = config["method"]
    model = build_transport(config, torch_device, payload.get("statistics"))
    if method in FLOW_STATE_METHODS:
        model.load_state_dict(payload["flow_state_dict"], strict=True)
        return model.eval(), config, payload
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if method in ("plain_ode", "brainode"):
        wrapped = core()["E"].ODETransport(model, int(config["training"]["integration_substeps"])).to(torch_device)
        return wrapped.eval(), config, payload
    return model.eval(), config, payload


def dumps(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=bc._json_default)
