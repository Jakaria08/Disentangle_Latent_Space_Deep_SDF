#!/usr/bin/env python3
"""Stage 1b: encode every cohort scan with the frozen ADNI representations (R1).

For each cohort the scan set is the union of its strict manifest and, where one exists, its
inclusive manifest. Vertices are read once from the correspondence PLYs and cached under the
bulk root. Each representation then produces 128-D raw codes plus per-scan reconstruction
errors (decoded code vs the real mesh).

ADNI is special: its codes are *copied* from the archives the existing longitudinal results
were trained on (August_Version for PCA/SpiralNet/Adaptive, task3_v3_lamm_latest for LAMM),
so those results remain reproducible bit for bit. ADNI is still re-encoded here, and the
difference between the copied and recomputed codes is the check that this encoding path -
the one used for AIBL, OASIS and CALSNIC - is correct.

Run with the pytorch_geo environment on GPU 0 or 2 only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import benchmark_common as bc

T3_SCRIPTS = bc.REPO_ROOT / "examples" / "ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth" / "task3_latent_flow_128_v1" / "scripts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cohorts", nargs="+", default=["adni", "aibl", "oasis", "calsnic"])
    parser.add_argument("--representations", nargs="+", default=list(bc.REPRESENTATIONS))
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true", help="Encode 8 scans per cohort and write nothing.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


# --------------------------------------------------------------------------------------
# scans and vertices
# --------------------------------------------------------------------------------------


def cohort_scans(cohort: str, sources: dict[str, Any]) -> pd.DataFrame:
    strict = bc.read_strict_manifest(cohort, sources)
    columns = ["cohort", "scan_key", "subject_key", "scan_id", "mesh_path_mm", "correspondence_volume_mm3"]
    frames = [strict.loc[:, columns].assign(in_strict=True)]
    inclusive_path = bc.STAGE1_ROOT / "cohorts" / cohort / "inclusive_manifest.csv"
    if sources["cohorts"][cohort].get("inclusive_qc_root"):
        if not inclusive_path.is_file():
            raise FileNotFoundError(f"{cohort}: run stage1_build_inclusive_cohorts.py first ({inclusive_path})")
        inclusive = pd.read_csv(inclusive_path, dtype={"scan_id": str, "subject_id": str})
        frames.append(inclusive.loc[:, columns].assign(in_strict=False))
    scans = pd.concat(frames, ignore_index=True)
    scans = scans.sort_values(["scan_key", "in_strict"], ascending=[True, False]).drop_duplicates("scan_key", keep="first")
    if scans.groupby("scan_key")["mesh_path_mm"].nunique().gt(1).any():
        raise ValueError(f"{cohort}: one scan maps to two mesh paths")
    return scans.sort_values("scan_key", kind="stable").reset_index(drop=True)


def load_vertices(cohort: str, scans: pd.DataFrame, vertex_count: int, dry_run: bool) -> np.ndarray:
    import trimesh

    cache = bc.require_bulk(bc.STAGE1_ROOT / "vertices" / f"{cohort}_vertices_mm.npy")
    sidecar = cache.with_name(f"{cohort}_vertices_scans.json")
    keys = scans["scan_key"].tolist()
    if not dry_run and cache.is_file() and sidecar.is_file() and bc.read_json(sidecar)["scan_keys"] == keys:
        return np.load(cache, mmap_mode="r")
    stack = np.empty((len(scans), vertex_count, 3), dtype=np.float32)
    for index, path in enumerate(scans["mesh_path_mm"]):
        vertices = np.asarray(trimesh.load(path, process=False).vertices, dtype=np.float32)
        if vertices.shape != (vertex_count, 3):
            raise ValueError(f"{path}: {vertices.shape} vertices, expected ({vertex_count}, 3)")
        stack[index] = vertices
    if not dry_run:
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_name(f".{cache.stem}.tmp.npy")
        np.save(temporary, stack)
        temporary.replace(cache)
        bc.atomic_json(sidecar, {"cohort": cohort, "scan_keys": keys, "sha256": bc.sha256_file(cache)})
    return stack


# --------------------------------------------------------------------------------------
# representations
# --------------------------------------------------------------------------------------


class Encoder:
    """Frozen R1 representation: raw codes in, millimetre vertices out."""

    def __init__(self, name: str, registry: dict[str, Any], device: torch.device) -> None:
        self.name = name
        self.spec = registry["representations"][name]
        self.device = device
        self.hashes: dict[str, str] = {}
        if self.spec["kind"] == "pca":
            root = bc.resolve(self.spec["pca_model_root"])
            self.mean = np.load(root / "mean.npy").astype(np.float64).reshape(-1)
            self.components = np.load(root / self.spec["components_file"]).astype(np.float64)[: bc.LATENT_DIM]
            for file in ("mean.npy", self.spec["components_file"], "faces.npy"):
                self.hashes[file] = bc.sha256_file(root / file)
            return
        t3 = task3_common()
        checkpoint = Path(self.spec["checkpoint"])
        self.hashes["checkpoint"] = bc.sha256_file(checkpoint)
        if self.hashes["checkpoint"] != self.spec["checkpoint_sha256"]:
            raise ValueError(f"{name}: checkpoint hash mismatch")
        self.model, _payload = t3.load_ae_model(name, device, t3_registry(registry))
        mesh_mean, mesh_std = t3.ae_normalization(t3_registry(registry))
        self.mesh_mean = torch.from_numpy(mesh_mean).to(device)
        self.mesh_std = torch.from_numpy(mesh_std).to(device)

    @torch.no_grad()
    def encode(self, vertices: np.ndarray, batch_size: int) -> np.ndarray:
        if self.spec["kind"] == "pca":
            flat = np.asarray(vertices, dtype=np.float64).reshape(len(vertices), -1)
            return ((flat - self.mean) @ self.components.T).astype(np.float32)
        outputs = []
        for start in range(0, len(vertices), batch_size):
            batch = torch.from_numpy(np.asarray(vertices[start : start + batch_size], dtype=np.float32).copy()).to(self.device)
            outputs.append(self.model.encode((batch - self.mesh_mean) / self.mesh_std).float().cpu().numpy())
        return np.concatenate(outputs).astype(np.float32)

    @torch.no_grad()
    def decode(self, codes: np.ndarray, batch_size: int) -> np.ndarray:
        if self.spec["kind"] == "pca":
            flat = np.asarray(codes, dtype=np.float64) @ self.components + self.mean
            return flat.reshape(len(codes), -1, 3).astype(np.float32)
        outputs = []
        for start in range(0, len(codes), batch_size):
            batch = torch.from_numpy(np.asarray(codes[start : start + batch_size], dtype=np.float32)).to(self.device)
            outputs.append((self.model.decode(batch) * self.mesh_std + self.mesh_mean).float().cpu().numpy())
        return np.concatenate(outputs)


_T3_REGISTRY: dict[str, Any] | None = None
_T3_COMMON = None


def task3_common():
    """task3_latent_flow_128_v1/scripts/common.py, imported by path under a unique name.

    A bare ``import common`` would resolve to whichever ``common.py`` comes first on sys.path,
    and the AE/LAMM script folders are also on sys.path.
    """
    global _T3_COMMON
    if _T3_COMMON is None:
        import importlib.util

        spec = importlib.util.spec_from_file_location("task3_latent_flow_common", T3_SCRIPTS / "common.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["task3_latent_flow_common"] = module
        spec.loader.exec_module(module)
        _T3_COMMON = module
    return _T3_COMMON


def t3_registry(registry: dict[str, Any]) -> dict[str, Any]:
    """The subset of R1 that task3's loader functions read; integrity is verified here instead."""
    global _T3_REGISTRY
    if _T3_REGISTRY is None:
        _T3_REGISTRY = {
            "latent_dim": bc.LATENT_DIM,
            "ae_source_root": registry["ae_source_root"],
            "ae_bulk_root": registry["ae_bulk_root"],
            "lamm_source_root": registry["lamm_source_root"],
            "faces_path": registry["faces_path"],
            "representations": registry["representations"],
        }
    return _T3_REGISTRY


def prepare_imports(registry: dict[str, Any]) -> dict[str, str]:
    """Verify pinned sources, then make task3 common and the pinned LAMM builder importable."""
    integrity = registry["source_integrity"]
    checks = {
        "faces": (Path(registry["faces_path"]), integrity["faces_sha256"]),
        "lamm_hierarchy": (Path(integrity["lamm_hierarchy"]), integrity["lamm_hierarchy_sha256"]),
        "lamm_model_source": (bc.resolve(integrity["lamm_model_source"]), integrity["lamm_model_source_sha256"]),
    }
    verified = {}
    for label, (path, expected) in checks.items():
        actual = bc.sha256_file(path)
        if actual != expected:
            raise ValueError(f"{label} hash mismatch at {path}: {actual} != {expected}")
        verified[label] = actual
    pinned = registry["pinned_sources"]["train_lamm"]
    nominal = bc.resolve(pinned["nominal_file"])
    bc.load_pinned_git_module(
        "train_lamm", pinned["git_blob"], pinned["sha256"],
        [bc.resolve(registry["ae_source_root"]) / "scripts", nominal.parent], nominal,
    )
    verified["train_lamm_pinned_blob"] = pinned["git_blob"]
    return verified


def reference_codes(name: str, registry: dict[str, Any], scan_ids: list[str]) -> np.ndarray:
    """ADNI raw codes exactly as stored in the archives the anchors were trained on."""
    directory = Path(registry["representations"][name]["adni_reference_archive_dir"])
    lookup: dict[str, np.ndarray] = {}
    for split in bc.SPLITS:
        archive = bc.load_npz(directory / f"{split}_subject_sequences_128.npz")
        for scan, code in zip(archive["visit_scan_ids"].astype(str), archive["visit_latent_raw_128"]):
            lookup[scan] = code
    missing = [scan for scan in scan_ids if scan not in lookup]
    if missing:
        raise KeyError(f"{name}: {len(missing)} ADNI scans missing from reference archives, e.g. {missing[:3]}")
    return np.stack([lookup[scan] for scan in scan_ids]).astype(np.float32)


def reconstruction_errors(decoded: np.ndarray, vertices: np.ndarray) -> dict[str, np.ndarray]:
    delta = decoded.astype(np.float64) - np.asarray(vertices, dtype=np.float64)
    return {
        "recon_coordinate_rmse_mm": np.sqrt(np.mean(delta**2, axis=(1, 2))).astype(np.float32),
        "recon_coordinate_mae_mm": np.mean(np.abs(delta), axis=(1, 2)).astype(np.float32),
        "recon_vertex_euclidean_mm": np.linalg.norm(delta, axis=2).mean(axis=1).astype(np.float32),
    }


def main() -> int:
    args = parse_args()
    bc.require_allowed_gpu(args.device)
    device = torch.device(args.device)
    sources = bc.load_cohort_sources()
    registry = bc.load_registry()
    verified = prepare_imports(registry)
    vertex_count = int(sources["topology"]["vertex_count"])
    encoders = {name: Encoder(name, registry, device) for name in args.representations}
    summary: dict[str, Any] = {"verified_sources": verified, "device": args.device, "cohorts": {}}

    summary_path = bc.STAGE1_ROOT / "latents" / "encoding_summary.json"
    if summary_path.is_file() and not args.dry_run:
        previous = bc.read_json(summary_path)
        summary["cohorts"] = previous.get("cohorts", {})
    for cohort in args.cohorts:
        started = time.time()
        report_path = bc.STAGE1_ROOT / "latents" / cohort / "encoding_report.json"
        todo = {
            name: encoder for name, encoder in encoders.items()
            if args.dry_run or args.overwrite or not (bc.STAGE1_ROOT / "latents" / cohort / f"{name}.npz").is_file()
        }
        if not todo:
            print(f"[{cohort}] every representation already encoded, skipping (pass --overwrite to redo)", flush=True)
            continue
        scans = cohort_scans(cohort, sources)
        if args.dry_run:
            scans = scans.head(8)
        vertices = load_vertices(cohort, scans, vertex_count, args.dry_run)
        use_reference = bool(sources["cohorts"][cohort].get("use_reference_archives"))
        strict_split = bc.read_strict_manifest(cohort, sources).set_index("scan_key")["split"]
        split_of = scans["scan_key"].map(strict_split).fillna("inclusive_only").to_numpy()
        cohort_report: dict[str, Any] = bc.read_json(report_path) if report_path.is_file() and not args.dry_run else {}
        cohort_report.update({"scans": int(len(scans)), "vertex_cache_seconds": round(time.time() - started, 1)})
        for name, encoder in todo.items():
            destination = bc.require_bulk(bc.STAGE1_ROOT / "latents" / cohort / f"{name}.npz")
            recomputed = encoder.encode(vertices, args.batch_size)
            codes = reference_codes(name, registry, scans["scan_id"].tolist()) if use_reference else recomputed
            if not np.isfinite(codes).all() or codes.shape != (len(scans), bc.LATENT_DIM):
                raise RuntimeError(f"{cohort}/{name}: invalid codes {codes.shape}")
            errors = reconstruction_errors(encoder.decode(codes, args.batch_size), vertices)
            difference = np.abs(codes - recomputed)
            report = {
                "code_source": "adni_reference_archive" if use_reference else "recomputed_r1",
                "recomputed_vs_stored_max_abs": float(difference.max()),
                "recomputed_vs_stored_max_rel_to_code_std": float((difference / np.maximum(codes.std(axis=0), 1e-12)).max()),
                "hashes": encoder.hashes,
                "recon_coordinate_rmse_mm_by_split": {
                    split: float(errors["recon_coordinate_rmse_mm"][split_of == split].mean())
                    for split in sorted(set(split_of))
                },
            }
            cohort_report[name] = report
            print(f"[{cohort}/{name}] codes={report['code_source']} max|stored-recomputed|={report['recomputed_vs_stored_max_abs']:.2e} "
                  f"recon RMSE by split {json.dumps({k: round(v, 5) for k, v in report['recon_coordinate_rmse_mm_by_split'].items()})}",
                  flush=True)
            if args.dry_run:
                continue
            bc.atomic_npz(destination, {
                "scan_keys": scans["scan_key"].to_numpy().astype(str),
                "codes": codes,
                "codes_recomputed": recomputed,
                **errors,
                "representation": np.asarray(name),
            })
            report["archive"] = str(destination)
        cohort_report["seconds"] = round(time.time() - started, 1)
        summary["cohorts"][cohort] = cohort_report
        if not args.dry_run:
            bc.atomic_json(report_path, cohort_report)
            bc.atomic_json(summary_path, summary)
    print(json.dumps({"dry_run": args.dry_run, "cohorts": {c: r["scans"] for c, r in summary["cohorts"].items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
