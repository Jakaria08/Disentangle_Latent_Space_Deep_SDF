#!/usr/bin/env python3
"""Stage 5 step 2: evaluate BrainODE-full on p5_converter_pooled (E-C1 rows for the comparator, E-C4).

Every task prediction integrates the P3 BrainODE-core field with RK4, holding the condition fixed within each
substep and re-estimating it before the next substep from the decoded predicted shape (solid voxelization -> the
trained cognition estimator -> sigmoid(logit / temperature)). The subject's label is never used. Voxelization runs in
a spawn process pool; the arithmetic is task3's CognitionFeedbackTransport.

E-C4 (estimator accuracy, BrainODE T3 analog): scan- and subject-level metrics on the split's stable subjects, and the
mean estimated condition at converters' pre- and post-conversion visits (descriptive).
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

import benchmark_common as bc
import converter_line as L
import dynamics_core as D
import evaluate_converter as EC
import evaluate_dynamics as EV
import train_brainode_full as BF
import train_converter_cocycle as TC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=int(BF.SETTINGS["voxel_workers"]))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def pooled_feedback_class(B, PV):
    class PooledFeedbackTransport(B.CognitionFeedbackTransport):
        """task3 CognitionFeedbackTransport with voxelization in a process pool (identical masks and conditions)."""

        def __init__(self, *args, executor, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.executor = executor

        @torch.no_grad()
        def voxel_masks(self, latent: torch.Tensor) -> np.ndarray:
            vertices = self.geometry.vertices(latent).detach().cpu().numpy()
            masks = np.zeros((len(vertices), self.grid.resolution, self.grid.resolution, self.grid.resolution), dtype=np.uint8)
            for index, mask, _quality in self.executor.map(PV._voxelize_worker, ((i, vertices[i]) for i in range(len(vertices))), chunksize=4):
                masks[index] = mask
            return masks

        @torch.no_grad()
        def estimate_condition(self, latent: torch.Tensor) -> torch.Tensor:
            tensor = torch.from_numpy(self.voxel_masks(latent)).to(next(self.estimator.parameters()).device, dtype=torch.float32)
            return torch.sigmoid(self.estimator(tensor) / self.temperature).to(latent.device, dtype=latent.dtype)

    return PooledFeedbackTransport


def main() -> int:
    args = parse_args()
    torch.multiprocessing.set_start_method("spawn", force=True)
    started = time.time()
    B, PV, _TB = BF.task3_modules()
    parts = D.core()
    C = parts["C"]
    device = D.device(args.device)
    checkpoint = args.run_dir / "checkpoints" / "estimator_best.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config, view = payload["config"], payload["view"]
    representation = config["representation"]
    output_dir = bc.require_bulk(args.output_dir or args.run_dir / f"evaluation__{view}" / args.split, "evaluation output")
    if (output_dir / "summary.json").exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {output_dir}")

    registry = D.view_registry(view)
    train = C.load_archive(representation, "train", registry)
    archive = C.load_archive(representation, args.split, registry)
    geometry = C.build_geometry(representation, train, device, registry)
    values = TC.prepared(archive, geometry, device, args.batch_size)
    raw = D.view_vertices(archive)
    ode, _ode_config, _ode_payload = D.load_trained_transport(Path(config["ode_checkpoint"]), device)
    estimator = B.VoxelCognitionCNN(int(config["settings"]["estimator_settings"]["base_channels"]),
                                    float(config["settings"]["estimator_settings"]["dropout"])).to(device)
    estimator.load_state_dict(payload["model_state_dict"], strict=True)
    estimator.eval()
    grid = B.VoxelGrid.from_mapping(config["grid"])
    faces = geometry.faces.cpu().numpy()
    group_of = dict(zip(archive["subject_ids"].astype(str), archive["subject_trajectory_groups"].astype(str)))

    with ProcessPoolExecutor(max_workers=args.workers, initializer=PV._initialize_worker, initargs=(faces, grid.mapping())) as executor:
        feedback = pooled_feedback_class(B, PV)(ode.function, geometry, estimator, grid, int(config["settings"]["feedback_substeps"]),
                                                 float(payload["temperature"]), executor=executor).to(device).eval()
        predict = lambda src, tgt: feedback.transport(values["z"][src], values["age"][src], values["age"][tgt])
        rows = EV.task_rows(feedback, "brainode_full", geometry, values, raw, view, args.split, args.batch_size, predict=predict)
        rows = rows.assign(rule="feedback", group=rows["subject_id"].astype(str).map(group_of))

        groups = L.stable_groups(archive)
        visits = np.arange(len(groups))
        probabilities = np.concatenate([feedback.estimate_condition(values["z"][torch.as_tensor(visits[i:i + 256], device=device)]).cpu().numpy()
                                        for i in range(0, len(visits), 256)])
    stable = groups >= 0
    labels = archive["visit_trajectory_labels"].astype(str)
    converter = np.isin(np.repeat(archive["subject_trajectory_groups"].astype(str), np.diff(archive["subject_visit_offsets"])), L.AD_CONVERTERS)
    estimator_report: dict[str, Any] = {
        "stable": B.scan_and_subject_metrics(archive["visit_label_ad"][stable].astype(np.int64), probabilities[stable], archive["visit_subject_ids"][stable]),
        "converter_mean_condition_before_conversion": float(probabilities[converter & (labels != "AD")].mean()) if (converter & (labels != "AD")).any() else None,
        "converter_mean_condition_after_conversion": float(probabilities[converter & (labels == "AD")].mean()) if (converter & (labels == "AD")).any() else None,
    }
    report = {
        "run_dir": str(args.run_dir), "estimator_checkpoint_sha256": bc.sha256_file(checkpoint), "method": "brainode_full",
        "representation": representation, "seed": config["seed"], "view": view, "split": args.split,
        "tasks": EC.summarize(rows, args.bootstrap), "estimator_E_C4": estimator_report,
        "test_loaded_during_training": bool(payload.get("test_data_loaded", False)), "git_commit": D.git_commit(),
        "seconds": round(time.time() - started, 1),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    bc.atomic_csv(output_dir / "task_rows.csv", rows)
    bc.atomic_json(output_dir / "summary.json", report)
    primary = [r for r in report["tasks"] if r["task"] == "one_shot_first" and r["group"] in ("stable", "AD converters")]
    print(" | ".join(f"{r['group']}: {r['euclidean_mm_mean']:.4f} mm (n={r['subjects']})" for r in primary),
          f"| stable subject AUROC {estimator_report['stable']['subject']['auroc']:.3f}")
    print(f"WROTE {output_dir / 'summary.json'} ({report['seconds']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
