#!/usr/bin/env python3
"""Validation/test evaluator for volume-aware exact-coboundary V2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


AUGUST_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
V1_SCRIPTS = Path(__file__).resolve().parents[2] / "coboundary_v1" / "scripts"
for path in (V1_SCRIPTS, Path(__file__).resolve().parent, AUGUST_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import common as C  # noqa: E402
import evaluate_coboundary as E  # noqa: E402
import volume_objective as O  # noqa: E402
from volume_coboundary_model import build_flow  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--evaluation-name", default="evaluation")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    C.validate_run_name(args.evaluation_name)
    run_dir = args.run_dir.expanduser().resolve()
    resolved = C.read_json(run_dir / "resolved_config.json")
    config = resolved["config"]
    if config.get("method") != "volume_exact_coboundary_c4_v2":
        raise ValueError("Evaluator accepts only volume_exact_coboundary_c4_v2")
    representation = str(config["representation"])
    checkpoint_path = args.checkpoint.expanduser().resolve() if args.checkpoint else run_dir / "checkpoints" / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    for key in ("coboundary_used", "structural_exactness", "train_only_volume_axis"):
        if not bool(checkpoint.get(key)):
            raise ValueError(f"Checkpoint contract is missing {key}")
    device = C.choose_device(args.device)
    registry = C.load_registry()
    train_archive = C.load_archive(representation, "train", registry)
    archive = C.load_archive(representation, args.split, registry)
    geometry = C.build_geometry(representation, train_archive, device, registry)
    values = C.values_on_device(archive, device)
    O.attach_reference_geometry(values, geometry, int(config["training"].get("decoder_batch_size", 64)))
    raw_vertices = C.cached_vertices(args.split, registry)
    volume_axis = checkpoint["statistics"]["volume_axis"]
    transport = build_flow(config, volume_axis["coefficient"]).to(device)
    transport.load_state_dict(checkpoint["flow_state_dict"], strict=True)
    transport.eval()
    forward = E.limited_pairs(C.load_pairs(args.split, archive, registry), args.max_pairs)
    first_last = E.limited_pairs(C.first_last_pairs(archive), args.max_pairs, first_last=True)
    backward, first_last_backward = E.reverse_rows(forward), E.reverse_rows(first_last)
    batch_size = int(config["training"].get("evaluation_batch_size", config["training"].get("batch_size", 64)))
    categories = {
        "all_forward": forward,
        "adjacent_forward": [row for row in forward if row.pair_type == "adjacent"],
        "nonadjacent_forward": [row for row in forward if row.pair_type == "nonadjacent"],
        "first_last_forward": first_last,
        "all_backward": backward,
        "adjacent_backward": [row for row in backward if row.pair_type.endswith("adjacent") and "nonadjacent" not in row.pair_type],
        "nonadjacent_backward": [row for row in backward if row.pair_type.endswith("nonadjacent")],
        "first_last_backward": first_last_backward,
    }
    categories.update({f"horizon_{name}_forward": rows for name, rows in E.horizon_groups(forward).items() if rows})
    categories.update({f"horizon_{name}_backward": rows for name, rows in E.horizon_groups(backward).items() if rows})
    pair_metrics = {
        name: O.evaluate_pairs(
            transport, geometry, values, rows, raw_vertices, batch_size, include_rows=(name == "all_forward")
        )
        for name, rows in categories.items() if rows
    }
    defects = O.cocycle_defects(transport, values, forward, checkpoint["statistics"], batch_size)
    maximum_defect = max(defects.values())
    threshold = float(config["selection"]["implementation_audit_max_relative_defect"])
    if maximum_defect > threshold:
        raise RuntimeError(f"Exact algebra audit failed: {maximum_defect:.3e} > {threshold:.3e}")
    sequences = E.sequence_metrics(transport, geometry, values, archive, raw_vertices, args.max_subjects)
    floor = E.representation_floor(values, raw_vertices, geometry, batch_size)
    bootstrap = E.subject_bootstrap(pair_metrics["all_forward"].get("row_metrics", []), args.bootstrap_samples)
    first_last_groups = pair_metrics["first_last_forward"]["groups"]
    trend = {
        diagnosis: {
            "predicted_signed_log_volume_rate_per_year": first_last_groups[diagnosis]["predicted_signed_rate_mean"],
            "observed_signed_log_volume_rate_per_year": first_last_groups[diagnosis]["observed_signed_rate_mean"],
            "predicted_percent_change_per_year": 100.0 * np.expm1(first_last_groups[diagnosis]["predicted_signed_rate_mean"]),
            "observed_percent_change_per_year": 100.0 * np.expm1(first_last_groups[diagnosis]["observed_signed_rate_mean"]),
        }
        for diagnosis in ("CN", "AD")
    }
    report = {
        "run_dir": str(run_dir), "evaluation_name": args.evaluation_name,
        "checkpoint": str(checkpoint_path), "checkpoint_epoch": int(checkpoint["epoch"]),
        "split": args.split, "representation": representation, "method": config["method"],
        "volume_axis": {key: value for key, value in volume_axis.items() if key != "coefficient"},
        "cn_ad_volume_trend": trend, "representation_floor": floor,
        "pair_metrics": pair_metrics, "sequence_metrics": sequences,
        "consistency_defects": defects, "subject_bootstrap": bootstrap,
        "structural_audit_passed": True,
        "test_loaded_during_training": bool(checkpoint.get("test_data_loaded", False)),
        "source_meshes_modified": False,
    }
    C.assert_finite_mapping({key: value for key, value in report.items() if key not in {"pair_metrics", "sequence_metrics"}})
    if args.dry_run:
        print("V2 EVALUATION DRY RUN PASSED — no files written.")
        print(json.dumps({key: value for key, value in report.items() if key != "pair_metrics"}, indent=2, sort_keys=True))
        return 0
    destination = run_dir / args.evaluation_name / args.split
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    C.atomic_json(destination / "summary.json", report)
    print(f"WROTE {destination / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

