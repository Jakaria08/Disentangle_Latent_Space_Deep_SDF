#!/usr/bin/env python3
"""Perform no-write static validation of CALSNIC task code and configurations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


TASK_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = TASK_DIR.parents[2]
CONFIGS = (
    TASK_DIR / "configs" / "calsnic_control_L_multires64_z256_exact.json",
    TASK_DIR / "configs" / "calsnic_control_L_multires128_z256_exact.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-generated-data", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sampling = REPO_ROOT / "examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task2_inr_representations_v1/scripts/shared_grid_common.py"
    sampling_hash = hashlib.sha256(sampling.read_bytes()).hexdigest()
    reports = []
    for path in CONFIGS:
        config = json.loads(path.read_text())
        output = Path(config["output_dir"]).resolve()
        if not output.is_relative_to(Path("/mnt/bulk10tb")):
            raise ValueError(f"Non-bulk output in {path}: {output}")
        if config["reference_sampling_sha256"] != sampling_hash:
            raise RuntimeError(
                f"Pinned sampling hash is stale in {path.name}: expected current {sampling_hash}"
            )
        resolutions = config["network_specs"]["grid_resolutions"]
        scheduled = [item["resolution"] for item in config["level_schedule"]]
        if resolutions != sorted(set(resolutions)) or set(resolutions) != set(scheduled):
            raise ValueError(f"Grid/schedule mismatch in {path}")
        if config["network_specs"]["grid_resolution"] != max(resolutions):
            raise ValueError(f"Sampling grid does not match maximum level in {path}")
        evaluator = REPO_ROOT / config["periodic_evaluation"]["script"]
        if not evaluator.is_file():
            raise FileNotFoundError(evaluator)
        if config["periodic_evaluation"]["splits"] != ["val"]:
            raise ValueError("Periodic evaluation must remain validation-only.")
        if args.require_generated_data:
            for field in ("manifest", "required_exact_audit"):
                if not Path(config[field]).is_file():
                    raise FileNotFoundError(Path(config[field]))
            pca = Path(config["periodic_evaluation"]["pca_model_dir"])
            for name in ("metadata.json", "mean_vertices.npy", "components.npy", "faces.npy"):
                if not (pca / name).is_file():
                    raise FileNotFoundError(pca / name)
        parameters = sum(int(value) ** 3 for value in resolutions) * int(config["network_specs"]["grid_channels_per_level"])
        reports.append(
            {
                "config": path.name,
                "levels": resolutions,
                "grid_parameters": parameters,
                "grid_fp32_mib": parameters * 4 / 1024**2,
                "output": str(output),
            }
        )
    multires_common = REPO_ROOT / "examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task2_inr_multires_single_field_v1/scripts/multires_common.py"
    shared_validator = multires_common.read_text()
    if "mesh_center_x" not in shared_validator:
        raise RuntimeError("Shared manifest validator lacks optional CALSNIC centring support.")
    trainer = REPO_ROOT / "examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task2_inr_multires_single_field_v1/scripts/train_multires_sdf.py"
    if "configured_script" not in trainer.read_text():
        raise RuntimeError("Shared trainer lacks the configured periodic-evaluator hook.")
    print(json.dumps({"passed": True, "no_files_written": True, "configs": reports}, indent=2))


if __name__ == "__main__":
    main()
