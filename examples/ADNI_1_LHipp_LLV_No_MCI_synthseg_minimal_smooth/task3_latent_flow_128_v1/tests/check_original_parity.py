#!/usr/bin/env python3
"""Check the new direct-C4 weights/selection against the original PCA C4."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


TASK = Path(__file__).resolve().parents[1]
PROJECT = TASK.parents[2]
ORIGINAL = PROJECT / "examples" / "ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth" / "hippocampus_pca_cocycle_v4" / "cocycle_v5" / "configs" / "c4_cocycle_v5.json"
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(TASK / "scripts"))

from models import DirectC4Flow  # noqa: E402
from train_adni_synthseg_pca_cocycle_v5 import DirectDiagnosisResidualCocycleFlow  # noqa: E402


def main() -> int:
    original = json.loads(ORIGINAL.read_text())
    mapping = {
        "real_pca_weight": "real_latent_weight",
        "real_vertex_weight": "real_vertex_weight",
        "observed_semigroup_weight": "observed_semigroup_weight",
        "virtual_semigroup_weight": "virtual_semigroup_weight",
        "inverse_weight": "inverse_weight",
        "sequence_pca_weight": "sequence_latent_weight",
        "sequence_vertex_weight": "sequence_vertex_weight",
        "sequence_semigroup_weight": "sequence_semigroup_weight",
        "volume_weight": "volume_weight",
        "rate_weight": "rate_weight",
        "slope_weight": "slope_weight",
        "group_rate_weight": "group_rate_weight",
        "disease_gap_weight": "disease_gap_weight",
    }
    for path in sorted((TASK / "configs").glob("*_direct_c4_s42.json")):
        current = json.loads(path.read_text())
        assert current["model"]["width"] == original["model"]["width"]
        assert current["model"]["residual_blocks"] == original["model"]["residual_blocks"]
        assert current["model"]["variant"] == "direct"
        assert current["loss"]["coboundary_weight"] == 0.0
        for old, new in mapping.items():
            assert current["loss"][new] == original["loss"][old], (path, old, new)
        assert current["selection"]["latent_nochange_tolerance"] == original["selection"]["pca_nochange_tolerance"]
        for key in ("all_pair_shape_weight", "volume_tiebreak_weight", "coordinate_nochange_tolerance", "max_relative_semigroup_defect", "max_relative_inverse_defect"):
            assert current["selection"][key] == original["selection"][key], (path, key)

    torch.manual_seed(2026)
    reference = DirectDiagnosisResidualCocycleFlow(128, 256, 2).eval()
    current = DirectC4Flow(128, 256, 2, 0.0).eval()
    current.load_state_dict(reference.state_dict(), strict=True)
    latent = torch.randn(7, 128)
    source = torch.randn(7)
    target = source + torch.randn(7)
    label = torch.randint(0, 2, (7,), dtype=torch.float32)
    torch.testing.assert_close(
        current.transport(latent, source, target, label),
        reference.transport(latent, source, target, label),
        rtol=0.0,
        atol=0.0,
    )
    print("ORIGINAL C4 PARITY PASSED: exact direct-flow function, 13 active weights, and selection thresholds match; coboundary=0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
