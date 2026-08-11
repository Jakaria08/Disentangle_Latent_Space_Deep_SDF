#!/usr/bin/env python3
"""Train original DeepSDF calibration with readable live terminal progress."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from original_deepsdf_adapter import experiment_dir

import train_large_siren_anchored_cocycle as implementation  # type: ignore  # noqa: E402


def _config_from_cli() -> dict[str, Any]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", required=True)
    args, _ = parser.parse_known_args()
    return json.loads(Path(args.config).read_text())


def main() -> int:
    config = _config_from_cli()
    progress_every = int(config.get("ProgressEveryBatches", 10))
    original_losses = implementation._losses
    state = {"step": 0}

    def losses_with_progress(*args: Any, **kwargs: Any):
        total, terms = original_losses(*args, **kwargs)
        if bool(kwargs.get("training", False)):
            state["step"] += 1
            if progress_every > 0 and (state["step"] == 1 or state["step"] % progress_every == 0):
                print(
                    "progress "
                    f"batch={state['step']:05d} "
                    f"total={float(total.detach().cpu()):.5f} "
                    f"sdf={float(terms['target_sdf'].detach().cpu()):.5f} "
                    f"ad_volume_rate={float(terms['ad_volume_rate'].detach().cpu()):.5f} "
                    f"speed={float(terms['mean_speed'].detach().cpu()):.3f}",
                    flush=True,
                )
        return total, terms

    implementation.experiment_dir = experiment_dir
    implementation._losses = losses_with_progress
    return implementation.main()


if __name__ == "__main__":
    raise SystemExit(main())
