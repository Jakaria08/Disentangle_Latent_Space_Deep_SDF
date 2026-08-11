#!/usr/bin/env python3
"""Verify that BrainODE predictions cannot depend on unrelated batch companions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from siren256_common import ensure_prepared, load_basis, load_cache, read_json, root_dir, write_json
from siren256_transport_models import BrainODEAttention, build_transport


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    root, run, device = root_dir(), args.run.resolve(), torch.device(args.device)
    ensure_prepared(root)
    config = read_json(run / "config.json")
    if config["ModelType"] != "brainode_attention_ode":
        raise ValueError("Batch-context audit applies only to the BrainODE attention run.")
    model = build_transport(config, load_basis(root)).to(device)
    if not isinstance(model, BrainODEAttention):
        raise TypeError("Configured model is not BrainODEAttention.")
    checkpoint = run / "checkpoints" / "best.pt"
    if not checkpoint.exists():
        checkpoint = run / "checkpoints" / "best_candidate.pt"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu")["flow_state_dict"], strict=True)
    model.eval()
    cache = load_cache(config)
    latent = torch.from_numpy(np.array(cache["latents"][:8], copy=True)).to(device)
    time = torch.from_numpy(np.array(cache["times"][:8], copy=True)).to(device)[:, None]
    condition = torch.from_numpy(np.array(cache["labels"][:8], copy=True)).to(device=device, dtype=torch.float32)[:, None]
    with torch.no_grad():
        alone = model.velocity(latent[:1], time[:1], condition[:1])
        with_companions = model.velocity(latent, time, condition)[:1]
        permuted = model.velocity(torch.cat((latent[:1], latent[torch.tensor([7, 6, 5, 4, 3, 2, 1], device=device)])), torch.cat((time[:1], time[torch.tensor([7, 6, 5, 4, 3, 2, 1], device=device)])), torch.cat((condition[:1], condition[torch.tensor([7, 6, 5, 4, 3, 2, 1], device=device)])))[:1]
    difference = max(float((alone - with_companions).abs().max().cpu()), float((alone - permuted).abs().max().cpu()))
    result = {"attention_contract": model.attention_contract, "source_velocity_max_abs_difference_with_companions": float((alone - with_companions).abs().max().cpu()), "source_velocity_max_abs_difference_after_companion_permutation": float((alone - permuted).abs().max().cpu()), "singleton_attention_weight": 1.0, "context_invariant_within_fp32_tolerance": bool(difference <= 1.0e-7), "interpretation": "A difference at or below 1e-7 is required: each Q/K/V calculation is a singleton and cannot encode cross-subject biological interaction."}
    write_json(run / "evaluation" / "brainode_batch_context_audit.json", result)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
