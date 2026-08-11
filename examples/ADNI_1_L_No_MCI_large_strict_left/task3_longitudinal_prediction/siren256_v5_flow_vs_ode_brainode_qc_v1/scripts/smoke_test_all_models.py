#!/usr/bin/env python3
"""Non-writing shape, RK4-gradient, and frozen-decoder smoke test for all models."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from siren256_common import ensure_prepared, load_basis, load_cache, load_config, load_frozen_decoder, root_dir
from siren256_transport_models import build_transport


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    root, device = root_dir(), torch.device(args.device)
    ensure_prepared(root)
    configs = [root / "configs" / name for name in ("v5_flow_c3.json", "plain_ode_c3_matched.json", "brainode_attention_c3_matched.json", "pca_parity_full_flow_v1.json", "pca_parity_full_flow_v2_pareto.json", "pca_parity_full_flow_v2_geometry_curriculum.json", "pca_parity_full_flow_v2_decoder_whitened.json")]
    basis, cache = load_basis(root), load_cache(load_config(configs[0]))
    source = torch.from_numpy(cache["latents"][:2].copy()).to(device).requires_grad_(True)
    source_time = torch.tensor([[0.2], [0.3]], device=device)
    target_time = torch.tensor([[0.4], [0.5]], device=device)
    condition = torch.tensor([[0.0], [1.0]], device=device)
    for path in configs:
        config = load_config(path)
        model = build_transport(config, basis).to(device)
        prediction = model.transport(source, source_time, target_time, condition)
        identity = model.transport(source, source_time, source_time, condition)
        loss = prediction.square().mean()
        model.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        gradient = max((float(parameter.grad.abs().max().cpu()) for parameter in model.parameters() if parameter.grad is not None), default=0.0)
        if prediction.shape != source.shape or not torch.isfinite(prediction).all() or not torch.allclose(identity, source):
            raise RuntimeError(f"Transport smoke failure: {config['ModelType']}")
        print(f"{config['ModelType']}: shape={tuple(prediction.shape)} max_parameter_grad={gradient:.3g} attention={getattr(model, 'attention_contract', 'n/a')}")
    decoder = load_frozen_decoder(load_config(configs[0]), device)
    decoded = decoder(torch.cat((source[:1], torch.zeros(1, 3, device=device)), dim=1))
    if not torch.isfinite(decoded).all():
        raise RuntimeError("Frozen decoder produced a non-finite value.")
    print("all matched models and frozen decoder passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
