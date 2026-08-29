#!/usr/bin/env python3
"""Fast implementation/config contract tests; no result files are written."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from coboundary_model import ExactCouplingCoboundaryFlow, build_flow  # noqa: E402
from train_coboundary import validate_config  # noqa: E402


def maximum_rmse(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((left - right).square(), dim=1)).max())


def main() -> int:
    torch.manual_seed(9137)
    configs = []
    for path in sorted((ROOT / "configs").glob("*_exact_coboundary_c4_s42.json")):
        config = json.loads(path.read_text(encoding="utf-8"))
        validate_config(config)
        configs.append(config)
    if {config["representation"] for config in configs} != {"pca128", "spiralnet128", "adaptive128"}:
        raise AssertionError("Expected exactly three representation configs")
    comparable = [json.loads(json.dumps(config)) for config in configs]
    for config in comparable:
        config.pop("name")
        config.pop("representation")
        config["training"].pop("run_name")
        for batch_key in ("batch_size", "decoder_batch_size", "evaluation_batch_size"):
            config["training"].pop(batch_key)
    if not all(config == comparable[0] for config in comparable[1:]):
        raise AssertionError("Representation configs differ beyond names and memory-driven batch sizes")

    flow: ExactCouplingCoboundaryFlow = build_flow(configs[0])
    # Move away from identity initialization so the test exercises real affine inverses.
    with torch.no_grad():
        for layer in flow.layers:
            torch.nn.init.normal_(layer.conditioner.cn_head.weight, std=0.01)
            torch.nn.init.normal_(layer.conditioner.cn_head.bias, std=0.01)
            torch.nn.init.normal_(layer.conditioner.ad_residual_head.weight, std=0.01)
            torch.nn.init.normal_(layer.conditioner.ad_residual_head.bias, std=0.01)
    flow.eval()
    batch, dim = 11, 128
    z = torch.randn(batch, dim)
    context = torch.randn(batch, dim)
    context_age = torch.randn(batch) * 0.2
    source = context_age + torch.rand(batch) * 0.7
    middle = source + torch.rand(batch) * 0.8
    target = middle + torch.rand(batch) * 0.9
    label = torch.randint(0, 2, (batch,), dtype=torch.float32)
    direct = flow.transport(z, source, target, label, context, context_age)
    first_leg = flow.transport(z, source, middle, label, context, context_age)
    composed = flow.transport(first_leg, middle, target, label, context, context_age)
    recovered = flow.transport(direct, target, source, label, context, context_age)
    identity = flow.transport(z, source, source, label, context, context_age)
    defects = {
        "composition_rmse_max": maximum_rmse(direct, composed),
        "inverse_rmse_max": maximum_rmse(z, recovered),
        "identity_rmse_max": maximum_rmse(z, identity),
    }
    if max(defects.values()) > 3.0e-6:
        raise AssertionError(f"Exact algebra defect too large: {defects}")
    loss = direct.square().mean() + composed.abs().mean()
    loss.backward()
    gradient = sum(float(parameter.grad.abs().sum()) for parameter in flow.parameters() if parameter.grad is not None)
    if not torch.isfinite(loss) or gradient <= 0.0:
        raise AssertionError("Gradient audit failed")
    if any(isinstance(module, torch.nn.Dropout) for module in flow.modules()):
        raise AssertionError("Dropout found inside exact invertible flow")
    print(json.dumps({
        "status": "passed", "configs": [config["name"] for config in configs],
        "parameters": sum(parameter.numel() for parameter in flow.parameters()),
        "algebraic_defects": defects, "gradient_l1_sum": gradient,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

