#!/usr/bin/env python3
"""Fast V2 architecture/config tests; writes no experiment artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from train_volume_coboundary import validate_config  # noqa: E402
from volume_coboundary_model import ExactVolumeCoboundaryFlow, build_flow  # noqa: E402


def maximum_rmse(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((left - right).square(), dim=1)).max())


def main() -> int:
    torch.manual_seed(7291)
    configs = []
    for path in sorted((ROOT / "configs").glob("*_volume_exact_coboundary_v2_s42.json")):
        config = json.loads(path.read_text(encoding="utf-8"))
        validate_config(config)
        configs.append(config)
    if {config["representation"] for config in configs} != {"pca128", "spiralnet128", "adaptive128"}:
        raise AssertionError("Expected exactly three V2 representation configs")
    comparable = [json.loads(json.dumps(config)) for config in configs]
    for config in comparable:
        config.pop("name")
        config.pop("representation")
        config["training"].pop("run_name")
        for key in ("batch_size", "decoder_batch_size", "evaluation_batch_size"):
            config["training"].pop(key)
    if not all(config == comparable[0] for config in comparable[1:]):
        raise AssertionError("V2 configs differ beyond name and memory-driven batches")

    coefficient = torch.randn(128)
    flow: ExactVolumeCoboundaryFlow = build_flow(configs[0], coefficient)
    with torch.no_grad():
        for head in (flow.volume_head, *[layer.conditioner for layer in flow.shape_layers]):
            torch.nn.init.normal_(head.cn_head.weight, std=0.01)
            torch.nn.init.normal_(head.cn_head.bias, std=0.01)
            torch.nn.init.normal_(head.ad_residual_head.weight, std=0.01)
            torch.nn.init.normal_(head.ad_residual_head.bias, std=0.01)
    flow.eval()
    batch = 13
    z, context = torch.randn(batch, 128), torch.randn(batch, 128)
    context_age = torch.randn(batch) * 0.1
    source = context_age + torch.rand(batch) * 0.5
    middle = source + torch.rand(batch) * 0.5
    target = middle + torch.rand(batch) * 0.5
    label = torch.randint(0, 2, (batch,), dtype=torch.float32)
    direct = flow.transport(z, source, target, label, context, context_age)
    composed = flow.transport(
        flow.transport(z, source, middle, label, context, context_age),
        middle, target, label, context, context_age,
    )
    recovered = flow.transport(direct, target, source, label, context, context_age)
    identity = flow.transport(z, source, source, label, context, context_age)
    defects = {
        "composition_rmse_max": maximum_rmse(direct, composed),
        "inverse_rmse_max": maximum_rmse(z, recovered),
        "identity_rmse_max": maximum_rmse(z, identity),
    }
    if max(defects.values()) > 1.0e-5:
        raise AssertionError(f"V2 algebra defect too large: {defects}")
    baseline_chart = flow.chart(z, context_age, label, context, context_age)
    target_chart = flow.chart(z, target, label, context, context_age)
    potential = flow.volume_potential(context, context_age, target, label).reshape(-1)
    linear_change = (target_chart - baseline_chart) @ coefficient
    isolation_error = float(torch.max(torch.abs(linear_change - potential)))
    if isolation_error > 3.0e-5:
        raise AssertionError(f"Volume-axis isolation failed: {isolation_error}")
    loss = direct.square().mean()
    loss.backward()
    gradient = sum(float(parameter.grad.abs().sum()) for parameter in flow.parameters() if parameter.grad is not None)
    if not torch.isfinite(loss) or gradient <= 0.0:
        raise AssertionError("V2 gradient audit failed")
    if any(isinstance(module, torch.nn.Dropout) for module in flow.modules()):
        raise AssertionError("Dropout found in V2 invertible map")
    print(json.dumps({
        "status": "passed", "configs": [config["name"] for config in configs],
        "parameters": sum(parameter.numel() for parameter in flow.parameters()),
        "algebraic_defects": defects, "volume_axis_isolation_max_abs": isolation_error,
        "gradient_l1_sum": gradient,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

