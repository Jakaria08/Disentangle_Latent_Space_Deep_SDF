#!/usr/bin/env python3
"""Build the pre-registered five-metric test comparison table for the three runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from siren256_common import read_json, write_json


PRIMARY = ["registered_normal_mae", "chamfer_l2_squared", "assd", "volume_relative_error", "annual_log_volume_rate_mae"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--flow-run", default="v5_flow_c3")
    parser.add_argument("--plain-ode-run", default="plain_ode_c3_matched")
    parser.add_argument("--brainode-run", default="brainode_attention_c3_matched")
    args = parser.parse_args()
    root = args.root.resolve()
    names = {"v5_direct_flow": args.flow_run, "plain_ode": args.plain_ode_run, "brainode_attention_ode": args.brainode_run}
    summaries: dict[str, Any] = {}
    validation: dict[str, Any] = {}
    contract = read_json(root / "metadata" / "input_contract.json")
    for label, name in names.items():
        run = root / "runs" / name
        summary = read_json(run / "evaluation" / "test" / "summary.json")
        if int(summary["pair_count"]) != int(contract["forward_pair_counts"]["test"]):
            raise ValueError(f"{name} does not use the common complete QC test-pair table.")
        summaries[label] = summary
        checkpoint = run / "checkpoints" / "best.pt"
        if checkpoint.exists():
            validation[label] = torch.load(checkpoint, map_location="cpu")["validation"]
    table = {label: {metric: float(summary["subject_macro"]["all_subject_macro"][metric]) for metric in PRIMARY} for label, summary in summaries.items()}
    flow = table["v5_direct_flow"]
    wins = {metric: flow[metric] < table["plain_ode"][metric] and flow[metric] < table["brainode_attention_ode"][metric] for metric in PRIMARY}
    rate_order = summaries["v5_direct_flow"].get("test_rate_ordering", {})
    val = validation.get("v5_direct_flow", {})
    flow_config = torch.load(root / "runs" / args.flow_run / "checkpoints" / "best.pt", map_location="cpu").get("config", {}) if (root / "runs" / args.flow_run / "checkpoints" / "best.pt").exists() else {}
    feasibility = flow_config.get("Feasibility", {})
    feasible = bool(val) and float(val.get("semigroup_scaled", float("inf"))) <= float(feasibility.get("semigroup_max_scaled", float("inf"))) and float(val.get("inverse_scaled", float("inf"))) <= float(feasibility.get("inverse_max_scaled", float("inf")))
    correct_order = rate_order.get("predicted_matches_observed_order")
    result = {"metric_direction": {metric: "lower_is_better" for metric in PRIMARY}, "subject_macro_test_table": table, "v5_wins_against_both_odes": wins, "v5_primary_win_count": int(sum(wins.values())), "v5_validation_feasible": feasible, "v5_test_rate_ordering": rate_order, "pre_registered_success_rule_met": bool(sum(wins.values()) >= 3 and feasible and correct_order is True), "validation": validation, "attention_note": summaries["brainode_attention_ode"].get("attention_contract")}
    write_json(root / "comparison_test.json", result)
    print(f"v5 wins {result['v5_primary_win_count']}/5 primary metrics; success={result['pre_registered_success_rule_met']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
