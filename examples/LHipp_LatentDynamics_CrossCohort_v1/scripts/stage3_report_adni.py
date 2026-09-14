#!/usr/bin/env python3
"""Stage 3 report: the ADNI internal matrix (protocol P0), tested once.

Reads only files the stage 3 jobs wrote (results_index.json, evaluation summaries and task rows,
training statuses, condition sweeps) plus stage 1 baselines and the stage 2 anchor verification.
It never evaluates a model.

Aggregation: every metric is first averaged per subject over the three seeds; tables report
mean +- SD across subjects, with the SD of the three per-seed means as seed variability. Paired
tests compare subject-level seed means with the cocycle of the same representation.

Outputs under stage3_adni/reports/: stage3_adni_report.md, stage3_gates.json, tables/*.csv,
figures/*.png. ``--allow-partial`` renders whatever exists and marks the report PARTIAL; the final
report requires all 60 seed-runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import benchmark_common as bc
import benchmark_stats as ST
import stage3_build_jobs as S3

REPORT = S3.STAGE3_ROOT / "reports"
TASKS = ("one_shot_first", "one_shot_prev", "all_prior_k", "four_shot")
METHODS = ("direct_c4", "plain_ode", "brainode", "latent_ode", "latent_ode_residual")
METHOD_LABEL = {"direct_c4": "Cocycle (direct C4)", "plain_ode": "Plain ODE", "brainode": "BrainODE",
                "latent_ode": "Latent ODE (faithful)", "latent_ode_residual": "Latent ODE (residual)"}
REP_LABEL = {"pca128": "PCA-128", "spiralnet128": "SpiralNet-128", "adaptive128": "Adaptive-128", "lamm128": "LAMM-128"}
# Reference palette (dataviz skill, light mode): two validated categorical slots + chrome/ink tokens.
INK = {"surface": "#fcfcfb", "primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
       "baseline": "#c3c2b7", "series1": "#2a78d6", "series2": "#eb6834"}


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"subject_id": str, "prefix_indices": str, "prefix_scan_ids": str, "prefix_years": str})


def load(allow_partial: bool) -> tuple[list[dict[str, Any]], list[str]]:
    runs, missing = [], []
    for run in bc.read_json(S3.STAGE3_ROOT / "results_index.json")["runs"]:
        entry = dict(run)
        paths = {split: Path(run["evaluations"][split]) for split in ("val", "test")}
        if not all((p / "summary.json").is_file() for p in paths.values()):
            missing.append(run["key"])
            continue
        entry["summary"] = {split: bc.read_json(p / "summary.json") for split, p in paths.items()}
        entry["test_rows"] = read_csv(paths["test"] / "task_rows.csv")
        status_path = (Path(run["run_dir"]) if run["source"] != "august_anchor" else Path(run["checkpoint"]).parent.parent) / "training_status.json"
        entry["status"] = bc.read_json(status_path) if status_path.is_file() else {}
        history = status_path.with_name("history.jsonl")
        if "elapsed_minutes" not in entry["status"] and history.is_file():
            # August anchors never wrote elapsed_minutes to their status; their history's last row has it.
            lines = [line for line in history.read_text(encoding="utf-8").splitlines() if line.strip()]
            if lines:
                import json

                entry["status"]["elapsed_minutes"] = json.loads(lines[-1]).get("elapsed_minutes")
        if run["source"] != "august_anchor" and entry["status"].get("status") != "complete":
            missing.append(run["key"])
            continue
        if "min_epoch_evaluations" in run and (Path(run["min_epoch_evaluations"]["test"]) / "summary.json").is_file():
            entry["min_epoch_test_rows"] = read_csv(Path(run["min_epoch_evaluations"]["test"]) / "task_rows.csv")
        sweep = S3.STAGE3_ROOT / "condition_sweeps" / f"{run['key']}.json"
        if sweep.is_file():
            entry["sweep"] = bc.read_json(sweep)
            entry["sweep_rows"] = pd.read_csv(sweep.with_suffix(".csv"), dtype={"subject_id": str})
        runs.append(entry)
    if missing and not allow_partial:
        raise RuntimeError(f"{len(missing)} seed-runs are incomplete (e.g. {missing[:4]}); pass --allow-partial for a draft")
    return runs, missing


def task_frame(runs: list[dict[str, Any]], key: str = "test_rows") -> pd.DataFrame:
    frames = []
    for run in runs:
        if key in run:
            frames.append(run[key].assign(representation=run["representation"], method=run["method"], seed=run["seed"]))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------------------


def gates(runs: list[dict[str, Any]], missing: list[str]) -> list[dict[str, Any]]:
    out = [{"id": "G3.1", "check": "all 60 seed-runs complete with val and test evaluations", "passed": not missing,
            "detail": f"{len(runs)} complete, {len(missing)} missing"}]
    leaked = [r["key"] for r in runs if any(r["summary"][s].get("test_loaded_during_training") for s in r["summary"])
              or r["status"].get("test_data_loaded")]
    out.append({"id": "G3.2", "check": "no run loaded test data during training or selection", "passed": not leaked, "detail": leaked})
    verification = bc.read_json(bc.BULK_ROOT / "stage2_validation" / "reports" / "anchor_verification.json")
    out.append({"id": "G3.3", "check": "reused anchors reproduce their stored summaries (stage 2 G2.7a)", "passed": bool(verification["passed"]),
                "detail": {k: v["max_abs_difference"] for k, v in verification["anchors"].items()}})
    baseline_val = read_csv(bc.STAGE1_ROOT / "baselines" / "p0_internal_adni" / "val_baseline_rows.csv")
    nochange = baseline_val[(baseline_val.task == "one_shot_first") & (baseline_val.baseline == "nochange_decoded")].groupby("representation")["euclidean_mm"].mean()
    flags = []
    for r in runs:
        record = next(t for t in r["summary"]["val"]["tasks"] if t["task"] == "one_shot_first" and t["variant"] == "averaged" and t["group"] == "overall")
        if record["euclidean_mm_mean"] >= nochange[r["representation"]]:
            flags.append(f"{r['key']} ({record['euclidean_mm_mean']:.4f} >= no-change {nochange[r['representation']]:.4f})")
    out.append({"id": "G3.4", "check": "models that do not beat no-change on validation (flagged, never dropped)", "passed": True, "detail": flags})
    bad = []
    for r in runs:
        defects = r["summary"]["test"]["consistency_defects"]
        if r["method"] == "direct_c4" and max(defects["relative_semigroup_defect_mean"], defects["relative_inverse_defect_mean"]) > 0.25:
            bad.append(r["key"])
        if r["method"] in ("plain_ode", "brainode") and defects["relative_semigroup_defect_mean"] > 1e-5:
            bad.append(r["key"])
    out.append({"id": "G3.5", "check": "cocycle defects <= 0.25; ODE semigroup defect <= 1e-5", "passed": not bad, "detail": bad})
    hashes = {r["key"]: bc.sha256_file(Path(r["evaluations"]["test"]) / "summary.json") for r in runs}
    out.append({"id": "G3.6", "check": "one sealed test summary per seed-run (hashes recorded)", "passed": len(hashes) == len(runs), "detail": hashes})
    return out


# --------------------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------------------


def fmt(mean: float, sd: float, digits: int = 4) -> str:
    return f"{mean:.{digits}f} ± {sd:.{digits}f}" if np.isfinite(mean) else "-"


def table_errors(frame: pd.DataFrame) -> pd.DataFrame:
    """P3-A: Euclidean error per representation x model x task (subject seed-means)."""
    records = []
    variants = frame[["method", "variant"]].drop_duplicates().itertuples(index=False)
    for method, variant in sorted(variants):
        for representation in REP_LABEL:
            for task in TASKS:
                part = frame[(frame.method == method) & (frame.variant == variant) & (frame.representation == representation) & (frame.task == task)]
                if part.empty:
                    continue
                per_subject = part.groupby("subject_id")[["euclidean_mm", "coordinate_mae_mm"]].mean()
                per_seed = part.groupby("seed")["euclidean_mm"].mean()
                records.append({"representation": representation, "model": method if variant == "averaged" else f"{method}__native", "task": task,
                                "subjects": len(per_subject), "seeds": int(part.seed.nunique()),
                                "euclidean_mm_mean": per_subject.euclidean_mm.mean(), "euclidean_mm_sd": per_subject.euclidean_mm.std(ddof=1),
                                "seed_sd": per_seed.std(ddof=1) if len(per_seed) > 1 else np.nan,
                                "coordinate_mae_mm_mean": per_subject.coordinate_mae_mm.mean()})
    baseline = read_csv(bc.STAGE1_ROOT / "baselines" / "p0_internal_adni" / "test_baseline_rows.csv")
    for (representation, name, task), part in baseline.groupby(["representation", "baseline", "task"]):
        if name not in ("nochange_decoded", "linear_decoded", "nochange_raw", "linear_raw") or task not in TASKS:
            continue
        records.append({"representation": representation, "model": f"baseline:{name}", "task": task, "subjects": len(part), "seeds": 0,
                        "euclidean_mm_mean": part.euclidean_mm.mean(), "euclidean_mm_sd": part.euclidean_mm.std(ddof=1), "seed_sd": np.nan,
                        "coordinate_mae_mm_mean": part.coordinate_mae_mm.mean()})
    return pd.DataFrame(records)


def table_fidelity(frame: pd.DataFrame) -> pd.DataFrame:
    """P3-B: AD/CN atrophy capture, predicted vs observed AD/CN rate ratio, subject slope agreement."""
    records = []
    part = frame[(frame.task == "one_shot_first") & (frame.variant == "averaged")]
    for (representation, method), cell in part.groupby(["representation", "method"]):
        per_seed = []
        for seed, rows in cell.groupby("seed"):
            ad, cn = rows[rows.diagnosis == "AD"], rows[rows.diagnosis == "CN"]
            per_seed.append({"ad_capture": ad.predicted_log_volume_rate.mean() / ad.observed_log_volume_rate.mean(),
                             "cn_capture": cn.predicted_log_volume_rate.mean() / cn.observed_log_volume_rate.mean(),
                             "predicted_ad_cn_ratio": ad.predicted_log_volume_rate.mean() / cn.predicted_log_volume_rate.mean(),
                             "observed_ad_cn_ratio": ad.observed_log_volume_rate.mean() / cn.observed_log_volume_rate.mean()})
        seeds = pd.DataFrame(per_seed)
        subjects = cell.groupby(["subject_id", "diagnosis"])[["predicted_log_volume_rate", "observed_log_volume_rate"]].mean().reset_index()
        ad_subjects = subjects[subjects.diagnosis == "AD"]
        corr_ad = ST.correlations(ad_subjects.predicted_log_volume_rate, ad_subjects.observed_log_volume_rate)
        corr_all = ST.correlations(subjects.predicted_log_volume_rate, subjects.observed_log_volume_rate)
        records.append({"representation": representation, "model": method, "seeds": len(seeds),
                        **{f"{c}_mean": seeds[c].mean() for c in seeds}, **{f"{c}_seed_sd": seeds[c].std(ddof=1) for c in seeds},
                        "ad_slope_pearson": corr_ad["pearson_r"], "ad_slope_spearman": corr_ad["spearman_rho"],
                        "all_slope_pearson": corr_all["pearson_r"], "all_slope_spearman": corr_all["spearman_rho"]})
    return pd.DataFrame(records)


def table_consistency(runs: list[dict[str, Any]]) -> pd.DataFrame:
    """P3-C and P3-D: composition defects, training cost, best epochs, error by horizon."""
    rows = []
    for r in runs:
        pairs = r["summary"]["test"]["pair_metrics"]
        rows.append({"representation": r["representation"], "model": r["method"], "seed": r["seed"], "source": r["source"],
                     "semigroup_defect": r["summary"]["test"]["consistency_defects"]["relative_semigroup_defect_mean"],
                     "inverse_defect": r["summary"]["test"]["consistency_defects"]["relative_inverse_defect_mean"],
                     "train_minutes": r["status"].get("elapsed_minutes", np.nan), "best_epoch": r["status"].get("best_epoch", np.nan),
                     **{f"horizon_{h}_euclidean_mm": pairs.get(f"horizon_{h}_forward", {}).get("groups", {}).get("overall", {}).get("end_to_end_euclidean_mean", np.nan)
                        for h in ("le_1y", "gt_1_le_2y", "gt_2y")}})
    frame = pd.DataFrame(rows)
    numeric = [c for c in frame.columns if c not in ("representation", "model", "seed", "source")]
    return frame.groupby(["representation", "model"])[numeric].mean().reset_index()


def table_sweeps(runs: list[dict[str, Any]]) -> pd.DataFrame:
    """P3-E: what the disease condition does from identical baselines."""
    rows = []
    for r in runs:
        if "sweep" not in r:
            continue
        g = r["sweep"]["groups"]
        rows.append({"representation": r["representation"], "model": r["method"], "seed": r["seed"],
                     "effect_annual_log_rate_0_2y": g["all"]["condition_effect_annual_log_rate_0_2y"],
                     "effect_over_observed_ad_cn_gap": r["sweep"].get("condition_effect_over_observed_gap", np.nan),
                     "cn_d0_rate": g["CN"]["d0_annual_log_rate_0_2y"], "ad_d1_rate": g["AD"]["d1_annual_log_rate_0_2y"],
                     "cn_observed_rate": g["CN"]["observed_first_last_log_rate_mean"], "ad_observed_rate": g["AD"]["observed_first_last_log_rate_mean"]})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    numeric = [c for c in frame.columns if c not in ("representation", "model", "seed")]
    mean = frame.groupby(["representation", "model"])[numeric].mean()
    sd = frame.groupby(["representation", "model"])[numeric].std(ddof=1).add_suffix("_seed_sd")
    return mean.join(sd).reset_index()


def table_paired(frame: pd.DataFrame) -> pd.DataFrame:
    """P3-F: per-subject error difference vs the cocycle of the same representation, Holm-corrected."""
    records = []
    part = frame[frame.variant == "averaged"]
    for representation in REP_LABEL:
        for task in ("one_shot_first", "all_prior_k"):
            cell = part[(part.representation == representation) & (part.task == task)]
            means = cell.groupby(["method", "subject_id"])["euclidean_mm"].mean().unstack(0)
            if "direct_c4" not in means:
                continue
            block = {}
            for method in METHODS[1:]:
                if method not in means:
                    continue
                aligned = means[["direct_c4", method]].dropna()
                block[method] = ST.paired_difference(aligned[method], aligned["direct_c4"])
            adjusted = ST.holm({m: v["wilcoxon_p"] for m, v in block.items()})
            for method, result in block.items():
                records.append({"representation": representation, "task": task, "model": method, **result, "holm_p": adjusted[method]})
    return pd.DataFrame(records)


def table_sensitivity(runs: list[dict[str, Any]]) -> pd.DataFrame:
    """P3-G: cocycle best.pt vs best_min_epoch.pt, per seed."""
    rows = []
    for r in runs:
        if r["method"] != "direct_c4" or "min_epoch_test_rows" not in r:
            continue
        for label, key in (("best", "test_rows"), ("min_epoch_15", "min_epoch_test_rows")):
            part = r[key][(r[key].task == "one_shot_first") & (r[key].variant == "averaged")]
            ad = part[part.diagnosis == "AD"]
            rows.append({"representation": r["representation"], "seed": r["seed"], "checkpoint": label, "euclidean_mm": part.euclidean_mm.mean(),
                         "ad_capture": ad.predicted_log_volume_rate.mean() / ad.observed_log_volume_rate.mean()})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------


def style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": INK["surface"], "axes.facecolor": INK["surface"], "savefig.facecolor": INK["surface"],
        "axes.edgecolor": INK["baseline"], "axes.linewidth": 0.6, "axes.grid": True, "grid.color": INK["grid"], "grid.linewidth": 0.5,
        "grid.linestyle": "-", "axes.spines.top": False, "axes.spines.right": False, "text.color": INK["primary"],
        "axes.labelcolor": INK["secondary"], "xtick.color": INK["muted"], "ytick.color": INK["muted"], "font.size": 8,
        "font.family": "sans-serif", "lines.linewidth": 1.5, "lines.solid_capstyle": "round", "legend.frameon": False,
    })
    return plt


def figure_sweeps(runs: list[dict[str, Any]], path: Path) -> bool:
    frames = [r["sweep_rows"].assign(representation=r["representation"], method=r["method"], seed=r["seed"]) for r in runs if "sweep_rows" in r]
    if not frames:
        return False
    plt = style()
    data = pd.concat(frames, ignore_index=True)
    curves = data.groupby(["representation", "method", "seed", "condition", "horizon_years"])["log_change_from_h0"].mean().groupby(
        ["representation", "method", "condition", "horizon_years"]).mean().reset_index()
    observed = {(r["representation"], r["method"]): (r["sweep"]["groups"]["CN"]["observed_first_last_log_rate_mean"],
                                                     r["sweep"]["groups"]["AD"]["observed_first_last_log_rate_mean"]) for r in runs if "sweep" in r}
    fig, axes = plt.subplots(len(METHODS), len(REP_LABEL), figsize=(10, 11), sharex=True, sharey=True)
    for i, method in enumerate(METHODS):
        for j, representation in enumerate(REP_LABEL):
            ax = axes[i][j]
            cell = curves[(curves.method == method) & (curves.representation == representation)]
            if (representation, method) in observed:
                cn_rate, ad_rate = observed[(representation, method)]
                h = np.array([0.0, 8.0])
                ax.plot(h, 100 * cn_rate * h, color=INK["muted"], linewidth=0.8)
                ax.plot(h, 100 * ad_rate * h, color=INK["muted"], linewidth=0.8)
            for condition, color in ((0, INK["series1"]), (1, INK["series2"])):
                line = cell[cell.condition == condition]
                ax.plot(line.horizon_years, 100 * line.log_change_from_h0, color=color, linewidth=1.5)
            if i == 0:
                ax.set_title(REP_LABEL[representation], color=INK["primary"], fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{METHOD_LABEL[method]}\nlog-volume change (%)")
            if i == len(METHODS) - 1:
                ax.set_xlabel("years from first visit")
    handles = [plt.Line2D([], [], color=INK["series1"], linewidth=1.5), plt.Line2D([], [], color=INK["series2"], linewidth=1.5),
               plt.Line2D([], [], color=INK["muted"], linewidth=0.8)]
    fig.legend(handles, ["condition d = 0 (control)", "condition d = 1 (disease)", "observed CN (upper) and AD (lower) first-to-last rate"],
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Predicted hippocampal volume change from identical ADNI test baselines (seed mean)", y=1.02, fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def figure_paired(paired: pd.DataFrame, path: Path) -> bool:
    if paired.empty:
        return False
    plt = style()
    tasks = [t for t in ("one_shot_first", "all_prior_k") if t in set(paired.task)]
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.2 * len(tasks), 5.6), sharey=True)
    axes = np.atleast_1d(axes)
    order = [(rep, m) for rep in REP_LABEL for m in METHODS[1:]]
    labels = [f"{REP_LABEL[rep]} · {METHOD_LABEL[m]}" for rep, m in order]
    for ax, task in zip(axes, tasks):
        part = paired[paired.task == task].set_index(["representation", "model"])
        ax.axvline(0.0, color=INK["baseline"], linewidth=0.8)
        for y, key in enumerate(order):
            if key not in part.index:
                continue
            row = part.loc[key]
            color = INK["series1"] if row.holm_p < 0.05 else INK["muted"]
            ax.plot([row.ci95_low, row.ci95_high], [y, y], color=color, linewidth=1.5)
            ax.plot(row.mean_difference, y, "o", color=color, markersize=5, markeredgecolor=INK["surface"], markeredgewidth=1.2)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(labels)
        ax.set_title(f"{task} (n = {int(part['n'].max())} subjects)", fontsize=9)
        ax.set_xlabel("Euclidean error vs cocycle (mm); > 0 means worse than cocycle")
    # Shared y axis: set the (inverted, padded) limits once. Calling invert_yaxis per panel flips it back.
    axes[0].set_ylim(len(order) - 0.5, -0.5)
    handles = [plt.Line2D([], [], color=INK["series1"], marker="o", linewidth=1.5), plt.Line2D([], [], color=INK["muted"], marker="o", linewidth=1.5)]
    fig.legend(handles, ["Holm-adjusted Wilcoxon p < 0.05", "not significant"], loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


# --------------------------------------------------------------------------------------
# markdown
# --------------------------------------------------------------------------------------


def md_table(frame: pd.DataFrame, columns: list[str], headers: list[str]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in columns) + " |")
    return lines


def model_name(model: str) -> str:
    if model.startswith("baseline:"):
        return {"baseline:nochange_decoded": "no-change (decoded)", "baseline:linear_decoded": "linear extrapolation (decoded)",
                "baseline:nochange_raw": "no-change (raw mesh)", "baseline:linear_raw": "linear extrapolation (raw mesh)"}[model]
    if model.endswith("__native"):
        return METHOD_LABEL[model[: -len("__native")]] + " - native k-observation encoding"
    return METHOD_LABEL[model]


def build_markdown(gate_list, errors, fidelity, consistency, sweeps, paired, sensitivity, partial, figures) -> str:
    lines = [f"# Stage 3 - ADNI internal matrix (P0){' - PARTIAL DRAFT' if partial else ''}", "",
             "Left hippocampus, ADNI test split (61 subjects), evaluated once. Every model is selected on validation only. "
             "Errors are per-vertex Euclidean distances (mm) to the real mesh, first averaged per subject over seeds 42/43/44; "
             "values are mean ± SD across subjects. BrainODE's published mm values come from a different template and pipeline and are not comparable.", "",
             "## Gates", "", "| gate | check | result |", "|---|---|---|"]
    lines += [f"| {g['id']} | {g['check']} | {'PASS' if g['passed'] else 'FAIL'} |" for g in gate_list]
    flags = next(g for g in gate_list if g["id"] == "G3.4")["detail"]
    if flags:
        lines += ["", "Flagged (not better than no-change on validation): " + "; ".join(flags)]

    lines += ["", "## P3-A. Shape prediction error (BrainODE Table 7 analog)", ""]
    for representation in REP_LABEL:
        part = errors[errors.representation.isin([representation, "raw_mesh"])]
        if part.empty:
            continue
        pivot = part.assign(cell=[fmt(m, s) for m, s in zip(part.euclidean_mm_mean, part.euclidean_mm_sd)]).pivot_table(
            index="model", columns="task", values="cell", aggfunc="first")
        counts = part.groupby("task")["subjects"].max()
        headers = ["model"] + [f"{t} (n={counts.get(t, 0)})" for t in TASKS if t in pivot.columns]
        lines += [f"### {REP_LABEL[representation]}", "", "| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
        order = [m for m in METHODS if m in pivot.index] + [f"{m}__native" for m in METHODS if f"{m}__native" in pivot.index] + \
                [b for b in ("baseline:nochange_decoded", "baseline:linear_decoded", "baseline:nochange_raw", "baseline:linear_raw") if b in pivot.index]
        for model in order:
            cells = [pivot.loc[model].get(t, "-") for t in TASKS if t in pivot.columns]
            lines.append("| " + model_name(model) + " | " + " | ".join(c if isinstance(c, str) else "-" for c in cells) + " |")
        lines.append("")
    lines += ["four_shot is almost entirely CN in ADNI test (1 AD subject). all_prior_k averages min(n-1, 4) predictions per subject.", ""]

    if not fidelity.empty:
        f = fidelity.copy()
        f["model"] = [METHOD_LABEL[m] for m in f.model]
        f["representation"] = [REP_LABEL[r] for r in f.representation]
        for c in ("ad_capture", "cn_capture", "predicted_ad_cn_ratio"):
            f[c] = [fmt(m, s, 2) for m, s in zip(f[f"{c}_mean"], f[f"{c}_seed_sd"].fillna(0))]
        f["observed_ratio"] = f["observed_ad_cn_ratio_mean"].round(2)
        f["ad_slope"] = f["ad_slope_pearson"].round(2)
        lines += ["## P3-B. Disease fidelity (one_shot_first)", "",
                  "Capture = mean predicted log-volume rate / mean observed rate (1.00 = the observed atrophy speed); ± is seed SD. "
                  "AD slope r = Pearson correlation of each AD subject's predicted and observed rate.", ""]
        lines += md_table(f, ["representation", "model", "ad_capture", "cn_capture", "predicted_ad_cn_ratio", "observed_ratio", "ad_slope"],
                          ["representation", "model", "AD capture", "CN capture", "predicted AD/CN rate", "observed AD/CN rate", "AD slope r"])
        lines.append("")
    if not sweeps.empty:
        s = sweeps.copy()
        s["model"] = [METHOD_LABEL[m] for m in s.model]
        s["representation"] = [REP_LABEL[r] for r in s.representation]
        s["effect"] = [fmt(m, sd) for m, sd in zip(s.effect_annual_log_rate_0_2y, s.effect_annual_log_rate_0_2y_seed_sd.fillna(0))]
        s["share"] = s.effect_over_observed_ad_cn_gap.round(2)
        lines += ["## P3-E. Condition sweep (BrainODE Fig. 3 analog)", "",
                  "Same baselines transported with d = 0 and d = 1. Effect = annual log-volume rate(d=1) - rate(d=0) over 0-2 y; "
                  "share = effect / observed AD-minus-CN rate gap.", ""]
        lines += md_table(s, ["representation", "model", "effect", "share"], ["representation", "model", "condition effect (per year)", "share of observed gap"])
        lines.append("")
    if not paired.empty:
        p = paired.copy()
        p["model"] = [METHOD_LABEL[m] for m in p.model]
        p["representation"] = [REP_LABEL[r] for r in p.representation]
        p["difference"] = [f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]" for m, lo, hi in zip(p.mean_difference, p.ci95_low, p.ci95_high)]
        p["holm"] = [f"{v:.3g}" for v in p.holm_p]
        lines += ["## P3-F. Paired error difference vs the cocycle", "", "> 0 means larger error than the cocycle of the same representation. Holm correction within representation x task.", ""]
        lines += md_table(p, ["representation", "task", "model", "difference", "holm"], ["representation", "task", "model", "mean difference mm [95% CI]", "Holm p"])
        lines.append("")
    if not consistency.empty:
        c = consistency.copy()
        c["model"] = [METHOD_LABEL[m] for m in c.model]
        c["representation"] = [REP_LABEL[r] for r in c.representation]
        for col in ("semigroup_defect", "inverse_defect"):
            c[col] = [f"{v:.2e}" for v in c[col]]
        for col in ("train_minutes", "best_epoch", "horizon_le_1y_euclidean_mm", "horizon_gt_1_le_2y_euclidean_mm", "horizon_gt_2y_euclidean_mm"):
            c[col] = c[col].round(3)
        lines += ["## P3-C/D. Consistency, cost and error by horizon (seed means)", ""]
        lines += md_table(c, ["representation", "model", "semigroup_defect", "inverse_defect", "train_minutes", "best_epoch",
                              "horizon_le_1y_euclidean_mm", "horizon_gt_1_le_2y_euclidean_mm", "horizon_gt_2y_euclidean_mm"],
                          ["representation", "model", "semigroup defect", "inverse defect", "train min", "best epoch", "<=1 y mm", "1-2 y mm", ">2 y mm"])
        lines.append("")
    if not sensitivity.empty:
        g = sensitivity.groupby(["representation", "checkpoint"])[["euclidean_mm", "ad_capture"]].mean().round(4).reset_index()
        g["representation"] = [REP_LABEL[r] for r in g.representation]
        lines += ["## P3-G. Cocycle selection sensitivity (best.pt vs best epoch >= 15)", ""]
        lines += md_table(g, ["representation", "checkpoint", "euclidean_mm", "ad_capture"], ["representation", "checkpoint", "one_shot_first mm", "AD capture"])
        lines.append("")
    lines += ["## Figures", ""] + [f"![{name}](figures/{name})" for name in figures] + [
        "", "Tables are in `tables/` (the data behind every figure).",
        "Figure colors are categorical slots 1-2 of the dataviz reference palette, which that palette documents as validated "
        "(adjacent and all-pairs CVD checks, light mode); Node.js is not installed on this machine, so the validator script was not re-run here. "
        "Identity never rests on color alone: figures carry legends and axis labels."]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--device", default=None, help="Ignored; accepted because the orchestrator appends it.")
    args = parser.parse_args()
    runs, missing = load(args.allow_partial)
    frame = task_frame(runs)
    gate_list = gates(runs, missing)
    errors, fidelity = table_errors(frame), table_fidelity(frame)
    consistency, sweeps, paired, sensitivity = table_consistency(runs), table_sweeps(runs), table_paired(frame), table_sensitivity(runs)
    tables = bc.require_bulk(REPORT / "tables")
    for name, table in (("p3a_errors", errors), ("p3b_fidelity", fidelity), ("p3cd_consistency_horizons", consistency),
                        ("p3e_condition_sweeps", sweeps), ("p3f_paired_vs_cocycle", paired), ("p3g_selection_sensitivity", sensitivity)):
        bc.atomic_csv(tables / f"{name}.csv", table)
    figures = []
    if figure_sweeps(runs, REPORT / "figures" / "p3e_condition_sweeps.png"):
        figures.append("p3e_condition_sweeps.png")
    if figure_paired(paired, REPORT / "figures" / "p3f_paired_vs_cocycle.png"):
        figures.append("p3f_paired_vs_cocycle.png")
    bc.atomic_json(REPORT / "stage3_gates.json", {"partial": bool(missing), "gates": gate_list})
    name = "stage3_adni_report_partial.md" if missing else "stage3_adni_report.md"
    bc.atomic_write_text(REPORT / name, build_markdown(gate_list, errors, fidelity, consistency, sweeps, paired, sensitivity, bool(missing), figures))
    print(f"wrote {REPORT / name}: {len(runs)} complete seed-runs, {len(missing)} missing; gates " +
          ", ".join(f"{g['id']}={'PASS' if g['passed'] else 'FAIL'}" for g in gate_list))
    hard = [g for g in gate_list if g["id"] in ("G3.1", "G3.2", "G3.3", "G3.5", "G3.6") and not g["passed"]]
    return 0 if (not hard or args.allow_partial) else 1


if __name__ == "__main__":
    raise SystemExit(main())
