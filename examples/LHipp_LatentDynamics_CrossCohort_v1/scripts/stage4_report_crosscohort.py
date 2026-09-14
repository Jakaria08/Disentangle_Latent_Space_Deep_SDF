#!/usr/bin/env python3
"""Stage 4 report: cross-cohort protocols P1-P4c, BrainODE-style per-dataset tables.

Reads only evaluation outputs (stage 4 results index, stage 3 ADNI results, stage 1 baselines) and
never evaluates a model. Every model number is paired with the no-change error on the *same*
subjects and representation, because protocols evaluate different subject sets (test split, whole
cohort, out-of-fold); skill = 1 - model error / no-change error.

Tables (reports/tables): s4a_per_dataset, s4c_unified_pooled, s4d_cross_benchmark, s4e_gaps,
s4f_interval_shift, s4g_ad_fidelity, s4h_calsnic_ood. Figure: s4f_interval_shift.png.
``--allow-partial`` renders whatever exists.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import benchmark_common as bc
import benchmark_stats as ST
import stage3_build_jobs as S3
import stage3_report_adni as R3
import stage4_build_jobs as S4

REPORT = S4.STAGE4_ROOT / "reports"
COHORTS = ("adni", "aibl", "oasis", "calsnic")
METHODS = R3.METHODS
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # dataviz reference categorical slots 1-5 (fixed order)


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------


def read_rows(output: str | Path) -> pd.DataFrame | None:
    path = Path(output) / "task_rows.csv"
    if not (Path(output) / "summary.json").is_file() or not path.is_file():
        return None
    rows = pd.read_csv(path, dtype={"subject_id": str})
    return rows[rows.variant == "averaged"]


def load_rows(allow_partial: bool) -> tuple[pd.DataFrame, list[str], dict]:
    index = bc.read_json(S4.STAGE4_ROOT / "results_index.json")
    frames, missing = [], []

    def add(rows, **meta):
        frames.append(rows.assign(**meta))

    for run in bc.read_json(S3.STAGE3_ROOT / "results_index.json")["runs"]:
        rows = read_rows(run["evaluations"]["test"])
        if rows is None:
            missing.append(f"P0:{run['key']}")
        else:
            add(rows, protocol="P0", view="p0_internal_adni", override=-1, representation=run["representation"], method=run["method"], seed=run["seed"])
    for run in index["trained_runs"]:
        if run["blocked"]:
            continue
        rows = read_rows(run["evaluations"]["test"])
        if rows is None:
            missing.append(run["key"])
        else:
            add(rows, protocol=run["protocol"], view=run["view"], override=-1, representation=run["representation"], method=run["method"], seed=run["seed"])
    for entry in index["external_evaluations"] + index["transfer_evaluations"]:
        rows = read_rows(entry["output"])
        if rows is None:
            missing.append(entry["key"])
        else:
            override = -1 if entry["condition_override"] is None else int(entry["condition_override"])
            add(rows, protocol=entry["protocol"], view=entry["view"], override=override, representation=entry["representation"],
                method=entry["method"], seed=entry["seed"])
    if missing and not allow_partial:
        raise RuntimeError(f"{len(missing)} evaluations missing (e.g. {missing[:4]}); pass --allow-partial for a draft")
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), missing, index


def load_baselines(views: set[str]) -> pd.DataFrame:
    frames = []
    for view in sorted(views):
        path = bc.STAGE1_ROOT / "baselines" / view / "test_baseline_rows.csv"
        if path.is_file():
            frame = pd.read_csv(path, dtype={"subject_id": str})
            frames.append(frame[frame.baseline == "nochange_decoded"][["view", "task", "representation", "subject_id", "euclidean_mm"]])
    return pd.concat(frames, ignore_index=True).rename(columns={"euclidean_mm": "nochange_mm"}) if frames else pd.DataFrame()


# --------------------------------------------------------------------------------------
# scopes: which rows answer "model X on cohort C under protocol P"
# --------------------------------------------------------------------------------------


def scopes() -> list[tuple[str, str, str, callable]]:
    """(scope label, cohort, description, row filter)."""
    out = [("P0 internal (test split)", "adni", "ADNI-trained, ADNI test", lambda f: f.view == "p0_internal_adni")]
    for c in ("aibl", "oasis", "calsnic"):
        out += [
            ("P1 zero-shot (whole cohort)", c, "ADNI-trained", lambda f, c=c: (f.view == f"p1_external_{c}_wholecohort") & (f.override == -1)),
            ("P1 zero-shot (test split)", c, "ADNI-trained", lambda f, c=c: (f.view == f"p1_external_{c}_testsplit") & (f.override == -1)),
            ("P2 internal (test split)", c, f"{c}-trained", lambda f, c=c: f.view == f"p2_internal_{c}"),
            ("P4 LOCO (whole cohort)" if c != "calsnic" else "P4c pooled -> CALSNIC (whole)", c, "trained without this cohort",
             (lambda f, c=c: f.view == f"p4_loco_without_{c}") if c != "calsnic" else (lambda f: (f.view == "p4c_exp2_pooled_to_calsnic") & (f.override == -1))),
        ]
        if c != "calsnic":
            out += [("P2 cross-fit (whole, out-of-fold)", c, f"{c}-trained, 5 folds", lambda f, c=c: f.view.str.startswith(f"p2_crossfit_{c}_fold")),
                    ("P3 pooled (test split)", c, "ADNI+AIBL+OASIS-trained", lambda f, c=c: (f.view == "p3_pooled") & (f.cohort == c))]
    out += [("P3 pooled (test split)", "adni", "ADNI+AIBL+OASIS-trained", lambda f: (f.view == "p3_pooled") & (f.cohort == "adni")),
            ("P4 LOCO (whole cohort)", "adni", "trained on AIBL+OASIS", lambda f: f.view == "p4_loco_without_adni")]
    return out


def summarize_cell(rows: pd.DataFrame, baselines: pd.DataFrame, task: str) -> dict:
    part = rows[rows.task == task]
    if part.empty:
        return {}
    per_subject = part.groupby(["view", "subject_id", "diagnosis"])[["euclidean_mm", "predicted_log_volume_rate", "observed_log_volume_rate"]].mean().reset_index()
    record = {"subjects": len(per_subject), "seeds": int(part.seed.nunique()), "euclidean_mean": per_subject.euclidean_mm.mean(),
              "euclidean_sd": per_subject.euclidean_mm.std(ddof=1) if len(per_subject) > 1 else 0.0}
    rep = part.representation.iloc[0]
    base = baselines[(baselines.task == task) & (baselines.representation == rep)] if not baselines.empty else baselines
    if not base.empty:
        matched = per_subject.merge(base, on=["view", "subject_id"], how="inner")
        if len(matched) == len(per_subject):
            record["nochange_mean"] = matched.nochange_mm.mean()
            record["skill"] = 1.0 - matched.euclidean_mm.mean() / matched.nochange_mm.mean()
    for label, name in (("AD", "ad"), ("CN", "cn")):
        group = per_subject[per_subject.diagnosis == label]
        record[f"{name}_subjects"] = len(group)
        if len(group) >= 3 and abs(group.observed_log_volume_rate.mean()) > 1e-9:
            record[f"{name}_capture"] = group.predicted_log_volume_rate.mean() / group.observed_log_volume_rate.mean()
    return record


def table_per_dataset(rows: pd.DataFrame, baselines: pd.DataFrame, task: str = "one_shot_first") -> pd.DataFrame:
    records = []
    for label, cohort, description, select in scopes():
        chosen = rows[select(rows)]
        if cohort != "adni" or label.startswith("P0") or label.startswith("P3") or label.startswith("P4"):
            chosen = chosen[chosen.cohort == cohort]
        for (method, representation), cell in chosen.groupby(["method", "representation"]):
            summary = summarize_cell(cell, baselines, task)
            if summary:
                records.append({"cohort": cohort, "scope": label, "training": description, "method": method, "representation": representation, **summary})
    return pd.DataFrame(records)


def table_unified(rows: pd.DataFrame) -> pd.DataFrame:
    part = rows[(rows.view == "p3_pooled") & rows.task.isin(["one_shot_first", "all_prior_k", "four_shot"])]
    out = []
    for (method, representation, task), cell in part.groupby(["method", "representation", "task"]):
        per_subject = cell.groupby(["cohort", "subject_id"]).euclidean_mm.mean()
        out.append({"method": method, "representation": representation, "task": task, "subjects": len(per_subject),
                    "euclidean_mean": per_subject.mean(), "euclidean_sd": per_subject.std(ddof=1)})
    return pd.DataFrame(out)


def table_cross_benchmark(rows: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    targets = [("LOCO -> ADNI", lambda f: f.view == "p4_loco_without_adni"), ("LOCO -> AIBL", lambda f: f.view == "p4_loco_without_aibl"),
               ("LOCO -> OASIS", lambda f: f.view == "p4_loco_without_oasis"),
               ("Exp1: AIBL -> ADNI+OASIS", lambda f: f.view == "p4b_exp1_aibl_to_adni_oasis"),
               ("Exp2 analog: pooled -> CALSNIC Control", lambda f: (f.view == "p4c_exp2_pooled_to_calsnic") & (f.override == -1) & (f.diagnosis == "CN"))]
    out = []
    for label, select in targets:
        for (method, representation), cell in rows[select(rows)].groupby(["method", "representation"]):
            summary = summarize_cell(cell, baselines, "one_shot_first")
            if summary:
                out.append({"benchmark": label, "method": method, "representation": representation, **summary})
    return pd.DataFrame(out)


def table_gaps(rows: pd.DataFrame) -> pd.DataFrame:
    """Transfer gap = P1 (ADNI-trained) - P2 (cohort-trained); pooling gain = P2 - P3; same test-split subjects."""
    out = []
    part = rows[rows.task == "one_shot_first"]
    for cohort in ("aibl", "oasis", "calsnic"):
        comparisons = [("transfer gap: zero-shot minus internal", f"p1_external_{cohort}_testsplit", f"p2_internal_{cohort}")]
        if cohort != "calsnic":
            comparisons.append(("pooling gain: internal minus pooled", f"p2_internal_{cohort}", "p3_pooled"))
        for label, view_a, view_b in comparisons:
            a = part[(part.view == view_a) & (part.override == -1) & (part.cohort == cohort)]
            b = part[(part.view == view_b) & (part.cohort == cohort)]
            for (method, representation), cell_a in a.groupby(["method", "representation"]):
                cell_b = b[(b.method == method) & (b.representation == representation)]
                if cell_b.empty:
                    continue
                joined = pd.concat([cell_a.groupby("subject_id").euclidean_mm.mean().rename("a"),
                                    cell_b.groupby("subject_id").euclidean_mm.mean().rename("b")], axis=1).dropna()
                if len(joined) >= 3:
                    out.append({"cohort": cohort, "comparison": label, "method": method, "representation": representation,
                                **ST.paired_difference(joined.a, joined.b)})
    return pd.DataFrame(out)


def table_interval_shift(rows: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    views = ["p0_internal_adni"] + [f"p1_external_{c}_wholecohort" for c in ("aibl", "oasis", "calsnic")]
    part = rows[rows.view.isin(views) & (rows.task == "one_shot_first") & (rows.override == -1)]
    if part.empty:
        return pd.DataFrame()
    per_subject = part.groupby(["view", "method", "representation", "subject_id"]).agg(
        euclidean_mm=("euclidean_mm", "mean"), horizon=("horizon_from_first_prefix_years", "first")).reset_index()
    base = baselines[(baselines.task == "one_shot_first")]
    merged = per_subject.merge(base, on=["view", "representation", "subject_id"], how="left")
    merged["horizon_bin"] = pd.cut(merged.horizon, [0, 1, 2, 3, 99], labels=["<=1 y", "1-2 y", "2-3 y", ">3 y"])
    grouped = merged.groupby(["method", "horizon_bin"], observed=True).agg(subjects=("subject_id", "size"), euclidean_mean=("euclidean_mm", "mean"),
                                                                           nochange_mean=("nochange_mm", "mean")).reset_index()
    grouped["skill"] = 1.0 - grouped.euclidean_mean / grouped.nochange_mean
    return grouped


def table_calsnic(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    part = rows[(rows.task == "one_shot_first") & rows.view.isin(["p1_external_calsnic_wholecohort", "p4c_exp2_pooled_to_calsnic", "p2_internal_calsnic"])]
    for (view, override, method, representation), cell in part.groupby(["view", "override", "method", "representation"]):
        per_subject = cell.groupby(["subject_id", "diagnosis"])[["predicted_log_volume_rate", "observed_log_volume_rate", "euclidean_mm"]].mean().reset_index()
        record = {"view": view, "condition": "labels (ALS = disease)" if override == -1 else f"all d = {override}", "method": method, "representation": representation}
        for label, name in (("AD", "als"), ("CN", "control")):
            group = per_subject[per_subject.diagnosis == label]
            record.update({f"{name}_subjects": len(group), f"{name}_predicted_rate": group.predicted_log_volume_rate.mean(),
                           f"{name}_observed_rate": group.observed_log_volume_rate.mean(), f"{name}_euclidean_mm": group.euclidean_mm.mean()})
        out.append(record)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------------------
# gates, figure, markdown
# --------------------------------------------------------------------------------------


def gates(rows: pd.DataFrame, missing: list[str], index: dict) -> list[dict]:
    out = [{"id": "G4.1", "check": "every non-blocked training run, external and transfer evaluation has a test summary",
            "passed": not missing, "detail": f"{len(missing)} missing"}]
    problems = []
    for held in ("adni", "aibl", "oasis"):
        for split in ("train", "val"):
            archive = bc.load_npz(bc.STAGE1_ROOT / "views" / f"p4_loco_without_{held}" / "dataset" / f"{split}_subject_sequences.npz")
            if held in set(archive["visit_cohorts"].astype(str)):
                problems.append(f"{held} in LOCO {split}")
    calsnic = bc.load_npz(bc.STAGE1_ROOT / "views" / "p2_internal_calsnic" / "dataset" / "train_subject_sequences.npz")
    if set(calsnic["visit_cohorts"].astype(str)) != {"calsnic"}:
        problems.append("CALSNIC internal training contains other cohorts")
    out.append({"id": "G4.2", "check": "LOCO held-out cohorts absent from train/val; CALSNIC ALS head trained on CALSNIC only", "passed": not problems, "detail": problems})
    oof = []
    cf = rows[rows.view.str.startswith("p2_crossfit_") & (rows.task == "one_shot_first")] if not rows.empty else rows
    for cohort, expected in (("aibl", 148), ("oasis", 385)):
        for (method, representation), cell in cf[cf.view.str.contains(f"_{cohort}_")].groupby(["method", "representation"]):
            if len(cell) != expected or cell.subject_id.nunique() != expected:
                oof.append(f"{cohort}/{representation}/{method}: {len(cell)} rows, {cell.subject_id.nunique()} subjects")
    out.append({"id": "G4.3", "check": "cross-fit gives every subject exactly one out-of-fold prediction", "passed": not oof, "detail": oof})
    leaked = []
    for run in index["trained_runs"]:
        status = Path(run["run_dir"]) / "training_status.json"
        if not run["blocked"] and status.is_file() and bc.read_json(status).get("test_data_loaded"):
            leaked.append(run["key"])
    out.append({"id": "G4.4", "check": "no training run loaded test data", "passed": not leaked, "detail": leaked})
    return out


def figure_interval_shift(table: pd.DataFrame, path: Path) -> bool:
    if table.empty:
        return False
    plt = R3.style()
    order = ["<=1 y", "1-2 y", "2-3 y", ">3 y"]
    # The faithful Latent ODE is tens of points below no-change, which flattens the other four near zero,
    # so the right panel zooms in without it.
    fig, (full, zoom) = plt.subplots(1, 2, figsize=(9.6, 4.0), gridspec_kw={"width_ratios": [1.0, 1.35]})
    ends = []
    for method, color in zip(METHODS, SLOTS):
        line = table[table.method == method].set_index("horizon_bin").reindex(order)
        style = dict(color=color, linewidth=1.5, marker="o", markersize=4.5, markeredgecolor=R3.INK["surface"], markeredgewidth=1.0)
        full.plot(range(len(order)), 100 * line.skill, label=R3.METHOD_LABEL[method], **style)
        if method == "latent_ode":
            continue
        zoom.plot(range(len(order)), 100 * line.skill, **style)
        last = line.skill.dropna()
        if len(last):
            ends.append((100 * last.iloc[-1], R3.METHOD_LABEL[method], order.index(last.index[-1])))
    low, high = zoom.get_ylim()
    gap = 0.075 * (high - low)
    placed = []
    for y, label, x in sorted(ends):
        y_text = y if not placed or y - placed[-1] >= gap else placed[-1] + gap
        placed.append(y_text)
        leader = dict(arrowstyle="-", color=R3.INK["baseline"], linewidth=0.6) if abs(y_text - y) > 1e-9 else None
        zoom.annotate(label, (x, y), xytext=(x + 0.15, y_text), textcoords="data", va="center", fontsize=7,
                      color=R3.INK["secondary"], arrowprops=leader)
    if placed:
        zoom.set_ylim(low, max(high, placed[-1] + gap))
    faithful = 100 * table[table.method == "latent_ode"].skill.dropna()
    omitted = f" ({faithful.min():.0f}% to {faithful.max():.0f}%)" if len(faithful) else ""
    for ax, title in ((full, "All five models"), (zoom, f"Zoom: faithful Latent ODE omitted{omitted}")):
        ax.axhline(0.0, color=R3.INK["baseline"], linewidth=0.8)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order)
        ax.set_title(title, fontsize=8)
    full.set_ylabel("skill vs no-change (%)")
    full.legend(loc="lower right", fontsize=7)
    zoom.set_xlim(-0.3, len(order) - 1 + 1.3)
    fig.supxlabel("first-to-last horizon (ADNI test + AIBL, OASIS, CALSNIC whole cohorts, zero-shot)", fontsize=8)
    fig.suptitle("Where transported predictions beat no-change, by horizon", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def cell_text(record: pd.Series) -> str:
    skill = f", skill {100 * record.skill:+.1f}%" if "skill" in record and pd.notna(record.get("skill")) else ""
    return f"{record.euclidean_mean:.4f}{skill} (n={int(record.subjects)})"


def markdown(gate_list, per_dataset, unified, cross, gaps, interval, fidelity_note, calsnic, missing, figures) -> str:
    lines = [f"# Stage 4 - cross-cohort protocols{' - PARTIAL DRAFT' if missing else ''}", "",
             "one_shot_first per-vertex Euclidean error (mm); skill = 1 - error / no-change error on the same subjects and representation. "
             "Multi-seed protocols (P0, P1, P3) average each subject over seeds first.",
             "The cocycle is not estimable on OASIS-only training views (no AD subject with >= 3 training visits), so those cells are absent.", "",
             "## Gates", "", "| gate | check | result |", "|---|---|---|"]
    lines += [f"| {g['id']} | {g['check']} | {'PASS' if g['passed'] else 'FAIL'} |" for g in gate_list]
    for cohort in COHORTS:
        part = per_dataset[per_dataset.cohort == cohort] if not per_dataset.empty else per_dataset
        if part.empty:
            continue
        part = part.assign(model=[f"{R3.METHOD_LABEL[m]} · {R3.REP_LABEL[r]}" for m, r in zip(part.method, part.representation)],
                           cell=[cell_text(r) for _, r in part.iterrows()])
        pivot = part.pivot_table(index="model", columns="scope", values="cell", aggfunc="first")
        scopes_present = [s for s in dict.fromkeys(label for label, c, _, _ in scopes() if c == cohort) if s in pivot.columns]
        lines += ["", f"## S4-A. {cohort.upper()} (BrainODE Table 7 analog)", "", "| model | " + " | ".join(scopes_present) + " |", "|" + "---|" * (len(scopes_present) + 1)]
        for model in pivot.index:
            lines.append(f"| {model} | " + " | ".join(str(pivot.loc[model].get(s)) if isinstance(pivot.loc[model].get(s), str) else "-" for s in scopes_present) + " |")
    if not cross.empty:
        lines += ["", "## S4-D. Cross-benchmark (BrainODE Table 8 analog)", "", "| benchmark | model | error | skill | n |", "|---|---|---|---|---|"]
        for _, r in cross.sort_values(["benchmark", "method", "representation"]).iterrows():
            skill = f"{100 * r.skill:+.1f}%" if pd.notna(r.get("skill")) else "-"
            lines.append(f"| {r.benchmark} | {R3.METHOD_LABEL[r.method]} · {R3.REP_LABEL[r.representation]} | {r.euclidean_mean:.4f} | {skill} | {int(r.subjects)} |")
    if not unified.empty:
        lines += ["", "## S4-C. Unified pooled test (BrainODE Table 2 analog)", "", "| model | task | error | n |", "|---|---|---|---|"]
        for _, r in unified.sort_values(["task", "method", "representation"]).iterrows():
            lines.append(f"| {R3.METHOD_LABEL[r.method]} · {R3.REP_LABEL[r.representation]} | {r.task} | {r.euclidean_mean:.4f} ± {r.euclidean_sd:.4f} | {int(r.subjects)} |")
    if not gaps.empty:
        lines += ["", "## S4-E. Transfer and pooling gaps (same test-split subjects)", "", "> 0 means the first protocol has larger error.", "",
                  "| cohort | comparison | model | mean difference mm [95% CI] | Wilcoxon p | n |", "|---|---|---|---|---|---|"]
        for _, r in gaps.iterrows():
            lines.append(f"| {r.cohort} | {r.comparison} | {R3.METHOD_LABEL[r.method]} · {R3.REP_LABEL[r.representation]} | "
                         f"{r.mean_difference:+.4f} [{r.ci95_low:+.4f}, {r.ci95_high:+.4f}] | {r.wilcoxon_p:.3g} | {int(r.n)} |")
    if not interval.empty:
        lines += ["", "## S4-F. Error and skill by horizon (zero-shot cohorts + ADNI test)", "", "| model | horizon | error | skill | n |", "|---|---|---|---|---|"]
        for _, r in interval.iterrows():
            lines.append(f"| {R3.METHOD_LABEL[r.method]} | {r.horizon_bin} | {r.euclidean_mean:.4f} | {100 * r.skill:+.1f}% | {int(r.subjects)} |")
    lines += ["", "## S4-G. Disease fidelity outside ADNI", "", fidelity_note]
    if not calsnic.empty:
        lines += ["", "## S4-H. CALSNIC out-of-distribution (ALS is not Alzheimer's)", "",
                  "| training | condition fed | model | ALS predicted / observed rate | Control predicted / observed rate |", "|---|---|---|---|---|"]
        for _, r in calsnic.sort_values(["view", "condition", "method", "representation"]).iterrows():
            lines.append(f"| {r.view} | {r.condition} | {R3.METHOD_LABEL[r.method]} · {R3.REP_LABEL[r.representation]} | "
                         f"{r.als_predicted_rate:.4f} / {r.als_observed_rate:.4f} | {r.control_predicted_rate:.4f} / {r.control_observed_rate:.4f} |")
    lines += ["", "## Figures", ""] + [f"![{f}](figures/{f})" for f in figures] + ["", "Tables in `tables/`. Colors: dataviz reference categorical slots 1-5 in fixed order, "
              "the left panel has a legend; the right panel zooms in without the off-scale faithful Latent ODE and carries direct end labels "
              "spread apart so they never overlap (the slot-3/4/5 contrast relief rule)."]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--device", default=None, help="Ignored; accepted because the orchestrator appends it.")
    args = parser.parse_args()
    rows, missing, index = load_rows(args.allow_partial)
    baselines = load_baselines(set(rows.view.unique()) if not rows.empty else set())
    per_dataset = table_per_dataset(rows, baselines) if not rows.empty else pd.DataFrame()
    unified = table_unified(rows) if not rows.empty else pd.DataFrame()
    cross = table_cross_benchmark(rows, baselines) if not rows.empty else pd.DataFrame()
    gaps = table_gaps(rows) if not rows.empty else pd.DataFrame()
    interval = table_interval_shift(rows, baselines) if not rows.empty else pd.DataFrame()
    calsnic = table_calsnic(rows) if not rows.empty else pd.DataFrame()
    fidelity = per_dataset[per_dataset.cohort.isin(["aibl", "oasis"])] if not per_dataset.empty else per_dataset
    fidelity_cols = [c for c in ("cohort", "scope", "method", "representation", "ad_subjects", "ad_capture", "cn_subjects", "cn_capture") if c in fidelity.columns]
    fidelity_note = ("Capture = mean predicted / mean observed log-volume rate, shown only for groups with >= 3 subjects; "
                     "see `tables/s4g_ad_fidelity.csv`. OASIS has 6 AD subjects in total, so its AD values are descriptive only.")
    gate_list = gates(rows, missing, index)
    tables = bc.require_bulk(REPORT / "tables")
    for name, table in (("s4a_per_dataset", per_dataset), ("s4c_unified_pooled", unified), ("s4d_cross_benchmark", cross), ("s4e_gaps", gaps),
                        ("s4f_interval_shift", interval), ("s4g_ad_fidelity", fidelity[fidelity_cols] if fidelity_cols else fidelity), ("s4h_calsnic_ood", calsnic)):
        bc.atomic_csv(tables / f"{name}.csv", table)
    figures = ["s4f_interval_shift.png"] if figure_interval_shift(interval, REPORT / "figures" / "s4f_interval_shift.png") else []
    bc.atomic_json(REPORT / "stage4_gates.json", {"partial": bool(missing), "missing": missing[:200], "gates": gate_list})
    name = "stage4_crosscohort_report_partial.md" if missing else "stage4_crosscohort_report.md"
    bc.atomic_write_text(REPORT / name, markdown(gate_list, per_dataset, unified, cross, gaps, interval, fidelity_note, calsnic, missing, figures))
    print(f"wrote {REPORT / name}: {len(missing)} evaluations missing; gates " + ", ".join(f"{g['id']}={'PASS' if g['passed'] else 'FAIL'}" for g in gate_list))
    return 0 if (all(g["passed"] for g in gate_list) or args.allow_partial) else 1


if __name__ == "__main__":
    raise SystemExit(main())
