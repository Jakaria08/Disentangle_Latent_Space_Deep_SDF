#!/usr/bin/env python3
"""Stage 5 step 3: the consolidated BrainODE-style report (PLAN Part 5A).

Built only from stored results (stage 1 views and baselines, stage 3-5 evaluation summaries and per-subject task
rows); no model is run. Output is deterministic: fixed bootstrap seeds, sorted groupings, no timestamps.

Tables (reports/tables):
  r_t5_composition, r_t6_evaluable      cohort composition and evaluable subjects per task
  r_t7_per_dataset, r_t1_aibl_regular   per-dataset and AIBL-regular tables (one-shot and all-prior-k)
  r_t2_unified, r_t8_cross_benchmark    unified pooled test and cross-benchmark transfer
  r_t3_ablations, r_t3_converter        ablations A1-A4 and the converter line
  endpoints_e1_e4                       pre-registered endpoints with CIs and the decision rule
  fidelity, consistency_cost            disease fidelity; defects, training time, best epochs
  sensitivity_age_subset, sensitivity_min_epoch, sensitivity_pooled_pca
  context_brainode_published            BrainODE's published numbers beside ours (not head-to-head)
  traceability                          every summary used: path, sha256, checkpoint sha256, git commit
Figures (reports/figures): condition effects, error vs shots, skill vs horizon, paired endpoint forest, AD slope scatter.
Gates: G5.2 (10 random R-T7 cells match raw summaries) and G5.3 (E1-E4 reported with CI and decision outcome) in
gates.json; G5.1 (byte-identical regeneration) with --verify, written to verification.json. Converter gates
G5.4-G5.7 are collected from their runs.
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import benchmark_common as bc
import benchmark_stats as ST
import converter_line as L
import stage3_report_adni as R3
import stage4_report_crosscohort as R4

STAGE5_ROOT = bc.BULK_ROOT / "stage5_brainode_style"
REPORT = STAGE5_ROOT / "reports"
METHODS = R3.METHODS
REPS = tuple(R3.REP_LABEL)
SLOTS = R4.SLOTS
DRAWS, SEED = 2000, 12345
CONVERTER = bc.read_json(bc.CONFIG_DIR / "converter_line.json")
PUBLISHED = [  # BrainODE (NeurIPS 2025), transcribed in PLAN section 0.2; different template, fitting, PCA-150 and subjects
    {"table": "T7", "setting": "AIBL", "metric": "4-shot / 1-shot Euclidean (mm)", "brainode": "0.52 ± 0.07 / 0.46 ± 0.05"},
    {"table": "T7", "setting": "ADNI", "metric": "4-shot / 1-shot Euclidean (mm)", "brainode": "0.59 ± 0.14 / 0.54 ± 0.15"},
    {"table": "T7", "setting": "OASIS", "metric": "4-shot / 1-shot Euclidean (mm)", "brainode": "0.52 ± 0.07 / 0.48 ± 0.08"},
    {"table": "T8", "setting": "Exp1 AIBL -> ADNI+OASIS", "metric": "Euclidean (mm)", "brainode": "0.518"},
    {"table": "T8", "setting": "Exp2 AIBL+ADNI+OASIS -> LBC1936", "metric": "Euclidean (mm)", "brainode": "0.522"},
    {"table": "T3", "setting": "shape accuracy NC&AD / CONV", "metric": "Euclidean (mm)", "brainode": "0.606 / 0.216"},
    {"table": "T3", "setting": "cognition estimator", "metric": "accuracy", "brainode": "0.891"},
]


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------


class Sources:
    """Every summary the report reads, for the traceability table."""

    def __init__(self) -> None:
        self.entries: dict[str, set[str]] = {}

    def add(self, summary: Path, used_in: str) -> None:
        self.entries.setdefault(str(summary), set()).add(used_in)

    def table(self) -> pd.DataFrame:
        records = []
        for path, used in sorted(self.entries.items()):
            summary = bc.read_json(path)
            records.append({"summary": path, "summary_sha256": bc.sha256_file(path), "checkpoint": summary.get("checkpoint", summary.get("run_dir", "")),
                            "checkpoint_sha256": summary.get("checkpoint_sha256", summary.get("estimator_checkpoint_sha256", "")),
                            "git_commit": summary.get("git_commit", ""), "used_in": ";".join(sorted(used))})
        return pd.DataFrame(records)


def rows_at(output: Path, sources: Sources, used_in: str) -> pd.DataFrame | None:
    if not (output / "summary.json").is_file() or not (output / "task_rows.csv").is_file():
        return None
    sources.add(output / "summary.json", used_in)
    rows = pd.read_csv(output / "task_rows.csv", dtype={"subject_id": str})
    return rows[rows.variant == "averaged"] if "variant" in rows else rows


def ratio_ci(predicted: np.ndarray, observed: np.ndarray) -> tuple[float, float, float]:
    predicted, observed = np.asarray(predicted, float), np.asarray(observed, float)
    if len(predicted) < 2 or abs(observed.mean()) < 1e-12:
        return float("nan"), float("nan"), float("nan")
    index = np.random.default_rng(SEED).integers(0, len(predicted), size=(DRAWS, len(predicted)))
    ratios = predicted[index].mean(axis=1) / observed[index].mean(axis=1)
    return float(predicted.mean() / observed.mean()), float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))


def capture_distance_difference(pred_a: np.ndarray, pred_b: np.ndarray, observed: np.ndarray) -> tuple[float, float, float]:
    """|1 - capture_a| - |1 - capture_b| on the same AD subjects, with a paired subject bootstrap (< 0: a is closer to 100%)."""
    pred_a, pred_b, observed = (np.asarray(v, float) for v in (pred_a, pred_b, observed))
    distance = lambda p, o: np.abs(1.0 - p.mean(axis=-1) / o.mean(axis=-1))
    point = float(distance(pred_a, observed) - distance(pred_b, observed))
    index = np.random.default_rng(SEED).integers(0, len(observed), size=(DRAWS, len(observed)))
    samples = distance(pred_a[index], observed[index]) - distance(pred_b[index], observed[index])
    return point, float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def subject_means(frame: pd.DataFrame, extra: tuple[str, ...] = ()) -> pd.DataFrame:
    keys = ["method", "representation", "subject_id", "diagnosis", *extra]
    return frame.groupby(keys)[["euclidean_mm", "predicted_log_volume_rate", "observed_log_volume_rate"]].mean().reset_index()


def fmt_ci(mean: float, low: float, high: float, digits: int = 4) -> str:
    return f"{mean:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]" if np.isfinite(mean) else "-"


def md(frame: pd.DataFrame, columns: list[str] | None = None, headers: list[str] | None = None, floats: int = 4) -> list[str]:
    if frame is None or frame.empty:
        return ["(no rows)"]
    columns = columns or list(frame.columns)
    headers = headers or columns
    render = lambda v: f"{v:.{floats}f}" if isinstance(v, (float, np.floating)) and np.isfinite(v) else ("-" if isinstance(v, (float, np.floating)) else str(v))
    return ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] + ["| " + " | ".join(render(row[c]) for c in columns) + " |" for _, row in frame.iterrows()]


def label(method: str, representation: str | None = None) -> str:
    name = R3.METHOD_LABEL.get(method, method)
    return f"{name} · {R3.REP_LABEL.get(representation, representation)}" if representation else name


# --------------------------------------------------------------------------------------
# composition
# --------------------------------------------------------------------------------------


def view_archives(view: str) -> dict[str, dict[str, np.ndarray]]:
    root = bc.STAGE1_ROOT / "views" / view / "dataset"
    return {split: bc.load_npz(root / f"{split}_subject_sequences.npz") for split in bc.SPLITS}


def table_composition() -> pd.DataFrame:
    records = []
    for cohort, view, disease in (("ADNI", "p0_internal_adni", "AD"), ("AIBL", "p2_internal_aibl", "AD"), ("OASIS", "p2_internal_oasis", "AD"),
                                  ("CALSNIC", "p2_internal_calsnic", "ALS")):
        archives = view_archives(view).values()
        labels = np.concatenate([a["subject_label_ad"] for a in archives])
        ages = np.concatenate([a["subject_baseline_age_years"] for a in archives]).astype(float)
        visits = np.concatenate([np.diff(a["subject_visit_offsets"]) for a in archives]).astype(float)
        intervals = np.concatenate([np.diff(a["visit_time_years_from_baseline"][a["subject_visit_offsets"][i]:a["subject_visit_offsets"][i + 1]])
                                    for a in archives for i in range(len(a["subject_ids"]))]).astype(float)
        records.append({"cohort": cohort, "subjects": len(labels), "control": int((labels == 0).sum()), "disease": f"{int((labels == 1).sum())} {disease}",
                        "scans": int(visits.sum()), "age_first_visit": f"{ages.mean():.1f} ± {ages.std(ddof=1):.1f}",
                        "observations": f"{visits.mean():.2f} ± {visits.std(ddof=1):.2f}", "interval_years": f"{intervals.mean():.2f} ± {intervals.std(ddof=1):.2f}"})
    converter = view_archives(CONVERTER["view"])
    groups = pd.Series(np.concatenate([a["subject_trajectory_groups"][a["subject_cohorts"] != "adni"] for a in converter.values()]))
    records.append({"cohort": "AIBL+OASIS inclusive (converter line)", "subjects": int(len(groups)), "control": int((groups == "CN-stable").sum()),
                    "disease": ", ".join(f"{int((groups == g).sum())} {g}" for g in ("AD-stable", "CN->AD", "MCI->AD", "CN->MCI")),
                    "scans": int(sum((a["visit_cohorts"] != "adni").sum() for a in converter.values())), "age_first_visit": "-", "observations": "-", "interval_years": "-"})
    return pd.DataFrame(records)


def table_evaluable() -> pd.DataFrame:
    tasks = list(bc.load_tasks()["tasks"])
    records = []
    for name, view in (("P0 ADNI test", "p0_internal_adni"), ("P1 AIBL whole cohort", "p1_external_aibl_wholecohort"),
                       ("P1 OASIS whole cohort", "p1_external_oasis_wholecohort"), ("P1 CALSNIC whole cohort", "p1_external_calsnic_wholecohort"),
                       ("P3 unified test", "p3_pooled"), ("P5 converter test", CONVERTER["view"])):
        manifest = bc.read_json(bc.STAGE1_ROOT / "views" / view / "view_manifest.json")
        records.append({"set": name, **{task: manifest["tasks"].get(f"test/{task}", {}).get("subjects", 0) for task in tasks}})
    return pd.DataFrame(records)


# --------------------------------------------------------------------------------------
# endpoints (pre-registered E1-E4 and the decision rule)
# --------------------------------------------------------------------------------------


def table_endpoints(rows: pd.DataFrame, baselines: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    records, decisions = [], []
    p0 = subject_means(rows[(rows.protocol == "P0") & (rows.task == "one_shot_first")])
    for representation in REPS:
        cell = p0[p0.representation == representation]
        nochange = float(baselines[(baselines.view == "p0_internal_adni") & (baselines.task == "one_shot_first")
                                   & (baselines.representation == representation)].nochange_mm.mean())
        cocycle = cell[cell.method == "direct_c4"].set_index("subject_id")
        outcome = {"representation": representation, "nochange_mm": nochange, "margin_mm": 0.02 * nochange, "preferred_over_cocycle": []}
        for method in METHODS:
            model = cell[cell.method == method].set_index("subject_id")
            if model.empty:
                continue
            ad = model[model.diagnosis == "AD"]
            e1_low, e1_high = ST.bootstrap_mean_ci(model.euclidean_mm, DRAWS, SEED)
            e2, e2_low, e2_high = ratio_ci(ad.predicted_log_volume_rate, ad.observed_log_volume_rate)
            record = {"endpoint_set": "P0 ADNI test", "representation": representation, "method": method, "subjects": len(model), "ad_subjects": len(ad),
                      "E1_mm": float(model.euclidean_mm.mean()), "E1_ci": f"[{e1_low:.4f}, {e1_high:.4f}]",
                      "E2_capture": e2, "E2_ci": f"[{e2_low:.3f}, {e2_high:.3f}]"}
            if method != "direct_c4" and not cocycle.empty:
                aligned = model.join(cocycle, rsuffix="_c", how="inner")
                d1 = ST.paired_difference(aligned.euclidean_mm, aligned.euclidean_mm_c, DRAWS, SEED)
                ad_aligned = aligned[aligned.diagnosis == "AD"]
                d2, d2_low, d2_high = capture_distance_difference(ad_aligned.predicted_log_volume_rate, ad_aligned.predicted_log_volume_rate_c, ad_aligned.observed_log_volume_rate)
                non_inferior = d1["ci95_high"] <= 0.02 * nochange
                improves = d2_high < 0.0
                record.update({"dE1_vs_cocycle_mm": d1["mean_difference"], "dE1_ci": f"[{d1['ci95_low']:.4f}, {d1['ci95_high']:.4f}]",
                               "E1_non_inferior": bool(non_inferior), "dE2_distance_to_100pct": d2, "dE2_ci": f"[{d2_low:.3f}, {d2_high:.3f}]",
                               "E2_improves": bool(improves), "preferred_over_cocycle": bool(non_inferior and improves)})
                if non_inferior and improves:
                    outcome["preferred_over_cocycle"].append(method)
            records.append(record)
        ranking = cell.groupby("method").euclidean_mm.mean().sort_values()
        outcome["ranking_by_E1"] = [f"{m} ({v:.4f})" for m, v in ranking.items()]
        outcome["decision"] = (f"preferred over the cocycle: {', '.join(outcome['preferred_over_cocycle'])}" if outcome["preferred_over_cocycle"]
                               else "no method is preferred over the cocycle; ranking follows E1")
        decisions.append(outcome)
    for name, select, task in (("E3 P3 AIBL all-prior-k", (rows.view == "p3_pooled") & (rows.cohort == "aibl"), "all_prior_k"),
                               ("E4 P1 AIBL whole cohort one-shot", (rows.view == "p1_external_aibl_wholecohort") & (rows.override == -1), "one_shot_first")):
        part = subject_means(rows[select & (rows.task == task)])
        for (method, representation), cell in part.groupby(["method", "representation"]):
            low, high = ST.bootstrap_mean_ci(cell.euclidean_mm, DRAWS, SEED)
            records.append({"endpoint_set": name, "representation": representation, "method": method, "subjects": len(cell),
                            "E1_mm": float(cell.euclidean_mm.mean()), "E1_ci": f"[{low:.4f}, {high:.4f}]"})
    return pd.DataFrame(records), decisions


# --------------------------------------------------------------------------------------
# ablations and sensitivity
# --------------------------------------------------------------------------------------


def run_metrics(rows: pd.DataFrame, summary: dict[str, Any]) -> dict[str, float]:
    part = rows[rows.task == "one_shot_first"]
    ad = part[part.diagnosis == "AD"]
    return {"euclidean_mm": float(part.euclidean_mm.mean()), "ad_capture": float(ad.predicted_log_volume_rate.mean() / ad.observed_log_volume_rate.mean()),
            "semigroup_defect": summary.get("consistency_defects", {}).get("relative_semigroup_defect_mean", np.nan)}


def table_ablations(sources: Sources, stage3_runs: list[dict[str, Any]]) -> pd.DataFrame:
    index = bc.read_json(STAGE5_ROOT / "results_index.json")
    records = []
    for run in index["ablation_runs"]:
        output = Path(run["run_dir"]) / "evaluation__p0_internal_adni" / "test"
        rows = rows_at(output, sources, "r_t3_ablations")
        if rows is None:
            continue
        sweep_path = STAGE5_ROOT / "ablations" / "condition_sweeps" / f"{run['key']}.json"
        sweep = bc.read_json(sweep_path) if sweep_path.is_file() else {}
        status = bc.read_json(Path(run["run_dir"]) / "training_status.json")
        records.append({"row": f"{run['ablation']} {run['method']}", "representation": run["representation"], "seed": run["seed"],
                        **run_metrics(rows, bc.read_json(output / "summary.json")), "condition_share_of_gap": sweep.get("condition_effect_over_observed_gap", np.nan),
                        "epochs_run": status.get("epoch", np.nan), "best_epoch": status.get("best_epoch", np.nan), "train_minutes": status.get("elapsed_minutes", np.nan)})
    for run in stage3_runs:
        if run["representation"] != "pca128" and not (run["method"] == "direct_c4" and run["representation"] == "spiralnet128"):
            continue
        summary = run["summary"]["test"]
        rows = run["test_rows"][run["test_rows"].variant == "averaged"]
        records.append({"row": f"reference {run['method']}", "representation": run["representation"], "seed": run["seed"], **run_metrics(rows, summary),
                        "condition_share_of_gap": run.get("sweep", {}).get("condition_effect_over_observed_gap", np.nan),
                        "epochs_run": run["status"].get("epoch", np.nan), "best_epoch": run["status"].get("best_epoch", np.nan),
                        "train_minutes": run["status"].get("elapsed_minutes", np.nan)})
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    numeric = ["euclidean_mm", "ad_capture", "semigroup_defect", "condition_share_of_gap", "epochs_run", "best_epoch", "train_minutes"]
    grouped = frame.groupby(["row", "representation"])
    table = grouped[numeric].mean().join(grouped.size().rename("seeds")).join(grouped[["euclidean_mm", "ad_capture"]].std(ddof=1).add_suffix("_seed_sd"))
    return table.reset_index()


def table_age_subset(rows: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    ages = {}
    for view in sorted(set(rows.view)):
        archive = bc.load_npz(bc.STAGE1_ROOT / "views" / view / "dataset" / "test_subject_sequences.npz")
        ages[view] = dict(zip(archive["subject_ids"].astype(str), archive["subject_baseline_age_years"].astype(float)))
    frame = rows[(rows.task == "one_shot_first") & (rows.override == -1)].copy()
    frame["age_first_visit"] = [ages[v].get(s, np.nan) for v, s in zip(frame.view, frame.subject_id)]
    scopes = [("P0 ADNI test", "adni", frame.view == "p0_internal_adni")]
    scopes += [(f"P1 {c.upper()} whole cohort", c, frame.view == f"p1_external_{c}_wholecohort") for c in ("aibl", "oasis", "calsnic")]
    scopes += [(f"P3 {c.upper()} test", c, (frame.view == "p3_pooled") & (frame.cohort == c)) for c in ("adni", "aibl", "oasis")]
    records = []
    for name, cohort, select in scopes:
        part = frame[select]
        for (method, representation), cell in part.groupby(["method", "representation"]):
            per_subject = cell.groupby(["view", "subject_id", "age_first_visit"]).euclidean_mm.mean().reset_index()
            base = baselines[(baselines.task == "one_shot_first") & (baselines.representation == representation)]
            per_subject = per_subject.merge(base, on=["view", "subject_id"], how="left")
            subset = per_subject[per_subject.age_first_visit.between(65, 95)]
            record = {"scope": name, "method": method, "representation": representation, "subjects_all": len(per_subject), "subjects_65_95": len(subset),
                      "euclidean_all": per_subject.euclidean_mm.mean(), "euclidean_65_95": subset.euclidean_mm.mean() if len(subset) else np.nan}
            if per_subject.nochange_mm.notna().all() and len(subset):
                record["skill_all_pct"] = 100 * (1 - per_subject.euclidean_mm.mean() / per_subject.nochange_mm.mean())
                record["skill_65_95_pct"] = 100 * (1 - subset.euclidean_mm.mean() / subset.nochange_mm.mean())
            records.append(record)
    return pd.DataFrame(records)


def table_min_epoch(sources: Sources, stage3_runs: list[dict[str, Any]]) -> pd.DataFrame:
    index = bc.read_json(STAGE5_ROOT / "results_index.json")
    records = []
    for run in index["min_epoch_runs"]:
        best = rows_at(Path(run["run_dir"]) / f"evaluation__{run['view']}" / "test", sources, "sensitivity_min_epoch")
        later = rows_at(Path(run["run_dir"]) / f"evaluation_min_epoch__{run['view']}" / "test", sources, "sensitivity_min_epoch")
        if best is None or later is None:
            continue
        for checkpoint, rows in (("best", best), ("min_epoch_15", later)):
            records.append({"protocol": run["protocol"], "view": run["view"], "representation": run["representation"], "seed": run["seed"], "checkpoint": checkpoint,
                            **{k: v for k, v in run_metrics(rows, {}).items() if k != "semigroup_defect"}})
    for run in stage3_runs:
        if "min_epoch_test_rows" not in run:
            continue
        for checkpoint, key in (("best", "test_rows"), ("min_epoch_15", "min_epoch_test_rows")):
            rows = run[key][run[key].variant == "averaged"]
            records.append({"protocol": "P0", "view": "p0_internal_adni", "representation": run["representation"], "seed": run["seed"], "checkpoint": checkpoint,
                            **{k: v for k, v in run_metrics(rows, {}).items() if k != "semigroup_defect"}})
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    wide = frame.pivot_table(index=["protocol", "representation"], columns="checkpoint", values=["euclidean_mm", "ad_capture"], aggfunc="mean")
    wide.columns = [f"{metric}_{checkpoint}" for metric, checkpoint in wide.columns]
    wide = wide.reset_index()
    wide["euclidean_difference_mm"] = wide["euclidean_mm_min_epoch_15"] - wide["euclidean_mm_best"]
    counts = frame[frame.checkpoint == "best"].groupby(["protocol", "representation"]).size().rename("runs").reset_index()
    return wide.merge(counts, on=["protocol", "representation"])


def table_pooled_pca(sources: Sources, rows: pd.DataFrame) -> pd.DataFrame:
    index = bc.read_json(STAGE5_ROOT / "results_index.json")
    frames = []
    for run in index["pooled_pca_runs"]:
        found = rows_at(Path(run["run_dir"]) / "evaluation__p3_pooled" / "test", sources, "sensitivity_pooled_pca")
        if found is not None:
            frames.append(found.assign(method=run["method"], seed=run["seed"]))
    if not frames:
        return pd.DataFrame()
    pooled = pd.concat(frames, ignore_index=True)
    r1 = rows[(rows.view == "p3_pooled") & (rows.representation == "pca128")]
    records = []
    for task in ("one_shot_first", "all_prior_k"):
        a = pooled[pooled.task == task].groupby(["method", "cohort", "subject_id"]).euclidean_mm.mean()
        b = r1[r1.task == task].groupby(["method", "cohort", "subject_id"]).euclidean_mm.mean()
        joined = pd.concat({"pooled": a, "r1": b}, axis=1).dropna().reset_index()
        for (method, cohort), cell in joined.groupby(["method", "cohort"]):
            result = ST.paired_difference(cell.pooled, cell.r1, DRAWS, SEED)
            records.append({"task": task, "method": method, "cohort": cohort, "pooled_mm": cell.pooled.mean(), "r1_adni_pca_mm": cell.r1.mean(), **result})
    return pd.DataFrame(records)


# --------------------------------------------------------------------------------------
# converter line
# --------------------------------------------------------------------------------------


def converter_rows(sources: Sources) -> pd.DataFrame:
    index_path = STAGE5_ROOT / "converter_index.json"
    if not index_path.is_file():
        return pd.DataFrame()
    index = bc.read_json(index_path)
    view = CONVERTER["view"]
    test = bc.load_npz(bc.STAGE1_ROOT / "views" / view / "dataset" / "test_subject_sequences.npz")
    group_of = dict(zip(test["subject_ids"].astype(str), test["subject_trajectory_groups"].astype(str)))
    frames = []
    for run in index["c1_runs"]:
        found = rows_at(Path(run["run_dir"]) / f"evaluation__{view}" / "test", sources, "r_t3_converter")
        if found is not None:
            frames.append(found.assign(model=[f"C1 {run['variant']} ({rule})" for rule in found["rule"]], representation=run["representation"], seed=run["seed"]))
    for run in index["comparator_runs"]:
        found = rows_at(Path(run["output_root"]) / "test", sources, "r_t3_converter")
        if found is not None:
            name = "C0 (source label)" if run["method"] == "direct_c4" else "BrainODE-core (source label)"
            frames.append(found.assign(model=name, representation=run["representation"], seed=run["seed"], group=found["subject_id"].astype(str).map(group_of)))
    for run in index["brainode_full_runs"]:
        found = rows_at(Path(run["run_dir"]) / f"evaluation__{view}" / "test", sources, "r_t3_converter")
        if found is not None:
            frames.append(found.assign(model="BrainODE-full (feedback)", representation=CONVERTER["brainode_full"]["representation"], seed=run["seed"]))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def table_converter(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    records = []
    for task in ("one_shot_first", "all_prior_k"):
        part = rows[rows.task == task]
        for (model, representation), cell in part.groupby(["model", "representation"]):
            per_subject = cell.groupby(["subject_id", "group"])[["euclidean_mm", "predicted_log_volume_rate", "observed_log_volume_rate"]].mean().reset_index()
            record = {"task": task, "model": model, "representation": representation, "seeds": int(cell.seed.nunique())}
            for name, members in (("CN-stable", ["CN-stable"]), ("AD-stable", ["AD-stable"]), ("CN->AD", ["CN->AD"]), ("MCI->AD", ["MCI->AD"]),
                                  ("stable", ["CN-stable", "AD-stable"]), ("AD converters", list(L.AD_CONVERTERS))):
                group = per_subject[per_subject.group.isin(members)]
                record[f"{name}_mm"] = group.euclidean_mm.mean() if len(group) else np.nan
                record[f"{name}_n"] = len(group)
            mci = per_subject[per_subject.group == "MCI->AD"]
            record["MCI->AD_capture"] = mci.predicted_log_volume_rate.mean() / mci.observed_log_volume_rate.mean() if len(mci) and abs(mci.observed_log_volume_rate.mean()) > 1e-12 else np.nan
            records.append(record)
    return pd.DataFrame(records)


def converter_gates(rows: pd.DataFrame) -> list[dict[str, Any]]:
    index_path = STAGE5_ROOT / "converter_index.json"
    if not index_path.is_file():
        return []
    index = bc.read_json(index_path)
    view = CONVERTER["view"]
    gaps, g55, missing = [], [], []
    for run in index["c1_runs"]:
        status_path, summary_path = Path(run["run_dir"]) / "training_status.json", Path(run["run_dir"]) / f"evaluation__{view}" / "test" / "summary.json"
        if not status_path.is_file() or not summary_path.is_file():
            missing.append(run["key"])
            continue
        gaps.append(bc.read_json(status_path).get("reduction_gap", np.nan))
        g55.append(bool(bc.read_json(summary_path)["gate_G5_5"]["passed"]))
    synthetic = {rep: bc.read_json(STAGE5_ROOT / "converter" / "synthetic_onset" / f"{rep}_s42.json")
                 for rep in CONVERTER["representations"] if (STAGE5_ROOT / "converter" / "synthetic_onset" / f"{rep}_s42.json").is_file()}
    # A gate over runs that do not all exist yet is PENDING (passed = None), never PASS or FAIL.
    complete = not missing
    out = [{"id": "G5.4", "check": "dose additivity <= 1e-12 (unit test) and C1 == direct_c4 on stable subjects <= 1e-6 (every run)",
            "passed": (max(gaps) <= 1e-6) if (gaps and complete) else (False if gaps and max(gaps) > 1e-6 else None),
            "detail": {"max_reduction_gap": max(gaps) if gaps else None, "runs": len(gaps), "missing": missing}},
           {"id": "G5.5", "check": "test evaluations never change or use learned onsets",
            "passed": (all(g55) if complete else (False if not all(g55) else None)) if g55 else None, "detail": {"runs": len(g55), "missing": len(missing)}},
           {"id": "G5.6", "check": "synthetic onset recovery median error <= 0.5 y", "passed": bool(synthetic) and all(s["gate_passed"] if "gate_passed" in s else s["gate"]["passed"] for s in synthetic.values()),
            "detail": {rep: {"median_abs_error_years": s["widths"][f"{CONVERTER['dose']['width_years']:g}"]["median_abs_error_years"],
                             "window_midpoint_error_years": s["widths"][f"{CONVERTER['dose']['width_years']:g}"]["median_midpoint_abs_error_years"]} for rep, s in synthetic.items()}}]
    detail, passed = {}, True
    if not rows.empty:
        stable = rows[(rows.task == "one_shot_first") & rows.group.isin(["CN-stable", "AD-stable"])]
        for representation in sorted(set(stable.representation)):
            c0 = stable[(stable.model == "C0 (source label)") & (stable.representation == representation)].euclidean_mm.mean()
            for model in sorted(m for m in set(stable.model) if m.startswith("C1") and m.endswith("(prefix)")):
                value = stable[(stable.model == model) & (stable.representation == representation)].euclidean_mm.mean()
                if np.isfinite(value) and np.isfinite(c0):
                    ok = value <= 1.01 * c0
                    passed &= ok
                    detail[f"{model} · {representation}"] = {"stable_mm": value, "c0_stable_mm": c0, "within_1pct": bool(ok)}
    out.append({"id": "G5.7", "check": "C1 stable-subject test error within 1% of the P3 cocycle (C0)",
                "passed": (passed if complete else (False if not passed else None)) if detail else None, "detail": detail})
    return out


# --------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------


def save(fig, plt, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    return path.name


def figure_condition_effects(stage3_sweeps: pd.DataFrame, ablations: pd.DataFrame, path: Path) -> str | None:
    if stage3_sweeps.empty:
        return None
    plt = R3.style()
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    effects = stage3_sweeps.set_index(["model", "representation"]).effect_over_observed_ad_cn_gap
    rows = [(f"{label(m)} · {R3.REP_LABEL[r]}", effects[(m, r)]) for m in METHODS for r in REPS if (m, r) in effects.index]
    if not ablations.empty:
        rows += [(f"{row.row} · {R3.REP_LABEL.get(row.representation, row.representation)}", row.condition_share_of_gap)
                 for row in ablations[ablations.row.str.startswith("A")].sort_values("row").itertuples()]
    rows = [(name, value) for name, value in rows if np.isfinite(value)]
    cap = 160.0  # the faithful Latent ODE reaches ~275% and would flatten every other model; off-scale values are labelled
    for y, (name, value) in enumerate(rows):
        color = SLOTS[1] if name.startswith("A") else SLOTS[0]
        shown = min(100 * value, cap)
        ax.plot([0, shown], [y, y], color=color, linewidth=1.5)
        ax.plot(shown, y, ">" if 100 * value > cap else "o", color=color, markersize=5, markeredgecolor=R3.INK["surface"], markeredgewidth=1.0)
        if 100 * value > cap:
            ax.annotate(f"{100 * value:.0f}%", (shown, y), xytext=(-4, 5), textcoords="offset points", ha="right", fontsize=6, color=R3.INK["secondary"])
    ax.set_xlim(-5, cap + 5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([name for name, _ in rows], fontsize=6)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.axvline(0.0, color=R3.INK["baseline"], linewidth=0.8)
    ax.axvline(100.0, color=R3.INK["baseline"], linewidth=0.8, linestyle=":")
    ax.set_xlabel("condition effect (d = 1 minus d = 0, 0-2 y) as % of the observed AD-CN rate gap (dotted: 100%)")
    ax.set_title("How much of the AD-CN atrophy gap the disease condition produces (ADNI test)", fontsize=9)
    handles = [plt.Line2D([], [], color=SLOTS[0], marker="o"), plt.Line2D([], [], color=SLOTS[1], marker="o")]
    ax.legend(handles, ["main models (stage 3)", "ablations A1-A4 (stage 5)"], loc="lower right", fontsize=7)
    return save(fig, plt, path)


def figure_error_vs_shots(errors: pd.DataFrame, path: Path) -> str | None:
    part = errors[errors.model.isin(METHODS)] if not errors.empty else errors
    if part.empty:
        return None
    plt = R3.style()
    tasks = ["one_shot_first", "all_prior_k", "four_shot"]
    faithful = part[(part.model == "latent_ode") & part.task.isin(tasks)].euclidean_mm_mean
    fig, axes = plt.subplots(1, len(REPS), figsize=(11, 3.4), sharey=True)
    for ax, representation in zip(axes, REPS):
        for method, color in zip(METHODS, SLOTS):
            if method == "latent_ode":
                continue  # 0.40-0.57 mm; shown in R-T3/R-T7, omitted so the four competitive models stay readable
            line = part[(part.representation == representation) & (part.model == method)].set_index("task").reindex(tasks)
            ax.plot(range(len(tasks)), line.euclidean_mm_mean, color=color, marker="o", markersize=4, markeredgecolor=R3.INK["surface"], label=label(method))
        ax.set_xticks(range(len(tasks)))
        ax.set_xticklabels(["1-shot", "all prior (k<=4)", "4-shot (CN)"])
        ax.set_title(R3.REP_LABEL[representation], fontsize=9)
    axes[0].set_ylabel("Euclidean error (mm), ADNI test")
    axes[-1].legend(fontsize=6, loc="upper right")
    fig.suptitle(f"Error vs number of observed visits (faithful Latent ODE omitted: {faithful.min():.2f}-{faithful.max():.2f} mm)", fontsize=10)
    fig.tight_layout()
    return save(fig, plt, path)


def figure_skill_by_horizon(interval: pd.DataFrame, path: Path) -> str | None:
    if interval.empty:
        return None
    plt = R3.style()
    order = ["<=1 y", "1-2 y", "2-3 y", ">3 y"]
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for method, color in zip(METHODS, SLOTS):
        if method == "latent_ode":
            continue
        line = interval[interval.method == method].set_index("horizon_bin").reindex(order)
        ax.plot(range(len(order)), 100 * line.skill, color=color, marker="o", markersize=4, markeredgecolor=R3.INK["surface"], label=label(method))
    ax.axhline(0.0, color=R3.INK["baseline"], linewidth=0.8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_xlabel("first-to-last horizon (ADNI test + AIBL, OASIS, CALSNIC whole cohorts)")
    ax.set_ylabel("skill vs no-change (%)")
    ax.set_title("Skill by horizon (faithful Latent ODE omitted: -63% to -33%)", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    return save(fig, plt, path)


def figure_endpoint_forest(endpoints: pd.DataFrame, path: Path) -> str | None:
    part = endpoints[(endpoints.endpoint_set == "P0 ADNI test") & endpoints.get("dE1_vs_cocycle_mm", pd.Series(dtype=float)).notna()] if "dE1_vs_cocycle_mm" in endpoints else pd.DataFrame()
    if part.empty:
        return None
    plt = R3.style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2), sharey=True)
    labels = [label(row.method, row.representation) for row in part.itertuples()]
    for ax, value, interval, xlabel in ((axes[0], "dE1_vs_cocycle_mm", "dE1_ci", "E1: error minus cocycle (mm); > 0 worse"),
                                        (axes[1], "dE2_distance_to_100pct", "dE2_ci", "E2: |1 - AD capture| minus cocycle's; < 0 closer to 100%")):
        for y, row in enumerate(part.itertuples()):
            low, high = (float(x) for x in getattr(row, interval).strip("[]").split(","))
            color = SLOTS[0] if row.preferred_over_cocycle else R3.INK["muted"]
            ax.plot([low, high], [y, y], color=color, linewidth=1.5)
            ax.plot(getattr(row, value), y, "o", color=color, markersize=5, markeredgecolor=R3.INK["surface"])
        ax.axvline(0.0, color=R3.INK["baseline"], linewidth=0.8)
        ax.set_xlabel(xlabel)
    axes[0].set_yticks(range(len(labels)))
    axes[0].set_yticklabels(labels, fontsize=6)
    axes[0].set_ylim(len(labels) - 0.5, -0.5)
    handles = [plt.Line2D([], [], color=SLOTS[0], marker="o"), plt.Line2D([], [], color=R3.INK["muted"], marker="o")]
    fig.legend(handles, ["preferred over the cocycle by the pre-registered rule", "not preferred"], loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    return save(fig, plt, path)


def figure_ad_slopes(rows: pd.DataFrame, path: Path) -> str | None:
    part = subject_means(rows[(rows.protocol == "P0") & (rows.task == "one_shot_first") & (rows.representation == "pca128")])
    part = part[part.diagnosis == "AD"]
    if part.empty:
        return None
    plt = R3.style()
    shown = [m for m in METHODS if m != "latent_ode"]
    fig, axes = plt.subplots(1, len(shown), figsize=(11, 3.2), sharex=True, sharey=True)
    for ax, method, color in zip(axes, shown, [SLOTS[0], SLOTS[1], SLOTS[2], SLOTS[4]]):
        cell = part[part.method == method]
        ax.plot(100 * cell.observed_log_volume_rate, 100 * cell.predicted_log_volume_rate, "o", color=color, markersize=4, markeredgecolor=R3.INK["surface"], markeredgewidth=0.8)
        limits = np.array([-15.0, 5.0])
        ax.plot(limits, limits, color=R3.INK["baseline"], linewidth=0.8)
        corr = ST.correlations(cell.observed_log_volume_rate, cell.predicted_log_volume_rate)
        ax.set_title(f"{label(method)}\nr = {corr['pearson_r']:.2f}", fontsize=8)
        ax.set_xlabel("observed %/y")
    axes[0].set_ylabel("predicted %/y (1-shot)")
    fig.suptitle("ADNI test AD subjects: predicted vs observed annual log-volume change (PCA-128, seed mean)", fontsize=9)
    fig.tight_layout()
    return save(fig, plt, path)


# --------------------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------------------


def gate_cells(per_dataset: pd.DataFrame, sources: Sources) -> dict[str, Any]:
    """G5.2: 10 seeded random R-T7 cells recomputed from the raw summary.json files."""
    index4, index3 = bc.read_json(bc.BULK_ROOT / "stage4_crosscohort" / "results_index.json"), bc.read_json(bc.BULK_ROOT / "stage3_adni" / "results_index.json")
    candidates = per_dataset[~per_dataset.scope.str.startswith("P2 cross-fit")].reset_index(drop=True)
    picks = np.random.default_rng(2026).choice(len(candidates), size=min(10, len(candidates)), replace=False)
    checks, passed = [], True
    for position in sorted(int(p) for p in picks):
        cell = candidates.loc[position]
        cohort, scope, method, representation = cell.cohort, cell.scope, cell.method, cell.representation
        group = "overall"
        if scope.startswith("P0"):
            outputs = [Path(r["evaluations"]["test"]) for r in index3["runs"] if r["representation"] == representation and r["method"] == method]
        elif scope.startswith("P1"):
            view = f"p1_external_{cohort}_{'wholecohort' if 'whole' in scope else 'testsplit'}"
            outputs = [Path(e["output"]) for e in index4["external_evaluations"] if e["view"] == view and e["condition_override"] is None
                       and e["representation"] == representation and e["method"] == method]
        elif scope.startswith("P4c"):
            outputs = [Path(e["output"]) for e in index4["transfer_evaluations"] if e["view"] == "p4c_exp2_pooled_to_calsnic" and e["condition_override"] is None
                       and e["representation"] == representation and e["method"] == method]
        else:
            view = {"P2 internal (test split)": f"p2_internal_{cohort}", "P3 pooled (test split)": "p3_pooled", "P4 LOCO (whole cohort)": f"p4_loco_without_{cohort}"}[scope]
            outputs = [Path(r["evaluations"]["test"]) for r in index4["trained_runs"] if r["view"] == view and not r["blocked"]
                       and r["representation"] == representation and r["method"] == method]
            group = f"cohort={cohort}" if view == "p3_pooled" else "overall"
        values = []
        for output in outputs:
            sources.add(output / "summary.json", "gate_G5.2")
            record = next(t for t in bc.read_json(output / "summary.json")["tasks"] if t["task"] == "one_shot_first" and t["variant"] == "averaged" and t["group"] == group)
            values.append(record["euclidean_mm_mean"])
        raw = float(np.mean(values)) if values else float("nan")
        ok = bool(values) and abs(raw - float(cell.euclidean_mean)) <= 1e-7
        passed &= ok
        checks.append({"cohort": cohort, "scope": scope, "method": method, "representation": representation, "table": float(cell.euclidean_mean),
                       "raw_summaries": raw, "summaries": len(values), "match": ok})
    return {"id": "G5.2", "check": "10 random R-T7 cells match their raw summary.json values (|diff| <= 1e-7 mm)", "passed": bool(checks) and passed, "detail": checks}


# --------------------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------------------


def build(output_root: Path, allow_partial: bool) -> dict[str, Any]:
    sources = Sources()
    tables, figures_dir = output_root / "tables", output_root / "figures"
    stage3_runs, stage3_missing = R3.load(allow_partial)
    for run in stage3_runs:
        sources.add(Path(run["evaluations"]["test"]) / "summary.json", "stage3")
    rows, stage4_missing, _index4 = R4.load_rows(allow_partial)
    baselines = R4.load_baselines(set(rows.view))
    frame3 = R3.task_frame(stage3_runs)

    composition, evaluable = table_composition(), table_evaluable()
    per_dataset = R4.table_per_dataset(rows, baselines, "one_shot_first")
    per_dataset_all_prior = R4.table_per_dataset(rows, baselines, "all_prior_k")
    aibl_regular = pd.concat([per_dataset.assign(task="one_shot_first"), per_dataset_all_prior.assign(task="all_prior_k")])
    aibl_regular = aibl_regular[(aibl_regular.cohort == "aibl") & aibl_regular.scope.str.match(r"P[123]")]
    unified, cross = R4.table_unified(rows), R4.table_cross_benchmark(rows, baselines)
    interval = R4.table_interval_shift(rows, baselines)
    endpoints, decisions = table_endpoints(rows, baselines)
    errors, fidelity = R3.table_errors(frame3), R3.table_fidelity(frame3)
    consistency, sweeps = R3.table_consistency(stage3_runs), R3.table_sweeps(stage3_runs)
    ablations = table_ablations(sources, stage3_runs)
    age_subset = table_age_subset(rows, baselines)
    min_epoch = table_min_epoch(sources, stage3_runs)
    pooled = table_pooled_pca(sources, rows)
    conv_rows = converter_rows(sources)
    converter = table_converter(conv_rows)
    external_fidelity = R4.table_per_dataset(rows, baselines, "one_shot_first")[["cohort", "scope", "method", "representation", "ad_subjects", "ad_capture", "cn_capture"]]
    context = pd.DataFrame(PUBLISHED)

    written = {
        "r_t5_composition": composition, "r_t6_evaluable": evaluable, "r_t7_per_dataset": per_dataset, "r_t1_aibl_regular": aibl_regular,
        "r_t2_unified": unified, "r_t8_cross_benchmark": cross, "r_t3_ablations": ablations, "r_t3_converter": converter,
        "endpoints_e1_e4": endpoints, "fidelity_adni": fidelity, "fidelity_external": external_fidelity, "consistency_cost": consistency,
        "sensitivity_age_subset": age_subset, "sensitivity_min_epoch": min_epoch, "sensitivity_pooled_pca": pooled, "context_brainode_published": context,
    }
    for name, table in written.items():
        bc.atomic_csv(tables / f"{name}.csv", table if table is not None else pd.DataFrame())

    figures = [f for f in (
        figure_condition_effects(sweeps, ablations, figures_dir / "f1_condition_effects.png"),
        figure_error_vs_shots(errors, figures_dir / "f2_error_vs_shots.png"),
        figure_skill_by_horizon(interval, figures_dir / "f3_skill_by_horizon.png"),
        figure_endpoint_forest(endpoints, figures_dir / "f4_endpoint_forest.png"),
        figure_ad_slopes(rows, figures_dir / "f5_ad_slopes.png"),
    ) if f]
    maps = REPORT / "figures" / "f6_error_maps.png"  # rendered by stage5_error_maps.py (needs model inference, so not rebuilt here)
    if maps.is_file():
        if output_root != REPORT:
            # Carry the error-map figure and its data into a verification rebuild so the byte comparison covers every file.
            figures_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(maps, figures_dir / maps.name)
            if (REPORT / "error_maps.npz").is_file():
                shutil.copyfile(REPORT / "error_maps.npz", output_root / "error_maps.npz")
        figures.append(maps.name)

    gate_list = [gate_cells(per_dataset, sources)]
    e_sets = set(endpoints.endpoint_set) if not endpoints.empty else set()
    gate_list.append({"id": "G5.3", "check": "E1-E4 reported with CIs and the decision-rule outcome",
                      "passed": {"P0 ADNI test", "E3 P3 AIBL all-prior-k", "E4 P1 AIBL whole cohort one-shot"} <= e_sets and bool(decisions)
                      and endpoints[endpoints.endpoint_set == "P0 ADNI test"][["E1_ci", "E2_ci"]].notna().all().all(), "detail": decisions})
    gate_list += converter_gates(conv_rows)
    bc.atomic_csv(tables / "traceability.csv", sources.table())
    missing = stage3_missing + stage4_missing
    bc.atomic_json(output_root / "gates.json", {"gates": gate_list, "missing": missing, "partial": bool(missing)})
    report = markdown(written, decisions, gate_list, figures, missing)
    bc.atomic_write_text(output_root / "brainode_style_report.md", report)
    return {"gates": gate_list, "figures": figures, "missing": missing}


def markdown(t: dict[str, pd.DataFrame], decisions, gate_list, figures, missing) -> str:
    lines = ["# LHipp latent dynamics - consolidated BrainODE-style report", "",
             "Built from stored evaluation outputs only. Absolute errors are not comparable with BrainODE's (different template, "
             "fitting, PCA and subjects); only rankings, skill and fidelity inside this benchmark are claims. One-shot = first visit "
             "to latest visit; skill = 1 - error / no-change error on the same subjects.", ""]
    if missing:
        lines += [f"**Partial draft:** {len(missing)} evaluations missing (e.g. {', '.join(missing[:4])}).", ""]
    lines += ["## Gates", "", "| gate | check | result |", "|---|---|---|"]
    lines += [f"| {g['id']} | {g['check']} | {'PENDING' if g['passed'] is None else ('PASS' if g['passed'] else 'FAIL')} |" for g in gate_list]
    lines += ["", "G5.1 (byte-identical regeneration) is recorded in verification.json by `--verify`.", ""]
    lines += ["## Pre-registered endpoints and decision", "",
              "E2 improvement means AD capture closer to 100% (|1 - capture| lower). Non-inferior on E1 means the upper 95% limit of the paired "
              "error difference to the cocycle is at most 2% of the no-change error.", ""]
    for outcome in decisions:
        lines.append(f"- **{R3.REP_LABEL[outcome['representation']]}:** {outcome['decision']}. E1 ranking: {', '.join(outcome['ranking_by_E1'])}.")
    lines += [""] + md(t["endpoints_e1_e4"], floats=4)
    sections = [("R-T5. Cohort composition", "r_t5_composition", 2), ("R-T6. Evaluable subjects per task (test)", "r_t6_evaluable", 0),
                ("R-T7. Per dataset, one-shot (BrainODE T7 analog)", "r_t7_per_dataset", 4), ("R-T1. AIBL regular intervals", "r_t1_aibl_regular", 4),
                ("R-T2. Unified pooled test", "r_t2_unified", 4), ("R-T8. Cross-benchmark", "r_t8_cross_benchmark", 4),
                ("R-T3a. Ablations A1-A4 (ADNI test, one-shot)", "r_t3_ablations", 4), ("R-T3b. Converter line (test)", "r_t3_converter", 4),
                ("Disease fidelity, ADNI", "fidelity_adni", 3), ("Disease fidelity, external cohorts", "fidelity_external", 3),
                ("Consistency and cost", "consistency_cost", 4), ("Sensitivity: BrainODE-matched age subset (65-95)", "sensitivity_age_subset", 4),
                ("Sensitivity: min-epoch-15 cocycle selection", "sensitivity_min_epoch", 4), ("Sensitivity: pooled PCA basis (P3)", "sensitivity_pooled_pca", 4),
                ("Context: BrainODE published numbers (not head-to-head)", "context_brainode_published", 3)]
    for title, key, floats in sections:
        lines += ["", f"## {title}", ""] + md(t[key], floats=floats)
    lines += ["", "## Figures", ""] + [f"![{name}](figures/{name})" for name in figures]
    lines += ["", "Traceability: `tables/traceability.csv` lists every summary used with its sha256, checkpoint sha256 and git commit."]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--verify", action="store_true", help="Rebuild into a scratch folder and compare every output byte for byte (G5.1).")
    parser.add_argument("--device", default=None, help="Ignored; accepted because the orchestrator appends it.")
    args = parser.parse_args()
    result = build(bc.require_bulk(REPORT), args.allow_partial)
    print(f"wrote {REPORT / 'brainode_style_report.md'}: " + ", ".join(
        f"{g['id']}={'PENDING' if g['passed'] is None else ('PASS' if g['passed'] else 'FAIL')}" for g in result["gates"]))
    if args.verify:
        scratch = bc.require_bulk(STAGE5_ROOT / "reports_verify")
        shutil.rmtree(scratch, ignore_errors=True)
        build(scratch, args.allow_partial)
        compared = sorted(p.relative_to(REPORT) for p in REPORT.rglob("*") if p.is_file())
        mismatched = [str(p) for p in compared if not (scratch / p).is_file() or not filecmp.cmp(REPORT / p, scratch / p, shallow=False)]
        verification = {"id": "G5.1", "check": "report regenerates byte-identically", "files": len(compared), "mismatched": mismatched, "passed": not mismatched,
                        "report_sha256": bc.sha256_file(REPORT / "brainode_style_report.md")}
        bc.atomic_json(STAGE5_ROOT / "verification.json", verification)
        shutil.rmtree(scratch, ignore_errors=True)
        print(f"G5.1 {'PASS' if verification['passed'] else 'FAIL'}: {len(compared)} files, {len(mismatched)} mismatched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
