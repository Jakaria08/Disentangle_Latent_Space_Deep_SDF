#!/usr/bin/env python3
"""Helpers for the results notebook (notebooks/LHipp_LatentDynamics_Results.ipynb).

Paths to the stored stage 1-5 results, the report palette (dataviz reference instance, light mode), table rendering,
mesh rendering, and the small CPU inference behind the mesh figures. Everything reads the experiment's bulk outputs;
the only model evaluations are CPU forward passes of checkpoints whose sealed test results are already in the tables.
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXPERIMENT = Path(__file__).resolve().parents[1]
SCRIPTS = EXPERIMENT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import benchmark_common as bc  # noqa: E402

BULK = bc.BULK_ROOT
STAGE1 = bc.STAGE1_ROOT
STAGE3 = BULK / "stage3_adni"
STAGE4 = BULK / "stage4_crosscohort"
STAGE5 = BULK / "stage5_brainode_style"
RUNS = BULK / "runs"
TABLE_DIRS = {"stage3": STAGE3 / "reports" / "tables", "stage4": STAGE4 / "reports" / "tables", "stage5": STAGE5 / "reports" / "tables"}

METHODS = ("direct_c4", "plain_ode", "brainode", "latent_ode", "latent_ode_residual")
METHOD_LABEL = {"direct_c4": "Cocycle (direct C4)", "plain_ode": "Plain ODE", "brainode": "BrainODE",
                "latent_ode": "Latent ODE (faithful)", "latent_ode_residual": "Latent ODE (residual)"}
REPS = ("pca128", "spiralnet128", "adaptive128", "lamm128")
REP_LABEL = {"pca128": "PCA-128", "spiralnet128": "SpiralNet-128", "adaptive128": "Adaptive-128", "lamm128": "LAMM-128"}
COHORTS = ("adni", "aibl", "oasis", "calsnic")
COHORT_LABEL = {"adni": "ADNI", "aibl": "AIBL", "oasis": "OASIS-3", "calsnic": "CALSNIC"}
DISEASE = {"adni": "AD", "aibl": "AD", "oasis": "AD", "calsnic": "ALS"}
STRICT_VIEWS = {"adni": "p0_internal_adni", "aibl": "p2_internal_aibl", "oasis": "p2_internal_oasis", "calsnic": "p2_internal_calsnic"}
CONVERTER_VIEW = "p5_converter_pooled"

# Dataviz reference palette, light mode: categorical slots in fixed order, text/chrome tokens, one-hue sequential ramp,
# and a warm/cool diverging pair with a neutral grey midpoint.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK = {"surface": "#fcfcfb", "primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "baseline": "#c3c2b7"}
SEQUENTIAL_STEPS = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
                    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
DIVERGING_MID = "#f0efec"
MESH_GREY = "#c9c8c1"
METHOD_COLOR = dict(zip(METHODS, SLOTS))
METHOD_MARKER = dict(zip(METHODS, ("o", "s", "D", "v", "^")))
REP_COLOR = dict(zip(REPS, SLOTS))
REP_MARKER = dict(zip(REPS, ("o", "s", "D", "^")))


# --------------------------------------------------------------------------------------
# style, colour maps, tables
# --------------------------------------------------------------------------------------


def style():
    import matplotlib

    import matplotlib.pyplot as plt

    matplotlib.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "figure.facecolor": INK["surface"], "axes.facecolor": INK["surface"],
        "savefig.facecolor": INK["surface"], "axes.edgecolor": INK["baseline"], "axes.linewidth": 0.6, "axes.grid": True,
        "grid.color": INK["grid"], "grid.linewidth": 0.6, "grid.linestyle": "-", "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 9.5, "axes.labelsize": 8.5, "text.color": INK["primary"], "axes.labelcolor": INK["secondary"],
        "axes.titlecolor": INK["primary"], "xtick.color": INK["muted"], "ytick.color": INK["muted"], "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "font.size": 8, "font.family": "sans-serif", "lines.linewidth": 1.6, "lines.markersize": 6,
        "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round", "legend.frameon": False, "legend.fontsize": 7.5,
        "figure.titlesize": 10.5, "axes.axisbelow": True,
    })
    return plt


def sequential_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("report_sequential_blue", SEQUENTIAL_STEPS)


def diverging_cmap():
    """Negative (inward, volume loss) red, zero grey, positive (outward) blue."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("report_diverging", [SLOTS[7], DIVERGING_MID, SLOTS[0]])


def table_csv(stage: str, name: str) -> pd.DataFrame:
    return pd.read_csv(TABLE_DIRS[stage] / f"{name}.csv")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def show(frame: pd.DataFrame, caption: str, precision: int = 3, formats: dict[str, str] | None = None, index: bool = False):
    """A paper-style table: caption on top, hairline header rule, no index."""
    formatter = {}
    for column in frame.columns:  # one formatter for every column; a second format() call would reset the others
        if pd.api.types.is_bool_dtype(frame[column]):
            continue
        if pd.api.types.is_integer_dtype(frame[column]):
            formatter[column] = "{:,.0f}"
        elif pd.api.types.is_float_dtype(frame[column]):
            formatter[column] = f"{{:.{precision}f}}"
    formatter.update({column: spec for column, spec in (formats or {}).items() if column in frame.columns})
    styler = frame.style.format(formatter=formatter, na_rep="–")
    if not index:
        styler = styler.hide(axis="index")
    return styler.set_caption(caption).set_table_styles([
        {"selector": "caption", "props": "caption-side: top; text-align: left; font-weight: 600; color: #0b0b0b; padding: 4px 0 6px 0;"},
        {"selector": "th", "props": "text-align: left; color: #52514e; font-weight: 600; border-bottom: 1px solid #c3c2b7; padding: 3px 10px;"},
        {"selector": "td", "props": "padding: 2px 10px; color: #0b0b0b;"},
    ])


def interval(ax, y, center, low, high, color, marker="o", label=None, size=6.5):
    ax.plot([low, high], [y, y], color=color, linewidth=1.6, solid_capstyle="round", zorder=2)
    ax.plot([center], [y], marker, color=color, markersize=size, markeredgecolor=INK["surface"], markeredgewidth=1.2, label=label, zorder=3)


def dot(ax, x, y, color, marker="o", label=None, size=6.5, filled=True):
    ax.plot([x], [y], marker, color=color, markersize=size, markeredgecolor=INK["surface"] if filled else color,
            markerfacecolor=color if filled else INK["surface"], markeredgewidth=1.2, label=label, zorder=3, linestyle="none")


# --------------------------------------------------------------------------------------
# provenance, gates, key findings
# --------------------------------------------------------------------------------------


def provenance_markdown() -> str:
    verification = read_json(STAGE5 / "verification.json")
    gates = read_json(STAGE5 / "reports" / "gates.json")
    return (f"**Results root:** `{BULK}`  \n**Consolidated report:** `{STAGE5 / 'reports' / 'brainode_style_report.md'}` "
            f"(sha256 `{verification['report_sha256'][:16]}…`, rebuilt byte-identically: {'yes' if verification['passed'] else 'no'}, "
            f"{verification['files']} files)  \n**Missing evaluations:** {len(gates['missing'])}")


def gates_table() -> pd.DataFrame:
    rows = []
    for gate in read_json(STAGE1 / "reports" / "stage1_audit.json")["gates"]:
        rows.append({"stage": "1 data foundation", "gate": gate["id"], "check": gate["name"], "passed": gate["passed"]})
    for stage, path in (("3 ADNI benchmark", STAGE3 / "reports" / "stage3_gates.json"), ("4 cross-cohort", STAGE4 / "reports" / "stage4_gates.json")):
        for gate in read_json(path)["gates"]:
            rows.append({"stage": stage, "gate": gate["id"], "check": gate["check"], "passed": gate["passed"]})
    verification = read_json(STAGE5 / "verification.json")
    rows.append({"stage": "5 report", "gate": "G5.1", "check": verification["check"], "passed": verification["passed"]})
    for gate in read_json(STAGE5 / "reports" / "gates.json")["gates"]:
        rows.append({"stage": "5 report", "gate": gate["id"], "check": gate["check"], "passed": gate["passed"]})
    frame = pd.DataFrame(rows)
    frame["result"] = frame.passed.map({True: "PASS", False: "FAIL"}).fillna("PENDING")
    return frame.drop(columns="passed")


def key_findings_markdown() -> str:
    endpoints = table_csv("stage5", "endpoints_e1_e4")
    p0 = endpoints[endpoints.endpoint_set == "P0 ADNI test"]
    errors = table_csv("stage3", "p3a_errors")
    nochange = errors[(errors.model == "baseline:nochange_decoded") & (errors.task == "one_shot_first")].euclidean_mm_mean.mean()
    fidelity = table_csv("stage5", "fidelity_adni")
    per_dataset = table_csv("stage5", "r_t7_per_dataset")
    ablations = table_csv("stage5", "r_t3_ablations").set_index(["row", "representation"])
    converter = table_csv("stage5", "r_t3_converter")
    pooled = table_csv("stage5", "sensitivity_pooled_pca")
    decisions = next(g for g in read_json(STAGE5 / "reports" / "gates.json")["gates"] if g["id"] == "G5.3")["detail"]
    g57 = next(g for g in read_json(STAGE5 / "reports" / "gates.json")["gates"] if g["id"] == "G5.7")
    e1 = p0.groupby("method").E1_mm.agg(["min", "max"])
    capture = fidelity.groupby("model").ad_capture_mean.agg(["min", "max"])
    zero_shot = per_dataset[per_dataset.scope == "P1 zero-shot (whole cohort)"].groupby(["cohort", "method"]).skill.mean().unstack()
    preferred = sorted({m for d in decisions for m in d["preferred_over_cocycle"]})
    leaders = "; ".join(f"{REP_LABEL[d['representation']]}: {METHOD_LABEL[d['ranking_by_E1'][0].split(' ')[0]]}" for d in decisions)
    one_shot = converter[converter.task == "one_shot_first"].set_index(["model", "representation"])
    lines = [
        f"- **ADNI one-shot error (range over the four representations):** "
        + "; ".join(f"{METHOD_LABEL[m]} {e1.loc[m, 'min']:.3f}–{e1.loc[m, 'max']:.3f} mm" for m in METHODS)
        + f"; no-change {nochange:.3f} mm.",
        "- **Pre-registered decision (E1 non-inferiority + E2 closer to 100%):** "
        + (f"preferred over the cocycle: {', '.join(METHOD_LABEL[m] for m in preferred)}" if preferred else "no method is preferred over the cocycle in any representation")
        + f". Lowest E1 per representation — {leaders}.",
        "- **AD atrophy captured on ADNI (predicted / observed AD rate):** "
        + "; ".join(f"{METHOD_LABEL[m]} {capture.loc[m, 'min']:.2f}–{capture.loc[m, 'max']:.2f}" for m in METHODS) + ".",
        "- **Zero-shot skill on whole external cohorts (mean over representations):** "
        + "; ".join(f"{COHORT_LABEL[c]}: " + ", ".join(f"{METHOD_LABEL[m]} {100 * zero_shot.loc[c, m]:+.1f}%" for m in ("direct_c4", "latent_ode_residual", "plain_ode", "brainode"))
                    for c in ("aibl", "oasis", "calsnic")) + ".",
        "- **Ablations (PCA, ADNI):** AD capture "
        + f"cocycle {ablations.loc[('reference direct_c4', 'pca128'), 'ad_capture']:.2f}, without disease head {ablations.loc[('A3 direct_c4_no_disease', 'pca128'), 'ad_capture']:.2f}; "
        + f"BrainODE {ablations.loc[('reference brainode', 'pca128'), 'ad_capture']:.2f}, BrainODE field + cocycle loss {ablations.loc[('A2 brainode_v', 'pca128'), 'ad_capture']:.2f}; "
        + f"exact coboundary {ablations.loc[('A1 exact_coboundary_c4', 'pca128'), 'ad_capture']:.2f} (PCA) / {ablations.loc[('A1 exact_coboundary_c4', 'spiralnet128'), 'ad_capture']:.2f} (SpiralNet).",
        "- **Converter line (5 test AD converters, descriptive):** AD-converter one-shot error C0 "
        + f"{one_shot.loc[('C0 (source label)', 'pca128'), 'AD converters_mm']:.3f} mm, C1 prefix-only {one_shot.loc[('C1 c1 (prefix)', 'pca128'), 'AD converters_mm']:.3f} mm, "
        + f"C1 oracle window {one_shot.loc[('C1 c1 (oracle)', 'pca128'), 'AD converters_mm']:.3f} mm (PCA); gate G5.7 (stable non-inferiority) "
        + ("passes" if g57["passed"] else "fails for the fine-tuned C1 variants (C1-frozen passes)") + ".",
        "- **Pooled PCA basis (sensitivity):** largest paired change "
        + f"{pooled[pooled.task == 'one_shot_first'].mean_difference.min():+.4f} mm; the ranking of dynamics is unchanged.",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# cohorts
# --------------------------------------------------------------------------------------


@lru_cache(maxsize=None)
def strict_archives(cohort: str) -> tuple[dict[str, np.ndarray], ...]:
    root = STAGE1 / "views" / STRICT_VIEWS[cohort] / "dataset"
    return tuple(bc.load_npz(root / f"{split}_subject_sequences.npz") for split in bc.SPLITS)


def subjects_frame(cohort: str) -> pd.DataFrame:
    rows = []
    for archive in strict_archives(cohort):
        offsets = archive["subject_visit_offsets"]
        for index, subject in enumerate(archive["subject_ids"].astype(str)):
            start, end = int(offsets[index]), int(offsets[index + 1])
            years = archive["visit_time_years_from_baseline"][start:end].astype(float)
            volumes = archive["visit_volume_mm3"][start:end].astype(float)
            rate = np.polyfit(years, np.log(volumes), 1)[0] * 100 if end - start >= 2 and np.ptp(years) >= 0.25 else np.nan
            rows.append({"cohort": cohort, "subject": subject, "label": int(archive["subject_label_ad"][index]), "visits": end - start,
                         "span_years": float(np.ptp(years)), "first_scan": str(archive["visit_scan_ids"][start]), "last_scan": str(archive["visit_scan_ids"][end - 1]),
                         "first_volume": float(volumes[0]), "last_volume": float(volumes[-1]), "age": float(archive["visit_age_years"][start]),
                         "rate_pct_per_year": rate})
    return pd.DataFrame(rows)


def observed_rates() -> pd.DataFrame:
    frame = pd.concat([subjects_frame(c) for c in COHORTS], ignore_index=True).dropna(subset=["rate_pct_per_year"])
    summary = frame.groupby(["cohort", "label"]).rate_pct_per_year.agg(subjects="size", median="median",
                                                                         q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75)).reset_index()
    return summary


# --------------------------------------------------------------------------------------
# meshes
# --------------------------------------------------------------------------------------


@lru_cache(maxsize=None)
def faces() -> np.ndarray:
    return np.load(bc.load_registry()["faces_path"]).astype(np.int64)


def real_vertices(scan_keys) -> np.ndarray:
    import dynamics_core as D

    return D.VertexStore().get([str(k) for k in scan_keys])


def vertex_normals(vertices: np.ndarray) -> np.ndarray:
    import trimesh

    mesh = trimesh.Trimesh(np.asarray(vertices, dtype=np.float64), faces(), process=False)
    return np.asarray(mesh.vertex_normals) * (1.0 if mesh.volume >= 0 else -1.0)


def mesh_volume(vertices: np.ndarray) -> float:
    import trimesh

    return float(abs(trimesh.Trimesh(np.asarray(vertices, dtype=np.float64), faces(), process=False).volume))


def normal_displacement(before: np.ndarray, after: np.ndarray, remove_centroid: bool = True) -> np.ndarray:
    """Signed displacement along the outward normal of ``before`` (negative = inward)."""
    shift = after.mean(axis=0) - before.mean(axis=0) if remove_centroid else 0.0
    return np.sum((after - shift - before) * vertex_normals(before), axis=1)


def render_meshes(panels: list[dict[str, Any]], ncols: int, cmap=None, norm=None, colorbar_label: str | None = None,
                  title: str | None = None, view: tuple[float, float] = (15.0, -60.0), size: tuple[float, float] = (2.25, 2.05)):
    """Grid of hippocampus surfaces. A panel has ``vertices`` and optional ``scalar`` (per vertex), ``title``, ``subtitle``."""
    plt = style()
    from matplotlib import cm

    triangles = faces()
    nrows = int(np.ceil(len(panels) / ncols))
    extra = 0.95 if cmap is not None else 0.25  # inches for the colour bar, its ticks and its label
    fig = plt.figure(figsize=(size[0] * ncols, size[1] * nrows + extra))
    for position, panel in enumerate(panels):
        ax = fig.add_subplot(nrows, ncols, position + 1, projection="3d")
        vertices = np.asarray(panel["vertices"])
        scalar = panel.get("scalar")
        surface = ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=triangles, linewidth=0.0, antialiased=False,
                                  shade=scalar is None, color=MESH_GREY if scalar is None else None)
        if scalar is not None:
            surface.set_facecolor(cmap(norm(np.asarray(scalar)[triangles].mean(axis=1))))
        try:
            ax.set_box_aspect(tuple(np.ptp(vertices, axis=0)), zoom=1.35)
        except TypeError:
            ax.set_box_aspect(tuple(np.ptp(vertices, axis=0)))
        ax.view_init(*panel.get("view", view))
        ax.set_axis_off()
        ax.set_title(panel.get("title", ""), fontsize=7.5, color=INK["primary"], y=0.98)
        if panel.get("subtitle"):
            ax.text2D(0.5, 0.04, panel["subtitle"], transform=ax.transAxes, ha="center", fontsize=6.5, color=INK["secondary"])
    height = fig.get_size_inches()[1]
    bottom = (extra / height) if cmap is not None else 0.02
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0 - 0.45 / height if title else 0.98, bottom=bottom, wspace=0.0, hspace=0.08)
    if cmap is not None:
        cax = fig.add_axes([0.3, 0.48 / height, 0.4, 0.12 / height])
        bar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
        bar.outline.set_visible(False)
        bar.ax.tick_params(labelsize=7, colors=INK["muted"])
        if colorbar_label:
            bar.set_label(colorbar_label, fontsize=7.5, color=INK["secondary"])
    if title:
        fig.suptitle(title, fontsize=10, color=INK["primary"])
    return fig


def cohort_example_panels() -> list[dict[str, Any]]:
    panels = []
    for cohort in COHORTS:
        frame = subjects_frame(cohort)
        for label in (0, 1):
            group = frame[frame.label == label]
            chosen = group.iloc[int(np.argmin(np.abs(group.first_volume - group.first_volume.median())))]
            panels.append({"vertices": real_vertices([chosen.first_scan])[0],
                           "title": f"{COHORT_LABEL[cohort]} · {'control' if label == 0 else DISEASE[cohort]}",
                           "subtitle": f"{chosen.first_volume:,.0f} mm³ · {chosen.age:.0f} y"})
    return panels


def observed_change_panels() -> tuple[list[dict[str, Any]], float]:
    panels = []
    for cohort, label, min_span in (("adni", 0, 1.0), ("adni", 1, 1.0), ("aibl", 1, 1.0), ("oasis", 1, 1.0), ("calsnic", 0, 0.5), ("calsnic", 1, 0.5)):
        frame = subjects_frame(cohort)
        group = frame[(frame.label == label) & (frame.span_years >= min_span)].dropna(subset=["rate_pct_per_year"])
        chosen = group.iloc[int(np.argmin(np.abs(group.rate_pct_per_year - group.rate_pct_per_year.median())))]
        first, last = real_vertices([chosen.first_scan, chosen.last_scan])
        panels.append({"vertices": first, "scalar": normal_displacement(first, last),
                       "title": f"{COHORT_LABEL[cohort]} · {'control' if label == 0 else DISEASE[cohort]}",
                       "subtitle": f"{chosen.span_years:.1f} y · volume {100 * (chosen.last_volume / chosen.first_volume - 1):+.1f}%"})
    vmax = float(np.quantile(np.abs(np.concatenate([p["scalar"] for p in panels])), 0.98))
    return panels, vmax


# --------------------------------------------------------------------------------------
# CPU inference for mesh figures
# --------------------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _core():
    import dynamics_core as D

    return D, D.core()


def device():
    import torch

    return torch.device("cpu")


@lru_cache(maxsize=None)
def archive(view: str, split: str, representation: str = "pca128") -> dict[str, np.ndarray]:
    D, parts = _core()
    return parts["C"].load_archive(representation, split, D.view_registry(view))


@lru_cache(maxsize=None)
def geometry(view: str, representation: str = "pca128"):
    D, parts = _core()
    return parts["C"].build_geometry(representation, archive(view, "train", representation), device(), D.view_registry(view))


@lru_cache(maxsize=None)
def values(view: str, split: str = "test", representation: str = "pca128"):
    _D, parts = _core()
    return parts["C"].values_on_device(archive(view, split, representation), device())


def age_range(view: str) -> float:
    normalization = read_json(STAGE1 / "views" / view / "view_manifest.json")["normalization"]
    return float(normalization["age_max_years"]) - float(normalization["age_min_years"])


def decode(view: str, codes, representation: str = "pca128") -> np.ndarray:
    import torch

    with torch.no_grad():
        return geometry(view, representation).vertices(torch.as_tensor(np.asarray(codes), dtype=torch.float32)).numpy()


def stage3_run(method: str, representation: str = "pca128", seed: int = 42) -> dict[str, Any]:
    return next(r for r in read_json(STAGE3 / "results_index.json")["runs"]
                if r["method"] == method and r["representation"] == representation and r["seed"] == seed)


@lru_cache(maxsize=None)
def transport_for(checkpoint: str):
    D, _parts = _core()
    return D.load_trained_transport(Path(checkpoint), device())[0]


def visits_of(view: str, subject: str, split: str = "test") -> tuple[int, int]:
    arch = archive(view, split)
    index = int(np.flatnonzero(arch["subject_ids"].astype(str) == subject)[0])
    return int(arch["subject_visit_offsets"][index]), int(arch["subject_visit_offsets"][index + 1] - 1)


def median_subject(rows: pd.DataFrame, diagnosis: str) -> str:
    part = rows[(rows.task == "one_shot_first") & (rows.variant == "averaged") & (rows.diagnosis == diagnosis)]
    return str(part.iloc[int(np.argmin(np.abs(part.euclidean_mm - part.euclidean_mm.median())))].subject_id)


def reconstruction_table() -> pd.DataFrame:
    rows = []
    for cohort in COHORTS:
        report = read_json(STAGE1 / "latents" / cohort / "encoding_report.json")
        row = {"cohort": COHORT_LABEL[cohort]}
        for representation in REPS:
            row[REP_LABEL[representation]] = float(report[representation]["recon_coordinate_rmse_mm_by_split"]["test"])
        rows.append(row)
    frame = pd.DataFrame(rows)
    bases = pd.read_csv(STAGE5 / "sensitivity" / "pooled_pca" / "reconstruction_by_cohort_split_basis.csv")
    test = bases[bases.split == "test"].pivot_table(index="cohort", columns="basis", values="recon_coordinate_rmse_mm")
    frame["PCA-128 pooled basis"] = [test.loc[c, "pooled P3 train"] for c in COHORTS]
    frame["PCA-128 own-cohort basis"] = [test.loc[c, "own cohort train"] for c in COHORTS]
    return frame


def reconstruction_error_panels() -> tuple[list[dict[str, Any]], float]:
    panels = []
    for title, view in (("ADNI test", "p0_internal_adni"), ("OASIS-3 (ADNI encoders)", "p1_external_oasis_wholecohort")):
        arch = archive(view, "test")
        offsets = arch["subject_visit_offsets"]
        firsts = offsets[:-1]
        cn = [int(f) for f, label in zip(firsts, arch["subject_label_ad"]) if label == 0]
        volumes = arch["visit_volume_mm3"][cn]
        visit = cn[int(np.argmin(np.abs(volumes - np.median(volumes))))]
        real = real_vertices([arch["visit_scan_ids"][visit]])[0]
        for representation in REPS:
            code = archive(view, "test", representation)["visit_latent_standardized_128"][visit:visit + 1]
            decoded = decode(view, code, representation)[0]
            error = np.linalg.norm(decoded - real, axis=1)
            panels.append({"vertices": real, "scalar": error, "title": f"{title} · {REP_LABEL[representation]}",
                           "subtitle": f"mean {error.mean():.3f} mm"})
    vmax = float(np.quantile(np.concatenate([p["scalar"] for p in panels]), 0.99))
    return panels, vmax


def one_shot_error_panels(seed: int = 42) -> tuple[list[dict[str, Any]], float]:
    import torch

    D, _parts = _core()
    adni_rows = pd.read_csv(Path(stage3_run("direct_c4", seed=seed)["evaluations"]["test"]) / "task_rows.csv", dtype={"subject_id": str})
    index4 = read_json(STAGE4 / "results_index.json")
    aibl_output = next(Path(e["output"]) for e in index4["external_evaluations"] if e["view"] == "p1_external_aibl_wholecohort"
                       and e["condition_override"] is None and e["representation"] == "pca128" and e["method"] == "direct_c4" and e["seed"] == seed)
    aibl_rows = pd.read_csv(aibl_output / "task_rows.csv", dtype={"subject_id": str})
    cases = [("ADNI test · CN", "p0_internal_adni", median_subject(adni_rows, "CN")),
             ("ADNI test · AD", "p0_internal_adni", median_subject(adni_rows, "AD")),
             ("AIBL · AD (zero-shot)", "p1_external_aibl_wholecohort", median_subject(aibl_rows, "AD"))]
    panels = []
    for title, view, subject in cases:
        first, last = visits_of(view, subject)
        arch, vals = archive(view, "test"), values(view)
        real = real_vertices([arch["visit_scan_ids"][last]])[0]
        years = float(arch["visit_time_years_from_baseline"][last] - arch["visit_time_years_from_baseline"][first])
        predictions = {"No-change": decode(view, vals["z"][first:first + 1].numpy())[0]}
        source, target = torch.tensor([first]), torch.tensor([last])
        for method in METHODS:
            transport = transport_for(stage3_run(method, seed=seed)["checkpoint"])
            with torch.no_grad():
                code = D.transport_call(transport, method, vals, source, target)
            predictions[METHOD_LABEL[method]] = decode(view, code.numpy())[0]
        for name, predicted in predictions.items():
            error = np.linalg.norm(predicted - real, axis=1)
            panels.append({"vertices": real, "scalar": error, "title": name if title.startswith("ADNI test · CN") else "",
                           "subtitle": f"{title} ({years:.1f} y) · {error.mean():.3f} mm" if name == "No-change" else f"{error.mean():.3f} mm"})
    vmax = float(np.quantile(np.concatenate([p["scalar"] for p in panels]), 0.99))
    return panels, vmax


def condition_sweep_panels(years: float = 4.0, seed: int = 42, methods=("direct_c4", "plain_ode", "brainode", "latent_ode_residual")):
    import torch

    view = "p0_internal_adni"
    rows = pd.read_csv(Path(stage3_run("direct_c4", seed=seed)["evaluations"]["test"]) / "task_rows.csv", dtype={"subject_id": str})
    subject = median_subject(rows, "CN")
    first, _last = visits_of(view, subject)
    vals = values(view)
    z0, age0 = vals["z"][first:first + 1], vals["age"][first:first + 1]
    target = age0 + years / age_range(view)
    base = decode(view, z0.numpy())[0]
    panels, volume0 = [], mesh_volume(base)
    for method in methods:
        transport = transport_for(stage3_run(method, seed=seed)["checkpoint"])
        with torch.no_grad():
            predicted = {d: decode(view, transport.transport(z0, age0, target, torch.tensor([float(d)])).numpy())[0] for d in (0, 1)}
        maps = {d: normal_displacement(base, predicted[d]) for d in (0, 1)}
        for key, scalar, label in ((0, maps[0], "d = 0 (control)"), (1, maps[1], "d = 1 (disease)"), ("diff", maps[1] - maps[0], "d = 1 minus d = 0")):
            subtitle = f"volume {100 * (mesh_volume(predicted[key]) / volume0 - 1):+.1f}%" if key in (0, 1) else f"extra inward {-np.minimum(scalar, 0).mean():.3f} mm"
            panels.append({"vertices": base, "scalar": scalar, "title": f"{METHOD_LABEL[method]} · {label}", "subtitle": subtitle})
    vmax = float(np.quantile(np.abs(np.concatenate([p["scalar"] for p in panels])), 0.99))
    return panels, vmax, subject


# --------------------------------------------------------------------------------------
# converter line
# --------------------------------------------------------------------------------------


def converter_run_dir(variant: str, representation: str = "pca128", seed: int = 42) -> Path:
    return RUNS / CONVERTER_VIEW / representation / variant / f"{representation}_{variant}_s{seed}"


def learned_onsets() -> pd.DataFrame:
    frames = []
    for path in sorted(RUNS.glob(f"{CONVERTER_VIEW}/*/*/*/learned_onsets.csv")):
        representation, variant = path.parts[-4], path.parts[-3]
        frame = pd.read_csv(path)
        if not frame.empty:
            frames.append(frame.assign(representation=representation, variant=variant, seed=int(path.parent.name.rsplit("_s", 1)[1])))
    return pd.concat(frames, ignore_index=True)


def brainode_full_estimator() -> pd.DataFrame:
    rows = []
    for path in sorted(RUNS.glob(f"{CONVERTER_VIEW}/pca128/brainode_full/*/evaluation__{CONVERTER_VIEW}/test/summary.json")):
        summary = read_json(path)
        estimator = summary["estimator_E_C4"]
        rows.append({"seed": summary["seed"], "subject AUROC": estimator["stable"]["subject"]["auroc"],
                     "subject balanced accuracy": estimator["stable"]["subject"]["balanced_accuracy"],
                     "scan AUROC": estimator["stable"]["scan"]["auroc"], "mean c before conversion": estimator["converter_mean_condition_before_conversion"],
                     "mean c after conversion": estimator["converter_mean_condition_after_conversion"]})
    return pd.DataFrame(rows)


def synthetic_onsets() -> pd.DataFrame:
    frames = []
    for representation in ("pca128", "adaptive128"):
        for width in ("0.25", "0.5"):
            path = STAGE5 / "converter" / "synthetic_onset" / f"{representation}_s42_w{width}.csv"
            if path.is_file():
                frames.append(pd.read_csv(path).assign(representation=representation, width_years=float(width)))
    return pd.concat(frames, ignore_index=True)


def converter_example(seed: int = 42) -> dict[str, Any]:
    import torch

    import converter_line as L
    import evaluate_converter as EC

    view = CONVERTER_VIEW
    arch, vals = archive(view, "test"), values(view)
    offsets = arch["subject_visit_offsets"]
    years = arch["visit_time_years_from_baseline"]
    candidates = [i for i, g in enumerate(arch["subject_trajectory_groups"].astype(str)) if g == "MCI->AD"]
    spans = [float(years[offsets[i + 1] - 1] - years[offsets[i]]) for i in candidates]
    index = candidates[int(np.argsort(spans)[len(spans) // 2])]
    first, last = int(offsets[index]), int(offsets[index + 1] - 1)
    labels = arch["visit_trajectory_labels"][first:last + 1].tolist()
    real_first, real_last = real_vertices([arch["visit_scan_ids"][first], arch["visit_scan_ids"][last]])
    source, target = torch.tensor([first]), torch.tensor([last])
    c0 = transport_for(str(RUNS / "p3_pooled" / "pca128" / "direct_c4" / f"pca128_direct_c4_s{seed}" / "checkpoints" / "best.pt"))
    model, _payload = EC.load_model(converter_run_dir("c1", seed=seed) / "checkpoints" / "best.pt", device())
    predictions = {}
    with torch.no_grad():
        predictions["C0 · source label"] = c0.transport(vals["z"][source], vals["age"][source], vals["age"][target], vals["label"][source])
        for rule, prefixes in (("prefix-only onset", {index: [0]}), ("oracle-window onset", None)):
            model.set_table(L.build_condition_table(arch, "prefix" if prefixes else "oracle", prefixes, theta_count=model.theta.numel()))
            predictions[f"C1 · {rule}"] = model.transport_visits(vals["z"][source], vals["age"][source], vals["age"][target], source)
    decoded = {name: decode(view, code.numpy())[0] for name, code in predictions.items()}
    return {"subject": str(arch["subject_ids"][index]), "labels": labels, "years": float(years[last] - years[first]),
            "real_first": real_first, "real_last": real_last, "predictions": decoded}


def dose_curves(onset: float = 1.0, width: float = 0.25, span: tuple[float, float] = (-1.0, 3.0)) -> pd.DataFrame:
    import torch

    import converter_line as L

    tau = torch.linspace(span[0], span[1], 401, dtype=torch.float64)
    start = torch.full_like(tau, span[0])
    instantaneous = torch.sigmoid((tau - onset) / width)
    average = L.average_dose(start, tau, torch.full_like(tau, onset), width)
    return pd.DataFrame({"years": tau.numpy(), "instantaneous": instantaneous.numpy(), "average_from_start": average.numpy()})
