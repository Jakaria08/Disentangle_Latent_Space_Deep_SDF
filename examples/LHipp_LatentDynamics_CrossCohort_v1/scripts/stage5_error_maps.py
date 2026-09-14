#!/usr/bin/env python3
"""Stage 5 step 3: per-vertex error maps for representative subjects (BrainODE Fig. 2 analog).

Three subjects, chosen deterministically as the subject whose PCA-128 cocycle (seed 42) one-shot error is closest to
its group median: an ADNI test CN subject, an ADNI test AD subject (P0 checkpoints) and an AIBL AD subject (zero-shot,
same ADNI-trained checkpoints on the AIBL whole cohort). For each, the first visit is transported to the latest visit
by every dynamics model (PCA-128, seed 42) and decoded; the per-vertex Euclidean distance to the real latest mesh is
drawn on the real mesh with one shared colour scale. The no-change prediction (decoded first visit) is the first panel.

CPU only; writes stage5_brainode_style/reports/figures/f6_error_maps.png and error_maps.npz (the report embeds them).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import benchmark_common as bc
import dynamics_core as D
import stage3_report_adni as R3

STAGE5_ROOT = bc.BULK_ROOT / "stage5_brainode_style"
REPRESENTATION, SEED = "pca128", 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu", help="Ignored beyond cpu; accepted because the orchestrator appends it.")
    return parser.parse_args()


def median_subject(rows: pd.DataFrame, diagnosis: str) -> str:
    part = rows[(rows.task == "one_shot_first") & (rows.variant == "averaged") & (rows.diagnosis == diagnosis)]
    return str(part.iloc[int(np.argmin(np.abs(part.euclidean_mm - part.euclidean_mm.median())))].subject_id)


def main() -> int:
    parse_args()
    parts = D.core()
    C = parts["C"]
    device = torch.device("cpu")
    stage3 = {(r["method"]): r for r in bc.read_json(bc.BULK_ROOT / "stage3_adni" / "results_index.json")["runs"]
              if r["representation"] == REPRESENTATION and r["seed"] == SEED}
    stage4 = bc.read_json(bc.BULK_ROOT / "stage4_crosscohort" / "results_index.json")
    aibl_output = next(Path(e["output"]) for e in stage4["external_evaluations"] if e["view"] == "p1_external_aibl_wholecohort"
                       and e["condition_override"] is None and e["representation"] == REPRESENTATION and e["method"] == "direct_c4" and e["seed"] == SEED)
    adni_rows = pd.read_csv(Path(stage3["direct_c4"]["evaluations"]["test"]) / "task_rows.csv", dtype={"subject_id": str})
    aibl_rows = pd.read_csv(aibl_output / "task_rows.csv", dtype={"subject_id": str})
    cases = [("ADNI test CN", "p0_internal_adni", median_subject(adni_rows, "CN")), ("ADNI test AD", "p0_internal_adni", median_subject(adni_rows, "AD")),
             ("AIBL AD (zero-shot)", "p1_external_aibl_wholecohort", median_subject(aibl_rows, "AD"))]

    transports = {method: D.load_trained_transport(Path(stage3[method]["checkpoint"]), device)[0] for method in R3.METHODS}
    panels, errors, faces = [], {}, None
    for title, view, subject in cases:
        registry = D.view_registry(view)
        train = C.load_archive(REPRESENTATION, "train", registry)
        archive = C.load_archive(REPRESENTATION, "test", registry)
        geometry = C.build_geometry(REPRESENTATION, train, device, registry)
        faces = geometry.faces.cpu().numpy()
        index = int(np.flatnonzero(archive["subject_ids"].astype(str) == subject)[0])
        first, last = int(archive["subject_visit_offsets"][index]), int(archive["subject_visit_offsets"][index + 1] - 1)
        values = C.values_on_device(archive, device)
        real = D.view_vertices(archive)[last]
        source, target = torch.tensor([first]), torch.tensor([last])
        with torch.no_grad():
            predictions = {"No-change": geometry.vertices(values["z"][source])[0].numpy()}
            for method, transport in transports.items():
                predictions[R3.METHOD_LABEL[method]] = geometry.vertices(D.transport_call(transport, method, values, source, target))[0].numpy()
        years = float(archive["visit_time_years_from_baseline"][last] - archive["visit_time_years_from_baseline"][first])
        for name, predicted in predictions.items():
            error = np.linalg.norm(predicted - real, axis=1)
            errors[f"{title} | {name}"] = error
            panels.append((f"{title} ({subject}, {years:.1f} y)", name, real, error))

    bc.atomic_npz(bc.require_bulk(STAGE5_ROOT / "reports" / "error_maps.npz"), {
        "labels": np.asarray(list(errors)), "errors_mm": np.stack(list(errors.values())), "faces": faces})
    plt = R3.style()
    import matplotlib
    from matplotlib import cm, colors

    names = ["No-change"] + [R3.METHOD_LABEL[m] for m in R3.METHODS]
    vmax = float(np.quantile(np.concatenate(list(errors.values())), 0.99))
    norm, cmap = colors.Normalize(0.0, vmax), matplotlib.colormaps["Blues"]  # sequential: one hue, light to dark
    fig = plt.figure(figsize=(2.2 * len(names), 2.3 * len(cases)))
    for position, (case, name, real, error) in enumerate(panels):
        ax = fig.add_subplot(len(cases), len(names), position + 1, projection="3d")
        surface = ax.plot_trisurf(real[:, 0], real[:, 1], real[:, 2], triangles=faces, linewidth=0.0, antialiased=False, shade=False)
        surface.set_facecolor(cmap(norm(error[faces].mean(axis=1))))
        ax.view_init(elev=15, azim=-60)
        ax.set_axis_off()
        row, column = divmod(position, len(names))
        if row == 0:
            ax.set_title(name, fontsize=7)
        if column == 0:
            ax.text2D(-0.15, 0.5, case, transform=ax.transAxes, rotation=90, va="center", fontsize=6, color=R3.INK["secondary"])
        ax.text2D(0.5, -0.02, f"mean {error.mean():.3f} mm", transform=ax.transAxes, ha="center", fontsize=6, color=R3.INK["secondary"])
    bar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=fig.axes, shrink=0.5, pad=0.02)
    bar.set_label("per-vertex Euclidean error to the real latest mesh (mm)")
    fig.suptitle("One-shot prediction error on the real mesh (PCA-128, seed 42; shared scale, 99th percentile cap)", fontsize=9)
    path = bc.require_bulk(STAGE5_ROOT / "reports" / "figures" / "f6_error_maps.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    print(f"WROTE {path}: " + "; ".join(f"{c[0]} = {c[2]}" for c in cases))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
