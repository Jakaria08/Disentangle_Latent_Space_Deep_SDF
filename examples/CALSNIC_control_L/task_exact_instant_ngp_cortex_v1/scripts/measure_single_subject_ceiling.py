#!/usr/bin/env python3
"""How much cortical surface can this field family represent AT ALL?

Every arm in this task has landed between 60% and 65% of true pial area, against
75% for a PCA-172 oracle: MR64/MR128 dense grids, three Instant-NGP variants, a
two-branch Compact-SDF, a grid-free band-limited Fourier field, and a deformed
implicit field.  Latent size, latent bound, hash capacity, marching-cubes
resolution, sampling density and the curvature penalty have each been tested and
each found non-binding.

That leaves one hypothesis nobody has tested: the field family itself simply
cannot represent a folded cortex, and ~65% is its ceiling rather than a
generalization failure.

This script settles it by removing every source of difficulty except
representation.  It fits arm D's **canonical field** -- the same band-limited
Fourier MLP, verbatim, via ``DeformedDecoder.template_sdf`` -- to a **single
subject**, with:

* no latent code, so there is no cross-subject conflict to resolve;
* no Eikonal and no curvature hinge, so nothing competes with the data term;
* the subject's own exact SDF samples as the only objective.

The result is an upper bound.  Whatever area this reaches is the most arm D's
``template_only`` readout could ever reach, and close to the most any member of
this family could reach.

    reaches ~95%  -> the field is fine; the 65% plateau is generalization,
                     and the project should attack correspondence or capacity
                     per subject, not the architecture.
    reaches ~65%  -> the ceiling is the field family. Marching cubes on a
                     coordinate MLP cannot carry cortical folding at all, and
                     every arm has been measuring the same wall.

Read-only with respect to experiment state: writes one JSON report and, with
--save-meshes, the fitted surfaces.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch

from hashgrid_common import (
    CALSNIC_TASK,
    REPO_ROOT,
    build_decoder,
    choose_device,
    clamped_l1,
    decode_variant_to_mesh,
    load_config,
    load_manifest,
    require_bulk_path,
    set_global_seed,
    stable_seed,
    write_json,
)


def _calsnic_evaluator():
    """Load the CALSNIC evaluator by path: two files share this module name."""
    path = CALSNIC_TASK / "scripts" / "periodic_evaluate_multires.py"
    spec = importlib.util.spec_from_file_location("calsnic_periodic_evaluate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Arm D config; supplies the template field spec.")
    parser.add_argument("--subjects", type=int, default=3, help="Train one field per subject.")
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--points-per-step", type=int, default=16384)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[256, 512])
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--save-meshes", action="store_true")
    return parser.parse_args()


def load_samples(row: dict[str, str], device: torch.device):
    """The subject's exact SDF samples: [N,3] coordinates and [N,1] distances."""
    payload = np.load(row["sdf_npz_path"])
    stacked = np.concatenate([payload["pos"], payload["neg"]], axis=0)
    xyz = torch.from_numpy(np.ascontiguousarray(stacked[:, :3])).float().to(device)
    sdf = torch.from_numpy(np.ascontiguousarray(stacked[:, 3:4])).float().to(device)
    return xyz, sdf


def fit_template(decoder, xyz, sdf, args, clamp_distance: float, near_band: float):
    """Fit the canonical field to one subject. Data term only, by design."""
    parameters = list(decoder.template_hidden.parameters()) + list(
        decoder.template_output.parameters()
    )
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
    schedule = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=max(1, args.steps // 4), gamma=0.5
    )
    # Match the population runs' near/broad split so the comparison is fair.
    near_index = torch.nonzero(sdf.abs().squeeze(1) <= near_band, as_tuple=False)[:, 0]
    half = args.points_per_step // 2
    history = []
    for step in range(1, args.steps + 1):
        near = near_index[torch.randint(len(near_index), (half,), device=xyz.device)]
        broad = torch.randint(len(xyz), (half,), device=xyz.device)
        chosen = torch.cat((near, broad))
        predicted = decoder.template_sdf(xyz[chosen])
        loss = clamped_l1(predicted, sdf[chosen], clamp_distance)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        schedule.step()
        if step % max(1, args.steps // 20) == 0 or step == 1:
            history.append({"step": step, "clamped_l1": float(loss.detach().cpu())})
            print(f"    step {step:>6}/{args.steps}  clamped_l1={float(loss):.6f}", flush=True)
    return history


def occupied_fraction(decoder, device, resolution: int = 48) -> float:
    """Fraction of a coarse lattice the field calls interior.

    Guard, not a metric.  An unconverged field sits near zero everywhere, so its
    zero level set fills the volume and marching cubes at 256**3 or 512**3
    allocates tens of millions of triangles and takes the process down with no
    traceback.  One cheap 48**3 probe catches that and lets the run report the
    failure instead of dying.  A left pial cortex occupies roughly 5-15% of this
    bounding cube.
    """
    step = 2.0 / (resolution - 1)
    index = torch.arange(resolution**3, device=device)
    x = torch.div(index, resolution * resolution, rounding_mode="floor")
    y = torch.div(index, resolution, rounding_mode="floor") % resolution
    z = index % resolution
    xyz = torch.stack((x, y, z), dim=1).float() * step - 1.0
    with torch.no_grad():
        values = decoder.template_sdf(xyz).squeeze(1)
    return float((values < 0.0).float().mean().cpu())


def decode(decoder, resolution: int, max_batch: int, device, scratch: Path):
    """Marching cubes on the canonical field, through the project's decode path.

    Routed via ``decode_variant_to_mesh`` with the ``template_only`` readout
    rather than a local marching-cubes call: the lattice indexing convention
    (x = i // R**2, no axis permutation) has to match every other mesh in this
    project exactly, and duplicating it here is how a silently mirrored surface
    gets made.
    """
    report = decode_variant_to_mesh(
        decoder,
        np.zeros(decoder.latent_size, dtype=np.float32),
        scratch,
        resolution,
        max_batch,
        device,
        "template_only",
    )
    return report


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = choose_device(args.device)
    evaluator = _calsnic_evaluator()
    set_global_seed(int(config["seed"]))

    rows = [r for r in load_manifest(config["manifest"]) if r["split"] == "train"]
    rows = rows[: int(args.subjects)]
    clamp_distance = float(config["clamp_distance"])
    near_band = float(config["sampling"]["near_band"])
    max_batch = int(config["reconstruction"]["max_batch"])
    surface_points = int(config["periodic_evaluation"]["surface_points"])

    output = require_bulk_path(
        args.output
        or Path(config["output_dir"]).parent.parent / "audits" / "single_subject_ceiling"
    )
    results = []
    for number, row in enumerate(rows, start=1):
        scan = row["scan_id"]
        print(f"[{number}/{len(rows)}] {scan}", flush=True)
        started = time.time()
        decoder = build_decoder(config, device)
        xyz, sdf = load_samples(row, device)
        history = fit_template(decoder, xyz, sdf, args, clamp_distance, near_band)
        decoder.eval()

        truth = evaluator.load_mesh(row["mesh_path_mm"])
        occupied = occupied_fraction(decoder, device)
        entry = {
            "scan_id": scan,
            "subject_id": row["subject_id"],
            "steps": int(args.steps),
            "final_clamped_l1": history[-1]["clamped_l1"],
            "ground_truth_area_mm2": float(truth.area),
            "fit_seconds": round(time.time() - started, 1),
            "occupied_fraction": occupied,
            "by_resolution": {},
        }
        print(f"    fit done, occupied_fraction={occupied:.4f}", flush=True)
        # Save before decoding: a decode failure must not cost the fit.
        weights = output / "fields" / f"{scan}_template.pth"
        weights.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"template_hidden": decoder.template_hidden.state_dict(),
             "template_output": decoder.template_output.state_dict(),
             "scan_id": scan, "steps": int(args.steps)},
            weights,
        )
        if not 0.005 <= occupied <= 0.40:
            entry["error"] = (
                f"field did not converge: {occupied:.4f} of the lattice is interior, "
                "outside the plausible 0.005-0.40 for a pial cortex; skipping decode"
            )
            print(f"    SKIPPED: {entry['error']}", flush=True)
            results.append(entry)
            continue
        for resolution in args.resolutions:
            scratch = output / "_scratch" / f"{scan}_mc{resolution}.ply"
            print(f"    decoding MC{resolution} ...", flush=True)
            try:
                decode(decoder, resolution, max_batch, device, scratch)
            except RuntimeError as error:   # no zero level set
                entry["by_resolution"][str(resolution)] = {"error": repr(error)}
                continue
            print(f"    decoded, mapping to mm ...", flush=True)
            mm_mesh = evaluator.mesh_sdf_to_mm(evaluator.load_mesh(scratch), row)
            print(f"    mesh {len(mm_mesh.faces):,} faces, scoring ...", flush=True)
            scratch.unlink()
            metrics = evaluator.surface_metrics(
                truth, mm_mesh, surface_points, stable_seed(scan, int(config["seed"]))
            )
            entry["by_resolution"][str(resolution)] = {
                "area_mm2": float(mm_mesh.area),
                "area_fraction_of_ground_truth": float(mm_mesh.area / truth.area),
                "faces": int(len(mm_mesh.faces)),
                "assd_mm": float(metrics["assd_mm"]),
                "hd95_mm": float(metrics["hd95_mm"]),
                "fscore_1mm": float(metrics["fscore_1mm"]),
                "connected_components": int(metrics["predicted_connected_components"]),
            }
            print(
                f"    MC{resolution}: area={mm_mesh.area / truth.area * 100:.1f}% "
                f"assd={metrics['assd_mm']:.4f} f1={metrics['fscore_1mm']:.4f}",
                flush=True,
            )
            if args.save_meshes:
                evaluator.atomic_export(mm_mesh, output / "meshes" / f"{scan}_mc{resolution}.ply")
        results.append(entry)

    report = {
        "question": "upper bound on pial area representable by arm D's canonical field",
        "method": "single-subject fit, no latent, no Eikonal, no curvature penalty",
        "config": str(Path(args.config).resolve()),
        "architecture": config["network_arch"],
        "subjects": len(results),
        "per_subject": results,
        "mean_area_fraction": {
            str(r): float(
                np.mean(
                    [
                        e["by_resolution"][str(r)]["area_fraction_of_ground_truth"]
                        for e in results
                        if "area_fraction_of_ground_truth" in e["by_resolution"].get(str(r), {})
                    ]
                )
            )
            for r in args.resolutions
        },
        "mean_assd_mm": {
            str(r): float(
                np.mean(
                    [
                        e["by_resolution"][str(r)]["assd_mm"]
                        for e in results
                        if "assd_mm" in e["by_resolution"].get(str(r), {})
                    ]
                )
            )
            for r in args.resolutions
        },
        "reference": {
            "pca_172_oracle_area_fraction": 0.751,
            "arm_D_single_area_fraction": 0.625,
            "arm_D_template_only_area_fraction": 0.611,
        },
    }
    write_json(output / "single_subject_ceiling.json", report)
    print("\nmean area fraction:", {k: round(v, 4) for k, v in report["mean_area_fraction"].items()})
    print("mean ASSD mm      :", {k: round(v, 4) for k, v in report["mean_assd_mm"].items()})
    print("report:", output / "single_subject_ceiling.json")


if __name__ == "__main__":
    main()
