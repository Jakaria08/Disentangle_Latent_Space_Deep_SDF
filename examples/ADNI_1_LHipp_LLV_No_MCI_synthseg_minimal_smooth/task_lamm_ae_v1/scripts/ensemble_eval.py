#!/usr/bin/env python3
"""Seed variance and prediction ensembling for LAMM checkpoints.

Reports three things this project has never had:
  * single-model mean +/- std over seeds -- the noise floor. Every ~1% claim made here
    (mixup 0.83%, expE 1.2%, expG 1.0%) rests on an unmeasured one, which is why the mixup
    effect had to be revised three times.
  * the ensemble curve for N = 1..k, so you can see where averaging saturates.
  * both backbones at matched seeds.

Ensembling averages the PREDICTED VERTICES (all models share the template topology), then
runs the standard metric -- it is not weight averaging, which would be meaningless across
independently initialised runs in different basins.
"""
from __future__ import annotations

import argparse, itertools, json, sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SPIRAL = Path("/home/jakaria/INR/Deep3DComp/examples/"
              "ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task_spiral_ae_v1/scripts")
for p in (str(HERE), str(SPIRAL)):
    if p not in sys.path: sys.path.insert(0, p)

import spiral_common as sc
import train_eval as te
import train_lamm as T

OUT = Path("/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task_lamm_ae_v1")
REF = {"pca128": 0.033668, "spiralnet128": 0.036784, "meshmae": 0.037867}


@torch.no_grad()
def predict(run_name, data, device, split):
    ck = torch.load(OUT / "studies" / run_name / "best.pt", map_location="cpu")
    a = argparse.Namespace(**ck["args"])
    model, _ = T.build(a, device)
    model.load_state_dict(ck["model_state_dict"]); model.eval()
    x = data.split(split)
    pred = torch.cat([model(x[i:i+32]) for i in range(0, len(x), 32)])
    del model; torch.cuda.empty_cache()
    return data.denormalize(pred).cpu()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True, help="run names sharing one config")
    p.add_argument("--label", default="ensemble")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--splits", nargs="+", default=["val", "test"])
    a = p.parse_args()

    device = torch.device("cuda", a.gpu); torch.cuda.set_device(device)
    data = te.load_mesh_tensors(device)
    report = {"label": a.label, "runs": a.runs, "reference": REF, "splits": {}}

    for split in a.splits:
        gt = data.denormalize(data.split(split)).cpu()
        preds = [predict(r, data, device, split) for r in a.runs]
        singles = [float(sc.vertex_rmse_mm(p_, gt).mean()) for p_ in preds]
        curve = {}
        for k in range(1, len(preds) + 1):                # mean over all k-subsets
            vals = [float(sc.vertex_rmse_mm(torch.stack([preds[i] for i in c]).mean(0), gt).mean())
                    for c in itertools.combinations(range(len(preds)), k)]
            curve[k] = {"mean": float(np.mean(vals)), "min": float(np.min(vals)),
                        "n_subsets": len(vals)}
        report["splits"][split] = {
            "single": {"values": singles, "mean": float(np.mean(singles)),
                       "std": float(np.std(singles, ddof=1)) if len(singles) > 1 else 0.0,
                       "min": float(np.min(singles))},
            "ensemble_curve": curve}
        s = report["splits"][split]["single"]
        print(f"\n== {split}   n={len(singles)} seeds")
        print(f"  single model : mean {s['mean']:.6f}  std {s['std']:.6f}  min {s['min']:.6f}")
        for k, v in curve.items():
            tag = "  <- full ensemble" if k == len(preds) else ""
            print(f"  ensemble k={k}: mean {v['mean']:.6f}  best {v['min']:.6f} "
                  f"({v['n_subsets']} subsets){tag}")
        if split == "val":
            full = curve[len(preds)]["mean"]
            print(f"  vs spiralnet {REF['spiralnet128']:.6f}: single {s['mean']/REF['spiralnet128']:.4f}x"
                  f"  ensemble {full/REF['spiralnet128']:.4f}x")

    fp = OUT / "reports" / f"{a.label}.json"
    fp.parent.mkdir(parents=True, exist_ok=True)
    sc.atomic_write_json(fp, report)
    print(f"\nwrote {fp}")


if __name__ == "__main__":
    main()
