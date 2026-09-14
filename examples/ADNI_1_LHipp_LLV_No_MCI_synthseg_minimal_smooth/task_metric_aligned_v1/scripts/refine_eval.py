#!/usr/bin/env python3
"""Post-hoc latent refinement -- ARM 1 of the semi-amortised experiment.

PCA's encoder z = U^T(x-mu) is the exact least-squares argmin for its own decoder, so PCA's
amortisation gap is ZERO by construction. A learned encoder is only an approximation to its
decoder's optimal code, and on this project's best checkpoint that approximation was measured
to cost 1.09% on val and 1.08% on test. Closing it does not advantage the learned model; it
removes a handicap PCA never had.

The refinement can only help: it starts from the encoder's own output and descends the same
reconstruction objective, so per sample the loss is non-increasing. Nothing else tried in this
project has that property.

PCA is included as a control -- it must gain ~0, which validates the harness.

Handles both LAMM checkpoints (task_lamm_ae_v1 / task_metric_aligned_v1) and the aligned
spiral checkpoints, dispatching on whether the saved args carry a `backbone` field.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
LAMM = HERE.parent.parent / "task_lamm_ae_v1" / "scripts"
SPIRAL = HERE.parent.parent / "task_spiral_ae_v1" / "scripts"
for p in (str(HERE), str(LAMM), str(SPIRAL)):
    if p not in sys.path: sys.path.insert(0, p)

import spiral_common as sc
import train_eval as te

OUT = Path("/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task_metric_aligned_v1")


def _defaults_of(mod, argv):
    """Current parser defaults, so checkpoints saved before a flag existed still load."""
    old = sys.argv
    try:
        sys.argv = argv
        return vars(mod.parse_args())
    finally:
        sys.argv = old


def load_model(ckpt_fp, device):
    ck = torch.load(ckpt_fp, map_location="cpu")
    saved = ck["args"]
    if "backbone" in saved:                          # LAMM
        import train_lamm as T
        a = argparse.Namespace(**{**_defaults_of(T, ["x"]), **saved})
        model, _ = T.build(a, device)
    else:                                            # aligned spiral / adaptive
        import train_spiral_aligned as S
        from network import build_model, resolve_conv_types
        a = argparse.Namespace(**{**_defaults_of(S, ["x", "--run-name", "x"]), **saved})
        ds = [int(v) for v in a.ds_factors.split(",")]
        oc = [a.base_channels, a.base_channels, a.base_channels * 2][:len(ds)]
        tr = sc.get_transform(ds)
        ct = resolve_conv_types(len(ds), a.conv_type, a.adaptive_levels)
        dyn = [max(1, round(a.dyn_frac * int(tr["vertices"][i].shape[0])))
               if ct[i] == "adaptive" else 1 for i in range(len(ds))]
        sp, dy, dn, up = sc.build_spiral_stack(tr, a.seq_length, a.dilation, dyn, device)
        model = build_model(tr, sp, dy, dn, up, oc, a.latent, a.conv_type, a.adaptive_levels,
                            conv_types=ct, dropout=a.dropout, linear_skip=False).to(device)
    model.load_state_dict(ck["model_state_dict"]); model.eval()
    for p_ in model.parameters(): p_.requires_grad_(False)
    return model, a


def obj(d, a):
    if getattr(a, "loss", "l1") == "l2":
        return d.pow(2).mean()
    if getattr(a, "loss", "l1") == "huber":
        return F.huber_loss(d, torch.zeros_like(d), delta=getattr(a, "huber_delta", 0.1))
    return d.abs().mean()


def refine(model, data, a, split, steps, lr, sigma, bs=64):
    x = data.split(split); outs = []
    for i in range(0, len(x), bs):
        xb = x[i:i+bs]
        with torch.no_grad():
            z = model.encode(xb).clone()
        z.requires_grad_(True)
        opt = torch.optim.Adam([z], lr=lr)
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            d = model.decode(z) - xb
            obj(d if sigma is None else d * sigma, a).backward()
            opt.step()
        with torch.no_grad():
            outs.append(model.decode(z).detach())
    return torch.cat(outs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True, help="run names under --root/studies")
    p.add_argument("--root", default=str(OUT))
    p.add_argument("--label", default="refine")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--steps", nargs="+", type=int, default=[50, 200, 1000])
    p.add_argument("--lrs", nargs="+", type=float, default=[1e-3, 3e-3, 1e-2])
    p.add_argument("--splits", nargs="+", default=["val", "test"])
    p.add_argument("--pca-control", action="store_true",
                   help="also refine PCA-128 coefficients; the gain must be ~0")
    a = p.parse_args()

    device = torch.device("cuda", a.gpu); torch.cuda.set_device(device)
    data = te.load_mesh_tensors(device)
    root = Path(a.root)
    report = {"label": a.label, "runs": a.runs, "results": {}}

    for run in a.runs:
        model, ra = load_model(root / "studies" / run / "best.pt", device)
        d2 = te.load_mesh_tensors(device)
        if getattr(ra, "norm_mode", "std") == "center":
            for at in ("train", "val", "test"):
                setattr(d2, at, (getattr(d2, at) * d2.std).contiguous())
            d2.std = torch.ones_like(d2.std)
        sig = (d2.std / d2.std.mean()).detach() if getattr(ra, "metric_weighted_loss", False) else None
        rr = {}
        for split in a.splits:
            gt = d2.denormalize(d2.split(split))
            with torch.no_grad():
                base = torch.cat([model(d2.split(split)[i:i+64])
                                  for i in range(0, len(d2.split(split)), 64)])
            r0 = float(sc.vertex_rmse_mm(d2.denormalize(base), gt).mean())
            grid = {}
            for st in a.steps:
                for lr in a.lrs:
                    pred = refine(model, d2, ra, split, st, lr, sig)
                    r1 = float(sc.vertex_rmse_mm(d2.denormalize(pred), gt).mean())
                    grid[f"steps{st}_lr{lr:g}"] = r1
                    print(f"  {run:16s} {split:5s} steps={st:5d} lr={lr:.0e}  "
                          f"{r0:.6f} -> {r1:.6f}  ({100*(r0-r1)/r0:+.2f}%)")
            best = min(grid.values())
            rr[split] = {"encoder": r0, "grid": grid, "best": best,
                         "gain_pct": 100 * (r0 - best) / r0}
            print(f"  {run:16s} {split:5s} BEST {best:.6f}  gain {rr[split]['gain_pct']:.2f}%")
        report["results"][run] = rr
        del model; torch.cuda.empty_cache()

    if a.pca_control:
        # THE CONTROL. Not an assertion -- PCA is pushed through the SAME refine() loop the
        # learned models use, and the gain must come back ~0.
        #
        # Why 0 is the right answer, and why it is a real check rather than a tautology:
        # PCA-128 is fitted in mm, so z = U^T(x-mu) is the exact argmin of squared mm error
        # over its own 128-d subspace. The score, vertex_rmse_mm, is monotone in that same
        # squared mm error, so NO code in the subspace can beat the projection under ANY
        # refinement objective. A nonzero gain here therefore means the harness is buggy --
        # e.g. denormalising twice, scoring a different split, or refining against the input
        # instead of the target. That is exactly the failure mode the spiral's 6.9% would be
        # indistinguishable from without this run.
        Vtr = sc.load_split_vertices("train")
        mu_np = Vtr.reshape(len(Vtr), -1).mean(0)
        U_np = np.linalg.svd(Vtr.reshape(len(Vtr), -1) - mu_np, full_matrices=False)[2][:128]
        mu_t = torch.from_numpy(mu_np).to(device)
        U_t = torch.from_numpy(np.ascontiguousarray(U_np)).to(device)
        nv = Vtr.shape[1]

        class PCAModel(torch.nn.Module):
            """Same encode/decode surface as the learned models, operating in mm."""
            def encode(self, x):
                return (x.reshape(len(x), -1) - mu_t) @ U_t.T
            def decode(self, z):
                return (z @ U_t + mu_t).reshape(len(z), nv, 3)
            def forward(self, x):
                return self.decode(self.encode(x))

        pm = PCAModel().to(device).eval()
        # A data view in mm: mean 0 / std 1, so denormalize() is the identity and
        # vertex_rmse_mm is computed on exactly the millimetre coordinates PCA was fitted to.
        dp = te.load_mesh_tensors(device)
        for at in ("train", "val", "test"):
            setattr(dp, at, torch.from_numpy(sc.load_split_vertices(at)).to(device).contiguous())
        dp.mean = torch.zeros_like(dp.mean); dp.std = torch.ones_like(dp.std)

        ctrl = {}
        for s in a.splits:
            gt = dp.split(s)
            with torch.no_grad():
                r0 = float(sc.vertex_rmse_mm(pm(gt), gt).mean())
            grid = {}
            for st in a.steps:
                for lr in a.lrs:
                    # l2 is the objective the score is monotone in; sigma=None = unweighted.
                    pred = refine(pm, dp, argparse.Namespace(loss="l2"), s, st, lr, None)
                    r1 = float(sc.vertex_rmse_mm(pred, gt).mean())
                    grid[f"steps{st}_lr{lr:g}"] = r1
                    print(f"  PCA-128 control     {s:5s} steps={st:5d} lr={lr:.0e}  "
                          f"{r0:.6f} -> {r1:.6f}  ({100*(r0-r1)/r0:+.2f}%)")
            best = min(grid.values())
            ctrl[s] = {"encoder": r0, "grid": grid, "best": best,
                       "gain_pct": 100 * (r0 - best) / r0}
            print(f"  PCA-128 control     {s:5s} BEST {best:.6f}  gain "
                  f"{ctrl[s]['gain_pct']:.2f}%  (must be ~0)")
        report["pca_control"] = ctrl

    fp = OUT / "reports" / f"{a.label}.json"
    fp.parent.mkdir(parents=True, exist_ok=True)
    sc.atomic_write_json(fp, report)
    print(f"\nwrote {fp}")


if __name__ == "__main__":
    main()
