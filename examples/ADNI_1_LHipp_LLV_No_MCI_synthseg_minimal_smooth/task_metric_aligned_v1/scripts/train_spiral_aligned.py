#!/usr/bin/env python3
"""SpiralNet++ / Adaptive Spiral with the metric-aligned objective. EXPERIMENT C.

WHY THIS FILE EXISTS. Every model in this study trains L1 on (x-mu)/sigma but is scored on
per-coordinate RMSE in MILLIMETRES, i.e. |p-t|*sigma. sigma spans 0.1223-1.2365 mm across
coordinates (10.1x), so a coordinate enters the objective at 1/sigma relative to its weight
in the score. On LAMM, correcting that was worth 10.8% -- 0.038051 -> 0.033916, roughly 16
sigma against a measured seed noise floor of 0.77%.

SpiralNet++ (0.036784), Adaptive (0.037237) and MeshMAE (0.037867) were all produced under
the same defect and have never been re-run. Until they are, the comparison table is not
trustworthy in EITHER direction: if SpiralNet++ gains comparably it lands near 0.0331 and
leads again. PCA is unaffected -- it has no training objective.

This is a NEW file. It does not touch task_spiral_ae_v1; the published checkpoints and
studies there stay exactly as they are.

Defaults reproduce spiralnet_z128_v7 trial_0036 (val 0.036784), the best SpiralNet++ on
record, so arm N1 here is a re-run of the published result and acts as the control.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SPIRAL = Path("/home/jakaria/INR/Deep3DComp/examples/"
              "ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task_spiral_ae_v1/scripts")
if str(SPIRAL) not in sys.path:
    sys.path.insert(0, str(SPIRAL))

import spiral_common as sc
import train_eval as te
from network import build_model, resolve_conv_types

OUT = Path("/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task_metric_aligned_v1")
REF = {"pca128": 0.033668, "spiralnet128_published": 0.036784,
       "adaptive128_published": 0.037237, "lamm_aligned": 0.033916}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    p.add_argument("--out-root", default=str(OUT))
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--conv-type", choices=("spiral", "adaptive"), default="spiral")
    # --- spiralnet_z128_v7 trial_0036 (val 0.036784) ---
    p.add_argument("--ds-factors", default="2,4,4")
    p.add_argument("--base-channels", type=int, default=96)
    p.add_argument("--seq-length", type=int, default=17)
    p.add_argument("--dilation", type=int, default=2)
    p.add_argument("--latent", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.05)
    # adaptive_z128_v7 trial_0023 (val 0.037237): adaptive on the COARSEST level only, with
    # dyn_seq = round(dyn_frac * n_vertices at that level) -> [1, 1, 187] for ds=2,4,4.
    p.add_argument("--adaptive-levels", type=int, default=1)
    p.add_argument("--dyn-frac", type=float, default=0.5431357742843035)
    p.add_argument("--lr", type=float, default=2.3035207977537586e-4)
    p.add_argument("--lr-decay", type=float, default=0.9116161230466987)
    p.add_argument("--decay-step", type=int, default=6)
    p.add_argument("--weight-decay", type=float, default=1.8340316433865296e-4)
    p.add_argument("--noise-std", type=float, default=0.06)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=400)
    # --- the correction under test ---
    p.add_argument("--norm-mode", choices=("std", "center"), default="std")
    p.add_argument("--metric-weighted-loss", action=argparse.BooleanOptionalAction,
                   default=False)
    # --- EXPERIMENT E: semi-amortisation, ported from task_lamm_ae_v1/scripts/train_lamm.py ---
    p.add_argument("--semi-amortized", type=int, default=0,
                   help="K inner gradient steps refining z before a SECOND decoder loss. "
                        "Post-hoc refinement (refine_eval.py) measured this model's "
                        "amortisation gap at 6.10%% val / 6.90%% test -- the largest in the "
                        "study, and 3.8x LAMM's 1.61%%. Semi-amortisation was only ever run "
                        "on LAMM (SA_K1/K3), where it moved nothing (0.033665 -> 0.033665); "
                        "it has never been applied to the model that actually has the gap.")
    p.add_argument("--sa-opt", choices=("adam", "sgd"), default="adam",
                   help="inner optimiser for z. MUST be adam. Measured on C_N3_sp_s1, K=3 "
                        "plain-SGD steps at lr 3e-3 move z by ||dz||=2e-6 against ||z||=17.9 "
                        "and change the loss by +0.000%%; on the LAMM checkpoints ||dz|| is "
                        "exactly 0.0 -- the update underflows float32. Adam normalises by "
                        "sqrt(v), so 3 steps at the same lr move ||dz||=0.098 for -1.7%%. "
                        "'sgd' exists only to reproduce the inert LAMM SA_K1/K3/K5 runs.")
    p.add_argument("--sa-lr", type=float, default=1e-2, help="inner step size for z")
    p.add_argument("--sa-weight", type=float, default=1.0, help="weight on the refined term")
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--min-epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--time-budget", type=float, default=0.0)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def recon_loss(pred, target, sigma=None):
    d = pred - target
    if sigma is not None:
        d = d * sigma            # |d|*sigma is exactly the mm-space L1
    return d.abs().mean()


def main():
    a = parse_args()
    device = torch.device("cuda", a.gpu); torch.cuda.set_device(device)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = sc.require_bulk_path(a.out_root, "aligned spiral output root")
    run = out / "studies" / a.run_name; run.mkdir(parents=True, exist_ok=True)
    log_fp = out / "logs" / f"{a.run_name}.log"; log_fp.parent.mkdir(parents=True, exist_ok=True)
    lg = open(log_fp, "a", buffering=1)
    def log(m): print(m, flush=True); lg.write(m + "\n")

    ds = [int(v) for v in a.ds_factors.split(",")]
    oc = [a.base_channels, a.base_channels, a.base_channels * 2][:len(ds)]
    tr = sc.get_transform(ds)
    conv_types = resolve_conv_types(len(ds), a.conv_type, a.adaptive_levels)
    dyn_seq = [max(1, round(a.dyn_frac * int(tr["vertices"][i].shape[0])))
               if conv_types[i] == "adaptive" else 1 for i in range(len(ds))]
    sp, dyn, dn, up = sc.build_spiral_stack(tr, a.seq_length, a.dilation, dyn_seq, device)
    model = build_model(tr, sp, dyn, dn, up, oc, a.latent, a.conv_type, a.adaptive_levels,
                        conv_types=conv_types, dropout=a.dropout, linear_skip=False).to(device)

    data = te.load_mesh_tensors(device)
    if a.norm_mode == "center":
        for at in ("train", "val", "test"):
            setattr(data, at, (getattr(data, at) * data.std).contiguous())
        data.std = torch.ones_like(data.std)
    sigma = (data.std / data.std.mean()).detach() if a.metric_weighted_loss else None

    info = {"conv_type": a.conv_type, "conv_types": conv_types, "dyn_seq": dyn_seq,
            "ds_factors": ds, "out_channels": oc,
            "seq_length": a.seq_length, "dilation": a.dilation, "latent": a.latent,
            "dropout": a.dropout, "noise_std": a.noise_std, "norm_mode": a.norm_mode,
            "metric_weighted_loss": bool(a.metric_weighted_loss), "seed": a.seed,
            "semi_amortized": int(a.semi_amortized), "sa_opt": a.sa_opt,
            "sa_lr": a.sa_lr, "sa_weight": a.sa_weight,
            "n_params": sum(p.numel() for p in model.parameters())}
    log("=" * 96)
    log(f"{a.run_name}: {json.dumps(info)}")
    log(f"PCA-128 {REF['pca128']:.6f} | spiralnet published {REF['spiralnet128_published']:.6f} "
        f"| LAMM aligned {REF['lamm_aligned']:.6f}")
    log("=" * 96)

    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.StepLR(opt, max(1, a.decay_step), gamma=a.lr_decay)
    epochs = 6 if a.smoke else a.epochs
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    xt = data.train; n = len(xt)
    best, best_state, best_ep, stale, hist = float("inf"), None, -1, 0, []
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=g).to(device)
        for i in range(0, n, a.batch_size):
            b = xt[perm[i:i + a.batch_size]]
            inp = b if a.noise_std <= 0 else b + a.noise_std * torch.randn_like(b)
            opt.zero_grad(set_to_none=True)
            if a.semi_amortized > 0:
                # Amortised term trains encoder+decoder; refined term trains the DECODER on
                # codes of the kind it is actually given at test time. z is detached between
                # inner steps, so no second-order graph is built -- cheap, and the encoder
                # still gets its gradient from the first term. The inner steps descend toward
                # the CLEAN target b, matching refine_eval.py's objective.
                z = model.encode(inp)
                loss = recon_loss(model.decode(z), b, sigma)
                zr = z.detach().clone().requires_grad_(True)
                inner = (torch.optim.Adam([zr], lr=a.sa_lr) if a.sa_opt == "adam" else None)
                for _ in range(a.semi_amortized):
                    li = recon_loss(model.decode(zr), b, sigma)
                    # autograd.grad, not backward(): the inner loop must not accumulate into
                    # the model's .grad, which the outer optimiser is about to consume.
                    gz, = torch.autograd.grad(li, zr)
                    if inner is None:
                        zr = (zr - a.sa_lr * gz).detach().requires_grad_(True)
                    else:
                        zr.grad = gz          # same tensor object, so Adam keeps its state
                        inner.step()
                loss = loss + a.sa_weight * recon_loss(model.decode(zr.detach()), b, sigma)
            else:
                loss = recon_loss(model(inp), b, sigma)
            loss.backward()
            opt.step()
        sched.step()
        if ep % a.eval_every == 0 or ep == epochs:
            cur = te.val_rmse(model, data)
            if cur < best - 1e-9:
                best, best_ep, stale = cur, ep, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            hist.append({"epoch": ep, "val_rmse_mm": cur, "best": best})
            log(f"  epoch {ep:4d} val_rmse_mm={cur:.6f} best={best:.6f} ({time.time()-t0:.0f}s)")
            if ep >= a.min_epochs and a.patience and stale >= a.patience:
                log(f"  early stop at epoch {ep}"); break
            if a.time_budget > 0 and (time.time() - t0) > a.time_budget:
                log(f"  stopped at epoch {ep} on the {a.time_budget:.0f}s budget"); break

    if best_state is not None:
        model.load_state_dict(best_state)
    metrics = {s: te.evaluate(model, data, s, with_volume=True) for s in sc.SPLITS}
    torch.save({"run": a.run_name, "args": vars(a), "info": info, "best_val_rmse_mm": best,
                "best_epoch": best_ep, "metrics": metrics,
                "model_state_dict": best_state}, run / "best.pt")
    sc.atomic_write_json(run / "summary.json",
        {"run": a.run_name, "args": vars(a), "info": info, "best_val_rmse_mm": best,
         "best_epoch": best_ep, "metrics": metrics, "history": hist, "reference": REF})
    for s in sc.SPLITS:
        log(f"  {s:5s} rmse_mm={metrics[s]['vertex_rmse_mm_mean']:.6f}")
    log(f"wrote {run}")
    lg.close()


if __name__ == "__main__":
    main()
