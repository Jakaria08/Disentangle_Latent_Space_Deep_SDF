#!/usr/bin/env python3
"""Train the LAMM autoencoder on the ADNI left hippocampus.

Metric is the project-standard per-coordinate vertex RMSE in mm, so results are directly
comparable to PCA-128 (0.033668), SpiralNet++ (0.036784), Adaptive (0.037237) and the tuned
MeshMAE transformer (0.037867). NOTE that LAMM's own paper reports mean per-vertex EUCLIDEAN
distance -- a factor of sqrt(3) larger -- so its published numbers must never be put in the
same table as these without conversion.
"""
from __future__ import annotations

import argparse, collections, json, math, sys, time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
SPIRAL = Path("/home/jakaria/INR/Deep3DComp/examples/"
              "ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task_spiral_ae_v1/scripts")
for p in (str(HERE), str(SPIRAL)):
    if p not in sys.path: sys.path.insert(0, p)

import spiral_common as sc
import train_eval as te
from lamm_model import (LAMMAutoencoder, MultiScaleLAMM, build_patches,
                        build_patch_members)

OUT = Path("/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task_lamm_ae_v1")

# Region counts reachable from the decimation hierarchy. LAMM splits 12k vertices into 11
# regions; the first search here used 86 on 2746 vertices, an 8x longer token sequence than
# the paper, and that was never examined. [2,4,4,4] shares its first four levels with the
# already-validated [2,4,4], so K=86 is bit-identical to every earlier run.
SCALE_SOURCE = {11: ([4, 4, 4, 4], 4),  43: ([4, 4, 4, 4], 3), 172: ([4, 4, 4, 4], 2),
                22: ([2, 4, 4, 4], 4),  86: ([2, 4, 4, 4], 3), 344: ([2, 4, 4, 4], 2)}
REF = {"pca128_val": 0.033668, "spiralnet128_val": 0.036784,
       "adaptive128_val": 0.037237, "meshmae_tuned_val": 0.037867}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", default="lamm_z128")
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--latent", type=int, default=128)
    p.add_argument("--backbone", choices=("transformer", "mlpmixer"), default="transformer")
    p.add_argument("--patch-level", type=int, default=3, help="decimation level for regions")
    p.add_argument("--dim", type=int, default=256, help="D; LAMM uses 512 on 6-10k meshes")
    p.add_argument("--enc-depth", type=int, default=5)     # LAMM: 5
    p.add_argument("--dec-depth", type=int, default=3)     # LAMM: 3
    p.add_argument("--heads", type=int, default=8)         # LAMM: 8 heads, dim_head 64
    p.add_argument("--dim-head", type=int, default=64,
                   help="attention head width; LAMM fixes 64 INDEPENDENT of D, so inner "
                        "width is heads*64. The first search used dim//heads instead.")
    p.add_argument("--scales", default="86",
                   help="comma-separated region counts, coarse first, e.g. 86 or 11,86,344. "
                        "More than one builds the multiscale residual model.")
    p.add_argument("--region-mode", choices=("raw", "both"), default="raw",
                   help="'both' appends the region's mean/max/min/std to its raw member "
                        "coordinates. LAMM uses raw only; the same change to the MeshMAE "
                        "tokenizer gave -7.7% val at ZERO training-error cost.")
    p.add_argument("--latent-split", default="",
                   help="multiscale only: comma-separated latent dims per scale, coarse "
                        "first, summing to --latent (e.g. 96,32). Empty = even split.")
    p.add_argument("--residual", action=argparse.BooleanOptionalAction, default=True,
                   help="multiscale only: X = X_coarse + R_medium + R_fine with the partial "
                        "sums supervised, instead of every scale reconstructing X")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--share-regions", action="store_true",
                   help="share tokenizer/head weights across regions (LAMM does NOT)")
    p.add_argument("--latent-mode", choices=("id_token", "flatten"), default="id_token")
    p.add_argument("--deep-sup", type=float, default=0.0,
                   help="weight on L1 at intermediate decoder layers (LAMM uses multilayer L1)")
    p.add_argument("--lr", type=float, default=1e-4)       # LAMM: 1e-4 -> 1e-6
    p.add_argument("--lr-final", type=float, default=1e-6)
    p.add_argument("--warmup-epochs", type=int, default=10)   # LAMM: 10
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=32)   # LAMM: 32
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--min-epochs", type=int, default=80)
    p.add_argument("--subject-weight", action=argparse.BooleanOptionalAction, default=False,
                   help="weight each training scan by 1/n_visits(subject). The cohort is 2037 "
                        "scans from only 475 subjects imaged 2-11 times, so per-scan weighting "
                        "gives an effective (sum n)^2/sum n^2 = 397.0 subjects against the 475 "
                        "actually present -- 16%% of the effective sample size discarded by an "
                        "accidental choice. It also matches the evaluation distribution, since "
                        "val and test are subject-clustered (61 subjects each).")
    p.add_argument("--norm-mode", choices=("std", "center"), default="std",
                   help="'std' = (x-mu)/sigma, this project's convention. 'center' = x-mu "
                        "only, which is what LAMM's paper does. sigma spans 0.1223-1.2365 mm "
                        "(10.1x) across coordinates, so dividing by it reweights the loss.")
    p.add_argument("--metric-weighted-loss", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="scale the residual by sigma before the loss. The reported metric is "
                        "per-coordinate RMSE in MM, i.e. |p-t|*sigma, but the loss is |p-t| -- "
                        "so a coordinate enters the objective at 1/sigma relative to its "
                        "weight in the score, up to 10.1x wrong. This makes them agree "
                        "exactly, while keeping sigma-normalisation for conditioning.")
    p.add_argument("--loss", choices=("l1", "l2", "huber"), default="l1",
                   help="training loss. Everything in this project trains on L1 (inherited "
                        "from guided_vae) but is SELECTED and REPORTED on per-coordinate "
                        "RMSE, which is L2. That mismatch has never been tested.")
    p.add_argument("--huber-delta", type=float, default=0.1)
    p.add_argument("--mixup-alpha", type=float, default=0.0,
                   help="cross-subject mixup: x = lam*x_i + (1-lam)*x_j with lam~Beta(a,a), "
                        "the AE target being the interpolant itself. 0 disables. LAMM has had "
                        "NO augmentation of any kind, while its whole remaining deficit is the "
                        "generalisation gap and the root cause is 475 independent subjects -- "
                        "mixup manufactures virtual ones without a new manifest.")
    p.add_argument("--mixup-prob", type=float, default=1.0,
                   help="fraction of batches to mix")
    p.add_argument("--time-budget", type=float, default=0.0)
    p.add_argument("--max-params", type=float, default=0.0,
                   help="exit 3 without training if the model exceeds this many parameters. "
                        "Lets a search explore scale/width combinations freely while "
                        "rejecting oversized ones UP FRONT, instead of letting the time "
                        "budget truncate them -- truncation biases the sampler against "
                        "capacity, which is exactly the variable under test here.")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


class EMA:
    """Shadow weights tracked alongside training and evaluated separately, so a run can never
    be worse than not using it. Measured at +1.1% on the MeshMAE encoder (S9 vs S7)."""

    def __init__(self, model, decay):
        self.decay = float(decay)
        self.pairs = [(p, p.detach().clone()) for p in model.parameters()]

    @torch.no_grad()
    def update(self):
        for p, s in self.pairs:
            s.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @contextmanager
    def applied(self):
        backup = [p.detach().clone() for p, _ in self.pairs]
        try:
            with torch.no_grad():
                for p, s in self.pairs: p.data.copy_(s)
            yield
        finally:
            with torch.no_grad():
                for (p, _), b in zip(self.pairs, backup): p.data.copy_(b)


def regions_for(k, V):
    if int(k) not in SCALE_SOURCE:
        raise SystemExit(f"region count {k} not available; choose from "
                         f"{sorted(SCALE_SOURCE)}")
    ds, level = SCALE_SOURCE[int(k)]
    centres = sc.get_transform(ds)["vertices"][level]
    pid, K = build_patches(V, centres)
    if K != int(k):
        raise SystemExit(f"expected {k} regions from ds={ds} level {level}, got {K}")
    return build_patch_members(pid, K)


def build(args, device):
    V, _ = sc.build_template()
    ks = [int(t) for t in str(args.scales).split(",") if t.strip()]
    scales = [regions_for(k, V) for k in ks]
    if len(scales) == 1:
        midx, mmask = scales[0]
        K = int(mmask.shape[0])
        m = LAMMAutoencoder(member_idx=midx, member_mask=mmask, n_vertices=V.shape[0],
                            latent=args.latent, dim=args.dim, enc_depth=args.enc_depth,
                            dec_depth=args.dec_depth, heads=args.heads,
                            dim_head=args.dim_head, dropout=args.dropout,
                            backbone=args.backbone, share_regions=args.share_regions,
                            latent_mode=args.latent_mode,
                            region_mode=args.region_mode).to(device)
    else:
        K = sum(int(mm.shape[0]) for _, mm in scales)
        m = MultiScaleLAMM(scales=scales, n_vertices=V.shape[0], latent=args.latent,
                           dim=args.dim, enc_depth=args.enc_depth, dec_depth=args.dec_depth,
                           heads=args.heads, dim_head=args.dim_head, dropout=args.dropout,
                           backbone=args.backbone, share_regions=args.share_regions,
                           residual=args.residual, region_mode=args.region_mode,
                           latent_split=([int(v) for v in args.latent_split.split(",")]
                                         if args.latent_split else None)).to(device)
    mmask = scales[-1][1]
    info = {"backbone": args.backbone, "scales": ks, "n_regions": K,
            "multiscale": len(scales) > 1, "residual": bool(args.residual),
            "dim_head": args.dim_head, "region_mode": args.region_mode,
            "latent_split_arg": args.latent_split,
            "patch_level": args.patch_level,
            "dim": args.dim, "enc_depth": args.enc_depth, "dec_depth": args.dec_depth,
            "heads": args.heads, "dropout": args.dropout, "latent": args.latent,
            "share_regions": bool(args.share_regions), "latent_mode": args.latent_mode,
            "deep_sup": args.deep_sup, "ema_decay": args.ema_decay,
            "mixup_alpha": args.mixup_alpha, "mixup_prob": args.mixup_prob,
            "loss": args.loss, "subject_weight": bool(args.subject_weight),
            "norm_mode": args.norm_mode,
            "metric_weighted_loss": bool(args.metric_weighted_loss),
            "region_width": int(mmask.shape[1]),
            "verts_per_region_mean": float(mmask.sum(1).float().mean()),
            "params": m.param_breakdown()}
    return m, info


def recon_loss(pred, target, a, w=None, sigma=None):
    """Per-sample loss, then a (optionally weighted) mean over the batch.

    `sigma` scales the RESIDUAL, which is exactly right for every loss here: L1 gives
    |d|*sigma, L2 gives (d*sigma)^2, huber gives huber(d*sigma) -- all of them the mm-space
    quantity. With sigma=None and w=None this is numerically identical to reduction='mean'.
    """
    d = pred - target
    if sigma is not None:
        d = d * sigma
    if a.loss == "l2":
        per = d.pow(2).mean(dim=(1, 2))
    elif a.loss == "huber":
        per = F.huber_loss(d, torch.zeros_like(d), delta=a.huber_delta,
                           reduction="none").mean(dim=(1, 2))
    else:
        per = d.abs().mean(dim=(1, 2))
    return per.mean() if w is None else (per * w).sum() / w.sum()


def lr_at(ep, args, total):
    """LAMM: linear warmup then one-cycle cosine decay from lr to lr_final."""
    if ep <= args.warmup_epochs:
        return args.lr * ep / max(1, args.warmup_epochs)
    t = (ep - args.warmup_epochs) / max(1, total - args.warmup_epochs)
    return args.lr_final + 0.5 * (args.lr - args.lr_final) * (1 + math.cos(math.pi * min(t, 1.0)))


@torch.no_grad()
def val_rmse(model, data, split="val", bs=32):
    model.eval(); x = data.split(split)
    pred = data.denormalize(torch.cat([model(x[i:i+bs]) for i in range(0, len(x), bs)]))
    return float(sc.vertex_rmse_mm(pred, data.denormalize(x)).mean())


def main():
    a = parse_args()
    device = torch.device("cuda", a.gpu); torch.cuda.set_device(device)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    run = OUT / "studies" / a.run_name; run.mkdir(parents=True, exist_ok=True)
    log_fp = OUT / "logs" / f"{a.run_name}.log"; log_fp.parent.mkdir(parents=True, exist_ok=True)
    lg = open(log_fp, "a", buffering=1)
    def log(m): print(m, flush=True); lg.write(m + "\n")

    model, info = build(a, device)
    if a.max_params > 0 and model.num_parameters() > a.max_params:
        print(f"REJECT {a.run_name}: {model.num_parameters()/1e6:.2f}M params "
              f"> --max-params {a.max_params/1e6:.2f}M", flush=True)
        raise SystemExit(3)
    data = te.load_mesh_tensors(device)
    if a.norm_mode == "center":
        # undo the sigma division; denormalize() stays consistent because std becomes 1,
        # so every reported metric keeps its meaning.
        for attr in ("train", "val", "test"):
            setattr(data, attr, (getattr(data, attr) * data.std).contiguous())
        data.std = torch.ones_like(data.std)
    sigma = None
    if a.metric_weighted_loss:
        sigma = (data.std / data.std.mean()).detach()      # mean 1, so the loss scale is kept
    log("=" * 96)
    log(f"{a.run_name}: {json.dumps(info)}")
    log(f"trainable params {model.num_parameters()/1e6:.3f}M")
    log(f"PCA-128 {REF['pca128_val']:.6f} | spiralnet {REF['spiralnet128_val']:.6f} "
        f"| meshmae tuned {REF['meshmae_tuned_val']:.6f}")
    log("=" * 96)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    epochs = 4 if a.smoke else a.epochs
    ema = EMA(model, a.ema_decay) if a.ema_decay > 0 else None
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    xt = data.train; n = len(xt)
    sw = None
    if a.subject_weight:
        tr_rows = sc.split_rows(sc.read_manifest(), "train")   # same order as data.train
        cnt = collections.Counter(r["subject_id"] for r in tr_rows)
        sw = torch.tensor([1.0 / cnt[r["subject_id"]] for r in tr_rows],
                          dtype=torch.float32, device=device)
        sw = sw / sw.mean()
        eff = len(tr_rows) ** 2 / sum(v * v for v in cnt.values())
        log(f"subject weighting ON: {len(tr_rows)} scans, {len(cnt)} subjects, "
            f"effective subjects {eff:.1f} -> {len(cnt)} ({len(cnt)/eff:.3f}x)")
    best, best_state, best_ep, stale, hist = float("inf"), None, -1, 0, []
    best_source = "live"
    t0 = time.time()

    for ep in range(1, epochs + 1):
        for grp in opt.param_groups: grp["lr"] = lr_at(ep, a, epochs)
        model.train()
        perm = torch.randperm(n, generator=g).to(device)
        for i in range(0, n, a.batch_size):
            idx = perm[i:i+a.batch_size]
            b = xt[idx]
            wb = None if sw is None else sw[idx]
            if a.mixup_alpha > 0 and float(torch.rand(1)) < a.mixup_prob and b.size(0) > 1:
                lam = float(np.random.beta(a.mixup_alpha, a.mixup_alpha))
                p2 = torch.randperm(b.size(0), device=b.device)
                b = lam * b + (1.0 - lam) * b[p2]
                if wb is not None:            # the mixed sample inherits a mixed weight
                    wb = lam * wb + (1.0 - lam) * wb[p2]
            opt.zero_grad(set_to_none=True)
            if a.deep_sup > 0:
                outs = model(b, all_layers=True)
                loss = recon_loss(outs[-1], b, a, wb, sigma)
                if len(outs) > 1:
                    loss = loss + a.deep_sup * sum(
                        recon_loss(o, b, a, wb, sigma) for o in outs[:-1]) / (len(outs) - 1)
            else:
                loss = recon_loss(model(b), b, a, wb, sigma)
            loss.backward(); opt.step()
            if ema is not None: ema.update()

        if ep % a.eval_every == 0 or ep == epochs:
            cur = val_rmse(model, data)
            cur_ema = None
            if ema is not None:
                with ema.applied(): cur_ema = val_rmse(model, data)
            improved = False
            for score, src in ((cur, "live"), (cur_ema, "ema")):
                if score is not None and score < best - 1e-9:
                    best, best_ep, best_source, improved = score, ep, src, True
            if improved:
                stale = 0
                with (ema.applied() if best_source == "ema" else nullcontext()):
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}
            else:
                stale += 1
            hist.append({"epoch": ep, "val_rmse_mm": cur, "val_rmse_mm_ema": cur_ema,
                         "best": best, "best_source": best_source,
                         "lr": opt.param_groups[0]["lr"]})
            emastr = "" if cur_ema is None else f" ema={cur_ema:.6f}"
            log(f"  [train] epoch {ep:4d} val_rmse_mm={cur:.6f}{emastr} "
                f"best={best:.6f}({best_source}) ({time.time()-t0:.0f}s)")
            if ep >= a.min_epochs and a.patience and stale >= a.patience:
                log(f"  early stop at epoch {ep}"); break
            if a.time_budget > 0 and (time.time() - t0) > a.time_budget:
                log(f"  stopped at epoch {ep} on the {a.time_budget:.0f}s budget"); break

    if best_state is not None: model.load_state_dict(best_state)

    @torch.no_grad()
    def split_metrics(sp, bs=32):
        model.eval(); x = data.split(sp)
        pred = torch.cat([model(x[i:i+bs]).detach() for i in range(0, len(x), bs)])
        return sc.reconstruction_metrics(data.denormalize(pred), data.denormalize(x),
                                         faces=data.faces)
    metrics = {sp: split_metrics(sp) for sp in sc.SPLITS}
    torch.save({"run": a.run_name, "args": vars(a), "info": info,
                "best_val_rmse_mm": best, "best_epoch": best_ep, "best_source": best_source,
                "metrics": metrics, "model_state_dict": best_state}, run / "best.pt")
    sc.atomic_write_json(run / "summary.json",
        {"run": a.run_name, "args": vars(a), "info": info, "best_val_rmse_mm": best,
         "best_epoch": best_ep, "best_source": best_source, "metrics": metrics,
         "history": hist, "n_params": model.num_parameters(), "reference": REF})
    for sp in sc.SPLITS:
        log(f"  {sp:5s} rmse_mm={metrics[sp]['vertex_rmse_mm_mean']:.6f}")
    log(f"wrote {run}")
    lg.close()


if __name__ == "__main__":
    main()
