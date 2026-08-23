#!/usr/bin/env python3
"""Who gains more from data -- PCA or the mesh AE?

PCA-128 val on CALSNIC is still improving steeply at n=173 (4.555 @ n=80 -> 3.953 @ n=173),
so PCA is data-limited, not saturated. If the AE's curve is steeper, pooling more scans closes
the gap; if it is parallel or flatter, extra data helps PCA at least as much and the AE never
catches up. Both are fit on the SAME training subsets, so the comparison is paired.
"""
import argparse, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import calsnic_data as cd
from search_calsnic import Data, make_model, train

ap = argparse.ArgumentParser()
ap.add_argument("--arch", default="spiral"); ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--latent", type=int, default=128); ap.add_argument("--epochs", type=int, default=400)
a = ap.parse_args()

dev = torch.device("cuda", a.gpu); torch.cuda.set_device(dev)
data = Data(dev)
full_x, full_raw = data.x["train"].clone(), data.raw["train"].clone()
rows = cd.read_manifest()
Ftr = cd.load_split_vertices("train", rows=rows).reshape(173, -1).astype(np.float64)
Fva = cd.load_split_vertices("val", rows=rows).reshape(15, -1).astype(np.float64)
Sva = cd.scale_factors(rows, "val")

def pca_val(idx, k):
    X = Ftr[idx]; mean = X.mean(0); Xc = X - mean
    G = Xc @ Xc.T / max(1, len(Xc) - 1)
    ev, evec = np.linalg.eigh(G); o = np.argsort(ev)[::-1]
    comp = evec[:, o].T @ Xc
    comp /= np.maximum(np.linalg.norm(comp, axis=1, keepdims=True), 1e-12)
    comp = comp[:min(k, len(Xc) - 1)]
    return cd.corresponded_vertex_rmse_mm((Fva - mean) @ comp.T @ comp + mean, Fva, Sva).mean()

rng = np.random.default_rng(0)
print(f"# arch={a.arch} latent={a.latent} epochs={a.epochs}", flush=True)
print(f"{'n_train':>8}{'AE_val':>9}{'PCA_val':>9}{'AE/PCA':>8}", flush=True)
for n in (40, 80, 120, 173):
    idx = rng.choice(173, n, replace=False) if n < 173 else np.arange(173)
    t = torch.as_tensor(idx, device=dev)
    data.x["train"], data.raw["train"] = full_x[t], full_raw[t]
    cfg = dict(n_levels=5, base_channels=32, seq_length=13, dilation=1, dropout=0.1,
               adaptive_levels=1, dynamic_seq_frac=0.5)
    torch.manual_seed(1); torch.cuda.empty_cache()
    m, _ = make_model(cfg, a.arch, a.latent, dev)
    best, _, _ = train(m, data, epochs=a.epochs, batch_size=8, lr=3e-4, lr_decay=0.99,
                       decay_step=10, weight_decay=1e-5, noise_std=0.04,
                       scheduler_type="cosine", patience=0, min_epochs=0, eval_every=50, seed=1)
    p = pca_val(idx, a.latent)
    print(f"{n:>8}{best:9.3f}{p:9.3f}{best/p:8.3f}", flush=True)
    del m; torch.cuda.empty_cache()
