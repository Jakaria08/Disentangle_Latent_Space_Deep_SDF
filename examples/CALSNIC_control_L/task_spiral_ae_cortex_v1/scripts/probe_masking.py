#!/usr/bin/env python3
"""Does icosahedral patch masking (MAE-style augmentation) help CALSNIC reconstruction?

Motivation. The plain AE has converged and is insensitive to every knob tried: capacity
(base 16-96 flat), training length (250 vs 2500 epochs identical), and latent size
(25D->128D gains only 5%). Those are signatures of a model that solved an easy problem --
full-mesh autoencoding admits a near-identity shortcut -- not one starved of capacity.

Masking removes that shortcut. A fraction of icosahedral patches is deleted from the INPUT;
the target stays the clean full mesh, so the network must infer missing cortex from context.
It is also strong structured augmentation: 173 meshes x many mask draws. This is the cheap
one-hour version of surface-MAE pretraining -- if masking-as-augmentation does nothing here,
a full two-stage sMAE pipeline almost certainly will not either.

Patches come free from the mesh's icosphere structure: the 162-vertex level of the nested
hierarchy gives 162 contiguous patches of ~253 vertices each.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent))
import calsnic_data as cd
from search_calsnic import Data, make_model, rmse_mm

ap = argparse.ArgumentParser()
ap.add_argument("--arch", required=True, choices=["spiral", "adaptive"])
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--latent", type=int, default=150)
ap.add_argument("--base", type=int, default=32)
ap.add_argument("--epochs", type=int, default=400)
ap.add_argument("--n-patches", type=int, default=162)
ap.add_argument("--ratios", nargs="+", type=float, default=[0.0, 0.25, 0.50, 0.75])
ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
a = ap.parse_args()

dev = torch.device("cuda", a.gpu); torch.cuda.set_device(dev)
data = Data(dev)

# ---- patch assignment: nearest of the 162 coarse icosphere vertices -------------------
from scipy.spatial import cKDTree
tmpl_v, _ = cd.build_template()
tr = cd.get_transform_ico(5)
centers = tr["vertices"][tr["level_sizes"].index(a.n_patches)]
patch_id = torch.from_numpy(cKDTree(centers).query(tmpl_v)[1].astype(np.int64)).to(dev)
n_patch = int(patch_id.max().item()) + 1
counts = torch.bincount(patch_id)
print(f"# {a.arch} latent={a.latent} base={a.base} epochs={a.epochs}", flush=True)
print(f"# {n_patch} patches, {counts.float().mean():.0f}+-{counts.float().std():.0f} vertices each",
      flush=True)
# PCA reference at this latent (interpolated between the measured 128 and 150/172 points)
PCA_VAL = {25: 4.703, 128: 3.953, 150: 3.905, 172: 3.864}
ref = PCA_VAL.get(a.latent, 3.905)
print(f"# PCA-{a.latent} val reference = {ref:.3f} mm", flush=True)
print(f"{'mask':>6}{'seed':>6}{'val_mm':>9}{'vs_PCA':>8}{'min':>7}", flush=True)


def train_masked(model, ratio, seed, epochs):
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    g = torch.Generator(device="cpu").manual_seed(seed)
    xt = data.x["train"]; n = len(xt); best = float("inf")
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=g).to(dev)
        for i in range(0, n, 8):
            tgt = xt[perm[i:i + 8]]
            if ratio > 0:
                # independent patch mask per sample; masked vertices zeroed in the INPUT only
                keep = (torch.rand(tgt.size(0), n_patch, device=dev) >= ratio).float()
                inp = tgt * keep[:, patch_id].unsqueeze(-1)
            else:
                inp = tgt
            opt.zero_grad(set_to_none=True)
            F.l1_loss(model(inp), tgt).backward()   # target is the CLEAN full mesh
            opt.step()
        sch.step()
        if ep % 50 == 0 or ep == epochs:
            best = min(best, float(rmse_mm(model, data, "val").mean()))
    return best


for ratio in a.ratios:
    for seed in a.seeds:
        cfg = dict(n_levels=5, base_channels=a.base, seq_length=13, dilation=1, dropout=0.1,
                   adaptive_levels=1, dynamic_seq_frac=0.5)
        torch.manual_seed(seed); torch.cuda.empty_cache()
        m, _ = make_model(cfg, a.arch, a.latent, dev)
        t0 = time.time()
        v = train_masked(m, ratio, seed, a.epochs)
        print(f"{ratio:6.2f}{seed:6d}{v:9.3f}{v/ref:8.3f}{(time.time()-t0)/60:7.1f}", flush=True)
        del m; torch.cuda.empty_cache()
