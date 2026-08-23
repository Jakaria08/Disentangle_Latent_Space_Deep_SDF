#!/usr/bin/env python3
"""Short capacity ladder on CALSNIC: where does the optimum sit, and is 25D viable?

ADNI showed a U-shaped capacity curve with a minimum at base~96 on 2037 training meshes.
CALSNIC has 173 -- 12x fewer -- so the optimum should sit lower. This probe finds roughly
where, cheaply, before committing to a long search.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import calsnic_data as cd
from search_calsnic import Data, make_model, train, rmse_mm

p = argparse.ArgumentParser()
p.add_argument("--arch", required=True, choices=["spiral", "adaptive"])
p.add_argument("--gpu", type=int, default=0)
p.add_argument("--epochs", type=int, default=250)
p.add_argument("--latents", nargs="+", type=int, default=[25, 128])
p.add_argument("--bases", nargs="+", type=int, default=[16, 32, 48, 64, 96])
a = p.parse_args()

dev = torch.device("cuda", a.gpu)
torch.cuda.set_device(dev)
data = Data(dev)
PCA = {25: (4.703, 4.947), 128: (3.953, 4.017), 172: (3.864, 3.844)}

print(f"# arch={a.arch} epochs={a.epochs}  (PCA val: 25D={PCA[25][0]}  128D={PCA[128][0]}  "
      f"172D_ceiling={PCA[172][0]})", flush=True)
print(f"{'latent':>7}{'base':>6}{'params':>10}{'peak':>8}{'val_mm':>9}{'vs_PCA':>8}"
      f"{'vs_PCA128':>10}{'min':>7}", flush=True)
for latent in a.latents:
    for base in a.bases:
        cfg = dict(n_levels=5, base_channels=base, seq_length=13, dilation=1, dropout=0.1,
                   adaptive_levels=1, dynamic_seq_frac=0.5)
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)
            torch.manual_seed(1)
            m, meta = make_model(cfg, a.arch, latent, dev)
            t0 = time.time()
            best, st, info = train(m, data, epochs=a.epochs, batch_size=8, lr=3e-4,
                                   lr_decay=0.99, decay_step=10, weight_decay=1e-5,
                                   noise_std=0.04, scheduler_type="cosine",
                                   patience=0, min_epochs=0, eval_every=25, seed=1)
            pk = torch.cuda.max_memory_allocated(dev)/1e9
            print(f"{latent:>7}{base:>6}{m.num_parameters()/1e6:9.2f}M{pk:7.1f}G{best:9.3f}"
                  f"{best/PCA[latent][0]:8.3f}{best/PCA[128][0]:10.3f}"
                  f"{(time.time()-t0)/60:7.1f}", flush=True)
            del m; torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"{latent:>7}{base:>6}   {'OOM' if 'out of memory' in str(e).lower() else str(e)[:40]}", flush=True)
            torch.cuda.empty_cache()
