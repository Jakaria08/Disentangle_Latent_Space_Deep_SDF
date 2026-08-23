#!/usr/bin/env python3
"""Is the CALSNIC AE undertrained? 173 meshes at bs=8 is only 22 steps/epoch, so 250 epochs
is ~5.5k gradient steps -- about a tenth of what the ADNI runs used. Sweep epoch count."""
import sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from search_calsnic import Data, make_model, train

dev = torch.device("cuda", 1); torch.cuda.set_device(dev)
data = Data(dev)
PCA = {25: 4.703, 128: 3.953}
print(f"{'latent':>7}{'epochs':>8}{'steps':>8}{'val_mm':>9}{'vs_PCA':>8}{'min':>7}", flush=True)
for latent in (25, 128):
    for ep in (250, 1000, 2500):
        cfg = dict(n_levels=5, base_channels=32, seq_length=13, dilation=1, dropout=0.1,
                   adaptive_levels=1, dynamic_seq_frac=0.5)
        torch.manual_seed(1); torch.cuda.empty_cache()
        m, _ = make_model(cfg, "spiral", latent, dev)
        t0 = time.time()
        best, _, _ = train(m, data, epochs=ep, batch_size=8, lr=3e-4, lr_decay=0.99,
                           decay_step=10, weight_decay=1e-5, noise_std=0.04,
                           scheduler_type="cosine", patience=0, min_epochs=0,
                           eval_every=50, seed=1)
        print(f"{latent:>7}{ep:>8}{ep*22:>8}{best:9.3f}{best/PCA[latent]:8.3f}"
              f"{(time.time()-t0)/60:7.1f}", flush=True)
        del m; torch.cuda.empty_cache()
