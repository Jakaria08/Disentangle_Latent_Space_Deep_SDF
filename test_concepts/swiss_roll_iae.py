#!/usr/bin/env python
"""
swiss_roll_iae.py — Isometric Auto-Encoder (I-AE) faithful PyTorch replica

References:
 • Gropp, Atzmon & Lipman, “Isometric Autoencoders,” NeurIPS 2020 :contentReference[oaicite:5]{index=5}
 • El Be Ji, “Isometric Embedding of Manifolds…,” TUM MSc Thesis 2022 :contentReference[oaicite:6]{index=6}
"""

import argparse
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt

# reproducibility
torch.manual_seed(42)

# 1. Dataset helper
def make_dataset(n_points=5000):
    X, t = make_swiss_roll(n_points, noise=0.0)  # zero noise :contentReference[oaicite:7]{index=7}
    X -= X.mean(axis=0, keepdims=True)          # center only :contentReference[oaicite:8]{index=8}
    return torch.tensor(X, dtype=torch.float32), torch.tensor(t, dtype=torch.float32)

# 2. MLP backbone
class MLP(nn.Module):
    def __init__(self, in_d, out_d, hidden=(256,)*3):
        super().__init__()
        layers = []
        dims = (in_d, *hidden, out_d)
        for a,b in zip(dims, dims[1:]):
            layers += [nn.Linear(a,b), nn.ReLU()]
        layers.pop()  # drop last ReLU
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

# 3. Isometric Auto-Encoder
class IAE(nn.Module):
    def __init__(self, x_dim=3, z_dim=2, dirs=8):
        super().__init__()
        self.enc, self.dec = MLP(x_dim,z_dim), MLP(z_dim,x_dim)
        self.dirs, self.step = dirs, 0

    @staticmethod
    def jvp(func, x, v):
        # forward-mode JVP :contentReference[oaicite:9]{index=9}
        import torch.func as F
        try:
            return F.jvp(func, (x,), (v,))[1]  # no create_graph
        except Exception:
            # fallback for higher-order grads :contentReference[oaicite:10]{index=10}
            _, jv = torch.autograd.functional.jvp(func, (x,), (v,), create_graph=True)
            return jv

    @staticmethod
    def unit_vectors(shape, device):
        v = torch.randn(shape, device=device)
        return v / (v.norm(dim=-1, keepdim=True) + 1e-9)

    def forward(self, x, λ_iso=300., λ_piso=300.):
        z    = self.enc(x)
        x̂    = self.dec(z)
        L_rec = (x̂ - x).pow(2).mean()

        # warm-up first 10% of steps (~120 steps) :contentReference[oaicite:11]{index=11}
        ramp = min(1.0, self.step / 120)
        self.step += 1

        B, d = z.shape
        v = self.unit_vectors((self.dirs, B, d), z.device)
        # vectorized JVP over 8 unit directions :contentReference[oaicite:12]{index=12}
        try:
            jvs = torch.func.vmap(self.jvp, in_dims=(None, None, 0))(self.dec, z, v)
        except Exception:
            jvs = torch.stack([self.jvp(self.dec, z, v[k]) for k in range(self.dirs)])
        L_iso = ((jvs.norm(dim=-1) - 1.0).pow(2)).mean()  # target = 1.0 :contentReference[oaicite:13]{index=13}

        # pseudo-inverse consistency (no detach) :contentReference[oaicite:14]{index=14}
        jv0   = jvs[0]
        JeJv0 = self.jvp(self.enc, x̂, jv0)
        L_piso = (JeJv0 - v[0]).pow(2).mean()

        return L_rec + ramp*(λ_iso*L_iso + λ_piso*L_piso), \
               {'rec':L_rec.detach(), 'iso':L_iso.detach(), 'piso':L_piso.detach()}

# 4. Training helper
def train(model, X, epochs=1200, batch=256, lr=1e-3, amp=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    ds = TensorDataset(X)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, pin_memory=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)  # lr from paper :contentReference[oaicite:15]{index=15}

    scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.type=='cuda'))
    for ep in range(1, epochs+1):
        sums = {'rec':0., 'iso':0., 'piso':0.}
        for (xb,) in dl:
            xb = xb.to(device)
            with torch.cuda.amp.autocast(enabled=(amp and device.type=='cuda')):
                loss, logs = model(xb)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            for k in sums: sums[k] += logs[k].item()*xb.size(0)
        if ep==1 or ep%100==0:
            N = len(ds)
            print(f"Epoch {ep:4d} | " +
                  " | ".join(f"{k}:{sums[k]/N:.2e}" for k in sums))
    return model

# 5. Demo & plotting
def run_demo(save_plot=True):
    X, t = make_dataset()
    model = IAE()
    train(model, X)
    device = next(model.parameters()).device
    Z = model.enc(X.to(device)).cpu().detach().numpy()
    if save_plot:
        plt.figure(figsize=(5,4))
        plt.scatter(Z[:,0], Z[:,1], c=t.numpy(), s=4, cmap='viridis')
        plt.axis('equal'); plt.tight_layout()
        plt.savefig('iae_swiss_roll_flattened.png', dpi=300, bbox_inches='tight')
        print("Saved → iae_swiss_roll_flattened.png")
    return model

if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cpu', action='store_true', help='force CPU')
    p.add_argument('--noplot', action='store_true', help='no plot')
    args = p.parse_args()
    if args.cpu: torch.cuda.is_available=lambda: False
    run_demo(save_plot=not args.noplot)
