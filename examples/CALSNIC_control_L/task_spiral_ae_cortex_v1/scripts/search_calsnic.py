#!/usr/bin/env python3
"""SpiralNet++ / Adaptive Spiral autoencoder on the CALSNIC left pial surface, vs PCA.

Same architectures and protocol as the ADNI hippocampus pure runs; the operators, AE and
decimation code are imported unchanged from that task. What differs is the data regime:

  173 training meshes, 40,962 vertices  =>  PCA is rank-capped at 172 components.
  PCA-256 does not exist, so the K=256 model's honest comparator is the PCA-172 ceiling.

Metric is CALSNIC's `corresponded_vertex_rmse_mm` (per-vertex Euclidean, unscaled to mm) --
NOT the hippocampus per-coordinate convention, which is sqrt(3) smaller.
"""
from __future__ import annotations

import argparse, copy, datetime, json, sys, time, traceback
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calsnic_data as cd
from network import build_model, resolve_conv_types   # from the hippocampus task

MAX_ADAPTIVE_NODES = 400
MAX_DYNAMIC_SEQ = 320
# The fs6 mesh is an icosphere subdivision (level 6) with nested edge-midpoint ordering, so
# the hierarchy is exact and free (2.4 s) instead of ~4-5 h of psbody quadric decimation.
# Levels: 40962 -> 10242 -> 2562 -> 642 -> 162 -> 42.
N_LEVEL_CHOICES = [4, 5]

EXPERIMENTS = {
    "calsnic_spiralnet_z128": ("spiral", 128), "calsnic_adaptive_z128": ("adaptive", 128),
    "calsnic_spiralnet_z256": ("spiral", 256), "calsnic_adaptive_z256": ("adaptive", 256),
}
# PCA comparator per latent budget. At K=256 PCA cannot exceed 172 components.
PCA_COMPARATOR = {128: (128, 3.953402, 4.017216), 256: (172, 3.863790, 3.843948)}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exp", required=True, choices=sorted(EXPERIMENTS))
    p.add_argument("--tag", default="v1")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n-trials", type=int, default=40)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--seed-repeats", type=int, default=5)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--min-epochs", type=int, default=150)
    p.add_argument("--trial-time-budget", type=float, default=2400.0)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


class Tee:
    def __init__(self, *s): self.streams = s
    def write(self, d):
        for x in self.streams: x.write(d)
        return len(d)
    def flush(self):
        for x in self.streams: x.flush()
    def isatty(self): return False


_SPIRALS = {}


def cached_spirals(transform, ds_key, seq, dil, dyn, device):
    import spiral_common as sc
    key = (ds_key, int(seq), int(dil), tuple(int(d) for d in dyn), str(device))
    if key not in _SPIRALS:
        _SPIRALS[key] = sc.build_spiral_stack(transform, seq, dil, dyn, device)
    return _SPIRALS[key]


def level_sizes(t): return [int(v.shape[0]) for v in t["vertices"][:-1]]


def plan_conv_types(n, conv_type, adaptive_levels, sizes):
    types = resolve_conv_types(n, conv_type, adaptive_levels)
    return ["spiral" if (t == "adaptive" and sizes[i] > MAX_ADAPTIVE_NODES) else t
            for i, t in enumerate(types)]


def plan_dyn(types, sizes, frac):
    return [1 if t != "adaptive" else int(max(8, min(MAX_DYNAMIC_SEQ, round(frac * n), n)))
            for t, n in zip(types, sizes)]


class Data:
    """Normalised splits on GPU plus what is needed to score in CALSNIC's mm convention."""
    def __init__(self, device):
        rows = cd.read_manifest()
        self.rows = rows
        self.faces = cd.load_faces(rows=rows)
        raw = {s: cd.load_split_vertices(s, rows=rows) for s in cd.SPLITS}
        self.scales = {s: cd.scale_factors(rows, s) for s in cd.SPLITS}
        mean = raw["train"].mean(axis=0)
        std = np.maximum(raw["train"].std(axis=0), 1e-8)
        self.mean = torch.from_numpy(mean).to(device)
        self.std = torch.from_numpy(std).to(device)
        self.x = {s: ((torch.from_numpy(v).to(device) - self.mean) / self.std).contiguous()
                  for s, v in raw.items()}
        self.raw = {s: torch.from_numpy(v).to(device) for s, v in raw.items()}
        self.device = device

    def denorm(self, t): return t * self.std + self.mean


@torch.no_grad()
def reconstruct(model, data, split, batch=8):
    model.eval()
    x = data.x[split]
    return data.denorm(torch.cat([model(x[i:i + batch]) for i in range(0, len(x), batch)]))


def rmse_mm(model, data, split):
    pred = reconstruct(model, data, split)
    return cd.corresponded_vertex_rmse_mm(pred, data.raw[split], data.scales[split])


def train(model, data, *, epochs, batch_size, lr, lr_decay, decay_step, weight_decay,
          noise_std, scheduler_type, patience, min_epochs, eval_every, seed,
          time_budget_s=None, log=None, report_fn=None, should_prune=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, int(epochs)))
             if scheduler_type == "cosine"
             else torch.optim.lr_scheduler.StepLR(opt, max(1, int(decay_step)), gamma=lr_decay))
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    xt = data.x["train"]; n = len(xt)
    best, state, best_ep, stale = float("inf"), None, -1, 0
    hist, on_budget, on_pat, on_prune = [], False, False, False
    t0 = time.time()
    for ep in range(1, int(epochs) + 1):
        model.train()
        perm = torch.randperm(n, generator=gen).to(data.device)
        tot = cnt = 0
        for i in range(0, n, batch_size):
            b = xt[perm[i:i + batch_size]]
            inp = b if noise_std <= 0 else b + noise_std * torch.randn_like(b)
            opt.zero_grad(set_to_none=True)
            loss = F.l1_loss(model(inp), b)
            loss.backward(); opt.step()
            tot += float(loss.detach()); cnt += 1
        sched.step()
        if ep % eval_every == 0 or ep == int(epochs):
            cur = float(rmse_mm(model, data, "val").mean())
            if cur < best - 1e-9:
                best, best_ep, stale = cur, ep, 0
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            hist.append({"epoch": ep, "train_l1": tot / max(1, cnt), "val_rmse_mm": cur})
            if log: log(f"    epoch {ep:4d}/{int(epochs)} train_l1={tot/max(1,cnt):.5f} "
                        f"val_rmse_mm={cur:.4f} best={best:.4f} ({time.time()-t0:.0f}s)")
            if report_fn: report_fn(cur, ep)
            if ep >= min_epochs and should_prune and should_prune():
                on_prune = True
                if log: log(f"    pruner signalled at epoch {ep}")
                break
            if ep >= min_epochs and patience and stale >= patience:
                on_pat = True
                if log: log(f"    early stop at epoch {ep} (best={best:.4f} @ {best_ep})")
                break
            if time_budget_s and (time.time() - t0) > time_budget_s:
                on_budget = True
                if log: log(f"    stopped at epoch {ep} on time budget (best={best:.4f})")
                break
    return best, state, {"history": hist, "best_epoch": best_ep,
                         "epochs_completed": hist[-1]["epoch"] if hist else 0,
                         "stopped_on_time_budget": on_budget, "stopped_on_patience": on_pat,
                         "stopped_on_prune": on_prune}


def sample_config(trial, arch, smoke):
    """Memory-aware ranges. A fine-level spiral conv materialises [B, 40962, seq, C]; at
    seq=27, C=128, B=32 that is ~18 GB, so batch/width/seq are capped well below the
    hippocampus settings. With 173 training meshes small batches are natural anyway."""
    cfg = {
        "n_levels": trial.suggest_categorical("n_levels", N_LEVEL_CHOICES),
        # Measured on GPU 1 at 40,962 vertices: base=160/bs=4 peaks at only 5.7 GB and
        # base=128/bs=8 at 8.8 GB, so the original 16-64 ceiling was far too conservative.
        # The ADNI runs showed the search saturating against a too-low base ceiling, and here
        # there is more to gain because PCA is rank-limited and weak.
        "base_channels": trial.suggest_int("base_channels", 32, 192, step=32),
        "seq_length": trial.suggest_int("seq_length", 9, 23, step=2),
        "dilation": trial.suggest_int("dilation", 1, 2),
        "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16]),
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "lr_decay": trial.suggest_float("lr_decay", 0.90, 0.999),
        "decay_step": trial.suggest_int("decay_step", 1, 20),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3, step=0.05),
        "noise_std": trial.suggest_float("noise_std", 0.0, 0.10, step=0.02),
        "scheduler": trial.suggest_categorical("scheduler", ["step", "cosine"]),
        "epochs": 4 if smoke else trial.suggest_int("epochs", 300, 900, step=100),
    }
    if arch == "adaptive":
        cfg["adaptive_levels"] = trial.suggest_int("adaptive_levels", 1, 2)
        cfg["dynamic_seq_frac"] = trial.suggest_float("dynamic_seq_frac", 0.25, 1.0)
    return cfg


def make_model(cfg, arch, latent, device):
    transform = cd.get_transform_ico(cfg["n_levels"])
    sizes = level_sizes(transform)
    n_lv = int(cfg["n_levels"])
    types = plan_conv_types(n_lv, arch, cfg.get("adaptive_levels", 0), sizes)
    dyn = plan_dyn(types, sizes, cfg.get("dynamic_seq_frac", 0.0))
    sp, dynidx, dn, up = cached_spirals(transform, f"ico{cfg['n_levels']}",
                                        cfg["seq_length"], cfg["dilation"], dyn, device)
    base = cfg["base_channels"]
    oc = [base] * (n_lv - 1) + [2 * base]
    model = build_model(transform=transform, spiral_indices=sp, dynamic_spiral_indices=dynidx,
                        down_transform=dn, up_transform=up, out_channels=oc,
                        latent_channels=latent, conv_type=arch,
                        adaptive_levels=cfg.get("adaptive_levels", 0), conv_types=types,
                        dropout=cfg["dropout"], linear_skip=False).to(device)
    meta = {"conv_types": types, "dyn": dyn, "out_channels": oc, "level_sizes": sizes,
            "pre_latent_dim": int(transform["vertices"][-1].shape[0]) * oc[-1]}
    return model, meta


def main():
    args = parse_args()
    arch, K = EXPERIMENTS[args.exp]
    pca_k, pca_val, pca_test = PCA_COMPARATOR[K]
    key = f"{args.exp}_{args.tag}"
    dirs = {n: cd.makedirs(cd.require_bulk_path(cd.OUTPUT_ROOT / n / key, n))
            for n in ("studies", "logs", "best")}
    trial_dir = cd.makedirs(dirs["studies"] / "trials")
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_fp = dirs["logs"] / f"{key}_{stamp}{'_smoke' if args.smoke else ''}.log"
    handle = open(log_fp, "a", buffering=1)
    sys.stdout, sys.stderr = Tee(sys.__stdout__, handle), Tee(sys.__stderr__, handle)
    def log(m): print(m, flush=True)

    try:
        device = torch.device("cuda", args.gpu)
        torch.cuda.set_device(device)
        log("=" * 88)
        log(f"{key}: arch={arch} K={K} gpu={args.gpu} trials={args.n_trials} smoke={args.smoke}")
        log(f"comparator PCA-{pca_k} (rank ceiling {cd.PCA_MAX_RANK}): "
            f"val {pca_val:.4f}  test {pca_test:.4f} mm")
        log("=" * 88)
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        data = Data(device)
        log(f"data: train={len(data.x['train'])} val={len(data.x['val'])} "
            f"test={len(data.x['test'])} verts={data.x['train'].shape[1]}")

        # Resume-safe -- see the note in the ADNI driver: an empty list truncates the CSV.
        trial_rows = []
        _existing = dirs["studies"] / "trial_metrics.csv"
        if _existing.exists():
            import csv as _csv
            with open(_existing, newline="") as _h:
                trial_rows = [r for r in _csv.DictReader(_h) if r.get("val_rmse_mm")]
            log(f"resumed trial_metrics.csv with {len(trial_rows)} prior rows")

        def objective(trial):
            cfg = sample_config(trial, arch, args.smoke)
            model, meta = make_model(cfg, arch, K, device)
            npar = model.num_parameters()
            for k, v in meta.items():
                trial.set_user_attr(k, v)
            trial.set_user_attr("n_params", npar)
            log(f"  trial {trial.number}: levels={cfg['n_levels']}{meta['level_sizes']} "
                f"oc={meta['out_channels']} "
                f"seq={cfg['seq_length']} bs={cfg['batch_size']} ep={cfg['epochs']} "
                f"lr={cfg['lr']:.2e} conv={meta['conv_types']} params={npar/1e6:.2f}M")
            torch.cuda.reset_peak_memory_stats(device)
            t0 = time.time()
            try:
                best, state, info = train(
                    model, data, epochs=cfg["epochs"], batch_size=cfg["batch_size"],
                    lr=cfg["lr"], lr_decay=cfg["lr_decay"], decay_step=cfg["decay_step"],
                    weight_decay=cfg["weight_decay"], noise_std=cfg["noise_std"],
                    scheduler_type=cfg["scheduler"],
                    patience=0 if args.smoke else args.patience,
                    min_epochs=0 if args.smoke else args.min_epochs,
                    eval_every=args.eval_every, seed=args.seed + trial.number,
                    time_budget_s=None if args.smoke else args.trial_time_budget,
                    log=log, report_fn=lambda v, e: trial.report(v, e),
                    should_prune=trial.should_prune)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    del model; torch.cuda.empty_cache()
                    log(f"  trial {trial.number}: CUDA OOM -> pruned")
                    raise optuna.TrialPruned("CUDA OOM") from exc
                raise
            if info["stopped_on_prune"]:
                del model; torch.cuda.empty_cache()
                raise optuna.TrialPruned()

            dur = time.time() - t0
            peak = torch.cuda.max_memory_allocated(device) / 1e9
            ck = trial_dir / f"trial_{trial.number:04d}_val{best:.4f}.pt"
            torch.save({"trial": trial.number, "experiment": args.exp, "arch": arch,
                        "latent_channels": K, "config": cfg, **meta,
                        "adaptive_reset_parameters_fixed": True,
                        "n_params": npar, "best_val_rmse_mm": float(best),
                        "model_state_dict": state}, ck)
            trial_rows.append({"trial": trial.number, "val_rmse_mm": float(best),
                               "pca_val_ref": pca_val, "ratio_vs_pca": float(best / pca_val),
                               "n_params": npar, "pre_latent_dim": meta["pre_latent_dim"],
                               "epochs_completed": info["epochs_completed"],
                               "stopped_on_time_budget": info["stopped_on_time_budget"],
                               "stopped_on_patience": info["stopped_on_patience"],
                               "duration_s": round(dur, 1), "peak_gb": round(peak, 2),
                               "checkpoint": str(ck),
                               **{f"p_{a}": b for a, b in trial.params.items()}})
            cd.atomic_write_csv(dirs["studies"] / "trial_metrics.csv", trial_rows)
            log(f"  trial {trial.number}: val_rmse_mm={best:.4f} "
                f"(PCA-{pca_k}={pca_val:.4f}, {best/pca_val:.3f}x) "
                f"params={npar/1e6:.2f}M peak={peak:.2f}GB {dur:.0f}s")
            del model; torch.cuda.empty_cache()
            return float(best)

        study = optuna.create_study(
            direction="minimize", study_name=key,
            storage=f"sqlite:///{dirs['studies'] / 'study.db'}", load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=args.seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=10))
        study.optimize(objective, n_trials=2 if args.smoke else args.n_trials,
                       gc_after_trial=True)

        done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not done:
            raise SystemExit("no completed trials")
        best_trial = study.best_trial
        log(f"\nBEST trial {best_trial.number}: val={best_trial.value:.4f} "
            f"(PCA-{pca_k} {pca_val:.4f})")

        # seed repeats + paired per-scan comparison against PCA on the same test scans
        bck = torch.load([r for r in trial_rows
                          if r["trial"] == best_trial.number][0]["checkpoint"],
                         map_location="cpu")
        cfg = bck["config"]
        n_seeds = 2 if args.smoke else args.seed_repeats
        seed_rows, test_per_scan = [], []
        log(f"\nseed repeats: {n_seeds}")
        for si in range(n_seeds):
            torch.manual_seed(3000 + si)
            m, _ = make_model(cfg, arch, K, device)
            bv, st, _ = train(m, data, epochs=cfg["epochs"], batch_size=cfg["batch_size"],
                              lr=cfg["lr"], lr_decay=cfg["lr_decay"],
                              decay_step=cfg["decay_step"], weight_decay=cfg["weight_decay"],
                              noise_std=cfg["noise_std"], scheduler_type=cfg["scheduler"],
                              patience=args.patience, min_epochs=args.min_epochs,
                              eval_every=args.eval_every, seed=3000 + si,
                              time_budget_s=args.trial_time_budget)
            if st is not None: m.load_state_dict(st)
            per_scan = rmse_mm(m, data, "test")
            test_per_scan.append(per_scan)
            seed_rows.append({"seed": 3000 + si, "val_rmse_mm": bv,
                              "test_rmse_mm": float(per_scan.mean())})
            log(f"  seed {3000+si}: val={bv:.4f} test={per_scan.mean():.4f}")
            del m; torch.cuda.empty_cache()
        cd.atomic_write_csv(dirs["best"] / "seed_repeats.csv", seed_rows)

        vals = np.array([r["val_rmse_mm"] for r in seed_rows])
        tests = np.array([r["test_rmse_mm"] for r in seed_rows])
        ae_per_scan = np.mean(np.stack(test_per_scan), axis=0)
        pca_per_scan = pca_test_per_scan(data, pca_k)
        paired = cd.paired_bootstrap(ae_per_scan, pca_per_scan)
        log(f"\n  seeds: val {vals.mean():.4f}+-{vals.std():.4f}  "
            f"test {tests.mean():.4f}+-{tests.std():.4f}   PCA-{pca_k} test {pca_test:.4f}")
        log(f"  paired vs PCA on the same 15 test scans: "
            f"mean diff {paired['mean_candidate_minus_reference']:+.4f} mm, "
            f"better on {paired['fraction_candidate_better']*100:.0f}% of scans, "
            f"95% CI {paired['bootstrap_mean_95ci']}, "
            f"{'SIGNIFICANT' if paired['significant'] else 'not significant'}")

        cd.atomic_write_json(dirs["best"] / "best_summary.json", {
            "experiment": args.exp, "arch": arch, "latent_channels": K,
            "best_trial": int(best_trial.number), "best_params": best_trial.params,
            "config": cfg, "n_params": bck["n_params"],
            "pca_comparator": {"components": pca_k, "val": pca_val, "test": pca_test,
                               "rank_ceiling": cd.PCA_MAX_RANK,
                               "note": "PCA cannot exceed 172 components on 173 training meshes"},
            "seed_repeats": {"n": len(seed_rows), "val_mean": float(vals.mean()),
                             "val_sd": float(vals.std()), "test_mean": float(tests.mean()),
                             "test_sd": float(tests.std())},
            "paired_vs_pca_test": paired,
            "beats_pca": bool(paired["mean_candidate_minus_reference"] < 0
                              and paired["significant"]),
            "n_trials_complete": len(done), "log_file": str(log_fp)})
        log(f"wrote {dirs['best']}")
    except Exception:
        traceback.print_exc(); raise
    finally:
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
        handle.close()


def pca_test_per_scan(data, k):
    """PCA-k reconstruction error per test scan, for the paired comparison."""
    model_dir = cd.OUTPUT_ROOT / "pca" / "model"
    mean = np.load(model_dir / "mean.npy").astype(np.float64)
    comp = np.load(model_dir / f"components_{k}.npy").astype(np.float64)
    gt = data.raw["test"].detach().cpu().numpy().reshape(len(data.raw["test"]), -1)
    recon = (gt - mean) @ comp.T @ comp + mean
    return cd.corresponded_vertex_rmse_mm(recon, gt, data.scales["test"])


if __name__ == "__main__":
    main()
