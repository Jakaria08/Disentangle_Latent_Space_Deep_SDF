#!/usr/bin/env python3
"""v4 search: PCA + neural-residual autoencoder at a matched latent budget.

One study per (residual architecture, latent budget K). k -- how much of the budget goes to
the frozen PCA basis -- is searched over {0, .25K, .5K, .75K, .9K, K}, so a single study spans
the ladder from a pure autoencoder (k=0) to pure PCA (k=K).

Stage 1 searches, stage 2 re-runs the winning config over several seeds (the reportable
number), stage 3 asserts the k=K control reproduces the published PCA figures.
"""

from __future__ import annotations

import argparse, datetime, json, sys, time, traceback
from pathlib import Path

import numpy as np
import optuna
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spiral_common as sc
import train_eval as te
from network import build_model, resolve_conv_types
from residual_model import MLPAE, ResidualAE, residual_stats

MAX_ADAPTIVE_NODES = 400
MAX_DYNAMIC_SEQ = 320
DS_CHOICES = ["4,4,4,4", "4,4,4", "4,4,2", "2,4,4"]
K_FRACTIONS = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]

EXPERIMENTS_V4 = {
    "spiralnet_z128": ("spiral", 128), "spiralnet_z256": ("spiral", 256),
    "adaptive_z128": ("adaptive", 128), "adaptive_z256": ("adaptive", 256),
    "mlp_z128": ("mlp", 128), "mlp_z256": ("mlp", 256),
}
PCA_TEST_REF = {128: 0.034380, 256: 0.010787}
PCA_VAL_REF = {128: 0.033668, 256: 0.010338}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exp", required=True, choices=sorted(EXPERIMENTS_V4))
    p.add_argument("--tag", default="v4")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n-trials", type=int, default=40)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--seed-repeats", type=int, default=5)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--trial-time-budget", type=float, default=1200.0)
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


_SPIRAL_CACHE, _RESSTAT_CACHE = {}, {}


def load_raw(device):
    """Raw mm tensors -- the PCA branch operates in the same space as the baseline table."""
    rows = sc.read_manifest()
    out = {s: torch.from_numpy(sc.load_split_vertices(s, rows=rows)).to(device) for s in sc.SPLITS}
    return rows, out, sc.load_faces(rows=rows)


def load_pca(device):
    model_dir = sc.OUTPUT_ROOT / "pca" / "model"
    mean = torch.from_numpy(np.load(model_dir / "mean.npy")).float().to(device)
    comps = torch.from_numpy(np.load(model_dir / "components_256.npy")).float().to(device)
    return mean, comps


def cached_res_stats(x_train, pca_mean, comps_k, key):
    if key not in _RESSTAT_CACHE:
        _RESSTAT_CACHE[key] = residual_stats(x_train, pca_mean, comps_k)
    return _RESSTAT_CACHE[key]


def cached_spirals(transform, ds_key, seq, dil, dyn, device):
    key = (ds_key, int(seq), int(dil), tuple(int(d) for d in dyn), str(device))
    if key not in _SPIRAL_CACHE:
        _SPIRAL_CACHE[key] = sc.build_spiral_stack(transform, seq, dil, dyn, device)
    return _SPIRAL_CACHE[key]


def level_sizes(t): return [int(v.shape[0]) for v in t["vertices"][:-1]]


def plan_conv_types(n_levels, conv_type, adaptive_levels, sizes):
    types = resolve_conv_types(n_levels, conv_type, adaptive_levels)
    return ["spiral" if (t == "adaptive" and sizes[i] > MAX_ADAPTIVE_NODES) else t
            for i, t in enumerate(types)]


def plan_dyn(conv_types, sizes, frac):
    return [1 if t != "adaptive" else int(max(8, min(MAX_DYNAMIC_SEQ, round(frac * n), n)))
            for t, n in zip(conv_types, sizes)]


def build_residual_ae(cfg, arch, latent_total, k, x_train, pca_mean, pca_comps, device, n_vertices):
    """Assemble PCA branch + residual network for one configuration."""
    comps_k = None if k == 0 else pca_comps[:k].contiguous()
    res_mean, res_std = cached_res_stats(x_train, pca_mean, comps_k, k)
    residual_latent = latent_total - k

    if residual_latent <= 0:                       # k == K -> the model is PCA-K exactly
        net = None
    elif arch == "mlp":
        net = MLPAE(n_vertices, residual_latent, hidden=cfg["hidden_dim"],
                    n_hidden=cfg["n_hidden_layers"], dropout=cfg["dropout"])
    else:
        transform = sc.get_transform(cfg["ds_factors"])
        sizes = level_sizes(transform)
        conv_types = plan_conv_types(len(cfg["ds_factors"]), arch, cfg.get("adaptive_levels", 0), sizes)
        dyn = plan_dyn(conv_types, sizes, cfg.get("dynamic_seq_frac", 0.0))
        spirals, dynamic, down, up = cached_spirals(
            transform, sc.ds_tag(cfg["ds_factors"]), cfg["seq_length"], cfg["dilation"], dyn, device)
        base = cfg["base_channels"]
        n_lv = len(cfg["ds_factors"])
        out_channels = [base] * (n_lv - 1) + [2 * base]
        net = build_model(transform=transform, spiral_indices=spirals, dynamic_spiral_indices=dynamic,
                          down_transform=down, up_transform=up, out_channels=out_channels,
                          latent_channels=residual_latent, conv_type=arch,
                          adaptive_levels=cfg.get("adaptive_levels", 0), conv_types=conv_types,
                          dropout=cfg["dropout"], linear_skip=False)
        cfg["_conv_types"], cfg["_dyn"] = conv_types, dyn
        cfg["_out_channels"], cfg["_pre_latent"] = out_channels, int(transform["vertices"][-1].shape[0]) * out_channels[-1]

    if net is not None:
        net = net.to(device)
    return ResidualAE(pca_mean, comps_k, net, res_mean, res_std, latent_total).to(device)


# ------------------------------------------------------------------------------------------
# training
# ------------------------------------------------------------------------------------------
@torch.no_grad()
def reconstruct(model, x, batch=64):
    model.eval()
    return torch.cat([model(x[i:i + batch]) for i in range(0, len(x), batch)])


def eval_metrics(model, x, faces=None):
    return sc.reconstruction_metrics(reconstruct(model, x), x, faces=faces)


def val_rmse(model, x_val):
    return float(sc.vertex_rmse_mm(reconstruct(model, x_val), x_val).mean())


def train_residual(model, x_train, x_val, *, epochs, batch_size, lr, lr_decay, decay_step,
                   weight_decay, noise_std, scheduler_type, patience, eval_every, seed,
                   time_budget_s=None, log=None, report_fn=None, should_prune=None):
    """Trains only the residual network; the PCA branch is frozen.

    Loss is L1 on the normalised residual, which keeps gradients well scaled -- residuals are
    ~30x smaller than the shapes. Model selection is on full-reconstruction val RMSE in mm.
    """
    if model.residual_net is None:                       # k == K, nothing to train
        return val_rmse(model, x_val), None, {"history": [], "best_epoch": 0,
                                              "epochs_completed": 0, "stopped_on_time_budget": False,
                                              "stopped_on_patience": False}

    net = model.residual_net
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, int(epochs)))
             if scheduler_type == "cosine"
             else torch.optim.lr_scheduler.StepLR(opt, max(1, int(decay_step)), gamma=lr_decay))

    with torch.no_grad():                                # residual target is fixed (PCA frozen)
        r_train = model.residual_target(x_train)

    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    n = len(r_train)
    best, best_state, best_epoch, stale = float("inf"), None, -1, 0
    hist, on_budget, on_patience = [], False, False
    start = time.time()

    for epoch in range(1, int(epochs) + 1):
        net.train()
        perm = torch.randperm(n, generator=gen).to(r_train.device)
        tot = cnt = 0
        for i in range(0, n, batch_size):
            b = r_train[perm[i:i + batch_size]]
            inp = b if noise_std <= 0 else b + noise_std * torch.randn_like(b)
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.l1_loss(net(inp), b)
            loss.backward(); opt.step()
            tot += float(loss.detach()); cnt += 1
        sched.step()

        if epoch % eval_every == 0 or epoch == int(epochs):
            cur = val_rmse(model, x_val)
            if cur < best - 1e-9:
                best, best_epoch, stale = cur, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            else:
                stale += 1
            hist.append({"epoch": epoch, "train_l1": tot / max(1, cnt), "val_rmse_mm": cur,
                         "best_val_rmse_mm": best, "elapsed_s": time.time() - start})
            if log: log(f"    epoch {epoch:4d}/{int(epochs)}  train_l1={tot/max(1,cnt):.5f}  "
                        f"val_rmse_mm={cur:.6f}  best={best:.6f}  ({time.time()-start:.0f}s)")
            if report_fn: report_fn(cur, epoch)
            if should_prune and should_prune():
                if log: log(f"    pruned at epoch {epoch}")
                break
            if patience and stale >= int(patience):
                on_patience = True
                if log: log(f"    early stop at epoch {epoch} (best={best:.6f} @ {best_epoch})")
                break
            if time_budget_s is not None and (time.time() - start) > time_budget_s:
                on_budget = True
                if log: log(f"    stopped at epoch {epoch}/{int(epochs)} on time budget "
                            f"(best={best:.6f})")
                break

    return best, best_state, {"history": hist, "best_epoch": best_epoch,
                              "epochs_completed": hist[-1]["epoch"] if hist else 0,
                              "stopped_on_time_budget": on_budget,
                              "stopped_on_patience": on_patience}


def sample_config(trial, arch, K, smoke):
    frac = trial.suggest_categorical("k_fraction", K_FRACTIONS)
    cfg = {
        "k_fraction": frac,
        "k": int(round(frac * K)),
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "lr_decay": trial.suggest_float("lr_decay", 0.90, 0.999),
        "decay_step": trial.suggest_int("decay_step", 1, 20),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3, step=0.05),
        "noise_std": trial.suggest_float("noise_std", 0.0, 0.08, step=0.02),
        "scheduler": trial.suggest_categorical("scheduler", ["step", "cosine"]),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32, 64]),
        "epochs": 4 if smoke else trial.suggest_int("epochs", 200, 600, step=100),
    }
    if arch == "mlp":
        cfg["hidden_dim"] = trial.suggest_categorical("hidden_dim", [512, 1024, 2048, 4096])
        cfg["n_hidden_layers"] = trial.suggest_int("n_hidden_layers", 1, 3)
    else:
        cfg["ds_factors"] = [int(v) for v in trial.suggest_categorical("ds_factors", DS_CHOICES).split(",")]
        cfg["base_channels"] = trial.suggest_int("base_channels", 32, 128, step=16)
        cfg["seq_length"] = trial.suggest_int("seq_length", 9, 27, step=2)
        cfg["dilation"] = trial.suggest_int("dilation", 1, 2)
        if arch == "adaptive":
            cfg["adaptive_levels"] = trial.suggest_int("adaptive_levels", 1, 2)
            cfg["dynamic_seq_frac"] = trial.suggest_float("dynamic_seq_frac", 0.25, 1.0)
    return cfg


# ------------------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------------------
def main():
    args = parse_args()
    arch, K = EXPERIMENTS_V4[args.exp]
    key = f"{args.exp}_{args.tag}"
    root = sc.OUTPUT_ROOT
    dirs = {n: sc.makedirs(sc.require_bulk_path(root / n / key, n)) for n in
            ("studies", "logs", "best", "latents")}
    trial_dir = sc.makedirs(dirs["studies"] / "trials")

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
        log(f"PCA-{K} reference  val={PCA_VAL_REF[K]:.6f}  test={PCA_TEST_REF[K]:.6f}")
        log(f"start={datetime.datetime.now().isoformat(timespec='seconds')}")
        log("=" * 88)

        torch.manual_seed(args.seed); np.random.seed(args.seed)
        rows, X, faces = load_raw(device)
        pca_mean, pca_comps = load_pca(device)
        n_vertices = X["train"].shape[1]
        log(f"data: train={len(X['train'])} val={len(X['val'])} test={len(X['test'])} "
            f"verts={n_vertices} | pca basis {tuple(pca_comps.shape)}")

        trial_rows: list[dict] = []

        def objective(trial):
            cfg = sample_config(trial, arch, K, args.smoke)
            k = cfg["k"]
            model = build_residual_ae(cfg, arch, K, k, X["train"], pca_mean, pca_comps,
                                      device, n_vertices)
            n_params = model.num_parameters()
            trial.set_user_attr("k", k)
            trial.set_user_attr("residual_latent", K - k)
            trial.set_user_attr("n_params_trainable", n_params)
            if "_pre_latent" in cfg: trial.set_user_attr("pre_latent_dim", cfg["_pre_latent"])
            if "_conv_types" in cfg: trial.set_user_attr("conv_types", cfg["_conv_types"])

            log(f"  trial {trial.number}: k={k} (res_latent={K-k}) " +
                (f"hidden={cfg.get('hidden_dim')}x{cfg.get('n_hidden_layers')} " if arch == "mlp"
                 else f"ds={cfg.get('ds_factors')} base={cfg.get('base_channels')} "
                      f"seq={cfg.get('seq_length')} ") +
                f"bs={cfg['batch_size']} ep={cfg['epochs']} lr={cfg['lr']:.2e} "
                f"drop={cfg['dropout']} noise={cfg['noise_std']} params={n_params/1e6:.2f}M")

            torch.cuda.reset_peak_memory_stats(device)
            t0 = time.time()
            try:
                best, state, info = train_residual(
                    model, X["train"], X["val"], epochs=cfg["epochs"], batch_size=cfg["batch_size"],
                    lr=cfg["lr"], lr_decay=cfg["lr_decay"], decay_step=cfg["decay_step"],
                    weight_decay=cfg["weight_decay"], noise_std=cfg["noise_std"],
                    scheduler_type=cfg["scheduler"], patience=0 if args.smoke else args.patience,
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

            dur, peak = time.time() - t0, torch.cuda.max_memory_allocated(device) / 1e9
            ck = trial_dir / f"trial_{trial.number:04d}_val{best:.6f}.pt"
            torch.save({"trial": trial.number, "experiment": args.exp, "tag": args.tag,
                        "arch": arch, "latent_total": K, "k": k, "residual_latent": K - k,
                        "config": {kk: vv for kk, vv in cfg.items() if not kk.startswith("_")},
                        "conv_types": cfg.get("_conv_types"), "dynamic_seq_lengths": cfg.get("_dyn"),
                        "out_channels": cfg.get("_out_channels"),
                        "adaptive_reset_parameters_fixed": True,
                        "n_params_trainable": n_params, "best_val_rmse_mm": float(best),
                        "best_epoch": info["best_epoch"], "residual_state_dict": state}, ck)

            trial_rows.append({
                "trial": trial.number, "experiment": args.exp, "arch": arch, "K": K, "k": k,
                "residual_latent": K - k, "val_rmse_mm": float(best),
                "pca_only_reference": PCA_VAL_REF[K], "best_epoch": info["best_epoch"],
                "epochs_completed": info["epochs_completed"],
                "stopped_on_time_budget": info["stopped_on_time_budget"],
                "stopped_on_patience": info["stopped_on_patience"],
                "n_params_trainable": n_params, "duration_s": round(dur, 1),
                "peak_gb": round(peak, 2), "checkpoint": str(ck),
                **{f"p_{a}": b for a, b in trial.params.items()}})
            sc.atomic_write_csv(dirs["studies"] / "trial_metrics.csv", trial_rows,
                                fieldnames=sorted({a for r in trial_rows for a in r}))
            log(f"  trial {trial.number}: val_rmse_mm={best:.6f} (PCA-{K}={PCA_VAL_REF[K]:.6f}) "
                f"k={k} params={n_params/1e6:.2f}M peak={peak:.2f}GB {dur:.0f}s")
            del model; torch.cuda.empty_cache()
            return float(best)

        study = optuna.create_study(
            direction="minimize", study_name=key,
            storage=f"sqlite:///{dirs['studies'] / 'study.db'}", load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=args.seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=10))
        study.optimize(objective, n_trials=2 if args.smoke else args.n_trials, gc_after_trial=True)

        done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not done:
            raise SystemExit("no completed trials")
        best_trial = study.best_trial
        log(f"\nBEST trial {best_trial.number}: val={best_trial.value:.6f} "
            f"k={best_trial.user_attrs.get('k')}")

        # ---- k-sweep: best val at each k the search visited (the main figure) ----
        sweep: dict[int, float] = {}
        for r in trial_rows:
            sweep[r["k"]] = min(sweep.get(r["k"], 1e9), r["val_rmse_mm"])
        sc.atomic_write_csv(dirs["studies"] / "k_sweep.csv",
                            [{"experiment": args.exp, "arch": arch, "K": K, "k": kk,
                              "residual_latent": K - kk, "best_val_rmse_mm": vv,
                              "pca_K_val": PCA_VAL_REF[K]} for kk, vv in sorted(sweep.items())])
        log("  k-sweep (best val per k): " +
            "  ".join(f"k={kk}:{vv:.6f}" for kk, vv in sorted(sweep.items())))

        # ---- stage 2: seed repeats of the winning config ----
        bck = torch.load([r for r in trial_rows if r["trial"] == best_trial.number][0]["checkpoint"],
                         map_location="cpu")
        cfg, k = bck["config"], bck["k"]
        n_seeds = 2 if args.smoke else args.seed_repeats
        log(f"\nstage 2: {n_seeds} seed repeats of trial {best_trial.number} (k={k})")
        seed_rows = []
        for s in range(n_seeds):
            torch.manual_seed(1000 + s)
            m = build_residual_ae(dict(cfg), arch, K, k, X["train"], pca_mean, pca_comps,
                                  device, n_vertices)
            bv, st, _ = train_residual(
                m, X["train"], X["val"], epochs=cfg["epochs"], batch_size=cfg["batch_size"],
                lr=cfg["lr"], lr_decay=cfg["lr_decay"], decay_step=cfg["decay_step"],
                weight_decay=cfg["weight_decay"], noise_std=cfg["noise_std"],
                scheduler_type=cfg["scheduler"], patience=args.patience,
                eval_every=args.eval_every, seed=1000 + s,
                time_budget_s=args.trial_time_budget, log=None)
            if st is not None: m.residual_net.load_state_dict(st)
            row = {"seed": 1000 + s, "val_rmse_mm": bv}
            for sp in sc.SPLITS:
                mt = eval_metrics(m, X[sp], faces=faces)
                row[f"{sp}_rmse_mm"] = mt["vertex_rmse_mm_mean"]
                row[f"{sp}_vol_err_pct"] = mt.get("volume_abs_relative_error_pct_mean")
            seed_rows.append(row)
            log(f"  seed {1000+s}: val={row['val_rmse_mm']:.6f} test={row['test_rmse_mm']:.6f}")
            del m; torch.cuda.empty_cache()
        sc.atomic_write_csv(dirs["best"] / "seed_repeats.csv", seed_rows)

        vals = np.array([r["val_rmse_mm"] for r in seed_rows])
        tests = np.array([r["test_rmse_mm"] for r in seed_rows])
        verdict = ("PCA-only (k=K); residual branch earned nothing" if k == K
                   else "beats PCA" if tests.mean() < PCA_TEST_REF[K] else "does not beat PCA")
        log(f"\n  seeds: val {vals.mean():.6f}+-{vals.std():.6f}  "
            f"test {tests.mean():.6f}+-{tests.std():.6f}  | PCA-{K} test {PCA_TEST_REF[K]:.6f}"
            f"  => {verdict}")

        sc.atomic_write_json(dirs["best"] / "best_summary.json", {
            "experiment": args.exp, "tag": args.tag, "arch": arch, "latent_total": K,
            "k": k, "residual_latent": K - k, "best_trial": int(best_trial.number),
            "best_params": best_trial.params, "config": cfg,
            "n_params_trainable": bck["n_params_trainable"],
            "pca_basis_numbers": int(k * n_vertices * 3),
            "search_val_rmse_mm": float(best_trial.value),
            "seed_repeats": {"n": len(seed_rows),
                             "val_mean": float(vals.mean()), "val_sd": float(vals.std()),
                             "test_mean": float(tests.mean()), "test_sd": float(tests.std())},
            "pca_reference": {"val": PCA_VAL_REF[K], "test": PCA_TEST_REF[K]},
            "verdict": verdict,
            "beats_pca_beyond_seed_noise": bool(
                tests.mean() + tests.std() < PCA_TEST_REF[K]) if k != K else False,
            "k_sweep": {str(a): b for a, b in sorted(sweep.items())},
            "n_trials_complete": len(done), "log_file": str(log_fp)})
        log(f"wrote {dirs['best']}")
        log(f"end={datetime.datetime.now().isoformat(timespec='seconds')}")
    except Exception:
        traceback.print_exc(); raise
    finally:
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
        handle.close()


if __name__ == "__main__":
    main()
