#!/usr/bin/env python3
"""Optuna search over the Compact MeshMAE hyperparameters.

Why this exists: SpiralNet++/Adaptive at z128 received 138 Optuna trials across 11 studies.
MeshMAE received 19 runs, ALL of them architecture ablations -- pool / latentset / flatten /
tokenizer / EMA / distance-bias / spiral-stem -- with `lr 3e-4, wd 0.05, dropout 0.1, cosine,
batch 16` hand-picked before the first run and never varied. Weight decay alone is 273x
SpiralNet++'s tuned optimum (0.05 vs 1.834e-4), carried over from transformer convention and
never validated on 2037 hippocampus meshes. The reported 4.4% gap therefore compares a model
tuned 138 times against one tuned zero times, and has to be closed before it means anything.

Two studies share this space and differ only in the latent head, so the flatten-vs-grouped
comparison is made AFTER both are tuned rather than between two untuned single runs.

Trials run as subprocesses: the trainer is already validated, each trial's artifacts land in
its own studies/ directory, and a crashed or OOM trial cannot poison the search process.

NO PRUNER, deliberately. Mid-run margins on this problem have twice proved non-predictive --
EMA read +15.1% at epoch 48 and +1.1% at convergence; the locality bias +4.5% then +0.14%.
A pruner ranking configs at epoch 50 would act on exactly the signal shown to be misleading.
Cost is bounded by --trial-time-budget plus the trainer's own patience instead.
"""
from __future__ import annotations

import argparse, csv, json, subprocess, sys, time
from pathlib import Path

import optuna

HERE = Path(__file__).resolve().parent
PY = "/home/jakaria/anaconda3/envs/pytorch_geo/bin/python"
OUT = Path("/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task_meshmae_ae_v1")
REF = {"pca128_val": 0.033668, "spiralnet128_val": 0.036784,
       "adaptive128_val": 0.037237, "s9_ema_val": 0.038463}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", required=True)
    p.add_argument("--head", choices=("flatten", "grouped"), default="flatten")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--epochs", type=int, default=220)
    p.add_argument("--min-epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--trial-time-budget", type=float, default=1100.0)
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def suggest(t, head):
    """Space centred so it SPANS both conventions: weight decay 1e-5..1e-1 covers
    SpiralNet++'s tuned 1.8e-4 and MeshMAE's inherited 0.05.

    The size knobs are bounded so that EVERY config converges inside --trial-time-budget.
    A first attempt used batch 8 / d_model 192 / depth 8; batch 8 doubles the steps per
    epoch and was truncated at epoch ~140 against the ~240 these runs need, so TPE would
    have penalised those configs for being cut short rather than for being worse -- a
    systematic bias, not noise. The slowest surviving config (d_model 128, depth 6,
    batch 16) is exactly S9's at 4.3 s/epoch, so 220 epochs fits the 1100 s budget with
    margin. A fair search over a slightly smaller space beats a biased search over a
    larger one, and capacity has been harmful at every point in this project anyway.
    """
    d_model = t.suggest_categorical("d_model", [64, 96, 128])
    cfg = {
        "lr": t.suggest_float("lr", 1e-4, 1e-3, log=True),
        "weight_decay": t.suggest_float("weight_decay", 1e-5, 1e-1, log=True),
        "dropout": t.suggest_float("dropout", 0.0, 0.3, step=0.05),
        "d_model": d_model,
        "depth": t.suggest_categorical("depth", [3, 4, 6]),
        "heads": t.suggest_categorical("heads", [4, 8]),      # divides every d_model above
        "batch_size": t.suggest_categorical("batch_size", [16, 32]),
        "ema_decay": t.suggest_categorical("ema_decay", [0.995, 0.999, 0.9995]),
    }
    if head == "grouped":
        cfg["head_rank"] = t.suggest_categorical("head_rank", [2, 4, 8, 16])
    return cfg


def main():
    a = parse_args()
    root = OUT / "studies" / a.study
    root.mkdir(parents=True, exist_ok=True)
    csv_fp = root / "trial_metrics.csv"
    log_fp = OUT / "logs" / f"{a.study}.log"
    log_fp.parent.mkdir(parents=True, exist_ok=True)

    def log(m):
        print(m, flush=True)
        with open(log_fp, "a") as h:
            h.write(m + "\n")

    def objective(trial):
        cfg = suggest(trial, a.head)
        name = f"{a.study}_t{trial.number:04d}"
        cmd = [PY, "-u", str(HERE / "train_meshmae.py"),
               "--run-name", name, "--gpu", str(a.gpu),
               # fixed: the architecture established by S2 (flatten/grouped head) and
               # S7 (raw tokenizer). Noise, distance-bias and spiral-stem are all excluded
               # -- measured at 0%, +0.14% and -0.08% respectively.
               "--patch-level", "3", "--tokenizer", "both", "--noise-std", "0.0",
               "--latent", "128", "--head", a.head,
               "--warm-start-decoder", "--no-freeze-decoder",
               "--epochs", str(a.epochs), "--eval-every", "1",
               "--min-epochs", str(a.min_epochs), "--patience", str(a.patience),
               "--time-budget", str(a.trial_time_budget)]
        for k, v in cfg.items():
            cmd += [f"--{k.replace('_', '-')}", str(v)]

        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        summary = OUT / "studies" / name / "summary.json"
        if proc.returncode != 0 or not summary.exists():
            log(f"  trial {trial.number} FAILED rc={proc.returncode}\n"
                f"    {proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else ''}")
            raise optuna.TrialPruned()

        d = json.loads(summary.read_text())
        val = float(d["best_val_rmse_mm"])
        row = {"trial": trial.number, "val_rmse_mm": val,
               "best_epoch": d.get("best_epoch"), "best_source": d.get("best_source"),
               "train_rmse_mm": d["metrics"]["train"]["vertex_rmse_mm_mean"],
               "test_rmse_mm": d["metrics"]["test"]["vertex_rmse_mm_mean"],
               "n_params": d.get("n_params"), "duration_s": round(time.time() - t0, 1),
               "head": a.head, **{f"p_{k}": v for k, v in cfg.items()}}
        row["gap"] = round(val / max(row["train_rmse_mm"], 1e-12), 4)
        write_header = not csv_fp.exists()
        with open(csv_fp, "a", newline="") as h:
            w = csv.DictWriter(h, fieldnames=list(row))
            if write_header:
                w.writeheader()
            w.writerow(row)
        log(f"  trial {trial.number:3d} val={val:.6f} train={row['train_rmse_mm']:.6f} "
            f"gap={row['gap']:.2f}x ep={row['best_epoch']} ({row['duration_s']:.0f}s) "
            f"lr={cfg['lr']:.2e} wd={cfg['weight_decay']:.2e} do={cfg['dropout']} "
            f"dm={cfg['d_model']} L={cfg['depth']} h={cfg['heads']} bs={cfg['batch_size']}"
            + (f" r={cfg['head_rank']}" if a.head == "grouped" else ""))
        return val

    study = optuna.create_study(
        study_name=a.study, direction="minimize",
        storage=f"sqlite:///{root/'study.db'}", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=a.seed, n_startup_trials=8))

    log("=" * 96)
    log(f"{a.study}: head={a.head} gpu={a.gpu} n_trials={a.n_trials} "
        f"epochs<={a.epochs} budget={a.trial_time_budget:.0f}s")
    log(f"targets  PCA-128 {REF['pca128_val']:.6f} | spiralnet {REF['spiralnet128_val']:.6f} "
        f"| S9_ema {REF['s9_ema_val']:.6f}")
    log("=" * 96)
    study.optimize(objective, n_trials=a.n_trials, catch=(Exception,))

    done = [t for t in study.trials if t.value is not None]
    if done:
        b = study.best_trial
        log(f"BEST trial {b.number}: val={b.value:.6f}  "
            f"({b.value/REF['spiralnet128_val']:.3f}x spiralnet, "
            f"{b.value/REF['pca128_val']:.3f}x PCA)")
        log(f"  params: {json.dumps(b.params)}")
    log(f"{len(done)}/{a.n_trials} trials completed; csv -> {csv_fp}")


if __name__ == "__main__":
    main()
