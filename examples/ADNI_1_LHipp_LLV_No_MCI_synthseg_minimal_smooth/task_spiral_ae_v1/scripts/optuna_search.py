#!/usr/bin/env python3
"""Optuna reconstruction search for one (conv_type, latent) experiment.

Latent size and conv type are FIXED per study -- they define the experiment -- so the search
is single-objective: minimise validation vertex_rmse_mm (mm).

The search deliberately includes hierarchy depth (`ds_factors`) and width (`base_channels`).
With the stock SpiralNet++ settings the encoder funnels everything through 11x64 = 704
features, which is fewer than the 8238 input dimensions and only 2.75x a 256-d latent; trials
whose pre-latent width is under 2x the latent are pruned rather than trained.

Everything (log, per-trial checkpoint, trial CSV, study db, summary) is written under bulk.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import optuna
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spiral_common as sc
import train_eval as te
from network import build_model, resolve_conv_types

# Adaptive-op guards: the ICCV op materialises [B, N, max_seq, C] tensors, so it is only
# applied on coarse levels (the paper's own CoMA config uses the two coarsest).
MAX_ADAPTIVE_NODES = 400
MAX_DYNAMIC_SEQ = 320

DS_CHOICES = ["4,4,4,4", "4,4,4", "4,4,2", "2,4,4"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exp", required=True, choices=sorted(sc.EXPERIMENTS))
    p.add_argument("--n-trials", type=int, default=80)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--tag", default="", help="suffix for study name + output dirs (e.g. v2)")
    p.add_argument("--linear-skip", action="store_true",
                   help="opt-in global linear branch; NOT part of the pure architecture "
                        "comparison -- use only for a separate augmented-variant study")
    # Search-range knobs. Defaults reproduce the v5 space exactly; v6 widens base_channels
    # because the v5 top-8 trials saturated at 112 of a 128 ceiling.
    p.add_argument("--base-min", type=int, default=32)
    p.add_argument("--base-max", type=int, default=128)
    p.add_argument("--base-step", type=int, default=16)
    p.add_argument("--epochs-min", type=int, default=300)
    p.add_argument("--epochs-max", type=int, default=900)
    p.add_argument("--dyn-frac-min", type=float, default=0.25)
    p.add_argument("--dyn-frac-max", type=float, default=1.0)
    p.add_argument("--min-epochs", type=int, default=150,
                   help="no pruning or patience before this epoch; protects slow starters")
    p.add_argument("--seed-repeats", type=int, default=5,
                   help="re-runs of the winning config, for the reportable mean+-sd")
    p.add_argument("--patience", type=int, default=8,
                   help="stop a trial after N evaluations with no new best (0 disables)")
    p.add_argument("--trial-time-budget", type=float, default=1200.0,
                   help="per-trial wall-clock cap in seconds; best-so-far is kept")
    p.add_argument("--smoke", action="store_true", help="2 trials x 4 epochs, wiring check only")
    return p.parse_args()


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()

    def isatty(self):
        return any(getattr(s, "isatty", lambda: False)() for s in self.streams)


_SPIRAL_CACHE: dict = {}


def cached_spiral_stack(transform, ds_key, seq_length, dilation, dyn_lengths, device):
    key = (ds_key, int(seq_length), int(dilation), tuple(int(d) for d in dyn_lengths), str(device))
    if key not in _SPIRAL_CACHE:
        _SPIRAL_CACHE[key] = sc.build_spiral_stack(
            transform, seq_length, dilation, dyn_lengths, device
        )
    return _SPIRAL_CACHE[key]


def level_sizes(transform) -> list[int]:
    """Vertex count at each level a conv operates on (i.e. every level except the coarsest)."""
    return [int(v.shape[0]) for v in transform["vertices"][:-1]]


def plan_conv_types(n_levels, conv_type, adaptive_levels, sizes):
    """Coarsest-k adaptive, then downgrade any level too large for the adaptive op."""
    types = resolve_conv_types(n_levels, conv_type, adaptive_levels)
    return [
        "spiral" if (t == "adaptive" and sizes[i] > MAX_ADAPTIVE_NODES) else t
        for i, t in enumerate(types)
    ]


def plan_dynamic_lengths(conv_types, sizes, frac):
    out = []
    for level_type, n in zip(conv_types, sizes):
        if level_type != "adaptive":
            out.append(1)
        else:
            out.append(int(max(8, min(MAX_DYNAMIC_SEQ, round(frac * n), n))))
    return out


def build_objective(args, spec, data, device, trial_rows, trial_dir, log):
    latent = int(spec["latent"])
    conv_type = spec["conv_type"]

    def objective(trial):
        ds_factors = [int(v) for v in trial.suggest_categorical("ds_factors", DS_CHOICES).split(",")]
        base = trial.suggest_int("base_channels", args.base_min, args.base_max,
                                 step=args.base_step)
        seq_length = trial.suggest_int("seq_length", 9, 27, step=2)
        dilation = trial.suggest_int("dilation", 1, 2)
        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        lr_decay = trial.suggest_float("lr_decay", 0.90, 0.999)
        decay_step = trial.suggest_int("decay_step", 1, 20)
        # v1 overfit badly, so the regularisation axes below were added and weight_decay
        # was widened by two orders of magnitude.
        weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.3, step=0.05)
        noise_std = trial.suggest_float("noise_std", 0.0, 0.10, step=0.02)
        scheduler_type = trial.suggest_categorical("scheduler", ["step", "cosine"])
        # NOT searched: a global linear branch would make this a linear autoencoder with a
        # spiral correction (measured: the spiral branch carried only 4.9% of the output),
        # which is no longer a SpiralNet++ / Adaptive-Spiral comparison. Opt-in flag only.
        linear_skip = bool(args.linear_skip)
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
        epochs = (4 if args.smoke else
                  trial.suggest_int("epochs", args.epochs_min, args.epochs_max, step=100))

        n_levels = len(ds_factors)
        out_channels = [base] * (n_levels - 1) + [2 * base]

        transform = sc.get_transform(ds_factors)
        sizes = level_sizes(transform)

        if conv_type == "adaptive":
            adaptive_levels = trial.suggest_int("adaptive_levels", 1, 2)
            dynamic_seq_frac = trial.suggest_float("dynamic_seq_frac",
                                                   args.dyn_frac_min, args.dyn_frac_max)
        else:
            adaptive_levels, dynamic_seq_frac = 0, 0.0

        conv_types = plan_conv_types(n_levels, conv_type, adaptive_levels, sizes)
        if conv_type == "adaptive" and "adaptive" not in conv_types:
            raise optuna.TrialPruned(
                f"no level under {MAX_ADAPTIVE_NODES} nodes for the adaptive op (sizes={sizes})"
            )
        dyn_lengths = plan_dynamic_lengths(conv_types, sizes, dynamic_seq_frac)

        pre_latent = int(transform["vertices"][-1].shape[0]) * out_channels[-1]
        trial.set_user_attr("pre_latent_dim", pre_latent)
        trial.set_user_attr("level_sizes", sizes)
        trial.set_user_attr("conv_types", conv_types)
        trial.set_user_attr("dynamic_seq_lengths", dyn_lengths)
        if pre_latent < 2 * latent:
            raise optuna.TrialPruned(
                f"pre-latent width {pre_latent} < 2x latent {latent}; structurally capped"
            )

        log(
            f"  trial {trial.number}: ds={ds_factors} oc={out_channels} seq={seq_length} "
            f"dil={dilation} bs={batch_size} ep={epochs} lr={lr:.2e} "
            f"pre_latent={pre_latent} conv={conv_types} dyn={dyn_lengths}"
        )

        ds_key = sc.ds_tag(ds_factors)
        spirals, dynamic, down, up = cached_spiral_stack(
            transform, ds_key, seq_length, dilation, dyn_lengths, device
        )

        torch.manual_seed(args.seed + trial.number)
        model = build_model(
            transform=transform,
            spiral_indices=spirals,
            dynamic_spiral_indices=dynamic,
            down_transform=down,
            up_transform=up,
            out_channels=out_channels,
            latent_channels=latent,
            conv_type=conv_type,
            adaptive_levels=adaptive_levels,
            conv_types=conv_types,  # guarded plan is authoritative
            dropout=dropout,
            linear_skip=linear_skip,
        ).to(device)
        n_params = model.num_parameters()
        trial.set_user_attr("n_params", n_params)

        torch.cuda.reset_peak_memory_stats(device)
        started = time.time()
        try:
            best_val, best_state, info = te.train_model(
                model,
                data,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                lr_decay=lr_decay,
                decay_step=decay_step,
                weight_decay=weight_decay,
                device=device,
                eval_every=args.eval_every,
                log_fn=log,
                report_fn=lambda value, epoch: trial.report(value, epoch),
                should_prune=trial.should_prune,
                seed=args.seed + trial.number,
                time_budget_s=None if args.smoke else args.trial_time_budget,
                noise_std=noise_std,
                scheduler_type=scheduler_type,
                patience=0 if args.smoke else args.patience,
                min_epochs=0 if args.smoke else args.min_epochs,
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                del model
                torch.cuda.empty_cache()
                log(f"  trial {trial.number}: CUDA OOM -> pruned")
                trial.set_user_attr("oom", True)
                raise optuna.TrialPruned("CUDA OOM") from exc
            raise

        if info.get("stopped_on_prune"):
            log(f"  trial {trial.number}: pruned (best so far {best_val:.6f})")
            del model
            torch.cuda.empty_cache()
            raise optuna.TrialPruned()

        duration = time.time() - started
        peak_gb = torch.cuda.max_memory_allocated(device) / 1e9

        checkpoint_fp = trial_dir / f"trial_{trial.number:04d}_val{best_val:.6f}.pt"
        torch.save(
            {
                "trial": int(trial.number),
                "experiment": args.exp,
                "conv_type": conv_type,
                "conv_types": conv_types,
                "latent_channels": latent,
                "out_channels": out_channels,
                "ds_factors": ds_factors,
                "seq_length": seq_length,
                "dilation": dilation,
                "adaptive_levels": adaptive_levels,
                "dynamic_seq_lengths": dyn_lengths,
                "adaptive_reset_parameters_fixed": True,
                "dropout": dropout,
                "noise_std": noise_std,
                "scheduler_type": scheduler_type,
                "linear_skip": linear_skip,
                "pre_latent_dim": pre_latent,
                "n_params": n_params,
                "best_val_rmse_mm": float(best_val),
                "best_epoch": info["best_epoch"],
                "stopped_on_time_budget": info["stopped_on_time_budget"],
                "stopped_on_patience": info["stopped_on_patience"],
                "epochs_completed": info["epochs_completed"],
                "params": dict(trial.params),
                "model_state_dict": best_state,
            },
            checkpoint_fp,
        )

        row = {
            "trial": trial.number,
            "experiment": args.exp,
            "val_rmse_mm": float(best_val),
            "best_epoch": info["best_epoch"],
            "epochs_completed": info["epochs_completed"],
            "stopped_on_time_budget": info["stopped_on_time_budget"],
            "stopped_on_patience": info["stopped_on_patience"],
            "linear_skip": linear_skip,
            "n_params": n_params,
            "pre_latent_dim": pre_latent,
            "conv_types": "|".join(conv_types),
            "dynamic_seq_lengths": "|".join(str(d) for d in dyn_lengths),
            "duration_s": round(duration, 1),
            "peak_gb": round(peak_gb, 2),
            "checkpoint": str(checkpoint_fp),
            **{f"p_{k}": v for k, v in trial.params.items()},
        }
        trial_rows.append(row)
        sc.atomic_write_csv(
            trial_dir.parent / "trial_metrics.csv",
            trial_rows,
            fieldnames=sorted({k for r in trial_rows for k in r}),
        )
        log(
            f"  trial {trial.number}: val_rmse_mm={best_val:.6f} params={n_params/1e6:.3f}M "
            f"peak={peak_gb:.2f}GB {duration:.0f}s"
        )

        del model
        torch.cuda.empty_cache()
        return float(best_val)

    return objective


def main():
    args = parse_args()
    spec = sc.EXPERIMENTS[args.exp]
    exp_key = f"{args.exp}_{args.tag}" if args.tag else args.exp
    dirs = sc.experiment_dirs(args.exp, tag=args.tag)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_fp = dirs["logs"] / f"{exp_key}_{stamp}{'_smoke' if args.smoke else ''}.log"
    handle = open(log_fp, "a", buffering=1)
    sys.stdout = Tee(sys.__stdout__, handle)
    sys.stderr = Tee(sys.__stderr__, handle)

    def log(message):
        print(message, flush=True)

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")
        device = torch.device("cuda", args.gpu)
        torch.cuda.set_device(device)

        log("=" * 88)
        log(f"experiment={args.exp} conv={spec['conv_type']} latent={spec['latent']}")
        log(f"gpu={args.gpu} ({torch.cuda.get_device_name(device)}) trials={args.n_trials} "
            f"smoke={args.smoke}")
        log(f"log={log_fp}")
        log(f"start={datetime.datetime.now().isoformat(timespec='seconds')}")
        log("=" * 88)

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        rows = sc.read_manifest()
        data = te.load_mesh_tensors(device, rows=rows)
        log(f"data: train={len(data.train)} val={len(data.val)} test={len(data.test)} "
            f"verts={data.train.shape[1]}")

        trial_dir = sc.makedirs(dirs["study"] / "trials")
        # Resume-safe: the CSV is rewritten wholesale from this list, so a restart with an
        # empty list truncates every row written before the restart. Seed it from disk.
        # (study.db and the per-trial checkpoints remain the authoritative record either way.)
        trial_rows: list[dict] = []
        _existing = dirs["study"] / "trial_metrics.csv"
        if _existing.exists():
            import csv as _csv
            with open(_existing, newline="") as _h:
                trial_rows = [r for r in _csv.DictReader(_h) if r.get("val_rmse_mm")]
            log(f"resumed trial_metrics.csv with {len(trial_rows)} prior rows")
        objective = build_objective(args, spec, data, device, trial_rows, trial_dir, log)

        storage = f"sqlite:///{dirs['study'] / 'study.db'}"
        study = optuna.create_study(
            direction="minimize",
            study_name=f"{exp_key}{'_smoke' if args.smoke else ''}",
            storage=storage,
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=args.seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=10),
        )
        n_trials = 2 if args.smoke else args.n_trials
        study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

        complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not complete:
            raise SystemExit("No trial completed; nothing to select.")
        best = study.best_trial
        log("")
        log(f"BEST trial {best.number}: val_rmse_mm={best.value:.6f}")
        log(f"  params: {json.dumps(best.params, indent=2, sort_keys=True)}")

        # Test set is touched exactly once, here, on the selected model.
        # int(...) on both sides: rows resumed from CSV have string fields, so a naive
        # `r["trial"] == best.number` silently matches nothing and the lookup blows up.
        _match = [r for r in trial_rows if int(r["trial"]) == int(best.number)]
        if not _match:
            raise SystemExit(
                f"no CSV/checkpoint row for best trial {best.number}; "
                f"run rebuild_trial_csv.py on this study first"
            )
        best_ckpt_fp = Path(_match[0]["checkpoint"])
        payload = torch.load(best_ckpt_fp, map_location="cpu")
        transform = sc.get_transform(payload["ds_factors"])
        spirals, dynamic, down, up = cached_spiral_stack(
            transform,
            sc.ds_tag(payload["ds_factors"]),
            payload["seq_length"],
            payload["dilation"],
            payload["dynamic_seq_lengths"],
            device,
        )
        model = build_model(
            transform=transform,
            spiral_indices=spirals,
            dynamic_spiral_indices=dynamic,
            down_transform=down,
            up_transform=up,
            out_channels=payload["out_channels"],
            latent_channels=payload["latent_channels"],
            conv_type=payload["conv_type"],
            adaptive_levels=payload["adaptive_levels"],
            conv_types=payload["conv_types"],
            dropout=payload.get("dropout", 0.0),
            linear_skip=payload.get("linear_skip", False),
        ).to(device)
        model.load_state_dict(payload["model_state_dict"])

        final = {
            split: te.evaluate(model, data, split, with_volume=True) for split in sc.SPLITS
        }
        for split in sc.SPLITS:
            log(f"  {split:5s} rmse_mm={final[split]['vertex_rmse_mm_mean']:.6f}")

        # ---- stage 2: seed repeats of the winning config (the reportable number) ----
        PCA_VAL = {128: 0.033668, 256: 0.010338}
        PCA_TEST = {128: 0.034380, 256: 0.010787}
        latent = int(spec["latent"])
        n_seeds = 0 if args.smoke else int(args.seed_repeats)
        seed_rows = []
        if n_seeds:
            log(f"\nstage 2: {n_seeds} seed repeats of trial {best.number}")
            bp = payload
            for si in range(n_seeds):
                torch.manual_seed(2000 + si)
                m = build_model(
                    transform=transform, spiral_indices=spirals,
                    dynamic_spiral_indices=dynamic, down_transform=down, up_transform=up,
                    out_channels=bp["out_channels"], latent_channels=bp["latent_channels"],
                    conv_type=bp["conv_type"], adaptive_levels=bp["adaptive_levels"],
                    conv_types=bp["conv_types"], dropout=bp.get("dropout", 0.0),
                    linear_skip=bp.get("linear_skip", False)).to(device)
                bv, st, _ = te.train_model(
                    m, data, epochs=bp["params"].get("epochs", 500),
                    batch_size=bp["params"]["batch_size"], lr=bp["params"]["lr"],
                    lr_decay=bp["params"]["lr_decay"], decay_step=bp["params"]["decay_step"],
                    weight_decay=bp["params"]["weight_decay"], device=device,
                    eval_every=args.eval_every, seed=2000 + si,
                    time_budget_s=args.trial_time_budget,
                    noise_std=bp.get("noise_std", 0.0),
                    scheduler_type=bp.get("scheduler_type", "step"),
                    patience=args.patience, min_epochs=args.min_epochs)
                if st is not None:
                    m.load_state_dict(st)
                row = {"seed": 2000 + si, "val_rmse_mm": bv}
                for sp in sc.SPLITS:
                    row[f"{sp}_rmse_mm"] = te.evaluate(m, data, sp)["vertex_rmse_mm_mean"]
                seed_rows.append(row)
                log(f"  seed {2000+si}: val={bv:.6f} test={row['test_rmse_mm']:.6f}")
                del m
                torch.cuda.empty_cache()
            sc.atomic_write_csv(dirs["best"] / "seed_repeats.csv", seed_rows)
            import numpy as _np
            vals = _np.array([r["val_rmse_mm"] for r in seed_rows])
            tests = _np.array([r["test_rmse_mm"] for r in seed_rows])
            beats = bool(tests.mean() + tests.std() < PCA_TEST[latent])
            log(f"\n  seeds: val {vals.mean():.6f}+-{vals.std():.6f}  "
                f"test {tests.mean():.6f}+-{tests.std():.6f}")
            log(f"  PCA-{latent} test {PCA_TEST[latent]:.6f}  ratio {tests.mean()/PCA_TEST[latent]:.3f}x"
                f"  => {'BEATS PCA' if beats else 'does not beat PCA'}")
            final["seed_repeats"] = {
                "n": len(seed_rows), "val_mean": float(vals.mean()), "val_sd": float(vals.std()),
                "test_mean": float(tests.mean()), "test_sd": float(tests.std()),
                "pca_test": PCA_TEST[latent], "pca_val": PCA_VAL[latent],
                "ratio_vs_pca_test": float(tests.mean() / PCA_TEST[latent]),
                "beats_pca_beyond_seed_noise": beats}

        best_payload = dict(payload)
        best_payload["metrics"] = final
        best_payload["study_best_trial"] = int(best.number)
        torch.save(best_payload, dirs["best"] / "best_model.pt")
        sc.atomic_write_json(
            dirs["best"] / "best_summary.json",
            {
                "experiment": args.exp,
                "tag": args.tag,
                "conv_type": spec["conv_type"],
                "latent_channels": spec["latent"],
                "best_trial": int(best.number),
                "best_val_rmse_mm": float(best.value),
                "best_params": best.params,
                "n_params": payload["n_params"],
                "pre_latent_dim": payload["pre_latent_dim"],
                "conv_types": payload["conv_types"],
                "dynamic_seq_lengths": payload["dynamic_seq_lengths"],
                "metrics": final,
                "n_trials_requested": n_trials,
                "n_trials_complete": len(complete),
                "log_file": str(log_fp),
                "source_checkpoint": str(best_ckpt_fp),
            },
        )
        sc.atomic_write_csv(
            dirs["study"] / "study_trials.csv",
            [
                {
                    "trial": t.number,
                    "state": str(t.state),
                    "value": t.value,
                    **{f"p_{k}": v for k, v in t.params.items()},
                    **{f"u_{k}": v for k, v in t.user_attrs.items()},
                }
                for t in study.trials
            ],
            fieldnames=None,
        )
        log(f"wrote best model + summary to {dirs['best']}")
        log(f"end={datetime.datetime.now().isoformat(timespec='seconds')}")
    except Exception:
        traceback.print_exc()
        raise
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        handle.close()


if __name__ == "__main__":
    main()
