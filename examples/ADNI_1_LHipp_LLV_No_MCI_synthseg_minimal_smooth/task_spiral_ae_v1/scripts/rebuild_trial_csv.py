#!/usr/bin/env python3
"""Rebuild trial_metrics.csv from study.db + per-trial checkpoints.

Needed because trial_metrics.csv was rewritten wholesale from an in-memory list, so a restart
truncated rows written before it. study.db and the checkpoints are the authoritative record;
this reconstructs the convenience CSV from them. The driver is now resume-safe, but this
recovers studies that were already truncated.

    python rebuild_trial_csv.py <study_dir> [<study_dir> ...]
"""
import sys, pathlib, csv, optuna, torch
optuna.logging.set_verbosity(optuna.logging.WARNING)

for d in map(pathlib.Path, sys.argv[1:]):
    name = d.name
    st = optuna.load_study(study_name=name, storage=f"sqlite:///{d/'study.db'}")
    by_num = {t.number: t for t in st.trials
              if t.state == optuna.trial.TrialState.COMPLETE}
    rows = []
    for ck in sorted((d / "trials").glob("*.pt")):
        p = torch.load(ck, map_location="cpu")
        n = int(p["trial"])
        t = by_num.get(n)
        row = {"trial": n, "val_rmse_mm": float(p["best_val_rmse_mm"]),
               "best_epoch": p.get("best_epoch"), "n_params": p.get("n_params"),
               "pre_latent_dim": p.get("pre_latent_dim"),
               "conv_types": "|".join(p.get("conv_types") or []),
               "dynamic_seq_lengths": "|".join(str(x) for x in (p.get("dynamic_seq_lengths") or [])),
               "epochs_completed": p.get("epochs_completed"),
               "stopped_on_time_budget": p.get("stopped_on_time_budget"),
               "linear_skip": p.get("linear_skip"), "checkpoint": str(ck),
               "recovered_from_checkpoint": True}
        for k, v in (p.get("params") or {}).items():
            row[f"p_{k}"] = v
        if t is not None:
            for k, v in t.user_attrs.items():
                row.setdefault(f"u_{k}", v)
        rows.append(row)
    rows.sort(key=lambda r: r["trial"])
    out = d / "trial_metrics.csv"
    fields = sorted({k for r in rows for k in r})
    with open(out, "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"{name}: rebuilt {len(rows)} rows -> {out}")
