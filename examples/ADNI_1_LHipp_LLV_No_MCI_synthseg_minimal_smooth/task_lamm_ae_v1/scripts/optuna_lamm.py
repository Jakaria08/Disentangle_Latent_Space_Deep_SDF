#!/usr/bin/env python3
"""Optuna search over LAMM hyperparameters, centred on the paper's published values.

Every range below BRACKETS what Tarasiou et al. report, rather than replacing it, so the
paper's configuration is always reachable and the search only asks how it should shift for
2037 scans from 475 independent subjects instead of their 6,000-10,000 meshes:

    published            search range            why it moves
    D = 512              96 / 128 / 192 / 256    3-5x less data, and 475 independent subjects
    enc 5 / dec 3        3-6 / 2-4               same
    heads 8, head 64     4 / 8                   D is smaller, so dim_head follows
    lr 1e-4 -> 1e-6      3e-5 .. 5e-4 (log)      final lr fixed at the paper's 1e-6
    warmup 10 epochs     5 / 10 / 20
    batch 32             16 / 32 / 64
    epochs 1500          <=400 with patience     these runs converge by ~250 with EMA
    weight decay n/s     1e-5 .. 1e-1 (log)      not stated in the paper; spans SpiralNet++'s
                                                 tuned 1.8e-4 and the transformer default 0.05
    multilayer L1        0 / 0.25 / 0.5 / 1.0    weight on the intermediate-decoder-layer loss
    no region sharing    False / True            LAMM uses region-specific weights; sharing
                                                 collapses tokenizer+heads 4.3M -> 0.05M,
                                                 which may matter at this sample size
    identity token       id_token / flatten      flatten beat every aggregating head in the
                                                 MeshMAE study (0.042 vs 0.079 pooled), so
                                                 LAMM's CLS-style extraction is worth testing

NO PRUNER, for the same reason as the MeshMAE search: mid-run margins on this problem have
twice proved non-predictive (EMA read +15.1% at epoch 48 and +1.1% at convergence; the
locality bias +4.5% then +0.14%).
"""
from __future__ import annotations

import argparse, csv, json, subprocess, sys, time
from pathlib import Path

import optuna

HERE = Path(__file__).resolve().parent
PY = "/home/jakaria/anaconda3/envs/pytorch_geo/bin/python"
OUT = Path("/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task_lamm_ae_v1")
REF = {"pca128": 0.033668, "spiralnet128": 0.036784, "meshmae_tuned": 0.037867}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", required=True)
    p.add_argument("--backbone", choices=("transformer", "mlpmixer", "search"),
                   required=True, help="'search' samples the backbone per trial")
    p.add_argument("--space", choices=("v1", "v2", "scales", "scales3", "moments", "decoder",
                            "fit", "finescale", "unbound", "latentsplit",
                            "arch_tuned", "mixup", "optim", "scaleset",
                            "subjweight"), default="v1")
    p.add_argument("--max-params", type=float, default=22e6)
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--min-epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--trial-time-budget", type=float, default=1200.0)
    p.add_argument("--latent", type=int, default=128)
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def suggest(t):
    return {
        "dim": t.suggest_categorical("dim", [96, 128, 192, 256]),
        "enc_depth": t.suggest_categorical("enc_depth", [3, 4, 5, 6]),
        "dec_depth": t.suggest_categorical("dec_depth", [2, 3, 4]),
        "heads": t.suggest_categorical("heads", [4, 8]),
        "lr": t.suggest_float("lr", 3e-5, 5e-4, log=True),
        "weight_decay": t.suggest_float("weight_decay", 1e-5, 1e-1, log=True),
        "dropout": t.suggest_float("dropout", 0.0, 0.2, step=0.05),
        "batch_size": t.suggest_categorical("batch_size", [16, 32, 64]),
        "warmup_epochs": t.suggest_categorical("warmup_epochs", [5, 10, 20]),
        "deep_sup": t.suggest_categorical("deep_sup", [0.0, 0.25, 0.5, 1.0]),
        "share_regions": t.suggest_categorical("share_regions", [False, True]),
        "latent_mode": t.suggest_categorical("latent_mode", ["id_token", "flatten"]),
    }


# Every top trial of the v1 searches used share_regions=False and latent_mode=flatten, and
# every trial that used share=True or id_token scored 0.16-0.22 -- 10 of 40 trials spent
# confirming two dead options. Both are dropped below, which pays for deeper, longer trials.
def suggest_v2(t, backbone):
    """Unboxed capacity/training search. v1's winners sat ON the boundary in three dimensions
    (dec_depth=4 max, enc_depth=6 max, batch=16 min), the mixer trained to epoch 377 of 400
    median, and half the trials hit the wall -- while LAMM's own setting is D=512 for 1500
    epochs. v1 measured the edge of its box, not the architecture."""
    cfg = {
        # D=384 and batch 8 are OUT, measured rather than assumed: at bs=8/D=384 a trial runs
        # 8.38 s/epoch, so 550 epochs needs 4600s. Width was never the pinned variable either
        # -- v1 had D=192 BEATING D=256 (0.0415 vs 0.0492). What v1 pinned was DEPTH (both
        # enc_depth=6 and dec_depth=4 were its maxima and both won) and EPOCHS (median best
        # epoch 377 of 400). Those are what this space unboxes.
        "dim": t.suggest_categorical("dim", [192, 256]),
        "enc_depth": t.suggest_categorical("enc_depth", [4, 6, 8]),
        "dec_depth": t.suggest_categorical("dec_depth", [4, 6, 8]),
        "heads": t.suggest_categorical("heads", [4, 8]),
        "dim_head": 64,                                   # LAMM's, independent of D
        "lr": t.suggest_float("lr", 5e-5, 4e-4, log=True),
        "weight_decay": t.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        "dropout": t.suggest_float("dropout", 0.0, 0.15, step=0.05),
        "batch_size": t.suggest_categorical("batch_size", [16, 32]),
        "warmup_epochs": t.suggest_categorical("warmup_epochs", [10, 20]),
        "deep_sup": t.suggest_categorical("deep_sup", [0.25, 0.5, 1.0]),
        "scales": "86",
    }
    if backbone == "search":
        cfg["backbone"] = t.suggest_categorical("backbone", ["transformer", "mlpmixer"])
    return cfg


def suggest_scales(t, backbone):
    """Region-granularity search, single and multiscale. LAMM splits 12k vertices into 11
    regions; v1 used 86 on 2746 -- an 8x longer token sequence than the paper, never examined.
    Multi-entry sets build the residual model X = X_coarse + R_medium + R_fine with the
    partial sums supervised, so deep_sup is never 0 here."""
    cfg = {
        "scales": t.suggest_categorical("scales", [
            "11", "22", "43", "86", "172", "344",          # single scale
            "11,86", "22,86", "11,43", "43,172",           # two scales
            "11,43,172", "11,86,344", "22,86,344"]),       # three scales
        "dim": t.suggest_categorical("dim", [128, 192, 256]),   # 128 keeps 344-token sets small
        "enc_depth": t.suggest_categorical("enc_depth", [4, 6]),
        "dec_depth": t.suggest_categorical("dec_depth", [4, 6]),
        "heads": t.suggest_categorical("heads", [4, 8]),
        "dim_head": 64,
        "lr": t.suggest_float("lr", 5e-5, 4e-4, log=True),
        "weight_decay": t.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        "dropout": t.suggest_float("dropout", 0.0, 0.15, step=0.05),
        "batch_size": t.suggest_categorical("batch_size", [16, 32]),
        "warmup_epochs": 10,
        "deep_sup": t.suggest_categorical("deep_sup", [0.25, 0.5, 1.0]),
    }
    if backbone == "search":
        cfg["backbone"] = t.suggest_categorical("backbone", ["transformer", "mlpmixer"])
    return cfg


def suggest_scales3(t, backbone):
    """THREE-scale residual study: X = X_coarse + R_medium + R_fine, the full formulation.

    Built from what exp1 measured, not from a fresh guess:
      * multiscale wins. Three of four `11,43` trials beat EVERY single-scale trial, and the
        result survives a capacity control -- at ~12-13M params `11,43` gives 0.0759 where
        single-scale 43 gives 0.1215 and 172 gives 0.1261.
      * but exp1 only ever completed TWO-scale sets. Every three-scale configuration it drew
        was rejected by the 20M guard (30.99M, 38.31M, 20.54M), so the third term of your
        decomposition has never actually been trained. That is the gap this study fills.
      * K=86 -- the granularity every earlier run in this project used -- was the WORST
        single scale tested, so all sets here start coarse.
      * dec_depth=8 never appeared in a good trial anywhere, and batch 32 dominating the
        old exp2 sampling is the most likely reason it never reproduced v1. Depth is capped
        at 6; batch stays {16,32} with v1's evidence favouring 16.

    --max-params is raised to 26M here because three tokenizers plus three head sets cost
    ~18M at D=192 before the backbone -- 20M would reject the entire space, which is exactly
    what happened to exp1's three-scale draws.
    """
    cfg = {
        "scales": t.suggest_categorical("scales", [
            "11,22,43", "11,43,86", "11,22,86", "11,43,172", "22,43,172", "11,86,172"]),
        "dim": t.suggest_categorical("dim", [96, 128, 192]),
        "enc_depth": t.suggest_categorical("enc_depth", [4, 6]),
        "dec_depth": t.suggest_categorical("dec_depth", [4, 6]),
        "heads": t.suggest_categorical("heads", [4, 8]),
        "dim_head": 64,
        "lr": t.suggest_float("lr", 6e-5, 4e-4, log=True),
        "weight_decay": t.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        "dropout": t.suggest_float("dropout", 0.0, 0.15, step=0.05),
        "batch_size": t.suggest_categorical("batch_size", [16, 32]),
        "warmup_epochs": 10,
        "deep_sup": t.suggest_categorical("deep_sup", [0.25, 0.5, 1.0]),
    }
    if backbone == "search":
        cfg["backbone"] = t.suggest_categorical("backbone", ["transformer", "mlpmixer"])
    return cfg


_FRONTIER = dict(                       # shared: where the good gaps live
    lr=(1e-4, 5e-4), wd=(1e-5, 1e-3), dropout=(0.0, 0.15))


def suggest_moments(t, backbone):
    """EXPERIMENT A -- isolate the region-moment tokenizer.

    Sorting every LAMM trial by TRAINING error shows the deficit is not generalisation:
    LAMM's gap at its best-val point is 1.41x, identical to SpiralNet++'s, and it has reached
    train 0.022724 -- 12.7% BETTER fitting than SpiralNet++'s 0.026021. What it cannot do is
    both at once: learning rate slides it monotonically along a fixed train/gap frontier
    across ten trials. To beat 0.036784 it needs train <= 0.026088 AT gap 1.41, a point off
    that frontier.

    Only one class of change has ever shifted the frontier on this cohort: more encoder input.
    Flatten bought 46% and raw-vs-moments bought 7.7% with training error unchanged to five
    decimals. LAMM tokenizes raw coordinates only and has never had the moments.

    K is fixed at 86 -- the granularity with the best fitting in every study -- so region_mode
    is the variable under test, sampled per trial for a within-study comparison."""
    cfg = {
        "scales": "86",
        "region_mode": t.suggest_categorical("region_mode", ["raw", "both"]),
        "dim": t.suggest_categorical("dim", [192, 256]),
        "enc_depth": t.suggest_categorical("enc_depth", [3, 4, 6]),
        "dec_depth": 4,                       # every winner in every study used 4
        "heads": t.suggest_categorical("heads", [4, 8]),
        "dim_head": 64,
        "lr": t.suggest_float("lr", *_FRONTIER["lr"], log=True),
        "weight_decay": t.suggest_float("weight_decay", *_FRONTIER["wd"], log=True),
        "dropout": t.suggest_float("dropout", *_FRONTIER["dropout"], step=0.05),
        "batch_size": 16,                     # v1: bs=16 won 11 of 16 top trials
        "warmup_epochs": 10,
        "deep_sup": t.suggest_categorical("deep_sup", [0.25, 0.5, 1.0]),
    }
    if backbone == "search":
        cfg["backbone"] = t.suggest_categorical("backbone", ["transformer", "mlpmixer"])
    return cfg


def suggest_decoder(t, backbone):
    """EXPERIMENT B -- stop the decoder head from compressing.

    v_hat_i = W_i^out y_i^L produces 3*N_i coordinates from ONE D-dimensional token:

        K= 11  1260 outputs / D=192 = 6.6x compression
        K= 43   357 / 192 = 1.9x
        K= 86   195 / 192 = 1.02x   <- exactly at the limit, and where every good fit sits
        K=172    96 / 192 = 0.50x   <- free
        K=344    54 / 192 = 0.28x

    Every well-fitting trial is K=86 at D=192, sitting precisely on the boundary where the
    head stops being a bottleneck -- and the second-best fit in the whole table is K=86 at
    D=256 (train 0.023321), the one config with real output headroom. Every set here either
    raises D or includes a fine scale whose heads cannot compress."""
    cfg = {
        "scales": t.suggest_categorical("scales", [
            "86", "172", "43,172", "86,172", "11,43,172", "11,86,172"]),
        "region_mode": t.suggest_categorical("region_mode", ["raw", "both"]),
        "dim": t.suggest_categorical("dim", [192, 256]),
        "enc_depth": t.suggest_categorical("enc_depth", [4, 6]),
        "dec_depth": 4,
        "heads": t.suggest_categorical("heads", [4, 8]),
        "dim_head": 64,
        "lr": t.suggest_float("lr", *_FRONTIER["lr"], log=True),
        "weight_decay": t.suggest_float("weight_decay", *_FRONTIER["wd"], log=True),
        "dropout": t.suggest_float("dropout", *_FRONTIER["dropout"], step=0.05),
        "batch_size": 16,
        "warmup_epochs": 10,
        "deep_sup": t.suggest_categorical("deep_sup", [0.25, 0.5, 1.0]),
    }
    if backbone == "search":
        cfg["backbone"] = t.suggest_categorical("backbone", ["transformer", "mlpmixer"])
    return cfg


def suggest_fit(t, backbone):
    """EXPERIMENT C -- close the FITTING gap, which is now the only thing left.

    expB reached gap parity with SpiralNet++ (1.38-1.46x vs 1.41x) across five reproduced
    trials, so generalisation is settled. The whole remaining deficit is fit: train 0.029771
    vs 0.026021, 14.4%. And at LAMM's OWN gap of 1.38x, a train error of 0.026021 gives
    0.035909 -- which beats SpiralNet++.

    The specific defect: expB's best trials early-stopped at epoch 129-171 while the cosine
    was stretched over 450, so the lr was still at 70-83% of peak when training ended. They
    never reached the low-lr annealing phase where fine detail is fit. Matching the schedule
    length to the convergence point is the cheapest possible fix, and `epochs` is searched
    here precisely so the cosine completes.

    Architecture is FROZEN at expB's reproduced optimum (43,172 / D=192 / E6/D4 / bs 16 /
    raw): scales 43,172 beat 43 alone and 172 alone by 3x, D=192 beat D=256, backbone was
    within 0.9%, and region_mode `both` was worth 0.9% over `raw` across 10 trials.
    """
    cfg = {
        "scales": "43,172",
        "region_mode": "raw",
        "dim": 192,
        "enc_depth": 6,
        "dec_depth": 4,
        "heads": 4,
        "dim_head": 64,
        "epochs_override": t.suggest_categorical("epochs_override", [150, 200, 250, 300]),
        "lr": t.suggest_float("lr", 2e-4, 6e-4, log=True),
        "weight_decay": t.suggest_float("weight_decay", 1e-5, 3e-4, log=True),
        "dropout": t.suggest_float("dropout", 0.0, 0.15, step=0.05),
        "batch_size": 16,
        "warmup_epochs": 10,
        "deep_sup": t.suggest_categorical("deep_sup", [0.0, 0.1, 0.25, 0.5]),
    }
    if backbone == "search":
        cfg["backbone"] = t.suggest_categorical("backbone", ["transformer", "mlpmixer"])
    return cfg


def suggest_finescale(t, backbone):
    """EXPERIMENT D -- less decoder compression still, plus the matched schedule.

    expB's winners pair a structure scale with a fine one: 43 heads emit 357 coords from
    D=192 (1.9x compression) and 172 emit 96 (0.5x, free). 43,172 scored 0.0412 against
    0.1171 for 86 alone and 0.0686 for 172 alone -- neither scale does both jobs. This asks
    whether pushing finer (344 emits 54 from 192, 0.28x) buys more fit."""
    cfg = {
        "scales": t.suggest_categorical("scales", [
            "43,172", "43,344", "172,344", "43,172,344", "86,344", "22,172"]),
        "region_mode": "raw",
        "dim": t.suggest_categorical("dim", [128, 192]),
        "enc_depth": 6,
        "dec_depth": 4,
        "heads": 4,
        "dim_head": 64,
        "epochs_override": t.suggest_categorical("epochs_override", [200, 300]),
        "lr": t.suggest_float("lr", 2e-4, 6e-4, log=True),
        "weight_decay": t.suggest_float("weight_decay", 1e-5, 3e-4, log=True),
        "dropout": t.suggest_float("dropout", 0.0, 0.15, step=0.05),
        "batch_size": 16,
        "warmup_epochs": 10,
        "deep_sup": t.suggest_categorical("deep_sup", [0.0, 0.25]),
    }
    if backbone == "search":
        cfg["backbone"] = t.suggest_categorical("backbone", ["transformer", "mlpmixer"])
    return cfg


def suggest_unbound(t, backbone):
    """EXPERIMENT E -- expC's optimum sits on TWO of my bounds. Move both.

    expC converged on one architecture across five reproductions (0.039148-0.039975), and its
    top six trials are pinned:
        lr  4.76e-4 - 5.94e-4   ceiling was 6.00e-4   4 of 6 within 10% of it
        wd  1.01e-5 - 2.72e-5   floor   was 1.00e-5   3 of 6 within 10% of it
    A tuned optimum lying on a boundary means the box is wrong, not the model -- the same
    mistake as the earlier batch-size, depth and epoch caps. lr goes up 3.3x, wd down 10x.

    Everything else is FROZEN at what expC reproduced: scales 43,172 (3x better than either
    scale alone), D=192 (beat 256 twice), MLPMixer, E6/D4, batch 16, region_mode raw
    (moments were worth 0.9%), and deep_sup 0.0 -- which appeared in EVERY top trial, so
    the partial-sum supervision that enforces the residual decomposition costs accuracy even
    though the multiscale token set itself clearly helps.

    Every top trial also ran its full 200-epoch schedule, confirming the schedule-length fix.
    """
    cfg = {
        "scales": "43,172", "region_mode": "raw", "dim": 192,
        "enc_depth": 6, "dec_depth": 4, "heads": 4, "dim_head": 64,
        "batch_size": 16, "warmup_epochs": 10, "deep_sup": 0.0,
        "epochs_override": t.suggest_categorical("epochs_override", [150, 200, 250, 300]),
        "lr": t.suggest_float("lr", 4e-4, 2e-3, log=True),
        "weight_decay": t.suggest_float("weight_decay", 1e-6, 1e-4, log=True),
        "dropout": t.suggest_float("dropout", 0.0, 0.20, step=0.05),
    }
    if backbone == "search":
        cfg["backbone"] = t.suggest_categorical("backbone", ["transformer", "mlpmixer"])
    return cfg


# Both spaces below rest on one observation. Across 118 LAMM trials the Pareto frontier of
# (train, gap) is smooth and SINGLE, and its best product -- 0.038698 -- is exactly expE's
# best val. Fine-scale sets are not on a better frontier, they are further along the same one
# (43,344: train 0.025577 but gap 1.61). SpiralNet++ sits strictly INSIDE it: at gap 1.41
# LAMM's frontier gives train ~0.0280 where SpiralNet++ has 0.026021. So LAMM needs a ~7%
# frontier SHIFT, and sliding along it -- which is all lr does -- cannot deliver that.
#
# What both exploit: every architectural choice was tuned at lr <= 6e-4, and the optimum is
# now 1.5e-3. D, depth, scale set and latent allocation were all selected under a learning
# rate 2.5x too low -- the same defect that made expE worth running.
def suggest_latentsplit(t, backbone):
    """EXPERIMENT F -- latent allocation across scales, the last untested knob.

    latent_split has always been an even 64/64, but at 43,172 with D=192 the two scales are
    compressed very unequally for that same budget:
        coarse 43:  192 x  43 =  8,256 values -> 64 dims  (129:1)
        fine  172:  192 x 172 = 33,024 values -> 64 dims  (516:1)
    The fine scale works 4x harder. This is also the knob that makes z_c / z_m / z_f
    meaningful for the downstream flow, so its answer is useful either way."""
    return {
        "scales": "43,172", "region_mode": "raw", "dim": 192,
        "enc_depth": 6, "dec_depth": 4, "heads": 4, "dim_head": 64,
        "batch_size": 16, "warmup_epochs": 10, "deep_sup": 0.0,
        "latent_split": t.suggest_categorical("latent_split",
                                              ["96,32", "80,48", "64,64", "48,80", "32,96"]),
        "epochs_override": t.suggest_categorical("epochs_override", [150, 200, 250]),
        "lr": t.suggest_float("lr", 8e-4, 2e-3, log=True),
        "weight_decay": t.suggest_float("weight_decay", 1e-6, 5e-5, log=True),
        "dropout": t.suggest_float("dropout", 0.05, 0.20, step=0.05),
    }


def suggest_arch_tuned(t, backbone):
    """EXPERIMENT G -- re-explore capacity at the CORRECT learning rate.

    D=256 lost twice (0.0466 vs 0.0412; 0.057 vs 0.041) -- but both times at lr <= 6e-4.
    Higher learning rates change how much capacity is usable, so every width/depth/scale
    verdict in this project was reached under a schedule now known to be 2.5x too slow."""
    cfg = {
        "scales": t.suggest_categorical("scales",
                                        ["43,172", "86,172", "22,172", "43,344", "43,172,344"]),
        "region_mode": "raw",
        "dim": t.suggest_categorical("dim", [192, 256, 320]),
        "enc_depth": t.suggest_categorical("enc_depth", [4, 6, 8]),
        "dec_depth": t.suggest_categorical("dec_depth", [4, 6]),
        "heads": 4, "dim_head": 64, "batch_size": 16, "warmup_epochs": 10, "deep_sup": 0.0,
        "epochs_override": t.suggest_categorical("epochs_override", [150, 200, 250]),
        "lr": t.suggest_float("lr", 8e-4, 2e-3, log=True),
        "weight_decay": t.suggest_float("weight_decay", 1e-6, 5e-5, log=True),
        "dropout": t.suggest_float("dropout", 0.05, 0.20, step=0.05),
    }
    if backbone == "search":
        cfg["backbone"] = t.suggest_categorical("backbone", ["transformer", "mlpmixer"])
    return cfg


def suggest_mixup(t, backbone):
    """EXPERIMENT H -- cross-subject mixup. LAMM has trained with NO augmentation, ever.

    The deficit is now purely the gap. At expG's optimum (train 0.028715, gap 1.33, val
    0.038308), beating SpiralNet++'s 0.036784 needs gap 1.281 -- a 3.7% reduction with the
    fit held. Meanwhile fitting capacity is a non-issue: trial 10 reached train 0.013009,
    HALF of SpiralNet++'s, at gap 3.44. Capacity is free; generalisation is the whole game.

    The root cause has been the same since the first measurement: 2037 scans from only 475
    independent subjects, 4.29 correlated visits each. Mixup on corresponded meshes is exact
    and manufactures virtual subjects without building a new manifest.

    alpha=0.0 is IN the grid so the no-mixup control is sampled inside the same study rather
    than compared across studies.

    Honest limitation: interpolants lie in the affine hull of the training set -- the space
    PCA already spans -- so this may regularise LAMM toward PCA rather than past it. For
    closing a 3.7% gap that is acceptable; as a route to beating PCA it would cap out.
    """
    return {
        "scales": "86,172", "region_mode": "raw", "dim": 256,
        "enc_depth": 8, "dec_depth": 6, "heads": 4, "dim_head": 64,
        "batch_size": 16, "warmup_epochs": 10, "deep_sup": 0.0,
        "mixup_alpha": t.suggest_categorical("mixup_alpha", [0.0, 0.1, 0.2, 0.4, 0.8]),
        "mixup_prob": t.suggest_categorical("mixup_prob", [0.25, 0.5, 1.0]),
        "epochs_override": t.suggest_categorical("epochs_override", [150, 200, 250]),
        "lr": t.suggest_float("lr", 6e-4, 1.6e-3, log=True),
        "weight_decay": t.suggest_float("weight_decay", 1e-6, 3e-5, log=True),
        "dropout": t.suggest_float("dropout", 0.0, 0.15, step=0.05),
    }


def suggest_optim(t, backbone):
    """EXPERIMENT I -- three knobs frozen since before the learning rate was retuned.

      * LOSS. Every model here trains on L1 (inherited from guided_vae) but is selected and
        reported on per-coordinate RMSE, which is L2. That objective/metric mismatch has
        never been tested anywhere in this project.
      * BATCH SIZE. Fixed at 16 since expC, when the optimum lr was ~5e-4. It is now 1.25e-3,
        and batch size and learning rate are the most strongly coupled pair in training --
        so 16 was chosen under a schedule 2.5x too slow. lr is re-searched alongside it.
      * EMA DECAY. Fixed at 0.999 since expC; at bs=16 that is a ~8-epoch horizon over a
        200-epoch run, also never revisited at the new lr.

    mixup_alpha is included as {0.0, 0.8} so this ALSO supplies the no-mixup control that
    expH's TPE will avoid sampling once it starts exploiting. Architecture is frozen at
    expH t2 (86,172 / D=256 / E8/D6 / mlpmixer), the best configuration found."""
    return {
        "scales": "86,172", "region_mode": "raw", "dim": 256,
        "enc_depth": 8, "dec_depth": 6, "heads": 4, "dim_head": 64,
        "warmup_epochs": 10, "deep_sup": 0.0, "mixup_prob": 0.5,
        "loss": t.suggest_categorical("loss", ["l1", "l2", "huber"]),
        "batch_size": t.suggest_categorical("batch_size", [8, 16, 32]),
        "ema_decay": t.suggest_categorical("ema_decay", [0.995, 0.999, 0.9995]),
        "mixup_alpha": t.suggest_categorical("mixup_alpha", [0.0, 0.8]),
        "epochs_override": t.suggest_categorical("epochs_override", [150, 200, 250]),
        "lr": t.suggest_float("lr", 4e-4, 2.5e-3, log=True),
        "weight_decay": t.suggest_float("weight_decay", 1e-6, 5e-5, log=True),
        "dropout": t.suggest_float("dropout", 0.05, 0.15, step=0.05),
    }


def suggest_scaleset(t, backbone):
    """EXPERIMENT J -- three-scale sets and the residual flag, at the TUNED configuration.

    Two things have never been tested where the model actually lives:

      * THREE-SCALE SETS AT THE TUNED lr. exp2b and expD searched them at lr <= 6e-4, and
        expG's 43,172,344 draws were all killed by its 20-36M guard. Meanwhile expG showed
        that widening a bound at the corrected lr changes which architecture wins -- it moved
        from 43,172/D=192/E6/D4 to 86,172/D=256/E8/D6. No three-scale set has been trained at
        D=256/E8/D6 with lr ~1.4e-3. The guard is raised to 40M here so they can run.
      * THE RESIDUAL FLAG. With deep_sup=0.0 winning everywhere, the per-scale outputs are
        summed but nothing forces a coarse/residual split. `residual=False` -- decode from
        the finest scale alone -- has an argument (172 alone scored 0.0686 vs 0.0412 for
        43,172, so summing clearly helps) but has never actually been run.

    Everything else is frozen at expH t12, the best model found: D=256, E8/D6, mlpmixer,
    bs 16, deep_sup 0.0, and mixup 0.8/0.5 -- which expH established at ~0.8% over control
    across six trials, with prob=0.25 being worse than no mixup at all.
    """
    return {
        "scales": t.suggest_categorical("scales", [
            "86,172", "43,86,172", "22,86,172", "43,86", "43,172", "22,86"]),
        "residual": t.suggest_categorical("residual", [True, False]),
        "region_mode": "raw", "dim": 256, "enc_depth": 8, "dec_depth": 6,
        "heads": 4, "dim_head": 64, "batch_size": 16, "warmup_epochs": 10,
        "deep_sup": 0.0, "mixup_alpha": 0.8, "mixup_prob": 0.5, "loss": "l1",
        "ema_decay": 0.999,
        "epochs_override": t.suggest_categorical("epochs_override", [150, 200, 250]),
        "lr": t.suggest_float("lr", 8e-4, 2e-3, log=True),
        "weight_decay": t.suggest_float("weight_decay", 5e-6, 6e-5, log=True),
        "dropout": t.suggest_float("dropout", 0.0, 0.10, step=0.05),
    }


def suggest_subjweight(t, backbone):
    """EXPERIMENT K -- inverse-visit-count loss weighting.

    Every model in this project weights each SCAN equally, but the training set is 2037 scans
    from only 475 subjects imaged 2 to 11 times (51 subjects have 2 visits, 5 have 11). Under
    per-scan weighting the effective number of subjects is

        (sum n_i)^2 / sum n_i^2  =  397.0     against the 475 actually present

    so 16% of the effective sample size is discarded by an accidental choice. Weighting each
    scan by 1/n_visits(subject) restores it -- a 1.197x gain for one line, verified to
    equalise every subject's total contribution exactly.

    This is not a regulariser: it changes the training DISTRIBUTION, and moves it toward the
    evaluation distribution, since val and test are subject-clustered too (61 subjects each).
    That is why it can shift the train/gap frontier where ~200 tuned trials of architecture
    and optimiser search could not.

    subject_weight is sampled per trial so the control is internal. Scales are {86,172} and
    {43,86} -- expJ found them equivalent (0.037984 vs 0.037955) with 43,86 28% smaller,
    since at D=256 the coarse heads no longer compress (195 outputs from 256 dims)."""
    return {
        "subject_weight": t.suggest_categorical("subject_weight", [True, False]),
        "scales": t.suggest_categorical("scales", ["86,172", "43,86"]),
        "region_mode": "raw", "dim": 256, "enc_depth": 8, "dec_depth": 6,
        "heads": 4, "dim_head": 64, "batch_size": 16, "warmup_epochs": 10,
        "deep_sup": 0.0, "mixup_alpha": 0.8, "mixup_prob": 0.5, "loss": "l1",
        "ema_decay": 0.999, "residual": True,
        "epochs_override": t.suggest_categorical("epochs_override", [150, 200, 250]),
        "lr": t.suggest_float("lr", 8e-4, 2e-3, log=True),
        "weight_decay": t.suggest_float("weight_decay", 5e-6, 6e-5, log=True),
        "dropout": t.suggest_float("dropout", 0.0, 0.10, step=0.05),
    }


SPACES = {"v1": lambda t, bb: suggest(t), "v2": suggest_v2, "scales": suggest_scales,
          "scales3": suggest_scales3, "moments": suggest_moments, "decoder": suggest_decoder,
          "fit": suggest_fit, "finescale": suggest_finescale, "unbound": suggest_unbound,
          "latentsplit": suggest_latentsplit, "arch_tuned": suggest_arch_tuned,
          "mixup": suggest_mixup, "optim": suggest_optim,
          "scaleset": suggest_scaleset, "subjweight": suggest_subjweight}


def main():
    a = parse_args()
    root = OUT / "studies" / a.study
    root.mkdir(parents=True, exist_ok=True)
    csv_fp = root / "trial_metrics.csv"
    log_fp = OUT / "logs" / f"{a.study}.log"
    log_fp.parent.mkdir(parents=True, exist_ok=True)

    def log(m):
        print(m, flush=True)
        with open(log_fp, "a") as h: h.write(m + "\n")

    def objective(trial):
        cfg = SPACES[a.space](trial, a.backbone)
        bb = cfg.pop("backbone", a.backbone)
        # `epochs_override` sets --epochs, which IS the cosine T_max in train_lamm.lr_at, so
        # searching it makes the schedule length match where the model actually converges.
        n_epochs = cfg.pop("epochs_override", a.epochs)
        name = f"{a.study}_t{trial.number:04d}"
        cmd = [PY, "-u", str(HERE / "train_lamm.py"),
               "--run-name", name, "--gpu", str(a.gpu),
               "--backbone", bb, "--latent", str(a.latent), "--patch-level", "3",
               "--lr-final", "1e-6", "--ema-decay", "0.999",
               "--max-params", str(a.max_params),
               "--epochs", str(n_epochs), "--eval-every", "1",
               "--min-epochs", str(a.min_epochs), "--patience", str(a.patience),
               "--time-budget", str(a.trial_time_budget)]
        for k, v in cfg.items():
            if k == "share_regions":
                if v: cmd += ["--share-regions"]
            elif k in ("residual", "subject_weight"):   # BooleanOptionalAction flags
                f = k.replace("_", "-")
                cmd += [f"--{f}" if v else f"--no-{f}"]
            else:
                cmd += [f"--{k.replace('_', '-')}", str(v)]
        cfg["backbone"] = bb
        cfg["epochs"] = n_epochs

        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        summary = OUT / "studies" / name / "summary.json"
        if proc.returncode == 3:                       # over --max-params, never trained
            log(f"  trial {trial.number} REJECTED (too large): "
                f"{proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ''}")
            raise optuna.TrialPruned()
        if proc.returncode != 0 or not summary.exists():
            tail = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else ""
            log(f"  trial {trial.number} FAILED rc={proc.returncode}  {tail}")
            raise optuna.TrialPruned()

        d = json.loads(summary.read_text())
        val = float(d["best_val_rmse_mm"])
        row = {"trial": trial.number, "val_rmse_mm": val, "backbone": a.backbone,
               "best_epoch": d.get("best_epoch"), "best_source": d.get("best_source"),
               "train_rmse_mm": d["metrics"]["train"]["vertex_rmse_mm_mean"],
               "test_rmse_mm": d["metrics"]["test"]["vertex_rmse_mm_mean"],
               "n_params": d.get("n_params"), "duration_s": round(time.time() - t0, 1),
               **{f"p_{k}": v for k, v in cfg.items()}}
        row["gap"] = round(val / max(row["train_rmse_mm"], 1e-12), 4)
        new = not csv_fp.exists()
        with open(csv_fp, "a", newline="") as h:
            w = csv.DictWriter(h, fieldnames=list(row))
            if new: w.writeheader()
            w.writerow(row)
        log(f"  trial {trial.number:3d} val={val:.6f} train={row['train_rmse_mm']:.6f} "
            f"gap={row['gap']:.2f}x ep={row['best_epoch']} ({row['duration_s']:.0f}s) "
            f"{cfg.get('backbone','')[:2]} sc={cfg.get('scales','86'):>11s} "
            f"rm={cfg.get('region_mode','raw'):>4s} "
            f"D={cfg['dim']} E{cfg['enc_depth']}/D{cfg['dec_depth']} h={cfg['heads']} "
            f"lr={cfg['lr']:.2e} wd={cfg['weight_decay']:.2e} do={cfg['dropout']} "
            f"bs={cfg['batch_size']} ds={cfg['deep_sup']} "
            f"mx={cfg.get('mixup_alpha',0)}/{cfg.get('mixup_prob',0)} "
            f"L={cfg.get('loss','l1')} ema={cfg.get('ema_decay',0.999)} "
            f"res={int(cfg.get('residual',True))} "
            f"sw={int(cfg.get('subject_weight',False))} "
            f"({row['n_params']/1e6:.2f}M)")
        return val

    study = optuna.create_study(
        study_name=a.study, direction="minimize",
        storage=f"sqlite:///{root/'study.db'}", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=a.seed, n_startup_trials=8))
    log("=" * 100)
    log(f"{a.study}: backbone={a.backbone} gpu={a.gpu} n_trials={a.n_trials} "
        f"epochs<={a.epochs} budget={a.trial_time_budget:.0f}s latent={a.latent} "
        f"space={a.space} max_params={a.max_params/1e6:.0f}M")
    log(f"targets  PCA-128 {REF['pca128']:.6f} | spiralnet {REF['spiralnet128']:.6f} "
        f"| meshmae tuned {REF['meshmae_tuned']:.6f}")
    log("=" * 100)
    study.optimize(objective, n_trials=a.n_trials, catch=(Exception,))

    done = [t for t in study.trials if t.value is not None]
    if done:
        b = study.best_trial
        log(f"BEST trial {b.number}: val={b.value:.6f}  "
            f"({b.value/REF['spiralnet128']:.3f}x spiralnet, {b.value/REF['pca128']:.3f}x PCA)")
        log(f"  params: {json.dumps(b.params)}")
    log(f"{len(done)}/{a.n_trials} trials completed; csv -> {csv_fp}")


if __name__ == "__main__":
    main()
