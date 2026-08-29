#!/usr/bin/env python3
"""Train the Compact MeshMAE autoencoder on the ADNI left hippocampus.

Two optional stages, both MeshMAE-derived:
  stage 1 (optional)  masked-patch pretraining -- delete a fraction of patch tokens and
                      reconstruct the full mesh, forcing the encoder to infer missing
                      geometry from context rather than copying its input.
  stage 2             plain autoencoding at mask ratio 0.

Metric is the project-standard per-coordinate vertex RMSE in mm, so numbers are directly
comparable to spiralnet128 (0.036784) / adaptive128 (0.037237) and PCA-128 (0.033668).
"""
from __future__ import annotations

import argparse, copy, json, sys, time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
SPIRAL = Path("/home/jakaria/INR/Deep3DComp/examples/"
              "ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task_spiral_ae_v1/scripts")
for p in (str(HERE), str(SPIRAL)):
    if p not in sys.path: sys.path.insert(0, p)

import spiral_common as sc
import train_eval as te
from meshmae_model import CompactMeshMAE, build_patches, build_patch_members

OUT = Path("/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task_meshmae_ae_v1")
# Default warm start is the checkpoint S1-S4 used, NOT the best one. v7 has since found
# trial_0036 (val 0.036784), but it carries seq_length=17 against trial_0005's 13, so
# switching would change the decoder as well as the variable under test and confound every
# comparison against S2. Pass --warm-path to use it deliberately; geometry follows the
# checkpoint automatically.
WARM = ("/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task_spiral_ae_v1/studies/"
        "spiralnet_z128_v7/trials/trial_0005_val0.036971.pt")
WARM_BEST = ("/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task_spiral_ae_v1/studies/"
             "spiralnet_z128_v7/trials/trial_0036_val0.036784.pt")   # seq_length 17


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", default="meshmae_z128")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--latent", type=int, default=128)
    p.add_argument("--patch-level", type=int, default=2, help="decimation level for centres")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--pretrain-epochs", type=int, default=0, help="0 disables MAE stage")
    p.add_argument("--mask-ratio", type=float, default=0.5)
    # Warm start + freeze are the defaults: the decoder is taken from the best pure
    # SpiralNet++ checkpoint and held fixed, so the Transformer encoder is the ONLY thing
    # learned. That isolates the comparison and cuts trainable parameters by ~73%, which
    # matters on 2037 training meshes. Use --no-warm-start-decoder / --no-freeze-decoder
    # to train the decoder as well.
    p.add_argument("--warm-start-decoder", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--freeze-decoder", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--head", choices=("pool","latentset","latentset_proj","flatten","grouped"),
                   default="pool",
                   help="latent head: mean+max pool (original), cross-attention latent set, "
                        "flatten-and-project (SpiralNet++-style), or grouped "
                        "(weight-shared per-patch projection then concat)")
    p.add_argument("--n-queries", type=int, default=16, help="latentset only")
    p.add_argument("--head-rank", type=int, default=4,
                   help="grouped head: channels kept per patch before concatenation")
    p.add_argument("--time-budget", type=float, default=0.0,
                   help="seconds; stop and write out what has been reached. 0 disables. "
                        "Caps a search trial without discarding its result.")
    p.add_argument("--tokenizer", choices=("moments", "raw", "both"), default="moments",
                   help="patch descriptor: order-invariant moments (original), raw member "
                        "coordinates, or both concatenated")
    p.add_argument("--warm-path", default=WARM,
                   help="decoder warm-start checkpoint; decoder geometry is read from it "
                        "(default: the trial_0005 checkpoint S1-S4 used)")
    p.add_argument("--distance-bias", action="store_true",
                   help="add a learned locality bias from inter-patch distance to attention")
    p.add_argument("--bias-init", type=float, default=1.0,
                   help="initial pre-softplus locality-bias weight. The bias is "
                        "-softplus(w)*distance, so w=-10 gives softplus~4.5e-5: the custom "
                        "attention block with the bias switched off, which is the control "
                        "for --distance-bias (it also swaps nn.TransformerEncoder for "
                        "BiasedEncoderBlock, and that must not be confounded with the bias)")
    p.add_argument("--stem-channels", type=int, default=0,
                   help="spiral-conv stem width before patch tokenization; 0 disables. The "
                        "stem output is concatenated with the raw coordinates, so the "
                        "tokenizer input is a strict superset of the no-stem case.")
    p.add_argument("--stem-layers", type=int, default=2, help="spiral conv layers in the stem")
    p.add_argument("--ema-decay", type=float, default=0.0,
                   help="per-step EMA of weights, 0 disables. Tracked ALONGSIDE the live "
                        "weights and evaluated separately, so it can never make a run worse: "
                        "both curves are logged and the better checkpoint is kept.")
    p.add_argument("--noise-std", type=float, default=0.0,
                   help="Gaussian input noise; 0.04-0.06 helped every winning spiral config")
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--min-epochs", type=int, default=150)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def build(args, device):
    # Decoder geometry must match the warm-start checkpoint exactly, so read it FROM the
    # checkpoint instead of hardcoding it: v7 trials differ in seq_length (13 vs 17) and a
    # mismatch surfaces only as an opaque load_state_dict shape error deep in the decoder.
    ck = None
    ds, oc, seq, dil = [2, 4, 4], [96, 96, 192], 13, 2       # only used without a warm start
    if args.warm_start_decoder:
        ck = torch.load(args.warm_path, map_location="cpu")
        ds, oc = list(ck["ds_factors"]), list(ck["out_channels"])
        seq, dil = int(ck["seq_length"]), int(ck["dilation"])
        if int(ck["latent_channels"]) != int(args.latent):
            raise SystemExit(f"--latent {args.latent} but {args.warm_path} is "
                             f"{ck['latent_channels']}-D: de_linear cannot load")
    tr = sc.get_transform(ds)
    V, _ = sc.build_template()
    centres = tr["vertices"][args.patch_level]
    pid, K = build_patches(V, centres)
    onehot = torch.zeros(V.shape[0], K); onehot[torch.arange(V.shape[0]), pid] = 1.0
    counts = onehot.sum(0).clamp_min(1.0)
    midx, mmask = build_patch_members(pid, K)
    sp, dyn, dn, up = sc.build_spiral_stack(tr, seq, dil, [1] * len(ds), device)
    m = CompactMeshMAE(
        n_patches=K, patch_onehot=onehot.to(device), patch_counts=counts.to(device),
        patch_centres=torch.tensor(np.asarray(centres), dtype=torch.float32, device=device),
        spiral_indices=sp, dynamic_spiral_indices=dyn, down_transform=dn, up_transform=up,
        out_channels=oc, latent=args.latent, d_model=args.d_model, depth=args.depth,
        heads=args.heads, dropout=args.dropout, decoder_dropout=0.1,
        head=args.head, n_queries=args.n_queries,
        distance_bias=args.distance_bias, bias_scale=args.bias_init,
        tokenizer_mode=args.tokenizer, member_idx=midx, member_mask=mmask,
        stem_channels=args.stem_channels, stem_layers=args.stem_layers,
        head_rank=args.head_rank).to(device)
    info = {"head": args.head, "n_queries": args.n_queries, "head_rank": int(args.head_rank),
            "distance_bias": bool(args.distance_bias), "bias_init": float(args.bias_init),
            "ema_decay": float(args.ema_decay),
            "stem_channels": int(args.stem_channels), "stem_layers": int(args.stem_layers),
            "tokenizer": args.tokenizer, "patch_width": int(mmask.shape[1]),
            "noise_std": float(args.noise_std),
            "n_patches": K, "ds_factors": ds, "out_channels": oc, "seq_length": seq,
            "dilation": dil, "patch_level": args.patch_level,
            "verts_per_patch_mean": float(counts.mean()), "verts_per_patch_std": float(counts.std())}
    if ck is not None:
        info["decoder_warm_start"] = args.warm_path
        info["decoder_tensors_loaded"] = m.load_decoder_weights(ck["model_state_dict"])
    if args.freeze_decoder:
        if not args.warm_start_decoder:
            raise SystemExit("--freeze-decoder without --warm-start-decoder would freeze "
                             "randomly initialised weights; pass --no-freeze-decoder")
        m.freeze_decoder(True); info["decoder_frozen"] = True
    return m, info


class EMA:
    """Exponential moving average of the weights, kept in parallel with training.

    S7's last ~100 epochs are flat to five decimal places (0.038900 repeatedly): under the
    cosine-decayed LR the optimiser is oscillating inside a basin rather than converging, and
    `best` is wherever a step happened to land. Averaging along that trajectory returns the
    basin's centre instead.

    Unlike input noise -- which penalises the objective and so trades train error for gap at
    roughly break-even (S5: gap -28%, train +44%, val worse) -- this leaves the objective
    untouched. Only buffers are excluded: every buffer here is a constant precomputed from
    the template, so averaging them would be a no-op.
    """

    def __init__(self, model, decay):
        self.decay = float(decay)
        self.pairs = [(p, p.detach().clone()) for p in model.parameters()]

    @torch.no_grad()
    def update(self):
        d = self.decay
        for p, shadow in self.pairs:
            shadow.mul_(d).add_(p.detach(), alpha=1.0 - d)

    @contextmanager
    def applied(self):
        backup = [p.detach().clone() for p, _ in self.pairs]
        try:
            with torch.no_grad():
                for p, shadow in self.pairs:
                    p.data.copy_(shadow)
            yield
        finally:
            with torch.no_grad():
                for (p, _), b in zip(self.pairs, backup):
                    p.data.copy_(b)


@torch.no_grad()
def val_rmse(model, data, split="val", bs=32):
    model.eval(); x = data.split(split)
    pred = data.denormalize(torch.cat([model(x[i:i+bs]) for i in range(0, len(x), bs)]))
    return float(sc.vertex_rmse_mm(pred, data.denormalize(x)).mean())


def main():
    a = parse_args()
    device = torch.device("cuda", a.gpu); torch.cuda.set_device(device)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    run = OUT / "studies" / a.run_name; run.mkdir(parents=True, exist_ok=True)
    log_fp = OUT / "logs" / f"{a.run_name}.log"; log_fp.parent.mkdir(parents=True, exist_ok=True)
    lg = open(log_fp, "a", buffering=1)
    def log(m): print(m, flush=True); lg.write(m + "\n")

    data = te.load_mesh_tensors(device)
    model, info = build(a, device)
    log("=" * 90)
    log(f"{a.run_name}: {json.dumps(info)}")
    log(f"trainable params {model.num_parameters()/1e6:.3f}M  "
        f"(encoder {sum(p.numel() for p in model.encoder.parameters())/1e6:.3f}M)")
    log("PCA-128 val 0.033668 | spiralnet128 val 0.036784 | adaptive128 val 0.037237"
        " | S2_flatten val 0.042143 (train 0.017323)")
    log("=" * 90)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    epochs = 4 if a.smoke else a.epochs
    pre = 2 if a.smoke else a.pretrain_epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    xt = data.train; n = len(xt)
    ema = EMA(model, a.ema_decay) if a.ema_decay > 0 else None
    best, best_state, best_ep, stale, hist = float("inf"), None, -1, 0, []
    best_source = "live"
    t0 = time.time()

    for ep in range(1, epochs + pre + 1):
        pretraining = ep <= pre
        model.train()
        perm = torch.randperm(n, generator=g).to(device)
        tot = cnt = 0
        for i in range(0, n, a.batch_size):
            b = xt[perm[i:i+a.batch_size]]
            mask = None
            if pretraining and a.mask_ratio > 0:
                mask = torch.rand(b.size(0), model.n_patches, device=device) < a.mask_ratio
            inp = b if a.noise_std <= 0 else b + a.noise_std * torch.randn_like(b)
            opt.zero_grad(set_to_none=True)
            F.l1_loss(model(inp, mask), b).backward()   # target is always the clean mesh
            opt.step()
            if ema is not None: ema.update()
            tot += 1; cnt += 1
        if not pretraining: sched.step()
        if ep % a.eval_every == 0 or ep == epochs + pre:
            cur = val_rmse(model, data)
            cur_ema = None
            if ema is not None:
                with ema.applied():
                    cur_ema = val_rmse(model, data)
            tag = "pretrain" if pretraining else "train"
            if not pretraining:
                improved = False
                for score, src in ((cur, "live"), (cur_ema, "ema")):
                    if score is not None and score < best - 1e-9:
                        best, best_ep, best_source, improved = score, ep, src, True
                if improved:
                    stale = 0
                    with (ema.applied() if best_source == "ema" else nullcontext()):
                        best_state = {k: v.detach().cpu().clone()
                                      for k, v in model.state_dict().items()}
                else:
                    stale += 1
            hist.append({"epoch": ep, "stage": tag, "val_rmse_mm": cur,
                         "val_rmse_mm_ema": cur_ema, "best": best, "best_source": best_source})
            emastr = "" if cur_ema is None else f" ema={cur_ema:.6f}"
            log(f"  [{tag}] epoch {ep:4d} val_rmse_mm={cur:.6f}{emastr} "
                f"best={best:.6f}({best_source}) ({time.time()-t0:.0f}s)")
            if (not pretraining and ep >= a.min_epochs and a.patience and stale >= a.patience):
                log(f"  early stop at epoch {ep}"); break
            if a.time_budget > 0 and (time.time() - t0) > a.time_budget:
                log(f"  stopped at epoch {ep} on the {a.time_budget:.0f}s budget "
                    f"(best={best:.6f})"); break

    if best_state is not None: model.load_state_dict(best_state)

    @torch.no_grad()                       # without this the 2037 train meshes accumulate
    def split_metrics(sp, bs=32):          # autograd graphs and OOM the card
        model.eval(); x = data.split(sp)
        pred = torch.cat([model(x[i:i+bs]).detach() for i in range(0, len(x), bs)])
        return sc.reconstruction_metrics(data.denormalize(pred),
                                         data.denormalize(x), faces=data.faces)
    metrics = {sp: split_metrics(sp) for sp in sc.SPLITS}
    torch.save({"run": a.run_name, "args": vars(a), "info": info,
                "best_val_rmse_mm": best, "best_epoch": best_ep, "best_source": best_source,
                "metrics": metrics, "model_state_dict": best_state}, run / "best.pt")
    sc.atomic_write_json(run / "summary.json",
        {"run": a.run_name, "args": vars(a), "info": info, "best_val_rmse_mm": best,
         "best_epoch": best_ep, "best_source": best_source,
         "metrics": metrics, "history": hist,
         "n_params": model.num_parameters(),
         "reference": {"pca128_val": 0.033668, "spiralnet128_val": 0.036784,
                       "adaptive128_val": 0.037237, "s2_flatten_val": 0.042143}})
    for sp in sc.SPLITS:
        log(f"  {sp:5s} rmse_mm={metrics[sp]['vertex_rmse_mm_mean']:.6f}")
    log(f"wrote {run}")
    lg.close()


if __name__ == "__main__":
    main()
