"""Train one attribute model on the cached images; evaluate on val after every epoch.

Saves to runs/<name>/: args.json, log.jsonl, best.pt / last.pt (EMA weights, fp16),
preds_{best,last}.npz (val probabilities, group probabilities, features; flip-averaged).
"best" = highest val retrieval score (mean mADM over the three val datasets).

Example (the four final models are in scripts/train_final.sh):
python train.py --name clipL_metaclip --backbone vit_large_patch14_clip_224.metaclip_2pt5b \
    --hw 252 126 --epochs 6 --bs 96 --lr 5e-5 --ld 0.8 --photo 1 --grad-ckpt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F
from timm.optim import param_groups_layer_decay
from timm.utils import ModelEmaV3

import scoring
from common import CACHE, CACHE_HW, domain_of, load_gt
from metrics import evaluate, subset_protocol
from model import GROUP_SIZES, AttrNet, resize_for_model
from scoring import BINARY, GROUPS

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--backbone", default="vit_base_patch16_clip_224.openai", help="timm model name")
p.add_argument("--hw", type=int, nargs=2, default=[256, 128], help="model input height width")
p.add_argument("--head", default="sigmoid", choices=["sigmoid", "grouped"],
               help="sigmoid = 40 independent outputs (used in all final models); grouped = softmax per group")
p.add_argument("--epochs", type=int, default=15)
p.add_argument("--bs", type=int, default=128)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--wd", type=float, default=0.05)
p.add_argument("--ld", type=float, default=0.75, help="layer-wise lr decay (1 = off)")
p.add_argument("--drop", type=float, default=0.1)
p.add_argument("--ls", type=float, default=0.05, help="label smoothing")
p.add_argument("--ema", type=float, default=0.9995)
p.add_argument("--warmup", type=float, default=1.0, help="warmup epochs")
p.add_argument("--color", type=float, default=0.2, help="brightness/contrast jitter strength")
p.add_argument("--photo", type=float, default=0.0,
               help="colour and lighting aug strength (white balance, gamma, saturation; 1 = default ranges)")
p.add_argument("--lp-epochs", type=int, default=0,
               help="LP-FT: first N of --epochs train the head only on frozen features (Kumar et al. 2022); "
                    "warmup + cosine then cover the remaining epochs")
p.add_argument("--lp-lr", type=float, default=1e-3, help="constant head lr during the LP phase")
p.add_argument("--domains", default="",
               help="comma list of training datasets for the leave-one-dataset-out check; empty = all")
p.add_argument("--grad-ckpt", action="store_true",
               help="activation checkpointing in the backbone: same math, less memory, slower")
p.add_argument("--eval-every", type=int, default=1)
p.add_argument("--save-epochs", action="store_true",
               help="also save ep<N>.pt (EMA) and preds_ep<N>.npz every epoch")
p.add_argument("--max-steps", type=int, default=0, help="debug: stop after N steps")
p.add_argument("--seed", type=int, default=0,
               help="data-order seed; != 0 also seeds torch (head init, dropout)")
args = p.parse_args()
if args.seed:
    torch.manual_seed(args.seed)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
dev = "cuda"
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", args.name)
os.makedirs(out_dir, exist_ok=True)
json.dump(vars(args), open(os.path.join(out_dir, "args.json"), "w"), indent=1)
H, W = args.hw


def log(d):
    print(json.dumps(d), flush=True)
    with open(os.path.join(out_dir, "log.jsonl"), "a") as fh:
        fh.write(json.dumps(d) + "\n")


# ---------------------------------------------------------------- data
def load_split(split):
    """Memory-mapped; callers copy only the rows they need."""
    names, y, qid, q = load_gt(split)
    x = np.load(os.path.join(CACHE, f"{split}_{CACHE_HW[0]}x{CACHE_HW[1]}.npy"), mmap_mode="r")
    return names, x, y, qid, q


class Rows:
    """Row subset of a memmap without copying it (leave-one-dataset-out runs)."""
    def __init__(self, x, rows):
        self.x, self.rows = x, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.x[self.rows[i]]


def take(x, mask):
    return x if mask.all() else Rows(x, np.where(mask)[0])


t0 = time.time()
tr_names, tr_x, tr_y, _, _ = load_split("train")
va_names, va_x, va_y, va_qid, va_q = load_split("val")
doms = args.domains.split(",") if args.domains else None
k_tr = np.isin(domain_of(tr_names), doms) if doms else np.ones(len(tr_names), bool)
tr_x, tr_y = take(tr_x, k_tr), tr_y[k_tr]
print(f"data loaded {time.time() - t0:.0f}s  train={len(tr_y)}", flush=True)

# Val is always evaluated on all three datasets (the held-out one included).
va_dom = domain_of(va_names)
protocols = {"all": (va_q, va_y, va_qid, np.ones(len(va_y), bool))}
for d in np.unique(va_dom):
    m = va_dom == d
    q, y, qid = subset_protocol(va_y, m)
    protocols[d] = (q, y, qid, m)

steps_per_epoch = len(tr_y) // args.bs
total_steps = steps_per_epoch * args.epochs
lp_steps = steps_per_epoch * args.lp_epochs


def batches():
    """Background thread: random batches of (uint8 images, labels) in pinned memory."""
    qu = queue.Queue(maxsize=6)

    def run():
        rng = np.random.default_rng(args.seed)
        for _ in range(args.epochs):
            perm = rng.permutation(len(tr_y))
            for i in range(steps_per_epoch):
                idx = np.sort(perm[i * args.bs:(i + 1) * args.bs])
                qu.put((torch.from_numpy(tr_x[idx]).pin_memory(),
                        torch.from_numpy(tr_y[idx]).pin_memory()))
        qu.put(None)

    threading.Thread(target=run, daemon=True).start()
    while (b := qu.get()) is not None:
        yield b


def augment(xb):
    """GPU aug: antialiased resize, random flip, random zoom/translate crop, colour jitter.
    No hue jitter and no random erasing (both hurt colour/region attributes)."""
    x = resize_for_model(xb, (H, W))
    B = x.shape[0]
    flip = (torch.rand(B, device=dev) < 0.5).float() * 2 - 1
    s = 1 / (1 + 0.2 * torch.rand(B, device=dev))                   # zoom in up to 1.2x
    sx = s * (1 + 0.05 * (torch.rand(B, device=dev) * 2 - 1))       # mild aspect jitter
    tx = (1 - sx) * (torch.rand(B, device=dev) * 2 - 1)
    ty = (1 - s) * (torch.rand(B, device=dev) * 2 - 1)
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0] = sx * flip
    theta[:, 0, 2] = tx
    theta[:, 1, 1] = s
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    if args.photo > 0:
        # Colour and lighting aug (label-preserving, no hue rotation): white balance = per-channel
        # gain exp(U(-.1,.1)*s) renormalised to mean 1, gamma exp(U(-.3,.3)*s), saturation 1+U(-.3,.3)*s.
        s = args.photo
        u = lambda *sh: torch.rand(*sh, device=dev) * 2 - 1
        gain = torch.exp(0.1 * s * u(B, 3, 1, 1))
        x = (x * gain / gain.mean(1, keepdim=True)).clamp_(0, 1)
        x = x.clamp_min(1e-4).pow(torch.exp(0.3 * s * u(B, 1, 1, 1)))
        gray = (0.299 * x[:, :1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:]).expand_as(x)
        x = (gray + (1 + 0.3 * s * u(B, 1, 1, 1)) * (x - gray)).clamp_(0, 1)
    c = args.color
    if c > 0:
        br = 1 + c * (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1)
        ct = 1 + c * (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1)
        m = x.mean((1, 2, 3), keepdim=True)
        x = ((x - m) * ct + m) * br
        x = x.clamp_(0, 1)
    return x


# ---------------------------------------------------------------- model / loss
model = AttrNet(args.backbone, (H, W), args.head, args.drop).to(dev)
if args.grad_ckpt:
    model.backbone.set_grad_checkpointing(True)
if "convnext" in args.backbone:
    model = model.to(memory_format=torch.channels_last)
groups = param_groups_layer_decay(model.backbone, weight_decay=args.wd, layer_decay=args.ld) \
    if args.ld < 1 else [{"params": list(model.backbone.parameters()), "lr_scale": 1.0}]
groups.append({"params": list(model.head.parameters()), "lr_scale": 1.0, "weight_decay": args.wd})
opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.999))
ema = ModelEmaV3(model, decay=args.ema)

pos_rate = torch.tensor(tr_y.mean(0), device=dev, dtype=torch.float32).clamp(1e-4, 1 - 1e-4)
w_pos, w_neg = torch.exp(1 - pos_rate), torch.exp(pos_rate)        # DeepMAR weights (Li et al. 2015)


def loss_fn(logits, y):
    y = y.float()
    a = args.ls
    if args.head == "sigmoid":
        t = (1 - a) * y + a * (1 - y)
        w = y * w_pos + (1 - y) * w_neg
        return F.binary_cross_entropy_with_logits(logits.float(), t, weight=w)
    logits = logits.float()
    loss, o = 0.0, 0
    for (idx, none), n in zip(GROUPS, GROUP_SIZES):
        g = y[:, idx]
        if none:
            g = torch.cat([g, (g.sum(1, keepdim=True) == 0).float()], 1)
        valid = g.sum(1) > 0
        t = g / g.sum(1, keepdim=True).clamp_min(1)
        t = (1 - a) * t + a / n
        ce = -(t * F.log_softmax(logits[:, o:o + n], 1)).sum(1)
        loss = loss + (ce * valid).sum() / valid.sum().clamp_min(1)
        o += n
    b = y[:, BINARY]
    tb = (1 - a) * b + a * (1 - b)
    wb = b * w_pos[BINARY] + (1 - b) * w_neg[BINARY]
    loss = loss + F.binary_cross_entropy_with_logits(logits[:, o:], tb, weight=wb, reduction="none").sum(1).mean()
    return loss / (len(GROUPS) + len(BINARY))


def lr_at(step):
    step -= lp_steps                                        # LP-FT: schedule restarts after the LP phase
    warm = args.warmup * steps_per_epoch
    if step < warm:
        return args.lr * step / warm
    return args.lr * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total_steps - lp_steps - warm)))


# ---------------------------------------------------------------- eval
@torch.no_grad()
def predict(net, x_all, bs=512):
    """Flip-averaged val probabilities, group probabilities and pooled features."""
    net.eval()
    P, GP, FE = [], [], []
    for s in range(0, len(x_all), bs):
        xb = torch.from_numpy(x_all[s:s + bs]).to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg, f = net(resize_for_model(xb, (H, W)))
            lg2, f2 = net(resize_for_model(xb, (H, W)).flip(3))
        p1, g1 = net.probs(lg)
        p2, g2 = net.probs(lg2)
        P.append(((p1 + p2) / 2).cpu())
        GP.append(((g1 + g2) / 2).cpu())
        FE.append(((f + f2) / 2).float().cpu())
    return torch.cat(P).numpy(), torch.cat(GP).numpy(), torch.cat(FE).half().numpy()


def eval_metrics(P, GP):
    res = {}
    Pt, GPt = torch.from_numpy(P).to(dev), torch.from_numpy(GP).to(dev)
    for pname, (q, y, qid, m) in protocols.items():
        qt = torch.from_numpy(q).float().to(dev)
        idx = torch.from_numpy(np.where(m)[0]).to(dev)
        for sname, fn in [("l1", lambda: scoring.l1(qt, Pt[idx])),
                          ("ll", lambda: scoring.bernoulli_ll(qt, Pt[idx], 1e-3)),
                          ("gll", lambda: scoring.grouped_ll(qt, GPt[idx], 1e-3))]:
            r = evaluate(fn(), q, y, qid)
            res[f"{pname}/{sname}/mADM"] = round(r["mADM"], 4)
            res[f"{pname}/{sname}/mAP"] = round(r["mAP"], 4)
    for sname in ["l1", "ll", "gll"]:
        res[f"dom/{sname}/mADM"] = round(float(np.mean(
            [res[f"{d}/{sname}/mADM"] for d in protocols if d != "all"])), 4)
    # recognition diagnostics at the plain 0.5 cut (all val images pooled)
    pred = P > 0.5
    yv = va_y.astype(bool)
    tp = (pred & yv).sum(1)
    f1 = (2 * tp / (pred.sum(1) + yv.sum(1)).clip(1)).mean()
    tpr = (pred & yv).sum(0) / yv.sum(0).clip(1)
    tnr = (~pred & ~yv).sum(0) / (~yv).sum(0).clip(1)
    res["inst_f1"] = round(float(f1), 4)
    res["mA"] = round(float(((tpr + tnr) / 2).mean()), 4)
    return res


# ---------------------------------------------------------------- train
step, best, t_ep = 0, -1.0, time.time()
model.train()
loss_avg = 0.0
for xb, yb in batches():
    xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
    lp = step < lp_steps
    if step == lp_steps and lp_steps:                       # end of LP: EMA restarts from the probed head
        ema.set(model)
    lr = args.lp_lr if lp else lr_at(step)
    for g in opt.param_groups:
        g["lr"] = (lr if g["params"][0] is model.head.weight else 0.0) if lp else lr * g.get("lr_scale", 1.0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if lp:                                              # frozen backbone: no graph, head-only grads
            with torch.no_grad():
                feats = model.backbone((augment(xb) - model.mean) / model.std)
            logits = model.head(model.drop(feats).to(model.head.weight.dtype))
        else:
            logits, feats = model(augment(xb))
    loss = loss_fn(logits, yb)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt.step()
    ema.update(model)
    loss_avg = 0.98 * loss_avg + 0.02 * loss.item() if step else loss.item()
    step += 1
    if step % 100 == 0:
        print(f"step {step}/{total_steps} loss {loss_avg:.4f} lr {lr:.2e} "
              f"{(time.time() - t_ep) / ((step - 1) % steps_per_epoch + 1):.3f}s/it", flush=True)
    if args.max_steps and step >= args.max_steps:
        break
    if step % steps_per_epoch == 0:
        ep = step // steps_per_epoch
        rec = {"epoch": ep, "loss": round(loss_avg, 4), "train_min": round((time.time() - t_ep) / 60, 2)}
        sd = {k: v.half() if v.is_floating_point() else v for k, v in ema.module.state_dict().items()}
        ckpt = {"args": vars(args), "state_dict": sd, "epoch": ep}
        if args.save_epochs:
            torch.save(ckpt, os.path.join(out_dir, f"ep{ep}.pt"))
        if ep % args.eval_every == 0 or ep == args.epochs:
            t1 = time.time()
            P, GP, FE = predict(ema.module, va_x)
            rec.update(eval_metrics(P, GP))
            rec["eval_s"] = round(time.time() - t1, 1)
            key = max(rec["dom/ll/mADM"], rec["dom/gll/mADM"])
            if key > best:
                best = key
                torch.save(ckpt, os.path.join(out_dir, "best.pt"))
                np.savez(os.path.join(out_dir, "preds_best.npz"), P=P, GP=GP, FE=FE, epoch=ep)
            if ep == args.epochs:
                np.savez(os.path.join(out_dir, "preds_last.npz"), P=P, GP=GP, FE=FE, epoch=ep)
            if args.save_epochs:
                np.savez(os.path.join(out_dir, f"preds_ep{ep}.npz"), P=P, GP=GP, FE=FE, epoch=ep)
        torch.save(ckpt, os.path.join(out_dir, "last.pt"))
        log(rec)
        model.train()
        t_ep = time.time()
print("done", out_dir)
