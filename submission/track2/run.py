"""UPAR 2027 Track 2 submission: attribute model(s) -> calibrated probabilities -> scoring.

Everything needed ships in this folder (vendored timm, weights, config.json). No network.
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np
import torch
from PIL import Image

CACHE_HW = (384, 192)          # must match common.CACHE_HW used in training
T_START = time.time()
BUDGET_S = float(os.environ.get("UPAR_BUDGET", 265.0))   # hard limit is 300 s for the whole run
GOV_WAIT_S = 40.0   # no governor decision in the first 40 s of inference: the A10 ramps up for ~30 s
_MODELS: list = []
_CFG: dict = {}
_DEV = "cuda" if torch.cuda.is_available() else "cpu"
_SIM_SLOW = os.environ.get("UPAR_SIM_SLOW", "")   # local governor test: "start" or "all"


def log(msg):
    print(f"[run.py {time.time() - T_START:6.1f}s] {msg}", flush=True)


def load_image(path):
    with Image.open(path) as im:
        im = im.convert("RGB").resize((CACHE_HW[1], CACHE_HW[0]), Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8)


def load_model() -> None:
    global _CFG
    if _MODELS:
        return
    from model import AttrNet
    with open(os.path.join(HERE, "config.json")) as fh:
        _CFG = json.load(fh)
    for m in _CFG["models"]:
        net = AttrNet(m["backbone"], tuple(m["hw"]), m["head"], pretrained=False)
        sd = torch.load(os.path.join(HERE, m["weights"]), map_location="cpu", weights_only=True)
        net.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True)
        net = net.to(_DEV).eval()
        if _DEV == "cuda":
            net = net.half()
            net.head.float()   # fp32 logits: fp16 rounding creates large tie groups
        _MODELS.append((net, m))
    log(f"loaded {len(_MODELS)} model(s) on {_DEV}")
    _warmup()


@torch.no_grad()
def _warmup():
    """One dummy batch per model, so CUDA init and kernel selection are not measured as slow inference."""
    if _DEV != "cuda":
        return
    from model import resize_for_model
    x = torch.zeros(int(_CFG.get("batch", 256)), *CACHE_HW, 3, dtype=torch.uint8, device=_DEV)
    for net, m in _MODELS:
        net(resize_for_model(x, tuple(m["hw"])).half())
    torch.cuda.synchronize()
    log("warm-up done")


def _batches(paths, bs, threads, ahead=8):
    """Decode + resize with a thread pool (PIL releases the GIL); yields uint8 [B,H,W,3].
    At most `ahead` batches are in flight: submitting the whole gallery at once made the decode
    threads race ahead (GBs of arrays, GIL contention) and starved the GPU-feeding main thread
    for the first ~50 s of inference."""
    from collections import deque
    starts = iter(range(0, len(paths), bs))
    with ThreadPoolExecutor(threads) as ex:
        futs = deque()

        def submit():
            s = next(starts, None)
            if s is not None:
                futs.append(ex.submit(lambda ps: np.stack([load_image(p) for p in ps]), paths[s:s + bs]))
        for _ in range(ahead):
            submit()
        while futs:
            xb = futs.popleft().result()
            submit()
            yield xb


@torch.no_grad()
def predict(paths):
    from model import resize_for_model
    bs = int(_CFG.get("batch", 256))
    flip = bool(_CFG.get("tta_flip", True))
    n = len(paths)
    # Per-model accumulators. If the governor drops models, only the models that saw every
    # image are used, so the whole gallery is scored by the same ensemble.
    P, GP, FE = [], [], []
    t0, done = time.time(), 0
    hist = []
    n_active = len(_MODELS)
    dtype = torch.float16 if _DEV == "cuda" else torch.float32
    for xb in _batches(paths, bs, int(_CFG.get("threads", 16))):
        x = torch.from_numpy(xb).to(_DEV, non_blocking=True)
        b = x.shape[0]
        for j, (net, m) in enumerate(_MODELS[:n_active]):
            xi = resize_for_model(x, tuple(m["hw"])).to(dtype)
            views = [xi, xi.flip(3)] if flip else [xi]
            p, gp, f = 0, 0, 0
            for v in views:
                lg, fv = net(v)
                pv, gpv = net.probs(lg.float())
                p, gp, f = p + pv, gp + gpv, f + fv.float()
            if j == len(P):
                P.append(torch.zeros(n, 40, device=_DEV))
                GP.append(torch.zeros(n, gp.shape[1], device=_DEV))
                FE.append(torch.zeros(n, f.shape[1], device=_DEV, dtype=torch.float16))
            P[j][done:done + b] = p / len(views)
            GP[j][done:done + b] = gp / len(views)
            FE[j][done:done + b] = torch.nn.functional.normalize(f, dim=1).half()
        done += b
        el = time.time() - t0
        if done == b or done % (20 * bs) < b or done == n:
            log(f"inference {done}/{n} imgs, {done / el:.0f} img/s, eta {(n - done) / max(done / el, 1e-6):.0f}s")
        # governor: if the projected finish overruns BUDGET_S, first drop flip TTA, then the
        # last model, then the next-to-last, ... (one per decision). Nothing is decided in the
        # first GOV_WAIT_S of inference (slow A10 cold start).
        hist.append((time.time(), done))
        if (flip or n_active > 1) and len(hist) > 8 and el > GOV_WAIT_S:
            (ta, na), (tb, nb) = hist[-9], hist[-1]
            # recent window OR cumulative rate, whichever is better: a genuinely slow machine is
            # slow on both; a transient I/O dip (cold disk) only drags the recent window down.
            rate = max((nb - na) / max(tb - ta, 1e-6), done / el)
            if (time.time() - T_START) + (n - done) / rate > BUDGET_S:
                if flip:
                    flip = False
                    log(f"governor: disabling flip TTA (recent {rate:.0f} img/s) to stay within the time budget")
                else:   # drop the LAST active model only (config order = importance)
                    n_active -= 1
                    log(f"governor: dropping model {n_active + 1} (recent {rate:.0f} img/s) to stay within the time budget")
                hist.clear()
        if _SIM_SLOW and (_SIM_SLOW == "all" or done <= 4 * bs):   # local testing only
            time.sleep(3.0)
    k = n_active
    if k < len(_MODELS):
        log(f"using model(s) 1..{k} only (dropped models did not see the whole gallery)")
    global _FE_DIMS
    _FE_DIMS = [f.shape[1] for f in FE[:k]]
    return sum(P[:k]) / k, sum(GP[:k]) / k, torch.cat(FE[:k], 1)


def score(queries, P, GP):
    import scoring
    sc = _CFG.get("scoring", {"type": "gll", "eps": 1e-3})
    t = sc.get("type", "gll")
    eps = float(sc.get("eps", 1e-3))
    if t == "l1":
        S = scoring.l1(queries, P)
    elif t == "ll":
        S = scoring.bernoulli_ll(queries, P, eps)
    elif t == "gll":
        S = scoring.grouped_ll(queries, GP, eps)
    elif t == "blend":
        lam = float(sc.get("lam", 0.5))
        S = lam * scoring.grouped_ll(queries, GP, eps) + (1 - lam) * float(sc.get("c", 1.0)) * scoring.agreement(queries, P)
    else:
        raise ValueError(t)
    return S


def rank_gallery(sample: dict) -> dict:
    load_model()
    gallery = sample["gallery"]
    paths = [g["image_path"] for g in gallery]
    queries = torch.as_tensor(np.asarray(sample["queries"], dtype=np.float32), device=_DEV)
    log(f"Q={queries.shape[0]} G={len(paths)}")
    P, GP, FE = predict(paths)
    try:
        kn = _CFG.get("knn")
        J = W = None
        if kn:
            import scoring
            wh = kn.get("whiten")
            if wh:  # gallery PCA-whitening of the graph features, per model block
                FE = scoring.whiten_blocks(FE, _FE_DIMS, float(wh.get("frac", 1.0)), float(wh.get("shrink", 0.03)))
            # one sparse graph serves probability diffusion and score-level smoothing
            J, W = scoring.knn_graph(FE, k=int(kn["k"]), tau=float(kn.get("tau", 0.05)),
                                     reciprocal=int(kn.get("recip", 0)))
            alpha = float(kn["beta"])
            if "gate_floor" in kn:  # per-image strength from the image's density rank in this gallery
                alpha = scoring.density_gate(scoring.neighbour_density(FE, int(kn.get("gate_m", 5))),
                                             alpha, float(kn.get("gate_gamma", 1.0)),
                                             float(kn["gate_floor"]))
            P = scoring.diffuse(P, J, W, alpha, int(kn.get("iters", 1)))
            GP = scoring.diffuse(GP, J, W, alpha, int(kn.get("iters", 1)))
            cl = kn.get("clus")
            if cl:  # mutual-kNN cluster consensus on the (whitened) graph, size-capped
                lab = scoring.mutual_clusters(FE, J, int(cl.get("k", 10)), float(cl["s0"]))
                P = scoring.cluster_consensus(P, lab, float(cl["beta"]), int(cl.get("cap", 64)))
                GP = scoring.cluster_consensus(GP, lab, float(cl["beta"]), int(cl.get("cap", 64)))
                log(f"clusters: {int(lab.max()) + 1} for {len(lab)} images")
            log(f"knn diffusion {kn}")
        S = score(queries, P, GP)
        lam = float(_CFG.get("scoring", {}).get("joint", 0))
        if lam > 0:  # closed-set query codebook: S - lam * logsumexp over all given queries (per image)
            S = S - lam * torch.logsumexp(S.float(), dim=0, keepdim=True)
            log(f"joint query normalisation lam={lam}")
        if kn and float(kn.get("score_beta", 0)) > 0:
            S = scoring.smooth_scores(S, J, W, float(kn["score_beta"]))
            log(f"score smoothing beta={kn['score_beta']}")
        if not torch.isfinite(S).all():
            raise FloatingPointError("non-finite scores")
    except Exception as e:  # never fail the submission: fall back to plain L1
        log(f"scoring failed ({e!r}); falling back to L1")
        import scoring
        S = scoring.l1(queries, P)
    S = S.float().cpu().numpy()
    log(f"done, scores {S.shape}")
    return {"similarities": S}
