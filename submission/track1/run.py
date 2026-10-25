"""UPAR 2027 Track 1 submission: attribute model ensemble -> 40 probabilities per image.

Same models as our Track 2 submission, per image only (no cross-image propagation). The 0.5
decision is calibrated with one logit shift per attribute fitted on the public val split
(config.json "calib"), identical for every test image. Optional "force1": in near-exclusive
groups (age, upper/lower colour, lower type) the group's top attribute is switched on when no
member passes 0.5. Everything ships in this folder (vendored timm, weights). No network.
Each predict_batch call is split into CHUNK-sized sub-batches whose decode runs on a thread pool
while the GPU works on the previous one (decode then infer in series left the GPU idle).
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
BATCH_SIZE = 512               # read by the ingestion program for predict_batch
CHUNK = 128                    # GPU sub-batch: decode of chunk k+1 overlaps inference of chunk k
T_START = time.time()
BUDGET_S = float(os.environ.get("UPAR_BUDGET", 255.0))   # hard limit is 300 s for the whole run
_MODELS: list = []
_CFG: dict = {}
_DEV = "cuda" if torch.cuda.is_available() else "cpu"
_POOL = ThreadPoolExecutor(16)
_STATE = {"n": 0, "single": False, "t_first": None}


def log(msg):
    print(f"[run.py {time.time() - T_START:6.1f}s] {msg}", flush=True)


def _to_array(sample):
    path = sample.get("image_path")
    if path:
        with Image.open(path) as im:
            im = im.convert("RGB").resize((CACHE_HW[1], CACHE_HW[0]), Image.BILINEAR)
            return np.asarray(im, dtype=np.uint8)
    im = sample["image"]
    if not isinstance(im, Image.Image):
        im = Image.fromarray(np.asarray(im))
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
            net.head.float()
        _MODELS.append((net, m))
    c = _CFG.get("calib", {})
    _CFG["_shift"] = torch.tensor(c.get("logit_shift", [0.0] * 40), dtype=torch.float32, device=_DEV)
    _CFG["_groups"] = c.get("force_groups", []) if c.get("force1") else []
    log(f"loaded {len(_MODELS)} model(s) on {_DEV}; calib force1={bool(_CFG['_groups'])}")
    _warmup()


@torch.no_grad()
def _warmup():
    if _DEV != "cuda":
        return
    from model import resize_for_model
    x = torch.zeros(CHUNK, *CACHE_HW, 3, dtype=torch.uint8, device=_DEV)
    for net, m in _MODELS:
        net(resize_for_model(x, tuple(m["hw"])).half())
    torch.cuda.synchronize()
    log("warm-up done")


@torch.no_grad()
def _probs(chunks):
    """iterable of uint8 [b,H,W,3] -> calibrated probabilities [sum b,40] (float32, cpu).
    GPU work is queued per chunk without a sync, so the next chunk decodes while the GPU runs;
    one sync at the end of the batch."""
    from model import resize_for_model
    dtype = torch.float16 if _DEV == "cuda" else torch.float32
    # governor: past the budget, keep only model 0 (each image is scored on its own, so a
    # late switch only changes the remaining images)
    if not _STATE["single"] and len(_MODELS) > 1 and time.time() - T_START > BUDGET_S:
        _STATE["single"] = True
        log(f"governor: model 1 only after {_STATE['n']} images to stay within the time budget")
    models = _MODELS[:1] if _STATE["single"] else _MODELS
    zs = []
    for xb in chunks:
        x = torch.from_numpy(xb).to(_DEV, non_blocking=True)
        p, wsum = 0, 0.0
        for net, m in models:   # optional per-model "weight" in config.json (default: equal)
            lg, _ = net(resize_for_model(x, tuple(m["hw"])).to(dtype))
            w = float(m.get("weight", 1.0))
            p, wsum = p + w * torch.sigmoid(lg.float()), wsum + w
        p = (p / wsum).clamp(1e-6, 1 - 1e-6)
        zs.append(torch.log(p / (1 - p)) + _CFG["_shift"])
    z = torch.cat(zs)
    out = torch.sigmoid(z)
    for idx in _CFG["_groups"]:
        g = z[:, idx]
        none = (g <= 0).all(1)
        if none.any():
            rows = torch.where(none)[0]
            cols = torch.as_tensor(idx, device=_DEV)[g[none].argmax(1)]
            out[rows, cols] = 0.51
    out = torch.nan_to_num(out, nan=0.0).clamp(0.0, 1.0)
    return out.cpu().numpy()


def predict_batch(samples):
    load_model()
    if _STATE["t_first"] is None:
        _STATE["t_first"] = time.time()
        im = samples[0].get("image") if samples else None
        log(f"first batch: {len(samples)} samples, sample['image'] is {type(im).__name__}")
    # one decode future per image (all pool threads busy); a batch is bounded, so memory is too
    futs = [_POOL.submit(_to_array, x) for x in samples]

    def chunks():
        for s in range(0, len(futs), CHUNK):
            yield np.stack([f.result() for f in futs[s:s + CHUNK]])
    try:
        P = _probs(chunks())
    except Exception as e:   # e.g. out of memory with two models: continue with model 1 only
        log(f"inference error {e!r}; retrying with model 1 only")
        _STATE["single"] = True
        P = _probs(chunks())
    names = samples[0].get("attribute_names") if samples else None
    order = _CFG.get("attribute_names")
    if names and order and list(names) != list(order):
        pos = {n: i for i, n in enumerate(order)}
        P = P[:, [pos[n] for n in names]]
    _STATE["n"] += len(samples)
    if _STATE["n"] % (10 * BATCH_SIZE) < len(samples):
        el = time.time() - _STATE["t_first"]
        log(f"{_STATE['n']} images, {_STATE['n'] / max(el, 1e-6):.0f} img/s since first batch")
    return [row.tolist() for row in P]


def predict_image(sample):
    return predict_batch([sample])[0]
