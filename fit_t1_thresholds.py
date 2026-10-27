"""Track 1: fit one decision threshold per attribute on the val set (as a logit shift with a fixed 0.5 cut).

The models' val probabilities are averaged with the given weights; for each attribute a shift b_a is
added to the log-odds. The 40 shifts are chosen by coordinate ascent (grid -3..+3, step 0.1, 3 sweeps)
to maximise the official Track 1 score (mean over val datasets of the harmonic mean of mA and
instance F1). Each attribute's grid is scored exactly in one vectorised pass. The shifts are then
shrunk towards their mean: b <- lam * b + (1 - lam) * mean(b).

python fit_t1_thresholds.py --preds runs/eva02L_lpft/preds_ep6.npz runs/clipL_metaclip/preds_ep4.npz \
    --weights 0.4 0.6 --shrink 0.75 --out configs/track1_thresholds.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import common
from t1_metrics import challenge, valid_attrs

GRID = np.round(np.arange(-3.0, 3.01, 0.1), 2)
# near-exclusive groups for the optional "force one answer per group" rule (not used in the final entry)
FORCE_GROUPS = [g["idx"] for g in common.GROUPS if g["name"] in ("age", "ub_color", "lb_color", "lb_type")]


def decide(z, b):
    return (z + b) > 0


class _Dom:
    def __init__(self, z, y):
        self.z, self.y = z, y.astype(bool)
        self.v = valid_attrs(y)
        self.npos = self.y.sum(0).clip(1)
        self.nneg = (~self.y).sum(0).clip(1)
        self.ngt = np.maximum(self.y[:, self.v].sum(1), 1)


def _scores(d, pred, a):
    """Track 1 score of one dataset for every GRID value of attribute a -> [len(GRID)]."""
    if not d.v[a]:
        return None
    p = pred[:, d.v]
    g = d.y[:, d.v]
    col = int(np.where(np.where(d.v)[0] == a)[0][0])
    base_i = (p & g).sum(1) - (p[:, col] & g[:, col])
    base_n = p.sum(1) - p[:, col]
    tp = (p & g).sum(0)
    tn = (~p & ~g).sum(0)
    ba = tp / d.npos[d.v] + tn / d.nneg[d.v]
    ba_rest = (ba.sum() - ba[col]) / 2
    pa = (d.z[:, a][:, None] + GRID[None, :]) > 0              # [N, G]
    ya = d.y[:, a][:, None]
    tp_a = (pa & ya).sum(0)
    tn_a = (~pa & ~ya).sum(0)
    mA = (ba_rest + (tp_a / d.npos[a] + tn_a / d.nneg[a]) / 2) / d.v.sum()
    inter = base_i[:, None] + (pa & ya)
    npred = base_n[:, None] + pa
    prec = (inter / np.maximum(npred, 1)).mean(0)
    rec = (inter / d.ngt[:, None]).mean(0)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return 2 * mA * f1 / np.maximum(mA + f1, 1e-12)


def fit(z, y, dom, sweeps=3):
    doms = [_Dom(z[dom == k], y[dom == k]) for k in np.unique(dom)]
    b = np.zeros(40)
    best = None
    for _ in range(sweeps):
        changed = False
        for a in range(40):
            preds = [decide(d.z, b) for d in doms]
            tot = np.zeros(len(GRID))
            for d, pr in zip(doms, preds):
                s = _scores(d, pr, a)
                if s is None:   # attribute not annotated in this dataset: its score does not depend on it
                    s = np.full(len(GRID), challenge(pr, d.y, np.zeros(len(pr)))["mean"]["avg"])
                tot += s
            tot /= len(doms)
            k = int(np.argmax(tot))
            if GRID[k] != b[a]:
                changed = True
            b[a] = GRID[k]
            best = tot[k]
        if not changed:
            break
    return b, best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True, help="val preds .npz written by train.py, one per model")
    ap.add_argument("--weights", type=float, nargs="+", required=True, help="probability weight per model")
    ap.add_argument("--shrink", type=float, default=0.75, help="lam in b <- lam*b + (1-lam)*mean(b)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert len(args.preds) == len(args.weights)

    names, y, _, _ = common.load_gt("val")
    dom = common.domain_of(names)
    P = [np.load(f)["P"].astype(np.float64) for f in args.preds]
    W = args.weights
    p = (sum(w * q for w, q in zip(W, P)) / sum(W)).clip(1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p))
    b, best = fit(z, y, dom)
    bs = np.round(args.shrink * b + (1 - args.shrink) * b.mean(), 3)
    print(f"val score: 0.5 cut {challenge(decide(z, np.zeros(40)), y, dom)['mean']['avg']:.4f} | "
          f"fitted {best:.4f} | shrunk {challenge(decide(z, bs), y, dom)['mean']['avg']:.4f}")
    json.dump({"logit_shift": [float(v) for v in bs], "force1": False, "force_groups": FORCE_GROUPS,
               "fitted_on": args.preds, "weights": W, "shrink_lam": args.shrink}, open(args.out, "w"), indent=1)
    print("saved", args.out)
