"""Track 1 metric (WACV 2024 script as described on the Codabench page): per dataset, prob > 0.5,
mA = mean over attrs of (TPR+TNR)/2, instance F1 = mean per-image 2|P∩G|/(|P|+|G|),
Challenge_Avg = HM(mA, inst F1); leaderboard = mean over datasets.
Attributes with no positive (or no negative) label in a dataset are treated as invalid there
and excluded from mA and from the instance sets (assumption: the official script excludes
attributes marked invalid per dataset; in UPAR those are exactly the never-annotated ones)."""
from __future__ import annotations
import numpy as np


def valid_attrs(y):
    pos = y.sum(0)
    return (pos > 0) & (pos < len(y))


def dataset_scores(pred, y, valid=None):
    """pred, y: bool [N,40]. Returns dict(mA, inst_f1, avg)."""
    v = valid_attrs(y) if valid is None else valid
    p, g = pred[:, v], y[:, v].astype(bool)
    tp = (p & g).sum(0); tn = (~p & ~g).sum(0)
    mA = float(((tp / g.sum(0).clip(1) + tn / (~g).sum(0).clip(1)) / 2).mean())
    inter = (p & g).sum(1)
    prec = (inter / np.maximum(p.sum(1), 1)).mean()          # empty prediction -> precision 0
    rec = (inter / np.maximum(g.sum(1), 1)).mean()
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)              # Rethinking-PAR form: F1 of mean P, R
    den = p.sum(1) + g.sum(1)
    f1_img = float(np.where(den > 0, 2 * inter / np.maximum(den, 1), 1.0).mean())
    avg = 2 * mA * f1 / max(mA + f1, 1e-12)
    return dict(mA=mA, inst_f1=float(f1), inst_f1_img=f1_img, avg=float(avg))


def challenge(pred, y, dom):
    res = {d: dataset_scores(pred[dom == d], y[dom == d]) for d in np.unique(dom)}
    res["mean"] = {k: float(np.mean([res[d][k] for d in res])) for k in ("mA", "inst_f1", "inst_f1_img", "avg")}
    return res
