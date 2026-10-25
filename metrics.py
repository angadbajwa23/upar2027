"""Local re-implementation of the official Track 2 metrics (mADM, mAP, Rank-k, mINP)."""
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def evaluate(scores, queries, gallery_labels, gallery_qid, higher_is_better=True,
             chunk=256, device="cuda" if torch.cuda.is_available() else "cpu", per_query=False):
    """scores: [Q,G] (similarities if higher_is_better, else distances).
    queries: [Q,40] 0/1. gallery_labels: [G,40] 0/1. gallery_qid: [G] index of matching query.
    Ties are broken by gallery order (stable sort), like the official scorer.
    per_query=True adds "per_query": {"ADM", "AP", "R"} arrays (R = number of matches)."""
    S = scores.float() if torch.is_tensor(scores) else torch.as_tensor(np.asarray(scores), dtype=torch.float32)
    Qm = torch.as_tensor(np.asarray(queries), dtype=torch.float32, device=device)
    Gm = torch.as_tensor(np.asarray(gallery_labels), dtype=torch.float32, device=device)
    gq = torch.as_tensor(np.asarray(gallery_qid), dtype=torch.long, device=device)
    nq, ng = S.shape
    n_attr = Qm.shape[1]
    out = dict(ADM=[], AP=[], INP=[], r1=[], R=[])
    for s in range(0, nq, chunk):
        e = min(nq, s + chunk)
        sc = S[s:e].to(device)
        if not higher_is_better:
            sc = -sc
        order = torch.sort(sc, dim=1, descending=True, stable=True).indices          # [c,G]
        q = Qm[s:e]
        agree = q @ Gm.T + (1 - q) @ (1 - Gm).T                                       # [c,G]
        dom = agree / n_attr
        mean = dom.mean(1, keepdim=True)
        mx = dom.max(1, keepdim=True).values
        ndom = ((dom - mean) / (mx - mean).clamp_min(1e-12)).clamp_min(0)
        match = (gq[None, :] == torch.arange(s, e, device=device)[:, None])          # [c,G]
        m_sorted = torch.gather(match, 1, order).float()
        nd_sorted = torch.gather(ndom, 1, order)
        ranks = torch.arange(1, ng + 1, device=device, dtype=torch.float32)[None, :]
        cum_m = m_sorted.cumsum(1)
        cum_nd = nd_sorted.cumsum(1)
        R = m_sorted.sum(1)
        out["AP"].append(((cum_m / ranks) * m_sorted).sum(1) / R)
        out["ADM"].append(((cum_nd / ranks) * m_sorted).sum(1) / R)
        last = (m_sorted * ranks).max(1).values
        out["INP"].append(R / last)
        out["R"].append(R)
        out["r1"].append(m_sorted.argmax(1).float() + 1)                             # first-hit rank
    res = {k: torch.cat(v).cpu().numpy() for k, v in out.items()}
    valid = np.isfinite(res["AP"])          # queries with >=1 match
    first = res["r1"][valid]
    m = {
        "mADM": float(res["ADM"][valid].mean()),
        "mAP": float(res["AP"][valid].mean()),
        "Rank-1": float((first <= 1).mean()),
        "Rank-5": float((first <= 5).mean()),
        "Rank-10": float((first <= 10).mean()),
        "mINP": float(res["INP"][valid].mean()),
        "n_queries": int(valid.sum()),
    }
    if per_query:
        m["per_query"] = {k: res[k] for k in ("ADM", "AP", "R")}
    return m


def subset_protocol(labels, mask):
    """Rebuild a retrieval protocol on a gallery subset: queries = unique attribute vectors."""
    y = labels[mask]
    q, qid = np.unique(y, axis=0, return_inverse=True)
    return q, y, qid.reshape(-1)


def fmt(m):
    return " ".join(f"{k}={v:.4f}" for k, v in m.items() if k not in ("n_queries", "per_query"))
