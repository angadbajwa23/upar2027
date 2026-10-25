"""Query x gallery scoring functions. Shipped verbatim inside the submission (torch 2.4 safe).

Every score is a bilinear form  phi(q) . psi(p)  so it is one matmul, never Q x G x 40.
All functions return SIMILARITIES (higher = better), float32 torch tensors [Q, G].
"""
from __future__ import annotations

import numpy as np
import torch

# Near-exclusive groups (see common.GROUPS); duplicated here so the submission is standalone.
GROUPS = [
    ([0, 1, 2], False),            # age
    ([4, 5, 6], True),             # hair (+none)
    (list(range(8, 20)), False),   # upper-body colour
    (list(range(21, 33)), False),  # lower-body colour
    ([33, 34], False),             # lower-body type
    ([37, 38], True),              # glasses (+none)
]
BINARY = [3, 7, 20, 35, 36, 39]


def _mm(a, b, chunk=4096):
    """a [Q,K] @ b[G,K].T on a's device, chunked over Q."""
    out = torch.empty(a.shape[0], b.shape[0], dtype=torch.float32, device=a.device)
    for s in range(0, a.shape[0], chunk):
        out[s:s + chunk] = a[s:s + chunk] @ b.T
    return out


def l1(q, p):
    """-sum|q - p|  (the example submission's distance, negated)."""
    return -(p.sum(1)[None, :] + _mm(q, 1 - 2 * p))


def agreement(q, p, w=None):
    """Expected number of agreeing attributes, optionally weighted per attribute."""
    w = torch.ones(q.shape[1], device=q.device) if w is None else w
    return _mm(torch.cat([q * w, (1 - q) * w], 1), torch.cat([p, 1 - p], 1))


def bernoulli_ll(q, p, eps=1e-4, w=None):
    """Log-likelihood of the query under independent Bernoullis."""
    p = p.clamp(eps, 1 - eps)
    w = torch.ones(q.shape[1], device=q.device) if w is None else w
    return _mm(torch.cat([q * w, (1 - q) * w], 1), torch.cat([p.log(), (1 - p).log()], 1))


def group_query_feats(q):
    """Query-side features for the grouped likelihood: per group a (soft) one-hot over the
    group's classes (+none), then [b, 1-b] for each independent binary."""
    parts = []
    for idx, none in GROUPS:
        g = q[:, idx]
        if none:
            g = torch.cat([g, (g.sum(1, keepdim=True) == 0).float()], 1)
        parts.append(g / g.sum(1, keepdim=True).clamp_min(1))
    b = q[:, BINARY]
    parts += [b, 1 - b]
    return torch.cat(parts, 1)


def sigmoid_to_group_probs(p):
    """Turn independent marginals into per-group categorical probabilities (+none)."""
    parts = []
    for idx, none in GROUPS:
        g = p[:, idx]
        if none:
            g = torch.cat([g, (1 - g).prod(1, keepdim=True)], 1)
        parts.append(g / g.sum(1, keepdim=True).clamp_min(1e-8))
    b = p[:, BINARY]
    parts += [b, 1 - b]
    return torch.cat(parts, 1)


def grouped_ll(q, gp, eps=1e-4):
    """gp: gallery group-probability vector (from a grouped head, or sigmoid_to_group_probs)."""
    return _mm(group_query_feats(q), gp.clamp(eps, 1).log())


def apply_temperature(p, t):
    """Rescale logits of marginals by 1/t (t scalar or [40])."""
    p = p.clamp(1e-6, 1 - 1e-6)
    return torch.sigmoid(torch.logit(p) / t)


def knn_smooth(feats, *probs, k=10, beta=0.3, tau=0.05, chunk=4096):
    """Within-gallery neighbour smoothing: p' = (1-beta) p + beta * softmax(sim/tau)-weighted mean
    of the k nearest gallery neighbours (cosine on L2-normalised feats). Applied to each array."""
    f = torch.nn.functional.normalize(feats.float(), dim=1)
    outs = [torch.empty_like(p) for p in probs]
    for s in range(0, f.shape[0], chunk):
        v, j = (f[s:s + chunk] @ f.T).topk(k + 1, dim=1)
        w = torch.softmax(v[:, 1:] / tau, 1)[..., None]          # drop self
        j = j[:, 1:]
        for p, o in zip(probs, outs):
            o[s:s + chunk] = (w * p[j]).sum(1)
    return [(1 - beta) * p + beta * o for p, o in zip(probs, outs)]


# ------------------------------------------------------------------ gallery propagation
def knn_graph(feats, k=10, tau=0.05, reciprocal=0, chunk=4096):
    """Sparse within-gallery neighbour graph (self excluded): indices J [G,k] and row weights
    W [G,k] = softmax(cos/tau) over the kept neighbours.
    reciprocal=K > 0: search the top-K and keep only neighbours j with i in top-K(j); the k
    best of those are kept, the rest get weight 0 (rows with no reciprocal neighbour are all 0)."""
    f = torch.nn.functional.normalize(feats.float(), dim=1)
    kk = max(k, reciprocal)
    V = torch.empty(f.shape[0], kk, device=f.device)
    J = torch.empty(f.shape[0], kk, dtype=torch.long, device=f.device)
    for s in range(0, f.shape[0], chunk):
        v, j = (f[s:s + chunk] @ f.T).topk(kk + 1, dim=1)
        V[s:s + chunk], J[s:s + chunk] = v[:, 1:], j[:, 1:]
    if reciprocal:
        rows = torch.arange(f.shape[0], device=f.device)[:, None]
        keep = torch.zeros_like(J, dtype=torch.bool)
        for s in range(0, f.shape[0], chunk):
            keep[s:s + chunk] = (J[J[s:s + chunk]] == rows[s:s + chunk, :, None]).any(-1)
        V = V.masked_fill(~keep, float("-inf"))
        V, o = V.topk(k, dim=1)
        J = torch.gather(J, 1, o)
    else:
        V, J = V[:, :k], J[:, :k]
    W = torch.softmax(V / tau, 1)
    W = torch.nan_to_num(W, nan=0.0)                # rows without any neighbour
    return J, W


def graph_mean(x, J, W):
    """Row i -> sum_j W[i,j] x[J[i,j]]; rows with zero total weight return x[i] unchanged."""
    m = (W[..., None] * x[J]).sum(1)
    empty = W.sum(1) == 0
    m[empty] = x[empty]
    return m


def diffuse(p, J, W, alpha=0.5, iters=1):
    """Label propagation (Zhou et al. 2004): p_{t+1} = (1-alpha) p_0 + alpha W p_t.
    iters=1 equals knn_smooth with beta=alpha."""
    q = p
    for _ in range(iters):
        q = (1 - alpha) * p + alpha * graph_mean(q, J, W)
    return q


def smooth_scores(S, J, W, b=0.5, chunk=2048):
    """Score-level propagation: S'[:, i] = (1-b) S[:, i] + b * sum_j W[i,j] S[:, J[i,j]]."""
    out = torch.empty_like(S)
    for s in range(0, S.shape[1], chunk):
        j, w = J[s:s + chunk], W[s:s + chunk]
        m = (S[:, j] * w[None]).sum(2)
        empty = w.sum(1) == 0
        m[:, empty] = S[:, s:s + chunk][:, empty]
        out[:, s:s + chunk] = (1 - b) * S[:, s:s + chunk] + b * m
    return out


def whiten_blocks(feats, dims, frac=1.0, shrink=0.03):
    """Gallery PCA-whitening per model block (transductive, label-free), then re-normalise and concat.
    Each L2-normalised block is centred, projected on its top frac*D eigenvectors (exact eigh of the
    gallery covariance) and scaled by 1/sqrt(var + shrink*mean(var))."""
    out = []
    for x in torch.split(feats.float(), dims, dim=1):
        x = torch.nn.functional.normalize(x, dim=1)
        x = x - x.mean(0, keepdim=True)
        var, V = torch.linalg.eigh(x.T @ x / x.shape[0])           # ascending
        d = max(1, min(int(round(x.shape[1] * frac)), x.shape[1] - 1))
        var, V = var[-d:].clamp_min(0), V[:, -d:]
        y = (x @ V) / (var + shrink * var.mean()).sqrt()
        out.append(torch.nn.functional.normalize(y, dim=1))
    return torch.nn.functional.normalize(torch.cat(out, 1), dim=1)


def mutual_clusters(feats, J, k=10, s0=0.45):
    """Connected components of the mutual-kNN graph (top-k of J) restricted to cosine >= s0. Labels [G]."""
    f = torch.nn.functional.normalize(feats.float(), dim=1)
    G = f.shape[0]
    jj = J[:, :k]
    sim = (f[:, None, :] * f[jj]).sum(-1)
    rows = torch.arange(G, device=f.device)[:, None].expand_as(jj)
    keep = (J[jj, :k] == rows[..., None]).any(-1) & (sim >= s0)
    a, b = rows[keep].cpu().numpy(), jj[keep].cpu().numpy()
    parent = np.arange(G)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for x, y in zip(a, b):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry
    lab = np.array([find(x) for x in range(G)])
    return torch.tensor(np.unique(lab, return_inverse=True)[1], device=f.device)


def cluster_consensus(p, lab, beta=0.5, cap=64):
    """p' = (1-beta) p + beta * mean of p over the image's cluster; clusters larger than cap are left
    unchanged (guard against chained / percolating clusters)."""
    n = int(lab.max()) + 1
    s = torch.zeros(n, p.shape[1], device=p.device, dtype=p.dtype).index_add_(0, lab, p)
    c = torch.zeros(n, device=p.device, dtype=p.dtype).index_add_(0, lab, torch.ones_like(p[:, 0]))
    w = beta * (c[lab] <= cap).to(p.dtype)[:, None]
    return (1 - w) * p + w * (s / c[:, None])[lab]


def neighbour_density(feats, m=5, chunk=4096):
    """Mean cosine similarity of each gallery image to its m nearest other gallery images."""
    f = torch.nn.functional.normalize(feats.float(), dim=1)
    out = torch.empty(f.shape[0], device=f.device)
    for s in range(0, f.shape[0], chunk):
        out[s:s + chunk] = (f[s:s + chunk] @ f.T).topk(m + 1, dim=1).values[:, 1:].mean(1)
    return out


def density_gate(dens, alpha, gamma=1.0, floor=0.0):
    """Per-image diffusion strength [G,1] from the image's density RANK within this gallery:
    alpha * (floor + (1-floor) * quantile**gamma). Rank-based, so no absolute similarity scale."""
    qnt = dens.argsort().argsort().float() / max(len(dens) - 1, 1)
    return (alpha * (floor + (1 - floor) * qnt.pow(gamma)))[:, None]
