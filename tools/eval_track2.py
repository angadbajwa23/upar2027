"""Run a Track 2 submission folder like the Codabench ingestion program (import run.py, call
rank_gallery once with the whole gallery), check the output format, and score it on val
(mADM, mAP, Rank-1/5/10, mINP).

python tools/eval_track2.py build/track2 [--domain PA100k] [--n 2000] [--save scores.npy]
"""
import argparse
import importlib.util
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import ATTRS, DATA, domain_of, load_gt  # noqa: E402
from metrics import evaluate, fmt, subset_protocol  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("sub")
ap.add_argument("--domain", default="", help="evaluate one val dataset only (Market1501, PA100k or PETA)")
ap.add_argument("--n", type=int, default=0, help="random gallery subset size (0 = all)")
ap.add_argument("--save", default="", help="save the score matrix (.npy)")
args = ap.parse_args()

names, y, qid, q = load_gt("val")
mask = np.ones(len(names), bool)
if args.domain:
    mask &= domain_of(names) == args.domain
if args.n:
    idx = np.where(mask)[0]
    keep = np.random.default_rng(0).choice(idx, min(args.n, len(idx)), replace=False)
    mask = np.zeros(len(names), bool)
    mask[keep] = True
qs, ys, qids = (q, y, qid) if mask.all() else subset_protocol(y, mask)
sel = np.where(mask)[0]
sample = {
    "attribute_names": list(ATTRS),
    "queries": qs.astype(int).tolist(),
    "gallery": [{"index": j, "image": names[i], "image_path": os.path.join(DATA, names[i])}
                for j, i in enumerate(sel)],
}
print(f"Q={len(qs)} G={len(sel)}")

sub = os.path.abspath(args.sub)
os.chdir(sub)
spec = importlib.util.spec_from_file_location("run", os.path.join(sub, "run.py"))
run = importlib.util.module_from_spec(spec)
t0 = time.time()
spec.loader.exec_module(run)
run.load_model()
out = run.rank_gallery(sample)
print(f"wall time {time.time() - t0:.1f}s")

assert isinstance(out, dict) and list(out) == ["similarities"], list(out)
S = np.asarray(out["similarities"])
assert S.shape == (len(qs), len(sel)) and np.isfinite(S).all(), S.shape
if args.save:
    np.save(args.save, S)
print(fmt(evaluate(S, qs, ys, qids, higher_is_better=True)))
