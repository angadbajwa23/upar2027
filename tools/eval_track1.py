"""Run a Track 1 submission folder like the Codabench ingestion program (import run.py, call
predict_batch in chunks of run.BATCH_SIZE), check the output format, and score it on a labelled split
(val by default; Track 1 score = harmonic mean of mA and instance F1, averaged over the datasets).

python tools/eval_track1.py build/track1 [--data /path/to/data] [--split val] [--domain PA100k] [--save probs.npy]
"""
import argparse
import importlib.util
import os
import sys
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("sub")
ap.add_argument("--data", default="", help="data root (default: $UPAR_DATA, else data/)")
ap.add_argument("--split", default="val",
                help="annotation split to score on: annotations/task2/<split>/ under the data root")
ap.add_argument("--domain", default="", help="evaluate one dataset only: first folder of the image paths (e.g. Market1501, PA100k, PETA)")
ap.add_argument("--save", default="", help="save the probabilities (.npy)")
args = ap.parse_args()
if args.data:
    os.environ["UPAR_DATA"] = os.path.abspath(args.data)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import ATTRS, DATA, domain_of, load_labels  # noqa: E402
from t1_metrics import challenge  # noqa: E402

names, y = load_labels(args.split)
dom = domain_of(names)
sel = np.where(dom == args.domain)[0] if args.domain else np.arange(len(names))

sub = os.path.abspath(args.sub)
if not os.path.isfile(os.path.join(sub, "run.py")):
    sys.exit(f"{sub} is not a built submission; run  bash scripts/build_submissions.sh  first")
os.chdir(sub)
spec = importlib.util.spec_from_file_location("run", os.path.join(sub, "run.py"))
run = importlib.util.module_from_spec(spec)
t0 = time.time()
spec.loader.exec_module(run)
run.load_model()
bs = int(getattr(run, "BATCH_SIZE", 64))
P = []
for s in range(0, len(sel), bs):
    batch = [{"image": names[i], "image_path": os.path.join(DATA, names[i]), "attribute_names": list(ATTRS)}
             for i in sel[s:s + bs]]
    P += run.predict_batch(batch)
print(f"wall time {time.time() - t0:.1f}s")

P = np.asarray(P, dtype=np.float64)
assert P.shape == (len(sel), 40) and np.isfinite(P).all() and (P >= 0).all() and (P <= 1).all(), P.shape
if args.save:
    np.save(args.save, P)
r = challenge(P > 0.5, y[sel], dom[sel])["mean"]
print(f"Track 1 score {r['avg']:.4f}  mA {r['mA']:.4f}  instance F1 {r['inst_f1']:.4f}")
