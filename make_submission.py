"""Build a Codabench submission folder (+ zip) for Track 1 or Track 2 from trained weights.

python make_submission.py --config configs/track2_final.json --out build/track2
python make_submission.py --config configs/track1_final.json --out build/track1

The folder holds run.py and metadata.yaml (from submission/track{1,2}/), model.py, scoring.py,
a vendored copy of timm (the platform image has no timm and no network), one fp16 state dict per
model (w0.pt, w1.pt, ...) and config.json. Each "checkpoint" in the config can be a checkpoint
written by train.py (runs/<name>/best.pt, last.pt or ep<N>.pt) or a released weight file.
"""
import argparse
import json
import os
import shutil
import subprocess

import timm
import torch

from common import ATTRS

HERE = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--out", required=True)
p.add_argument("--no-zip", action="store_true")
args = p.parse_args()

cfg = json.load(open(args.config))
track = int(cfg["track"])
out = os.path.abspath(args.out)
if os.path.exists(out):
    shutil.rmtree(out)
os.makedirs(out)
for f in ("run.py", "metadata.yaml"):
    shutil.copy(os.path.join(HERE, "submission", f"track{track}", f), out)
for f in ("model.py", "scoring.py"):
    shutil.copy(os.path.join(HERE, f), out)
shutil.copytree(os.path.dirname(timm.__file__), os.path.join(out, "timm"),
                ignore=shutil.ignore_patterns("__pycache__"))

models = []
for i, m in enumerate(cfg["models"]):
    ck = torch.load(os.path.join(HERE, m["checkpoint"]), map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    sd = {k: v.half() if v.is_floating_point() else v for k, v in sd.items()}
    torch.save(sd, os.path.join(out, f"w{i}.pt"))
    e = {"backbone": m["backbone"], "hw": m["hw"], "head": m.get("head", "sigmoid"), "weights": f"w{i}.pt"}
    if "weight" in m:
        e["weight"] = m["weight"]
    models.append(e)

if track == 2:
    sub = {"models": models, "scoring": cfg["scoring"], "tta_flip": cfg.get("tta_flip", False),
           "batch": cfg.get("batch", 256), "threads": cfg.get("threads", 16)}
    if cfg.get("knn"):
        sub["knn"] = cfg["knn"]
else:
    cal = json.load(open(os.path.join(HERE, cfg["thresholds"])))
    sub = {"models": models, "attribute_names": ATTRS,
           "calib": {k: cal[k] for k in ("logit_shift", "force1", "force_groups")}}
json.dump(sub, open(os.path.join(out, "config.json"), "w"), indent=1)
print("folder", out)

if not args.no_zip:
    zip_path = out.rstrip("/") + ".zip"
    if os.path.exists(zip_path):
        os.remove(zip_path)
    subprocess.run(["zip", "-qr", zip_path, ".", "-x", "*/__pycache__/*", "*.DS_Store"], cwd=out, check=True)
    print("zip   ", zip_path, f"{os.path.getsize(zip_path) / 1e6:.0f} MB")
