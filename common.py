"""Shared definitions: attributes, groups, annotation loading, image preprocessing."""
from __future__ import annotations

import csv
import os

import numpy as np

# Data root prepared with the organizers' kit (https://github.com/speckean/UPAR-Challenge-2027):
# it holds annotations/task2/{train,val}/{gt,ids,queries}.csv and the image folders
# Market1501/, PA100k/, PETA/ that the paths in gt.csv refer to.
DATA = os.environ.get("UPAR_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
ANN = os.path.join(DATA, "annotations", "task2")
CACHE = os.environ.get("UPAR_CACHE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache"))

ATTRS = [
    "Age-Young", "Age-Adult", "Age-Old", "Gender-Female",
    "Hair-Length-Short", "Hair-Length-Long", "Hair-Length-Bald", "UpperBody-Length-Short",
    "UpperBody-Color-Black", "UpperBody-Color-Blue", "UpperBody-Color-Brown", "UpperBody-Color-Green",
    "UpperBody-Color-Grey", "UpperBody-Color-Orange", "UpperBody-Color-Pink", "UpperBody-Color-Purple",
    "UpperBody-Color-Red", "UpperBody-Color-White", "UpperBody-Color-Yellow", "UpperBody-Color-Other",
    "LowerBody-Length-Short", "LowerBody-Color-Black", "LowerBody-Color-Blue", "LowerBody-Color-Brown",
    "LowerBody-Color-Green", "LowerBody-Color-Grey", "LowerBody-Color-Orange", "LowerBody-Color-Pink",
    "LowerBody-Color-Purple", "LowerBody-Color-Red", "LowerBody-Color-White", "LowerBody-Color-Yellow",
    "LowerBody-Color-Other", "LowerBody-Type-Trousers&Shorts", "LowerBody-Type-Skirt&Dress",
    "Accessory-Backpack", "Accessory-Bag", "Accessory-Glasses-Normal", "Accessory-Glasses-Sun",
    "Accessory-Hat",
]
N_ATTR = 40

# Near-exclusive groups. `none` = an extra "no attribute in this group" class.
GROUPS = [
    dict(name="age", idx=[0, 1, 2], none=False),
    dict(name="hair", idx=[4, 5, 6], none=True),
    dict(name="ub_color", idx=list(range(8, 20)), none=False),
    dict(name="lb_color", idx=list(range(21, 33)), none=False),
    dict(name="lb_type", idx=[33, 34], none=False),
    dict(name="glasses", idx=[37, 38], none=True),
]
BINARY = [3, 7, 20, 35, 36, 39]

# Cached image size (H, W). Every image is resized to this once, train and test alike.
CACHE_HW = (384, 192)


def load_gt(split: str):
    """Returns (image relpaths, labels uint8 [N,40], query ids [N], queries uint8 [Q,40])."""
    with open(os.path.join(ANN, split, "gt.csv")) as fh:
        rows = list(csv.reader(fh))[1:]
    names = [r[0] for r in rows]
    y = np.array([[int(v) for v in r[1:]] for r in rows], dtype=np.uint8)
    with open(os.path.join(ANN, split, "ids.csv")) as fh:
        idmap = {r[0]: int(r[1]) for r in csv.reader(fh) if not r[0].startswith("#")}
    qid = np.array([idmap[n] for n in names], dtype=np.int64)
    with open(os.path.join(ANN, split, "queries.csv")) as fh:
        q = np.array([[int(v) for v in r] for r in csv.reader(fh) if not r[0].startswith("#")],
                     dtype=np.uint8)
    return names, y, qid, q


def domain_of(names):
    return np.array([n.split("/")[0] for n in names])


def load_image(path: str) -> np.ndarray:
    """PIL decode -> RGB -> bilinear resize to CACHE_HW. Used identically at train and test."""
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB").resize((CACHE_HW[1], CACHE_HW[0]), Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8)
