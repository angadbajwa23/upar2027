"""Decode + resize every train/val image once into a uint8 memmap [N, H, W, 3] (cache/<split>_384x192.npy).

python build_cache.py [train val]
"""
import os
import sys
from multiprocessing import Pool

import numpy as np

from common import CACHE, CACHE_HW, DATA, load_gt, load_image


def work(args):
    i, path = args
    return i, load_image(os.path.join(DATA, path))


if __name__ == "__main__":
    os.makedirs(CACHE, exist_ok=True)
    for split in sys.argv[1:] or ["train", "val"]:
        names, *_ = load_gt(split)
        out = os.path.join(CACHE, f"{split}_{CACHE_HW[0]}x{CACHE_HW[1]}.npy")
        arr = np.lib.format.open_memmap(out, mode="w+", dtype=np.uint8,
                                        shape=(len(names), *CACHE_HW, 3))
        with Pool(min(32, os.cpu_count() or 1)) as pool:
            for k, (i, im) in enumerate(pool.imap_unordered(work, enumerate(names), chunksize=64)):
                arr[i] = im
                if k % 20000 == 0:
                    print(split, k, len(names), flush=True)
        arr.flush()
        print("wrote", out, arr.shape, flush=True)
