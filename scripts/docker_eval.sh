#!/bin/bash
# Run the built submissions on the val split inside the Codabench image (torch 2.4.1, no timm:
# the submission folders ship their own copy). Needs a GPU supported by CUDA 12.1 (e.g. A10, A100).
#   bash scripts/docker_eval.sh /path/to/data
set -e
DATA=$(realpath "$1")
cd "$(dirname "$0")/.."
for t in 2 1; do
  docker run --rm --gpus all --shm-size 8g -v "$PWD":/upar2027 -v "$DATA":/data -e UPAR_DATA=/data \
    -w /upar2027 pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime python tools/eval_track$t.py build/track$t
done
