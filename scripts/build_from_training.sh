#!/bin/bash
# Use your own trained models instead of the released weights: link the epochs used by the final
# entries into weights/, re-fit the Track 1 thresholds, and build both submissions.
set -e
cd "$(dirname "$0")/.."
mkdir -p weights
ln -sf ../runs/eva02L_lpft/ep6.pt          weights/eva02L_lpft_ep6.pt          # epoch 6 = LP epoch + 5 FT epochs
ln -sf ../runs/clipL_metaclip/ep4.pt       weights/clipL_metaclip_ep4.pt
ln -sf ../runs/clipL_openai/ep4.pt         weights/clipL_openai_ep4.pt
python fit_t1_thresholds.py --preds runs/eva02L_lpft/preds_ep6.npz runs/clipL_metaclip/preds_ep4.npz \
    --weights 0.4 0.6 --shrink 0.75 --out configs/track1_thresholds.json
bash scripts/build_submissions.sh
