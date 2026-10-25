#!/bin/bash
# Train the four models behind the final entries (about 1-2 h each on one 32 GB GPU).
# Every model: input 252x126, batch 96, lr 5e-5, layer-wise lr decay 0.8, weight decay 0.05,
# dropout 0.1, label smoothing 0.05, EMA 0.9995, 1 warm-up epoch + cosine, colour/lighting aug (--photo 1).
# --save-epochs keeps every epoch's checkpoint and val predictions (runs/<name>/ep<N>.pt, preds_ep<N>.npz).
set -e
cd "$(dirname "$0")/.."
COMMON="--hw 252 126 --bs 96 --lr 5e-5 --ld 0.8 --photo 1 --grad-ckpt --save-epochs"

# EVA02-L, LP-FT: 1 epoch linear probing (frozen backbone, head lr 1e-3) + 6 epochs fine-tuning.
# Used: epoch 6 in the log's numbering (LP epoch + 5 FT epochs; best val retrieval score).
python train.py --name eva02L_lpft --backbone eva02_large_patch14_224.mim_m38m --epochs 7 --lp-epochs 1 $COMMON

# CLIP ViT-L/14, MetaCLIP 2.5B weights. Track 2 uses epoch 4 (best val retrieval score).
python train.py --name clipL_metaclip --backbone vit_large_patch14_clip_224.metaclip_2pt5b --epochs 6 $COMMON

# Second run of the same recipe. Track 1 uses its epoch 6.
python train.py --name clipL_metaclip_run2 --backbone vit_large_patch14_clip_224.metaclip_2pt5b --epochs 6 $COMMON

# CLIP ViT-L/14, OpenAI weights. Track 2 uses epoch 4 (best val retrieval score).
python train.py --name clipL_openai --backbone vit_large_patch14_clip_224.openai --epochs 6 $COMMON
