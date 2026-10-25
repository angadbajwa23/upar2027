# UPAR Challenge 2027: Density-Aware Prediction Sharing for Attribute-Based Person Retrieval

Training code, trained models and the exact submission code of team `angadbajwa` for the
[UPAR Challenge 2027](https://vap.aau.dk/rws/challenge/) (Real-World Surveillance workshop, WACV 2027).

| Track | Final entry | Test score | Place |
|---|---|---|---|
| 2: Attribute-based person retrieval | 3 ViT-L models + density-controlled sharing | 69.11 mADM (mAP 40.66, mINP 28.38) | 3rd / 49 |
| 1: Pedestrian attribute recognition | 2 ViT-L models + one threshold per attribute | 79.44 (mA 79.56, instance F1 79.32) | 20th / 89 |

Both tracks use the same attribute models, trained with one recipe. Only the last step differs.

- **Track 2.** Each gallery image is scored by the log-likelihood of the query attributes under the
  image's predicted probabilities. Before scoring, images that look alike share their predictions:
  label propagation on a kNN graph of the backbone features (PCA-whitened on the gallery). How much an
  image takes from its neighbours depends on its density rank in the gallery, so isolated images
  mostly keep their own prediction. Small groups of mutual neighbours are then pulled towards their
  group mean. None of this uses labels or extra training.
- **Track 1.** The two models' probabilities are averaged and each attribute gets its own decision
  threshold, fitted on the validation set, instead of the fixed 0.5.

## Repository layout

```
train.py                 train one attribute model on the cached images, evaluate on val after every epoch
build_cache.py           decode and resize all train/val images once (cache/*.npy)
fit_t1_thresholds.py     Track 1: fit the 40 per-attribute thresholds on val
make_submission.py       build a Codabench submission folder + zip from a config
common.py                attribute list, annotation loading, image loading
model.py, scoring.py     model and scoring code (both are copied into the submissions)
metrics.py, t1_metrics.py   local versions of the Track 2 and Track 1 metrics
submission/track{1,2}/   run.py + metadata.yaml of our scored submissions
configs/                 track2_final.json, track1_final.json, track1_thresholds.json
scripts/                 train_final.sh, build_submissions.sh, build_from_training.sh, docker_eval.sh
tools/                   eval_track{1,2}.py: run a submission folder the way Codabench does and score it on val
```

`submission/*/run.py`, `model.py` and `scoring.py` are unchanged copies of the files in our uploaded
zips. They still contain a few options that the final configs switch off (other scoring types, score
smoothing, the Track 1 `force1` rule, flip TTA).

## 1. Environment

Training and building: Python 3.12, torch 2.8.0 (CUDA 12.8), timm 1.0.30, one NVIDIA RTX PRO 4500
(32 GB). `make_submission.py` also needs the `zip` command.

```bash
pip install -r requirements.txt
```

or in Docker:

```bash
docker run --gpus all --shm-size 16g -it -v $PWD:/upar2027 -w /upar2027 \
    pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime bash
pip install timm==1.0.30 && apt-get update && apt-get install -y zip
```

A built submission runs in the Codabench image `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime`
without extra packages or network access, since the folder carries its own copy of timm. On
Codabench it ran on one NVIDIA A10 (24 GB) within the 5-minute limit.

## 2. Data

Download the challenge data with the organizers' kit
([UPAR-Challenge-2027](https://github.com/speckean/UPAR-Challenge-2027), `python download_datasets.py`)
and put it (or a link to it) at `data/`, or point `UPAR_DATA` to it:

```
data/
├── annotations/task2/{train,val}/{gt.csv, ids.csv, queries.csv}
├── Market1501/   (bounding_box_train/, bounding_box_test/, query/)
├── PA100k/release_data/release_data/
└── PETA/images/
```

Image paths in `gt.csv` are relative to `data/`. Both tracks use these annotation files. We train on
the official training split only; the validation split is used to choose epochs, the Track 1
thresholds and the Track 2 settings.

## 3. Rebuild the leaderboard submissions from the released weights

The four weight files (fp16 state dicts, about 600 MB each) are on the Hugging Face Hub:
[Angad23/upar2027](https://huggingface.co/Angad23/upar2027).

```bash
pip install -U huggingface_hub
hf download Angad23/upar2027 --include "*.pt" --local-dir weights
(cd weights && sha256sum -c ../weights.sha256)
```

| File | Backbone (timm name) | Used in |
|---|---|---|
| `eva02L_lpft_ep6.pt` | EVA02-L/14, `eva02_large_patch14_224.mim_m38m`, LP-FT | Track 1, Track 2 |
| `clipL_metaclip_ep4.pt` | CLIP ViT-L/14, `vit_large_patch14_clip_224.metaclip_2pt5b` | Track 2 |
| `clipL_openai_ep4.pt` | CLIP ViT-L/14, `vit_large_patch14_clip_224.openai` | Track 2 |
| `clipL_metaclip_run2_ep6.pt` | CLIP ViT-L/14 MetaCLIP, second training run | Track 1 |

```bash
bash scripts/build_submissions.sh        # -> build/track2(.zip), build/track1(.zip)
```

This gives the same code, config and weights as the two zips we uploaded. To run them on the
validation split the way the Codabench ingestion program does:

```bash
python tools/eval_track2.py build/track2      # mADM, mAP, Rank-1/5/10, mINP
python tools/eval_track1.py build/track1      # Track 1 score, mA, instance F1
bash scripts/docker_eval.sh /path/to/data     # both, inside the Codabench image
```

## 4. Train from scratch

```bash
python build_cache.py                    # cache/train_384x192.npy, cache/val_384x192.npy (~28 GB)
bash scripts/train_final.sh              # the four models, 1-2 h each
bash scripts/build_from_training.sh      # link the epochs we used, re-fit the Track 1 thresholds, build both submissions
```

All four models share one recipe:

- input 252×126, linear layer on the backbone's pooled feature (dropout 0.1)
- binary cross-entropy with DeepMAR positive-rate weights, label smoothing 0.05
- AdamW, lr 5e-5, weight decay 0.05, batch 96, layer-wise lr decay 0.8
- one warm-up epoch, then cosine decay; EMA of the weights (0.9995)
- flip, zoom, shift, brightness and contrast augmentation, plus colour/lighting augmentation
  (`--photo 1`: per-channel gain up to ±10%, gamma, saturation up to ±30%)

EVA02-L first trains only the linear layer for one epoch (LP-FT); its log counts that as epoch 1,
so `ep6` means the LP epoch plus 5 fine-tuning epochs. For each model we used the epoch with the best
val retrieval score, except the Track 1 CLIP-L, which is the last epoch (6) of a second run.

GPU training is not bit-exact, so retrained models will differ slightly from the released ones. Our
first MetaCLIP run stopped after epoch 4 because the disk filled up; epoch 4 was also the best epoch of
the complete second run.

## 5. Settings of the final entries

**Track 2** (`configs/track2_final.json`)

- equal-weight average of the three models' probabilities, no flip at test time
- scoring: Bernoulli log-likelihood of the query, probabilities clipped to [1e-4, 1 − 1e-4]
- graph features: each model's pooled feature, L2-normalised and PCA-whitened on the test gallery
  (shrinkage 0.01), then concatenated
- kNN graph with k = 20, softmax weights with temperature 0.15, 3 propagation steps
- density = mean cosine similarity to the 5 nearest gallery images, converted to a rank u in [0, 1];
  sharing strength α = 0.95 · (0.2 + 0.8u)
- groups: mutual 10-NN with cosine ≥ 0.45; each image moves halfway to its group mean; groups larger
  than 64 are left alone
- if inference falls behind the time limit, the last model in the list is dropped for the whole gallery

**Track 1** (`configs/track1_final.json`)

- probabilities averaged with weights 0.4 (EVA02-L) and 0.6 (CLIP-L)
- one logit shift per attribute added before the 0.5 cut, fitted on the full val split by coordinate
  ascent (−3 to +3 in steps of 0.1, three sweeps, maximising the Track 1 score) and shrunk 25% towards
  their mean (`fit_t1_thresholds.py`)
- if inference falls behind the time limit, the remaining images are scored by the first model only

## Citation

```bibtex
@inproceedings{upar2027density,
  title     = {Density-Aware Prediction Sharing for Attribute-Based Person Retrieval},
  author    = {AUTHORS_TODO},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) Workshops},
  year      = {2027}
}
```

## License

The code is MIT licensed (see `LICENSE`). The trained weights are for non-commercial research only:
they are trained on the UPAR annotations (CC BY-NC-SA) and the PA100k, PETA and Market-1501 images,
and the MetaCLIP models start from MetaCLIP weights (CC BY-NC 4.0). The pre-trained backbones keep
their own licenses.
