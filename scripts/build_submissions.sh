#!/bin/bash
# Build the two Codabench submissions (folders + zips) in build/ from the weights in weights/.
set -e
cd "$(dirname "$0")/.."
python make_submission.py --config configs/track2_final.json --out build/track2
python make_submission.py --config configs/track1_final.json --out build/track1
