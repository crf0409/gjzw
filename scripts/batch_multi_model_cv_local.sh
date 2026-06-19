#!/bin/bash
# P1.3 (local box version): 仅跑 cv_aug_only + cv_efficientnetv2.
# convnext 由 remote 跑.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_ROOT}"

LOG=/tmp/p1_multi_local.log
echo "=== P1.3 local started: $(date) ===" | tee -a $LOG
EPOCHS=30
NPROC=4

python scripts/run_cv.py \
  --model resnet50 --dataset AL6 --img-size 224 224 \
  --folds 5 --seeds 42 1337 2024 \
  --epochs $EPOCHS --batch-size 32 --nproc $NPROC \
  --output-subdir cv_aug_only \
  --extra-args "--archaug --gauss-noise 0.5" 2>&1 | tee -a $LOG | tail -10
echo "[1/2] cv_aug_only done" | tee -a $LOG

python scripts/run_cv.py \
  --model efficientnet_v2_s_tv --dataset AL6 --img-size 224 224 \
  --folds 5 --seeds 42 1337 2024 \
  --epochs $EPOCHS --batch-size 32 --nproc $NPROC \
  --output-subdir cv_efficientnetv2 2>&1 | tee -a $LOG | tail -10
echo "[2/2] cv_efficientnetv2 done" | tee -a $LOG

echo "=== P1.3 LOCAL DONE: $(date) ===" | tee -a $LOG
