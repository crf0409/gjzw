#!/bin/bash
# P1.3 (remote box version): 仅跑 cv_convnext.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_ROOT}"

LOG=/tmp/p1_multi_remote.log
echo "=== P1.3 remote started: $(date) ===" | tee -a $LOG
EPOCHS=30
NPROC=4

python scripts/run_cv.py \
  --model convnext_tiny_tv --dataset AL6 --img-size 224 224 \
  --folds 5 --seeds 42 1337 2024 \
  --epochs $EPOCHS --batch-size 32 --nproc $NPROC \
  --output-subdir cv_convnext 2>&1 | tee -a $LOG | tail -10
echo "[1/1] cv_convnext done" | tee -a $LOG

echo "=== P1.3 REMOTE DONE: $(date) ===" | tee -a $LOG
