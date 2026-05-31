#!/bin/bash
# P1.3 Multi-model 5-fold × 3-seed CV on AL6 — for Friedman + Holm post-hoc.
# Adds 3 models so we have ≥4 models in the significance test.
# Already done: cv_baseline (resnet50), cv_aafnet_v2 (resnet50 with full AAFNet)
# New: archaug_noise_only resnet50 (= attrib_archaug_with_noise reused),
#      efficientnet_v2_s_tv baseline,
#      convnext_tiny_tv baseline.

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_ROOT}"

LOG=/tmp/p1_multi_cv.log
echo "=== P1.3 multi-model CV started: $(date) ===" | tee -a $LOG

EPOCHS=30
NPROC=4

# Model 1: resnet50 + ArchAug + GaussianNoise (no MSSA, no SupCon) — augmentation-only
python scripts/run_cv.py \
  --model resnet50 --dataset AL6 --img-size 224 224 \
  --folds 5 --seeds 42 1337 2024 \
  --epochs $EPOCHS --batch-size 32 --nproc $NPROC \
  --output-subdir cv_aug_only \
  --extra-args "--archaug --gauss-noise 0.5" 2>&1 \
  | tee -a $LOG | tail -20
echo "[1/3] cv_aug_only done" | tee -a $LOG

# Model 2: efficientnet_v2_s_tv baseline
python scripts/run_cv.py \
  --model efficientnet_v2_s_tv --dataset AL6 --img-size 224 224 \
  --folds 5 --seeds 42 1337 2024 \
  --epochs $EPOCHS --batch-size 32 --nproc $NPROC \
  --output-subdir cv_efficientnetv2 2>&1 \
  | tee -a $LOG | tail -20
echo "[2/3] cv_efficientnetv2 done" | tee -a $LOG

# Model 3: convnext_tiny_tv baseline
python scripts/run_cv.py \
  --model convnext_tiny_tv --dataset AL6 --img-size 224 224 \
  --folds 5 --seeds 42 1337 2024 \
  --epochs $EPOCHS --batch-size 32 --nproc $NPROC \
  --output-subdir cv_convnext 2>&1 \
  | tee -a $LOG | tail -20
echo "[3/3] cv_convnext done" | tee -a $LOG

echo "=== P1.3 MULTI-MODEL CV DONE: $(date) ===" | tee -a $LOG
