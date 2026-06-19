#!/bin/bash
# CapsNet 训练 + 鲁棒性 (3 seeds × 30 epochs)
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_ROOT}"
LOG=/tmp/capsnet_main.log
echo "=== CapsNet training started: $(date) ===" | tee $LOG

SEEDS=(42 1337 2024)
for seed in "${SEEDS[@]}"; do
  TRAIN_OUT="capsnet_seed${seed}"
  ROBUST_OUT="capsnet_robust_seed${seed}"

  EXISTING=$(ls outputs/${TRAIN_OUT}/*/capsnet/best_capsnet.pth 2>/dev/null | tail -1)
  if [ -z "$EXISTING" ]; then
    echo "[train] capsnet seed=${seed}" | tee -a $LOG
    EPOCHS=30 BS=32 NPROC=4 SEED=$seed OUT="$TRAIN_OUT" \
      bash scripts/train_ddp.sh capsnet 224 2>&1 | tee -a $LOG | tail -10
    EXISTING=$(ls outputs/${TRAIN_OUT}/*/capsnet/best_capsnet.pth 2>/dev/null | tail -1)
  else
    echo "[skip-train] $EXISTING" | tee -a $LOG
  fi

  if [ -n "$EXISTING" ]; then
    echo "[robust] capsnet seed=${seed}" | tee -a $LOG
    python scripts/run_robustness.py --model capsnet --dataset AL6 --img-size 224 224 \
      --ckpt "$EXISTING" --output-subdir "$ROBUST_OUT" 2>&1 | tee -a $LOG | tail -3
  fi
done

# Also CV (5-fold × 3-seed) for inclusion in §5.6 multi-model significance
echo "[cv] capsnet 5-fold × 3-seed" | tee -a $LOG
python scripts/run_cv.py --model capsnet --dataset AL6 --img-size 224 224 \
  --folds 5 --seeds 42 1337 2024 --epochs 30 --batch-size 32 --nproc 4 \
  --output-subdir cv_capsnet 2>&1 | tee -a $LOG | tail -10

echo "=== CapsNet DONE: $(date) ===" | tee -a $LOG
