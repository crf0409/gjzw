#!/bin/bash
# 补齐归因表: baseline + Full AAFNet 3-seed × 30ep + 鲁棒性
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_ROOT}"

LOG=/tmp/p1_attrib_sup.log
echo "=== P1.1 supplement started: $(date) ===" | tee -a $LOG

declare -A CONFIGS=(
    ["baseline"]=""
    ["aafnet_full"]="--mssa --loss-type focalls_supcon --supcon-weight 0.3 --archaug --gauss-noise 0.5"
)

SEEDS=(42 1337 2024)
EPOCHS=30
BS=32
NPROC=4

for cfg_name in "${!CONFIGS[@]}"; do
  cfg_args="${CONFIGS[$cfg_name]}"
  for seed in "${SEEDS[@]}"; do
    OUT_TRAIN="attrib_${cfg_name}_seed${seed}"
    OUT_ROBUST="attrib_robust_${cfg_name}_seed${seed}"

    EXISTING=$(ls outputs/${OUT_TRAIN}/*/resnet50/best_resnet50.pth 2>/dev/null | tail -1)
    if [ -n "$EXISTING" ]; then
      echo "[skip-train] ${cfg_name} seed=${seed} exists" | tee -a $LOG
      CKPT="$EXISTING"
    else
      echo "[train] ${cfg_name} seed=${seed} args=${cfg_args}" | tee -a $LOG
      EPOCHS=$EPOCHS BS=$BS NPROC=$NPROC SEED=$seed OUT="$OUT_TRAIN" \
        bash scripts/train_ddp.sh resnet50 224 $cfg_args 2>&1 | tee -a $LOG | tail -3
      CKPT=$(ls outputs/${OUT_TRAIN}/*/resnet50/best_resnet50.pth 2>/dev/null | tail -1)
    fi

    [ -z "$CKPT" ] && { echo "[ERR] no ckpt"; continue; }

    EXISTING_R=$(ls outputs/${OUT_ROBUST}/*/resnet50/results.json 2>/dev/null | tail -1)
    if [ -n "$EXISTING_R" ]; then
      echo "[skip-robust] ${cfg_name} seed=${seed} exists" | tee -a $LOG
    else
      echo "[robust] ${cfg_name} seed=${seed}" | tee -a $LOG
      python scripts/run_robustness.py \
        --model resnet50 --dataset AL6 --img-size 224 224 \
        --ckpt "$CKPT" --output-subdir "$OUT_ROBUST" 2>&1 | tee -a $LOG | tail -3
    fi
  done
done

echo "=== P1.1 SUPPLEMENT DONE: $(date) ===" | tee -a $LOG
