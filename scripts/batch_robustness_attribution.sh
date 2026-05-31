#!/bin/bash
# P1.1 Robustness attribution: 5 new configs × 3 seeds
# Each config: train_ddp.sh + run_robustness.py
# Output: outputs/attrib_<config>/<seed>/resnet50/{best_resnet50.pth, training_log.json, ...}
#         outputs/attrib_robust_<config>/<seed>/resnet50/results.json

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_ROOT}"

LOG=/tmp/p1_attrib.log
echo "=== P1.1 attribution started: $(date) ===" | tee -a $LOG

# 配置: 5 个新组别
declare -A CONFIGS=(
    ["nx_only"]="--gauss-noise 0.5"
    ["archaug_no_noise"]="--archaug --gauss-noise 0.0"
    ["archaug_with_noise"]="--archaug --gauss-noise 0.5"
    ["mssa_only"]="--mssa"
    ["mssa_archaug_noise"]="--mssa --archaug --gauss-noise 0.5"
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

    # 检查是否已经训练完成
    EXISTING=$(ls outputs/${OUT_TRAIN}/*/resnet50/best_resnet50.pth 2>/dev/null | tail -1)
    if [ -n "$EXISTING" ]; then
      echo "[skip-train] ${cfg_name} seed=${seed} already done: $EXISTING" | tee -a $LOG
      CKPT="$EXISTING"
    else
      echo "[train] ${cfg_name} seed=${seed} args=${cfg_args}" | tee -a $LOG
      EPOCHS=$EPOCHS BS=$BS NPROC=$NPROC SEED=$seed OUT="$OUT_TRAIN" \
        bash scripts/train_ddp.sh resnet50 224 $cfg_args 2>&1 \
        | tee -a $LOG | tail -5
      CKPT=$(ls outputs/${OUT_TRAIN}/*/resnet50/best_resnet50.pth 2>/dev/null | tail -1)
    fi

    if [ -z "$CKPT" ]; then
      echo "[ERR] no ckpt for ${cfg_name} seed=${seed}, skip robustness" | tee -a $LOG
      continue
    fi

    # 鲁棒性评估
    EXISTING_ROBUST=$(ls outputs/${OUT_ROBUST}/*/resnet50/results.json 2>/dev/null | tail -1)
    if [ -n "$EXISTING_ROBUST" ]; then
      echo "[skip-robust] ${cfg_name} seed=${seed} already done" | tee -a $LOG
    else
      echo "[robust] ${cfg_name} seed=${seed} ckpt=$CKPT" | tee -a $LOG
      python scripts/run_robustness.py \
        --model resnet50 --dataset AL6 --img-size 224 224 \
        --ckpt "$CKPT" --output-subdir "$OUT_ROBUST" 2>&1 \
        | tee -a $LOG | tail -3
    fi
  done
done

echo "=== P1.1 ATTRIBUTION DONE: $(date) ===" | tee -a $LOG
