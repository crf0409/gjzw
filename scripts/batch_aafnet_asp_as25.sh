#!/bin/bash
# P1.2 — AAFNet on ASP_clean / AS25_clean, baseline vs full, 3 seeds each.
# Pure training, no fold split (single train/test split per dataset, 3 seeds).
# Output: outputs/asp_as25_<role>_<dataset>_seed<seed>/.../<model>/

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_ROOT}"

LOG=/tmp/p1_aafnet_asp_as25.log
echo "=== P1.2 AAFNet on ASP/AS25 started: $(date) ===" | tee -a $LOG

declare -A ROLE_ARGS=(
    ["baseline"]=""
    ["aafnet"]="--mssa --loss-type focalls_supcon --supcon-weight 0.3 --archaug --gauss-noise 0.5"
)

DATASETS=("ASP_clean" "AS25_clean")
SEEDS=(42 1337 2024)
EPOCHS=30
BS=32
NPROC=4

for dataset in "${DATASETS[@]}"; do
  for role in "${!ROLE_ARGS[@]}"; do
    args="${ROLE_ARGS[$role]}"
    for seed in "${SEEDS[@]}"; do
      OUT_DIR="asp_as25_${role}_${dataset}_seed${seed}"

      EXISTING=$(ls outputs/${OUT_DIR}/*/resnet50/best_resnet50.pth 2>/dev/null | tail -1)
      if [ -n "$EXISTING" ]; then
        echo "[skip] ${dataset} ${role} seed=${seed} done: $EXISTING" | tee -a $LOG
        continue
      fi

      echo "[train] ${dataset} ${role} seed=${seed} args=${args}" | tee -a $LOG
      EPOCHS=$EPOCHS BS=$BS NPROC=$NPROC SEED=$seed OUT="$OUT_DIR" DATASET="$dataset" \
        bash scripts/train_ddp.sh resnet50 224 $args 2>&1 \
        | tee -a $LOG | tail -8
    done
  done
done

echo "=== P1.2 AAFNet on ASP/AS25 DONE: $(date) ===" | tee -a $LOG
