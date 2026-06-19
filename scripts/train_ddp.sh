#!/bin/bash
# 4 卡 DDP 训练启动器
#
# 用法:
#   bash scripts/train_ddp.sh resnet50              # 224x224
#   bash scripts/train_ddp.sh inception_v3 299      # 299x299
#   NPROC=2 bash scripts/train_ddp.sh resnet50      # 仅用 2 卡
#   EPOCHS=100 BS=64 bash scripts/train_ddp.sh resnet50

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

MODEL="${1:?Usage: $0 <model> [img_size]}"
SIZE="${2:-224}"
shift                # 消费 MODEL
[[ $# -gt 0 ]] && shift  # 消费 SIZE (若传了)

NPROC="${NPROC:-4}"
DATASET="${DATASET:-AL6}"
EPOCHS="${EPOCHS:-80}"
BS="${BS:-32}"
LR="${LR:-1e-4}"
OPT="${OPT:-adam}"
SCHED="${SCHED:-cosine}"
SEED="${SEED:-42}"
OUT="${OUT:-ddp}"

# 关闭 NCCL P2P 在某些 RTX 30 系上有时更稳
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
# 找一个空闲端口
PORT="${PORT:-$(python -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')}"

echo "=== DDP Training ==="
echo "  model:       ${MODEL}"
echo "  dataset:     ${DATASET}"
echo "  img size:    ${SIZE}x${SIZE}"
echo "  epochs:      ${EPOCHS}"
echo "  batch_size:  ${BS} (per rank)"
echo "  nproc:       ${NPROC}"
echo "  master port: ${PORT}"
echo

# 确保 PT 缓存存在
CACHE_TRAIN="${ROOT}/data/cache/${DATASET}_${SIZE}x${SIZE}_rgb_train.pt"
if [[ ! -f "${CACHE_TRAIN}" ]]; then
  echo "[!] missing cache: ${CACHE_TRAIN}"
  echo "[!] building it now..."
  python scripts/build_pt_cache.py --dataset "${DATASET}" --size "${SIZE}" "${SIZE}"
fi

torchrun --standalone --nproc_per_node="${NPROC}" --master_port="${PORT}" \
  scripts/train_ddp.py \
  --model "${MODEL}" \
  --dataset "${DATASET}" \
  --img-size "${SIZE}" "${SIZE}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BS}" \
  --lr "${LR}" \
  --optimizer "${OPT}" \
  --schedule "${SCHED}" \
  --seed "${SEED}" \
  --output-subdir "${OUT}" \
  "$@"
