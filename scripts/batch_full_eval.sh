#!/bin/bash
# 一键跑完后续所有关键实验 (导师要求的 P4–P6).
# 等待当前 GPU 队列空闲后开始. 顺序串行保证产出可比.
#
# 估计时间: 60-90 分钟.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_ROOT}"

EPOCHS=${EPOCHS:-30}

wait_gpu_idle () {
  while pgrep -f "torchrun" >/dev/null; do
    echo "[wait] torchrun still running ..."
    sleep 30
  done
  echo "[wait] GPU idle"
}

run_cv () {
  local model=$1 seed=$2 outdir=$3 extra=$4
  echo
  echo "=== CV: $outdir / model=$model / seed=$seed ==="
  python scripts/run_cv.py --model $model --dataset AL6 --img-size 224 224 \
    --folds 5 --seeds $seed --epochs $EPOCHS \
    --output-subdir $outdir --extra-args "$extra"
}

# ── P4 5-fold × 3-seed CV (baseline + AAFNet v2) ──
# seed=42 已经跑过, 这里补 1337 和 2024
wait_gpu_idle
run_cv resnet50 1337 cv_baseline_seed1337 ""
wait_gpu_idle
run_cv resnet50 2024 cv_baseline_seed2024 ""
wait_gpu_idle
run_cv resnet50 1337 cv_aafnet_v2_seed1337 \
  "--mssa --loss-type focalls_supcon --supcon-weight 0.3 --archaug --perspective 0.3 --arch-occlusion 0.3 --weather 0.3 --gauss-noise 0.5"
wait_gpu_idle
run_cv resnet50 2024 cv_aafnet_v2_seed2024 \
  "--mssa --loss-type focalls_supcon --supcon-weight 0.3 --archaug --perspective 0.3 --arch-occlusion 0.3 --weather 0.3 --gauss-noise 0.5"

# ── P5 方法消融 (Baseline → +MSSA → +Focal+LS → +SupCon → +ArchAug → +Noise → +KD) ──
wait_gpu_idle
echo
echo "=== P5 方法消融 ==="
python scripts/run_ablations.py --base-model resnet50 --epochs $EPOCHS \
  --seeds 42 1337 --axes a_mssa b_loss c_aug --output-subdir ablations_main

# ── P6 跨数据集 transfer (ImageNet → ASP_clean → AL6) ──
wait_gpu_idle
echo
echo "=== P6 跨数据集 transfer (ASP_clean -> AL6) ==="
python scripts/run_cross_dataset.py --mode transfer \
  --source ASP_clean --target AL6 --img-size 224 224 \
  --epochs $EPOCHS --seed 42

# 同样 AS25_clean -> AL6
wait_gpu_idle
echo
echo "=== P6 跨数据集 transfer (AS25_clean -> AL6) ==="
python scripts/run_cross_dataset.py --mode transfer \
  --source AS25_clean --target AL6 --img-size 224 224 \
  --epochs $EPOCHS --seed 42

# 单数据集 within (AL6 / ASP_clean / AS25_clean)
wait_gpu_idle
echo
echo "=== P6 跨数据集 within ==="
python scripts/run_cross_dataset.py --mode within \
  --model resnet50 --datasets AL6 ASP_clean AS25_clean \
  --img-size 224 224 --epochs $EPOCHS --seed 42

# 聚合
echo
echo "=== Aggregating results ==="
python scripts/aggregate_results.py

# 显著性
echo
echo "=== Significance ==="
mkdir -p outputs/sig_collect
python -c "
import json, glob, sys
out = {}
# 整合 baseline + AAFNet 各 seed 的 fold accuracies
for tag in ['cv_baseline', 'cv_baseline_seed1337', 'cv_baseline_seed2024',
            'cv_aafnet_v2', 'cv_aafnet_v2_seed1337', 'cv_aafnet_v2_seed2024']:
    files = glob.glob(f'outputs/{tag}/*/resnet50/cv_summary.json')
    if not files: continue
    s = json.load(open(files[0]))
    # 用 tag 区分 baseline vs aafnet, 不区分 seed
    key = 'baseline' if 'baseline' in tag else 'aafnet_v2'
    out.setdefault(key, []).extend(
        f['test_accuracy'] for f in s.get('folds', []) if 'test_accuracy' in f
    )
json.dump(out, open('outputs/sig_collect/fold_accs.json', 'w'), indent=2)
print('saved:', list(out.keys()), {k: len(v) for k, v in out.items()})
"
python -m src.evaluation.significance --in outputs/sig_collect/fold_accs.json \
  --out outputs/sig_collect

echo
echo "=== ALL DONE ==="
