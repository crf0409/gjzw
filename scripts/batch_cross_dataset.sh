#!/bin/bash
# 单独跑 cross_dataset 阶段 (transfer ASP/AS25 → AL6 + within 3 datasets)
# 估计 ~50 分钟.

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

# Transfer ASP_clean → AL6 (3 phases)
wait_gpu_idle
echo
echo "=== transfer ASP_clean → AL6 ==="
python scripts/run_cross_dataset.py --mode transfer \
  --source ASP_clean --target AL6 --img-size 224 224 \
  --epochs $EPOCHS --seed 42

# Transfer AS25_clean → AL6 (3 phases)
wait_gpu_idle
echo
echo "=== transfer AS25_clean → AL6 ==="
python scripts/run_cross_dataset.py --mode transfer \
  --source AS25_clean --target AL6 --img-size 224 224 \
  --epochs $EPOCHS --seed 42

# Within: AL6, ASP_clean, AS25_clean
wait_gpu_idle
echo
echo "=== within: AL6 / ASP_clean / AS25_clean ==="
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
import json, glob
out = {}
for tag in ['cv_baseline', 'cv_baseline_seed1337', 'cv_baseline_seed2024',
            'cv_aafnet_v2', 'cv_aafnet_v2_seed1337', 'cv_aafnet_v2_seed2024']:
    files = glob.glob(f'outputs/{tag}/*/resnet50/cv_summary.json')
    if not files: continue
    s = json.load(open(files[0]))
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
