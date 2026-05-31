#!/bin/bash
# 批量串行 CV: baseline / AAFNet v2 / ConvNeXt-Tiny
# 每个模型 5-fold × 1 seed (seed=42), 30 epoch
# 总时间估计: 15 fold × 3 min ≈ 45 分钟
#
# 完成后用 src.evaluation.significance 比较

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_ROOT}"

EPOCHS=${EPOCHS:-30}
SEED=${SEED:-42}

echo "=== CV plan ==="
echo "  baseline ResNet-50:    5 folds × seed $SEED × $EPOCHS epoch"
echo "  AAFNet v2 (resnet50):  5 folds × seed $SEED × $EPOCHS epoch  (with --mssa --supcon --archaug --gauss-noise)"
echo "  ConvNeXt-Tiny:         5 folds × seed $SEED × $EPOCHS epoch"
echo

# 等已跑的 CV baseline 跑完 (如果在跑)
while pgrep -f "run_cv.py.*resnet50.*output-subdir cv_baseline" >/dev/null; do
  sleep 30
done
echo "[batch] cv_baseline finished (or not running)"

if ! [ -f outputs/cv_baseline/*/resnet50/cv_summary.json ]; then
  python scripts/run_cv.py --model resnet50 --dataset AL6 --img-size 224 224 \
    --folds 5 --seeds $SEED --epochs $EPOCHS \
    --output-subdir cv_baseline
fi

# AAFNet v2 (with all components)
echo
echo "[batch] starting CV AAFNet v2 ..."
python scripts/run_cv.py --model resnet50 --dataset AL6 --img-size 224 224 \
  --folds 5 --seeds $SEED --epochs $EPOCHS \
  --output-subdir cv_aafnet_v2 \
  --extra-args "--mssa --loss-type focalls_supcon --supcon-weight 0.3 --archaug --perspective 0.3 --arch-occlusion 0.3 --weather 0.3 --gauss-noise 0.5"

# ConvNeXt-Tiny baseline
echo
echo "[batch] starting CV ConvNeXt-Tiny ..."
python scripts/run_cv.py --model convnext_tiny --dataset AL6 --img-size 224 224 \
  --folds 5 --seeds $SEED --epochs $EPOCHS \
  --output-subdir cv_convnext

echo
echo "=== ALL CV DONE ==="
echo "Summaries:"
for d in outputs/cv_baseline outputs/cv_aafnet_v2 outputs/cv_convnext; do
  s=$(find $d -name 'cv_summary.json' | head -1)
  if [ -n "$s" ]; then
    echo "  $s"
    python -c "import json; r=json.load(open('$s')); print('   ', r.get('model'), 'mean=', r.get('test_accuracy_mean'), 'std=', r.get('test_accuracy_std'))"
  fi
done
