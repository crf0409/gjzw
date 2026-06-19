#!/bin/bash
# 下载两个 Kaggle 古建筑数据集到 data/raw_kaggle/
# 用法: bash scripts/download_datasets.sh [dataset_name|all]
#   dataset_name: styles_periods | architectural_styles | all (默认)
# 依赖: curl 或 kaggle CLI 之一；后者需要 ~/.kaggle/kaggle.json

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="${ROOT}/data/raw_kaggle"
mkdir -p "${RAW_DIR}"

target="${1:-all}"

download_styles_periods() {
  local zip="${RAW_DIR}/styles_periods.zip"
  local out="${RAW_DIR}/styles_periods"
  if [[ -d "${out}" && -n "$(ls -A "${out}" 2>/dev/null || true)" ]]; then
    echo "[skip] styles_periods already extracted at ${out}"
    return
  fi
  echo "[1/2] Downloading gustavoachavez/architectural-styles-periods-dataset ..."
  curl -L --fail \
    -o "${zip}" \
    "https://www.kaggle.com/api/v1/datasets/download/gustavoachavez/architectural-styles-periods-dataset"
  echo "[1/2] Extracting ..."
  mkdir -p "${out}"
  unzip -q -o "${zip}" -d "${out}"
  echo "[1/2] Done -> ${out}"
}

download_architectural_styles() {
  local out="${RAW_DIR}/architectural_styles"
  if [[ -d "${out}" && -n "$(ls -A "${out}" 2>/dev/null || true)" ]]; then
    echo "[skip] architectural_styles already extracted at ${out}"
    return
  fi
  echo "[2/2] Downloading dumitrux/architectural-styles-dataset ..."
  if command -v kaggle >/dev/null 2>&1; then
    kaggle datasets download dumitrux/architectural-styles-dataset \
      -p "${RAW_DIR}/" --unzip
    if [[ ! -d "${out}" ]]; then
      mkdir -p "${out}"
      find "${RAW_DIR}" -maxdepth 1 -type d -name "*architectural*styles*" \
        ! -path "${RAW_DIR}" -exec mv {}/* "${out}/" \; 2>/dev/null || true
    fi
  else
    local zip="${RAW_DIR}/architectural_styles.zip"
    curl -L --fail \
      -o "${zip}" \
      "https://www.kaggle.com/api/v1/datasets/download/dumitrux/architectural-styles-dataset"
    mkdir -p "${out}"
    unzip -q -o "${zip}" -d "${out}"
  fi
  echo "[2/2] Done -> ${out}"
}

case "${target}" in
  styles_periods)        download_styles_periods ;;
  architectural_styles)  download_architectural_styles ;;
  all)
    download_styles_periods
    download_architectural_styles
    ;;
  *)
    echo "Unknown target: ${target}"
    echo "Usage: $0 [styles_periods|architectural_styles|all]"
    exit 2
    ;;
esac

echo
echo "All requested datasets downloaded under: ${RAW_DIR}"
echo "Next: python -m src.data.dataset_unifier --all"
