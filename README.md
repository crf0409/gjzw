# gjzw | 古建筑物图像分类系统 (Ancient-Building Image Classification)

基于深度学习的古建筑物自动识别与分类系统, 配套 SCI Q2 论文 *AAFNet: An
Architectural-Aware Fusion Network with Implicit Neural Texture Compression
for Robust Ancient-Building Image Classification* 的实验代码.

本开源仓库只包含代码、配置与必要入口说明; 原始/处理后数据集、模型权重、训练输出、
论文 Markdown/Word/PPT/PDF 文档及其生成脚本不随仓库发布.

## 功能特性

- 三个公开数据集 (AL6 / ASP / AS25), 配套数据审计 (MD5 + pHash 重复检测) 与
  脱重 cleaned 变体
- 多主干网络对比 (ResNet-50 / VGG / Inception / EfficientNet / MobileNet-V3 /
  ViT / ConvNeXt-Tiny / Swin-V2-Tiny / EfficientNetV2-S 等)
- AAFNet 框架: 多尺度风格注意力 MSSA + 跨尺度门控融合 CSGF + 监督对比蒸馏
  SASC-KD + 多样性加权专家混合 DW-MoE + 域内增强 ArchAug
- 5-fold × 3-seed 交叉验证 + Wilcoxon 配对显著性检验
- 鲁棒性套件 (5 类扰动 × 3 严重度) + Grad-CAM 可解释性 + Pareto 效率分析
- INR-AncientArch (探索性): 基于 SIREN 的隐式神经压缩, 与剪枝的存储/延迟对照

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 数据准备

```bash
# 下载 Kaggle 数据 (需要 kaggle credentials, 见 scripts/download_datasets.sh)
bash scripts/download_datasets.sh

# 统一三个数据集到 data/processed/{AL6,ASP,AS25}/{train,test}/
python -m src.data.dataset_unifier --dataset all

# 数据审计 + 输出 outputs/data_audit/<dataset>/duplicate_report.md
python scripts/data_audit.py --dataset AL6
python scripts/data_audit.py --dataset ASP
python scripts/data_audit.py --dataset AS25

# 生成脱重 cleaned 变体 (移除 MD5 重复 + 跨集泄漏)
python scripts/dedup_dataset.py --dataset ASP
python scripts/dedup_dataset.py --dataset AS25
# 可选: 进一步移除 pHash 近似重复 (生成 *_strict 变体)
# python scripts/dedup_dataset.py --dataset AS25 --strict

# 预生成 PT 缓存 (uint8 [N,C,H,W] tensor, 训练时 GPU 端归一化)
python scripts/build_pt_cache.py --dataset all --sizes 224x224
```

### 3. 训练

DDP 单机 4 卡训练 (默认 RTX 3090):

```bash
# Baseline ResNet-50 (无 AAFNet 增强)
EPOCHS=30 BS=32 NPROC=4 bash scripts/train_ddp.sh resnet50 224 \
    --output-subdir ddp_baseline

# AAFNet v2 (MSSA + SupCon + ArchAug + Gaussian noise aug)
EPOCHS=30 BS=32 NPROC=4 bash scripts/train_ddp.sh resnet50 224 \
    --mssa --loss-type focalls_supcon --supcon-weight 0.3 \
    --archaug --gauss-noise 0.5 \
    --output-subdir ddp_aafnet_v2
```

### 4. 评估与结果复现

```bash
# 5-fold × 3-seed CV
python scripts/run_cv.py --model resnet50 --dataset AL6 \
    --folds 5 --seeds 42 1337 2024 --epochs 30

# 鲁棒性套件
python scripts/run_robustness.py --model resnet50 \
    --ckpt outputs/ddp_aafnet_v2/latest/resnet50/best_resnet50.pth

# 跨数据集分析
python scripts/run_cross_dataset.py --mode transfer \
    --source ASP_clean --target AL6

# 消融矩阵
python scripts/run_ablations.py --base-model resnet50 --epochs 30

# INR 探索性实验
python scripts/fit_inr_dataset.py --dataset AL6 --split train --hidden 256 --layers 4
python scripts/train_inr_classifier.py --hidden 256 --layers 4 --head mlp
python scripts/benchmark_inr_vs_pruning.py --dataset AS25_clean

# DW-MoE 集成
python scripts/collect_member_predictions.py --dataset AL6 --img-size 224 224 \
    --members resnet50:CKPT:224 mobilenet_v3:CKPT:224 ... \
    --output-subdir ensemble_inputs
python scripts/train_ensemble.py --inputs outputs/ensemble_inputs/latest/members.npz \
    --output-subdir ensemble

# 显著性分析
python -m src.evaluation.significance --in outputs/cv_baseline outputs/cv_aafnet_v2

# 聚合实验数字到本地输出目录
python scripts/aggregate_results.py
```

## 目录结构

```
gjzw/
├── src/
│   ├── models/           # backbones / MSSA / DW-MoE / INR
│   ├── data/             # dataset_unifier / arch_aug / cached_dataset
│   ├── training/         # ddp_trainer / losses / cv_runner
│   └── evaluation/       # robustness / efficiency / significance / interpret
├── scripts/              # 全部入口脚本 (上面 quickstart 调用的)
├── config/               # YAML 配置 (default + 消融变体)
├── data/                 # 本地数据目录, 不纳入 Git
└── outputs/              # 本地训练结果目录, 不纳入 Git
```

## 开源边界

- 本项目代码以 MIT License 发布, 见 `LICENSE`.
- `data/`, `dataset/`, `outputs/`, 模型权重、缓存、压缩包和 Office/PDF 文档均被忽略.
- 如需复现实验, 请按数据集原始许可自行下载数据, 并在本地生成缓存与权重.
