# gjzw | 甲骨文字符分类系统

基于深度学习的甲骨文古文字自动识别与分类系统，支持多骨格类别的高精度图像分类。

## 功能特性

- 甲骨文图像预处理与增强（旋转、翻转、对比度）
- 主干网络对比（ResNet / ViT / ConvNeXt）
- 测试集分层划分（30% 独立测试）
- 分类报告与混淆矩阵可视化

## 快速开始

```bash
pip install -r requirements.txt

# 数据预处理
python src/data/preprocess.py --config config/default.yaml

# 训练
python src/training/train.py --config config/default.yaml

# 评估
python src/evaluation/evaluate.py --config config/default.yaml
```

## 目录结构

```
gjzw/
├── src/
│   ├── models/      # 分类器与主干网络
│   ├── data/        # 数据加载与增强
│   ├── training/    # 训练循环
│   └── evaluation/  # 评估指标
├── config/          # YAML 配置文件
├── data/            # 甲骨文图像数据集
└── outputs/         # 训练结果与可视化
```
