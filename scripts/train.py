#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
主训练脚本 - 训练图像分类模型

使用方法:
    python scripts/train.py --model inception_resnet_v2
    python scripts/train.py --model resnet50 --epochs 100 --batch-size 32
    python scripts/train.py --config config/my_config.yaml
"""

import argparse
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.utils.paths import paths
from src.models.backbones import get_backbone, list_backbones


def main():
    parser = argparse.ArgumentParser(
        description='Train ancient character image classifier'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='inception_resnet_v2',
        help='Model architecture (default: inception_resnet_v2)'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to config file'
    )
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=None,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=None,
        help='Batch size'
    )
    parser.add_argument(
        '--data-dir', '-d',
        type=str,
        default=None,
        help='Data directory path'
    )
    parser.add_argument(
        '--weights', '-w',
        type=str,
        default=None,
        help='Path to pretrained weights'
    )
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List available models'
    )

    args = parser.parse_args()

    if args.list_models:
        print("\nAvailable models:")
        for name in list_backbones():
            print(f"  - {name}")
        return

    # 确保输出目录存在
    paths.ensure_dirs()

    print("=" * 60)
    print("Ancient Character Image Classification Training")
    print(f"Architecture: {args.model}")
    print("=" * 60)

    # 构建覆盖配置
    overrides = {'model': {'name': args.model}}

    if args.epochs:
        overrides['training'] = {'epochs': args.epochs}
    if args.batch_size:
        overrides.setdefault('training', {})['batch_size'] = args.batch_size
    if args.data_dir:
        overrides['paths'] = {'data': args.data_dir}
    if args.weights:
        overrides['model']['weights_path'] = args.weights

    # 加载配置
    config = load_config(args.config, overrides)

    # 创建分类器
    ClassifierClass = get_backbone(args.model)
    classifier = ClassifierClass(config)

    # 加载数据
    X_train, X_val, y_train, y_val = classifier.load_data()

    # 构建模型
    classifier.build_model()

    # 训练
    classifier.train(X_train, X_val, y_train, y_val)

    # 绘制训练历史
    classifier.plot_training_history()

    # 评估测试集
    classifier.evaluate_test_set()

    # 预测可视化
    classifier.predict_sample_images()

    # 保存模型
    classifier.save_model()

    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
