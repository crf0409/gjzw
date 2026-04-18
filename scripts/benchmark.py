#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一的Benchmark脚本 - 替代6个重复的benchmark脚本

使用方法:
    python scripts/benchmark.py --model resnet50
    python scripts/benchmark.py --model vgg16 --epochs 150
    python scripts/benchmark.py --model all  # 运行所有模型
    python scripts/benchmark.py --list  # 列出可用模型
"""

import argparse
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.utils.paths import paths
from src.models.backbones import get_backbone, list_backbones, BACKBONE_REGISTRY


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark different CNN/ViT architectures for ancient character classification'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        help='Model name (e.g., resnet50, vgg16) or "all" to run all models'
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
        '--list', '-l',
        action='store_true',
        help='List available models'
    )
    parser.add_argument(
        '--data-dir', '-d',
        type=str,
        default=None,
        help='Data directory path'
    )

    args = parser.parse_args()

    # 列出可用模型
    if args.list:
        print("\nAvailable models:")
        for name in list_backbones():
            print(f"  - {name}")
        print("\nUse --model <name> to train a specific model")
        print("Use --model all to train all models")
        return

    if not args.model:
        parser.print_help()
        return

    # 确定要训练的模型
    if args.model == 'all':
        models = list_backbones()
    else:
        if args.model not in BACKBONE_REGISTRY:
            print(f"Unknown model: {args.model}")
            print(f"Available models: {list_backbones()}")
            return
        models = [args.model]

    # 确保输出目录存在
    paths.ensure_dirs()

    # 训练每个模型
    for model_name in models:
        print("\n" + "=" * 60)
        print(f"Training: {model_name}")
        print("=" * 60)

        # 加载配置
        config_path = args.config
        if config_path is None:
            # 尝试加载模型特定的配置
            model_config = paths.config_dir / 'experiments' / 'benchmark' / f'{model_name}.yaml'
            if model_config.exists():
                config_path = str(model_config)

        # 构建覆盖配置
        overrides = {'model': {'name': model_name}}

        if args.epochs:
            overrides['training'] = {'epochs': args.epochs}
        if args.batch_size:
            overrides.setdefault('training', {})['batch_size'] = args.batch_size
        if args.data_dir:
            overrides['paths'] = {'data': args.data_dir}

        config = load_config(config_path, overrides)

        # 创建并训练模型
        ClassifierClass = get_backbone(model_name)
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

        print(f"\n{model_name} training completed!")
        print("=" * 60)


if __name__ == "__main__":
    main()
