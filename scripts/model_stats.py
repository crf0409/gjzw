#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""统计所有模型的参数量和计算量 (PyTorch)"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from src.utils.config import load_config
from src.models.backbones import get_backbone, list_backbones

# 模型配置：(名称, 输入高度, 输入宽度)
MODEL_CONFIGS = [
    ('resnet50', 224, 224),
    ('vgg16', 224, 224),
    ('vgg19', 224, 224),
    ('inception_v3', 299, 299),
    ('inception_resnet_v2', 299, 299),
    ('efficientnet_b3', 300, 300),
    ('mobilenet_v3', 224, 224),
    ('vit_b16', 224, 224),
    ('custom_mlp', 224, 224),
]

NUM_CLASSES = 6


def count_params(model):
    """统计参数量"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, non_trainable


def get_flops(model, input_shape):
    """计算模型的FLOPs（使用 thop，若不可用则跳过）"""
    try:
        from thop import profile
        dummy = torch.randn(1, *input_shape)
        flops, _ = profile(model, inputs=(dummy,), verbose=False)
        return flops
    except Exception:
        return None


def format_number(num):
    """格式化数字显示"""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(int(num))


def main():
    print("=" * 80)
    print("模型参数量与计算量统计")
    print("=" * 80)
    print()

    results = []

    for name, height, width in MODEL_CONFIGS:
        # custom_mlp 使用 1 通道
        channels = 1 if name == 'custom_mlp' else 3
        input_shape = (channels, height, width)
        print(f"正在分析 {name}...", end=" ", flush=True)

        try:
            # 加载配置并设置模型参数
            config = load_config(overrides={
                'model': {'name': name},
                'data': {'img_height': height, 'img_width': width},
            })

            # 获取已注册的分类器类并实例化
            ClassifierClass = get_backbone(name)
            classifier = ClassifierClass(config)
            classifier.num_classes = NUM_CLASSES

            # 构建模型
            model = classifier.build_model()
            model.eval()

            trainable, non_trainable = count_params(model)
            total = trainable + non_trainable
            flops = get_flops(model, input_shape)

            results.append({
                'name': name,
                'input': f"{channels}x{height}x{width}",
                'trainable': trainable,
                'non_trainable': non_trainable,
                'total': total,
                'flops': flops
            })
            print("完成")

        except Exception as e:
            print(f"失败: {e}")
            results.append({
                'name': name,
                'input': f"{channels}x{height}x{width}",
                'trainable': 0,
                'non_trainable': 0,
                'total': 0,
                'flops': None,
                'error': str(e)
            })

    # 打印结果表格
    print()
    print("=" * 80)
    print(f"{'模型':<25} {'输入尺寸':<15} {'可训练参数':<12} {'总参数':<12} {'FLOPs':<12}")
    print("-" * 80)

    for r in results:
        flops_str = format_number(r['flops']) if r['flops'] else "N/A"
        print(f"{r['name']:<25} {r['input']:<15} {format_number(r['trainable']):<12} {format_number(r['total']):<12} {flops_str:<12}")

    print("=" * 80)

    # 按参数量排序
    print()
    print("按总参数量排序:")
    print("-" * 40)
    sorted_results = sorted(results, key=lambda x: x['total'], reverse=True)
    for i, r in enumerate(sorted_results, 1):
        print(f"{i}. {r['name']}: {format_number(r['total'])}")


if __name__ == "__main__":
    main()
