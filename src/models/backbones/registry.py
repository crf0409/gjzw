# -*- coding: utf-8 -*-
"""模型注册表 - 支持通过名称动态创建模型"""

BACKBONE_REGISTRY = {}


def register_backbone(name):
    """
    装饰器：注册模型到注册表

    Usage:
        @register_backbone('resnet50')
        class ResNet50Classifier(BaseClassifier):
            ...
    """
    def decorator(cls):
        BACKBONE_REGISTRY[name] = cls
        return cls
    return decorator


def get_backbone(name):
    """
    获取已注册的模型类

    Args:
        name: 模型名称

    Returns:
        模型类

    Raises:
        ValueError: 如果模型名称未注册
    """
    if name not in BACKBONE_REGISTRY:
        available = list(BACKBONE_REGISTRY.keys())
        raise ValueError(f"Unknown backbone: {name}. Available: {available}")
    return BACKBONE_REGISTRY[name]


def list_backbones():
    """列出所有已注册的模型"""
    return list(BACKBONE_REGISTRY.keys())
