# -*- coding: utf-8 -*-
"""配置管理模块"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from .paths import paths


class DictConfig:
    """支持点号访问的配置字典"""

    def __init__(self, data: dict):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, DictConfig(value))
            else:
                setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def to_dict(self) -> dict:
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, DictConfig):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


def deep_merge(base: dict, override: dict) -> dict:
    """深度合并两个字典"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def resolve_paths(config: dict, root: Path = None) -> dict:
    """解析配置中的路径占位符"""
    root = root or paths.root

    def resolve(value):
        if isinstance(value, str):
            # 替换路径占位符
            value = value.replace('${PROJECT_ROOT}', str(root))
            value = value.replace('${ROOT}', str(root))
            # 解析相对路径
            if value.startswith('./'):
                value = str(root / value[2:])
        elif isinstance(value, dict):
            value = {k: resolve(v) for k, v in value.items()}
        elif isinstance(value, list):
            value = [resolve(v) for v in value]
        return value

    return resolve(config)


def load_config(config_path: str = None, overrides: dict = None) -> DictConfig:
    """
    加载配置文件

    Args:
        config_path: 配置文件路径，如果为None则加载默认配置
        overrides: 覆盖配置的字典

    Returns:
        DictConfig: 配置对象
    """
    # 默认配置
    default_config = {
        'project': {
            'name': 'ancient-character-classification',
            'seed': 42,
        },
        'paths': {
            'root': str(paths.root),
            'data': str(paths.images_dir),
            'weights': str(paths.weights_dir),
            'outputs': str(paths.outputs_dir),
        },
        'data': {
            'img_height': None,  # 自动检测
            'img_width': None,
            'test_split': 0.3,
            'train_mapping': 'train_mapping.csv',
            'test_mapping': 'test_mapping.csv',
            'augmentation': {
                'rotation': 0.15,
                'translation': 0.1,
                'zoom': 0.15,
                'brightness': 0.1,
                'contrast_lower': 0.9,
                'contrast_upper': 1.1,
            },
        },
        'training': {
            'batch_size': 32,
            'epochs': 100,
            'early_stopping_patience': 15,
            'reduce_lr_patience': 8,
            'reduce_lr_factor': 0.5,
            'min_lr': 1e-7,
        },
        'model': {
            'name': 'resnet50',
            'weights': 'imagenet',
            'weights_path': None,
            'fine_tune_ratio': 0.8,
            'dropout1': 0.3,
            'dropout2': 0.2,
            'fc_units': 256,
            'l2_reg': 0.0001,
        },
        'optimizer': {
            'type': 'adam',
            'learning_rate': 0.0001,
            'momentum': 0.9,
            'schedule': 'cosine',  # 'cosine', 'exponential', 'constant'
        },
    }

    # 如果指定了配置文件，加载并合并
    if config_path:
        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                file_config = yaml.safe_load(f) or {}
            default_config = deep_merge(default_config, file_config)

    # 应用覆盖配置
    if overrides:
        default_config = deep_merge(default_config, overrides)

    # 解析路径
    default_config = resolve_paths(default_config)

    return DictConfig(default_config)


# 默认配置实例
default_config = load_config()
