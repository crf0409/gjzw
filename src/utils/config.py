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


def _lookup_dotted(cfg: dict, dotted: str):
    """从配置字典按 'data.dataset' 形式查找."""
    cur = cfg
    for part in dotted.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def resolve_paths(config: dict, root: Path = None) -> dict:
    """解析配置中的占位符:
       - ${PROJECT_ROOT} / ${ROOT} -> 项目根目录
       - ${data.dataset} 等 -> 同一配置树中按点号路径查找
    多次迭代直到稳定 (最多 5 次)。
    """
    root = root or paths.root

    def resolve(value, full_cfg):
        if isinstance(value, str):
            value = value.replace('${PROJECT_ROOT}', str(root))
            value = value.replace('${ROOT}', str(root))
            # 替换 ${a.b.c} 形式
            import re
            def sub(m):
                key = m.group(1)
                got = _lookup_dotted(full_cfg, key)
                return str(got) if got is not None else m.group(0)
            value = re.sub(r'\$\{([a-zA-Z0-9_.]+)\}', sub, value)
            if value.startswith('./'):
                value = str(root / value[2:])
        elif isinstance(value, dict):
            value = {k: resolve(v, full_cfg) for k, v in value.items()}
        elif isinstance(value, list):
            value = [resolve(v, full_cfg) for v in value]
        return value

    cfg = config
    for _ in range(5):
        new_cfg = resolve(cfg, cfg)
        if new_cfg == cfg:
            break
        cfg = new_cfg
    return cfg


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
            'name': 'ancient-architecture-classification',
            'seed': 42,
        },
        'paths': {
            'root': str(paths.root),
            # 注意: 实际 data 路径由 ${data.dataset} 插值, 见 config/default.yaml
            'data': '${PROJECT_ROOT}/data/processed/${data.dataset}',
            'weights': str(paths.weights_dir),
            'outputs': str(paths.outputs_dir),
        },
        'data': {
            'dataset': 'AL6',     # AL6 | ASP | AS25
            'img_height': None,   # 自动检测
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
                'arch_aug': {
                    'enabled': False,
                    'perspective': 0.0,
                    'arch_occlusion': 0.0,
                    'weather': 0.0,
                    'mixup': 0.0,
                    'cutmix': 0.0,
                    'randaugment': False,
                },
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
            'schedule': 'cosine',  # cosine | exponential | constant | plateau
        },
        # AAFNet 框架开关（消融时切此处）
        'aafnet': {
            'msa': {
                'enabled': False,
                'fused_dim': 512,
                'style_dim': 128,
                'placement': 'after_layer4',
            },
            'loss': {
                'type': 'ce',
                'focal_gamma': 2.0,
                'label_smoothing': 0.05,
                'supcon_weight': 0.0,
                'supcon_temp': 0.07,
                'proj_dim': 128,
                'kd_weight': 0.0,
                'kd_temp': 4.0,
                'teacher_ckpt': None,
            },
            'ensemble': {
                'enabled': False,
                'mode': 'soft_vote',
                'members': [],
            },
        },
        'cv': {
            'enabled': False,
            'n_folds': 5,
            'seeds': [42, 1337, 2024],
        },
    }

    # 优先从 config/default.yaml 加载（如果存在），与硬编码默认 deep_merge
    auto_default_yaml = paths.config_dir / 'default.yaml'
    if auto_default_yaml.exists() and config_path is None:
        try:
            with open(auto_default_yaml, 'r', encoding='utf-8') as f:
                yaml_default = yaml.safe_load(f) or {}
            default_config = deep_merge(default_config, yaml_default)
        except Exception as e:
            print(f"Warning: failed to load {auto_default_yaml}: {e}")

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
