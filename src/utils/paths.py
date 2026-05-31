# -*- coding: utf-8 -*-
"""路径管理工具"""

import os
from pathlib import Path


def get_project_root():
    """获取项目根目录"""
    current = Path(__file__).resolve()
    # 向上查找包含 src/ 目录的路径
    for parent in [current] + list(current.parents):
        if (parent / 'src').exists() and (parent / 'config').exists():
            return parent
    # 回退到 src 的上级目录
    return current.parent.parent.parent


PROJECT_ROOT = get_project_root()


class PathManager:
    """路径管理器"""

    def __init__(self, root=None):
        self.root = Path(root) if root else PROJECT_ROOT

    @property
    def config_dir(self):
        return self.root / 'config'

    @property
    def data_dir(self):
        return self.root / 'data'

    @property
    def raw_data_dir(self):
        return self.data_dir / 'raw'

    @property
    def processed_data_dir(self):
        return self.data_dir / 'processed'

    @property
    def images_dir(self):
        return self.processed_data_dir / 'images'

    @property
    def features_dir(self):
        return self.processed_data_dir / 'features'

    @property
    def mappings_dir(self):
        return self.processed_data_dir / 'mappings'

    @property
    def weights_dir(self):
        return self.data_dir / 'weights'

    @property
    def outputs_dir(self):
        return self.root / 'outputs'

    @property
    def models_dir(self):
        return self.outputs_dir / 'models'

    @property
    def logs_dir(self):
        return self.outputs_dir / 'logs'

    @property
    def figures_dir(self):
        return self.outputs_dir / 'figures'

    def ensure_dirs(self):
        """确保所有目录存在"""
        dirs = [
            self.config_dir,
            self.raw_data_dir,
            self.images_dir,
            self.features_dir,
            self.mappings_dir,
            self.weights_dir,
            self.models_dir,
            self.logs_dir,
            self.figures_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


# 默认路径管理器实例
paths = PathManager()
