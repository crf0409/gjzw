# -*- coding: utf-8 -*-
"""INR 子注册器 — 占位, 当前直接通过 nn.Module 接口使用, 不接 BaseClassifier."""


def register_inr_backbone(name):
    def decorator(cls):
        return cls
    return decorator
