#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DDP 训练入口 - 由 torchrun 启动

用法:
    torchrun --standalone --nproc_per_node=4 scripts/train_ddp.py \
        --model resnet50 --dataset AL6 --img-size 224 \
        --epochs 80 --batch-size 32

或经由 scripts/train_ddp.sh 包装.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.ddp_trainer import TrainArgs, run_ddp_training


def parse_args() -> TrainArgs:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True,
                   help="backbone 名 (resnet50, efficientnet_b3, ...)")
    p.add_argument("--dataset", type=str, default="AL6",
                   help="任意 data/processed/<name>/ 目录 (AL6/ASP/AS25 + _clean/_strict 等变体)")
    p.add_argument("--img-size", type=int, nargs=2, default=[224, 224],
                   metavar=("H", "W"))
    p.add_argument("--grayscale", action="store_true")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", "--learning-rate", dest="learning_rate",
                   type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--optimizer", choices=["adam", "adamw", "sgd"],
                   default="adam")
    p.add_argument("--schedule", choices=["cosine", "exponential", "constant"],
                   default="cosine")
    p.add_argument("--early-stopping-patience", type=int, default=15)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-subdir", type=str, default="ddp")
    p.add_argument("--no-sync-bn", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    # AAFNet 开关
    p.add_argument("--mssa", action="store_true",
                   help="启用 MSSA 多尺度模块")
    p.add_argument("--loss-type",
                   choices=["ce", "focal", "focalls",
                            "focalls_supcon", "focalls_supcon_kd"],
                   default="ce")
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--supcon-weight", type=float, default=0.5,
                   help="启用 supcon 时的权重 (loss-type 含 supcon 时生效)")
    p.add_argument("--supcon-temp", type=float, default=0.07)
    p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument("--kd-weight", type=float, default=1.0,
                   help="启用 kd 时的权重 (loss-type 含 kd 时生效)")
    p.add_argument("--kd-temp", type=float, default=4.0)
    p.add_argument("--teacher-ckpt", type=str, default=None,
                   help="教师模型权重路径 (用于 KD)")
    # ArchAug 域内增强
    p.add_argument("--archaug", action="store_true",
                   help="启用 ArchAug (单独开关; 需要至少配一个具体增强 >0)")
    p.add_argument("--perspective", type=float, default=0.3,
                   help="PerspectiveJitter distortion_scale, 0=关")
    p.add_argument("--arch-occlusion", type=float, default=0.5,
                   help="ArchOcclusion 概率, 0=关")
    p.add_argument("--weather", type=float, default=0.3,
                   help="WeatherSim 概率, 0=关")
    p.add_argument("--mixup", type=float, default=0.0,
                   help="MixUp alpha, 0=关")
    p.add_argument("--cutmix", type=float, default=0.0,
                   help="CutMix alpha, 0=关")
    p.add_argument("--randaugment", action="store_true")
    p.add_argument("--gauss-noise", type=float, default=0.0,
                   help="GaussianNoise 概率, 0=关 (针对噪声鲁棒性)")
    # 外部 indices (CV runner 用)
    p.add_argument("--train-indices-path", default=None,
                   help="外部 train indices .npy 路径 (CV / data efficiency 用)")
    p.add_argument("--val-indices-path", default=None,
                   help="外部 val indices .npy 路径")
    p.add_argument("--train-fraction", type=float, default=1.0,
                   help="训练数据子采样比例 (data efficiency curve, 1.0=全量)")
    p.add_argument("--init-ckpt", default=None,
                   help="初始权重 .pth (跨数据集 transfer, strict=False 加载)")

    a = p.parse_args()

    aafnet_overrides = None
    if a.mssa or a.loss_type != "ce":
        aafnet_overrides = {}
        if a.mssa:
            aafnet_overrides["msa"] = {"enabled": True}
        if a.loss_type != "ce":
            aafnet_overrides["loss"] = {
                "type": a.loss_type,
                "focal_gamma": a.focal_gamma,
                "label_smoothing": a.label_smoothing,
                "supcon_weight": a.supcon_weight if "supcon" in a.loss_type else 0.0,
                "supcon_temp": a.supcon_temp,
                "proj_dim": a.proj_dim,
                "kd_weight": a.kd_weight if "kd" in a.loss_type else 0.0,
                "kd_temp": a.kd_temp,
                "teacher_ckpt": a.teacher_ckpt,
            }

    # ArchAug 配置写到 data.augmentation.arch_aug, 由 build_train_aug 读取
    archaug_overrides = None
    if (a.archaug or a.mixup > 0 or a.cutmix > 0 or a.randaugment
            or a.gauss_noise > 0):
        archaug_overrides = {
            "enabled": True,
            "perspective": a.perspective if a.archaug else 0.0,
            "arch_occlusion": a.arch_occlusion if a.archaug else 0.0,
            "weather": a.weather if a.archaug else 0.0,
            "mixup": a.mixup,
            "cutmix": a.cutmix,
            "randaugment": a.randaugment,
            "gauss_noise": a.gauss_noise,
        }

    return TrainArgs(
        model=a.model,
        dataset=a.dataset,
        img_height=a.img_size[0],
        img_width=a.img_size[1],
        grayscale=a.grayscale,
        epochs=a.epochs,
        batch_size=a.batch_size,
        learning_rate=a.learning_rate,
        weight_decay=a.weight_decay,
        optimizer=a.optimizer,
        schedule=a.schedule,
        early_stopping_patience=a.early_stopping_patience,
        num_workers=a.num_workers,
        seed=a.seed,
        output_subdir=a.output_subdir,
        sync_bn=not a.no_sync_bn,
        amp=not a.no_amp,
        aafnet_overrides=aafnet_overrides,
        archaug_overrides=archaug_overrides,
        train_indices_path=a.train_indices_path,
        val_indices_path=a.val_indices_path,
        train_fraction=a.train_fraction,
        init_ckpt=a.init_ckpt,
    )


def main():
    args = parse_args()
    run_ddp_training(args)


if __name__ == "__main__":
    main()
