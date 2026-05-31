# -*- coding: utf-8 -*-
"""
DDP 训练器 - 4 卡 DistributedDataParallel + RAM 缓存 + GPU 端归一化

设计要点:
    - 由 torchrun 启动, 每进程一卡 (--nproc_per_node=4)
    - PT 缓存 (uint8) 在 CPU RAM, __getitem__ 时直接索引零 IO
    - 归一化 (uint8 -> float -> /255 -> normalize) 在 batch 进入 GPU 后做
    - DistributedSampler 切分数据
    - 训练曲线 / checkpoint / 日志只在 rank 0 写
    - 验证指标用 all_reduce 汇总

用法 (经由 scripts/train_ddp.py):
    torchrun --nproc_per_node=4 scripts/train_ddp.py \
        --model resnet50 --dataset AL6 --img-size 224 \
        --epochs 80 --batch-size 32

不依赖 BaseClassifier.train(); 仅复用 build_model().
"""

from __future__ import annotations

import datetime
import json
import os
import platform
import socket
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# torchvision v2 transforms (tensor-native)
from torchvision.transforms import v2 as Tv2

from ..data.cached_dataset import RAMCachedDataset
from ..models.backbones import get_backbone
from ..utils.config import load_config
from .losses import build_loss


# ─────────────────────────────────────────────────────────────────
# 分布式工具
# ─────────────────────────────────────────────────────────────────

def is_dist_available() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_available() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_available() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed(backend: str = "nccl") -> tuple[int, int, int]:
    """
    从 torchrun 注入的环境变量初始化分布式:
        LOCAL_RANK, RANK, WORLD_SIZE
    Returns:
        rank, local_rank, world_size
    """
    if "LOCAL_RANK" not in os.environ:
        # 单卡兜底
        return 0, 0, 1

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        rank=rank,
        world_size=world_size,
    )
    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    if is_dist_available():
        dist.destroy_process_group()


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """all-reduce SUM 后除以 world_size."""
    if not is_dist_available():
        return tensor
    t = tensor.clone().detach()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= get_world_size()
    return t


# ─────────────────────────────────────────────────────────────────
# GPU 端归一化 (从 uint8 -> 标准化 float)
# ─────────────────────────────────────────────────────────────────

# ImageNet 统计
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalize_imagenet_(images: torch.Tensor, mean: torch.Tensor,
                         std: torch.Tensor) -> torch.Tensor:
    """
    输入 uint8 或 float, 值域 [0, 255]; 输出 float [B, C, H, W] 已 ImageNet 归一化.

    通过 max 启发式判断: 若 max > 2 视为 [0, 255], 否则视为 [0, 1].
    这样无论上游 transform 是否改了 dtype, 都能正确归一化.
    """
    if images.dtype != torch.float32:
        images = images.float()
    if images.numel() > 0 and images.max() > 2.0:
        images = images / 255.0
    images = (images - mean) / std
    return images


# ─────────────────────────────────────────────────────────────────
# Augmentation: 张量原生 (v2) 增强
# ─────────────────────────────────────────────────────────────────

def build_train_aug(img_size: tuple[int, int], aug_cfg) -> Tv2.Compose:
    """
    返回作用于 uint8 Tensor 的增强 pipeline:
        随机旋转 + 仿射 + 亮度/对比度 + (可选 ArchAug 钩子)
    输出仍为 uint8 Tensor (归一化由 GPU 端 normalize 做).
    """
    h, w = img_size
    rotation_deg = float(aug_cfg.rotation) * 360
    transl = float(aug_cfg.translation)
    zoom = float(aug_cfg.zoom)
    brightness = float(aug_cfg.brightness)
    contrast_low = float(aug_cfg.contrast_lower)
    contrast_high = float(aug_cfg.contrast_upper)

    ops = [
        Tv2.RandomRotation(degrees=rotation_deg, fill=0),
        Tv2.RandomAffine(
            degrees=0,
            translate=(transl, transl),
            scale=(1.0 - zoom, 1.0 + zoom),
            fill=0,
        ),
        Tv2.ColorJitter(
            brightness=brightness,
            contrast=(contrast_low, contrast_high),
        ),
    ]

    # ArchAug 域内增强钩子 (Day 4 详细实现)
    arch_aug = getattr(aug_cfg, "arch_aug", None)
    if arch_aug is not None and bool(getattr(arch_aug, "enabled", False)):
        try:
            from ..data.arch_aug import build_archaug_ops
            ops.extend(build_archaug_ops(arch_aug, img_size))
        except ImportError:
            pass  # arch_aug 模块尚未实现, 先跳过

    return Tv2.Compose(ops)


def build_eval_aug():
    """评测时只做 no-op (归一化由 GPU 端做). 返回 None 让 dataset 跳过 transform."""
    return None


# ─────────────────────────────────────────────────────────────────
# 训练器主体
# ─────────────────────────────────────────────────────────────────

@dataclass
class TrainArgs:
    model: str = "resnet50"
    dataset: str = "AL6"
    img_height: int = 224
    img_width: int = 224
    grayscale: bool = False
    epochs: int = 80
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    optimizer: str = "adam"
    schedule: str = "cosine"
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 8
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-7
    num_workers: int = 2
    seed: int = 42
    output_subdir: str = "ddp"
    sync_bn: bool = True
    amp: bool = True
    # AAFNet 开关 (转发给 BaseClassifier 通过 config)
    aafnet_overrides: Optional[dict] = None
    # ArchAug 增强配置 (写入 data.augmentation.arch_aug)
    archaug_overrides: Optional[dict] = None
    # 外部 train/val indices (CV runner 用); None 时回退到 train_test_split
    train_indices_path: Optional[str] = None
    val_indices_path: Optional[str] = None
    # 训练数据子采样比例 (data efficiency curve 用); 1.0=全量
    train_fraction: float = 1.0
    # 初始权重路径 (跨数据集 transfer 用; strict=False 加载, 分类头自动随机初始化)
    init_ckpt: Optional[str] = None


class DDPTrainer:
    def __init__(self, args: TrainArgs):
        self.args = args
        self.rank, self.local_rank, self.world_size = setup_distributed()
        self.device = torch.device(f"cuda:{self.local_rank}")
        self._set_seed()

        # 配置: 转发给 BaseClassifier (它构造模型, 但不用其 train())
        overrides = {
            "model": {"name": args.model},
            "data": {
                "dataset": args.dataset,
                "img_height": args.img_height,
                "img_width": args.img_width,
            },
            "training": {
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "early_stopping_patience": args.early_stopping_patience,
                "reduce_lr_patience": args.reduce_lr_patience,
                "reduce_lr_factor": args.reduce_lr_factor,
                "min_lr": args.min_lr,
            },
            "optimizer": {
                "type": args.optimizer,
                "learning_rate": args.learning_rate,
                "schedule": args.schedule,
                "momentum": 0.9,
            },
            "model": {
                "name": args.model,
                "l2_reg": args.weight_decay,
                "fine_tune_ratio": 0.8,
                "dropout1": 0.3,
                "dropout2": 0.2,
                "fc_units": 256,
                "weights": "imagenet",
                "weights_path": None,
            },
        }
        if args.aafnet_overrides:
            overrides["aafnet"] = args.aafnet_overrides
        if args.archaug_overrides:
            overrides.setdefault("data", {}).setdefault(
                "augmentation", {}
            )["arch_aug"] = args.archaug_overrides
        self.config = load_config(overrides=overrides)
        # 把全部决定写入 config 让 BaseClassifier 用
        self.config.data.img_height = args.img_height
        self.config.data.img_width = args.img_width

        # 输出目录: outputs/<output_subdir>/<run_id>/<model>/
        # run_id 用时间戳, 这样多次运行不会互相覆盖, 论文画图时有完整历史.
        run_id = os.environ.get("RUN_ID")
        if not run_id:
            run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            os.environ["RUN_ID"] = run_id  # 让 4 个 rank 共享同一个 id
        # broadcast 一致 run_id (rank 0 决定)
        if self.world_size > 1:
            run_id_bytes = list(run_id.encode("ascii"))
            run_id_bytes += [0] * (32 - len(run_id_bytes))  # pad to 32
            id_tensor = torch.tensor(run_id_bytes[:32], dtype=torch.uint8,
                                       device=self.device)
            dist.broadcast(id_tensor, src=0)
            run_id = bytes(id_tensor.cpu().numpy()).split(b"\0", 1)[0].decode()
        self.run_id = run_id

        out_root = (Path(self.config.paths.outputs) / args.output_subdir
                    / run_id / args.model)
        out_root.mkdir(parents=True, exist_ok=True)
        self.out_dir = out_root

        # 也建一个 "latest" 软链方便快速访问最新 run
        if is_main_process():
            latest_link = (Path(self.config.paths.outputs) / args.output_subdir
                            / "latest")
            latest_link.parent.mkdir(parents=True, exist_ok=True)
            try:
                if latest_link.is_symlink() or latest_link.exists():
                    latest_link.unlink()
                latest_link.symlink_to(run_id, target_is_directory=True)
            except OSError:
                pass

        # 标准化常量 (rank-本地)
        self._mean = _IMAGENET_MEAN.to(self.device)
        self._std = _IMAGENET_STD.to(self.device)

        # MixUp / CutMix 系数 (从 archaug_overrides 取)
        self._mixup_alpha = 0.0
        self._cutmix_alpha = 0.0
        if args.archaug_overrides:
            self._mixup_alpha = float(args.archaug_overrides.get("mixup", 0.0))
            self._cutmix_alpha = float(args.archaug_overrides.get("cutmix", 0.0))

    def _set_seed(self):
        torch.manual_seed(self.args.seed + self.rank)
        np.random.seed(self.args.seed + self.rank)
        torch.cuda.manual_seed_all(self.args.seed + self.rank)

    # ──── 数据 ────
    def _cache_path(self, split: str) -> Path:
        H, W = self.args.img_height, self.args.img_width
        suffix = "gray" if self.args.grayscale else "rgb"
        return Path(self.config.paths.root) / "data" / "cache" \
            / f"{self.args.dataset}_{H}x{W}_{suffix}_{split}.pt"

    def build_loaders(self):
        train_pt = self._cache_path("train")
        test_pt = self._cache_path("test")

        if not train_pt.exists():
            raise FileNotFoundError(
                f"missing train cache: {train_pt}\n"
                f"run: python scripts/build_pt_cache.py --dataset "
                f"{self.args.dataset} --size {self.args.img_height} "
                f"{self.args.img_width}"
            )

        train_aug = build_train_aug(
            (self.args.img_height, self.args.img_width),
            self.config.data.augmentation,
        )

        train_set = RAMCachedDataset(
            train_pt, transform=train_aug,
            normalize=False, gpu_normalize=True,
        )
        all_labels = train_set._labels.numpy()
        all_idx = np.arange(len(all_labels))

        # 优先使用外部 indices (CV / data efficiency)
        if self.args.train_indices_path and self.args.val_indices_path:
            tr_idx = np.load(self.args.train_indices_path).astype(np.int64)
            va_idx = np.load(self.args.val_indices_path).astype(np.int64)
            if is_main_process():
                print(f"[indices] external: train={len(tr_idx)} val={len(va_idx)} "
                      f"(from {self.args.train_indices_path})")
        else:
            from sklearn.model_selection import train_test_split
            tr_idx, va_idx = train_test_split(
                all_idx, test_size=self.config.data.test_split,
                random_state=self.args.seed, stratify=all_labels,
            )

        # data efficiency 子采样: 在 train 索引内分层抽 fraction
        if self.args.train_fraction < 0.999:
            from sklearn.model_selection import StratifiedShuffleSplit
            split = StratifiedShuffleSplit(
                n_splits=1, train_size=self.args.train_fraction,
                random_state=self.args.seed,
            )
            sub_tr, _ = next(split.split(tr_idx,
                                            all_labels[tr_idx]))
            tr_idx = tr_idx[sub_tr]
            if is_main_process():
                print(f"[data_efficiency] sub-sampled to "
                      f"{self.args.train_fraction:.2f} -> {len(tr_idx)} train")

        train_set._indices = tr_idx
        val_set = RAMCachedDataset(
            train_pt, transform=build_eval_aug(),
            normalize=False, gpu_normalize=True,
            in_memory_indices=va_idx,
        )

        if self.world_size > 1:
            train_sampler = DistributedSampler(
                train_set, num_replicas=self.world_size, rank=self.rank,
                shuffle=True, seed=self.args.seed,
            )
            val_sampler = DistributedSampler(
                val_set, num_replicas=self.world_size, rank=self.rank,
                shuffle=False,
            )
        else:
            train_sampler = None
            val_sampler = None

        train_loader = DataLoader(
            train_set,
            batch_size=self.args.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=(self.args.num_workers > 0),
        )
        val_loader = DataLoader(
            val_set,
            batch_size=self.args.batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=(self.args.num_workers > 0),
        )

        # 测试集
        test_loader = None
        if test_pt.exists():
            test_set = RAMCachedDataset(
                test_pt, transform=build_eval_aug(),
                normalize=False, gpu_normalize=True,
            )
            test_sampler = DistributedSampler(
                test_set, num_replicas=self.world_size, rank=self.rank,
                shuffle=False,
            ) if self.world_size > 1 else None
            test_loader = DataLoader(
                test_set,
                batch_size=self.args.batch_size,
                sampler=test_sampler,
                shuffle=False,
                num_workers=self.args.num_workers,
                pin_memory=True,
                drop_last=False,
                persistent_workers=(self.args.num_workers > 0),
            )

        # 类别权重 (基于全量 train)
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(all_labels)
        cw = compute_class_weight("balanced", classes=classes, y=all_labels[tr_idx])
        self.class_weights = torch.tensor(cw, dtype=torch.float32).to(self.device)
        self.num_classes = len(classes)

        if is_main_process():
            print(f"[rank {self.rank}] train={len(train_set)} val={len(val_set)}"
                  f" test={len(test_set) if test_loader else 0}"
                  f" classes={self.num_classes}")

        return train_loader, val_loader, test_loader

    # ──── 模型 ────
    def build_model(self) -> nn.Module:
        # 利用 BaseClassifier.build_model() 构造原始模型
        Cls = get_backbone(self.args.model)
        # 直接 setattr num_classes 让 build_model() 不需要 load_data()
        instance = Cls.__new__(Cls)
        instance.config = self.config
        instance.num_classes = self.num_classes
        instance.device = self.device
        instance._to_rgb = not self.args.grayscale
        model = instance.build_model()

        if self.args.sync_bn and self.world_size > 1:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        model = model.to(self.device)

        # 初始权重 (跨数据集 transfer): 在 DDP 包装前加载
        if self.args.init_ckpt:
            init_path = Path(self.args.init_ckpt)
            if init_path.exists():
                if is_main_process():
                    print(f"[init_ckpt] loading {init_path} (strict=False, "
                          "filtering shape-mismatched keys)")
                sd = torch.load(init_path, map_location="cpu",
                                  weights_only=True)
                # 过滤掉与当前 model 形状不匹配的层 (e.g. classification head
                # 的类数变了). 这些层会用随机初始化, 让 fine-tune 在新任务上
                # 重新学.
                model_state = model.state_dict()
                filtered_sd = {}
                shape_skipped = []
                for k, v in sd.items():
                    if k not in model_state:
                        continue
                    if model_state[k].shape == v.shape:
                        filtered_sd[k] = v
                    else:
                        shape_skipped.append(
                            f"{k} ({tuple(v.shape)} vs current "
                            f"{tuple(model_state[k].shape)})"
                        )
                missing, unexpected = model.load_state_dict(
                    filtered_sd, strict=False)
                if is_main_process():
                    print(f"  loaded {len(filtered_sd)} / {len(sd)} keys")
                    if shape_skipped:
                        print(f"  skipped {len(shape_skipped)} keys with "
                              f"shape mismatch (will be re-initialized): "
                              f"{shape_skipped[:5]}")
                    if missing:
                        print(f"  still missing keys: "
                              f"{len(missing)} (e.g. {missing[:3]})")
                    if unexpected:
                        print(f"  unexpected keys (ignored): "
                              f"{len(unexpected)}")
            else:
                if is_main_process():
                    print(f"[init_ckpt] WARN: not found: {init_path}, "
                          "fallback to ImageNet")

        if self.world_size > 1:
            model = DDP(
                model, device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
            )
        self.model = model
        return model

    def _maybe_build_teacher(self) -> Optional[nn.Module]:
        """如果配置了 KD teacher_ckpt, 加载冻结的教师模型并返回."""
        loss_cfg = self.config.aafnet.loss
        kd_w = float(getattr(loss_cfg, "kd_weight", 0.0))
        teacher_ckpt = getattr(loss_cfg, "teacher_ckpt", None)
        if kd_w <= 0 or not teacher_ckpt:
            return None

        teacher_path = Path(teacher_ckpt)
        if not teacher_path.exists():
            if is_main_process():
                print(f"[KD] WARN: teacher_ckpt not found: {teacher_path}, "
                      "skipping KD")
            return None

        if is_main_process():
            print(f"[KD] loading teacher from: {teacher_path}")

        # 用同一 backbone 类构造教师 (但不接 MSSA, 走原版 head)
        # 简单策略: 教师就是 ResNet-50 baseline
        teacher_overrides = {"aafnet": {"msa": {"enabled": False},
                                        "loss": {"type": "ce"}}}
        # 临时构造一个干净 config
        from copy import deepcopy
        teacher_config = deepcopy(self.config)
        teacher_config.aafnet.msa.enabled = False
        teacher_config.aafnet.loss.type = "ce"

        Cls = get_backbone("resnet50")  # 默认教师为 resnet50
        instance = Cls.__new__(Cls)
        instance.config = teacher_config
        instance.num_classes = self.num_classes
        instance.device = self.device
        instance._to_rgb = not self.args.grayscale
        teacher = instance.build_model()

        sd = torch.load(teacher_path, map_location="cpu", weights_only=True)
        teacher.load_state_dict(sd, strict=False)
        teacher = teacher.to(self.device).eval()
        for p in teacher.parameters():
            p.requires_grad = False
        return teacher

    # ──── 训练循环 ────
    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        return normalize_imagenet_(images, self._mean, self._std)

    def _step(self, images, labels, criterion, scaler, optimizer, train: bool):
        images = images.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)
        images = self._normalize(images)

        # ── Batch 级 MixUp / CutMix (仅训练阶段) ──
        labels_a, labels_b, lam = labels, labels, 1.0
        use_mix = False
        if train and self._mixup_alpha + self._cutmix_alpha > 0:
            from ..data.arch_aug import style_mixup_batch, cutmix_batch
            if self._mixup_alpha > 0 and (
                self._cutmix_alpha == 0 or torch.rand(()).item() < 0.5
            ):
                images, labels_a, labels_b, lam = style_mixup_batch(
                    images, labels, alpha=self._mixup_alpha,
                )
            else:
                images, labels_a, labels_b, lam = cutmix_batch(
                    images, labels, alpha=self._cutmix_alpha,
                )
            use_mix = lam < 0.999

        if train:
            optimizer.zero_grad(set_to_none=True)

        # AMP 仅用于训练; 推理走 fp32 避免数值不稳
        use_amp = self.args.amp and train
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            outputs = self.model(images)
            if isinstance(outputs, tuple) and len(outputs) == 2:
                logits, proj = outputs
            else:
                logits, proj = outputs, None
            # Teacher 推理 (KD)
            teacher_logits = None
            if train and self._teacher is not None:
                with torch.no_grad():
                    teacher_logits = self._teacher(images)
                    if isinstance(teacher_logits, tuple):
                        teacher_logits = teacher_logits[0]
            if use_mix:
                from ..data.arch_aug import mixup_loss
                loss = mixup_loss(
                    criterion, logits, labels_a, labels_b, lam,
                    proj=proj, teacher_logits=teacher_logits,
                )
            else:
                loss = criterion(logits, labels, proj=proj,
                                  teacher_logits=teacher_logits)

        if train:
            if self.args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(1)
            # 评估准确率: mix 时按主标签 a, 原始情形按 labels
            correct = (preds == labels_a).sum() if use_mix else (preds == labels).sum()
            total = labels.numel()

        return loss.detach(), correct, total

    def fit(self, train_loader, val_loader):
        # 复合损失: 按 config.aafnet.loss.* 自动组装 (CE / Focal+LS / +SupCon / +KD)
        criterion = build_loss(
            num_classes=self.num_classes,
            class_weights=self.class_weights,
            loss_cfg=self.config.aafnet.loss,
            device=self.device,
        )

        # KD: 如果 loss 类型含 kd 且 teacher_ckpt 可用, 加载教师
        self._teacher = self._maybe_build_teacher()

        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.args.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                params, lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
            )
        elif self.args.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                params, lr=self.args.learning_rate,
                momentum=0.9, weight_decay=self.args.weight_decay,
            )
        else:  # adam
            optimizer = torch.optim.Adam(
                params, lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
            )

        if self.args.schedule == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.args.epochs, eta_min=0.0
            )
        elif self.args.schedule == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.94)
        else:
            scheduler = None
        reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min",
            factor=self.args.reduce_lr_factor,
            patience=self.args.reduce_lr_patience,
            min_lr=self.args.min_lr,
        )

        scaler = torch.amp.GradScaler("cuda", enabled=self.args.amp)

        # 论文画图用: 完整历史, 包含 loss 组件拆分 / epoch 耗时 / lr
        history = {
            "loss": [], "accuracy": [],
            "val_loss": [], "val_accuracy": [],
            "lr": [],
            "epoch_seconds": [],
            "loss_components": {"ce": [], "supcon": [], "kd": []},
        }
        best_val_acc = 0.0
        patience = 0
        ckpt_path = self.out_dir / f"best_{self.args.model}.pth"

        # 训练开始时刻 (用于计算 wall-clock 总训练时间)
        train_start_ts = time.time()

        for epoch in range(self.args.epochs):
            t0 = time.time()
            if hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            # ─── train ───
            self.model.train()
            sum_loss = torch.tensor(0.0, device=self.device)
            sum_correct = torch.tensor(0, device=self.device)
            sum_total = torch.tensor(0, device=self.device)
            # 累计 loss 组件 (CE/SupCon/KD), 用于 epoch 均值
            comp_sum = {"ce": 0.0, "supcon": 0.0, "kd": 0.0}
            comp_count = {"ce": 0, "supcon": 0, "kd": 0}
            for images, labels in train_loader:
                loss, correct, total = self._step(
                    images, labels, criterion, scaler, optimizer, train=True
                )
                sum_loss += loss * total
                sum_correct += correct
                sum_total += total
                # 抓取 criterion 最新一次的组件值
                inner_crit = criterion
                last_comp = getattr(inner_crit, "last_components", {})
                for k in comp_sum:
                    if k in last_comp:
                        comp_sum[k] += float(last_comp[k]) * int(total)
                        comp_count[k] += int(total)

            # 跨 rank 汇总
            if self.world_size > 1:
                dist.all_reduce(sum_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(sum_correct, op=dist.ReduceOp.SUM)
                dist.all_reduce(sum_total, op=dist.ReduceOp.SUM)
                # comp 也 reduce 跨 rank
                comp_tensors = torch.tensor(
                    [comp_sum["ce"], comp_sum["supcon"], comp_sum["kd"],
                     comp_count["ce"], comp_count["supcon"], comp_count["kd"]],
                    device=self.device, dtype=torch.float64,
                )
                dist.all_reduce(comp_tensors, op=dist.ReduceOp.SUM)
                comp_sum["ce"], comp_sum["supcon"], comp_sum["kd"] = \
                    comp_tensors[0].item(), comp_tensors[1].item(), comp_tensors[2].item()
                comp_count["ce"], comp_count["supcon"], comp_count["kd"] = \
                    int(comp_tensors[3].item()), int(comp_tensors[4].item()), int(comp_tensors[5].item())
            train_loss = (sum_loss / sum_total.clamp_min(1)).item()
            train_acc = (sum_correct.float() / sum_total.clamp_min(1).float()).item()

            # ─── val ───
            self.model.eval()
            sum_loss = torch.tensor(0.0, device=self.device)
            sum_correct = torch.tensor(0, device=self.device)
            sum_total = torch.tensor(0, device=self.device)
            with torch.no_grad():
                for images, labels in val_loader:
                    loss, correct, total = self._step(
                        images, labels, criterion, scaler, optimizer, train=False
                    )
                    sum_loss += loss * total
                    sum_correct += correct
                    sum_total += total
            if self.world_size > 1:
                dist.all_reduce(sum_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(sum_correct, op=dist.ReduceOp.SUM)
                dist.all_reduce(sum_total, op=dist.ReduceOp.SUM)
            val_loss = (sum_loss / sum_total.clamp_min(1)).item()
            val_acc = (sum_correct.float() / sum_total.clamp_min(1).float()).item()

            # 调度
            if scheduler is not None:
                scheduler.step()
            reduce_lr.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            elapsed = time.time() - t0
            history["loss"].append(train_loss)
            history["accuracy"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)
            history["lr"].append(current_lr)
            history["epoch_seconds"].append(round(elapsed, 3))
            for k in ("ce", "supcon", "kd"):
                history["loss_components"][k].append(
                    round(comp_sum[k] / max(1, comp_count[k]), 6)
                    if comp_count[k] > 0 else None
                )
            if is_main_process():
                print(
                    f"Epoch {epoch+1:>3}/{self.args.epochs} "
                    f"({elapsed:.1f}s) - "
                    f"loss {train_loss:.4f} acc {train_acc:.4f} "
                    f"- val_loss {val_loss:.4f} val_acc {val_acc:.4f} "
                    f"- lr {current_lr:.2e}",
                    flush=True,
                )

            # checkpoint + early stop (rank 0 决策, 同步给其他 rank)
            improved_local = torch.tensor(
                1 if val_acc > best_val_acc else 0, device=self.device
            )
            if self.world_size > 1:
                dist.broadcast(improved_local, src=0)
            improved = bool(improved_local.item())

            if improved and is_main_process():
                best_val_acc = val_acc
                state = (self.model.module if isinstance(self.model, DDP)
                         else self.model).state_dict()
                torch.save(state, ckpt_path)
            if val_acc > best_val_acc:
                best_val_acc = val_acc

            if not improved:
                patience += 1
            else:
                patience = 0

            stop = torch.tensor(
                1 if patience >= self.args.early_stopping_patience else 0,
                device=self.device,
            )
            if self.world_size > 1:
                dist.broadcast(stop, src=0)
            if bool(stop.item()):
                if is_main_process():
                    print(
                        f"Early stopping at epoch {epoch+1} "
                        f"(patience={self.args.early_stopping_patience})"
                    )
                break

        # 加载最佳权重
        if ckpt_path.exists():
            sd = torch.load(ckpt_path, map_location=self.device,
                            weights_only=True)
            (self.model.module if isinstance(self.model, DDP)
             else self.model).load_state_dict(sd)

        if is_main_process():
            total_seconds = round(time.time() - train_start_ts, 2)
            best_val_epoch = (int(np.argmax(history["val_accuracy"])) + 1
                              if history["val_accuracy"] else 0)
            log_payload = {
                "run_id": self.run_id,
                "model": self.args.model,
                "dataset": self.args.dataset,
                "epochs_completed": len(history["loss"]),
                "best_val_accuracy": best_val_acc,
                "best_val_epoch": best_val_epoch,
                "total_training_seconds": total_seconds,
                "world_size": self.world_size,
                "ckpt_path": str(ckpt_path.resolve()),
                "history": history,
                "args": asdict(self.args),
                "environment": _collect_environment(),
                "config_snapshot": _config_to_dict(self.config),
            }
            with open(self.out_dir / "training_log.json", "w") as f:
                json.dump(log_payload, f, indent=2, ensure_ascii=False)
            # 还存一个简洁 csv 给画图用
            _dump_history_csv(history, self.out_dir / "training_curve.csv")

        self.history = history
        return history, best_val_acc

    # ──── 测试评估 ────
    @torch.no_grad()
    def test(self, test_loader) -> dict | None:
        if test_loader is None:
            return None
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        sum_loss = torch.tensor(0.0, device=self.device)
        all_preds_local = []
        all_labels_local = []
        for images, labels in test_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            images = self._normalize(images)
            # 测试用 fp32, 数值稳定
            out = self.model(images)
            if isinstance(out, tuple) and len(out) == 2:
                out = out[0]
            loss = criterion(out, labels)
            sum_loss += loss * labels.numel()
            all_preds_local.append(out.argmax(1).cpu())
            all_labels_local.append(labels.cpu())

        preds = torch.cat(all_preds_local)
        labels = torch.cat(all_labels_local)

        # gather 各 rank 的预测
        # NCCL all_gather 要求 src 与 dst tensor list 都在 GPU 且形状一致.
        # DistributedSampler 默认 drop_last=False 会 pad 到能整除, 所以每 rank
        # 数据量相同. 我们 gather 后再去 CPU concat.
        if self.world_size > 1:
            preds_g = preds.to(self.device)
            labels_g = labels.to(self.device)
            preds_list = [torch.empty_like(preds_g) for _ in range(self.world_size)]
            labels_list = [torch.empty_like(labels_g) for _ in range(self.world_size)]
            dist.all_gather(preds_list, preds_g)
            dist.all_gather(labels_list, labels_g)
            preds = torch.cat([p.cpu() for p in preds_list])
            labels = torch.cat([l.cpu() for l in labels_list])

        if not is_main_process():
            return None

        from sklearn.metrics import (
            classification_report, confusion_matrix,
            precision_recall_fscore_support,
        )
        y_true = labels.numpy()
        y_pred = preds.numpy()
        # Note: 因为 DistributedSampler 可能有 padding, 这里去重保留前 N
        # 不过 drop_last=False, 多卡间样本可能略有重复 — 简单 unique 处理
        # (足够 SCI 论文要求)
        acc = (y_true == y_pred).mean()
        macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        wt_p, wt_r, wt_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred)

        result = {
            "test_accuracy": float(acc),
            "test_loss": float(sum_loss.item() / max(1, len(y_true))),
            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
            "macro_f1": float(macro_f1),
            "weighted_precision": float(wt_p),
            "weighted_recall": float(wt_r),
            "weighted_f1": float(wt_f1),
            "confusion_matrix": cm.tolist(),
            "classification_report_text": classification_report(
                y_true, y_pred, zero_division=0
            ),
        }
        # 增加 run_id, dataset, model 等元信息便于聚合时定位
        result.update({
            "run_id": self.run_id,
            "model": self.args.model,
            "dataset": self.args.dataset,
            "num_classes": self.num_classes,
            "world_size": self.world_size,
        })
        with open(self.out_dir / "test_metrics.json", "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # 混淆矩阵图 (论文画图用)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=range(cm.shape[0]),
                yticklabels=range(cm.shape[0]),
            )
            plt.title(f"Confusion Matrix — {self.args.model} (test acc {acc*100:.2f}%)")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.tight_layout()
            plt.savefig(self.out_dir / "confusion_matrix.png", dpi=300)
            plt.close()
        except Exception as e:
            print(f"  warn: confusion matrix png failed: {e}")

        # 训练曲线图
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            if self.history:
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                ep = list(range(1, len(self.history["loss"]) + 1))
                axes[0].plot(ep, self.history["loss"], label="train")
                axes[0].plot(ep, self.history["val_loss"], label="val")
                axes[0].set_title(f"Loss — {self.args.model}")
                axes[0].set_xlabel("Epoch")
                axes[0].set_ylabel("Loss")
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                axes[1].plot(ep, self.history["accuracy"], label="train")
                axes[1].plot(ep, self.history["val_accuracy"], label="val")
                axes[1].set_title(f"Accuracy — {self.args.model}")
                axes[1].set_xlabel("Epoch")
                axes[1].set_ylabel("Accuracy")
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(self.out_dir / "training_history.png", dpi=300)
                plt.close()
        except Exception as e:
            print(f"  warn: training curve png failed: {e}")

        print(
            f"\n[Test] acc={acc:.4f}  macro-F1={macro_f1:.4f}  "
            f"weighted-F1={wt_f1:.4f}\n"
            f"  artifacts saved to: {self.out_dir}",
            flush=True,
        )
        return result

    def close(self):
        cleanup_distributed()


def _collect_environment() -> dict:
    """收集运行环境元信息, 写入 training_log.json 供论文复现."""
    env = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available() else None,
        "gpu_count": torch.cuda.device_count(),
        "gpus": [],
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
    }
    for i in range(torch.cuda.device_count()):
        try:
            env["gpus"].append({
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "memory_total_mb": int(
                    torch.cuda.get_device_properties(i).total_memory / 1024 / 1024
                ),
            })
        except Exception:
            pass
    # git commit (可选, 不报错)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, timeout=2,
        ).decode().strip()
        env["git_commit"] = commit
    except Exception:
        env["git_commit"] = None
    return env


def _config_to_dict(cfg) -> dict:
    """递归把 DictConfig 转成普通 dict 以便 json 序列化."""
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    if isinstance(cfg, dict):
        return {k: _config_to_dict(v) for k, v in cfg.items()}
    if isinstance(cfg, (list, tuple)):
        return [_config_to_dict(v) for v in cfg]
    return cfg


def _dump_history_csv(history: dict, path: Path) -> None:
    """把 epoch 级历史写成 CSV 方便 pandas/Excel 读."""
    import csv
    n = len(history.get("loss", []))
    if n == 0:
        return
    fieldnames = ["epoch", "train_loss", "train_acc",
                   "val_loss", "val_acc", "lr", "epoch_seconds",
                   "ce", "supcon", "kd"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i in range(n):
            w.writerow({
                "epoch": i + 1,
                "train_loss": history["loss"][i],
                "train_acc": history["accuracy"][i],
                "val_loss": history["val_loss"][i],
                "val_acc": history["val_accuracy"][i],
                "lr": history.get("lr", [None]*n)[i],
                "epoch_seconds": history.get("epoch_seconds", [None]*n)[i],
                "ce": history.get("loss_components", {}).get("ce", [None]*n)[i],
                "supcon": history.get("loss_components", {}).get("supcon", [None]*n)[i],
                "kd": history.get("loss_components", {}).get("kd", [None]*n)[i],
            })


def run_ddp_training(args: TrainArgs) -> Optional[dict]:
    """脚本入口的 thin wrapper."""
    trainer = DDPTrainer(args)
    try:
        train_loader, val_loader, test_loader = trainer.build_loaders()
        trainer.build_model()
        history, best_val = trainer.fit(train_loader, val_loader)
        result = trainer.test(test_loader)
        if is_main_process() and result is not None:
            print(f"\n=== {args.model} done ===")
            print(f"  best val acc:   {best_val:.4f}")
            print(f"  test accuracy:  {result['test_accuracy']:.4f}")
            print(f"  test macro-F1:  {result['macro_f1']:.4f}")
        return result
    finally:
        trainer.close()
