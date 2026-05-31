#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
纹理分析 + 局部放大对比图

输出:  outputs/paper/figures/
  1. texture_detail_zoom_{class}.png  — 每类样本的局部放大细节对比
  2. texture_cross_model_zoom.png     — 跨模型同区域 Grad-CAM 细节放大对比
  3. texture_feature_analysis.png     — 纹理特征提取分析（边缘/梯度/频率）
  4. texture_multi_class_comparison.png — 六类建筑纹理特征对比
  5. texture_attention_zoom.png       — 注意力区域局部放大对比（最佳 vs 最差模型）
"""

import argparse
import copy
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from scipy import ndimage

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.utils.paths import paths
from src.models.backbones import get_backbone

# ── 配置 ──
PAPER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'outputs', 'paper')
FIG_DIR = os.path.join(PAPER_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

DISPLAY_NAMES = {
    'custom_mlp': 'Custom MLP',
    'resnet50': 'ResNet-50',
    'vgg16': 'VGG-16',
    'vgg19': 'VGG-19',
    'inception_v3': 'Inception V3',
    'inception_resnet_v2': 'Inception-ResNet V2',
    'efficientnet_b3': 'EfficientNet-B3',
    'mobilenet_v3': 'MobileNet V3',
    'vit_b16': 'ViT-B/16',
}

CLASS_NAMES = {
    0: 'Class 1', 1: 'Class 2', 2: 'Class 3',
    3: 'Class 4', 4: 'Class 5', 5: 'Class 6',
}

# 模型对应的输入尺寸
MODEL_CONFIGS = [
    ('resnet50', 224, 224),
    ('vgg16', 224, 224),
    ('vgg19', 224, 224),
    ('inception_v3', 299, 299),
    ('inception_resnet_v2', 299, 299),
    ('efficientnet_b3', 300, 300),
    ('mobilenet_v3', 224, 224),
    ('vit_b16', 224, 224),
]


# ══════════════════════════════════════════════════════════════════════
#  Grad-CAM 实现（与 paper_model_analysis.py 一致）
# ══════════════════════════════════════════════════════════════════════

class GradCAM:
    """Grad-CAM — 带 hook 自动清理 + inplace 安全"""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._hooks = []

    def _register_hooks(self):
        def fwd_hook(module, input, output):
            self.activations = output

        def bwd_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        h1 = self.target_layer.register_forward_hook(fwd_hook)
        h2 = self.target_layer.register_full_backward_hook(bwd_hook)
        self._hooks = [h1, h2]

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def generate(self, input_tensor, target_class=None):
        self._register_hooks()
        _inplace_states = {}
        try:
            self.model.eval()

            for name, m in self.model.named_modules():
                if isinstance(m, nn.ReLU) and m.inplace:
                    _inplace_states[name] = True
                    m.inplace = False

            input_tensor = input_tensor.requires_grad_(True)
            output = self.model(input_tensor)

            if target_class is None:
                target_class = output.argmax(dim=1).item()

            self.model.zero_grad()
            one_hot = torch.zeros_like(output)
            one_hot[0, target_class] = 1.0
            output.backward(gradient=one_hot)

            if self.gradients is None or self.activations is None:
                raise RuntimeError("Hooks did not capture activations/gradients")

            grads = self.gradients.detach()
            acts = self.activations.detach()

            if grads.dim() == 3:
                weights = grads.mean(dim=1, keepdim=True)
                cam = (weights * acts).sum(dim=2)
                cam = cam[:, 1:]
                n_patches = cam.shape[1]
                h = w = int(n_patches ** 0.5)
                cam = cam[:, :h * w].reshape(1, 1, h, w)
                cam = F.relu(cam)
                cam = cam.squeeze().cpu().numpy()
            else:
                weights = grads.mean(dim=(2, 3), keepdim=True)
                cam = (weights * acts).sum(dim=1, keepdim=True)
                cam = F.relu(cam)
                cam = cam.squeeze().cpu().numpy()

            if cam.max() > 0:
                cam = cam / cam.max()

            return cam, target_class
        finally:
            self._remove_hooks()
            self.gradients = None
            self.activations = None
            for name, m in self.model.named_modules():
                if name in _inplace_states:
                    m.inplace = True


def get_last_conv_layer(model_name, model):
    """获取最后一个卷积层（用于最终 Grad-CAM）"""
    if model_name == 'resnet50':
        return model[0].layer4[-1]
    elif model_name == 'vgg16':
        return model[0][28]  # Conv2d(512, 512)
    elif model_name == 'vgg19':
        return model[0][34]  # Conv2d(512, 512)
    elif model_name == 'inception_v3':
        return model[0].Mixed_7c
    elif model_name == 'inception_resnet_v2':
        backbone = model[0]
        if hasattr(backbone, 'conv2d_7b'):
            return backbone.conv2d_7b
        return backbone.mixed_7a
    elif model_name == 'efficientnet_b3':
        feat = model[0].features
        return feat[-1]
    elif model_name == 'mobilenet_v3':
        return model[0].features[-1]
    elif model_name == 'vit_b16':
        return model[0].encoder.layers[-1]
    return None


# ══════════════════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════════════════

def load_raw_image(img_path, target_is_landscape=True):
    """加载原始灰度图，并矫正朝向"""
    img = Image.open(img_path).convert('L')
    w, h = img.size
    is_landscape = w >= h
    if target_is_landscape != is_landscape:
        img = img.transpose(Image.ROTATE_90)
    return img


def get_sample_images_per_class(config, n_per_class=2):
    """每个类别取 n_per_class 张样本"""
    import pandas as pd
    data_dir = config.paths.data
    test_csv = os.path.join(data_dir, config.data.test_mapping)
    df = pd.read_csv(test_csv, encoding='utf-8-sig')

    label_col = '标签' if '标签' in df.columns else 'label'
    fname_col = '文件名' if '文件名' in df.columns else 'filename'

    samples = {}
    for label in sorted(df[label_col].unique()):
        subset = df[df[label_col] == label]
        chosen = subset.sample(n=min(n_per_class, len(subset)), random_state=42)
        imgs = []
        for _, row in chosen.iterrows():
            img_path = os.path.join(data_dir, 'test', row[fname_col])
            if os.path.exists(img_path):
                imgs.append(img_path)
        samples[int(label) - 1] = imgs  # 转为 0-indexed
    return samples


def compute_texture_features(img_gray):
    """计算纹理特征图: 边缘、梯度方向、频率"""
    img_arr = np.array(img_gray, dtype=np.float64)

    # 1. Sobel 边缘
    sobel_x = ndimage.sobel(img_arr, axis=1)
    sobel_y = ndimage.sobel(img_arr, axis=0)
    edge_magnitude = np.hypot(sobel_x, sobel_y)
    if edge_magnitude.max() > 0:
        edge_magnitude = edge_magnitude / edge_magnitude.max()

    # 2. 梯度方向
    gradient_direction = np.arctan2(sobel_y, sobel_x + 1e-10)
    gradient_direction = (gradient_direction + np.pi) / (2 * np.pi)  # normalize to [0, 1]

    # 3. Laplacian（高频纹理）
    laplacian = ndimage.laplace(img_arr)
    laplacian = np.abs(laplacian)
    if laplacian.max() > 0:
        laplacian = laplacian / laplacian.max()

    # 4. 局部方差（纹理复杂度）
    mean_filter = ndimage.uniform_filter(img_arr, size=11)
    mean_sq_filter = ndimage.uniform_filter(img_arr ** 2, size=11)
    local_var = np.clip(mean_sq_filter - mean_filter ** 2, 0, None)
    if local_var.max() > 0:
        local_var = local_var / local_var.max()

    return {
        'edge': edge_magnitude,
        'gradient_dir': gradient_direction,
        'laplacian': laplacian,
        'local_var': local_var,
    }


def find_roi_regions(img_gray, n_regions=3):
    """自动检测感兴趣区域（高纹理复杂度区域）"""
    img_arr = np.array(img_gray, dtype=np.float64)
    h, w = img_arr.shape

    # 使用局部方差找到高纹理区域
    mean_filter = ndimage.uniform_filter(img_arr, size=21)
    mean_sq_filter = ndimage.uniform_filter(img_arr ** 2, size=21)
    local_var = np.clip(mean_sq_filter - mean_filter ** 2, 0, None)

    # 同时考虑边缘信息
    sobel_x = ndimage.sobel(img_arr, axis=1)
    sobel_y = ndimage.sobel(img_arr, axis=0)
    edge_mag = np.hypot(sobel_x, sobel_y)

    # 综合分数
    score = 0.5 * local_var / (local_var.max() + 1e-10) + 0.5 * edge_mag / (edge_mag.max() + 1e-10)

    # 用大窗口滑动找到得分最高的区域
    crop_h, crop_w = h // 3, w // 3
    regions = []
    used_mask = np.zeros_like(score)

    for _ in range(n_regions):
        best_score = -1
        best_pos = (0, 0)

        for y in range(0, h - crop_h, crop_h // 4):
            for x in range(0, w - crop_w, crop_w // 4):
                region_score = score[y:y + crop_h, x:x + crop_w]
                overlap = used_mask[y:y + crop_h, x:x + crop_w].mean()
                s = region_score.mean() * (1 - overlap * 2)  # 惩罚重叠
                if s > best_score:
                    best_score = s
                    best_pos = (x, y)

        x, y = best_pos
        regions.append((x, y, crop_w, crop_h))
        used_mask[y:y + crop_h, x:x + crop_w] = 1.0

    return regions


def load_model_for_inference(model_name, config, device, img_h, img_w):
    """加载训练好的模型"""
    weight_path = os.path.join(PAPER_DIR, model_name, f'best_{model_name}.pth')
    if not os.path.exists(weight_path):
        weight_path = os.path.join(config.paths.outputs, 'models', f'best_{model_name}.pth')

    cfg = copy.deepcopy(config)
    cfg.data.img_height = img_h
    cfg.data.img_width = img_w
    cfg.model.name = model_name

    BackboneClass = get_backbone(model_name)
    classifier = BackboneClass(cfg)
    classifier.num_classes = 6
    classifier.device = device
    classifier.build_model()
    model = classifier.model

    if os.path.exists(weight_path):
        state_dict = torch.load(weight_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()
    return model


def prepare_input_tensor(img_path, img_h, img_w, to_rgb=True):
    """准备模型输入 tensor"""
    img = load_raw_image(img_path, target_is_landscape=True)
    img_resized = img.resize((img_w, img_h), Image.BILINEAR)

    if to_rgb:
        img_rgb = img_resized.convert('RGB')
        tensor = transforms.ToTensor()(img_rgb).unsqueeze(0)
    else:
        tensor = transforms.ToTensor()(img_resized).unsqueeze(0)

    return tensor


# ══════════════════════════════════════════════════════════════════════
#  图1: 每类建筑样本 — 局部放大细节对比
# ══════════════════════════════════════════════════════════════════════

def plot_texture_detail_zoom_per_class(config):
    """为每个类别生成局部放大细节对比图"""
    print("\n[1/5] Generating per-class texture detail zoom...")
    samples = get_sample_images_per_class(config, n_per_class=1)

    roi_colors = ['#FF4444', '#44BB44', '#4488FF']
    roi_labels = ['Region A', 'Region B', 'Region C']

    for cls_idx, img_paths in samples.items():
        if not img_paths:
            continue

        img_path = img_paths[0]
        img = load_raw_image(img_path, target_is_landscape=True)
        img_arr = np.array(img)
        h, w = img_arr.shape

        # 找到3个 ROI
        rois = find_roi_regions(img, n_regions=3)

        # 计算纹理特征
        features = compute_texture_features(img)

        # 创建图: 3行
        # 第1行: 原图(带ROI框) + 3个局部放大
        # 第2行: 边缘图(带ROI框) + 3个局部放大边缘
        # 第3行: 局部方差(带ROI框) + 3个局部放大方差
        fig = plt.figure(figsize=(18, 14))
        gs = GridSpec(3, 4, figure=fig, hspace=0.25, wspace=0.2)

        feature_maps = [
            ('Original', img_arr, 'gray'),
            ('Edge Detection (Sobel)', features['edge'], 'hot'),
            ('Texture Complexity', features['local_var'], 'inferno'),
        ]

        for row, (title, feat_map, cmap) in enumerate(feature_maps):
            # 左侧大图（带 ROI 框）
            ax_main = fig.add_subplot(gs[row, 0])
            ax_main.imshow(feat_map, cmap=cmap)
            ax_main.set_title(title, fontsize=11, fontweight='bold')
            ax_main.axis('off')

            for k, (rx, ry, rw, rh) in enumerate(rois):
                rect = patches.Rectangle(
                    (rx, ry), rw, rh,
                    linewidth=2.5, edgecolor=roi_colors[k],
                    facecolor='none', linestyle='-'
                )
                ax_main.add_patch(rect)
                ax_main.text(rx + 3, ry + 15, roi_labels[k],
                             color=roi_colors[k], fontsize=9, fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                       alpha=0.8, edgecolor=roi_colors[k]))

            # 右侧3个放大区域
            for k, (rx, ry, rw, rh) in enumerate(rois):
                ax_zoom = fig.add_subplot(gs[row, k + 1])
                crop = feat_map[ry:ry + rh, rx:rx + rw]
                ax_zoom.imshow(crop, cmap=cmap)
                ax_zoom.set_title(f'{roi_labels[k]}', fontsize=10,
                                  color=roi_colors[k], fontweight='bold')
                ax_zoom.axis('off')

                # 添加彩色边框
                for spine in ax_zoom.spines.values():
                    spine.set_edgecolor(roi_colors[k])
                    spine.set_linewidth(3)
                    spine.set_visible(True)

        cls_name = CLASS_NAMES.get(cls_idx, f'Class {cls_idx}')
        fig.suptitle(f'Texture Detail Analysis — {cls_name}',
                     fontsize=15, fontweight='bold', y=1.01)
        plt.tight_layout()

        path = os.path.join(FIG_DIR, f'texture_detail_zoom_class{cls_idx}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════
#  图2: 六类建筑纹理特征对比
# ══════════════════════════════════════════════════════════════════════

def plot_multi_class_texture_comparison(config):
    """六类建筑纹理特征横向对比"""
    print("\n[2/5] Generating multi-class texture comparison...")
    samples = get_sample_images_per_class(config, n_per_class=1)

    n_classes = len(samples)
    fig, axes = plt.subplots(4, n_classes, figsize=(n_classes * 3.5, 14))

    row_titles = ['Original', 'Edge (Sobel)', 'High-Freq (Laplacian)', 'Texture Complexity']

    for col, cls_idx in enumerate(sorted(samples.keys())):
        if not samples[cls_idx]:
            continue

        img = load_raw_image(samples[cls_idx][0], target_is_landscape=True)
        img_arr = np.array(img)
        features = compute_texture_features(img)

        feature_list = [
            (img_arr, 'gray'),
            (features['edge'], 'hot'),
            (features['laplacian'], 'magma'),
            (features['local_var'], 'inferno'),
        ]

        for row, (feat_map, cmap) in enumerate(feature_list):
            axes[row, col].imshow(feat_map, cmap=cmap)
            axes[row, col].axis('off')

            if row == 0:
                cls_name = CLASS_NAMES.get(cls_idx, f'Class {cls_idx}')
                axes[row, col].set_title(cls_name, fontsize=12, fontweight='bold')

        if col == 0:
            for row, title in enumerate(row_titles):
                axes[row, col].set_ylabel(title, fontsize=11, fontweight='bold',
                                          rotation=90, labelpad=15)
                axes[row, col].yaxis.set_label_position('left')

    fig.suptitle('Multi-Class Texture Feature Comparison',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()

    path = os.path.join(FIG_DIR, 'texture_multi_class_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════
#  图3: 纹理特征提取深度分析
# ══════════════════════════════════════════════════════════════════════

def plot_texture_feature_analysis(config):
    """单张图的完整纹理分析"""
    print("\n[3/5] Generating texture feature deep analysis...")
    samples = get_sample_images_per_class(config, n_per_class=1)

    # 选2张不同类的样本做详细分析
    chosen = []
    for cls_idx in sorted(samples.keys()):
        if samples[cls_idx]:
            chosen.append((cls_idx, samples[cls_idx][0]))
        if len(chosen) >= 2:
            break

    fig = plt.figure(figsize=(22, 12))
    gs = GridSpec(2, 6, figure=fig, hspace=0.3, wspace=0.3)

    for row, (cls_idx, img_path) in enumerate(chosen):
        img = load_raw_image(img_path, target_is_landscape=True)
        img_arr = np.array(img)
        features = compute_texture_features(img)

        # 找到最佳 ROI
        rois = find_roi_regions(img, n_regions=1)
        rx, ry, rw, rh = rois[0]

        items = [
            ('Original', img_arr, 'gray'),
            ('Edge (Sobel)', features['edge'], 'hot'),
            ('Gradient Dir.', features['gradient_dir'], 'hsv'),
            ('High-Freq (Lap.)', features['laplacian'], 'magma'),
            ('Texture Complex.', features['local_var'], 'inferno'),
        ]

        # 第一列: 原图 + ROI
        ax0 = fig.add_subplot(gs[row, 0])
        ax0.imshow(img_arr, cmap='gray')
        rect = patches.Rectangle((rx, ry), rw, rh, linewidth=3,
                                 edgecolor='red', facecolor='none')
        ax0.add_patch(rect)
        cls_name = CLASS_NAMES.get(cls_idx, f'Class {cls_idx}')
        ax0.set_title(f'{cls_name}\n(with ROI)', fontsize=10, fontweight='bold')
        ax0.axis('off')

        # 后续列: 各纹理特征的放大视图
        for col, (title, feat_map, cmap) in enumerate(items[1:], 1):
            ax = fig.add_subplot(gs[row, col])

            # 显示放大区域
            crop = feat_map[ry:ry + rh, rx:rx + rw]
            ax.imshow(crop, cmap=cmap)
            ax.set_title(f'{title}\n(Zoomed ROI)', fontsize=9, fontweight='bold')
            ax.axis('off')

            for spine in ax.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(2)
                spine.set_visible(True)

        # 第6列: 纹理统计直方图
        ax_hist = fig.add_subplot(gs[row, 5])
        edge_crop = features['edge'][ry:ry + rh, rx:rx + rw].flatten()
        var_crop = features['local_var'][ry:ry + rh, rx:rx + rw].flatten()
        ax_hist.hist(edge_crop, bins=50, alpha=0.6, color='#FF6B6B', label='Edge', density=True)
        ax_hist.hist(var_crop, bins=50, alpha=0.6, color='#4ECDC4', label='Texture', density=True)
        ax_hist.set_title(f'Feature Distribution\n(ROI)', fontsize=9, fontweight='bold')
        ax_hist.legend(fontsize=8)
        ax_hist.set_xlabel('Intensity', fontsize=8)
        ax_hist.set_ylabel('Density', fontsize=8)
        ax_hist.tick_params(labelsize=7)

    fig.suptitle('Texture Feature Analysis with Local Detail Zoom',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()

    path = os.path.join(FIG_DIR, 'texture_feature_analysis.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════
#  图4: 跨模型 Grad-CAM 局部放大对比
# ══════════════════════════════════════════════════════════════════════

def plot_cross_model_attention_zoom(config, device):
    """对同一张图，对比不同模型的注意力局部放大"""
    print("\n[4/5] Generating cross-model attention zoom comparison...")

    # 选择要对比的模型
    compare_models = ['resnet50', 'vgg16', 'inception_v3', 'efficientnet_b3', 'mobilenet_v3', 'vit_b16']

    # 取2张不同类的测试样本
    samples = get_sample_images_per_class(config, n_per_class=1)
    chosen_imgs = []
    for cls_idx in sorted(samples.keys()):
        if samples[cls_idx]:
            chosen_imgs.append((cls_idx, samples[cls_idx][0]))
        if len(chosen_imgs) >= 2:
            break

    n_models = len(compare_models)
    n_imgs = len(chosen_imgs)

    # 布局: 每张图一行，列 = [原图+ROI] + [各模型Grad-CAM全图] + [各模型ROI放大]
    # 简化：每张图 2 行 (全图行 + 放大行)
    fig = plt.figure(figsize=(n_models * 3.2 + 4, n_imgs * 6.5))
    gs = GridSpec(n_imgs * 2, n_models + 1, figure=fig, hspace=0.35, wspace=0.15)

    for img_idx, (cls_idx, img_path) in enumerate(chosen_imgs):
        # 原始灰度图
        raw_img = load_raw_image(img_path, target_is_landscape=True)
        raw_arr = np.array(raw_img)

        # 找到 ROI
        rois = find_roi_regions(raw_img, n_regions=1)
        rx, ry, rw, rh = rois[0]

        # 第1行（全图）的第1列：原图 + ROI
        row_full = img_idx * 2
        row_zoom = img_idx * 2 + 1

        ax_orig_full = fig.add_subplot(gs[row_full, 0])
        ax_orig_full.imshow(raw_arr, cmap='gray')
        rect = patches.Rectangle((rx, ry), rw, rh, linewidth=3,
                                 edgecolor='lime', facecolor='none')
        ax_orig_full.add_patch(rect)
        cls_name = CLASS_NAMES.get(cls_idx, f'Class {cls_idx}')
        ax_orig_full.set_title(f'Original\n{cls_name}', fontsize=10, fontweight='bold')
        ax_orig_full.axis('off')

        # 第2行的第1列：原图 ROI 放大
        ax_orig_zoom = fig.add_subplot(gs[row_zoom, 0])
        crop = raw_arr[ry:ry + rh, rx:rx + rw]
        ax_orig_zoom.imshow(crop, cmap='gray')
        ax_orig_zoom.set_title('Zoomed ROI', fontsize=10, fontweight='bold')
        ax_orig_zoom.axis('off')
        for spine in ax_orig_zoom.spines.values():
            spine.set_edgecolor('lime')
            spine.set_linewidth(3)
            spine.set_visible(True)

        # 对每个模型生成 Grad-CAM
        for m_idx, model_name in enumerate(compare_models):
            cfg_hw = [(h, w) for n, h, w in MODEL_CONFIGS if n == model_name]
            if not cfg_hw:
                continue
            img_h, img_w = cfg_hw[0]

            try:
                model = load_model_for_inference(model_name, config, device, img_h, img_w)
                target_layer = get_last_conv_layer(model_name, model)
                if target_layer is None:
                    continue

                input_tensor = prepare_input_tensor(img_path, img_h, img_w).to(device)
                gradcam = GradCAM(model, target_layer)
                cam, pred = gradcam.generate(input_tensor)

                # 放大到原图大小
                cam_full = np.array(Image.fromarray(cam).resize(
                    (raw_arr.shape[1], raw_arr.shape[0]), Image.BILINEAR
                ))

                # 第1行：全图 Grad-CAM overlay
                ax_full = fig.add_subplot(gs[row_full, m_idx + 1])
                ax_full.imshow(raw_arr, cmap='gray', alpha=0.5)
                ax_full.imshow(cam_full, cmap='jet', alpha=0.5)
                rect2 = patches.Rectangle((rx, ry), rw, rh, linewidth=2.5,
                                          edgecolor='lime', facecolor='none')
                ax_full.add_patch(rect2)
                display_name = DISPLAY_NAMES.get(model_name, model_name)
                ax_full.set_title(display_name, fontsize=10, fontweight='bold')
                ax_full.axis('off')

                # 第2行：ROI 放大 Grad-CAM
                ax_zoom = fig.add_subplot(gs[row_zoom, m_idx + 1])
                cam_crop = cam_full[ry:ry + rh, rx:rx + rw]
                raw_crop = raw_arr[ry:ry + rh, rx:rx + rw]
                ax_zoom.imshow(raw_crop, cmap='gray', alpha=0.4)
                ax_zoom.imshow(cam_crop, cmap='jet', alpha=0.6)
                ax_zoom.set_title('Zoomed', fontsize=9)
                ax_zoom.axis('off')
                for spine in ax_zoom.spines.values():
                    spine.set_edgecolor('lime')
                    spine.set_linewidth(2)
                    spine.set_visible(True)

                del model
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"    Warning: {model_name} failed: {e}")
                for r in [row_full, row_zoom]:
                    ax = fig.add_subplot(gs[r, m_idx + 1])
                    ax.text(0.5, 0.5, f'Error', ha='center', va='center', fontsize=9)
                    ax.axis('off')

    fig.suptitle('Cross-Model Attention Comparison with Local Detail Zoom',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()

    path = os.path.join(FIG_DIR, 'texture_cross_model_zoom.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════
#  图5: 最佳 vs 最差模型注意力对比 + 局部放大
# ══════════════════════════════════════════════════════════════════════

def plot_best_vs_worst_attention_zoom(config, device):
    """最佳模型 vs 最差模型的注意力区域局部放大对比"""
    print("\n[5/5] Generating best vs worst model attention zoom...")

    best_model = 'resnet50'
    worst_model = 'vit_b16'  # 在 CNN 中选一个表现相对差的

    # 加载 test metrics 确认
    summary_path = os.path.join(PAPER_DIR, 'summary.json')
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        # 找最佳和最差
        accs = {}
        for k, v in summary.items():
            if 'test' in v and 'test_accuracy' in v['test']:
                accs[k] = v['test']['test_accuracy']
        if accs:
            best_model = max(accs, key=accs.get)
            worst_model = min(accs, key=accs.get)
            # 排除 custom_mlp（太差无意义），取倒数第二
            if worst_model == 'custom_mlp':
                sorted_accs = sorted(accs.items(), key=lambda x: x[1])
                worst_model = sorted_accs[1][0] if len(sorted_accs) > 1 else sorted_accs[0][0]

    print(f"  Best: {best_model}, Contrast: {worst_model}")

    # 加载样本
    samples = get_sample_images_per_class(config, n_per_class=1)
    # 取4张不同类
    chosen = []
    for cls_idx in sorted(samples.keys()):
        if samples[cls_idx]:
            chosen.append((cls_idx, samples[cls_idx][0]))
        if len(chosen) >= 4:
            break

    n_imgs = len(chosen)
    # 布局: n_imgs 行, 7列 [原图+ROI | Best全图 | Best放大 | Worst全图 | Worst放大 | 边缘 | 边缘放大]
    fig = plt.figure(figsize=(26, n_imgs * 4))
    gs = GridSpec(n_imgs, 7, figure=fig, hspace=0.3, wspace=0.15)

    for i, (cls_idx, img_path) in enumerate(chosen):
        raw_img = load_raw_image(img_path, target_is_landscape=True)
        raw_arr = np.array(raw_img)

        rois = find_roi_regions(raw_img, n_regions=1)
        rx, ry, rw, rh = rois[0]

        features = compute_texture_features(raw_img)

        # Col 0: 原图 + ROI
        ax0 = fig.add_subplot(gs[i, 0])
        ax0.imshow(raw_arr, cmap='gray')
        rect = patches.Rectangle((rx, ry), rw, rh, linewidth=3,
                                 edgecolor='#FF4444', facecolor='none')
        ax0.add_patch(rect)
        cls_name = CLASS_NAMES.get(cls_idx, f'Class {cls_idx}')
        ax0.set_title(f'Original ({cls_name})' if i == 0 else cls_name,
                      fontsize=10, fontweight='bold')
        ax0.axis('off')

        for m_idx, (model_name, col_full, col_zoom, color) in enumerate([
            (best_model, 1, 2, '#22CC22'),
            (worst_model, 3, 4, '#FF8800'),
        ]):
            cfg_hw = [(h, w) for n, h, w in MODEL_CONFIGS if n == model_name]
            if not cfg_hw:
                continue
            img_h, img_w = cfg_hw[0]

            try:
                model = load_model_for_inference(model_name, config, device, img_h, img_w)
                target_layer = get_last_conv_layer(model_name, model)
                input_tensor = prepare_input_tensor(img_path, img_h, img_w).to(device)

                gradcam = GradCAM(model, target_layer)
                cam, pred = gradcam.generate(input_tensor)

                cam_full = np.array(Image.fromarray(cam).resize(
                    (raw_arr.shape[1], raw_arr.shape[0]), Image.BILINEAR
                ))

                display_name = DISPLAY_NAMES.get(model_name, model_name)

                # 全图
                ax_full = fig.add_subplot(gs[i, col_full])
                ax_full.imshow(raw_arr, cmap='gray', alpha=0.45)
                ax_full.imshow(cam_full, cmap='jet', alpha=0.55)
                rect2 = patches.Rectangle((rx, ry), rw, rh, linewidth=2.5,
                                          edgecolor=color, facecolor='none')
                ax_full.add_patch(rect2)
                label = 'Best' if m_idx == 0 else 'Contrast'
                ax_full.set_title(f'{label}: {display_name}' if i == 0 else display_name,
                                  fontsize=10, fontweight='bold')
                ax_full.axis('off')

                # 放大
                ax_zoom = fig.add_subplot(gs[i, col_zoom])
                cam_crop = cam_full[ry:ry + rh, rx:rx + rw]
                raw_crop = raw_arr[ry:ry + rh, rx:rx + rw]
                ax_zoom.imshow(raw_crop, cmap='gray', alpha=0.35)
                ax_zoom.imshow(cam_crop, cmap='jet', alpha=0.65)
                ax_zoom.set_title(f'Zoom' if i == 0 else '', fontsize=9)
                ax_zoom.axis('off')
                for spine in ax_zoom.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(3)
                    spine.set_visible(True)

                del model
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"    Warning: {model_name} failed for class {cls_idx}: {e}")

        # Col 5: 边缘图
        ax_edge = fig.add_subplot(gs[i, 5])
        ax_edge.imshow(features['edge'], cmap='hot')
        rect3 = patches.Rectangle((rx, ry), rw, rh, linewidth=2.5,
                                  edgecolor='cyan', facecolor='none')
        ax_edge.add_patch(rect3)
        ax_edge.set_title('Edge Map' if i == 0 else '', fontsize=10, fontweight='bold')
        ax_edge.axis('off')

        # Col 6: 边缘放大
        ax_edge_zoom = fig.add_subplot(gs[i, 6])
        edge_crop = features['edge'][ry:ry + rh, rx:rx + rw]
        ax_edge_zoom.imshow(edge_crop, cmap='hot')
        ax_edge_zoom.set_title('Edge Zoom' if i == 0 else '', fontsize=9)
        ax_edge_zoom.axis('off')
        for spine in ax_edge_zoom.spines.values():
            spine.set_edgecolor('cyan')
            spine.set_linewidth(3)
            spine.set_visible(True)

    fig.suptitle('Best vs Contrast Model: Attention & Texture Detail Comparison',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()

    path = os.path.join(FIG_DIR, 'texture_attention_zoom.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════
#  主函数
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Texture analysis with detail zoom')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--parts', type=str, default='1,2,3,4,5',
                        help='Comma-separated list of parts to run (1-5)')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    config = load_config()
    parts = [int(x.strip()) for x in args.parts.split(',')]

    # 纯 CPU 分析（不需要模型）
    if 1 in parts:
        plot_texture_detail_zoom_per_class(config)

    if 2 in parts:
        plot_multi_class_texture_comparison(config)

    if 3 in parts:
        plot_texture_feature_analysis(config)

    # 需要 GPU 和模型的分析
    if 4 in parts:
        plot_cross_model_attention_zoom(config, device)

    if 5 in parts:
        plot_best_vs_worst_attention_zoom(config, device)

    print("\n" + "=" * 70)
    print("  TEXTURE ANALYSIS COMPLETE!")
    print(f"  Figures saved to: {FIG_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
