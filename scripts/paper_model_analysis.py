#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
论文级模型深度分析脚本

输出:  outputs/paper/figures/
  1.  model_perf_metrics.json         — 各模型推理速度/模型大小/GFLOPs
  2.  inference_speed_comparison.png   — 推理速度对比柱状图
  3.  model_size_comparison.png        — 模型大小对比
  4.  flops_vs_accuracy.png            — FLOPs vs 精度
  5.  efficiency_radar.png             — 多维效率雷达图
  6.  gradcam_layers_<model>.png       — 各层 Grad-CAM 热力图
  7.  gradcam_best_model_grid.png      — 最佳模型多样本 Grad-CAM 网格
  8.  ablation_study.png               — 消融实验结果
  9.  module_contribution_heatmap.png  — 模块贡献热力可视化
  10. feature_channel_heatmap.png      — 通道激活热力图
"""

import argparse
import json
import os
import sys
import time
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.utils.paths import paths
from src.models.backbones import get_backbone, list_backbones
from src.models.base_classifier import AncientCharDataset

# ── 配置 ──
PAPER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'outputs', 'paper')
FIG_DIR = os.path.join(PAPER_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

MODEL_CONFIGS = [
    ("custom_mlp",          224, 224),
    ("resnet50",            224, 224),
    ("vgg16",               224, 224),
    ("vgg19",               224, 224),
    ("inception_v3",        299, 299),
    ("inception_resnet_v2", 299, 299),
    ("efficientnet_b3",     300, 300),
    ("mobilenet_v3",        224, 224),
    ("vit_b16",             224, 224),
]

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

MODEL_COLORS = {
    'custom_mlp': '#95a5a6',
    'resnet50': '#e74c3c',
    'vgg16': '#3498db',
    'vgg19': '#2980b9',
    'inception_v3': '#2ecc71',
    'inception_resnet_v2': '#27ae60',
    'efficientnet_b3': '#f39c12',
    'mobilenet_v3': '#9b59b6',
    'vit_b16': '#e67e22',
}


# =============================================================================
# 1. 模型性能指标: 推理速度 / 模型大小 / GFLOPs
# =============================================================================

def measure_model_metrics(model_name, img_h, img_w, model, device, n_warmup=10, n_runs=50):
    """测量模型推理速度、大小和 FLOPs"""
    model.eval()
    model.to(device)

    # 输入通道数
    in_channels = 1 if model_name == 'custom_mlp' else 3
    dummy_input = torch.randn(1, in_channels, img_h, img_w).to(device)

    # 1) 推理速度
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy_input)
        if device.type == 'cuda':
            torch.cuda.synchronize()

        times = []
        for _ in range(n_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms

    avg_latency = np.mean(times)
    std_latency = np.std(times)
    throughput = 1000.0 / avg_latency  # images/sec

    # 2) 模型大小 (MB)
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / (1024 * 1024)

    # 模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 3) GFLOPs
    gflops = None
    try:
        from thop import profile as thop_profile
        flops, _ = thop_profile(model, inputs=(dummy_input,), verbose=False)
        gflops = flops / 1e9
    except Exception:
        # Fallback: 手动估算
        try:
            from torch.utils.flop_counter import FlopCounterMode
            flop_counter = FlopCounterMode(display=False)
            with flop_counter:
                model(dummy_input)
            gflops = flop_counter.get_total_flops() / 1e9
        except Exception:
            gflops = None

    return {
        'avg_latency_ms': round(avg_latency, 2),
        'std_latency_ms': round(std_latency, 2),
        'throughput_fps': round(throughput, 1),
        'model_size_mb': round(model_size_mb, 2),
        'total_params': total_params,
        'trainable_params': trainable_params,
        'gflops': round(gflops, 2) if gflops else None,
    }


def run_all_metrics(device):
    """对所有模型运行性能指标测量"""
    print("\n" + "=" * 70)
    print("  MODEL PERFORMANCE METRICS")
    print("=" * 70)

    config = load_config()
    results = {}

    for model_name, img_h, img_w in MODEL_CONFIGS:
        print(f"\n--- {DISPLAY_NAMES[model_name]} ({model_name}) ---")

        # 查找已保存的模型权重
        weight_path = os.path.join(PAPER_DIR, model_name, f'best_{model_name}.pth')
        if not os.path.exists(weight_path):
            # 尝试默认输出路径
            weight_path = os.path.join(config.paths.outputs, 'models', f'best_{model_name}.pth')

        # 配置模型
        cfg = copy.deepcopy(config)
        cfg.data.img_height = img_h
        cfg.data.img_width = img_w
        cfg.model.name = model_name

        try:
            BackboneClass = get_backbone(model_name)
            classifier = BackboneClass(cfg)
            classifier.num_classes = 6  # 6 类
            model = classifier.build_model()

            # 加载权重（如果有）
            if os.path.exists(weight_path):
                state_dict = torch.load(weight_path, map_location=device, weights_only=True)
                model.load_state_dict(state_dict)
                print(f"  Loaded weights from {weight_path}")

            metrics = measure_model_metrics(model_name, img_h, img_w, model, device)
            results[model_name] = metrics

            print(f"  Latency: {metrics['avg_latency_ms']:.2f} ± {metrics['std_latency_ms']:.2f} ms")
            print(f"  Throughput: {metrics['throughput_fps']:.1f} FPS")
            print(f"  Model size: {metrics['model_size_mb']:.2f} MB")
            print(f"  Params: {metrics['total_params']:,} (trainable: {metrics['trainable_params']:,})")
            print(f"  GFLOPs: {metrics['gflops']}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # 保存 JSON
    out_path = os.path.join(FIG_DIR, 'model_perf_metrics.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics to {out_path}")

    return results


# =============================================================================
# 2. 性能指标可视化
# =============================================================================

def plot_inference_speed(metrics):
    """推理速度对比柱状图"""
    keys = [k for k in metrics if k in DISPLAY_NAMES]
    names = [DISPLAY_NAMES[k] for k in keys]
    latency = [metrics[k]['avg_latency_ms'] for k in keys]
    throughput = [metrics[k]['throughput_fps'] for k in keys]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 推理延迟
    colors = [MODEL_COLORS.get(k, '#333') for k in keys]
    bars = ax1.barh(names, latency, color=colors, edgecolor='white', height=0.6)
    ax1.set_xlabel('Inference Latency (ms)', fontsize=12)
    ax1.set_title('Inference Latency per Image', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    for bar, val in zip(bars, latency):
        ax1.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}ms', va='center', fontsize=9, fontweight='bold')

    # 吞吐量
    bars2 = ax2.barh(names, throughput, color=colors, edgecolor='white', height=0.6)
    ax2.set_xlabel('Throughput (images/sec)', fontsize=12)
    ax2.set_title('Inference Throughput', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    for bar, val in zip(bars2, throughput):
        ax2.text(val + 1, bar.get_y() + bar.get_height()/2,
                f'{val:.0f}', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'inference_speed_comparison.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_model_size(metrics):
    """模型大小对比"""
    keys = [k for k in metrics if k in DISPLAY_NAMES]
    names = [DISPLAY_NAMES[k] for k in keys]
    sizes = [metrics[k]['model_size_mb'] for k in keys]
    params = [metrics[k]['total_params'] / 1e6 for k in keys]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    colors = [MODEL_COLORS.get(k, '#333') for k in keys]

    # 模型大小 MB
    bars1 = ax1.barh(names, sizes, color=colors, edgecolor='white', height=0.6)
    ax1.set_xlabel('Model Size (MB)', fontsize=12)
    ax1.set_title('Model Storage Size', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    for bar, val in zip(bars1, sizes):
        ax1.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}MB', va='center', fontsize=9, fontweight='bold')

    # 参数量 M
    bars2 = ax2.barh(names, params, color=colors, edgecolor='white', height=0.6)
    ax2.set_xlabel('Parameters (M)', fontsize=12)
    ax2.set_title('Model Parameters', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    for bar, val in zip(bars2, params):
        ax2.text(val + 0.2, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}M', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'model_size_comparison.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_flops_vs_accuracy(metrics):
    """GFLOPs vs Accuracy 散点图"""
    # 加载 test metrics
    test_data = {}
    for k in metrics:
        tf = os.path.join(PAPER_DIR, k, 'test_metrics.json')
        if os.path.exists(tf):
            with open(tf) as f:
                test_data[k] = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 7))

    for k in metrics:
        if k not in test_data or metrics[k]['gflops'] is None:
            continue
        gflops = metrics[k]['gflops']
        acc = test_data[k]['test_accuracy'] * 100
        size_mb = metrics[k]['model_size_mb']
        color = MODEL_COLORS.get(k, '#333')
        name = DISPLAY_NAMES[k]

        # 圆的大小 = model size
        s = max(50, size_mb * 1.5)
        ax.scatter(gflops, acc, s=s, c=color, edgecolors='black',
                   linewidth=1, zorder=5, alpha=0.8)
        ax.annotate(name, (gflops, acc), textcoords='offset points',
                    xytext=(8, 5), fontsize=9)

    ax.set_xlabel('GFLOPs', fontsize=13)
    ax.set_ylabel('Test Accuracy (%)', fontsize=13)
    ax.set_title('Computational Cost vs. Accuracy\n(bubble size ∝ model size)',
                 fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'flops_vs_accuracy.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_efficiency_radar(metrics):
    """多维效率雷达图"""
    test_data = {}
    for k in metrics:
        tf = os.path.join(PAPER_DIR, k, 'test_metrics.json')
        if os.path.exists(tf):
            with open(tf) as f:
                test_data[k] = json.load(f)

    # 维度: Accuracy, Speed(1/latency), Size(1/size), Params(1/params), F1
    dims = ['Accuracy', 'Speed\n(1/latency)', 'Compactness\n(1/size)', 'Efficiency\n(1/params)', 'Macro F1']
    n_dims = len(dims)

    # 收集原始值
    raw = {}
    for k in metrics:
        if k not in test_data:
            continue
        raw[k] = [
            test_data[k]['test_accuracy'] * 100,
            1000.0 / metrics[k]['avg_latency_ms'],  # speed
            1.0 / max(metrics[k]['model_size_mb'], 0.1),  # compactness
            1.0 / max(metrics[k]['total_params'] / 1e6, 0.01),  # efficiency
            test_data[k]['macro_f1'] * 100,
        ]

    if not raw:
        print("  Skipping radar chart: no data")
        return

    # 归一化到 0-1
    all_vals = np.array(list(raw.values()))
    mins = all_vals.min(axis=0)
    maxs = all_vals.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0

    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for k, vals in raw.items():
        normalized = ((np.array(vals) - mins) / ranges).tolist()
        normalized += normalized[:1]
        color = MODEL_COLORS.get(k, '#333')
        ax.plot(angles, normalized, 'o-', linewidth=2, label=DISPLAY_NAMES[k],
                color=color)
        ax.fill(angles, normalized, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_title('Model Efficiency Radar Chart', fontsize=14, fontweight='bold', pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'efficiency_radar.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# 3. Grad-CAM 热力图
# =============================================================================

class GradCAM:
    """Grad-CAM 实现 — 带 hook 自动清理"""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._hooks = []

    def _register_hooks(self):
        """注册 hook（每次 generate 前调用）"""
        def fwd_hook(module, input, output):
            # 保留梯度链（不 detach）
            self.activations = output

        def bwd_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        h1 = self.target_layer.register_forward_hook(fwd_hook)
        h2 = self.target_layer.register_full_backward_hook(bwd_hook)
        self._hooks = [h1, h2]

    def _remove_hooks(self):
        """清理所有 hook"""
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def generate(self, input_tensor, target_class=None):
        """生成 Grad-CAM 热力图"""
        self._register_hooks()
        try:
            self.model.eval()

            # 关键修复1：禁用所有 inplace ReLU，避免 backward hook 报错
            _inplace_states = {}
            for name, m in self.model.named_modules():
                if isinstance(m, nn.ReLU) and m.inplace:
                    _inplace_states[name] = True
                    m.inplace = False

            # 关键修复2：给输入开启 requires_grad，确保梯度能流过冻结层
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

            # 处理非 4D 输出（ViT encoder 输出为 3D: B, N, D）
            if grads.dim() == 3:
                # (B, N, D) -> 对 N 维度做 GAP 得到权重
                weights = grads.mean(dim=1, keepdim=True)  # (B, 1, D)
                cam = (weights * acts).sum(dim=2)  # (B, N)
                cam = cam[:, 1:]  # 去掉 CLS token
                # reshape 为方形
                n_patches = cam.shape[1]
                h = w = int(n_patches ** 0.5)
                cam = cam[:, :h*w].reshape(1, 1, h, w)
                cam = F.relu(cam)
                cam = cam.squeeze().cpu().numpy()
            else:
                # 标准 4D 卷积特征图 (B, C, H, W)
                weights = grads.mean(dim=(2, 3), keepdim=True)  # GAP
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
            # 恢复 inplace 状态
            for name, m in self.model.named_modules():
                if name in _inplace_states:
                    m.inplace = True


def get_target_layers(model_name, model):
    """获取各模型适合做 Grad-CAM 的中间层（选取 Conv/ReLU 而非 MaxPool）"""
    layers = {}

    if model_name == 'resnet50':
        backbone = model[0]
        # 每个 layer 是 Sequential of Bottleneck，取最后一个 Bottleneck
        layers = {
            'layer1': backbone.layer1[-1],
            'layer2': backbone.layer2[-1],
            'layer3': backbone.layer3[-1],
            'layer4': backbone.layer4[-1],
        }
    elif model_name in ('vgg16', 'vgg19'):
        # model[0] IS the features Sequential
        feat = model[0]
        if model_name == 'vgg16':
            # 选 Conv2d 层（不选 ReLU，因为 inplace=True 会导致 backward hook 失败）
            layers = {
                'block2_conv': feat[7],   # Conv2d(128, 128)
                'block3_conv': feat[14],  # Conv2d(256, 256)
                'block4_conv': feat[21],  # Conv2d(512, 512)
                'block5_conv': feat[28],  # Conv2d(512, 512)
            }
        else:  # vgg19
            layers = {
                'block2_conv': feat[7],   # Conv2d(128, 128)
                'block3_conv': feat[16],  # Conv2d(256, 256)
                'block4_conv': feat[25],  # Conv2d(512, 512)
                'block5_conv': feat[34],  # Conv2d(512, 512)
            }
    elif model_name == 'inception_v3':
        backbone = model[0]
        layers = {
            'Mixed_5d': backbone.Mixed_5d,
            'Mixed_6e': backbone.Mixed_6e,
            'Mixed_7c': backbone.Mixed_7c,
        }
    elif model_name == 'inception_resnet_v2':
        backbone = model[0]
        layers = {
            'mixed_5b': backbone.mixed_5b if hasattr(backbone, 'mixed_5b') else None,
            'mixed_6a': backbone.mixed_6a if hasattr(backbone, 'mixed_6a') else None,
            'mixed_7a': backbone.mixed_7a if hasattr(backbone, 'mixed_7a') else None,
            'conv2d_7b': backbone.conv2d_7b if hasattr(backbone, 'conv2d_7b') else None,
        }
        layers = {k: v for k, v in layers.items() if v is not None}
    elif model_name == 'efficientnet_b3':
        backbone = model[0]
        feat = backbone.features
        layers = {
            'block2': feat[2],
            'block4': feat[4],
            'block6': feat[6],
            'block8': feat[8] if len(feat) > 8 else feat[-1],
        }
    elif model_name == 'mobilenet_v3':
        backbone = model[0]
        feat = backbone.features
        layers = {
            'block4': feat[4],
            'block8': feat[8],
            'block12': feat[12],
            'block_last': feat[-1],
        }
    elif model_name == 'vit_b16':
        backbone = model[0]
        if hasattr(backbone, 'encoder') and hasattr(backbone.encoder, 'layers'):
            enc_layers = backbone.encoder.layers
            n = len(enc_layers)
            layers = {
                f'encoder_{n//4}': enc_layers[n//4],
                f'encoder_{n//2}': enc_layers[n//2],
                f'encoder_{3*n//4}': enc_layers[3*n//4],
                f'encoder_{n-1}': enc_layers[n-1],
            }

    return layers


def load_sample_images(config, n_samples=8, to_rgb=True, img_h=224, img_w=224):
    """加载样本图片用于可视化"""
    data_dir = config.paths.data
    test_csv = os.path.join(data_dir, config.data.test_mapping)

    if not os.path.exists(test_csv):
        # 使用 train mapping
        test_csv = os.path.join(data_dir, config.data.train_mapping)

    df = pd.read_csv(test_csv)

    images = []
    raw_images = []
    labels = []

    transform = transforms.Compose([transforms.ToTensor()])

    # 每个类取 1-2 张
    for cls in sorted(df['标签'].unique()):
        cls_df = df[df['标签'] == cls]
        for _, row in cls_df.head(2).iterrows():
            subdir = 'test' if 'test' in test_csv else 'train'
            img_path = os.path.join(data_dir, subdir, row['文件名'])
            if not os.path.exists(img_path):
                img_path = os.path.join(data_dir, 'train', row['文件名'])
            if not os.path.exists(img_path):
                continue

            img = Image.open(img_path).convert('L')
            img = img.resize((img_w, img_h), Image.BILINEAR)
            raw_img = np.array(img)

            if to_rgb:
                img = img.convert('RGB')

            tensor = transform(img).unsqueeze(0)
            images.append(tensor)
            raw_images.append(raw_img)
            labels.append(row['标签'] - 1)

            if len(images) >= n_samples:
                break
        if len(images) >= n_samples:
            break

    return images, raw_images, labels


def generate_gradcam_for_model(model_name, model, config, device, img_h, img_w):
    """为指定模型生成各层 Grad-CAM 热力图"""
    if model_name == 'custom_mlp':
        print(f"  Skipping Grad-CAM for {model_name} (no convolutional layers)")
        return

    target_layers = get_target_layers(model_name, model)
    if not target_layers:
        print(f"  No target layers found for {model_name}")
        return

    to_rgb = model_name != 'custom_mlp'
    images, raw_images, labels = load_sample_images(
        config, n_samples=6, to_rgb=to_rgb, img_h=img_h, img_w=img_w
    )

    if not images:
        print(f"  No sample images available for Grad-CAM")
        return

    n_imgs = min(len(images), 6)
    n_layers = len(target_layers)
    layer_names = list(target_layers.keys())

    fig, axes = plt.subplots(n_imgs, n_layers + 1, figsize=((n_layers + 1) * 3, n_imgs * 3))
    if n_imgs == 1:
        axes = axes.reshape(1, -1)

    display_name = DISPLAY_NAMES.get(model_name, model_name)

    for i in range(n_imgs):
        # 原始图像
        axes[i, 0].imshow(raw_images[i], cmap='gray')
        axes[i, 0].set_title(f'Class {labels[i]}' if i == 0 else '', fontsize=9)
        if i == 0:
            axes[i, 0].set_title('Original', fontsize=10, fontweight='bold')
        axes[i, 0].axis('off')

        # 各层 Grad-CAM
        for j, layer_name in enumerate(layer_names):
            try:
                # 为每个层重新创建 GradCAM 实例 (避免 hook 冲突)
                gradcam = GradCAM(model, target_layers[layer_name])
                input_tensor = images[i].to(device)

                cam, pred_class = gradcam.generate(input_tensor)

                # resize CAM 到原始图像大小
                cam_resized = np.array(
                    Image.fromarray(cam).resize((img_w, img_h), Image.BILINEAR)
                )

                axes[i, j + 1].imshow(raw_images[i], cmap='gray', alpha=0.5)
                axes[i, j + 1].imshow(cam_resized, cmap='jet', alpha=0.5)
                if i == 0:
                    axes[i, j + 1].set_title(layer_name, fontsize=10, fontweight='bold')
                axes[i, j + 1].axis('off')
            except Exception as e:
                axes[i, j + 1].text(0.5, 0.5, f'Error\n{str(e)[:30]}',
                                    ha='center', va='center', fontsize=7,
                                    transform=axes[i, j + 1].transAxes)
                axes[i, j + 1].axis('off')

    plt.suptitle(f'Grad-CAM Layer Activations — {display_name}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, f'gradcam_layers_{model_name}.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def generate_gradcam_best_model_grid(best_model_name, model, config, device, img_h, img_w):
    """最佳模型的多样本 Grad-CAM 网格"""
    if best_model_name == 'custom_mlp':
        return

    target_layers = get_target_layers(best_model_name, model)
    if not target_layers:
        return

    # 使用最深层
    last_layer_name = list(target_layers.keys())[-1]
    last_layer = target_layers[last_layer_name]

    to_rgb = best_model_name != 'custom_mlp'
    images, raw_images, labels = load_sample_images(
        config, n_samples=12, to_rgb=to_rgb, img_h=img_h, img_w=img_w
    )

    n = min(len(images), 12)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 6, rows * 3))
    if rows == 1:
        axes = axes.reshape(1, -1)

    display_name = DISPLAY_NAMES.get(best_model_name, best_model_name)

    for i in range(n):
        r = i // cols
        c = i % cols

        try:
            gradcam = GradCAM(model, last_layer)
            input_tensor = images[i].to(device)
            cam, pred_class = gradcam.generate(input_tensor)
            cam_resized = np.array(
                Image.fromarray(cam).resize((img_w, img_h), Image.BILINEAR)
            )

            # 原图
            axes[r, c * 2].imshow(raw_images[i], cmap='gray')
            axes[r, c * 2].set_title(f'True: {labels[i]}', fontsize=9)
            axes[r, c * 2].axis('off')

            # Grad-CAM 叠加
            axes[r, c * 2 + 1].imshow(raw_images[i], cmap='gray', alpha=0.5)
            axes[r, c * 2 + 1].imshow(cam_resized, cmap='jet', alpha=0.5)
            axes[r, c * 2 + 1].set_title(f'Pred: {pred_class}', fontsize=9)
            axes[r, c * 2 + 1].axis('off')
        except Exception as e:
            axes[r, c * 2].axis('off')
            axes[r, c * 2 + 1].text(0.5, 0.5, 'Error', ha='center', va='center',
                                     transform=axes[r, c * 2 + 1].transAxes)
            axes[r, c * 2 + 1].axis('off')

    # 隐藏多余
    for i in range(n, rows * cols):
        r = i // cols
        c = i % cols
        axes[r, c * 2].set_visible(False)
        axes[r, c * 2 + 1].set_visible(False)

    plt.suptitle(f'Grad-CAM — {display_name} ({last_layer_name})',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'gradcam_best_model_grid.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# 4. 通道激活热力图 (Feature Channel Heatmap)
# =============================================================================

def generate_channel_heatmaps(model_name, model, config, device, img_h, img_w):
    """生成各层通道激活热力图"""
    if model_name == 'custom_mlp':
        print(f"  Skipping channel heatmap for {model_name}")
        return

    target_layers = get_target_layers(model_name, model)
    if not target_layers:
        return

    to_rgb = model_name != 'custom_mlp'
    images, raw_images, labels = load_sample_images(
        config, n_samples=1, to_rgb=to_rgb, img_h=img_h, img_w=img_w
    )
    if not images:
        return

    model.eval()
    input_tensor = images[0].to(device)

    # 收集各层的激活
    activations = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor) and output.dim() == 4:
                activations[name] = output.detach().cpu()
        return hook_fn

    for name, layer in target_layers.items():
        h = layer.register_forward_hook(make_hook(name))
        hooks.append(h)

    with torch.no_grad():
        _ = model(input_tensor)

    for h in hooks:
        h.remove()

    if not activations:
        return

    layer_names = list(activations.keys())
    n_layers = len(layer_names)

    # 每层显示 top-8 通道的平均激活
    n_show_channels = 8

    fig, axes = plt.subplots(n_layers, n_show_channels + 1, figsize=((n_show_channels + 1) * 2, n_layers * 2))
    if n_layers == 1:
        axes = axes.reshape(1, -1)

    display_name = DISPLAY_NAMES.get(model_name, model_name)

    for i, layer_name in enumerate(layer_names):
        act = activations[layer_name][0]  # shape: (C, H, W)
        n_channels = act.shape[0]

        # 计算每个通道的平均激活值, 选 top-N
        channel_means = act.mean(dim=(1, 2))
        top_indices = torch.argsort(channel_means, descending=True)[:n_show_channels]

        # 平均激活
        avg_act = act.mean(dim=0).numpy()
        axes[i, 0].imshow(avg_act, cmap='viridis')
        axes[i, 0].set_title(f'{layer_name}\n(avg, {n_channels}ch)', fontsize=8, fontweight='bold')
        axes[i, 0].axis('off')

        # Top-N 通道
        for j, ch_idx in enumerate(top_indices):
            ch_act = act[ch_idx].numpy()
            axes[i, j + 1].imshow(ch_act, cmap='hot')
            axes[i, j + 1].set_title(f'Ch{ch_idx.item()}\n({channel_means[ch_idx]:.2f})', fontsize=7)
            axes[i, j + 1].axis('off')

    plt.suptitle(f'Channel Activation Heatmaps — {display_name}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, f'channel_heatmap_{model_name}.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# 5. 消融实验 (Ablation Study)
# =============================================================================

def get_ablation_blocks(model_name, model):
    """获取模型可消融的模块"""
    blocks = {}

    if model_name == 'resnet50':
        backbone = model[0]
        blocks = {
            'layer1': list(backbone.layer1.parameters()),
            'layer2': list(backbone.layer2.parameters()),
            'layer3': list(backbone.layer3.parameters()),
            'layer4': list(backbone.layer4.parameters()),
            'Classification Head': list(model[1].parameters()),
        }
    elif model_name in ('vgg16', 'vgg19'):
        # VGG structure: nn.Sequential(features, pool, flatten, head)
        # model[0] IS the features Sequential, model[3] is the ClassificationHead
        feat = model[0]
        if model_name == 'vgg16':
            blocks = {
                'block1-2 (features[:10])': list(feat[:10].parameters()),
                'block3 (features[10:17])': list(feat[10:17].parameters()),
                'block4 (features[17:24])': list(feat[17:24].parameters()),
                'block5 (features[24:])': list(feat[24:].parameters()),
                'Classification Head': list(model[3].parameters()),
            }
        else:
            blocks = {
                'block1-2 (features[:10])': list(feat[:10].parameters()),
                'block3 (features[10:19])': list(feat[10:19].parameters()),
                'block4 (features[19:28])': list(feat[19:28].parameters()),
                'block5 (features[28:])': list(feat[28:].parameters()),
                'Classification Head': list(model[3].parameters()),
            }
    elif model_name == 'inception_v3':
        backbone = model[0]
        blocks = {
            'Conv2d_1-4': [p for n, p in backbone.named_parameters()
                          if any(x in n for x in ['Conv2d_1', 'Conv2d_2', 'Conv2d_3', 'Conv2d_4'])],
            'Mixed_5b-5d': [p for n, p in backbone.named_parameters()
                           if 'Mixed_5' in n],
            'Mixed_6a-6e': [p for n, p in backbone.named_parameters()
                           if 'Mixed_6' in n],
            'Mixed_7a-7c': [p for n, p in backbone.named_parameters()
                           if 'Mixed_7' in n],
            'Classification Head': list(model[1].parameters()),
        }
    elif model_name == 'efficientnet_b3':
        backbone = model[0]
        feat = backbone.features
        n_feat = len(feat)
        blocks = {
            f'features[0:{n_feat//3}]': list(feat[:n_feat//3].parameters()),
            f'features[{n_feat//3}:{2*n_feat//3}]': list(feat[n_feat//3:2*n_feat//3].parameters()),
            f'features[{2*n_feat//3}:]': list(feat[2*n_feat//3:].parameters()),
            'Classification Head': list(model[1].parameters()),
        }
    elif model_name == 'mobilenet_v3':
        backbone = model[0]
        feat = backbone.features
        n_feat = len(feat)
        blocks = {
            f'features[0:{n_feat//3}]': list(feat[:n_feat//3].parameters()),
            f'features[{n_feat//3}:{2*n_feat//3}]': list(feat[n_feat//3:2*n_feat//3].parameters()),
            f'features[{2*n_feat//3}:]': list(feat[2*n_feat//3:].parameters()),
            'Classification Head': list(model[1].parameters()),
        }
    elif model_name == 'vit_b16':
        backbone = model[0]
        if hasattr(backbone, 'encoder') and hasattr(backbone.encoder, 'layers'):
            enc = backbone.encoder.layers
            n = len(enc)
            blocks = {
                f'encoder[0:{n//3}]': list(enc[:n//3].parameters()),
                f'encoder[{n//3}:{2*n//3}]': list(enc[n//3:2*n//3].parameters()),
                f'encoder[{2*n//3}:]': list(enc[2*n//3:].parameters()),
                'Classification Head': list(model[1].parameters()),
            }

    return blocks


def run_ablation_study(model_name, model, config, device, img_h, img_w):
    """
    消融实验: 逐个冻结/置零模块, 测量准确率下降
    """
    if model_name == 'custom_mlp':
        print(f"  Skipping ablation for {model_name}")
        return None

    blocks = get_ablation_blocks(model_name, model)
    if not blocks:
        return None

    # 加载验证数据
    to_rgb = model_name != 'custom_mlp'
    test_csv = os.path.join(config.paths.data, config.data.train_mapping)
    df = pd.read_csv(test_csv)

    image_paths = []
    labels_list = []
    for _, row in df.iterrows():
        p = os.path.join(config.paths.data, 'train', row['文件名'])
        if os.path.exists(p):
            image_paths.append(p)
            labels_list.append(row['标签'] - 1)

    # 使用验证集 (30%)
    from sklearn.model_selection import train_test_split
    _, X_val, _, y_val = train_test_split(
        image_paths, labels_list, test_size=0.3, random_state=42,
        stratify=labels_list
    )

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = AncientCharDataset(
        X_val, y_val, transform,
        target_size=(img_h, img_w),
        target_is_landscape=None,
        to_rgb=to_rgb,
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    def evaluate_accuracy(mdl):
        mdl.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, lbls in loader:
                imgs = imgs.to(device)
                lbls = lbls.to(device)
                outputs = mdl(imgs)
                _, preds = outputs.max(1)
                correct += (preds == lbls).sum().item()
                total += lbls.size(0)
        return correct / total * 100 if total > 0 else 0

    # 基线准确率
    baseline_acc = evaluate_accuracy(model)
    results = {'baseline': baseline_acc}

    print(f"  Baseline accuracy: {baseline_acc:.2f}%")

    for block_name, params in blocks.items():
        if not params:
            continue

        # 保存原始参数
        original_data = [p.data.clone() for p in params]

        # 置零该模块参数
        with torch.no_grad():
            for p in params:
                p.data.zero_()

        ablated_acc = evaluate_accuracy(model)
        drop = baseline_acc - ablated_acc

        # 恢复参数
        with torch.no_grad():
            for p, orig in zip(params, original_data):
                p.data.copy_(orig)

        results[block_name] = {
            'accuracy_after_ablation': round(ablated_acc, 2),
            'accuracy_drop': round(drop, 2),
            'contribution_ratio': round(drop / baseline_acc * 100, 2) if baseline_acc > 0 else 0,
        }
        print(f"  Ablate [{block_name}]: acc={ablated_acc:.2f}% (drop={drop:.2f}%)")

    return results


def plot_ablation_results(all_ablation_results):
    """绘制消融实验结果"""
    valid_results = {k: v for k, v in all_ablation_results.items()
                     if v is not None and len(v) > 1}
    if not valid_results:
        print("  No ablation results to plot")
        return

    n_models = len(valid_results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6))
    if n_models == 1:
        axes = [axes]

    for ax, (model_name, results) in zip(axes, valid_results.items()):
        baseline = results['baseline']
        block_names = [k for k in results if k != 'baseline']
        drops = [results[k]['accuracy_drop'] for k in block_names]
        contributions = [results[k]['contribution_ratio'] for k in block_names]

        # 短化名称
        short_names = []
        for n in block_names:
            if len(n) > 25:
                n = n[:22] + '...'
            short_names.append(n)

        colors = ['#e74c3c' if d > 5 else '#f39c12' if d > 1 else '#2ecc71' for d in drops]
        bars = ax.barh(short_names, drops, color=colors, edgecolor='white', height=0.6)

        ax.set_xlabel('Accuracy Drop (%)', fontsize=11)
        ax.set_title(f'{DISPLAY_NAMES.get(model_name, model_name)}\n(baseline: {baseline:.1f}%)',
                     fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        for bar, val, contrib in zip(bars, drops, contributions):
            ax.text(val + 0.2, bar.get_y() + bar.get_height()/2,
                   f'{val:.1f}% ({contrib:.1f}%)',
                   va='center', fontsize=9, fontweight='bold')

    plt.suptitle('Ablation Study — Module Contribution Analysis',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'ablation_study.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# 6. 模块贡献热力可视化
# =============================================================================

def plot_module_contribution_heatmap(all_ablation_results):
    """各模型模块贡献比热力图"""
    valid_results = {k: v for k, v in all_ablation_results.items()
                     if v is not None and len(v) > 1}
    if not valid_results:
        return

    # 收集所有模块名
    all_blocks = set()
    for results in valid_results.values():
        for k in results:
            if k != 'baseline':
                all_blocks.add(k)
    all_blocks = sorted(all_blocks)

    # 构建矩阵
    data = []
    model_names = []
    for model_name, results in valid_results.items():
        row = []
        for block in all_blocks:
            if block in results:
                row.append(results[block]['contribution_ratio'])
            else:
                row.append(np.nan)
        data.append(row)
        model_names.append(DISPLAY_NAMES.get(model_name, model_name))

    df = pd.DataFrame(data, index=model_names, columns=all_blocks)

    # 短化列名
    short_cols = []
    for c in df.columns:
        if len(c) > 20:
            c = c[:17] + '...'
        short_cols.append(c)
    df.columns = short_cols

    fig, ax = plt.subplots(figsize=(max(12, len(all_blocks) * 1.5), len(model_names) * 0.8 + 2))
    sns.heatmap(df, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax,
                linewidths=0.5, vmin=0, mask=df.isna(),
                cbar_kws={'label': 'Contribution Ratio (%)'})
    ax.set_title('Module Contribution Ratio (%) — Ablation Results',
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12)
    ax.set_xlabel('Module', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=8)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'module_contribution_heatmap.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


# =============================================================================
# 7. 综合性能对比表 (包含所有新指标)
# =============================================================================

def plot_comprehensive_table(metrics):
    """生成包含所有性能指标的综合对比表"""
    test_data = {}
    for k in metrics:
        tf = os.path.join(PAPER_DIR, k, 'test_metrics.json')
        if os.path.exists(tf):
            with open(tf) as f:
                test_data[k] = json.load(f)

    rows = []
    for k in metrics:
        if k not in test_data:
            continue
        t = test_data[k]
        m = metrics[k]
        rows.append({
            'Model': DISPLAY_NAMES.get(k, k),
            'Params (M)': f"{m['total_params']/1e6:.1f}",
            'Size (MB)': f"{m['model_size_mb']:.1f}",
            'GFLOPs': f"{m['gflops']:.2f}" if m['gflops'] else 'N/A',
            'Latency (ms)': f"{m['avg_latency_ms']:.1f}",
            'FPS': f"{m['throughput_fps']:.0f}",
            'Acc (%)': f"{t['test_accuracy']*100:.2f}",
            'F1 (%)': f"{t['macro_f1']*100:.2f}",
        })

    if not rows:
        return

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(20, len(rows) * 0.7 + 2))
    ax.axis('off')

    table = ax.table(cellText=df.values, colLabels=df.columns,
                     cellLoc='center', loc='center',
                     colColours=['#2c3e50'] * len(df.columns))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)

    for i in range(len(df.columns)):
        table[0, i].set_text_props(color='white', fontweight='bold')

    # 高亮最佳行
    best_idx = df['Acc (%)'].astype(float).idxmax()
    for j in range(len(df.columns)):
        table[best_idx + 1, j].set_facecolor('#d5f5e3')

    ax.set_title('Comprehensive Model Performance Comparison',
                 fontsize=16, fontweight='bold', pad=20)

    path = os.path.join(FIG_DIR, 'comprehensive_performance_table.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")

    # LaTeX
    latex_path = os.path.join(FIG_DIR, 'comprehensive_performance_table.tex')
    with open(latex_path, 'w') as f:
        f.write(df.to_latex(index=False, escape=False))
    print(f"  Saved: {latex_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Paper-level model analysis')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    parser.add_argument('--skip-metrics', action='store_true', help='Skip performance metrics')
    parser.add_argument('--skip-gradcam', action='store_true', help='Skip Grad-CAM')
    parser.add_argument('--skip-ablation', action='store_true', help='Skip ablation study')
    parser.add_argument('--models', nargs='+', default=None, help='Specific models to analyze')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    config = load_config()

    # 过滤模型
    if args.models:
        model_configs = [(n, h, w) for n, h, w in MODEL_CONFIGS if n in args.models]
    else:
        # 只分析有 test_metrics.json 的模型
        model_configs = [(n, h, w) for n, h, w in MODEL_CONFIGS
                        if os.path.exists(os.path.join(PAPER_DIR, n, 'test_metrics.json'))]

    print(f"Models to analyze: {[m[0] for m in model_configs]}")

    # ── Part 1: 性能指标 ──
    metrics = {}
    if not args.skip_metrics:
        print("\n" + "=" * 70)
        print("  PART 1: MODEL PERFORMANCE METRICS")
        print("=" * 70)

        for model_name, img_h, img_w in model_configs:
            print(f"\n--- {DISPLAY_NAMES.get(model_name, model_name)} ---")

            cfg = copy.deepcopy(config)
            cfg.data.img_height = img_h
            cfg.data.img_width = img_w
            cfg.model.name = model_name

            try:
                BackboneClass = get_backbone(model_name)
                classifier = BackboneClass(cfg)
                classifier.num_classes = 6
                model = classifier.build_model()

                # 加载权重
                weight_path = os.path.join(PAPER_DIR, model_name, f'best_{model_name}.pth')
                if not os.path.exists(weight_path):
                    weight_path = os.path.join(config.paths.outputs, 'models', f'best_{model_name}.pth')
                if os.path.exists(weight_path):
                    state_dict = torch.load(weight_path, map_location=device, weights_only=True)
                    model.load_state_dict(state_dict)

                m = measure_model_metrics(model_name, img_h, img_w, model, device)
                metrics[model_name] = m
                print(f"  Latency: {m['avg_latency_ms']}ms | Size: {m['model_size_mb']}MB | "
                      f"GFLOPs: {m['gflops']} | Params: {m['total_params']:,}")
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

        # 保存 JSON
        out_path = os.path.join(FIG_DIR, 'model_perf_metrics.json')
        with open(out_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nSaved: {out_path}")

        # 可视化
        if metrics:
            print("\nGenerating performance charts...")
            plot_inference_speed(metrics)
            plot_model_size(metrics)
            plot_flops_vs_accuracy(metrics)
            plot_efficiency_radar(metrics)
            plot_comprehensive_table(metrics)
    else:
        # 从已有文件加载
        mf = os.path.join(FIG_DIR, 'model_perf_metrics.json')
        if os.path.exists(mf):
            with open(mf) as f:
                metrics = json.load(f)

    # ── Part 2: Grad-CAM & 通道热力图 ──
    if not args.skip_gradcam:
        print("\n" + "=" * 70)
        print("  PART 2: GRAD-CAM & CHANNEL HEATMAPS")
        print("=" * 70)

        best_model_name = None
        best_acc = 0

        for model_name, img_h, img_w in model_configs:
            if model_name == 'custom_mlp':
                continue

            print(f"\n--- {DISPLAY_NAMES.get(model_name, model_name)} ---")

            cfg = copy.deepcopy(config)
            cfg.data.img_height = img_h
            cfg.data.img_width = img_w
            cfg.model.name = model_name

            try:
                BackboneClass = get_backbone(model_name)
                classifier = BackboneClass(cfg)
                classifier.num_classes = 6
                model = classifier.build_model()

                weight_path = os.path.join(PAPER_DIR, model_name, f'best_{model_name}.pth')
                if not os.path.exists(weight_path):
                    weight_path = os.path.join(config.paths.outputs, 'models', f'best_{model_name}.pth')
                if os.path.exists(weight_path):
                    state_dict = torch.load(weight_path, map_location=device, weights_only=True)
                    model.load_state_dict(state_dict)

                model.to(device)
                model.eval()

                # 检查是否最佳模型
                tf = os.path.join(PAPER_DIR, model_name, 'test_metrics.json')
                if os.path.exists(tf):
                    with open(tf) as f:
                        acc = json.load(f)['test_accuracy']
                    if acc > best_acc:
                        best_acc = acc
                        best_model_name = model_name

                # Grad-CAM
                print(f"  Generating Grad-CAM...")
                generate_gradcam_for_model(model_name, model, config, device, img_h, img_w)

                # 通道热力图
                print(f"  Generating channel heatmaps...")
                generate_channel_heatmaps(model_name, model, config, device, img_h, img_w)

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

        # 最佳模型的详细 Grad-CAM 网格
        if best_model_name:
            print(f"\n--- Best model Grad-CAM grid: {DISPLAY_NAMES[best_model_name]} ---")
            bhw = [(h, w) for n, h, w in model_configs if n == best_model_name]
            if bhw:
                img_h, img_w = bhw[0]
                cfg = copy.deepcopy(config)
                cfg.data.img_height = img_h
                cfg.data.img_width = img_w
                cfg.model.name = best_model_name

                BackboneClass = get_backbone(best_model_name)
                classifier = BackboneClass(cfg)
                classifier.num_classes = 6
                model = classifier.build_model()

                weight_path = os.path.join(PAPER_DIR, best_model_name, f'best_{best_model_name}.pth')
                if not os.path.exists(weight_path):
                    weight_path = os.path.join(config.paths.outputs, 'models', f'best_{best_model_name}.pth')
                if os.path.exists(weight_path):
                    state_dict = torch.load(weight_path, map_location=device, weights_only=True)
                    model.load_state_dict(state_dict)

                model.to(device)
                model.eval()
                generate_gradcam_best_model_grid(best_model_name, model, config, device, img_h, img_w)

    # ── Part 3: 消融实验 ──
    if not args.skip_ablation:
        print("\n" + "=" * 70)
        print("  PART 3: ABLATION STUDY")
        print("=" * 70)

        all_ablation = {}
        for model_name, img_h, img_w in model_configs:
            if model_name == 'custom_mlp':
                continue

            print(f"\n--- {DISPLAY_NAMES.get(model_name, model_name)} ---")

            cfg = copy.deepcopy(config)
            cfg.data.img_height = img_h
            cfg.data.img_width = img_w
            cfg.model.name = model_name

            try:
                BackboneClass = get_backbone(model_name)
                classifier = BackboneClass(cfg)
                classifier.num_classes = 6
                model = classifier.build_model()

                weight_path = os.path.join(PAPER_DIR, model_name, f'best_{model_name}.pth')
                if not os.path.exists(weight_path):
                    weight_path = os.path.join(config.paths.outputs, 'models', f'best_{model_name}.pth')
                if os.path.exists(weight_path):
                    state_dict = torch.load(weight_path, map_location=device, weights_only=True)
                    model.load_state_dict(state_dict)

                model.to(device)
                results = run_ablation_study(model_name, model, config, device, img_h, img_w)
                all_ablation[model_name] = results

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

        if all_ablation:
            print("\nGenerating ablation charts...")
            plot_ablation_results(all_ablation)
            plot_module_contribution_heatmap(all_ablation)

            # 保存 JSON
            ablation_path = os.path.join(FIG_DIR, 'ablation_results.json')
            with open(ablation_path, 'w') as f:
                json.dump(all_ablation, f, indent=2, default=str)
            print(f"Saved: {ablation_path}")

    print("\n" + "=" * 70)
    print("  ALL ANALYSES COMPLETE!")
    print(f"  Figures saved to: {FIG_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
