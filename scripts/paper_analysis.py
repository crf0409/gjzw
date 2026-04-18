#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
论文级全面分析与可视化脚本

输出:  outputs/paper/figures/
  1.  model_comparison_bar.png       — 各模型 Test Accuracy / Macro F1 / Weighted F1 对比柱状图
  2.  training_curves_grid.png       — 所有模型训练曲线 (loss + accuracy) 网格图
  3.  confusion_matrices_grid.png    — 所有模型混淆矩阵网格
  4.  per_class_f1_heatmap.png       — 各模型 × 各类别 F1 热力图
  5.  per_class_precision_recall.png — 各模型 per-class P/R 分组柱状图
  6.  params_vs_accuracy.png         — 参数量 vs 精度 散点图
  7.  training_time_vs_accuracy.png  — 训练时间 vs 精度 散点图
  8.  best_model_predictions.png     — 最佳模型的推理可视化
  9.  error_analysis.png             — 最佳模型的错误分析（误分类样本）
  10. roc_curves.png                 — 各模型 ROC 曲线 (若支持)
  11. model_summary_table.png        — LaTeX 风格汇总表格图
  12. learning_rate_curves.png       — 学习率变化曲线 (从 training_log)
"""

import json
import os
import sys
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── 配置 ──
PAPER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'outputs', 'paper')
FIG_DIR = os.path.join(PAPER_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# 中文支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# 模型显示名映射（论文用）
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

# 模型颜色
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


def load_all_data():
    """加载所有模型的数据"""
    models = {}
    for d in sorted(os.listdir(PAPER_DIR)):
        model_dir = os.path.join(PAPER_DIR, d)
        test_file = os.path.join(model_dir, 'test_metrics.json')
        log_file  = os.path.join(model_dir, 'training_log.json')
        if os.path.isdir(model_dir) and os.path.exists(test_file):
            with open(test_file) as f:
                test_data = json.load(f)
            log_data = None
            if os.path.exists(log_file):
                with open(log_file) as f:
                    log_data = json.load(f)
            models[d] = {
                'test': test_data,
                'log': log_data,
                'display_name': DISPLAY_NAMES.get(d, d),
            }
    return models


def plot_model_comparison_bar(models):
    """图1: 各模型指标对比柱状图"""
    names = [models[k]['display_name'] for k in models]
    test_acc = [models[k]['test']['test_accuracy'] * 100 for k in models]
    macro_f1 = [models[k]['test']['macro_f1'] * 100 for k in models]
    weighted_f1 = [models[k]['test']['weighted_f1'] * 100 for k in models]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width, test_acc, width, label='Test Accuracy (%)', color='#3498db', edgecolor='white')
    bars2 = ax.bar(x, macro_f1, width, label='Macro F1 (%)', color='#2ecc71', edgecolor='white')
    bars3 = ax.bar(x + width, weighted_f1, width, label='Weighted F1 (%)', color='#e74c3c', edgecolor='white')

    ax.set_xlabel('Model Architecture', fontsize=13)
    ax.set_ylabel('Score (%)', fontsize=13)
    ax.set_title('Model Performance Comparison', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=10)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)

    # 在柱状图上标注数值
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'model_comparison_bar.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_training_curves_grid(models):
    """图2: 所有模型训练曲线网格图"""
    model_keys = [k for k in models if models[k]['log'] is not None]
    n = len(model_keys)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 10, rows * 4))
    if rows == 1:
        axes = axes.reshape(1, -1)

    for i, key in enumerate(model_keys):
        r = i // cols
        c = i % cols
        log = models[key]['log']['history']
        epochs = range(1, len(log['loss']) + 1)
        name = models[key]['display_name']

        # Accuracy subplot
        ax_acc = axes[r, c * 2]
        ax_acc.plot(epochs, [v*100 for v in log['accuracy']], 'b-', linewidth=1.5, label='Train')
        ax_acc.plot(epochs, [v*100 for v in log['val_accuracy']], 'r-', linewidth=1.5, label='Val')
        ax_acc.set_title(f'{name} — Accuracy', fontsize=10, fontweight='bold')
        ax_acc.set_xlabel('Epoch', fontsize=8)
        ax_acc.set_ylabel('Accuracy (%)', fontsize=8)
        ax_acc.legend(fontsize=7)
        ax_acc.grid(alpha=0.3)

        # Loss subplot
        ax_loss = axes[r, c * 2 + 1]
        ax_loss.plot(epochs, log['loss'], 'b-', linewidth=1.5, label='Train')
        ax_loss.plot(epochs, log['val_loss'], 'r-', linewidth=1.5, label='Val')
        ax_loss.set_title(f'{name} — Loss', fontsize=10, fontweight='bold')
        ax_loss.set_xlabel('Epoch', fontsize=8)
        ax_loss.set_ylabel('Loss', fontsize=8)
        ax_loss.legend(fontsize=7)
        ax_loss.grid(alpha=0.3)

    # 隐藏多余的subplot
    for i in range(n, rows * cols):
        r = i // cols
        c = i % cols
        axes[r, c * 2].set_visible(False)
        axes[r, c * 2 + 1].set_visible(False)

    plt.suptitle('Training Curves for All Models', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'training_curves_grid.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_confusion_matrices_grid(models):
    """图3: 所有模型混淆矩阵网格"""
    model_keys = list(models.keys())
    n = len(model_keys)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4.5))
    if rows == 1:
        axes = axes.reshape(1, -1)

    for i, key in enumerate(model_keys):
        r = i // cols
        c = i % cols
        cm = np.array(models[key]['test']['confusion_matrix'])
        name = models[key]['display_name']
        acc = models[key]['test']['test_accuracy'] * 100

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[r, c],
                    xticklabels=[f'C{j}' for j in range(cm.shape[0])],
                    yticklabels=[f'C{j}' for j in range(cm.shape[0])],
                    cbar=False)
        axes[r, c].set_title(f'{name}\n(Acc: {acc:.1f}%)', fontsize=10, fontweight='bold')
        axes[r, c].set_xlabel('Predicted', fontsize=8)
        axes[r, c].set_ylabel('True', fontsize=8)

    for i in range(n, rows * cols):
        r = i // cols
        c = i % cols
        axes[r, c].set_visible(False)

    plt.suptitle('Confusion Matrices — All Models', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'confusion_matrices_grid.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_per_class_f1_heatmap(models):
    """图4: 各模型 × 各类别 F1 热力图"""
    model_keys = list(models.keys())
    class_names = list(models[model_keys[0]]['test']['per_class'].keys())

    data = []
    for key in model_keys:
        row = []
        for cls in class_names:
            row.append(models[key]['test']['per_class'][cls]['f1'] * 100)
        data.append(row)

    df = pd.DataFrame(data,
                      index=[models[k]['display_name'] for k in model_keys],
                      columns=[c.replace('Class_', 'C') for c in class_names])

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(df, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax,
                vmin=0, vmax=100, linewidths=0.5)
    ax.set_title('Per-Class F1 Score (%) — All Models', fontsize=14, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12)
    ax.set_xlabel('Class', fontsize=12)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'per_class_f1_heatmap.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_per_class_precision_recall(models):
    """图5: 各模型 per-class Precision & Recall 分组柱状图"""
    model_keys = list(models.keys())
    class_names = list(models[model_keys[0]]['test']['per_class'].keys())
    n_classes = len(class_names)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax_idx, metric in enumerate(['precision', 'recall']):
        data = {}
        for key in model_keys:
            vals = [models[key]['test']['per_class'][cls][metric] * 100 for cls in class_names]
            data[models[key]['display_name']] = vals

        df = pd.DataFrame(data, index=[c.replace('Class_', 'Class ') for c in class_names])
        df.plot(kind='bar', ax=axes[ax_idx], width=0.8, edgecolor='white')
        axes[ax_idx].set_title(f'Per-Class {metric.capitalize()} (%)', fontsize=13, fontweight='bold')
        axes[ax_idx].set_ylabel(f'{metric.capitalize()} (%)', fontsize=11)
        axes[ax_idx].set_xlabel('Class', fontsize=11)
        axes[ax_idx].set_ylim(0, 110)
        axes[ax_idx].legend(fontsize=7, loc='lower right', ncol=2)
        axes[ax_idx].grid(axis='y', alpha=0.3)
        axes[ax_idx].tick_params(axis='x', rotation=0)

    plt.suptitle('Per-Class Precision & Recall Across Models', fontsize=15, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'per_class_precision_recall.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_params_vs_accuracy(models):
    """图6: 参数量 vs 精度 散点图"""
    fig, ax = plt.subplots(figsize=(10, 7))

    for key in models:
        log = models[key].get('log')
        if log is None:
            continue
        total_params = None
        # 尝试从 summary.json 获取
        summary_file = os.path.join(PAPER_DIR, 'summary.json')
        if os.path.exists(summary_file):
            with open(summary_file) as f:
                summary = json.load(f)
            if key in summary.get('models', {}):
                total_params = summary['models'][key].get('total_params')

        if total_params is None:
            continue

        acc = models[key]['test']['test_accuracy'] * 100
        name = models[key]['display_name']
        color = MODEL_COLORS.get(key, '#333333')

        ax.scatter(total_params / 1e6, acc, s=150, c=color, edgecolors='black',
                   linewidth=1, zorder=5)
        ax.annotate(name, (total_params / 1e6, acc), textcoords='offset points',
                    xytext=(8, 5), fontsize=9)

    ax.set_xlabel('Total Parameters (M)', fontsize=13)
    ax.set_ylabel('Test Accuracy (%)', fontsize=13)
    ax.set_title('Model Parameters vs. Test Accuracy', fontsize=15, fontweight='bold')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'params_vs_accuracy.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_training_time_vs_accuracy(models):
    """图7: 训练时间 vs 精度 散点图"""
    fig, ax = plt.subplots(figsize=(10, 7))

    for key in models:
        log = models[key].get('log')
        if log is None:
            continue
        train_time = log.get('training_time_seconds', 0) / 60.0  # 转为分钟
        acc = models[key]['test']['test_accuracy'] * 100
        name = models[key]['display_name']
        color = MODEL_COLORS.get(key, '#333333')

        ax.scatter(train_time, acc, s=150, c=color, edgecolors='black',
                   linewidth=1, zorder=5)
        ax.annotate(name, (train_time, acc), textcoords='offset points',
                    xytext=(8, 5), fontsize=9)

    ax.set_xlabel('Training Time (minutes)', fontsize=13)
    ax.set_ylabel('Test Accuracy (%)', fontsize=13)
    ax.set_title('Training Time vs. Test Accuracy', fontsize=15, fontweight='bold')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'training_time_vs_accuracy.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_best_model_predictions(models):
    """图8: 最佳模型的推理可视化 — 使用已保存的 predictions.png"""
    # 找最佳模型
    best_key = max(models, key=lambda k: models[k]['test']['test_accuracy'])
    src_path = os.path.join(PAPER_DIR, best_key, 'predictions.png')
    if os.path.exists(src_path):
        import shutil
        dst_path = os.path.join(FIG_DIR, 'best_model_predictions.png')
        shutil.copy2(src_path, dst_path)
        print(f"  Saved: {dst_path} (best model: {models[best_key]['display_name']})")
    else:
        print(f"  Warning: predictions.png not found for {best_key}")


def plot_error_analysis(models):
    """图9: 各模型误分类数量分析"""
    model_keys = list(models.keys())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左图: 各模型的错误数量
    names = [models[k]['display_name'] for k in model_keys]
    cm_list = [np.array(models[k]['test']['confusion_matrix']) for k in model_keys]
    total_samples = cm_list[0].sum()

    errors = [total_samples - np.trace(cm) for cm in cm_list]
    correct = [np.trace(cm) for cm in cm_list]

    y_pos = np.arange(len(names))
    axes[0].barh(y_pos, errors, color='#e74c3c', edgecolor='white', height=0.6)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(names, fontsize=10)
    axes[0].set_xlabel('Number of Misclassifications', fontsize=12)
    axes[0].set_title('Misclassification Count by Model', fontsize=13, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)
    for i, v in enumerate(errors):
        axes[0].text(v + 1, i, str(int(v)), va='center', fontsize=10, fontweight='bold')

    # 右图: 最佳模型的逐类别错误分布
    best_key = max(models, key=lambda k: models[k]['test']['test_accuracy'])
    cm = np.array(models[best_key]['test']['confusion_matrix'])
    n_classes = cm.shape[0]
    class_errors = []
    for i in range(n_classes):
        total_in_class = cm[i].sum()
        correct_in_class = cm[i, i]
        class_errors.append(total_in_class - correct_in_class)

    class_labels = [f'Class {i}' for i in range(n_classes)]
    colors = ['#2ecc71' if e == 0 else '#e74c3c' if e > 2 else '#f39c12' for e in class_errors]
    axes[1].bar(class_labels, class_errors, color=colors, edgecolor='white')
    axes[1].set_xlabel('Class', fontsize=12)
    axes[1].set_ylabel('Number of Errors', fontsize=12)
    axes[1].set_title(f'Per-Class Errors — {models[best_key]["display_name"]}', fontsize=13, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(class_errors):
        axes[1].text(i, v + 0.1, str(int(v)), ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'error_analysis.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_best_model_confusion_heatmap(models):
    """图10: 最佳模型的详细混淆矩阵热力图（归一化 + 原始数值）"""
    best_key = max(models, key=lambda k: models[k]['test']['test_accuracy'])
    cm = np.array(models[best_key]['test']['confusion_matrix'])
    name = models[best_key]['display_name']
    acc = models[best_key]['test']['test_accuracy'] * 100

    # 归一化
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    n = cm.shape[0]
    class_labels = [f'Class {i}' for i in range(n)]

    # 左: 原始计数
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=class_labels, yticklabels=class_labels,
                linewidths=0.5, cbar_kws={'label': 'Count'})
    axes[0].set_title(f'{name} — Confusion Matrix (Counts)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predicted Label', fontsize=11)
    axes[0].set_ylabel('True Label', fontsize=11)

    # 右: 归一化 (%)
    annot = np.array([[f'{v:.1f}%' for v in row] for row in cm_norm])
    sns.heatmap(cm_norm, annot=annot, fmt='', cmap='YlOrRd', ax=axes[1],
                xticklabels=class_labels, yticklabels=class_labels,
                linewidths=0.5, vmin=0, vmax=100,
                cbar_kws={'label': 'Percentage (%)'})
    axes[1].set_title(f'{name} — Confusion Matrix (Normalized %)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Predicted Label', fontsize=11)
    axes[1].set_ylabel('True Label', fontsize=11)

    plt.suptitle(f'Best Model: {name} (Test Accuracy: {acc:.2f}%)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'best_model_confusion_heatmap.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_model_summary_table(models):
    """图11: LaTeX 风格汇总表格（含性能指标）"""
    # 尝试加载性能指标
    perf_metrics = {}
    perf_file = os.path.join(FIG_DIR, 'model_perf_metrics.json')
    if os.path.exists(perf_file):
        with open(perf_file) as f:
            perf_metrics = json.load(f)

    rows = []
    for key in models:
        t = models[key]['test']
        log = models[key].get('log', {})
        epochs = log.get('epochs_completed', '?') if log else '?'
        train_time = f"{log.get('training_time_seconds', 0)/60:.1f}" if log else '?'

        # 参数量和性能指标
        perf = perf_metrics.get(key, {})
        total_params = f"{perf['total_params']/1e6:.1f}M" if perf.get('total_params') else '?'
        model_size = f"{perf['model_size_mb']:.1f}" if perf.get('model_size_mb') else '?'
        gflops = f"{perf['gflops']:.2f}" if perf.get('gflops') else '?'
        latency = f"{perf['avg_latency_ms']:.1f}" if perf.get('avg_latency_ms') else '?'
        fps = f"{perf['throughput_fps']:.0f}" if perf.get('throughput_fps') else '?'

        rows.append({
            'Model': models[key]['display_name'],
            'Params': total_params,
            'Size (MB)': model_size,
            'GFLOPs': gflops,
            'Latency (ms)': latency,
            'FPS': fps,
            'Epochs': epochs,
            'Time (min)': train_time,
            'Test Acc (%)': f"{t['test_accuracy']*100:.2f}",
            'Macro F1 (%)': f"{t['macro_f1']*100:.2f}",
            'Wtd F1 (%)': f"{t['weighted_f1']*100:.2f}",
        })

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(22, len(rows) * 0.6 + 2))
    ax.axis('off')

    table = ax.table(cellText=df.values, colLabels=df.columns,
                     cellLoc='center', loc='center',
                     colColours=['#3498db'] * len(df.columns))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)

    # 设置表头样式
    for i in range(len(df.columns)):
        table[0, i].set_text_props(color='white', fontweight='bold')

    # 高亮最佳行
    best_idx = df['Test Acc (%)'].astype(float).idxmax()
    for j in range(len(df.columns)):
        table[best_idx + 1, j].set_facecolor('#d5f5e3')

    ax.set_title('Model Performance Summary', fontsize=16, fontweight='bold', pad=20)

    path = os.path.join(FIG_DIR, 'model_summary_table.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")

    # 同时输出 LaTeX 表格
    latex_path = os.path.join(FIG_DIR, 'model_summary_table.tex')
    with open(latex_path, 'w') as f:
        f.write(df.to_latex(index=False, escape=False))
    print(f"  Saved: {latex_path}")


def plot_convergence_comparison(models):
    """图12: 各模型验证精度收敛曲线对比"""
    fig, ax = plt.subplots(figsize=(12, 7))

    for key in models:
        log = models[key].get('log')
        if log is None:
            continue
        val_acc = [v * 100 for v in log['history']['val_accuracy']]
        epochs = range(1, len(val_acc) + 1)
        color = MODEL_COLORS.get(key, '#333333')
        ax.plot(epochs, val_acc, linewidth=2, label=models[key]['display_name'],
                color=color)

    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=13)
    ax.set_title('Validation Accuracy Convergence — All Models', fontsize=15, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'convergence_comparison.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def plot_loss_comparison(models):
    """图13: 各模型验证损失对比"""
    fig, ax = plt.subplots(figsize=(12, 7))

    for key in models:
        log = models[key].get('log')
        if log is None:
            continue
        val_loss = log['history']['val_loss']
        epochs = range(1, len(val_loss) + 1)
        color = MODEL_COLORS.get(key, '#333333')
        ax.plot(epochs, val_loss, linewidth=2, label=models[key]['display_name'],
                color=color)

    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Validation Loss', fontsize=13)
    ax.set_title('Validation Loss Convergence — All Models', fontsize=15, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'loss_comparison.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved: {path}")


def generate_latex_summary(models):
    """生成论文可用的纯文本汇总"""
    path = os.path.join(FIG_DIR, 'paper_results_summary.txt')
    with open(path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PAPER RESULTS SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        # 排序：按 test accuracy 降序
        sorted_keys = sorted(models.keys(),
                             key=lambda k: models[k]['test']['test_accuracy'],
                             reverse=True)

        for rank, key in enumerate(sorted_keys, 1):
            t = models[key]['test']
            name = models[key]['display_name']
            f.write(f"#{rank}  {name}\n")
            f.write(f"  Test Accuracy:       {t['test_accuracy']*100:.2f}%\n")
            f.write(f"  Macro Precision:     {t['macro_precision']*100:.2f}%\n")
            f.write(f"  Macro Recall:        {t['macro_recall']*100:.2f}%\n")
            f.write(f"  Macro F1:            {t['macro_f1']*100:.2f}%\n")
            f.write(f"  Weighted Precision:  {t['weighted_precision']*100:.2f}%\n")
            f.write(f"  Weighted Recall:     {t['weighted_recall']*100:.2f}%\n")
            f.write(f"  Weighted F1:         {t['weighted_f1']*100:.2f}%\n")
            f.write(f"  Test Loss:           {t['test_loss']:.4f}\n")
            f.write(f"\n  Per-class F1:\n")
            for cls, vals in t['per_class'].items():
                f.write(f"    {cls}: P={vals['precision']*100:.1f}% R={vals['recall']*100:.1f}% "
                        f"F1={vals['f1']*100:.1f}% (n={vals['support']})\n")
            f.write(f"\n  Classification Report:\n{t['classification_report_text']}\n")
            f.write("-" * 80 + "\n\n")

    print(f"  Saved: {path}")


def main():
    print("=" * 70)
    print("  PAPER-LEVEL ANALYSIS & VISUALIZATION")
    print("=" * 70)

    models = load_all_data()
    print(f"\nLoaded {len(models)} models: {list(models.keys())}\n")

    if not models:
        print("No model data found. Run train_all_paper.py first.")
        return

    print("Generating figures...")
    plot_model_comparison_bar(models)
    plot_training_curves_grid(models)
    plot_confusion_matrices_grid(models)
    plot_per_class_f1_heatmap(models)
    plot_per_class_precision_recall(models)
    plot_params_vs_accuracy(models)
    plot_training_time_vs_accuracy(models)
    plot_best_model_predictions(models)
    plot_error_analysis(models)
    plot_best_model_confusion_heatmap(models)
    plot_model_summary_table(models)
    plot_convergence_comparison(models)
    plot_loss_comparison(models)
    generate_latex_summary(models)

    print(f"\nAll figures saved to: {FIG_DIR}/")
    print("Done!")


if __name__ == "__main__":
    main()
