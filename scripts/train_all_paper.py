#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
论文级全量训练脚本 — 逐个训练所有模型并收集详细信息

输出:
  outputs/paper/
    summary.json          — 所有模型的汇总指标 (test_acc, test_loss, params, 训练时间 …)
    <model>/
      training_log.json   — 逐 epoch 的 loss/accuracy/val_loss/val_accuracy
      test_metrics.json   — 测试集 classification_report (per-class P/R/F1) + confusion_matrix
      training_history.png
      confusion_matrix.png
      predictions.png
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)

from src.utils.config import load_config
from src.utils.paths import paths
from src.models.backbones import get_backbone, list_backbones

# ── 模型列表 & 对应输入尺寸 ────────────────────────────────
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


def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    return trainable, total


def detailed_evaluate(classifier, paper_dir, model_name):
    """在测试集上做详细推理, 返回 dict 并保存 JSON / 图"""
    import pandas as pd
    from torch.utils.data import DataLoader
    from src.models.base_classifier import AncientCharDataset
    import torch.nn as nn
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    test_dir = os.path.join(classifier.data_dir, 'test')
    test_csv = os.path.join(classifier.data_dir, classifier.config.data.test_mapping)
    df = pd.read_csv(test_csv)

    image_paths, labels = [], []
    for _, row in df.iterrows():
        p = os.path.join(test_dir, row['文件名'])
        if os.path.exists(p):
            image_paths.append(p)
            labels.append(row['标签'] - 1)

    test_loader = classifier.create_dataset(
        np.array(image_paths), np.array(labels), is_training=False
    )

    device = classifier.device
    classifier.model.to(device)
    classifier.model.eval()

    criterion = nn.CrossEntropyLoss()
    all_preds, all_labels, all_probs = [], [], []
    total_loss, total_samples = 0.0, 0

    with torch.no_grad():
        for images, batch_labels in test_loader:
            images = images.to(device)
            batch_labels = batch_labels.to(device)
            outputs = classifier.model(images)
            loss = criterion(outputs, batch_labels)
            probs = torch.softmax(outputs, dim=1)

            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)
    test_loss = total_loss / total_samples
    test_acc  = accuracy_score(y_true, y_pred)

    # per-class 指标
    target_names = [f"Class_{i}" for i in range(classifier.num_classes)]
    report_dict  = classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True
    )
    report_str   = classification_report(
        y_true, y_pred, target_names=target_names
    )
    cm = confusion_matrix(y_true, y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro'
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "per_class": {},
        "confusion_matrix": cm.tolist(),
        "classification_report_text": report_str,
    }
    for i, name in enumerate(target_names):
        metrics["per_class"][name] = {
            "precision": float(precision[i]),
            "recall":    float(recall[i]),
            "f1":        float(f1[i]),
            "support":   int(support[i]),
        }

    # 保存 JSON
    model_dir = os.path.join(paper_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # ── 混淆矩阵 (更美观) ──
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Confusion Matrix — {model_name}', fontsize=14)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "confusion_matrix.png"), dpi=300)
    plt.close()

    # ── 打印 ──
    print(f"\n{'='*60}")
    print(f" Test Results for {model_name}")
    print(f"{'='*60}")
    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}  ({test_acc*100:.2f}%)")
    print(f"  Macro    P/R/F1: {macro_p:.4f} / {macro_r:.4f} / {macro_f1:.4f}")
    print(f"  Weighted P/R/F1: {weighted_p:.4f} / {weighted_r:.4f} / {weighted_f1:.4f}")
    print(f"\n{report_str}")
    print(f"Confusion Matrix:\n{cm}\n")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", "-e", type=int, default=None)
    parser.add_argument("--batch-size", "-b", type=int, default=None)
    parser.add_argument("--models", nargs="*", default=None,
                        help="指定要训练的模型名(空=全部)")
    args = parser.parse_args()

    paths.ensure_dirs()
    paper_dir = os.path.join(str(paths.outputs_dir), "paper")
    os.makedirs(paper_dir, exist_ok=True)

    # 选择模型
    if args.models:
        selected = [(n, h, w) for n, h, w in MODEL_CONFIGS if n in args.models]
    else:
        selected = MODEL_CONFIGS

    # ── 环境信息 ──
    env_info = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "gpu_count": torch.cuda.device_count(),
    }
    print("\n" + "=" * 70)
    print("  PAPER-LEVEL FULL BENCHMARK")
    print("=" * 70)
    for k, v in env_info.items():
        print(f"  {k}: {v}")
    print(f"  Models to train: {[n for n,_,_ in selected]}")
    print("=" * 70 + "\n")

    summary = {"environment": env_info, "models": {}}

    for model_name, height, width in selected:
        print("\n" + "#" * 70)
        print(f"##  {model_name}  ({height}x{width})")
        print("#" * 70)

        overrides = {"model": {"name": model_name},
                     "data": {"img_height": height, "img_width": width}}
        if args.epochs:
            overrides["training"] = {"epochs": args.epochs}
        if args.batch_size:
            overrides.setdefault("training", {})["batch_size"] = args.batch_size

        config = load_config(overrides=overrides)
        Cls = get_backbone(model_name)
        classifier = Cls(config)

        # 1) load data
        X_train, X_val, y_train, y_val = classifier.load_data()

        # 2) build model
        model = classifier.build_model()
        trainable_params, total_params = count_params(model)
        print(f"\n  Trainable params : {trainable_params:>12,}")
        print(f"  Total params     : {total_params:>12,}")

        # 3) train
        t0 = time.time()
        history = classifier.train(X_train, X_val, y_train, y_val)
        train_time = time.time() - t0
        actual_epochs = len(history['loss'])
        print(f"\n  Training time: {train_time:.1f}s  ({actual_epochs} epochs)")

        # 保存逐 epoch 日志
        model_dir = os.path.join(paper_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "training_log.json"), "w") as f:
            json.dump({
                "epochs_completed": actual_epochs,
                "training_time_seconds": round(train_time, 2),
                "history": {k: [float(v) for v in vals] for k, vals in history.items()},
            }, f, indent=2)

        # 4) plot training history (保存到 paper 目录)
        classifier.plot_training_history(
            save_path=os.path.join(model_dir, "training_history.png")
        )

        # 5) detailed test evaluation
        test_metrics = detailed_evaluate(classifier, paper_dir, model_name)

        # 6) prediction samples
        classifier.predict_sample_images(
            save_path=os.path.join(model_dir, "predictions.png")
        )

        # 7) save model
        classifier.save_model()

        # 8) 写入 summary
        best_val_acc = max(history['val_accuracy'])
        best_val_epoch = int(np.argmax(history['val_accuracy'])) + 1
        summary["models"][model_name] = {
            "input_size": f"{height}x{width}",
            "trainable_params": trainable_params,
            "total_params": total_params,
            "epochs_completed": actual_epochs,
            "best_val_accuracy": float(best_val_acc),
            "best_val_epoch": best_val_epoch,
            "training_time_seconds": round(train_time, 2),
            "test_loss": test_metrics["test_loss"],
            "test_accuracy": test_metrics["test_accuracy"],
            "macro_precision": test_metrics["macro_precision"],
            "macro_recall": test_metrics["macro_recall"],
            "macro_f1": test_metrics["macro_f1"],
            "weighted_precision": test_metrics["weighted_precision"],
            "weighted_recall": test_metrics["weighted_recall"],
            "weighted_f1": test_metrics["weighted_f1"],
        }

        # 打印阶段性摘要
        s = summary["models"][model_name]
        print(f"\n  ┌─ {model_name} Summary ─────────────────────────")
        print(f"  │ Total/Trainable params : {s['total_params']:,} / {s['trainable_params']:,}")
        print(f"  │ Epochs completed       : {s['epochs_completed']}")
        print(f"  │ Best val accuracy      : {s['best_val_accuracy']:.4f} (epoch {s['best_val_epoch']})")
        print(f"  │ Training time          : {s['training_time_seconds']:.1f}s")
        print(f"  │ Test accuracy          : {s['test_accuracy']:.4f}")
        print(f"  │ Macro F1               : {s['macro_f1']:.4f}")
        print(f"  │ Weighted F1            : {s['weighted_f1']:.4f}")
        print(f"  └──────────────────────────────────────────────")

        # 清理 GPU 缓存
        del model, classifier
        torch.cuda.empty_cache()

    # ── 最终汇总 ──
    with open(os.path.join(paper_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n\n" + "=" * 80)
    print("  FINAL COMPARISON TABLE")
    print("=" * 80)
    header = (f"{'Model':<22} {'Input':>9} {'Params':>12} {'Train(s)':>9} "
              f"{'TestAcc':>8} {'MacroF1':>8} {'WtdF1':>8}")
    print(header)
    print("-" * 80)
    for name, info in summary["models"].items():
        print(f"{name:<22} {info['input_size']:>9} "
              f"{info['total_params']:>12,} {info['training_time_seconds']:>9.1f} "
              f"{info['test_accuracy']:>8.4f} {info['macro_f1']:>8.4f} "
              f"{info['weighted_f1']:>8.4f}")
    print("=" * 80)
    print(f"\nAll results saved to: {paper_dir}/")
    print(f"  summary.json          — 汇总表")
    print(f"  <model>/training_log.json   — 逐epoch记录")
    print(f"  <model>/test_metrics.json   — 测试集详细指标 + confusion matrix")
    print(f"  <model>/*.png               — 训练曲线/混淆矩阵/预测样例")


if __name__ == "__main__":
    main()
