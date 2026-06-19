#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
预下载 5 个 SOTA timm backbone 的预训练权重 (一次性).

后续 DDP 训练时所有 rank 都从本地 cache 读, 避免 4 进程并发下载.

用法:
    python scripts/prefetch_timm_weights.py             # 全部下
    python scripts/prefetch_timm_weights.py --models convnext_tiny swin_tiny

如果某个权重下载失败, 不阻塞其他模型继续.
"""

import argparse
import sys
import time

MODELS = {
    "convnext_tiny":      "convnext_tiny.fb_in22k_ft_in1k",
    "swin_tiny":          "swin_tiny_patch4_window7_224.ms_in1k",
    "maxvit_tiny":        "maxvit_tiny_tf_224.in1k",
    "efficientnetv2_s":   "tf_efficientnetv2_s.in21k_ft_in1k",
    "regnety_032":        "regnety_032.tv2_in1k",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*", default=None,
                   help="选择要拉取的模型, 默认全部")
    args = p.parse_args()

    import timm

    targets = args.models or list(MODELS.keys())
    failed = []

    for name in targets:
        if name not in MODELS:
            print(f"[skip] unknown model: {name} (available: {list(MODELS)})")
            continue
        timm_name = MODELS[name]
        print(f"\n[fetch] {name} = {timm_name}")
        t0 = time.time()
        try:
            m = timm.create_model(timm_name, pretrained=True, num_classes=0)
            n_params = sum(p.numel() for p in m.parameters())
            print(f"  ok ({time.time() - t0:.1f}s)  feat_dim={m.num_features}  "
                  f"params={n_params:,}")
        except Exception as e:
            print(f"  FAILED: {e}")
            failed.append(name)

    print("\n" + "=" * 50)
    print(f"  Done. Cached: {len(targets) - len(failed)}/{len(targets)}")
    if failed:
        print(f"  Failed: {failed}")
        print(f"  (可以在网络好时重试, 或者从 https://huggingface.co/timm 手动下载)")
        sys.exit(1)


if __name__ == "__main__":
    main()
