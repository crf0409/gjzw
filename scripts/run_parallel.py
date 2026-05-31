#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
并行训练脚本 - 在多个GPU上同时运行不同模型

使用方法:
    python scripts/run_parallel.py                    # 运行默认实验
    python scripts/run_parallel.py --batch 2         # 运行第2批实验
    python scripts/run_parallel.py --models vgg19 efficientnet_b3  # 指定模型
"""

import subprocess
import os
import sys
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Python解释器路径
PYTHON = sys.executable

# 实验批次配置
BATCH_1 = [
    (0, "resnet50"),
    (1, "vgg16"),
    (2, "inception_v3"),
    (3, "custom_mlp"),
]

BATCH_2 = [
    (0, "vgg19"),
    (1, "inception_resnet_v2"),
    (2, "efficientnet_b3"),
    (3, "mobilenet_v3"),
]

BATCH_3 = [
    (0, "vit_b16"),
]


def run_experiment(gpu_id, model_name):
    """在指定GPU上运行实验"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        PYTHON,
        "scripts/benchmark.py",
        "--model", model_name,
        "--data-dir", "data/processed/images"
    ]

    log_file = f"outputs/logs/{model_name}_gpu{gpu_id}.log"
    os.makedirs("outputs/logs", exist_ok=True)

    print(f"Starting {model_name} on GPU {gpu_id}...")
    print(f"  Log: {log_file}")

    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )

    return process, model_name, gpu_id, log_file


def run_batch(experiments):
    """运行一批实验"""
    print(f"\nRunning {len(experiments)} experiments in parallel:\n")

    processes = []
    for gpu_id, model_name in experiments:
        proc, name, gpu, log = run_experiment(gpu_id, model_name)
        processes.append((proc, name, gpu, log))

    print("\n" + "=" * 60)
    print("Waiting for completion...")
    print("=" * 60)

    # 等待所有进程完成
    results = []
    for proc, name, gpu, log in processes:
        proc.wait()
        status = "completed" if proc.returncode == 0 else f"failed (code {proc.returncode})"
        print(f"  {name} on GPU {gpu}: {status}")
        results.append((name, gpu, proc.returncode, log))

    return results


def main():
    parser = argparse.ArgumentParser(description='Run parallel GPU training')
    parser.add_argument('--batch', type=int, default=None,
                        help='Batch number (1, 2, or 3)')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Specific models to run')
    parser.add_argument('--all', action='store_true',
                        help='Run all models sequentially in batches')
    args = parser.parse_args()

    print("=" * 60)
    print("Parallel GPU Training")
    print("=" * 60)

    if args.models:
        # 运行指定模型
        experiments = [(i % 4, m) for i, m in enumerate(args.models)]
        run_batch(experiments)
    elif args.batch:
        # 运行指定批次
        batches = {1: BATCH_1, 2: BATCH_2, 3: BATCH_3}
        if args.batch in batches:
            run_batch(batches[args.batch])
        else:
            print(f"Invalid batch number: {args.batch}")
    elif args.all:
        # 运行所有批次
        for i, batch in enumerate([BATCH_1, BATCH_2, BATCH_3], 1):
            print(f"\n{'=' * 60}")
            print(f"BATCH {i}")
            print('=' * 60)
            run_batch(batch)
    else:
        # 默认运行第一批
        run_batch(BATCH_1)

    print("\n" + "=" * 60)
    print("All experiments finished!")
    print("Check logs in outputs/logs/")
    print("=" * 60)


if __name__ == "__main__":
    main()
