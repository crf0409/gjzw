# -*- coding: utf-8 -*-
"""
效率评测 — 参数量 / FLOPs / 显存 / 延迟 / 能耗

输出 (params, FLOPs, latency_cpu, latency_gpu, peak_mem) 字典, 供 Pareto 图和
论文 Table 5 (efficiency comparison) 使用.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn


def count_params(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "total_params_M": total / 1e6,
    }


def count_flops(model: nn.Module, input_shape: tuple,
                device: torch.device) -> dict:
    """用 fvcore 统计 FLOPs. 失败时返回 None."""
    try:
        from fvcore.nn import FlopCountAnalysis
        x = torch.randn(1, *input_shape, device=device)
        flops = FlopCountAnalysis(model.to(device).eval(), x).total()
        return {"flops": int(flops), "gflops": flops / 1e9}
    except Exception as e:
        return {"flops": None, "error": str(e)}


@torch.no_grad()
def measure_latency(model: nn.Module, input_shape: tuple,
                     device: torch.device,
                     batch_sizes: tuple = (1, 32),
                     n_warmup: int = 30, n_run: int = 200) -> dict:
    """测多 batch_size 下的 mean / median / p95 latency."""
    model = model.to(device).eval()
    out = {"device": str(device), "by_batch": {}}
    for bs in batch_sizes:
        x = torch.randn(bs, *input_shape, device=device)
        for _ in range(n_warmup):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(n_run):
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        times = np.array(times) * 1000.0  # ms
        out["by_batch"][f"bs{bs}"] = {
            "mean_ms": float(times.mean()),
            "median_ms": float(np.median(times)),
            "p95_ms": float(np.percentile(times, 95)),
            "ms_per_image": float(np.median(times) / bs),
            "throughput_img_per_s": float(bs * 1000 / np.median(times)),
        }
    return out


def measure_peak_memory(model: nn.Module, input_shape: tuple,
                         device: torch.device, batch_size: int = 32) -> dict:
    """测 GPU 峰值显存 (训练步 forward+backward)."""
    if device.type != "cuda":
        return {"peak_mem_mb": None, "note": "cpu device"}
    model = model.to(device).train()
    torch.cuda.reset_peak_memory_stats(device)
    x = torch.randn(batch_size, *input_shape, device=device)
    out = model(x)
    if isinstance(out, tuple):
        out = out[0]
    loss = out.mean()
    loss.backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    return {"peak_mem_mb": float(peak), "batch_size": batch_size}


def full_efficiency_report(model: nn.Module, input_shape: tuple,
                             device: torch.device, name: str = "model") -> dict:
    """一站式: 参数 + FLOPs + 多 bs latency + 峰值内存."""
    print(f"\n[efficiency] {name}")
    rep = {"name": name}
    rep.update(count_params(model))
    rep.update(count_flops(model, input_shape, device))

    # latency CPU + GPU
    cpu_lat = measure_latency(model, input_shape, torch.device("cpu"),
                                 batch_sizes=(1,), n_warmup=10, n_run=50)
    gpu_lat = (measure_latency(model, input_shape, device,
                                  batch_sizes=(1, 32), n_warmup=30, n_run=100)
                if device.type == "cuda" else {})
    rep["latency_cpu"] = cpu_lat
    rep["latency_gpu"] = gpu_lat

    # 显存峰值 (重新构造模型避免 CPU 状态污染)
    try:
        peak = measure_peak_memory(model, input_shape, device, batch_size=32)
        rep["peak_memory"] = peak
    except Exception as e:
        rep["peak_memory"] = {"peak_mem_mb": None, "error": str(e)}

    print(f"  params: {rep['total_params_M']:.1f} M")
    print(f"  FLOPs:  {rep.get('gflops', 'N/A')}")
    print(f"  latency CPU bs=1: "
          f"{cpu_lat['by_batch']['bs1']['median_ms']:.2f} ms")
    if gpu_lat:
        print(f"  latency GPU bs=1: "
              f"{gpu_lat['by_batch']['bs1']['median_ms']:.2f} ms")
        print(f"  latency GPU bs=32: "
              f"{gpu_lat['by_batch']['bs32']['median_ms']:.2f} ms "
              f"({gpu_lat['by_batch']['bs32']['ms_per_image']:.3f} ms/img)")
    if rep["peak_memory"].get("peak_mem_mb"):
        print(f"  peak GPU mem: {rep['peak_memory']['peak_mem_mb']:.0f} MB")

    return rep
