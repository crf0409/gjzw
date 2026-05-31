"""INR 纹理压缩可解释性可视化

加载已拟合好的 BatchedSIREN 权重（data/cache_inr/AL6_siren_h*_L4_224x224_*.pt），
重建图像并产出 5 张可解释图：

  F_INR_A_recon_grid.png          每类 1 张原图 vs h=128 vs h=256 vs 误差热图
  F_INR_B_psnr_storage_pareto.png 每图 PSNR / 存储 散点 + JPEG 对照线
  F_INR_C_zoom_textures.png       3 类纹理（瓦片 / 木构 / 彩画）局部放大
  F_INR_D_freq_spectrum.png       原图 vs SIREN 重建 的 FFT 幅值谱对比
  F_INR_E_progressive.png         单图重建质量随步数的进化（重新训一张做演示）
"""
from __future__ import annotations

import io
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.inr.siren import BatchedSIREN, build_coord_grid, SIREN  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = ROOT / "paper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRETTY_FONT_SIZE = 9
plt.rcParams.update({
    "font.size": PRETTY_FONT_SIZE,
    "axes.titlesize": PRETTY_FONT_SIZE + 1,
    "axes.labelsize": PRETTY_FONT_SIZE,
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.family": ["Noto Sans CJK JP", "AR PL UKai CN", "DejaVu Sans"],
    "axes.unicode_minus": False,
})


# ─────────────────────────────────────────────────────────────────
# 工具：把 flat 权重张量装回 BatchedSIREN
# ─────────────────────────────────────────────────────────────────


def load_siren_from_flat(flat_w: torch.Tensor, hidden: int, layers: int = 4,
                         omega_0: float = 30.0) -> BatchedSIREN:
    """flat_w: [B, P] -> BatchedSIREN(B, hidden, layers)"""
    B = flat_w.shape[0]
    siren = BatchedSIREN(batch_size=B, hidden_dim=hidden,
                         num_layers=layers, omega_0=omega_0).to(DEVICE)
    dims = [2] + [hidden] * (layers - 1) + [3]
    offset = 0
    with torch.no_grad():
        for i in range(layers):
            d_in, d_out = dims[i], dims[i + 1]
            w_size = d_in * d_out
            b_size = d_out
            W = flat_w[:, offset:offset + w_size].reshape(B, d_in, d_out)
            offset += w_size
            bb = flat_w[:, offset:offset + b_size]
            offset += b_size
            siren.weights[i].copy_(W.to(DEVICE))
            siren.biases[i].copy_(bb.to(DEVICE))
    return siren


def reconstruct(siren: BatchedSIREN, h: int = 224, w: int = 224,
                chunk: int = 8) -> torch.Tensor:
    """前向得到 [B, 3, H, W] 重建（分批避免 OOM）"""
    siren.eval()
    coords = build_coord_grid(h, w, device=DEVICE)
    B = siren.batch_size
    outs = []
    with torch.no_grad():
        for s in range(0, B, chunk):
            e = min(s + chunk, B)
            sub = BatchedSIREN(batch_size=e - s, hidden_dim=siren.hidden_dim,
                                num_layers=siren.num_layers,
                                omega_0=siren.omega_0).to(DEVICE)
            for i in range(siren.num_layers):
                sub.weights[i].copy_(siren.weights[i][s:e])
                sub.biases[i].copy_(siren.biases[i][s:e])
            o = sub(coords)                              # [chunk, H*W, 3]
            outs.append(o.reshape(e - s, h, w, 3).permute(0, 3, 1, 2).cpu())
            del sub
            torch.cuda.empty_cache()
    return torch.cat(outs, dim=0).clamp(-1.0, 1.0)


# ─────────────────────────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────────────────────────


def load_originals(split: str = "train", limit: int = 64) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """加载 AL6 原图 (uint8) -> [N, 3, 224, 224] in [-1, 1] 区间"""
    ckpt = torch.load(ROOT / f"data/cache/AL6_224x224_rgb_{split}.pt",
                      map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        imgs = ckpt.get("images", ckpt.get("data"))
        labels = ckpt.get("labels")
        class_names = ckpt.get("class_names", [f"c{i}" for i in range(6)])
    else:
        imgs, labels, class_names = ckpt, torch.zeros(len(ckpt), dtype=torch.long), [f"c{i}" for i in range(6)]
    if imgs.dtype == torch.uint8:
        imgs = imgs.float() / 127.5 - 1.0       # uint8 -> [-1, 1]
    imgs = imgs[:limit]
    labels = labels[:limit] if labels is not None else None
    return imgs, labels, class_names


def load_siren_cache(split: str = "train", hidden: int = 128) -> dict:
    """加载已拟合好的 SIREN 权重"""
    p = ROOT / f"data/cache_inr/AL6_siren_h{hidden}_L4_224x224_{split}.pt"
    return torch.load(p, map_location="cpu", weights_only=False)


def to_image(t: torch.Tensor) -> np.ndarray:
    """[3,H,W] in [-1,1] -> [H,W,3] uint8"""
    img = ((t.clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8).cpu().numpy()
    return np.transpose(img, (1, 2, 0))


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """两张 [-1,1] 图像的 PSNR"""
    mse = ((a - b) ** 2).mean().item()
    if mse < 1e-12:
        return 99.0
    return 10.0 * math.log10(4.0 / mse)


def jpeg_size(img_np_uint8: np.ndarray, quality: int) -> int:
    """模拟 JPEG 压缩后字节数"""
    pil = Image.fromarray(img_np_uint8)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    return len(buf.getvalue())


# ─────────────────────────────────────────────────────────────────
# Figure A: 6 类 × (原图 / h=128 / h=256 / 误差热图) 网格
# ─────────────────────────────────────────────────────────────────


def figure_recon_grid():
    print("=== Figure A: recon grid ===")
    imgs, labels, class_names = load_originals("train", limit=64)
    cache_128 = load_siren_cache("train", 128)
    cache_256 = load_siren_cache("train", 256)
    siren_128 = load_siren_from_flat(cache_128["weights"], hidden=128)
    siren_256 = load_siren_from_flat(cache_256["weights"], hidden=256)

    recon_128 = reconstruct(siren_128, chunk=16)
    recon_256 = reconstruct(siren_256, chunk=4)

    # 每类挑 1 张
    chosen = []
    for cid in range(6):
        idx = (labels == cid).nonzero(as_tuple=True)[0]
        if len(idx) > 0:
            chosen.append(int(idx[0]))
    if len(chosen) < 6:
        chosen = list(range(min(6, len(imgs))))

    fig, axes = plt.subplots(len(chosen), 4, figsize=(11, 1.7 * len(chosen)))
    for row, idx in enumerate(chosen):
        orig = imgs[idx]
        r128 = recon_128[idx]
        r256 = recon_256[idx]
        psnr_128 = psnr(orig, r128)
        psnr_256 = psnr(orig, r256)
        # 误差热图（h=128 残差）
        err = (orig - r128).abs().mean(dim=0).cpu().numpy()  # [H, W]
        # 可视化
        axes[row, 0].imshow(to_image(orig)); axes[row, 0].set_ylabel(class_names[labels[idx].item()], fontsize=PRETTY_FONT_SIZE)
        axes[row, 0].set_title("原图 (~150 KB)" if row == 0 else "")
        axes[row, 1].imshow(to_image(r128))
        axes[row, 1].set_title("h=128 SIREN (132 KB / 图)" if row == 0 else "")
        axes[row, 1].set_xlabel(f"PSNR {psnr_128:.1f} dB", fontsize=PRETTY_FONT_SIZE - 0.5)
        axes[row, 2].imshow(to_image(r256))
        axes[row, 2].set_title("h=256 SIREN (520 KB / 图)" if row == 0 else "")
        axes[row, 2].set_xlabel(f"PSNR {psnr_256:.1f} dB", fontsize=PRETTY_FONT_SIZE - 0.5)
        im = axes[row, 3].imshow(err, cmap="hot", vmin=0, vmax=err.max())
        axes[row, 3].set_title("h=128 残差 |原图−重建|" if row == 0 else "")
        for ax in axes[row]:
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("INR 纹理压缩可解释性 — 6 类古建筑样本逐尺寸重建对比",
                 fontsize=PRETTY_FONT_SIZE + 2, y=1.005)
    fig.tight_layout()
    out = OUT_DIR / "F_INR_A_recon_grid.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────
# Figure B: PSNR / 存储 散点 + JPEG Pareto 对照
# ─────────────────────────────────────────────────────────────────


def figure_psnr_storage_pareto():
    print("=== Figure B: PSNR/storage Pareto ===")
    imgs, labels, class_names = load_originals("train", limit=64)
    cache_128 = load_siren_cache("train", 128)
    cache_256 = load_siren_cache("train", 256)
    siren_128 = load_siren_from_flat(cache_128["weights"], hidden=128)
    siren_256 = load_siren_from_flat(cache_256["weights"], hidden=256)
    r128 = reconstruct(siren_128, chunk=16)
    r256 = reconstruct(siren_256, chunk=4)

    psnrs_128 = [psnr(imgs[i], r128[i]) for i in range(len(imgs))]
    psnrs_256 = [psnr(imgs[i], r256[i]) for i in range(len(imgs))]

    storage_128 = 33795 * 4 / 1024     # KB / image
    storage_256 = 133123 * 4 / 1024    # KB / image

    # JPEG 对照（同样 64 张）
    jpeg_quality_psnr = {}
    for q in [10, 20, 40, 60, 80, 95]:
        sizes = []; psn = []
        for i in range(len(imgs)):
            orig_uint8 = to_image(imgs[i])
            sz = jpeg_size(orig_uint8, q)
            sizes.append(sz / 1024)
            # 解码回去算 PSNR
            buf = io.BytesIO()
            Image.fromarray(orig_uint8).save(buf, format="JPEG", quality=q)
            buf.seek(0)
            dec = np.array(Image.open(buf).convert("RGB"))
            dec_t = torch.from_numpy(dec).float().permute(2, 0, 1) / 127.5 - 1.0
            psn.append(psnr(imgs[i], dec_t))
        jpeg_quality_psnr[q] = (np.mean(sizes), np.mean(psn))

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    # SIREN 散点
    ax.scatter([storage_128] * len(psnrs_128), psnrs_128, s=24, alpha=0.55,
               color="#d04848", label=f"SIREN h=128 (n={len(psnrs_128)})")
    ax.scatter([storage_256] * len(psnrs_256), psnrs_256, s=24, alpha=0.55,
               color="#2b6cb0", label=f"SIREN h=256 (n={len(psnrs_256)})")
    # 加均值标记
    ax.scatter([storage_128], [np.mean(psnrs_128)], s=180, marker="*",
               color="#d04848", edgecolors="black", linewidths=1.5,
               label=f"h=128 均值 {np.mean(psnrs_128):.1f} dB", zorder=5)
    ax.scatter([storage_256], [np.mean(psnrs_256)], s=180, marker="*",
               color="#2b6cb0", edgecolors="black", linewidths=1.5,
               label=f"h=256 均值 {np.mean(psnrs_256):.1f} dB", zorder=5)

    # JPEG 曲线
    qs_sorted = sorted(jpeg_quality_psnr.keys())
    jp_sizes = [jpeg_quality_psnr[q][0] for q in qs_sorted]
    jp_psnr = [jpeg_quality_psnr[q][1] for q in qs_sorted]
    ax.plot(jp_sizes, jp_psnr, "o-", color="#3aa17e", linewidth=1.5,
            markersize=6, label="JPEG (Q=10/20/40/60/80/95)")
    for q, (s, p) in jpeg_quality_psnr.items():
        ax.annotate(f"Q={q}", (s, p), xytext=(4, 3), textcoords="offset points",
                    fontsize=PRETTY_FONT_SIZE - 1, color="#0d3a25")

    ax.set_xlabel("每图存储（KB，对数刻度）")
    ax.set_ylabel("PSNR（dB）")
    ax.set_xscale("log")
    ax.set_title("纹理压缩 PSNR/存储 Pareto — SIREN vs JPEG（AL6 train 64 张）")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=PRETTY_FONT_SIZE - 0.5)

    out = OUT_DIR / "F_INR_B_psnr_storage_pareto.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────
# Figure C: 局部放大 — 看瓦片 / 木纹 / 彩画细节
# ─────────────────────────────────────────────────────────────────


def figure_zoom_textures():
    print("=== Figure C: zoom textures ===")
    imgs, labels, class_names = load_originals("train", limit=64)
    cache_128 = load_siren_cache("train", 128)
    cache_256 = load_siren_cache("train", 256)
    siren_128 = load_siren_from_flat(cache_128["weights"], hidden=128)
    siren_256 = load_siren_from_flat(cache_256["weights"], hidden=256)
    r128 = reconstruct(siren_128, chunk=16)
    r256 = reconstruct(siren_256, chunk=4)

    # 选 3 个不同类的样本 + 各自一个 64×64 patch
    picks = []
    for cid in range(6):
        idx = (labels == cid).nonzero(as_tuple=True)[0]
        if len(idx) > 0:
            picks.append(int(idx[0]))
        if len(picks) >= 3:
            break

    # 每个样本切 3 个不同位置 patch（顶部 / 中部 / 底部）
    patch_locs = [(40, 40), (96, 96), (152, 152)]
    fig, axes = plt.subplots(len(picks), 4 + 3, figsize=(14, 2.0 * len(picks)))

    for row, idx in enumerate(picks):
        orig = imgs[idx]
        r1 = r128[idx]
        r2 = r256[idx]
        cls = class_names[labels[idx].item()]
        # 全图与边框
        full = to_image(orig)
        axes[row, 0].imshow(full)
        axes[row, 0].set_title(f"原图（{cls}）" if row == 0 else "")
        # 用红框标记 patch 位置
        for (py, px) in patch_locs:
            axes[row, 0].add_patch(plt.Rectangle((px, py), 32, 32, fill=False,
                                                  edgecolor="red", linewidth=1.2))
        axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])
        axes[row, 0].set_ylabel(cls)

        # 接下来 3 列 × 2 行（各 patch 显示 原图 / h=128 / h=256）
        # 但表格只能给 6 列。简化：每行只展示 3 个 patch（h=128 重建对比）
        for j, (py, px) in enumerate(patch_locs):
            patch_orig = full[py:py+32, px:px+32]
            patch_128 = to_image(r1)[py:py+32, px:px+32]
            patch_256 = to_image(r2)[py:py+32, px:px+32]
            # 拼成 1 列 3 行的小图
            stacked = np.vstack([patch_orig, patch_128, patch_256])
            axes[row, 1 + j].imshow(stacked)
            if row == 0:
                axes[row, 1 + j].set_title(f"Patch{j+1} (32×32)\n原 / h=128 / h=256",
                                            fontsize=PRETTY_FONT_SIZE - 0.5)
            axes[row, 1 + j].set_xticks([]); axes[row, 1 + j].set_yticks([])

        # 接下来 3 列 给 残差热图
        for j, (py, px) in enumerate(patch_locs):
            err = (orig - r1).abs().mean(dim=0)[py:py+32, px:px+32].numpy()
            err256 = (orig - r2).abs().mean(dim=0)[py:py+32, px:px+32].numpy()
            stacked_err = np.vstack([err / max(err.max(), 1e-6),
                                     err256 / max(err256.max(), 1e-6)])
            axes[row, 4 + j].imshow(stacked_err, cmap="hot", vmin=0, vmax=1)
            if row == 0:
                axes[row, 4 + j].set_title(f"Patch{j+1} 残差\n上=h128 下=h256",
                                            fontsize=PRETTY_FONT_SIZE - 0.5)
            axes[row, 4 + j].set_xticks([]); axes[row, 4 + j].set_yticks([])

    fig.suptitle("INR 纹理压缩 — 局部 32×32 patch 重建与残差", fontsize=PRETTY_FONT_SIZE + 2, y=1.005)
    fig.tight_layout()
    out = OUT_DIR / "F_INR_C_zoom_textures.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────
# Figure D: FFT 幅值谱 — 看 SIREN 在哪些频段保留得好
# ─────────────────────────────────────────────────────────────────


def fft_log_spectrum(img_chw: torch.Tensor) -> np.ndarray:
    """单图 [3, H, W] -> 灰度 FFT log 幅值谱（中心化）"""
    gray = img_chw.mean(dim=0)
    f = torch.fft.fft2(gray)
    f = torch.fft.fftshift(f)
    mag = (f.abs() + 1e-6).log10()
    return mag.cpu().numpy()


def figure_freq_spectrum():
    print("=== Figure D: FFT spectrum ===")
    imgs, labels, class_names = load_originals("train", limit=64)
    cache_128 = load_siren_cache("train", 128)
    cache_256 = load_siren_cache("train", 256)
    siren_128 = load_siren_from_flat(cache_128["weights"], hidden=128)
    siren_256 = load_siren_from_flat(cache_256["weights"], hidden=256)
    r128 = reconstruct(siren_128, chunk=16)
    r256 = reconstruct(siren_256, chunk=4)

    # 选 3 个样本
    picks = [0, 12, 30]

    fig = plt.figure(figsize=(13, 4 + len(picks)))
    gs = fig.add_gridspec(len(picks), 5, width_ratios=[1, 1, 1, 1, 1.2])

    # 平均频谱（所有 64 张）作为右侧总结
    avg_orig = np.mean([fft_log_spectrum(imgs[i]) for i in range(len(imgs))], axis=0)
    avg_128 = np.mean([fft_log_spectrum(r128[i]) for i in range(len(imgs))], axis=0)
    avg_256 = np.mean([fft_log_spectrum(r256[i]) for i in range(len(imgs))], axis=0)

    vmin = min(avg_orig.min(), avg_128.min(), avg_256.min())
    vmax = max(avg_orig.max(), avg_128.max(), avg_256.max())

    for row, idx in enumerate(picks):
        ax_orig = fig.add_subplot(gs[row, 0])
        ax_orig.imshow(to_image(imgs[idx]))
        ax_orig.set_title("原图" if row == 0 else "")
        ax_orig.set_ylabel(f"#{idx} {class_names[labels[idx].item()]}",
                            fontsize=PRETTY_FONT_SIZE)
        ax_orig.set_xticks([]); ax_orig.set_yticks([])

        for col, (lab, src) in enumerate([("原图 FFT", imgs[idx]),
                                           ("h=128 重建 FFT", r128[idx]),
                                           ("h=256 重建 FFT", r256[idx])]):
            mag = fft_log_spectrum(src)
            ax = fig.add_subplot(gs[row, 1 + col])
            ax.imshow(mag, cmap="viridis", vmin=vmin, vmax=vmax)
            if row == 0:
                ax.set_title(lab)
            ax.set_xticks([]); ax.set_yticks([])

        # 最后一列：径向频谱
        ax_rad = fig.add_subplot(gs[row, 4])
        for series, lab in [(imgs[idx], "原图"), (r128[idx], "h=128"), (r256[idx], "h=256")]:
            mag = fft_log_spectrum(series)
            cy, cx = mag.shape[0] // 2, mag.shape[1] // 2
            yy, xx = np.indices(mag.shape)
            r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
            tbin = np.bincount(r.ravel(), mag.ravel())
            cnt = np.bincount(r.ravel())
            radial = tbin / np.maximum(cnt, 1)
            ax_rad.plot(radial, label=lab, linewidth=1.0)
        ax_rad.set_xlabel("径向频率 bin")
        if row == 0:
            ax_rad.set_title("径向频谱（均值）")
            ax_rad.legend(fontsize=PRETTY_FONT_SIZE - 1)
        ax_rad.grid(alpha=0.3)

    fig.suptitle("INR 频域可解释性 — SIREN 重建 vs 原图 FFT 对比",
                 fontsize=PRETTY_FONT_SIZE + 2, y=1.005)
    fig.tight_layout()
    out = OUT_DIR / "F_INR_D_freq_spectrum.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────
# Figure E: 单图重建质量随训练步数的进化
# ─────────────────────────────────────────────────────────────────


def figure_progressive():
    print("=== Figure E: progressive fitting ===")
    imgs, labels, class_names = load_originals("train", limit=64)
    # 选 1 张代表性样本 — 取第 0 张
    target = imgs[0:1].to(DEVICE)                  # [1, 3, H, W]
    siren = SIREN(in_dim=2, out_dim=3, hidden_dim=128, num_layers=4,
                  omega_0=30.0, final_linear=True).to(DEVICE)
    coords = build_coord_grid(224, 224, device=DEVICE)
    target_flat = target.permute(0, 2, 3, 1).reshape(-1, 3)  # [H*W, 3]

    opt = torch.optim.Adam(siren.parameters(), lr=5e-4)
    snapshots_steps = [0, 25, 50, 100, 200, 500]
    snapshots = {}
    psnr_curve = []

    for step in range(501):
        if step in snapshots_steps:
            with torch.no_grad():
                pred = siren(coords).reshape(224, 224, 3).permute(2, 0, 1)
                snapshots[step] = pred.detach().cpu().clone()
                psnr_curve.append((step, psnr(target[0].cpu(), pred.cpu())))
        if step == 500:
            break
        opt.zero_grad()
        pred = siren(coords)
        loss = ((pred - target_flat) ** 2).mean()
        loss.backward()
        opt.step()

    # 画图
    fig = plt.figure(figsize=(13, 5))
    gs = fig.add_gridspec(2, len(snapshots_steps) + 1)

    # 上排：每个快照
    for col, step in enumerate(snapshots_steps):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(to_image(snapshots[step]))
        psn = next(p for s, p in psnr_curve if s == step)
        ax.set_title(f"step {step}\nPSNR {psn:.1f} dB", fontsize=PRETTY_FONT_SIZE)
        ax.set_xticks([]); ax.set_yticks([])

    # 上排最右：原图
    ax = fig.add_subplot(gs[0, -1])
    ax.imshow(to_image(target[0].cpu()))
    ax.set_title("原图", fontsize=PRETTY_FONT_SIZE)
    ax.set_xticks([]); ax.set_yticks([])

    # 下排：PSNR 曲线 + loss 曲线（重新跑一次取所有 step 的 PSNR）
    siren_b = SIREN(in_dim=2, out_dim=3, hidden_dim=128, num_layers=4,
                    omega_0=30.0, final_linear=True).to(DEVICE)
    opt_b = torch.optim.Adam(siren_b.parameters(), lr=5e-4)
    full_psnr = []
    full_loss = []
    for step in range(500):
        opt_b.zero_grad()
        pred = siren_b(coords)
        loss = ((pred - target_flat) ** 2).mean()
        loss.backward()
        opt_b.step()
        if step % 5 == 0:
            with torch.no_grad():
                pred_full = siren_b(coords).reshape(224, 224, 3).permute(2, 0, 1)
                full_psnr.append((step, psnr(target[0].cpu(), pred_full.cpu())))
                full_loss.append((step, loss.item()))

    ax = fig.add_subplot(gs[1, :])
    steps_arr = [s for s, _ in full_psnr]
    psnr_arr = [p for _, p in full_psnr]
    loss_arr = [l for _, l in full_loss]
    ax.plot(steps_arr, psnr_arr, "-o", color="#d04848", markersize=3,
            label="PSNR（dB，左轴）")
    ax.set_xlabel("Adam 步数")
    ax.set_ylabel("PSNR（dB）", color="#d04848")
    ax.tick_params(axis="y", labelcolor="#d04848")
    ax.axhline(30.0, linestyle="--", color="#888", alpha=0.6, label="目标 30 dB")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=PRETTY_FONT_SIZE - 1)

    ax2 = ax.twinx()
    ax2.plot(steps_arr, loss_arr, "-", color="#2b6cb0", alpha=0.6, linewidth=1,
              label="重建 MSE（右轴对数）")
    ax2.set_yscale("log")
    ax2.set_ylabel("MSE", color="#2b6cb0")
    ax2.tick_params(axis="y", labelcolor="#2b6cb0")

    ax.set_title("INR 渐进拟合 — 单图 SIREN 重建质量随 Adam 步数演化（h=128, lr=5e-4）",
                  fontsize=PRETTY_FONT_SIZE + 1)
    fig.tight_layout()
    out = OUT_DIR / "F_INR_E_progressive.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    figure_recon_grid()
    figure_psnr_storage_pareto()
    figure_zoom_textures()
    figure_freq_spectrum()
    figure_progressive()
    print("\nDone.")
