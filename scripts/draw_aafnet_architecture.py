"""绘制 A 会风格 AAFNet 架构图 D1a / D1b

特点：
  - 半透明阶段色块作为分区背景
  - 圆角矩形 + drop-shadow 立体 box
  - FancyArrowPatch 多种箭头样式（实线/虚线/双线 wrap）
  - 张量 shape 标注在节点旁
  - 大字号阶段标题 + 模块子标题双层文字
  - 配色专门为论文打印调整（红 / 蓝 / 绿 / 橙 / 紫五色家族）
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.transforms import offset_copy
import numpy as np

OUT_DIR = Path(__file__).resolve().parents[1] / "paper" / "figures" / "diagrams"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": ["Noto Sans CJK JP", "Noto Sans", "DejaVu Sans"],
    "axes.unicode_minus": False,
})

# ─────────────────────────────────────────────────────────────────
# 颜色配色（A 会论文标准搭配）
# ─────────────────────────────────────────────────────────────────

PALETTE = {
    "input":   {"fill": "#FFF3D6", "edge": "#C89400", "text": "#3a2a00"},
    "backbone":{"fill": "#DCEAF7", "edge": "#2B6CB0", "text": "#1a3450"},
    "mssa":    {"fill": "#FCDADA", "edge": "#D04848", "text": "#3a0f0f"},
    "csgf":    {"fill": "#FFE6C8", "edge": "#D18438", "text": "#5a3500"},
    "head":    {"fill": "#D9EFE0", "edge": "#3AA17E", "text": "#0d3a25"},
    "proj":    {"fill": "#E1ECF8", "edge": "#5A86BD", "text": "#102640"},
    "loss":    {"fill": "#F0DCEC", "edge": "#9A3D99", "text": "#3a0d3a"},
    "teacher": {"fill": "#EAE1F4", "edge": "#7B5BA0", "text": "#2d164d"},
}

ZONE_COLORS = {
    "stage1": "#E8F0FA",
    "stage2": "#FCEEEE",
    "stage3": "#FFF5E6",
    "stage4": "#EDF6F0",
    "stage5": "#F5E8F3",
    "entry":  "#FFF8E5",
}

SHADOW_OFFSET = (0.04, -0.04)
SHADOW_COLOR = "#00000022"


# ─────────────────────────────────────────────────────────────────
# 基础绘制工具
# ─────────────────────────────────────────────────────────────────


def draw_box(ax, x, y, w, h, kind, lines, fontsize=10.5, lw=1.6,
             title_fontsize=None, title_lines=0, italic_lines=()):
    """
    在 (x, y) 为左下角，w×h 大小，绘制一个 A 会风格 box。
    kind: PALETTE 中的 key
    lines: 显示文字（list of str），第 1 行加粗
    """
    style = PALETTE[kind]
    # drop shadow
    shadow = FancyBboxPatch(
        (x + SHADOW_OFFSET[0], y + SHADOW_OFFSET[1]),
        w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=0, facecolor=SHADOW_COLOR, edgecolor="none",
        zorder=2,
    )
    ax.add_patch(shadow)
    # main box
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=lw, facecolor=style["fill"],
        edgecolor=style["edge"], zorder=3,
    )
    ax.add_patch(box)
    # text — vertically centered
    n = len(lines)
    line_h = (h - 0.18) / max(n, 1)
    for i, line in enumerate(lines):
        cy = y + h - 0.09 - (i + 0.5) * line_h
        weight = "bold" if i < max(1, title_lines) else "normal"
        size = title_fontsize if (i < title_lines and title_fontsize) else fontsize
        style_kw = {"fontstyle": "italic"} if i in italic_lines else {}
        ax.text(x + w / 2, cy, line, ha="center", va="center",
                fontsize=size, fontweight=weight, color=style["text"],
                **style_kw, zorder=4)


def draw_zone(ax, x, y, w, h, color, title, subtitle=""):
    """绘制半透明阶段分区色块 + 阶段标题（标题与副标题都置于顶部白色药丸内）"""
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.20",
        linewidth=2.5, linestyle="--",
        facecolor=color, edgecolor="#999999",
        alpha=0.55, zorder=1,
    )
    ax.add_patch(rect)
    # 阶段标题与副标题合并到顶部居中横条 pill 中
    pill_w = w - 0.4
    pill_h = 0.55
    pill_x = x + 0.2
    pill_y = y + h - pill_h - 0.18
    pill = FancyBboxPatch(
        (pill_x, pill_y), pill_w, pill_h,
        boxstyle="round,pad=0.01,rounding_size=0.10",
        linewidth=1.2, facecolor="white",
        edgecolor="#666666", zorder=4,
    )
    ax.add_patch(pill)
    if subtitle:
        ax.text(pill_x + 0.30, pill_y + pill_h / 2, title,
                fontsize=14, fontweight="bold", color="#222222",
                ha="left", va="center", zorder=6)
        # 副标题紧跟其后
        ax.text(pill_x + 0.30 + 1.55, pill_y + pill_h / 2,
                f"— {subtitle}",
                fontsize=10.5, fontstyle="italic", color="#555555",
                ha="left", va="center", zorder=6)
    else:
        ax.text(pill_x + pill_w / 2, pill_y + pill_h / 2, title,
                fontsize=14, fontweight="bold", color="#222222",
                ha="center", va="center", zorder=6)


def arrow(ax, x0, y0, x1, y1, label="", style="->", color="#222",
          lw=1.8, label_offset=(0, 0.18), label_size=9, dashed=False):
    """绘制带可选标签的箭头"""
    ls = "--" if dashed else "-"
    arr = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle=style, mutation_scale=14,
        linewidth=lw, color=color, linestyle=ls,
        connectionstyle="arc3,rad=0.0",
        zorder=4,
    )
    ax.add_patch(arr)
    if label:
        mx, my = (x0 + x1) / 2 + label_offset[0], (y0 + y1) / 2 + label_offset[1]
        ax.text(mx, my, label, fontsize=label_size, color=color,
                 ha="center", va="center", style="italic",
                 bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                           edgecolor="none", alpha=0.85), zorder=5)


def curved_arrow(ax, x0, y0, x1, y1, rad=0.25, label="", color="#222",
                 lw=1.8, label_offset=(0, 0.0), label_size=9, dashed=False):
    """弯曲箭头（用于汇入或分叉）"""
    ls = "--" if dashed else "-"
    arr = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="->", mutation_scale=14,
        linewidth=lw, color=color, linestyle=ls,
        connectionstyle=f"arc3,rad={rad}",
        zorder=4,
    )
    ax.add_patch(arr)
    if label:
        mx = (x0 + x1) / 2 + label_offset[0]
        my = (y0 + y1) / 2 + label_offset[1]
        ax.text(mx, my, label, fontsize=label_size, color=color,
                 ha="center", va="center", style="italic",
                 bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                           edgecolor="none", alpha=0.85), zorder=5)


# ─────────────────────────────────────────────────────────────────
# D1a — 前半段：输入 → 骨干 → MSSA → CSGF → fused
# ─────────────────────────────────────────────────────────────────


def draw_d1a():
    fig, ax = plt.subplots(figsize=(20, 7.5))
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 9)
    ax.set_aspect("equal")
    ax.axis("off")

    # ——— 标题 ———
    ax.text(12.5, 8.55, "AAFNet — Forward Pipeline (Part 1 / 2)",
            fontsize=18, fontweight="bold", ha="center", color="#222")
    ax.text(12.5, 8.10,
            "Input → Multi-scale Backbone → MSSA × 3 → CSGF Cross-Scale Fusion → Fused Vector",
            fontsize=11, fontstyle="italic", ha="center", color="#555")

    # ——— 阶段分区色块 ———
    draw_zone(ax, 0.4, 0.7, 4.2, 7.0, ZONE_COLORS["stage1"],
              "Stage 1", "ResNet-50 Multi-scale Backbone")
    draw_zone(ax, 5.0, 0.7, 5.4, 7.0, ZONE_COLORS["stage2"],
              "Stage 2", "MSSA Multi-Scale Stylistic Attention")
    draw_zone(ax, 10.8, 0.7, 13.8, 7.0, ZONE_COLORS["stage3"],
              "Stage 3", "CSGF Cross-Scale Gated Fusion")

    # ——— Stage 1: Input + ArchAug + ResNet-50 layers ———
    # Input image 下移：Stage 1 zone 顶部 7.7, pill 占用 6.97~7.52
    # 所以 Input image top 不能超过 6.6（留 0.4 gap）
    draw_box(ax, 0.7, 5.2, 1.8, 1.4, "input",
             ["Input image", "x ∈ ℝ³ˣ²²⁴ˣ²²⁴", "(RGB)"], fontsize=10)
    draw_box(ax, 0.7, 3.2, 1.8, 1.2, "input",
             ["ArchAug", "(train-time", "augmentation)"], fontsize=9.5)

    # ResNet-50 layers — 同步下移
    draw_box(ax, 2.85, 5.4, 1.55, 1.0, "backbone",
             ["layer2", "F₂: 512 × 28²"], fontsize=10)
    draw_box(ax, 2.85, 4.1, 1.55, 1.0, "backbone",
             ["layer3", "F₃: 1024 × 14²"], fontsize=10)
    draw_box(ax, 2.85, 2.8, 1.55, 1.0, "backbone",
             ["layer4", "F₄: 2048 × 7²"], fontsize=10)

    # arrows: Input → ArchAug → backbone
    arrow(ax, 1.6, 5.2, 1.6, 4.4, dashed=True, label="train", color="#888",
          label_offset=(0.35, 0))
    arrow(ax, 2.5, 5.9, 2.85, 5.9, label_size=8)
    arrow(ax, 2.5, 5.9, 2.85, 4.6, label_size=8)
    arrow(ax, 2.5, 5.9, 2.85, 3.3, label_size=8)
    arrow(ax, 2.5, 3.8, 2.85, 5.9, dashed=True, color="#888")
    arrow(ax, 2.5, 3.8, 2.85, 4.6, dashed=True, color="#888")
    arrow(ax, 2.5, 3.8, 2.85, 3.3, dashed=True, color="#888")

    # ——— Stage 2: MSSA modules ———
    mssa_y = [5.4, 4.1, 2.8]
    for i, y in enumerate(mssa_y):
        idx = i + 2
        draw_box(ax, 5.4, y, 4.6, 1.0, "mssa",
                 [f"MSSA_{idx}",
                  "GAP → style sᵢ ∈ ℝ¹²⁸",
                  "Fᵢ' = Fᵢ ⊙ σ(W·sᵢ)·σ(Conv₁(Fᵢ))"],
                 fontsize=9, title_lines=1, title_fontsize=10.5)
        # arrow from layer to MSSA
        arrow(ax, 4.4, y + 0.5, 5.4, y + 0.5, label_size=8)

    # ——— Stage 3: CSGF chain ———(也下移以对齐)
    csgf_y = 4.1
    # 投影 v_i
    draw_box(ax, 11.2, csgf_y, 3.0, 1.6, "csgf",
             ["proj projection",
              "vᵢ = Wᵢ · GAP(Fᵢ')",
              "vᵢ ∈ ℝ⁵¹²,  i ∈ {2,3,4}"],
             fontsize=9.5, title_lines=1, title_fontsize=11)
    # softmax gate — 拆为 4 行避免超框
    draw_box(ax, 14.8, csgf_y, 3.6, 1.6, "csgf",
             ["softmax gate",
              "α = softmax(W_g ·",
              "concat(GAP(F₂'), GAP(F₃'), GAP(F₄')))",
              "α ∈ ℝ³,  Σ αᵢ = 1"],
             fontsize=8.5, title_lines=1, title_fontsize=11)
    # fused vector
    draw_box(ax, 19.0, csgf_y, 4.6, 1.6, "csgf",
             ["fused vector",
              "v = α₂·v₂ + α₃·v₃ + α₄·v₄",
              "v ∈ ℝ⁵¹²"], fontsize=10, title_lines=1, title_fontsize=11.5)

    # arrows: MSSA → proj
    for y in mssa_y:
        curved_arrow(ax, 10.0, y + 0.5, 11.2, csgf_y + 0.8, rad=-0.15)
    # proj → gate → fused
    arrow(ax, 14.2, csgf_y + 0.8, 14.8, csgf_y + 0.8, label_size=9)
    arrow(ax, 18.4, csgf_y + 0.8, 19.0, csgf_y + 0.8, label_size=9)

    # tensor shape annotations — 下移到与 box 平齐
    ax.text(14.5, csgf_y + 1.9, "v₂ / v₃ / v₄  ∈ ℝ⁵¹²",
            fontsize=9, color="#5a3500",
            ha="center", style="italic",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="#FFF8E5",
                      edgecolor="#D18438", linewidth=0.8))
    ax.text(18.7, csgf_y + 1.9, "α  ∈ ℝ³", fontsize=9, color="#5a3500",
            ha="center", style="italic",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="#FFF8E5",
                      edgecolor="#D18438", linewidth=0.8))

    # ——— 流向指示带 ———
    ax.text(21.3, 2.8, "→ continues to D1b\n(heads + SASC-KD loss)",
            fontsize=10, color="#9A3D99", ha="center", va="center",
            fontweight="bold", style="italic",
            bbox=dict(boxstyle="round,pad=0.30", facecolor="#FAF3F8",
                      edgecolor="#9A3D99", linewidth=1.5))
    arrow(ax, 21.3, csgf_y, 21.3, 3.4, color="#9A3D99", lw=2.5)

    # ——— 图例（精确居中：估算总宽再计算起点）———
    legend_y = 1.15
    legend_items = [
        ("input",    "Input / Aug"),
        ("backbone", "Backbone feature"),
        ("mssa",     "MSSA module"),
        ("csgf",     "CSGF proj/gate/fused"),
    ]
    item_widths = [3.0, 3.4, 3.0, 4.0]   # 每个 item 估计占宽
    legend_total = sum(item_widths)
    canvas_w = 25.0
    legend_x_start = (canvas_w - legend_total) / 2

    cx = legend_x_start
    for (kind, label), iw in zip(legend_items, item_widths):
        rect = FancyBboxPatch((cx, legend_y), 0.45, 0.28,
                               boxstyle="round,pad=0.01,rounding_size=0.04",
                               facecolor=PALETTE[kind]["fill"],
                               edgecolor=PALETTE[kind]["edge"],
                               linewidth=1.2, zorder=3)
        ax.add_patch(rect)
        ax.text(cx + 0.58, legend_y + 0.14, label, fontsize=9,
                va="center", ha="left", color="#333")
        cx += iw

    fig.tight_layout()
    out = OUT_DIR / "D1a_aafnet_arch_part1.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────
# D1b — 后半段：fused → heads → SASC-KD loss
# ─────────────────────────────────────────────────────────────────


def draw_d1b():
    fig, ax = plt.subplots(figsize=(20, 8.5))
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")

    # 标题
    ax.text(12.5, 9.55, "AAFNet — Forward Pipeline (Part 2 / 2)",
            fontsize=18, fontweight="bold", ha="center", color="#222")
    ax.text(12.5, 9.10,
            "Fused Vector → Dual Output Heads → SASC-KD Composite Loss",
            fontsize=11, fontstyle="italic", ha="center", color="#555")

    # 阶段分区
    draw_zone(ax, 0.4, 0.5, 5.0, 8.2, ZONE_COLORS["entry"],
              "From Part 1", "fused vector")
    draw_zone(ax, 5.8, 0.5, 7.4, 8.2, ZONE_COLORS["stage4"],
              "Stage 4", "Dual Output Heads")
    draw_zone(ax, 13.6, 0.5, 11.0, 8.2, ZONE_COLORS["stage5"],
              "Stage 5", "SASC-KD Composite Loss")

    # ——— Entry: fused vector ———
    draw_box(ax, 1.0, 5.0, 4.0, 2.0, "csgf",
             ["fused vector  v",
              "v = Σ αᵢ · vᵢ  ∈ ℝ⁵¹²",
              "(from Part 1, CSGF output)"], fontsize=10.5,
             title_lines=1, title_fontsize=12)

    # ——— Stage 4: dual heads ———
    # ClsHead
    draw_box(ax, 6.4, 6.4, 6.2, 1.5, "head",
             ["ClsHead",
              "FC(512 → 6) → softmax",
              "ŷ ∈ ℝ⁶ (class probabilities)"],
             fontsize=9.5, title_lines=1, title_fontsize=11.5)
    # Proj head
    draw_box(ax, 6.4, 4.0, 6.2, 1.5, "proj",
             ["Proj head g_proj",
              "FC(512 → 256 → 128) + L2-norm",
              "z ∈ S¹²⁷ ⊂ ℝ¹²⁸ (unit sphere)"],
             fontsize=9.5, title_lines=1, title_fontsize=11.5)

    # arrows fused → heads
    curved_arrow(ax, 5.0, 6.0, 6.4, 7.15, rad=0.18)
    curved_arrow(ax, 5.0, 6.0, 6.4, 4.75, rad=-0.18)

    # ——— Stage 5: 损失 ———
    # 教师网络（独立位置，左下）
    draw_box(ax, 14.2, 1.4, 4.0, 1.6, "teacher",
             ["frozen RN50 teacher",
              "(separately trained)",
              "p_t = σ(z_t / T),  T = 4"], fontsize=9,
             title_lines=1, title_fontsize=11)

    # 三个损失
    draw_box(ax, 14.2, 6.4, 4.5, 1.5, "loss",
             ["L_FLS",
              "Focal-γ=2 + LS-ε=0.05",
              "(applied on ClsHead logits)"], fontsize=9,
             title_lines=1, title_fontsize=11)

    draw_box(ax, 14.2, 4.0, 4.5, 1.5, "loss",
             ["L_SupCon",
              "−Σ log[exp(zᵢ·zₚ/τ)/",
              "        Σ exp(zᵢ·zₐ/τ)]",
              "τ = 0.07,  same-class as +"], fontsize=8.5,
             title_lines=1, title_fontsize=11)

    draw_box(ax, 19.6, 4.6, 4.6, 1.4, "loss",
             ["L_KD",
              "T² · KL(σ(zₛ/T) ∥ σ(z_t/T))",
              "T = 4 (distill from teacher)"], fontsize=9,
             title_lines=1, title_fontsize=11)

    # 总损失
    draw_box(ax, 19.6, 1.6, 4.6, 2.4, "loss",
             ["Total Loss",
              "ℒ_total = ℒ_FLS",
              "+ λ₁ · ℒ_SupCon",
              "+ λ₂ · ℒ_KD",
              "(λ₁ = λ₂ = 1.0)"], fontsize=10.5,
             title_lines=1, title_fontsize=13)

    # arrows ClsHead → L_FLS
    arrow(ax, 12.6, 7.15, 14.2, 7.15, label="logits", label_size=8)
    # Proj → L_SupCon
    arrow(ax, 12.6, 4.75, 14.2, 4.75, label="z", label_size=8)
    # Proj → L_KD（实际还经过 student logits，简化为直接连）
    curved_arrow(ax, 12.6, 7.15, 19.6, 5.3, rad=0.15, color="#666",
                 label="z_s", label_size=8)
    # Teacher → L_KD
    curved_arrow(ax, 18.2, 2.2, 19.6, 4.6, rad=0.20, dashed=True,
                 color="#7B5BA0", label="z_t (no grad)",
                 label_offset=(0.7, 0.2), label_size=8)

    # 三个损失 → 总损失
    curved_arrow(ax, 18.7, 6.4, 19.6, 4.0, rad=-0.20)
    curved_arrow(ax, 18.7, 4.0, 19.6, 3.5, rad=-0.10)
    curved_arrow(ax, 24.2, 4.6, 23.0, 4.0, rad=0.10)

    # 图例（精确居中）
    legend_items = [
        ("csgf",    "Input (from Part 1)"),
        ("head",    "Classification head"),
        ("proj",    "Projection head"),
        ("teacher", "Frozen teacher"),
        ("loss",    "Loss term / total"),
    ]
    item_widths = [3.7, 3.9, 3.4, 3.3, 3.5]
    legend_total = sum(item_widths)
    canvas_w = 25.0
    legend_x_start = (canvas_w - legend_total) / 2
    legend_y = 0.85

    cx = legend_x_start
    for (kind, label), iw in zip(legend_items, item_widths):
        rect = FancyBboxPatch((cx, legend_y), 0.45, 0.28,
                               boxstyle="round,pad=0.01,rounding_size=0.04",
                               facecolor=PALETTE[kind]["fill"],
                               edgecolor=PALETTE[kind]["edge"],
                               linewidth=1.2, zorder=3)
        ax.add_patch(rect)
        ax.text(cx + 0.58, legend_y + 0.14, label, fontsize=9,
                va="center", ha="left", color="#333")
        cx += iw

    fig.tight_layout()
    out = OUT_DIR / "D1b_aafnet_arch_part2.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    print("Drawing AAFNet architecture diagrams (A-conf style)...")
    draw_d1a()
    draw_d1b()
    print("Done.")
