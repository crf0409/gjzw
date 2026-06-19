# -*- coding: utf-8 -*-
"""
显著性检验: Wilcoxon + Friedman + Holm-Bonferroni post-hoc

输入: per-fold accuracy 矩阵 [n_models, n_folds] 或 dict {model: [fold_accs]}
输出: 模型对之间的 p 值矩阵 + Friedman 全局 p 值 + 排名

用法 (作为 module):
    from src.evaluation.significance import run_significance_tests
    res = run_significance_tests(
        fold_accs={"resnet50": [0.99, 0.98, ...], "convnext_tiny": [...], ...}
    )
    print(res["wilcoxon_pvals"])  # dict[(a,b)] -> p
    print(res["friedman_pvalue"])
    print(res["mean_rank"])

CLI:
    python -m src.evaluation.significance --in outputs/cv/<run_id> [--out ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import scipy.stats


def run_significance_tests(fold_accs: dict[str, list[float]],
                            alpha: float = 0.05) -> dict:
    """
    Args:
        fold_accs: {model_name: [acc_per_fold]}.
                   所有模型 fold 数必须相同 (paired test).
        alpha: 显著性阈值 (用于 Holm-Bonferroni)

    Returns:
        dict {
            'models':           list[str]
            'fold_accs':        dict
            'mean':             dict
            'std':              dict
            'wilcoxon_pvals':   dict[(a, b)] -> p
            'wilcoxon_p_matrix': np.ndarray [M, M]
            'holm_significance': np.ndarray [M, M] of bool
            'friedman_stat':    float
            'friedman_pvalue':  float
            'mean_rank':        dict (Friedman ranks)
        }
    """
    models = sorted(fold_accs.keys())
    M = len(models)
    fold_lens = {len(fold_accs[m]) for m in models}
    if len(fold_lens) > 1:
        raise ValueError(f"models have different fold counts: {fold_lens}")
    n_folds = next(iter(fold_lens))

    means = {m: float(np.mean(fold_accs[m])) for m in models}
    stds = {m: float(np.std(fold_accs[m])) for m in models}

    # 两两 Wilcoxon signed-rank
    p_matrix = np.ones((M, M))
    pvals_dict = {}
    pairs = list(combinations(range(M), 2))
    raw_ps = []
    for i, j in pairs:
        a, b = models[i], models[j]
        x = np.array(fold_accs[a])
        y = np.array(fold_accs[b])
        if np.allclose(x, y):
            p = 1.0
        else:
            try:
                stat, p = scipy.stats.wilcoxon(x, y, zero_method="wilcox",
                                                  alternative="two-sided")
            except ValueError:
                p = 1.0
        p_matrix[i, j] = p_matrix[j, i] = float(p)
        pvals_dict[(a, b)] = float(p)
        raw_ps.append(p)

    # Holm-Bonferroni 校正
    raw_ps_arr = np.array(raw_ps)
    order = np.argsort(raw_ps_arr)
    n_tests = len(raw_ps_arr)
    holm_adj = np.empty_like(raw_ps_arr)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = raw_ps_arr[idx] * (n_tests - rank)
        running_max = max(running_max, adj)
        holm_adj[idx] = min(1.0, running_max)
    sig_pairs = holm_adj < alpha

    holm_matrix = np.zeros((M, M), dtype=bool)
    for k, (i, j) in enumerate(pairs):
        holm_matrix[i, j] = holm_matrix[j, i] = bool(sig_pairs[k])

    # Friedman 全局
    data = np.stack([fold_accs[m] for m in models], axis=0)  # [M, F]
    if data.shape[0] < 3 or data.shape[1] < 3:
        friedman_stat, friedman_p = None, None
    else:
        friedman_stat, friedman_p = scipy.stats.friedmanchisquare(*data)

    # 平均排名
    ranks = np.zeros((M, n_folds))
    for f in range(n_folds):
        col = data[:, f]
        ranks[:, f] = scipy.stats.rankdata(-col, method="average")
    mean_rank = {m: float(ranks[i].mean()) for i, m in enumerate(models)}

    # tuple key 转 str 便于 json 序列化
    pvals_str_keys = {f"{a}__vs__{b}": v for (a, b), v in pvals_dict.items()}
    return {
        "models": models,
        "fold_accs": fold_accs,
        "mean": means,
        "std": stds,
        "wilcoxon_pvals": pvals_str_keys,
        "wilcoxon_p_matrix": p_matrix.tolist(),
        "holm_significance": holm_matrix.tolist(),
        "friedman_stat": float(friedman_stat) if friedman_stat is not None else None,
        "friedman_pvalue": float(friedman_p) if friedman_p is not None else None,
        "mean_rank": mean_rank,
        "alpha": alpha,
        "n_folds": int(n_folds),
    }


def render_pvalue_matrix(result: dict, save_path: Path | None = None
                          ) -> None:
    """画 p 值矩阵热图 (论文 Table 6)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        models = result["models"]
        p = np.array(result["wilcoxon_p_matrix"])
        sig = np.array(result["holm_significance"])
        annot = np.empty_like(p, dtype=object)
        for i in range(len(models)):
            for j in range(len(models)):
                if i == j:
                    annot[i, j] = "—"
                else:
                    star = "*" if sig[i, j] else ""
                    annot[i, j] = f"{p[i, j]:.3f}{star}"
        plt.figure(figsize=(max(6, len(models) * 0.7),
                             max(5, len(models) * 0.6)))
        sns.heatmap(p, annot=annot, fmt="", cmap="rocket",
                     xticklabels=models, yticklabels=models,
                     cbar_kws={"label": "Wilcoxon p"})
        plt.title("Pairwise Wilcoxon p (Holm-Bonferroni *: p_adj < α)")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.close()
    except Exception as e:
        print(f"  warn: plot failed: {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_dir", required=True,
                   help="CV 结果目录 (含每模型的 cv_summary.json), "
                        "或一个汇总的 fold_accs.json")
    p.add_argument("--out", dest="out_dir", default=None)
    p.add_argument("--alpha", type=float, default=0.05)
    args = p.parse_args()

    in_path = Path(args.in_dir)
    fold_accs = {}
    if in_path.is_file() and in_path.suffix == ".json":
        with open(in_path) as f:
            fold_accs = json.load(f)
    else:
        # 在子目录里找 cv_summary.json
        for sub in in_path.iterdir():
            cv_summary = sub / "cv_summary.json"
            if cv_summary.exists():
                with open(cv_summary) as f:
                    s = json.load(f)
                accs = [f["test_accuracy"] for f in s.get("folds", [])
                        if "test_accuracy" in f]
                if accs:
                    fold_accs[s.get("model", sub.name)] = accs

    if not fold_accs:
        sys.exit(f"no fold_accs found at {in_path}")

    print(f"\n=== Significance Tests ===")
    for m, accs in fold_accs.items():
        print(f"  {m:<20} n={len(accs):>2}  "
              f"mean={np.mean(accs):.4f}  std={np.std(accs):.4f}")

    res = run_significance_tests(fold_accs, alpha=args.alpha)

    print(f"\n  Friedman: stat={res['friedman_stat']:.3f}  "
          f"p={res['friedman_pvalue']:.4f}" if res["friedman_pvalue"] else
          "\n  Friedman: skipped (need >=3 models, >=3 folds)")
    print(f"  Mean rank (lower=better):")
    for m, r in sorted(res["mean_rank"].items(), key=lambda x: x[1]):
        print(f"    {m:<20} {r:.2f}")

    out_dir = Path(args.out_dir) if args.out_dir else in_path
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "significance.json"
    with open(out_json, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  saved: {out_json}")

    render_pvalue_matrix(res, out_dir / "pvalue_matrix.png")


if __name__ == "__main__":
    main()
