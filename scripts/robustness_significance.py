"""
P2 #45 — Robustness statistical testing.

For each (config, seed):
  - read attrib_robust_<config>_seed<seed>/*/resnet50/results.json
  - compute mean robust accuracy across (a) all 15 cells, (b) Gaussian only,
    (c) Blur only, (d) Brightness only, (e) Occlusion only, (f) JPEG only

Then for each pair of configs in {baseline, nx_only, archaug_with_noise,
mssa_archaug_noise, aafnet_full}:
  - Wilcoxon signed-rank on the 3 paired (across seeds) means
  - Mean values per config

Output:
  outputs/p1_robustness_sig.json
  outputs/p1_robustness_sig.md
"""
from __future__ import annotations
import json
from pathlib import Path
from itertools import combinations
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]


CONFIGS = ["baseline", "nx_only", "archaug_no_noise", "archaug_with_noise",
           "mssa_only", "mssa_archaug_noise", "aafnet_full"]
SEEDS = [42, 1337, 2024]
PAIRWISE_CONFIGS = ["baseline", "nx_only", "archaug_with_noise",
                    "mssa_archaug_noise", "aafnet_full"]


def load_robust(config: str, seed: int) -> dict | None:
    p = sorted((ROOT / "outputs" / f"attrib_robust_{config}_seed{seed}").glob("*/resnet50/results.json"))
    if not p:
        return None
    with open(p[-1]) as f:
        return json.load(f)


def compute_means(rb: dict) -> dict:
    """Return mean accuracy per perturbation kind + overall mean."""
    out = {}
    accs_all = []
    for kind, cells in rb["robustness"].items():
        accs = [c["accuracy"] for c in cells]
        out[f"{kind}_mean"] = float(np.mean(accs))
        accs_all.extend(accs)
    out["overall_robust_mean"] = float(np.mean(accs_all))
    out["clean"] = float(rb.get("clean_accuracy", 0.0))
    return out


def main():
    # 收集每 config 每 seed 的 means
    per_seed_means = {}
    for cfg in CONFIGS:
        per_seed_means[cfg] = {}
        for seed in SEEDS:
            rb = load_robust(cfg, seed)
            if rb is None:
                continue
            per_seed_means[cfg][seed] = compute_means(rb)

    # 跨 seed 的 mean ± std
    summary = {}
    for cfg in CONFIGS:
        s = per_seed_means[cfg]
        if not s:
            continue
        keys = list(next(iter(s.values())).keys())
        agg = {}
        for k in keys:
            vals = [s[seed][k] for seed in s]
            agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)) if len(vals) > 1 else 0.0,
                      "n_seeds": len(vals)}
        summary[cfg] = agg

    # Wilcoxon 配对 (跨 seed) 在 overall_robust_mean 上
    pair_results = {}
    for a, b in combinations(PAIRWISE_CONFIGS, 2):
        if a not in per_seed_means or b not in per_seed_means:
            continue
        a_vals, b_vals = [], []
        common = sorted(set(per_seed_means[a].keys()) & set(per_seed_means[b].keys()))
        for seed in common:
            a_vals.append(per_seed_means[a][seed]["overall_robust_mean"])
            b_vals.append(per_seed_means[b][seed]["overall_robust_mean"])
        if len(common) < 2:
            pair_results[f"{a}__vs__{b}"] = {"note": "n<2", "n": len(common)}
            continue
        try:
            res = stats.wilcoxon(a_vals, b_vals)
            pair_results[f"{a}__vs__{b}"] = {
                "n": len(common),
                "a_mean": float(np.mean(a_vals)),
                "b_mean": float(np.mean(b_vals)),
                "delta": float(np.mean(a_vals)) - float(np.mean(b_vals)),
                "wilcoxon_p": float(res.pvalue),
            }
        except ValueError as e:
            pair_results[f"{a}__vs__{b}"] = {"error": str(e)}

    out = {
        "per_seed_means": per_seed_means,
        "summary": summary,
        "wilcoxon_pairs": pair_results,
    }

    out_path = ROOT / "outputs" / "p1_robustness_sig.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")

    # MD
    md = ["# P2 #45 — Robustness statistical summary\n",
          "Per-config aggregate accuracy across the 15 robustness cells, computed",
          "per seed then averaged. Wilcoxon signed-rank on the three paired",
          "(by seed) means.\n",
          "## Mean accuracies (3-seed mean ± std)",
          "",
          "| Config | Clean | All 15 cells | Gaussian | Motion blur | Brightness | Occlusion | JPEG |",
          "|---|---|---|---|---|---|---|---|"]
    for cfg in CONFIGS:
        s = summary.get(cfg, {})
        if not s:
            continue
        row = [cfg]
        for k in ["clean", "overall_robust_mean", "gauss_noise_mean",
                  "motion_blur_mean", "brightness_mean", "occlusion_mean", "jpeg_compress_mean"]:
            r = s.get(k, {})
            if not r:
                row.append("—")
            elif r.get("n_seeds", 0) > 1:
                row.append(f"{r['mean']*100:.2f} ± {r['std']*100:.2f} %")
            else:
                row.append(f"{r['mean']*100:.2f} %")
        md.append("| " + " | ".join(row) + " |")

    md.append("\n## Wilcoxon signed-rank (paired by seed) on overall robust accuracy mean")
    md.append("| Pair | a mean | b mean | Δ (pp) | n | p-value |")
    md.append("|---|---|---|---|---|---|")
    for pair, r in pair_results.items():
        if "error" in r:
            md.append(f"| {pair} | err | err | err | err | err |")
            continue
        if "note" in r:
            md.append(f"| {pair} | _{r.get('note')}_ | — | — | {r.get('n')} | — |")
            continue
        md.append(f"| {pair} | {r['a_mean']*100:.2f} % | {r['b_mean']*100:.2f} % | {r['delta']*100:+.2f} | {r['n']} | {r['wilcoxon_p']:.4f} |")

    md_path = ROOT / "outputs" / "p1_robustness_sig.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"Wrote {md_path}")
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
