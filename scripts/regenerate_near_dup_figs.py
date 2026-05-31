"""
Regenerate near_duplicate_pairs.png for AL6 / ASP / AS25.

Bug fix: original data_audit.py constructs paths as
  data/processed/<ds>/test/<filename>
but the actual layout is data/processed/<ds>/test/<class>/<filename>.
"""
from __future__ import annotations
import json
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def find_image(ds_root: Path, split: str, filename: str, label: int) -> Path | None:
    """Look in <ds_root>/<split>/<label>/<filename> first; fallback to glob."""
    p = ds_root / split / str(label) / filename
    if p.exists():
        return p
    matches = list(ds_root.glob(f"{split}/*/{filename}"))
    return matches[0] if matches else None


def render_for(dataset: str):
    audit_p = ROOT / "outputs" / "data_audit" / dataset / "audit.json"
    if not audit_p.exists():
        print(f"[skip] {dataset}: no audit.json"); return
    audit = json.loads(audit_p.read_text())
    near = audit.get("near_duplicate_phash", {}).get("examples_top10", [])
    if not near:
        print(f"[skip] {dataset}: no near-dup examples"); return

    ds_root = ROOT / "data" / "processed" / dataset
    out = ROOT / "outputs" / "data_audit" / dataset / "near_duplicate_pairs.png"

    pairs_resolved = []
    for d in near:
        test_p = find_image(ds_root, "test", d["test_file"], d["test_label"])
        train_p = find_image(ds_root, "train", d["train_file"], d["train_label"])
        if test_p and train_p:
            pairs_resolved.append((test_p, train_p, d))

    if not pairs_resolved:
        print(f"[err] {dataset}: 0 pairs resolvable")
        return

    n_show = min(len(pairs_resolved), 10)
    fig, axes = plt.subplots(n_show, 2, figsize=(7, 2.6 * n_show))
    if n_show == 1:
        axes = axes.reshape(1, -1)

    for i, (test_p, train_p, d) in enumerate(pairs_resolved[:n_show]):
        try:
            test_img = Image.open(test_p).convert("RGB")
            train_img = Image.open(train_p).convert("RGB")
            axes[i, 0].imshow(test_img)
            axes[i, 1].imshow(train_img)
            axes[i, 0].set_title(f"test  cls #{d['test_label']}  ({test_p.name})", fontsize=9)
            axes[i, 1].set_title(f"train cls #{d['train_label']}  Hamming={d['phash_distance']}", fontsize=9)
        except Exception as e:
            print(f"  failed pair {i}: {e}")
        axes[i, 0].axis("off")
        axes[i, 1].axis("off")

    fig.suptitle(f"{dataset} — top-{n_show} pHash near-duplicate cross-split pairs (test ↔ train)",
                 fontsize=12, y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[ok] {dataset}: wrote {out} ({n_show} pairs)")


def main():
    for ds in ["AL6", "ASP", "AS25"]:
        render_for(ds)

    # Copy AL6 (used as the canonical one in paper/figures/data_audit/) plus the others
    paper_dir = ROOT / "paper" / "figures" / "data_audit"
    paper_dir.mkdir(parents=True, exist_ok=True)
    for ds in ["AL6", "ASP", "AS25"]:
        src = ROOT / "outputs" / "data_audit" / ds / "near_duplicate_pairs.png"
        if src.exists():
            dst = paper_dir / f"near_duplicate_pairs_{ds}.png"
            dst.write_bytes(src.read_bytes())
            print(f"[copy] {dst}")

    # Replace the blank `near_duplicate_pairs.png` with AL6's (7 pairs)
    al6_src = ROOT / "outputs" / "data_audit" / "AL6" / "near_duplicate_pairs.png"
    if al6_src.exists():
        legacy = paper_dir / "near_duplicate_pairs.png"
        legacy.write_bytes(al6_src.read_bytes())
        print(f"[copy] replaced legacy → {legacy}")


if __name__ == "__main__":
    main()
