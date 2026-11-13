"""Failure-case analysis on the real-world typographic sets (RTA-100, PAINT, optionally Disentangling).

Uses the cached image features (same order as the authors' dataset objects, shuffle=False) to get the
predictions of CLIP, the authors' token, the reproduced DP tokens and the option-B tokens, then pulls the
corresponding raw images from the authors' dataset classes to draw labelled example grids.

Categories (per image):
  fixed       CLIP wrong, every DP seed right           -> what the defense repairs
  broken      CLIP right, every DP seed wrong           -> what the defense breaks
  persistent  every DP seed AND every B seed wrong      -> residual failures neither version fixes
  b_wins      every B seed right, every DP seed wrong
  b_loses     every DP seed right, every B seed wrong

Image order: the authors' real-world loaders list files with os.listdir(), whose order depends on the file system,
so the order in which the features were cached (Colab) need not match the order on the machine running this script.
Each cached feature is therefore matched to its image by re-encoding the images and taking the nearest neighbour
(cosine ~ 1); the match is checked to be one-to-one and label-consistent.

Runs on CPU (re-encoding 1,000 images takes a few minutes; a GPU is used if present). Run from the Defense-Prefix repo root:
  python failure_cases.py --cache CACHE_DIR --runs RUNS_DIR --sets rta-100 paint --out OUT_DIR
"""
import argparse
import csv
import json
import os
import random
import sys

import torch

sys.path.insert(0, os.getcwd())
from cache_features import build              # authors' dataset classes, same arguments as feature caching
from eval_cached import class_text_features, load_model, device

DP = ["dp_seed0", "dp_seed1", "dp_seed2"]
B = ["vocab_seed0", "vocab_seed1", "vocab_seed2"]
CATS = ["fixed", "broken", "persistent", "b_wins", "b_loses"]
# "Soft Nordic" pastel palette, same as scripts/make_figures.py
PASTEL = {"gray": "#cfcbc4", "blue": "#6b8fc2", "apricot": "#efab7d", "mint": "#9fcfbf"}
CAT_COLOR = {"fixed": PASTEL["mint"], "broken": PASTEL["apricot"], "persistent": PASTEL["gray"],
             "b_wins": PASTEL["blue"], "b_loses": PASTEL["apricot"]}
CAT_LABEL = {"fixed": "Fixed by DP", "broken": "Broken by DP", "persistent": "Wrong for DP and DP + everyday words",
             "b_wins": "Only DP + everyday words right", "b_loses": "Only DP right"}
CAT_SHORT = {"fixed": "Fixed\nby DP", "broken": "Broken\nby DP", "persistent": "Wrong for both\n(DP, DP + everyday)",
             "b_wins": "Only DP +\neveryday right", "b_loses": "Only DP\nright"}
INK, INK2 = "#0b0b0b", "#52514e"


def edge(c, f=0.72):
    c = c.lstrip("#"); return "#" + "".join(f"{int(int(c[i:i + 2], 16) * f):02x}" for i in (0, 2, 4))


def load_token(path, dtype):
    t = torch.load(path, map_location=device, weights_only=False).detach()
    return (t.unsqueeze(0) if t.dim() == 1 else t).to(device).type(dtype)


def match_images(name, model, c, root):
    """Map every cached feature to its image file (see module docstring). Returns (dataset, list of file indices)."""
    from clip.clip import _transform
    from torch.utils.data import DataLoader
    data = build(name, _transform(224), root=root)
    feats = []
    with torch.no_grad():
        for img, _, _ in DataLoader(data, batch_size=64, shuffle=False, num_workers=2):
            f = model.encode_image(img.to(device).type(model.dtype)).float()
            feats.append(torch.nn.functional.normalize(f, dim=-1).cpu())
    feats = torch.cat(feats)
    sim = c["clean"].float() @ feats.T
    best, j = sim.max(dim=1)
    assert len(set(j.tolist())) == len(j), f"{name}: image matching is not one-to-one"
    bad = sum(data.true_labels[int(k)] != c["classes"][int(l)] for k, l in zip(j, c["labels"]))
    assert bad == 0, f"{name}: {bad} matched images have a different label"
    print(f"{name}: matched {len(j)} cached features to images, cosine min {best.min():.4f} / mean {best.mean():.4f}")
    return data, j.tolist()


def written_text(fname):
    """'label=cup_text=iPhone.jpg' -> 'iPhone' (RTA-100 and PAINT file-name convention)."""
    try:
        return fname.split("_text=")[1].rsplit(".", 1)[0]
    except IndexError:
        return None


def majority(preds):
    return max(set(preds), key=preds.count)


def grid(items, classes, title, path, color, ncol=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not items:
        return
    nrow = (len(items) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.9 * nrow + 0.6), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (img, r) in zip(axes.ravel(), items):
        ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(img)
        for sp in ax.spines.values():          # pastel frame = category colour
            sp.set_color(color); sp.set_linewidth(5)
        ax.set_title(f"true: {classes[r['true']]}   written: {r['text']}\nCLIP: {classes[r['clip']]}\n"
                     f"DP: {classes[r['dp_major']]}\nDP + everyday: {classes[r['b_major']]}",
                     fontsize=7.5, loc="left", color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.55 / (3.9 * nrow + 0.6)), h_pad=2.2)
    fig.suptitle(title, fontsize=11, x=0.01, y=0.995, ha="left", va="top", color=INK,
                 bbox={"boxstyle": "round,pad=0.35", "facecolor": color, "edgecolor": edge(color)})
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def counts_figure(summary, path):
    """Share of images per category for each real-world set (pastel bars, one colour per set)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    sets = list(summary)
    cols = [PASTEL["blue"], PASTEL["apricot"], PASTEL["mint"]]
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    w = 0.8 / len(sets)
    for i, name in enumerate(sets):
        n = summary[name]["counts"]["n"]
        vals = [100 * summary[name]["counts"][c] / n for c in CATS]
        x = np.arange(len(CATS)) + (i - (len(sets) - 1) / 2) * w
        ax.bar(x, vals, w * 0.92, color=cols[i % 3], edgecolor=edge(cols[i % 3]), linewidth=0.7,
               hatch="///" if i == 2 else None, label=f"{name} (n={n})")
        for xi, v, c in zip(x, vals, CATS):
            ax.text(xi, v + 0.6, f"{v:.1f}", ha="center", va="bottom", fontsize=6.5, color=INK2)
    ax.set_xticks(range(len(CATS)))
    ax.set_xticklabels([CAT_SHORT[c] for c in CATS], fontsize=7.5)
    ax.set_ylabel("% of images", color=INK2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="#e4e3df"); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--authors_token", default="learned_token/dp_vit-b32.pt")
    ap.add_argument("--sets", nargs="+", default=["rta-100", "paint"])
    ap.add_argument("--root", default="datasets")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_cat", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import torchvision.transforms as T
    show = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224)])
    model = load_model()
    toks = {"authors": load_token(args.authors_token, model.dtype)}
    toks.update({t: load_token(f"{args.runs}/{t}/dp_token.pt", model.dtype) for t in DP + B})
    summary = {}
    for name in args.sets:
        c = torch.load(f"{args.cache}/{name}.pt", map_location="cpu", weights_only=False)
        x = c["clean"].float().to(device)
        preds = {"clip": (x @ class_text_features(model, c["classes"], c["template"]).float().T).argmax(-1).cpu()}
        for k, t in toks.items():
            preds[k] = (x @ class_text_features(model, c["classes"], c["template"], t).float().T).argmax(-1).cpu()
        y = c["labels"]
        data, file_idx = match_images(name, model, c, args.root)
        acc = {k: round(100 * p.eq(y).float().mean().item(), 2) for k, p in preds.items()}
        print(name, "accuracy check (compare with results/*.csv):", acc)

        rows = []
        for i in range(len(y)):
            fname = data.img_files[file_idx[i]]
            r = {"idx": i, "file": fname, "text": written_text(fname), "true": int(y[i]),
                 **{k: int(p[i]) for k, p in preds.items()}}
            dp_ok = [r[k] == r["true"] for k in DP]
            b_ok = [r[k] == r["true"] for k in B]
            clip_ok = r["clip"] == r["true"]
            r["dp_major"], r["b_major"] = majority([r[k] for k in DP]), majority([r[k] for k in B])
            r["fixed"] = (not clip_ok) and all(dp_ok)
            r["broken"] = clip_ok and not any(dp_ok)
            r["persistent"] = not any(dp_ok) and not any(b_ok)
            r["b_wins"] = all(b_ok) and not any(dp_ok)
            r["b_loses"] = all(dp_ok) and not any(b_ok)
            # residual failure that lands on CLIP's own (attacked) answer: the defense did not move the prediction
            r["persistent_same_as_clip"] = r["persistent"] and r["dp_major"] == r["clip"]
            r.update({"true_name": c["classes"][r["true"]], "clip_name": c["classes"][r["clip"]],
                      "dp_name": c["classes"][r["dp_major"]], "b_name": c["classes"][r["b_major"]]})
            # did the model output the WRITTEN word (attack success) rather than some other wrong class?
            r["clip_is_text"] = r["clip_name"] == r["text"]
            r["dp_is_text"] = r["dp_name"] == r["text"]
            r["b_is_text"] = r["b_name"] == r["text"]
            rows.append(r)
        with open(f"{args.out}/preds_{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        counts = {k: sum(r[k] for r in rows) for k in CATS + ["persistent_same_as_clip"]}
        counts["n"] = len(rows)
        pers = [r for r in rows if r["persistent"]]
        text_stats = {"attack_success_clip": sum(r["clip_is_text"] for r in rows),
                      "attack_success_dp_majority": sum(r["dp_is_text"] for r in rows),
                      "attack_success_everyday_majority": sum(r["b_is_text"] for r in rows),
                      "persistent_dp_predicts_text": sum(r["dp_is_text"] for r in pers),
                      "persistent_clip_predicts_text": sum(r["clip_is_text"] for r in pers)}
        summary[name] = {"accuracy": acc, "counts": counts, "written_word": text_stats}
        print(name, text_stats)
        print(name, counts)

        from PIL import Image
        rng = random.Random(args.seed)
        for cat in CATS:
            pick = [r for r in rows if r[cat]]
            rng.shuffle(pick)
            items = [(show(Image.open(os.path.join(data.img_dir, r["file"])).convert("RGB")), r) for r in pick[:args.per_cat]]
            grid(items, c["classes"], f"{name}: {CAT_LABEL[cat]} ({counts[cat]} of {len(rows)} images)",
                 f"{args.out}/{name}_{cat}.png", CAT_COLOR[cat])
    json.dump(summary, open(f"{args.out}/failure_summary.json", "w"), indent=1)
    counts_figure(summary, f"{args.out}/fig6_failure_categories.png")
    print("saved", args.out)


if __name__ == "__main__":
    main()
