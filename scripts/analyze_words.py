"""Word-level control for raw attack strength in the seen/unseen experiment.

Words differ in how strongly they fool plain CLIP (length-matching does not remove this).
For every attack word we compute, over the images it was written on:
  clip_hit  - % of images plain CLIP predicts as the written word (raw attack strength)
  dp_hit    - % of images the DP model still predicts as the written word
  dp_acc    - DP accuracy on those images
Then a weighted least-squares fit over words (weights = number of images per word):
  dp_hit = a + b * clip_hit + c * seen
c answers: at EQUAL raw attack strength, do seen words get through DP less (c < 0) or more?
A 95% CI for c comes from a bootstrap over words. A scatter plot is saved per dataset.

Usage:
  python analyze_words.py --preds results/seen_unseen_authors/preds_pets.pt [...] --model authors --out results/words
"""
import argparse
import csv
import json
import os

import numpy as np
import torch


def per_word(d, model, v_train=()):
    rows = []
    for cond, wkey, ikey in (("seen", "word_seen", "w_seen"), ("unseen", "word_unseen", "w_unseen")):
        words = np.array(d[wkey])
        target = d[ikey].numpy()
        clip_hit = d["preds"]["clip"][cond].numpy() == target
        dp_hit = d["preds"][model][cond].numpy() == target
        dp_ok = d["preds"][model][cond].numpy() == d["labels"].numpy()
        for w in np.unique(words):
            m = words == w
            group = "imagenet100" if cond == "seen" else ("everyday_train" if str(w) in v_train else "everyday_heldout")
            rows.append({"word": str(w), "seen": int(cond == "seen"), "group": group, "n": int(m.sum()),
                         "clip_hit": 100 * clip_hit[m].mean(), "dp_hit": 100 * dp_hit[m].mean(),
                         "dp_acc": 100 * dp_ok[m].mean()})
    return rows


def wls(rows):
    X = np.array([[1.0, r["clip_hit"], r["seen"]] for r in rows])
    y = np.array([r["dp_hit"] for r in rows])
    w = np.sqrt(np.array([r["n"] for r in rows], dtype=float))
    beta, *_ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
    return beta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--model", default="authors")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--vocab", default=None, help="vocab.json / v_train.txt: tag unseen words used in option-B training")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    v_train = set()
    if args.vocab:
        v_train = set(json.load(open(args.vocab))["v_train"]) if args.vocab.endswith(".json") else \
            {l.strip() for l in open(args.vocab) if l.strip()}
    rng = np.random.default_rng(0)
    summary = []
    for path in args.preds:
        d = torch.load(path, map_location="cpu", weights_only=False)
        name = os.path.basename(path)[len("preds_"):-3]
        rows = per_word(d, args.model, v_train)
        beta = wls(rows)
        seen_rows = [r for r in rows if r["seen"]]
        unseen_rows = [r for r in rows if not r["seen"]]
        boots = []
        for _ in range(args.n_boot):
            sample = [seen_rows[i] for i in rng.integers(len(seen_rows), size=len(seen_rows))] + \
                     [unseen_rows[i] for i in rng.integers(len(unseen_rows), size=len(unseen_rows))]
            boots.append(wls(sample)[2])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        s = {"dataset": name, "model": args.model, "n_words_seen": len(seen_rows), "n_words_unseen": len(unseen_rows),
             "mean_clip_hit_seen": round(float(np.mean([r["clip_hit"] for r in seen_rows])), 2),
             "mean_clip_hit_unseen": round(float(np.mean([r["clip_hit"] for r in unseen_rows])), 2),
             "slope_on_clip_hit": round(float(beta[1]), 3),
             "seen_effect_c": round(float(beta[2]), 2), "seen_effect_ci95": [round(float(lo), 2), round(float(hi), 2)]}
        for g in ("imagenet100", "everyday_train", "everyday_heldout"):
            gr = [r for r in rows if r["group"] == g]
            if gr:
                n = sum(r["n"] for r in gr)
                s[f"{g}_words"] = len(gr)
                s[f"{g}_clip_hit"] = round(sum(r["clip_hit"] * r["n"] for r in gr) / n, 2)
                s[f"{g}_model_hit"] = round(sum(r["dp_hit"] * r["n"] for r in gr) / n, 2)
        summary.append(s)
        print(json.dumps(s))
        with open(os.path.join(args.out, f"words_{name}_{args.model}.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows([{k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()} for r in rows])
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            for flag, label, marker in ((1, "seen (ImageNet-100 name)", "o"), (0, "unseen (length-matched)", "^")):
                pts = [r for r in rows if r["seen"] == flag]
                ax.scatter([r["clip_hit"] for r in pts], [r["dp_hit"] for r in pts], s=[6 + r["n"] / 2 for r in pts],
                           marker=marker, alpha=0.6, label=label)
            xs = np.linspace(0, 100, 50)
            for flag, style in ((1, "-"), (0, "--")):
                ax.plot(xs, beta[0] + beta[1] * xs + beta[2] * flag, style, linewidth=1)
            ax.set_xlabel("plain CLIP: % predicted as the written word")
            ax.set_ylabel(f"{args.model}: % still predicted as the written word")
            ax.set_title(f"{name}: per-word attack success")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(args.out, f"words_{name}_{args.model}.png"), dpi=150)
        except ImportError:
            pass
    with open(os.path.join(args.out, f"summary_{args.model}.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
