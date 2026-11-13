"""Evaluate CLIP and one or more DP tokens on the seen/unseen test sets from make_seen_unseen.py.

Scoring is the authors' eval.py rule (dataset template, "* {class}" with * -> DP vector,
cosine x 100, argmax) over the shared candidate list (dataset classes + seen + unseen words).

Reported per dataset and token:
  acc_clean, acc_seen, acc_unseen         top-1 accuracy (%)
  hit_seen, hit_unseen                    % of images predicted as the written word (attack success)
  gap = acc_seen - acc_unseen             paired over the same images, 95% bootstrap CI
  For DP tokens additionally:
  did = (DP - CLIP)_seen - (DP - CLIP)_unseen (difference-in-differences, absolute points)
  rec_seen / rec_unseen: recovery = % of the images on which CLIP predicted the written word that the
      DP model classifies correctly (conditions on images the attack actually fooled), and the paired
      difference rec_seen - rec_unseen with a 95% bootstrap CI over images fooled in both conditions.
Per-image predictions are saved (preds.pt) for the word-level analysis in analyze_words.py.

Usage (from the Defense-Prefix repo root):
  python eval_seen_unseen.py --data cache_su/su_pets.pt cache_su/su_caltech.pt \
      --tokens authors=learned_token/dp_vit-b32.pt seed0=runs/dp_seed0/dp_token.pt --out results/seen_unseen
"""
import argparse
import csv
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())
import clip  # noqa: E402
from utils.non_nv import encode_text_with_learnt_tokens  # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def text_feats(model, names, template, token=None):
    if token is None:
        f = model.encode_text(torch.cat([clip.tokenize(template.format(c)) for c in names]).to(device))
        return F.normalize(f, dim=-1)
    toks = torch.cat([clip.tokenize(template.format(f"* {c}")) for c in names]).to(device)
    star = clip.tokenize(["*"]).to(device)[0][1]
    return F.normalize(model.encode_text_with_learnt_tokens(toks, star, token.unsqueeze(0), is_emb=False), dim=-1)


@torch.no_grad()
def predict(x, t):
    return torch.cat([(100.0 * x[i:i + 8192].to(device) @ t.T).argmax(-1).cpu() for i in range(0, len(x), 8192)])


def boot_ci(diff, n_boot=2000, seed=0):
    """95% percentile bootstrap CI of the mean of a per-image difference vector (resampling images)."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(len(diff), (n_boot, len(diff)), generator=g)
    means = diff[idx].mean(1)
    return round(100 * means.quantile(0.025).item(), 2), round(100 * means.quantile(0.975).item(), 2)


def load_token(path, dtype):
    t = torch.load(path, map_location=device, weights_only=False).detach()
    if t.dim() == 1:
        t = t.unsqueeze(0)
    return t.to(device).type(dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--tokens", nargs="+", required=True, help="name=path pairs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()
    model.encode_text_with_learnt_tokens = type(model.encode_text)(encode_text_with_learnt_tokens, model)
    tokens = {k: load_token(v, model.dtype) for k, v in (s.split("=", 1) for s in args.tokens)}

    rows = []
    for path in args.data:
        d = torch.load(path, map_location="cpu", weights_only=False)
        cand, y = d["candidates"], d["labels"]
        w_seen = torch.tensor([cand.index(w) for w in d["word_seen"]])
        w_unseen = torch.tensor([cand.index(w) for w in d["word_unseen"]])
        correct, preds = {}, {}
        for name, tok in [("clip", None)] + list(tokens.items()):
            t = text_feats(model, cand, d["template"], tok)
            p = {k: predict(d[k], t) for k in ("clean", "seen", "unseen")}
            ok = {k: p[k].eq(y).float() for k in p}
            correct[name] = ok
            preds[name] = p
            gap = ok["seen"] - ok["unseen"]
            r = {"dataset": d["dataset"], "model": name, "n": len(y),
                 "acc_clean": round(100 * ok["clean"].mean().item(), 2),
                 "acc_seen": round(100 * ok["seen"].mean().item(), 2),
                 "acc_unseen": round(100 * ok["unseen"].mean().item(), 2),
                 "hit_seen": round(100 * p["seen"].eq(w_seen).float().mean().item(), 2),
                 "hit_unseen": round(100 * p["unseen"].eq(w_unseen).float().mean().item(), 2),
                 "gap_seen_minus_unseen": round(100 * gap.mean().item(), 2)}
            r["gap_ci95"] = boot_ci(gap)
            if name != "clip":
                did = (ok["seen"] - correct["clip"]["seen"]) - (ok["unseen"] - correct["clip"]["unseen"])
                r["did"] = round(100 * did.mean().item(), 2)
                r["did_ci95"] = boot_ci(did)
                fooled_s = preds["clip"]["seen"].eq(w_seen)
                fooled_u = preds["clip"]["unseen"].eq(w_unseen)
                r["rec_seen"] = round(100 * ok["seen"][fooled_s].mean().item(), 2)
                r["rec_unseen"] = round(100 * ok["unseen"][fooled_u].mean().item(), 2)
                both = fooled_s & fooled_u
                r["n_fooled_both"] = int(both.sum())
                if both.sum() > 0:
                    dr = ok["seen"][both] - ok["unseen"][both]
                    r["rec_diff_both"] = round(100 * dr.mean().item(), 2)
                    r["rec_diff_both_ci95"] = boot_ci(dr)
            rows.append(r)
            print(json.dumps(r))
        torch.save({"preds": preds, "labels": y, "w_seen": w_seen, "w_unseen": w_unseen,
                    "word_seen": d["word_seen"], "word_unseen": d["word_unseen"]},
                   os.path.join(args.out, f"preds_{d['dataset']}.pt"))

    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(args.out, "seen_unseen.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("saved", os.path.join(args.out, "seen_unseen.csv"))


if __name__ == "__main__":
    main()
