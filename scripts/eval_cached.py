"""Evaluate CLIP and CLIP + Defense-Prefix on cached image features.

Text features and scoring follow the authors' eval.py exactly:
one template per dataset (templates[0]), "* {class}" with the * embedding
replaced by the DP vector, cosine similarity x 100, argmax.

Usage (from the Defense-Prefix repo root):
  python eval_cached.py --cache CACHE_DIR --token learned_token/dp_vit-b32.pt --tag authors_token
"""
import argparse
import csv
import glob
import json
import os
import sys

import clip
import torch
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())
from utils.non_nv import encode_text_with_learnt_tokens  # authors' function

device = "cuda" if torch.cuda.is_available() else "cpu"

ORDER = ["imagenet", "caltech", "pets", "cars", "flowers", "food", "aircraft", "dtd", "sun", "eurosat",
         "disentangling", "paint", "rta-100"]
# Paper, Tables B, C and 3 (ViT-B/32): (CLIP, CLIP + DP)
PAPER = {
    "clean": {"imagenet": (62.02, 62.48), "caltech": (88.64, 89.28), "pets": (87.35, 87.22), "cars": (58.72, 57.47),
              "flowers": (66.32, 63.82), "food": (84.14, 83.65), "aircraft": (18.99, 19.26), "dtd": (44.57, 40.64),
              "sun": (61.74, 61.41), "eurosat": (42.98, 43.85)},
    "synthetic": {"imagenet": (39.10, 49.83), "caltech": (63.97, 79.54), "pets": (58.95, 72.88), "cars": (21.02, 28.64),
                  "flowers": (31.32, 44.12), "food": (56.27, 67.79), "aircraft": (10.83, 14.49), "dtd": (25.53, 31.6),
                  "sun": (34.02, 43.50), "eurosat": (4.86, 9.65)},
    "real": {"disentangling": (43.27, 71.93), "paint": (50.00, 63.64), "rta-100": (47.20, 58.00)},
}


def load_model():
    model, _ = clip.load("ViT-B/32", device=device)
    model.encode_text_with_learnt_tokens = type(model.encode_text)(encode_text_with_learnt_tokens, model)
    model.eval()
    return model


@torch.no_grad()
def class_text_features(model, classes, template, token=None):
    if token is None:
        toks = torch.cat([clip.tokenize(template.format(c)) for c in classes]).to(device)
        f = model.encode_text(toks)
        return f / f.norm(dim=-1, keepdim=True)
    toks = torch.cat([clip.tokenize(template.format(f"* {c}")) for c in classes]).to(device)
    star = clip.tokenize(["*"]).to(device)[0][1]
    f = model.encode_text_with_learnt_tokens(toks, star, token.unsqueeze(0), is_emb=False)
    return F.normalize(f, dim=-1)


@torch.no_grad()
def predictions(img_feats, text_feats, bs=8192):
    out = []
    for i in range(0, len(img_feats), bs):
        sim = 100.0 * (img_feats[i:i + bs].to(device) @ text_feats.T)
        out.append(sim.argmax(-1).cpu())
    return torch.cat(out)


def evaluate(model, cache_dir, token):
    """Returns rows: dataset, condition, n, acc_clip, acc_dp, plus per-image correctness for later stats."""
    rows, per_image = [], {}
    files = {os.path.basename(p)[:-3]: p for p in glob.glob(os.path.join(cache_dir, "*.pt"))}
    for name in [d for d in ORDER if d in files]:
        c = torch.load(files[name], map_location="cpu", weights_only=False)
        t_clip = class_text_features(model, c["classes"], c["template"])
        t_dp = class_text_features(model, c["classes"], c["template"], token)
        conds = [("real", c["clean"])] if c["real_world"] else [("clean", c["clean"]), ("synthetic", c["attack"])]
        for cond, feats in conds:
            y = c["labels"]
            ok_clip = predictions(feats, t_clip).eq(y)
            ok_dp = predictions(feats, t_dp).eq(y)
            per_image[f"{name}/{cond}"] = {"clip": ok_clip, "dp": ok_dp}
            rows.append({"dataset": name, "condition": cond, "n": len(y),
                         "acc_clip": round(100 * ok_clip.float().mean().item(), 2),
                         "acc_dp": round(100 * ok_dp.float().mean().item(), 2)})
    return rows, per_image


def with_paper(rows):
    for r in rows:
        p = PAPER.get(r["condition"], {}).get(r["dataset"])
        r["paper_clip"], r["paper_dp"] = (p if p else (None, None))
    return rows


def print_table(rows):
    print(f"{'dataset':<14}{'cond':<10}{'n':>7} | {'paper CLIP':>10}{'ours CLIP':>10} | {'paper DP':>9}{'ours DP':>9}")
    for r in rows:
        pc = "" if r["paper_clip"] is None else f"{r['paper_clip']:.2f}"
        pd = "" if r["paper_dp"] is None else f"{r['paper_dp']:.2f}"
        print(f"{r['dataset']:<14}{r['condition']:<10}{r['n']:>7} | {pc:>10}{r['acc_clip']:>10.2f} | {pd:>9}{r['acc_dp']:>9.2f}")
    for cond in ["clean", "synthetic", "real"]:
        sub = [r for r in rows if r["condition"] == cond]
        if sub:
            mc = sum(r["acc_clip"] for r in sub) / len(sub)
            md = sum(r["acc_dp"] for r in sub) / len(sub)
            print(f"  mean {cond:<9} over {len(sub)} sets: CLIP {mc:.2f} | DP {md:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--token", default="learned_token/dp_vit-b32.pt")
    ap.add_argument("--tag", default="authors_token")
    ap.add_argument("--out", default=None, help="results folder (default: <cache>/../results)")
    args = ap.parse_args()

    token = torch.load(args.token, map_location=device, weights_only=False)
    token = token.detach()
    if token.dim() == 1:          # authors save shape (1, 512); eval.py passes token.unsqueeze(0)
        token = token.unsqueeze(0)
    model = load_model()
    token = token.to(device).type(model.dtype)

    rows, per_image = evaluate(model, args.cache, token)
    rows = with_paper(rows)
    print_table(rows)

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.cache)), "results")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"{args.tag}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    torch.save(per_image, os.path.join(out, f"{args.tag}_per_image.pt"))
    print("saved", os.path.join(out, f"{args.tag}.csv"))


if __name__ == "__main__":
    main()
