"""Paired per-image comparison: B (vocab seeds) vs reproduction (dp seeds), and controls vs reproduction."""
"""Run from the repo root:  python scripts/analyze_paired.py  -> outputs/results/paired06.json"""
import sys, glob, os, json, numpy as np
from math import comb
sys.path.insert(0, os.path.dirname(__file__))
try:
    import torch
    def load(p): return torch.load(p, map_location="cpu", weights_only=False)
except ImportError:
    from ptload import load
U = {os.path.basename(f)[:-len("_per_image.pt")]: f for f in glob.glob("outputs/results/per_image/*_per_image.pt")}
D = {t: load(f) for t, f in U.items() if t != "smoke"}
rng = np.random.default_rng(0)
def ok(tag, key): return np.asarray(D[tag][key]["dp"]).astype(float)
def mean_ok(tags, key): return np.mean([ok(t, key) for t in tags], axis=0)
def mcnemar(a, b):
    n01, n10 = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum()); n = n01 + n10
    p = min(1.0, 2 * sum(comb(n, k) for k in range(0, min(n01, n10) + 1)) / 2 ** n) if n else 1.0
    return n01, n10, round(p, 4)
REPRO = ["dp_seed0", "dp_seed1", "dp_seed2"]; VOCAB = ["vocab_seed0", "vocab_seed1", "vocab_seed2"]
RAND = [f"ctrl_random_s{i}" for i in range(5)]
sets = {"disentangling": ["disentangling/real"], "paint": ["paint/real"], "rta-100": ["rta-100/real"],
        "real_pooled": ["disentangling/real", "paint/real", "rta-100/real"]}
syn = [k for k in D["dp_seed0"] if k.endswith("synthetic")]; cln = [k for k in D["dp_seed0"] if k.endswith("clean")]
out = {}
def compare(name, A, B, keys):
    a = np.concatenate([mean_ok(A, k) for k in keys]); b = np.concatenate([mean_ok(B, k) for k in keys])
    d = a - b; bs = [d[rng.integers(len(d), size=len(d))].mean() for _ in range(5000)]
    r = {"n": len(d), "A": round(100 * a.mean(), 2), "B": round(100 * b.mean(), 2), "diff": round(100 * d.mean(), 2),
         "ci95": [round(100 * x, 2) for x in np.percentile(bs, [2.5, 97.5])]}
    # seed-paired McNemar (vocab_seed_i vs dp_seed_i) on pooled images
    if len(A) == len(B) == 3:
        r["mcnemar_per_seed"] = [mcnemar(np.concatenate([ok(x, k) for k in keys]), np.concatenate([ok(y, k) for k in keys])) for x, y in zip(A, B)]
    return r
for nm, ks in sets.items():
    out[f"vocab_vs_repro/{nm}"] = compare(nm, VOCAB, REPRO, ks)
out["vocab_vs_repro/synthetic_pooled_9"] = compare("syn", VOCAB, REPRO, syn)
out["vocab_vs_repro/clean_pooled_9"] = compare("cln", VOCAB, REPRO, cln)
for c in ["ctrl_renderer", "ctrl_distractors"]:
    out[f"{c}_vs_repro/real_pooled"] = compare("r", [c], REPRO, sets["real_pooled"])
out["repro_vs_random5/real_pooled"] = compare("r", REPRO, RAND, sets["real_pooled"])
out["random5_vs_identity/real_pooled"] = compare("r", RAND, ["ctrl_identity_only"], sets["real_pooled"])
# CLIP baseline (same in all files)
clip = np.concatenate([np.asarray(D["dp_seed0"][k]["clip"]).astype(float) for k in sets["real_pooled"]])
out["clip_real_pooled_acc"] = round(100 * clip.mean(), 2)
for k, v in out.items(): print(f"{k:<38}", json.dumps(v))
json.dump(out, open("outputs/results/paired06.json", "w"), indent=1)
