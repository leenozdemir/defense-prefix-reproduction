"""Per-word-group hit/accuracy for every model on the seen/unseen sets, seed means, paired bootstrap vocab vs repro."""
"""Run from the repo root:  python scripts/analyze_word_groups.py  -> outputs/results/groups06.json"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
try:
    import torch
    def load(p): return torch.load(p, map_location="cpu", weights_only=False)
except ImportError:
    from ptload import load
V = {l.strip() for l in open("data/v_train.txt") if l.strip()}
R = "outputs/results/seen_unseen_06"
rng = np.random.default_rng(0)
out = {}
for ds in ("caltech", "pets"):
    d = load(f"{R}/preds_{ds}.pt")
    y = np.asarray(d["labels"]); ws, wu = np.asarray(d["w_seen"]), np.asarray(d["w_unseen"])
    grp_u = np.array(["everyday_train" if w in V else "everyday_heldout" for w in d["word_unseen"]])
    P = d["preds"]
    res = {}
    for m, p in P.items():
        ps, pu, pc = np.asarray(p["seen"]), np.asarray(p["unseen"]), np.asarray(p["clean"])
        r = {"acc_clean": (pc == y), "hit_in100": (ps == ws), "acc_in100": (ps == y)}
        for g in ("everyday_train", "everyday_heldout"):
            k = grp_u == g
            r[f"hit_{g}"] = (pu == wu)[k]; r[f"acc_{g}"] = (pu == y)[k]
        res[m] = r
    def mean_of(models, key): return np.mean([res[m][key] for m in models], axis=0).astype(float)
    groups = {"clip": ["clip"], "authors": ["authors"], "repro": ["dp_seed0", "dp_seed1", "dp_seed2"],
              "vocab": ["vocab_seed0", "vocab_seed1", "vocab_seed2"], "ctrl_renderer": ["ctrl_renderer"],
              "ctrl_distractors": ["ctrl_distractors"]}
    tab = {}
    for key in res["clip"]:
        row = {g: round(100 * mean_of(ms, key).mean(), 2) for g, ms in groups.items()}
        row["n"] = int(len(res["clip"][key]))
        row["repro_sd"] = round(float(np.std([100 * res[m][key].mean() for m in groups["repro"]], ddof=1)), 2)
        row["vocab_sd"] = round(float(np.std([100 * res[m][key].mean() for m in groups["vocab"]], ddof=1)), 2)
        diff = mean_of(groups["vocab"], key) - mean_of(groups["repro"], key)
        b = [diff[rng.integers(len(diff), size=len(diff))].mean() for _ in range(2000)]
        row["vocab_minus_repro"] = round(100 * diff.mean(), 2)
        row["ci95"] = [round(100 * float(x), 2) for x in np.percentile(b, [2.5, 97.5])]
        tab[key] = row
    out[ds] = tab
    print("==", ds, "| unseen split:", {g: int((grp_u == g).sum()) for g in set(grp_u)})
    for k, r in tab.items(): print(f"{k:<22}", json.dumps(r))
json.dump(out, open("outputs/results/groups06.json", "w"), indent=1)
