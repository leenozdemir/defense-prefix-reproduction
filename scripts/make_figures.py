"""Report figures (fig1-fig5) from outputs/results and outputs/runs. Run from the repo root:  python scripts/make_figures.py"""
import csv, json, os, sys, statistics as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
try:
    import torch
    def load_pt(p): return torch.load(p, map_location="cpu", weights_only=False)
except ImportError:
    from ptload import load as load_pt

R, RUNS, OUT = "outputs/results", "outputs/runs", "figures"
os.makedirs(OUT, exist_ok=True)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
THEME = os.environ.get("FIG_THEME", "pastel")
if THEME == "pastel":   # soft colours; separation checked for colour-blind readers, bars carry a darker outline + value labels
    REF, C1, C2, C3 = "#cfcbc4", "#6b8fc2", "#efab7d", "#9fcfbf"   # "Soft Nordic"
    HEAT = ["#fcfcfb", "#e6edf6", "#cbd9ec", "#a9c0df", "#89a7d0", "#6b8fc2"]
else:                   # default saturated theme
    REF, C1, C2, C3 = "#6f6e69", "#2a78d6", "#eb6834", "#1baf7a"
    HEAT = ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
OUT = os.environ.get("FIG_OUT", OUT)
os.makedirs(OUT, exist_ok=True)


def edge(c, f=0.72):
    """Darker shade of a fill colour, used as the bar outline so light fills stay visible on white."""
    if THEME != "pastel":
        return "white"
    c = c.lstrip("#"); return "#" + "".join(f"{int(int(c[i:i + 2], 16) * f):02x}" for i in (0, 2, 4))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "figure.facecolor": "white", "legend.frameon": False})

def rows(tag): return list(csv.DictReader(open(f"{R}/{tag}.csv")))
def cond_mean(tag, cond, col="acc_dp"):
    return st.mean(float(r[col]) for r in rows(tag) if r["condition"] == cond)
def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.15); fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig); print("saved", name)
def bars(ax, x, vals, errs, color, label, w, fs=6.5):
    b = ax.bar(x, vals, w * 0.92, yerr=errs, color=color, label=label, capsize=2,
               hatch="///" if (THEME == "pastel" and color == C3) else None,
               error_kw={"elinewidth": 0.8, "ecolor": INK2}, edgecolor=edge(color), linewidth=0.7)
    for i, (xi, v) in enumerate(zip(x, vals)):
        top = v + (errs[i] if errs is not None else 0)
        ax.text(xi, top + 0.8, f"{v:.1f}", ha="center", va="bottom", fontsize=fs, color=INK2)
    return b

# ---------- Fig 1: main benchmark ----------
groups = [("CLIP", None, REF), ("DP reproduction (3 seeds)", ["dp_seed0", "dp_seed1", "dp_seed2"], C1),
          ("DP + diversified vocab. (3 seeds)", ["vocab_seed0", "vocab_seed1", "vocab_seed2"], C2),
          ("Random token (5 draws)", [f"ctrl_random_s{i}" for i in range(5)], C3)]
conds = [("clean", "Clean (9 sets)"), ("synthetic", "Synthetic attack (9 sets)"), ("real", "Real-world attack (3 sets)")]
fig, ax = plt.subplots(figsize=(7.0, 3.3)); w = 0.2
for gi, (lab, tags, col) in enumerate(groups):
    x = np.arange(3) + (gi - 1.5) * w
    if tags is None:
        vals, errs = [cond_mean("dp_seed0", c, "acc_clip") for c, _ in conds], None
    else:
        per = [[cond_mean(t, c) for t in tags] for c, _ in conds]
        vals, errs = [st.mean(p) for p in per], [st.stdev(p) for p in per]
    bars(ax, x, vals, errs, col, lab, w)
ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in conds]); ax.set_ylabel("Top-1 accuracy (%)")
ax.set_ylim(0, 78); ax.legend(ncol=4, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, 1.13))
ax.text(1.0, -0.17, "Bars: mean over datasets; whiskers: SD over seeds/draws. CLIP ViT-B/32, frozen.",
        transform=ax.transAxes, ha="right", fontsize=6.5, color=INK2)
save(fig, "fig1_main_benchmark")

# ---------- Fig 2: paired differences on real-world sets (forest plot) ----------
P = json.load(open(f"{R}/paired06.json"))
items = [("Diversified vocab. − reproduction", "vocab_vs_repro", C2), ("Renderer control − reproduction", "ctrl_renderer_vs_repro", C1),
         ("Distractor control − reproduction", "ctrl_distractors_vs_repro", C3)]
sets = [("disentangling", "Disentangling (n=160)"), ("paint", "PAINT (n=110)"), ("rta-100", "RTA-100 (n=1000)"),
        ("real_pooled", "All real-world (n=1270)")]
fig, ax = plt.subplots(figsize=(6.2, 3.0)); y = 0; yt, yl = [], []
for key, lab in sets:
    for j, (nm, pref, col) in enumerate(items):
        k = f"{pref}/{key}"
        if k not in P: continue
        d, (lo, hi) = P[k]["diff"], P[k]["ci95"]
        yy = y - j * 0.25
        ax.plot([lo, hi], [yy, yy], color=col, linewidth=2.4, solid_capstyle="round")
        ax.plot(d, yy, "o", color=col, markersize=7, markeredgecolor=edge(col), markeredgewidth=1,
                label=nm if key == "disentangling" or (key == "real_pooled" and pref != "vocab_vs_repro") else None)
    yt.append(y - 0.25); yl.append(lab); y -= 1.2
ax.axvline(0, color=INK2, linewidth=0.8); ax.set_yticks(yt); ax.set_yticklabels(yl)
ax.grid(axis="x", color=GRID); ax.grid(axis="y", visible=False)
ax.set_xlabel("Accuracy difference, percentage points (paired image bootstrap, 95% CI)")
h, l = ax.get_legend_handles_labels(); seen = {}
for hh, ll in zip(h, l): seen.setdefault(ll, hh)
ax.legend(seen.values(), seen.keys(), fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.42, 1.14))
save(fig, "fig2_paired_real_world")

# ---------- Fig 3: residual vulnerability per word group ----------
G = json.load(open(f"{R}/groups06.json"))
wg = [("hit_in100", "IN-100 names\n(all training)"), ("hit_everyday_train", "Added words\n(train split)"),
      ("hit_everyday_heldout", "Added words\n(held-out split)")]
models = [("CLIP", "clip", REF), ("DP reproduction", "repro", C1), ("DP + diversified vocab.", "vocab", C2),
          ("Distractor control", "ctrl_distractors", C3)]
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3), sharey=True); w = 0.21
for ax, ds, title in zip(axes, ["pets", "caltech"], ["Oxford Pets", "Caltech-101"]):
    for mi, (lab, key, col) in enumerate(models):
        x = np.arange(3) + (mi - 1.5) * w
        bars(ax, x, [G[ds][g][key] for g, _ in wg], None, col, lab, w, fs=5.6)
    ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in wg], fontsize=7.5); ax.set_title(title, fontsize=9, color=INK)
    ax.set_ylim(0, 68)
axes[0].set_ylabel("Attack success (% predicted as the written word)")
axes[0].legend(ncol=4, fontsize=7.5, loc="upper center", bbox_to_anchor=(1.05, 1.2))
save(fig, "fig3_word_groups")

# ---------- Fig 4: cosine similarity between learned tokens ----------
order = ["dp_seed0", "ctrl_renderer", "ctrl_distractors", "vocab_seed0", "dp_seed1", "vocab_seed1", "dp_seed2",
         "vocab_seed2", "ctrl_identity_only", "authors", "ctrl_random_s0", "ctrl_random_s1"]
names = {"dp_seed0": "DP seed 0", "ctrl_renderer": "renderer ctrl (s0)", "ctrl_distractors": "distractor ctrl (s0)",
         "vocab_seed0": "Diversified seed 0", "dp_seed1": "DP seed 1", "vocab_seed1": "Diversified seed 1", "dp_seed2": "DP seed 2",
         "vocab_seed2": "Diversified seed 2", "ctrl_identity_only": "identity-only (s0)", "authors": "authors' token",
         "ctrl_random_s0": "random #0", "ctrl_random_s1": "random #1"}
AUTH = os.environ.get("AUTHORS_TOKEN", "../ext/Defense-Prefix/learned_token/dp_vit-b32.pt")
vec = {}
for t in order:
    p = AUTH if t == "authors" else f"{RUNS}/{t}/dp_token.pt"
    v = np.asarray(load_pt(p), dtype=np.float32).reshape(-1); vec[t] = v / np.linalg.norm(v)
M = np.stack([vec[t] for t in order]); S = M @ M.T
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("blue", HEAT)
fig, ax = plt.subplots(figsize=(5.6, 4.8))
im = ax.imshow(np.clip(S, 0, 1), cmap=cmap, vmin=0, vmax=1)
for i in range(len(order)):
    for j in range(len(order)):
        ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center", fontsize=5.8, color="white" if (S[i, j] > 0.6 and THEME != "pastel") else INK)
ax.set_xticks(range(len(order))); ax.set_yticks(range(len(order)))
ax.set_xticklabels([names[t] for t in order], rotation=60, ha="right", fontsize=7)
ax.set_yticklabels([names[t] for t in order], fontsize=7); ax.grid(False)
for s in ax.spines.values(): s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("cosine similarity (negatives shown as 0)", fontsize=7)
cb.outline.set_visible(False)
save(fig, "fig4_token_similarity")

# ---------- Fig 5: training cost ----------
runs = [("DP reproduction", ["dp_seed0", "dp_seed1", "dp_seed2"], C1), ("DP + diversified vocab.", ["vocab_seed0", "vocab_seed1", "vocab_seed2"], C2),
        ("Distractor control", ["ctrl_distractors"], C3)]
logs = {t: json.load(open(f"{RUNS}/{t}/train_log.json")) for _, ts, _ in runs for t in ts}
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6))
for ax, key, lab, scale in [(axes[0], "train_seconds", "Training time (min, T4)", 1 / 60), (axes[1], "peak_gpu_mem_mb", "Peak GPU memory (GB)", 1 / 1024)]:
    vals = [st.mean(logs[t][key] for t in ts) * scale for _, ts, _ in runs]
    b = ax.bar(range(3), vals, 0.6, color=[c for *_, c in runs], edgecolor=[edge(c) for *_, c in runs], linewidth=0.7,
               hatch=[("///" if (THEME == "pastel" and c == C3) else None) for *_, c in runs])
    for i, v in enumerate(vals): ax.text(i, v * 1.02, f"{v:.1f}", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(range(3)); ax.set_xticklabels(["DP repro\n(100 classes)", "Diversified vocab.\n(300 classes)", "Distractor ctrl\n(300 classes)"], fontsize=7)
    ax.set_title(lab, fontsize=8.5, color=INK); ax.set_ylim(0, max(vals) * 1.2)
save(fig, "fig5_training_cost")

# ---------- Fig 7: what kind of error remains on the real-world sets (needs outputs/failure_cases from notebook 07) ----------
FC = "outputs/failure_cases"
if os.path.exists(f"{FC}/preds_rta-100.csv"):
    sets7 = [s for s in ("rta-100", "paint") if os.path.exists(f"{FC}/preds_{s}.csv")]
    fig, axes = plt.subplots(1, len(sets7), figsize=(3.4 * len(sets7) + 0.6, 3.0), sharey=False)
    axes = np.atleast_1d(axes)
    models7 = [("CLIP", "clip_name"), ("DP", "dp_name"), ("DP +\ndiversified vocab.", "b_name")]
    for ax, ds in zip(axes, sets7):
        R7 = list(csv.DictReader(open(f"{FC}/preds_{ds}.csv")))
        n = len(R7)
        txt = [100 * sum(r[k] == r["text"] for r in R7) / n for _, k in models7]
        oth = [100 * sum(r[k] not in (r["text"], r["true_name"]) for r in R7) / n for _, k in models7]
        x = np.arange(len(models7)); w = 0.38
        ax.bar(x - w / 2, txt, w * 0.92, color=C2, edgecolor=edge(C2), linewidth=0.7, label="predicts the written word")
        ax.bar(x + w / 2, oth, w * 0.92, color=REF, edgecolor=edge(REF), linewidth=0.7,
               hatch="///" if THEME == "pastel" else None, label="other wrong class")
        for xi, a, b in zip(x, txt, oth):
            ax.text(xi - w / 2, a + 0.6, f"{a:.1f}", ha="center", va="bottom", fontsize=6.5, color=INK2)
            ax.text(xi + w / 2, b + 0.6, f"{b:.1f}", ha="center", va="bottom", fontsize=6.5, color=INK2)
        ax.set_xticks(x); ax.set_xticklabels([m for m, _ in models7], fontsize=7.5)
        ax.set_title(f"{ds} (n={n})", fontsize=9, color=INK); ax.set_ylim(0, 42)
    axes[0].set_ylabel("% of images")
    axes[0].legend(fontsize=7.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5 * len(sets7) + 0.05, 1.22))
    fig.text(0.99, -0.04, "DP and DP + diversified vocab.: majority prediction over 3 seeds.", ha="right", fontsize=6.5, color=INK2)
    save(fig, "fig7_error_types_real_world")
