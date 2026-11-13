"""Write the LaTeX tables of the report from the saved outputs (run from the repo root):
    python scripts/make_tables.py            -> report/tables/*.tex
Every number in the result tables of the report comes from this script; nothing is typed by hand.
"""
import csv
import json
import os
import statistics as st

R, RUNS, OUT = "outputs/results", "outputs/runs", "report/tables"
os.makedirs(OUT, exist_ok=True)
NAMES = {"imagenet": "ImageNet", "caltech": "Caltech101", "pets": "OxfordPets", "cars": "StanfordCars",
         "flowers": "Flowers102", "food": "Food101", "aircraft": "FGVCAircraft", "dtd": "DTD", "eurosat": "EuroSAT",
         "disentangling": "Disentangling$^\\dagger$", "paint": "PAINT", "rta-100": "RTA-100"}


def rows(tag):
    return list(csv.DictReader(open(f"{R}/{tag}.csv")))


def pm(s):
    """'62.09 ± 0.26' -> '62.09{\\scriptsize$\\pm$0.26}'"""
    if "±" in s:
        a, b = [x.strip() for x in s.split("±")]
        return f"{a}{{\\tiny$\\pm${b}}}"
    return s


def write(name, body):
    import re
    body = re.sub(r"(?<![\w$.])-(?=\d)", "$-$", body)          # typographic minus for negative numbers
    body = re.sub(r",\s+", ", ", body).replace("1,000", "1{,}000")
    body = body.replace(",.000)", ",<.001)")
    open(f"{OUT}/{name}.tex", "w").write(body)
    print("wrote", f"{OUT}/{name}.tex")


# ---------------- Table: per-dataset reproduction ----------------
S = list(csv.DictReader(open(f"{R}/summary_06.csv")))
lines = []
for cond, title in (("clean", "Clean images"), ("synthetic", "Synthetic typographic attack"), ("real", "Real-world typographic attack")):
    lines.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{title}}}}}\\\\")
    for r in S:
        if r["condition"] != cond or r["dataset"].startswith("MEAN"):
            continue
        lines.append(" & ".join([NAMES[r["dataset"]], r["paper_clip"], r["paper_dp"], r["ours_clip"], r["authors_dp"],
                                 pm(r["repro_dp"]), pm(r["vocab_dp"]), pm(r["random5_dp"])]) + "\\\\")
    m = next(r for r in S if r["dataset"].startswith("MEAN") and r["condition"] == cond)
    pc = [float(r["paper_clip"]) for r in S if r["condition"] == cond and not r["dataset"].startswith("MEAN")]
    pd = [float(r["paper_dp"]) for r in S if r["condition"] == cond and not r["dataset"].startswith("MEAN")]
    lines.append("\\midrule")
    lines.append(" & ".join([f"\\textbf{{Mean}} ({len(pc)})", f"{st.mean(pc):.2f}", f"{st.mean(pd):.2f}", m["ours_clip"],
                             m["authors_dp"], pm(m["repro_dp"]), pm(m["vocab_dp"]), pm(m["random5_dp"])]) + "\\\\")
    lines.append("\\midrule")
write("tab_reproduction", "\n".join(lines[:-1]) + "\n")

# ---------------- Table: condition means of every run + resources ----------------
def cond_means(tags):
    out = {}
    for cond in ("clean", "synthetic", "real"):
        per = [st.mean(float(r["acc_dp"]) for r in rows(t) if r["condition"] == cond) for t in tags]
        out[cond] = (st.mean(per), st.stdev(per) if len(per) > 1 else None)
    return out


def logs(tags):
    L = [json.load(open(f"{RUNS}/{t}/train_log.json")) for t in tags if os.path.exists(f"{RUNS}/{t}/train_log.json")]
    if not L:
        return None, None, None
    return st.mean(l["train_seconds"] for l in L) / 60, st.mean(l["peak_gpu_mem_mb"] for l in L) / 1024, L[0]["n_candidates"]


runs = [("CLIP (no prefix)", None, "--"),
        ("DP, authors' token", ["authors_token"], "1"),
        ("DP, reproduction", ["dp_seed0", "dp_seed1", "dp_seed2"], "3"),
        ("DP + diversified vocabulary", ["vocab_seed0", "vocab_seed1", "vocab_seed2"], "3"),
        ("\\quad renderer control", ["ctrl_renderer"], "1"),
        ("\\quad distractor control", ["ctrl_distractors"], "1"),
        ("\\quad identity loss only", ["ctrl_identity_only"], "1"),
        ("\\quad random token", [f"ctrl_random_s{i}" for i in range(5)], "5")]
lines = []
for name, tags, n in runs:
    if tags is None:
        c = {cond: (st.mean(float(r["acc_clip"]) for r in rows("dp_seed0") if r["condition"] == cond), None)
             for cond in ("clean", "synthetic", "real")}
        t = m_ = k = None
    else:
        c = cond_means(tags)
        t, m_, k = logs(tags)
    cells = [name, n]
    for cond in ("clean", "synthetic", "real"):
        mu, sd = c[cond]
        cells.append(f"{mu:.2f}" + (f"{{\\tiny$\\pm${sd:.2f}}}" if sd is not None else ""))
    cells += [f"{t:.1f}" if t else "--", f"{m_:.2f}" if m_ else "--", str(k) if k else "--"]
    lines.append(" & ".join(cells) + "\\\\")
write("tab_conditions", "\n".join(lines) + "\n")

# ---------------- Table: seen / unseen diagnostic (authors' token + reproduction) ----------------
SU = list(csv.DictReader(open(f"{R}/seen_unseen_06/seen_unseen.csv")))
W = {(d["dataset"], d["model"]): d for m in ("authors", "dp_seed0") for d in json.load(open(f"{R}/words_06/summary_{m}.json"))}
lines = []
for ds, nm in (("pets", "Pets"), ("caltech", "Caltech101")):
    clip = next(r for r in SU if r["dataset"] == ds and r["model"] == "clip")
    auth = next(r for r in SU if r["dataset"] == ds and r["model"] == "authors")
    w = W[(ds, "authors")]
    ci = lambda s: s.strip("()[] ").replace(",", ", ")
    lines.append(f"\\multirow{{2}}{{*}}{{{nm} (n={clip['n']})}} & CLIP & {clip['hit_seen']} & {clip['hit_unseen']} & "
                 f"{clip['acc_seen']} & {clip['acc_unseen']} & -- & -- \\\\")
    lines.append(f" & DP (authors) & {auth['hit_seen']} & {auth['hit_unseen']} & {auth['acc_seen']} & {auth['acc_unseen']} & "
                 f"{auth['did']} [{ci(auth['did_ci95'])}] & {w['seen_effect_c']:+.2f} [{w['seen_effect_ci95'][0]}, {w['seen_effect_ci95'][1]}]\\\\")
    if ds == "pets":
        lines.append("\\midrule")
write("tab_seen_unseen", "\n".join(lines) + "\n")

# ---------------- Table: paired tests on the real-world sets ----------------
P = json.load(open(f"{R}/paired06.json"))
lines = []
for key, nm in (("disentangling", "Disentangling$^\\dagger$"), ("paint", "PAINT"), ("rta-100", "RTA-100"),
                ("real_pooled", "All real-world"), ("synthetic_pooled_9", "Synthetic (9 sets)"), ("clean_pooled_9", "Clean (9 sets)")):
    p = P[f"vocab_vs_repro/{key}"]
    mc = " ".join(f"({a},{b},{pv:.3f})".replace("0.", ".") if pv < 1 else f"({a},{b},1)" for a, b, pv in p["mcnemar_per_seed"])
    lines.append(f"{nm} & {p['n']:,} & {p['A']:.2f} & {p['B']:.2f} & {p['diff']:+.2f} [{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}] & {mc}\\\\")
lines.append("\\midrule")
for key, nm in (("ctrl_renderer_vs_repro/real_pooled", "Renderer control $-$ repro."),
                ("ctrl_distractors_vs_repro/real_pooled", "Distractor control $-$ repro."),
                ("repro_vs_random5/real_pooled", "Repro. $-$ random token"),
                ("random5_vs_identity/real_pooled", "Random token $-$ identity only")):
    p = P[key]
    lines.append(f"{nm} (real) & {p['n']:,} & {p['A']:.2f} & {p['B']:.2f} & {p['diff']:+.2f} [{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}] & --\\\\")
write("tab_paired", "\n".join(lines) + "\n")

# ---------------- Table: word groups (Pets / Caltech seen-unseen sets) ----------------
G = json.load(open(f"{R}/groups06.json"))
lines = []
for ds, nm in (("pets", "Pets"), ("caltech", "Caltech101")):
    for key, lab in (("acc_in100", "IN-100 names"), ("acc_everyday_train", "Added words, train split"),
                     ("acc_everyday_heldout", "Added words, held-out split")):
        g = G[ds][key]
        lines.append(f"{nm if key == 'acc_in100' else ''} & {lab} & {g['n']:,} & {g['clip']:.2f} & {g['repro']:.2f} & "
                     f"{g['vocab']:.2f} & {g['vocab_minus_repro']:+.2f} [{g['ci95'][0]:+.2f}, {g['ci95'][1]:+.2f}] & {g['ctrl_distractors']:.2f}\\\\")
    if ds == "pets":
        lines.append("\\midrule")
write("tab_groups", "\n".join(lines) + "\n")

# small macro file with numbers used in the running text
A = json.load(open(f"{R}/attack_strength_check.json"))
macros = {
    "premiseInHundred": f"{A['clip_attack_success_imagenet100_words']:.2f}",
    "premiseAdded": f"{A['clip_attack_success_everyday_words']:.2f}",
}
write("numbers", "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()))
