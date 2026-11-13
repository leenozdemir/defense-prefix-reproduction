"""Aggregate eval_cached.py CSVs over seeds: mean and sample std per dataset/condition, next to the paper.

Usage:
  python summarize_runs.py --results RESULTS_DIR --group repro=dp_seed0,dp_seed1,dp_seed2 --group authors=authors_token
Each name in a group is a tag written by eval_cached.py (RESULTS_DIR/<tag>.csv).
"""
import argparse
import csv
import os
import statistics as st


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fmt(vals):
    m = st.mean(vals)
    return f"{m:.2f} ± {st.stdev(vals):.2f}" if len(vals) > 1 else f"{m:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--group", action="append", required=True, help="name=tag1,tag2,...")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    groups = {}
    for g in args.group:
        name, tags = g.split("=", 1)
        groups[name] = [read(os.path.join(args.results, f"{t}.csv")) for t in tags.split(",")]

    first = next(iter(groups.values()))[0]
    keys = [(r["dataset"], r["condition"]) for r in first]
    header = ["dataset", "condition", "paper_clip", "paper_dp", "ours_clip"] + [f"{g}_dp" for g in groups]
    lines = []
    for ds, cond in keys:
        row = {"dataset": ds, "condition": cond}
        ref = next(r for r in first if r["dataset"] == ds and r["condition"] == cond)
        row["paper_clip"], row["paper_dp"], row["ours_clip"] = ref["paper_clip"], ref["paper_dp"], ref["acc_clip"]
        for g, runs in groups.items():
            vals = [float(next(r for r in run if r["dataset"] == ds and r["condition"] == cond)["acc_dp"]) for run in runs]
            row[f"{g}_dp"] = fmt(vals)
        lines.append(row)
    # condition means: mean over datasets within each run, then mean ± std over runs
    for cond in ["clean", "synthetic", "real"]:
        sub = [k for k in keys if k[1] == cond]
        if not sub:
            continue
        row = {"dataset": f"MEAN ({len(sub)} sets)", "condition": cond, "paper_clip": "", "paper_dp": ""}
        row["ours_clip"] = f"{st.mean(float(next(r for r in first if (r['dataset'], r['condition']) == k)['acc_clip']) for k in sub):.2f}"
        for g, runs in groups.items():
            per_run = [st.mean(float(next(r for r in run if (r["dataset"], r["condition"]) == k)["acc_dp"]) for k in sub)
                       for run in runs]
            row[f"{g}_dp"] = fmt(per_run)
        lines.append(row)

    width = [max(len(h), *(len(str(l.get(h, ""))) for l in lines)) for h in header]
    print("  ".join(h.ljust(w) for h, w in zip(header, width)))
    for l in lines:
        print("  ".join(str(l.get(h, "")).ljust(w) for h, w in zip(header, width)))
    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            w.writerows(lines)
        print("saved", args.out)


if __name__ == "__main__":
    main()
