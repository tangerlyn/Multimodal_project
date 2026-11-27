#!/usr/bin/env python3
"""Truncation analysis on MMMU val: what happens to the answers that hit the token cap.

    python MMD/scripts/truncation_analysis.py                 # print the tables
    python MMD/scripts/truncation_analysis.py --plot docs/figures/truncation.png

Compares, item by item on the same 900 questions (Pipeline scores from MMD/results/*/scored.jsonl):
  base       best_r2u      base model, chain-of-thought recipe (truncated answers recovered by the fallback pass)
  hsft       h_sft_s3407   H-SFT, same recipe
  direct     raw_baseline  base model, MMMU-repo prompt "answer ... directly" (Assignment 1 baseline run)
The direct run also differs in seed (42), image size (<= 1,003,520 px) and token cap (8,192), so it is a different
prompt *and* setting, not a prompt-only ablation. P-values are exact two-sided McNemar tests.
--plot needs matplotlib.
"""
import argparse
import json
import os
from math import comb

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
RUNS = {"base": "best_r2u", "hsft": "h_sft_s3407", "direct": "raw_baseline"}


def load(name):
    with open(os.path.join(ROOT, name, "scored.jsonl")) as f:
        return {r["id"]: r for r in map(json.loads, f)}


def mcnemar(a_only, b_only):
    n, k = a_only + b_only, min(a_only, b_only)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def ok(run, i):
    return bool(run[i]["pipeline_correct"])


def compare(a, b, items):
    """(correct in a, correct in b, only a, only b, p)."""
    a_only = sum(ok(a, i) and not ok(b, i) for i in items)
    b_only = sum(ok(b, i) and not ok(a, i) for i in items)
    return sum(ok(a, i) for i in items), sum(ok(b, i) for i in items), a_only, b_only, mcnemar(a_only, b_only)


def truncated(run):
    return {i for i, r in run.items() if r["finish_reason"] == "length"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plot", help="write the figure to this path (PNG or SVG)")
    args = ap.parse_args()
    R = {k: load(v) for k, v in RUNS.items()}
    base, hsft, direct = R["base"], R["hsft"], R["direct"]
    assert set(base) == set(hsft) == set(direct), "runs cover different items"
    total = {k: sum(ok(r, i) for i in r) for k, r in R.items()}
    tb, th = truncated(base), truncated(hsft)
    out = {}

    print(f"totals (of 900): base {total['base']}, H-SFT {total['hsft']}, answer-directly {total['direct']}")
    print(f"truncated: base {len(tb)}, H-SFT {len(th)}, both {len(tb & th)}, only base {len(tb - th)}, "
          f"only H-SFT {len(th - tb)}\n")

    rows = [("base-truncated items: base+fallback vs answer-directly", base, direct, tb),
            ("base-truncated items: H-SFT vs base+fallback", hsft, base, tb),
            ("base-truncated items: H-SFT vs answer-directly", hsft, direct, tb),
            ("H-SFT-truncated items: H-SFT+fallback vs answer-directly", hsft, direct, th),
            ("newly truncated by H-SFT: H-SFT vs base", hsft, base, th - tb)]
    for label, a, b, items in rows:
        ca, cb, ao, bo, p = compare(a, b, items)
        print(f"{label} (n={len(items)}): {ca} vs {cb} correct (only first {ao}, only second {bo}, p={p:.3f})")

    for name, run, tr in (("base", base, tb), ("H-SFT", hsft, th)):
        swapped = total["base" if name == "base" else "hsft"] - sum(ok(run, i) for i in tr) + sum(ok(direct, i) for i in tr)
        own = total["base" if name == "base" else "hsft"]
        print(f"hybrid on {name}: truncated answers replaced by answer-directly -> {swapped}/900 "
              f"({swapped / 9:.2f}%, {swapped - own:+d} items)")

    out["trunc"] = {"MMMU val": (len(tb), len(th))}
    for label, b_name, h_name in (("MMMU-Pro standard", "pro10_base_s3407", "pro10_h_sft_s3407"),
                                  ("MMMU-Pro vision", "prov_base_s3407", "prov_h_sft_s3407")):
        out["trunc"][label] = (len(truncated(load(b_name))), len(truncated(load(h_name))))
    print("\ntruncated answers before retry:", {k: f"{b} -> {h}" for k, (b, h) in out["trunc"].items()})

    out["on_base_trunc"] = [("Base\n+ fallback", sum(ok(base, i) for i in tb)),
                            ("Base,\nanswer directly", sum(ok(direct, i) for i in tb)),
                            ("H-SFT", sum(ok(hsft, i) for i in tb))]
    if args.plot:
        plot(out, len(tb), args.plot)
        print("wrote", args.plot)


def plot(out, n_trunc, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    blue, orange, gray = "#2a78d6", "#eb6834", "#8a8984"   # categorical slots 1-2 (validated) + neutral
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": grid, "axes.labelcolor": ink2, "xtick.color": ink2,
                         "ytick.color": ink2, "text.color": ink, "font.family": "DejaVu Sans"})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={"width_ratios": [1.45, 1]}, facecolor=surface)

    labels = list(out["trunc"])
    w, x = 0.36, range(len(labels))
    for k, (name, color) in enumerate((("Base", blue), ("H-SFT", orange))):
        vals = [out["trunc"][l][k] for l in labels]
        xs = [i + (k - 0.5) * (w + 0.02) for i in x]
        a1.bar(xs, vals, w, color=color, label=name, zorder=2)
        for xi, v in zip(xs, vals):
            a1.text(xi, v + 3, str(v), ha="center", va="bottom", fontsize=9, color=ink2)
    a1.set_xticks(list(x), labels)
    a1.set_title("Truncated answers (hit the token cap), before retry", loc="left", fontsize=11, color=ink)
    a1.legend(frameon=False, loc="upper left")
    a1.set_ylim(0, max(max(v) for v in out["trunc"].values()) * 1.18)

    names = [n for n, _ in out["on_base_trunc"]]
    vals = [v for _, v in out["on_base_trunc"]]
    colors = [blue, gray, orange]
    a2.bar(range(3), vals, 0.6, color=colors, zorder=2)
    for i, v in enumerate(vals):
        a2.text(i, v + 1, f"{v}/{n_trunc}", ha="center", va="bottom", fontsize=9, color=ink2)
    a2.set_xticks(range(3), names)
    a2.set_title(f"Correct answers on the {n_trunc} MMMU val\nitems that Base truncates", loc="left", fontsize=11,
                 color=ink)
    a2.set_ylim(0, max(vals) * 1.25)

    for ax in (a1, a2):
        ax.set_facecolor(surface)
        ax.yaxis.grid(True, color=grid, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(length=0)
    fig.tight_layout(w_pad=3)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=surface)


if __name__ == "__main__":
    main()
