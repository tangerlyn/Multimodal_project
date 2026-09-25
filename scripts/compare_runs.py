#!/usr/bin/env python3
"""Side-by-side table for several scored runs (results/<name>/summary.json).

    python scripts/compare_runs.py ab_forced ab_qwen ab_control
    python scripts/compare_runs.py --labels forced,qwen,control ab_forced ab_qwen ab_control

Rows: pipeline / strict accuracy, response groups (answer-only, reasoned, truncated), verbatim loops,
mean response length, multiple-choice vs open accuracy, MMMU disciplines. Stdlib only.
"""
import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name):
    path = name if name.endswith(".json") else os.path.join(ROOT, "results", name, "summary.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def pct(x):
    return "n/a" if x is None or x != x else f"{x:.1f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="result names under results/, or paths to summary.json")
    ap.add_argument("--labels", default=None, help="comma-separated column labels")
    args = ap.parse_args()
    S = [load(r) for r in args.runs]
    labels = args.labels.split(",") if args.labels else [os.path.basename(r.rstrip("/")).replace("summary.json", "") or r for r in args.runs]

    rows = []
    def row(name, vals):
        rows.append((name, vals))
    row("items", [str(s["n"]) for s in S])
    row("accuracy, pipeline (%)", [pct(s["pipeline"]["micro_acc"]) for s in S])
    row("accuracy, strict (%)", [pct(s["strict"]["micro_acc"]) for s in S])
    row("multiple-choice acc (%)", [pct(s["by_question_type"]["multiple-choice"]["pipeline_acc"]) for s in S])
    row("open acc (%)", [pct(s["by_question_type"]["open"]["pipeline_acc"]) if s["by_question_type"]["open"]["n"] else "n/a" for s in S])
    for g, label in (("answer_only", "answer-only"), ("reasoned", "reasoned & finished"), ("truncated", "truncated")):
        row(f"{label}: n", [str(s["by_response_group"][g]["n"]) for s in S])
        row(f"{label}: acc (%)", [pct(s["by_response_group"][g]["pipeline_acc"]) if s["by_response_group"][g]["n"] else "n/a" for s in S])
    row("verbatim loops", [str(s["generation"]["loops"]) for s in S])
    row("mean response chars", [f"{s['generation']['mean_chars']:.0f}" for s in S])
    row("median response chars", [f"{s['generation']['median_chars']:.0f}" for s in S])
    row("parse failed (pipeline)", [str(s["pipeline"]["parse_failed"]) for s in S])
    row("fallback items (+right/-wrong)", [f"{s['pipeline']['fallback_items']} (+{s['pipeline']['fallback_gain']}/-{s['pipeline']['fallback_loss']})" for s in S])
    for d, v in S[0]["by_discipline"].items():
        row(f"{d} acc (%)", [pct(s["by_discipline"][d]["pipeline_acc"]) if s["by_discipline"][d]["n"] else "n/a" for s in S])

    w0 = max(len(r[0]) for r in rows)
    print("| " + "metric".ljust(w0) + " | " + " | ".join(labels) + " |")
    print("|" + "-" * (w0 + 2) + "|" + "|".join("-" * (len(l) + 2) for l in labels) + "|")
    for name, vals in rows:
        print("| " + name.ljust(w0) + " | " + " | ".join(v.rjust(len(l)) for v, l in zip(vals, labels)) + " |")


if __name__ == "__main__":
    main()
