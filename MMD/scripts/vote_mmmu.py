"""Experiment G1: self-consistency (majority vote) over several sampled MMMU runs. No training.

Each run is the same preset with a different --seed (vLLM per-request seed = hash(seed, id)), scored
by score_mmmu.py. This script reads the scored.jsonl files and, per item, takes the most common
pipeline_pred among the samples that have one (truncated / unparsed samples abstain). Ties go to the
earliest run in the argument order, so the first run (the existing seed-3407 run) breaks ties.

  python scripts/vote_mmmu.py results/best_r2u results/g1_s1 results/g1_s2 --out results/g1_vote
  python scripts/vote_mmmu.py ... --emit_ids outputs/g1_stage2_ids.txt    # adaptive stage 2

--emit_ids writes the ids whose current votes are NOT unanimous (or have < 2 valid votes), so later
seeds only run there (--ids_file in run_mmmu_eval.sh); unanimous items keep their answer.
Reports: voted accuracy, accuracy of the first run alone, gain/loss vs the first run with an exact
McNemar p-value, accuracy vs number of votes, and per-discipline accuracy.
"""

import argparse
import collections
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_mmmu import DISCIPLINES  # noqa: E402


def key_of(pred, qtype):
    if pred is None or pred == []:
        return None
    if qtype == "multiple-choice":
        return str(pred).strip().upper()
    vals = pred if isinstance(pred, list) else [pred]
    return "|".join(sorted(str(v).strip().lower().rstrip(".") for v in vals))


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def load(run_dir):
    path = os.path.join(run_dir, "scored.jsonl")
    return {r["id"]: r for r in map(json.loads, open(path, encoding="utf-8"))}


def vote(samples, qtype):
    """samples: list of scored rows in run order. Returns (key, correct, n_valid, top_count)."""
    keys = [(key_of(s["pipeline_pred"], qtype), s) for s in samples]
    valid = [(k, s) for k, s in keys if k is not None]
    if not valid:
        return None, False, 0, 0
    counts = collections.Counter(k for k, _ in valid)
    top = max(counts.values())
    winner = next(k for k, _ in valid if counts[k] == top)          # earliest run wins a tie
    correct = next(s["pipeline_correct"] for k, s in valid if k == winner)
    return winner, bool(correct), len(valid), top


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="results/<stem> dirs; the first must cover all 900 items")
    ap.add_argument("--out", default=None, help="write voted.jsonl + summary.json here")
    ap.add_argument("--emit_ids", default=None, help="write non-unanimous ids for the next stage")
    args = ap.parse_args()

    runs = [load(d) for d in args.runs]
    base = runs[0]
    if len(base) != 900:
        sys.exit(f"first run has {len(base)} items, expected 900")
    for d, r in zip(args.runs, runs):
        extra = set(r) - set(base)
        if extra:
            sys.exit(f"{d}: ids not in the first run, e.g. {sorted(extra)[:3]}")
        print(f"{d}: {len(r)} items", flush=True)

    rows, need_more = [], []
    for i, (rid, b) in enumerate(base.items()):
        samples = [r[rid] for r in runs if rid in r]
        key, ok, n_valid, top = vote(samples, b["question_type"])
        unanimous = n_valid >= 2 and top == n_valid
        if not unanimous:
            need_more.append(rid)
        rows.append({"id": rid, "subject": b["subject"], "question_type": b["question_type"],
                     "answer": b["answer"], "n_samples": len(samples), "n_valid": n_valid,
                     "top_votes": top, "voted_pred": key, "voted_correct": ok,
                     "first_correct": bool(b["pipeline_correct"]), "unanimous": unanimous})

    n = len(rows)
    acc = 100 * sum(r["voted_correct"] for r in rows) / n
    acc1 = 100 * sum(r["first_correct"] for r in rows) / n
    gain = sum(r["voted_correct"] and not r["first_correct"] for r in rows)
    loss = sum(r["first_correct"] and not r["voted_correct"] for r in rows)
    by_n = collections.defaultdict(list)
    for r in rows:
        by_n[r["n_samples"]].append(r)
    disc = {}
    for d, subs in DISCIPLINES.items():
        items = [r for r in rows if r["subject"] in subs]
        if items:
            disc[d] = {"n": len(items),
                       "first": round(100 * sum(r["first_correct"] for r in items) / len(items), 2),
                       "voted": round(100 * sum(r["voted_correct"] for r in items) / len(items), 2)}
    unan = [r for r in rows if r["unanimous"]]
    summary = {
        "runs": args.runs, "n": n, "first_run_acc": round(acc1, 2), "voted_acc": round(acc, 2),
        "delta_pp": round(acc - acc1, 2), "gain": gain, "loss": loss,
        "mcnemar_p": round(mcnemar_exact(gain, loss), 4),
        "unanimous_items": len(unan),
        "unanimous_acc": round(100 * sum(r["voted_correct"] for r in unan) / max(len(unan), 1), 2),
        "no_valid_vote": sum(r["n_valid"] == 0 for r in rows),
        "by_num_samples": {k: {"n": len(v), "voted_acc": round(100 * sum(r["voted_correct"] for r in v) / len(v), 2),
                               "first_acc": round(100 * sum(r["first_correct"] for r in v) / len(v), 2)}
                           for k, v in sorted(by_n.items())},
        "by_discipline": disc,
    }
    print(json.dumps(summary, indent=2))

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "voted.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(os.path.join(args.out, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")
    if args.emit_ids:
        os.makedirs(os.path.dirname(args.emit_ids) or ".", exist_ok=True)
        with open(args.emit_ids, "w") as f:
            f.write("\n".join(need_more) + "\n")
        print(f"{len(need_more)} non-unanimous ids -> {args.emit_ids}")


if __name__ == "__main__":
    main()
