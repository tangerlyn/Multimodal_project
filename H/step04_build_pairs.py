"""Experiment H, step 2: difficulty filter + preference pairs from gen.jsonl.

  python step04_build_pairs.py --gen build/gen.jsonl --out build

Keeps questions whose pass rate (correct / valid samples, at least --min_valid valid) is in [--lo, --hi]
(Zhang et al. 2024 use 0.25-0.85; Vision-G1 uses 0.2-0.8). For each kept question, up to --pairs
(pos, neg) pairs: positives are correct samples, shortest first; negatives are wrong samples, preferring
distinct wrong answers. Questions are split 90/10 into train/val by id hash, so SFT and DPO share the split.
With --trunc_pairs N, each question that has both a short correct sample and a sample cut off at the token limit
also gives up to N (short correct, cut-off) pairs, so the model is pushed away from answers that never finish.
Writes pairs_train.jsonl, pairs_val.jsonl and stats.json.
"""
import argparse
import collections
import hashlib
import json
import os
import random


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lo", type=float, default=0.25)
    ap.add_argument("--hi", type=float, default=0.85)
    ap.add_argument("--min_valid", type=int, default=6)
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--max_tokens", type=int, default=700, help="drop samples longer than this (training memory)")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--trunc_pairs", type=int, default=0,
                    help="extra pairs per question: pos = a short correct sample, neg = a sample that hit the token "
                         "limit without answering (any pass rate; 0 = off, the original experiment)")
    ap.add_argument("--trunc_pos_max", type=int, default=700, help="longest positive allowed in a truncation pair")
    ap.add_argument("--neg_cap", type=int, default=768, help="truncation negatives are cut to about this many tokens")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rng_t = random.Random(args.seed + 1)   # separate stream: the original pairs stay identical when --trunc_pairs is on

    rows = [json.loads(l) for l in open(args.gen) if l.strip()]
    hist, kept, pairs = collections.Counter(), [], {"train": [], "val": []}
    by_cat = collections.defaultdict(lambda: [0, 0])
    n_samples = n_valid = n_correct = 0
    for r in rows:
        ss = r["samples"]
        n_samples += len(ss)
        valid = [s for s in ss if s["valid"]]
        n_valid += len(valid)
        n_correct += sum(s["correct"] for s in valid)
        by_cat[r["category"]][0] += 1
        split = "val" if int(hashlib.sha256(r["id"].encode()).hexdigest()[:8], 16) % 10 == 0 else "train"
        if args.trunc_pairs:
            cut = [s for s in ss if s["finish"] == "length"]
            short = sorted({s["text"]: s for s in valid if s["correct"] and s["n_tokens"] <= args.trunc_pos_max}.values(),
                           key=lambda s: s["n_tokens"])
            if cut and short:
                rng_t.shuffle(cut)
                hist["trunc_questions"] += 1
                for k in range(min(args.trunc_pairs, len(cut))):
                    p, n = short[k % len(short)], cut[k]
                    keep = int(len(n["text"]) * min(1.0, args.neg_cap / max(n["n_tokens"], 1)))
                    pairs[split].append({"id": r["id"], "image": r["image"], "prompt": r["prompt"], "gold": r["gold"],
                                         "kind": r["kind"], "category": r["category"], "pass_rate": None,
                                         "pair_type": "trunc", "pos": p["text"].strip(), "neg": n["text"][:keep].strip(),
                                         "pos_tokens": p["n_tokens"], "neg_tokens": min(n["n_tokens"], args.neg_cap)})
        if len(valid) < args.min_valid:
            hist["too_few_valid"] += 1
            continue
        rate = sum(s["correct"] for s in valid) / len(valid)
        hist[f"{sum(s['correct'] for s in valid)}/{len(valid)}"] += 1
        if not (args.lo <= rate <= args.hi):
            continue
        pos = sorted({s["text"]: s for s in valid if s["correct"] and s["n_tokens"] <= args.max_tokens}.values(),
                     key=lambda s: s["n_tokens"])
        neg = [s for s in valid if not s["correct"] and s["n_tokens"] <= args.max_tokens]
        rng.shuffle(neg)
        seen, neg_div = set(), []
        for s in neg:                      # distinct wrong answers first
            if s["pred"] not in seen:
                seen.add(s["pred"]); neg_div.append(s)
        neg = neg_div + [s for s in neg if s not in neg_div]
        if not pos or not neg:
            hist["no_pair_after_length_cut"] += 1
            continue
        kept.append(r["id"])
        by_cat[r["category"]][1] += 1
        for k in range(min(args.pairs, max(len(pos), len(neg)))):
            p, n = pos[k % len(pos)], neg[k % len(neg)]
            pairs[split].append({"id": r["id"], "image": r["image"], "prompt": r["prompt"], "gold": r["gold"],
                                 "kind": r["kind"], "category": r["category"], "pass_rate": round(rate, 3),
                                 "pair_type": "wrong", "pos": p["text"].strip(), "neg": n["text"].strip(),
                                 "pos_tokens": p["n_tokens"], "neg_tokens": n["n_tokens"]})
    os.makedirs(args.out, exist_ok=True)
    for split, ps in pairs.items():
        rng.shuffle(ps)
        with open(os.path.join(args.out, f"pairs_{split}.jsonl"), "w") as f:
            for p in ps:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
    stats = {
        "questions": len(rows), "samples": n_samples, "valid_samples": n_valid,
        "sample_accuracy": round(n_correct / max(n_valid, 1), 4), "kept_questions": len(kept),
        "pairs_train": len(pairs["train"]), "pairs_val": len(pairs["val"]),
        "pairs_train_by_type": dict(collections.Counter(p["pair_type"] for p in pairs["train"])),
        "sft_examples_train": len({(p["id"], p["pos"]) for p in pairs["train"]}),
        "mean_pos_tokens": round(sum(p["pos_tokens"] for p in pairs["train"]) / max(len(pairs["train"]), 1), 1),
        "mean_neg_tokens": round(sum(p["neg_tokens"] for p in pairs["train"]) / max(len(pairs["train"]), 1), 1),
        "filter": {"lo": args.lo, "hi": args.hi, "min_valid": args.min_valid, "max_tokens": args.max_tokens},
        "pass_histogram": dict(sorted(hist.items())),
        "by_category": {k: {"questions": v[0], "kept": v[1]} for k, v in sorted(by_cat.items())},
    }
    json.dump(stats, open(os.path.join(args.out, "stats.json"), "w"), indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
