#!/usr/bin/env python3
"""Score a raw.jsonl with the official MMMU parser and write the report tables.

    python scripts/score_mmmu.py outputs/raw.jsonl                      # -> results/raw/
    python scripts/score_mmmu.py outputs/best.jsonl --partial            # smoke / ablation files

Two scores are always produced from the same responses:

* strict   : MMMU official parser (mmmu_official_eval_utils.py @ 268471d, byte-identical, SHA256 checked)
             with the random guess on a parse failure replaced by "no answer" (counted wrong). Nothing else.
* pipeline : strict, preceded by two deterministic steps that are documented in the report:
             1. if <stem>.fallback.jsonl has the item, the second-pass continuation replaces the response
                (multiple-choice: "(" + continuation; open: the continuation);
             2. an explicit final answer is taken before the official rules: for a fallback item the
                letter the second pass produced; otherwise the last "Answer: X" line, else the last
                \\boxed{X}; for open questions the last "Answer: ..." line is parsed instead of the
                whole response.
             Everything else is the official parser.

Extra metrics (finish reasons, verbatim loops, response length groups, MMMU disciplines) are the ones the
team asked for to judge fine-tuning runs. numpy is the only non-stdlib dependency (the official parser needs it).
"""
import argparse
import ast
import collections
import csv
import hashlib
import importlib.util
import json
import os
import re
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from prompt import trim_loop  # noqa: E402

OFFICIAL_PATH = os.path.join(ROOT, "scripts", "mmmu_official_eval_utils.py")
OFFICIAL_SHA256 = "cc1a89b4a27f697702d340bc6a7578b5237135754ddaa9ef105239a3d9847367"
OFFICIAL_COMMIT = "268471d0d488258990025331c7528359c324aa25"
RANDOM_LINE = "pred_index = random.choice(all_choices)"
OFFICIAL_SCORE = 67.4

# MMMU official discipline grouping (MMMU-Benchmark/MMMU mmmu/utils/data_utils.py DOMAIN_CAT2SUB_CAT).
DISCIPLINES = {
    "Art & Design": ["Art", "Art_Theory", "Design", "Music"],
    "Business": ["Accounting", "Economics", "Finance", "Manage", "Marketing"],
    "Science": ["Biology", "Chemistry", "Geography", "Math", "Physics"],
    "Health & Medicine": ["Basic_Medical_Science", "Clinical_Medicine", "Diagnostics_and_Laboratory_Medicine",
                          "Pharmacy", "Public_Health"],
    "Humanities & Social Science": ["History", "Literature", "Sociology", "Psychology"],
    "Tech & Engineering": ["Agriculture", "Architecture_and_Engineering", "Computer_Science", "Electronics",
                           "Energy_and_Power", "Materials", "Mechanical_Engineering"],
}
SUBJECT_ORDER = [s for group in DISCIPLINES.values() for s in group]
SUBJECT_ORDER = sorted(SUBJECT_ORDER)
ANSWER_ONLY_CHARS = 60

ANSWER_LINE = re.compile(r"(?im)^\W{0,3}answer\W{0,3}\s*[:：]\s*(.+?)\s*$")
LETTER_AT_START = re.compile(r"^\W{0,3}([A-I])(?![A-Za-z])")
BOXED = re.compile(r"\\boxed\{\s*\(?([A-I])\)?\s*\}")


def load_official():
    src = open(OFFICIAL_PATH, "rb").read()
    if hashlib.sha256(src).hexdigest() != OFFICIAL_SHA256:
        sys.exit(f"{OFFICIAL_PATH} does not match the pinned MMMU commit {OFFICIAL_COMMIT}")
    text = src.decode()
    if text.count(RANDOM_LINE) != 1:
        sys.exit("expected exactly one random fallback in the official parser")
    text = text.replace(RANDOM_LINE, "pred_index = None")
    spec = importlib.util.spec_from_loader("mmmu_official_no_random", loader=None)
    mod = importlib.util.module_from_spec(spec)
    exec(compile(text, OFFICIAL_PATH, "exec"), mod.__dict__)
    return mod


def gold_of(rec):
    a = rec["answer"]
    if rec["question_type"] == "open" and a.strip().startswith("["):
        return [str(x) for x in ast.literal_eval(a)]
    return a


def official_mc(E, rec, text):
    choices = [chr(65 + i) for i in range(len(rec["options"]))]
    return E.parse_multi_choice_response(text, choices, dict(zip(choices, rec["options"])))


def score_mc(E, rec, text, explicit_first, fb=None):
    """Returns (pred, method)."""
    choices = {chr(65 + i) for i in range(len(rec["options"]))}
    if fb is not None:
        # The second pass continues "...the final answer is (", so its first letter is the answer.
        m = LETTER_AT_START.match(fb["fallback_response"].strip())
        if m and m.group(1) in choices:
            return m.group(1), "fallback_letter"
    if explicit_first:
        lines = [m.group(1) for m in ANSWER_LINE.finditer(text)]
        for line in reversed(lines):
            m = LETTER_AT_START.match(line.strip("*` "))
            if m and m.group(1) in choices:
                return m.group(1), "answer_line"
        boxed = [m.group(1) for m in BOXED.finditer(text) if m.group(1) in choices]
        if boxed:
            return boxed[-1], "boxed"
    pred = official_mc(E, rec, text)
    return pred, ("official" if pred is not None else "none")


def score_open(E, rec, text, explicit_first):
    """Returns (pred_list, method)."""
    if explicit_first:
        lines = [m.group(1) for m in ANSWER_LINE.finditer(text)]
        if lines:
            line = lines[-1].strip("*` ").rstrip(".")
            pred = E.parse_open_response(line)
            if pred:
                return pred, "answer_line"
    if not text.strip():
        return [], "none"
    pred = E.parse_open_response(text)
    return pred, ("official" if pred else "none")


def evaluate(E, rec, text, explicit_first, fb=None):
    gold = gold_of(rec)
    if rec["question_type"] == "multiple-choice":
        pred, method = score_mc(E, rec, text, explicit_first, fb)
        correct = pred is not None and bool(E.eval_multi_choice(gold, pred))
    else:
        pred, method = score_open(E, rec, text, explicit_first)
        correct = bool(pred) and bool(E.eval_open(gold, pred))
    return pred, method, correct


def fallback_text(rec, fb):
    if rec["question_type"] == "multiple-choice":
        return "(" + fb["fallback_response"]
    return fb["fallback_response"]


def pct(a, b):
    return 100.0 * a / b if b else float("nan")


def fmt(x, nd=2):
    return "n/a" if x != x else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw")
    ap.add_argument("--fallback", default="auto", help="fallback jsonl; 'auto' = <stem>.fallback.jsonl if present; 'none'")
    ap.add_argument("--out_dir", default=None, help="default results/<stem>")
    ap.add_argument("--partial", action="store_true", help="do not require 900 records / 30 per subject")
    args = ap.parse_args()

    E = load_official()
    stem = os.path.splitext(os.path.basename(args.raw))[0]
    out_dir = args.out_dir or os.path.join(ROOT, "results", stem)
    os.makedirs(out_dir, exist_ok=True)

    rows = [json.loads(l) for l in open(args.raw, encoding="utf-8") if l.strip()]
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        sys.exit("duplicate ids; run check_raw.py first")
    if not args.partial and len(rows) != 900:
        sys.exit(f"expected 900 records, found {len(rows)} (use --partial for smoke/ablation files)")

    fb_path = None
    if args.fallback == "auto":
        cand = os.path.join(os.path.dirname(os.path.abspath(args.raw)), f"{stem}.fallback.jsonl")
        fb_path = cand if os.path.exists(cand) else None
    elif args.fallback != "none":
        fb_path = args.fallback
    fallback = {}
    if fb_path:
        for l in open(fb_path, encoding="utf-8"):
            if l.strip():
                fb = json.loads(l)
                fallback[fb["id"]] = fb

    scored = []
    for r in rows:
        s_pred, s_method, s_ok = evaluate(E, r, r["response"], explicit_first=False)
        fb = fallback.get(r["id"])
        text = fallback_text(r, fb) if fb else r["response"]
        p_pred, p_method, p_ok = evaluate(E, r, text, explicit_first=True, fb=fb)
        if fb and p_method != "fallback_letter":
            p_method = "fallback+" + p_method
        loop_removed = trim_loop(r["response"])[1] if r["finish_reason"] == "length" else 0
        resp = r["response"].strip()
        group = ("truncated" if r["finish_reason"] == "length"
                 else "answer_only" if len(resp) <= ANSWER_ONLY_CHARS and "\n" not in resp
                 else "reasoned")
        scored.append({
            "id": r["id"], "subject": r["subject"], "question_type": r["question_type"], "answer": r["answer"],
            "finish_reason": r["finish_reason"], "num_images": r["num_images"], "response_chars": len(r["response"]),
            "response_group": group, "loop_chars_removed": loop_removed,
            "strict_pred": s_pred, "strict_method": s_method, "strict_correct": s_ok,
            "pipeline_pred": p_pred, "pipeline_method": p_method, "pipeline_correct": p_ok,
            "used_fallback": r["id"] in fallback,
        })

    with open(os.path.join(out_dir, "scored.jsonl"), "w", encoding="utf-8") as f:
        for s in scored:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    def agg(items):
        n = len(items)
        return {"n": n, "strict_correct": sum(s["strict_correct"] for s in items),
                "pipeline_correct": sum(s["pipeline_correct"] for s in items),
                "strict_acc": pct(sum(s["strict_correct"] for s in items), n),
                "pipeline_acc": pct(sum(s["pipeline_correct"] for s in items), n),
                "truncated": sum(s["finish_reason"] == "length" for s in items),
                "loops": sum(s["loop_chars_removed"] > 0 for s in items),
                "mean_chars": statistics.mean(s["response_chars"] for s in items) if n else float("nan")}

    subjects = sorted({s["subject"] for s in scored}, key=lambda x: SUBJECT_ORDER.index(x) if x in SUBJECT_ORDER else 99)
    by_subject = {sub: agg([s for s in scored if s["subject"] == sub]) for sub in subjects}
    if not args.partial and (len(subjects) != 30 or any(v["n"] != 30 for v in by_subject.values())):
        sys.exit("expected 30 subjects with 30 records each")
    with open(os.path.join(out_dir, "by_subject.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["subject", "n", "strict_correct", "strict_acc", "pipeline_correct", "pipeline_acc",
                                       "truncated", "loops", "mean_chars"])
        for sub, v in by_subject.items():
            w.writerow([sub, v["n"], v["strict_correct"], fmt(v["strict_acc"]), v["pipeline_correct"], fmt(v["pipeline_acc"]),
                        v["truncated"], v["loops"], f"{v['mean_chars']:.0f}"])

    overall = agg(scored)
    macro_strict = statistics.mean(v["strict_acc"] for v in by_subject.values())
    macro_pipe = statistics.mean(v["pipeline_acc"] for v in by_subject.values())
    by_disc = {d: agg([s for s in scored if s["subject"] in subs]) for d, subs in DISCIPLINES.items()}
    by_type = {t: agg([s for s in scored if s["question_type"] == t]) for t in ("multiple-choice", "open")}
    by_group = {g: agg([s for s in scored if s["response_group"] == g]) for g in ("answer_only", "reasoned", "truncated")}
    by_images = {k: agg([s for s in scored if s["num_images"] == k]) for k in sorted({s["num_images"] for s in scored})}
    methods = collections.Counter(s["pipeline_method"] for s in scored)
    fb_used = [s for s in scored if s["used_fallback"]]
    fb_gain = sum(s["pipeline_correct"] and not s["strict_correct"] for s in fb_used)
    fb_loss = sum(s["strict_correct"] and not s["pipeline_correct"] for s in fb_used)
    explicit = [s for s in scored if not s["used_fallback"]]
    ex_gain = sum(s["pipeline_correct"] and not s["strict_correct"] for s in explicit)
    ex_loss = sum(s["strict_correct"] and not s["pipeline_correct"] for s in explicit)

    summary = {
        "raw": os.path.abspath(args.raw), "raw_sha256": hashlib.sha256(open(args.raw, "rb").read()).hexdigest(),
        "fallback_file": fb_path, "parser": {"source": f"MMMU-Benchmark/MMMU @ {OFFICIAL_COMMIT} mmmu/utils/eval_utils.py",
                                             "sha256": OFFICIAL_SHA256, "change": "random guess on parse failure -> no answer (wrong)"},
        "n": overall["n"],
        "strict": {"correct": overall["strict_correct"], "micro_acc": overall["strict_acc"], "macro_acc": macro_strict,
                   "parse_failed": sum(s["strict_method"] == "none" for s in scored)},
        "pipeline": {"correct": overall["pipeline_correct"], "micro_acc": overall["pipeline_acc"], "macro_acc": macro_pipe,
                     "parse_failed": sum(s["pipeline_method"].endswith("none") for s in scored),
                     "methods": dict(methods), "fallback_items": len(fb_used), "fallback_gain": fb_gain, "fallback_loss": fb_loss,
                     "explicit_answer_gain": ex_gain, "explicit_answer_loss": ex_loss},
        "delta_vs_official_pp": {"strict": overall["strict_acc"] - OFFICIAL_SCORE, "pipeline": overall["pipeline_acc"] - OFFICIAL_SCORE},
        "generation": {"truncated": overall["truncated"], "loops": overall["loops"], "mean_chars": overall["mean_chars"],
                       "median_chars": statistics.median(s["response_chars"] for s in scored),
                       "answer_only": by_group["answer_only"]["n"]},
        "by_discipline": by_disc, "by_question_type": by_type, "by_response_group": by_group, "by_num_images": by_images,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    md = []
    md.append(f"# Results for `{os.path.basename(args.raw)}`\n")
    md.append("Acc = pipeline score (%, two decimals). Strict = official parser only. See `summary.json`.\n")
    md.append("| No. | Subject | Data Num | Acc | Strict | length | loops |\n|---|---|---|---|---|---|---|")
    for i, (sub, v) in enumerate(by_subject.items(), 1):
        md.append(f"| {i} | {sub} | {v['n']} | {fmt(v['pipeline_acc'])} | {fmt(v['strict_acc'])} | {v['truncated']} | {v['loops']} |")
    md.append(f"| | **Overall (macro avg)** | **{overall['n']}** | **{fmt(macro_pipe)}** | **{fmt(macro_strict)}** | {overall['truncated']} | {overall['loops']} |\n")
    md.append(f"Overall = mean of the {len(by_subject)} subject accuracies = {fmt(macro_pipe, 4)}. "
              f"Micro = {overall['pipeline_correct']} / {overall['n']} × 100 = {fmt(overall['pipeline_acc'], 4)}"
              + (" (equal because every subject has 30 items)." if not args.partial else "."))
    md.append("")
    md.append("| | Overall (MMMU val) |\n|---|---:|")
    md.append(f"| Official (Qwen3-VL Technical Report) | {OFFICIAL_SCORE:.2f} |")
    md.append(f"| Ours, pipeline | {fmt(overall['pipeline_acc'])} |")
    md.append(f"| Ours, strict official parser | {fmt(overall['strict_acc'])} |")
    md.append(f"| Δ pipeline − official | {overall['pipeline_acc'] - OFFICIAL_SCORE:+.2f} pp |\n")
    md.append("## Generation behaviour\n")
    md.append("| Response group | n | Acc | Strict |\n|---|---:|---:|---:|")
    for g, v in by_group.items():
        md.append(f"| {g} | {v['n']} | {fmt(v['pipeline_acc'])} | {fmt(v['strict_acc'])} |")
    md.append(f"\nTruncated (finish_reason=length): {overall['truncated']}. Verbatim loops among them: {overall['loops']}. "
              f"Mean response length: {overall['mean_chars']:.0f} chars, median {summary['generation']['median_chars']:.0f}. "
              f"Answer-only responses (≤{ANSWER_ONLY_CHARS} chars, one line): {by_group['answer_only']['n']}.\n")
    md.append("## By MMMU discipline\n")
    md.append("| Discipline | n | Acc | Strict | length | loops | mean chars |\n|---|---:|---:|---:|---:|---:|---:|")
    for d, v in by_disc.items():
        md.append(f"| {d} | {v['n']} | {fmt(v['pipeline_acc'])} | {fmt(v['strict_acc'])} | {v['truncated']} | {v['loops']} | {v['mean_chars']:.0f} |")
    md.append("\n## By question type\n")
    md.append("| Type | n | Acc | Strict |\n|---|---:|---:|---:|")
    for t, v in by_type.items():
        md.append(f"| {t} | {v['n']} | {fmt(v['pipeline_acc'])} | {fmt(v['strict_acc'])} |")
    md.append("\n## Parser\n")
    md.append(f"Extraction methods (pipeline): {dict(methods)}. Explicit-answer rule changed {ex_gain} wrong→right and {ex_loss} right→wrong. "
              f"Fallback covered {len(fb_used)} items: {fb_gain} wrong→right, {fb_loss} right→wrong. "
              f"Parse failures: strict {summary['strict']['parse_failed']}, pipeline {summary['pipeline']['parse_failed']}.")
    with open(os.path.join(out_dir, "results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(f"{args.raw}: n={overall['n']}  strict {overall['strict_correct']}/{overall['n']} = {fmt(overall['strict_acc'])}%  "
          f"pipeline {overall['pipeline_correct']}/{overall['n']} = {fmt(overall['pipeline_acc'])}%  "
          f"(Δ vs {OFFICIAL_SCORE}: {overall['pipeline_acc'] - OFFICIAL_SCORE:+.2f} pp)")
    print(f"truncated {overall['truncated']}, loops {overall['loops']}, answer-only {by_group['answer_only']['n']}, "
          f"fallback items {len(fb_used)} (+{fb_gain}/-{fb_loss}), explicit-answer rule (+{ex_gain}/-{ex_loss})")
    print(f"wrote {out_dir}/{{scored.jsonl,by_subject.csv,summary.json,results.md}}")


if __name__ == "__main__":
    main()
