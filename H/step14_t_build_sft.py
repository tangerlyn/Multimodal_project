"""Experiment T, step 2: turn the teacher's solutions (step13_t_teacher_solutions.py) into SFT rows for step05_train.py --mode sft.

  python step14_t_build_sft.py --teacher buildT/teacher.jsonl --out buildT

Keeps a solution only if it finished, its final answer equals the gold answer, and it has at most --max_tokens
tokens. Rows: {"id", "image", "prompt", "pos", "student_pass"}; "prompt" is the student's eval prompt (without the
extra "keep it short" sentence the teacher saw). Questions are split 90/10 into train/val by the same id hash as
step04_build_pairs.py. Writes pairs_train.jsonl, pairs_val.jsonl and stats.json.
"""
import argparse
import hashlib
import json
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_tokens", type=int, default=700)
    ap.add_argument("--max_student_pass", type=float, default=1.0,
                    help="keep only questions the student solves at most this often (1.0 = keep all)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.teacher, encoding="utf-8") if l.strip()]
    out = {"train": [], "val": []}
    st = {"questions": len(rows), "teacher_correct": 0, "dropped_long": 0, "dropped_student_pass": 0,
          "teacher_correct_where_student_pass_le_0.25": 0, "questions_student_pass_le_0.25": 0}
    for r in rows:
        t, sp = r["teacher"], r.get("student_pass")
        hard = sp is not None and sp <= 0.25
        st["questions_student_pass_le_0.25"] += hard
        if not (t["correct"] and t["finish"] == "stop"):
            continue
        st["teacher_correct"] += 1
        st["teacher_correct_where_student_pass_le_0.25"] += hard
        if t["n_tokens"] > args.max_tokens:
            st["dropped_long"] += 1
            continue
        if sp is not None and sp > args.max_student_pass:
            st["dropped_student_pass"] += 1
            continue
        split = "val" if int(hashlib.sha256(r["id"].encode()).hexdigest()[:8], 16) % 10 == 0 else "train"
        out[split].append({"id": r["id"], "image": r["image"], "prompt": r["prompt"], "pos": t["text"],
                           "student_pass": sp})
    os.makedirs(args.out, exist_ok=True)
    for k, v in out.items():
        with open(os.path.join(args.out, f"pairs_{k}.jsonl"), "w", encoding="utf-8") as f:
            for x in v:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        st[f"rows_{k}"] = len(v)
    json.dump(st, open(os.path.join(args.out, "stats.json"), "w"), indent=2)
    print(json.dumps(st, indent=2))


if __name__ == "__main__":
    main()
