"""Experiment H, step 1b: re-sample questions that lost samples to the token limit.

  python step03_redo_truncated.py select --gen build/gen.jsonl --out build/items_redo.jsonl
  python step02_sample_answers.py --items build/items_redo.jsonl --out build/gen_redo.jsonl --max_tokens 2048
  python step03_redo_truncated.py merge --gen build/gen.jsonl --redo build/gen_redo.jsonl --out build/gen2.jsonl

select: questions with fewer than --min_valid valid samples and at least one finish=length sample.
merge : gen rows, with every re-sampled question replaced by its new row (all 8 samples new).
"""
import argparse
import json


def rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["select", "merge"])
    ap.add_argument("--gen", required=True)
    ap.add_argument("--redo")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_valid", type=int, default=6)
    args = ap.parse_args()
    gen = rows(args.gen)
    if args.cmd == "select":
        out = [{k: v for k, v in r.items() if k != "samples"} for r in gen
               if sum(s["valid"] for s in r["samples"]) < args.min_valid
               and any(s["finish"] == "length" for s in r["samples"])]
    else:
        new = {r["id"]: r for r in rows(args.redo)}
        out = [new.get(r["id"], r) for r in gen]
        print(f"replaced {sum(r['id'] in new for r in gen)} of {len(gen)} rows")
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{args.cmd}: wrote {len(out)} rows to {args.out}")


if __name__ == "__main__":
    main()
