"""Retry on truncation: regenerate the answers that hit the output cap with new seeds, keep a retry only if it finished.

  python step11_retry.py select  outputs/NAME.jsonl            outputs/NAME_retry_r1.ids
  python step11_retry.py select  outputs/NAME_retry_r1.jsonl   outputs/NAME_retry_r2.ids
  python step11_retry.py merge   outputs/NAME.jsonl outputs/NAME_retry   (reads NAME_retry_r1.jsonl, NAME_retry_r2.jsonl)

merge writes outputs/NAME_retry.jsonl (+ .fallback.jsonl): every record of NAME, with a truncated one replaced by the
first finished retry (round 1 before round 2). This is the rule behind every "+ 재풀이" number in the report.
"""
import json
import os
import sys


def load(path):
    return [json.loads(line) for line in open(path) if line.strip()] if os.path.exists(path) else []


def select(src, dst):
    ids = [r["id"] for r in load(src) if r["finish_reason"] == "length"]
    with open(dst, "w") as f:
        f.write("".join(i + "\n" for i in ids))
    print(f"{src}: {len(ids)} truncated -> {dst}")


def merge(base_path, out_stem, rounds=(1, 2)):
    rep, fb = {}, {}
    for k in rounds:
        fallback = {r["id"]: r for r in load(f"{out_stem}_r{k}.fallback.jsonl")}
        for r in load(f"{out_stem}_r{k}.jsonl"):
            if r["finish_reason"] != "length" and r["id"] not in rep:
                rep[r["id"]] = r
                if r["id"] in fallback:
                    fb[r["id"]] = fallback[r["id"]]
    rows = load(base_path)
    with open(f"{out_stem}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(rep.get(r["id"], r), ensure_ascii=False) + "\n")
    base_fb = base_path[:-len(".jsonl")] + ".fallback.jsonl"
    with open(f"{out_stem}.fallback.jsonl", "w") as f:
        for r in [r for r in load(base_fb) if r["id"] not in rep] + list(fb.values()):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = sum(r["finish_reason"] == "length" for r in rows)
    print(f"truncated {n}, replaced by a finished retry {len(rep)}, still truncated {n - len(rep)} -> {out_stem}.jsonl")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "select" and len(sys.argv) == 4:
        select(sys.argv[2], sys.argv[3])
    elif cmd == "merge" and len(sys.argv) == 4:
        merge(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)
