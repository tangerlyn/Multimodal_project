"""Turn a scored MMMU test run into the answer file of the official MMMU evaluation server.
  python step12_make_submission.py results/<run>/scored.jsonl submissions/<run>.json
Multiple-choice items get the option letter, open items the first parsed answer; an item without a prediction
gets an empty string. The format ({"test_<Subject>_<n>": answer}) follows the MMMU repo's example submission,
which should be checked against the current instructions on the evaluation server before uploading.
"""
import json
import os
import sys

src, dst = sys.argv[1:3]
out, empty = {}, 0
for line in open(src):
    r = json.loads(line)
    pred = r.get("pipeline_pred")
    if isinstance(pred, list):
        pred = str(pred[0]) if pred else ""
    pred = "" if pred is None else str(pred)
    empty += pred == ""
    out[r["id"]] = pred
os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
json.dump(out, open(dst, "w"), ensure_ascii=False, indent=0)
print(f"{len(out)} answers written to {dst}, {empty} empty")
