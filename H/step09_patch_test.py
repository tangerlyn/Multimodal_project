"""Add the MMMU test split (10,500 items) to the evaluator, selected with MMMU_DATASET=mmmu_test. Idempotent.
  python step09_patch_test.py /content/MMD
The Hugging Face MMMU/MMMU test split now ships the answers, so score_mmmu.py scores a test run locally;
step12_make_submission.py also turns the scored file into the answer file for the official MMMU evaluation server.
Without the variable nothing changes.
"""
import sys
root = sys.argv[1]
p = f"{root}/src/data.py"
s = open(p).read()
mark = "# --- MMMU test split (patch_test.py) ---"
if mark not in s:
    s += f'''

{mark}
import os as _os2
if _os2.environ.get("MMMU_DATASET", "") == "mmmu_test":
    SPLIT = "test"   # same repo and revision as validation; load_subject reads SPLIT when it is called
'''
    open(p, "w").write(s)
print("patched" if mark in open(p).read() else "NOT patched")
