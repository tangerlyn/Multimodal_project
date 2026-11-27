"""Add MMMU-Pro to the MMMU evaluator, selected with the MMMU_DATASET environment variable. Idempotent.
  python step08_patch_pro.py /content/MMD
  MMMU_DATASET=pro10       MMMU/MMMU_Pro "standard (10 options)", test split, 1730 items
  MMMU_DATASET=pro4        "standard (4 options)"
  MMMU_DATASET=pro_vision  "vision": the question and options are inside one screenshot; the text prompt is the
                           instruction sentence of the MMMU-Pro repo ("Write out the multiple-choice question in the
                           image and then solve it.") followed by the same CoT and answer-line sentences as elsewhere
Without the variable the evaluator behaves as before (MMMU val). Option letters are widened from A-I to A-L
because MMMU-Pro has up to 12 options. Item ids keep the dataset's form (test_<Subject>_<n>), so use
score_mmmu.py --partial and skip check_raw.py.
"""
import sys
root = sys.argv[1]

p = f"{root}/src/data.py"; s = open(p).read()
if "MMMU_DATASET" not in s:
    s += '''

# --- MMMU-Pro (patch_pro.py) ---
import os as _os

_PRO = {"pro10": "standard (10 options)", "pro4": "standard (4 options)", "pro_vision": "vision"}.get(
    _os.environ.get("MMMU_DATASET", ""))
VISION_QUESTION = "Write out the multiple-choice question in the image and then solve it."
if _PRO:
    DATASET_REPO = "MMMU/MMMU_Pro"
    DATASET_REVISION = f"main (unpinned), config {_PRO}"
    SPLIT = "test"
    _cache = {}

    def load_subject(subject, data_root=None, revision=None):
        if not _cache:
            ds = load_dataset(DATASET_REPO, _PRO, split=SPLIT, cache_dir=data_root)
            by = {}
            for i, sub in enumerate(ds["subject"]):
                by.setdefault(sub, []).append(i)
            _cache["ds"], _cache["by"] = ds, by
        samples = []
        for i in _cache["by"].get(subject, []):
            row = _cache["ds"][i]
            vision = _PRO == "vision"
            if vision:
                images = {1: row["image"]}
            else:
                images = {n: row[f"image_{n}"] for n in range(1, MAX_IMAGES + 1) if row[f"image_{n}"] is not None}
            samples.append({
                "id": row["id"], "subject": subject, "question_type": "multiple-choice",
                "question": VISION_QUESTION if vision else row["question"],
                "options": [str(o) for o in ast.literal_eval(row["options"])],
                "answer": row["answer"], "images": images, "vision": vision,
            })
        return samples
'''
    open(p, "w").write(s)

p = f"{root}/src/prompt.py"; s = open(p).read()
if 'sample.get("vision")' not in s:
    a = "        text = text.rstrip()\n    else:"
    assert s.count(a) == 1
    s = s.replace(a, '        text = text.rstrip()\n        if sample.get("vision"):\n            text = sample["question"]\n    else:')
    open(p, "w").write(s)

for p in (f"{root}/src/eval_mmmu.py", f"{root}/scripts/score_mmmu.py"):
    s = open(p).read()
    if "[A-I]" in s:
        open(p, "w").write(s.replace("[A-I]", "[A-L]"))
print("pro patch present:", "MMMU_DATASET" in open(f"{root}/src/data.py").read(),
      'sample.get("vision")' in open(f"{root}/src/prompt.py").read(),
      "[A-I]" not in open(f"{root}/src/eval_mmmu.py").read() + open(f"{root}/scripts/score_mmmu.py").read())
