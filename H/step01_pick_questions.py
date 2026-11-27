"""Experiment H, step 0: pick a category-balanced, auto-verifiable ViRL39K subset and drop MMMU-val overlaps.

  python step01_pick_questions.py --out ../data/items.jsonl   (--virl defaults to $VIRL_DIR or ../data/virl)

Kept: single-image questions whose gold is one option letter (with parsable "Choices: (A) ...") or a plain number.
ViRL39K is 65% grade-school math; math-heavy RL data lowered MMMU in prior work, so the subset is balanced
across categories (QUOTA) instead of sampled proportionally.
Contamination: 64-bit dHash of every MMMU validation image vs every candidate image (Hamming <= 4 dropped),
plus exact match of the normalised question text.
Prompt = the MMMU eval text (vlmevalkit style + Qwen CoT sentence + answer line).
"""
import argparse
import ast
import collections
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "MMD", "src"))
from data import SUBJECTS, load_subject  # noqa: E402
from prompt import build_text  # noqa: E402

QUOTA = {"(GradeSchool) Science": 900, "Broader STEM Topics": 500, "Social Science": 600,
         "Tables/Diagrams/Charts": 500, "Commonsense": 120, "Spatial Reasoning": 250,
         "(GradeSchool) Geometric": 200, "(GradeSchool) Non-Geo Math": 200}
OPTION = re.compile(r"^\(?([A-J])[).]\s+(.*)$")
PLAIN_NUM = re.compile(r"^-?\d+(\.\d+)?$")


def dhash(img):
    g = img.convert("L").resize((9, 8))
    px = list(g.getdata())
    bits = 0
    for r in range(8):
        for c in range(8):
            bits = (bits << 1) | (px[r * 9 + c] > px[r * 9 + c + 1])
    return bits


def norm(q):
    return re.sub(r"[^a-z0-9]+", " ", q.lower()).strip()


def parse(row):
    """-> (question, options, gold, kind) or None."""
    m = re.fullmatch(r"\\boxed\{(.*)\}", row["answer"].strip(), re.S)
    if not m:
        return None
    gold = m.group(1).strip()
    q = row["question"].replace("<image>/n", "").replace("<image>", "").strip()
    if "<image" in q:
        return None
    lines = [l.strip() for l in q.split("\n") if l.strip()]
    opts = []
    while lines:                                   # trailing block of option lines, A.. in order
        mo = OPTION.match(lines[-1])
        if not mo:
            break
        opts.append((mo.group(1), mo.group(2).strip()))
        lines.pop()
    opts.reverse()
    if opts:
        if [l for l, _ in opts] != [chr(65 + i) for i in range(len(opts))] or len(opts) < 2:
            return None
        if lines and lines[-1].rstrip(":").lower() in ("choices", "options"):
            lines.pop()
        stem = re.sub(r"^Question:\s*", "", "\n".join(lines)).strip()
        if not stem or not re.fullmatch(r"[A-J]", gold) or ord(gold) - 65 >= len(opts):
            return None
        return stem, [t for _, t in opts], gold, "mc"
    q = re.sub(r"^Question:\s*", "", q).strip()
    if PLAIN_NUM.match(gold):
        return q, [], gold, "num"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--virl", default=os.environ.get("VIRL_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "virl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every category quota")
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    import pandas as pd
    from PIL import Image

    df = pd.read_parquet(os.path.join(args.virl, "39Krelease.parquet"))
    rng = random.Random(args.seed)
    cands = collections.defaultdict(list)
    drop = collections.Counter()
    for row in df.to_dict("records"):
        imgs = ast.literal_eval(row["image"]) if isinstance(row["image"], str) else list(row["image"])
        if len(imgs) != 1:
            drop["multi_image"] += 1
            continue
        path = os.path.join(args.virl, imgs[0])
        if not os.path.isfile(path):
            drop["missing_image"] += 1
            continue
        p = parse(row)
        if p is None:
            drop["not_verifiable"] += 1
            continue
        q, opts, gold, kind = p
        cands[row["category"]].append({"id": row["qid"], "image": path, "question": q, "options": opts, "gold": gold,
                                       "kind": kind, "category": row["category"], "source": row["source"]})
    print("verifiable candidates:", {k: len(v) for k, v in cands.items()}, dict(drop), flush=True)

    print("hashing MMMU validation images ...", flush=True)
    mmmu_hash, mmmu_q = set(), set()
    for s in SUBJECTS:
        for it in load_subject(s):
            mmmu_q.add(norm(it["question"]))
            for im in it["images"].values():
                mmmu_hash.add(dhash(im))
    mmmu_hash = list(mmmu_hash)
    print(f"MMMU: {len(mmmu_hash)} image hashes, {len(mmmu_q)} questions", flush=True)

    def contaminated(it):
        if norm(it["question"]) in mmmu_q:
            return "question"
        with Image.open(it["image"]) as im:
            h = dhash(im)
        return "image" if any(bin(h ^ m).count("1") <= 4 for m in mmmu_hash) else None

    items, overlap, per_cat = [], collections.Counter(), {}
    for cat, quota in QUOTA.items():
        pool = cands.get(cat, [])
        rng.shuffle(pool)
        want, got = int(quota * args.scale), 0
        for it in pool:
            if got >= want:
                break
            why = contaminated(it)
            if why:
                overlap[why] += 1
                continue
            sample = {"question_type": "multiple-choice" if it["kind"] == "mc" else "open",
                      "question": it["question"], "options": it["options"]}
            it["prompt"] = build_text(sample, "vlmevalkit", True, "qwen")
            items.append(it)
            got += 1
        per_cat[cat] = {"available": len(pool), "taken": got}
    rng.shuffle(items)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    info = {"items": len(items), "per_category": per_cat, "kinds": dict(collections.Counter(i["kind"] for i in items)),
            "sources": dict(collections.Counter(i["source"] for i in items)), "dropped": dict(drop),
            "mmmu_overlap_dropped": dict(overlap), "mmmu_image_hashes": len(mmmu_hash)}
    json.dump(info, open(os.path.splitext(args.out)[0] + ".info.json", "w"), indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
