"""Experiment S, step 0: collect a pool of public images (with their original text) for question synthesis.

  python step15_s_collect_images.py --out ../synth --scale 0.1        # about 100 images; --scale 1 is about 1,000

Sources are public Hugging Face datasets, streamed (no full download). Each source lists the MMMU subjects its
images can serve. Kept: single-image rows whose image is at least 224 px on both sides, not a near-duplicate of
an image already in the pool, and not a near-duplicate of any MMMU validation image (64-bit dHash, Hamming <= 4,
the same rule as step01_pick_questions.py). Writes <out>/images/*.jpg, <out>/pool.jsonl and <out>/pool.info.json.
pool rows: {"id", "image", "source", "subjects", "context"}; "context" is the dataset's own question/answer or
label text for that image, shown to the teacher only as background.
"""
import argparse
import collections
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "MMD", "src"))

CAULDRON = "HuggingFaceM4/the_cauldron"
SCI = ["Biology", "Chemistry", "Physics", "Geography", "Basic_Medical_Science"]
# name: (repo, config, kind, quota at scale 1, subjects)
SOURCES = {
    "ai2d": (CAULDRON, "ai2d", "cauldron", 100, SCI),
    "tqa": (CAULDRON, "tqa", "cauldron", 100, SCI + ["Energy_and_Power"]),
    "finqa": (CAULDRON, "finqa", "cauldron", 80, ["Accounting", "Finance", "Economics"]),
    "tat_qa": (CAULDRON, "tat_qa", "cauldron", 60, ["Accounting", "Finance"]),
    "chartqa": (CAULDRON, "chartqa", "cauldron", 80, ["Economics", "Marketing", "Manage", "Sociology", "Public_Health"]),
    "hitab": (CAULDRON, "hitab", "cauldron", 40, ["Economics", "Sociology", "Public_Health", "Manage"]),
    "geomverse": (CAULDRON, "geomverse", "cauldron", 50, ["Math"]),
    "intergps": (CAULDRON, "intergps", "cauldron", 40, ["Math"]),
    "tabmwp": (CAULDRON, "tabmwp", "cauldron", 40, ["Math", "Accounting"]),
    "mapqa": (CAULDRON, "mapqa", "cauldron", 40, ["Geography", "Public_Health"]),
    "infographic_vqa": (CAULDRON, "infographic_vqa", "cauldron", 40, ["Marketing", "Public_Health", "Sociology"]),
    "datikz": (CAULDRON, "datikz", "cauldron_user", 120, ["Computer_Science", "Electronics", "Math", "Physics"]),
    "vqarad": (CAULDRON, "vqarad", "cauldron", 30, ["Clinical_Medicine", "Diagnostics_and_Laboratory_Medicine"]),
    "pathvqa": ("flaviagiammarino/path-vqa", None, "qa", 100,
                ["Diagnostics_and_Laboratory_Medicine", "Basic_Medical_Science", "Clinical_Medicine"]),
    "wikiart": ("huggan/wikiart", None, "wikiart", 80, ["Art", "Art_Theory", "Design"]),
}


def dhash(img):
    g = img.convert("L").resize((9, 8))
    px = list(g.getdata())
    bits = 0
    for r in range(8):
        for c in range(8):
            bits = (bits << 1) | (px[r * 9 + c] > px[r * 9 + c + 1])
    return bits


def buffered_shuffle(rows, rng, size=200):
    """Seeded shuffle of a stream through a fixed-size buffer (plain Python, so it does not depend on dill)."""
    buf = []
    for row in rows:
        buf.append(row)
        if len(buf) >= size:
            yield buf.pop(rng.randrange(len(buf)))
    rng.shuffle(buf)
    yield from buf


def row_to_image_context(row, kind, features):
    """-> (PIL image, context text) or None."""
    if kind.startswith("cauldron"):
        if len(row["images"]) != 1:
            return None
        turns = row["texts"][:2]
        if kind == "cauldron_user":   # the answer is source code, not useful as background
            ctx = "\n".join(t["user"] for t in turns)
        else:
            ctx = "\n".join(f"Q: {t['user']}\nA: {t['assistant']}" for t in turns)
        return row["images"][0], ctx
    if kind == "qa":
        return row["image"], f"Q: {row['question']}\nA: {row['answer']}"
    if kind == "wikiart":
        lab = {k: features[k].int2str(row[k]) for k in ("artist", "genre", "style")}
        return row["image"], f"Artist: {lab['artist']}; genre: {lab['genre']}; style: {lab['style']}".replace("_", " ")
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every source quota")
    ap.add_argument("--sources", default=None, help="comma-separated subset of: " + ",".join(SOURCES))
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--no_mmmu_check", action="store_true", help="skip the MMMU overlap check (local smoke test only)")
    args = ap.parse_args()

    from datasets import load_dataset

    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    pool_path = os.path.join(args.out, "pool.jsonl")
    have = collections.Counter()
    seen_hash = []
    if os.path.exists(pool_path):
        from PIL import Image
        for l in open(pool_path, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                have[r["source"]] += 1
                with Image.open(r["image"]) as im:
                    seen_hash.append(dhash(im))
    print(f"pool already has {sum(have.values())} images", flush=True)

    mmmu_hash = []
    if not args.no_mmmu_check:
        from data import SUBJECTS, load_subject
        hs = set()
        for s in SUBJECTS:
            for it in load_subject(s):
                for im in it["images"].values():
                    hs.add(dhash(im))
        mmmu_hash = list(hs)
        print(f"MMMU validation: {len(mmmu_hash)} image hashes", flush=True)

    def near(h, hashes):
        return any(bin(h ^ m).count("1") <= 4 for m in hashes)

    names = args.sources.split(",") if args.sources else list(SOURCES)
    drop = collections.Counter()
    with open(pool_path, "a", encoding="utf-8") as fout:
        for name in names:
            repo, config, kind, quota, subjects = SOURCES[name]
            want = max(1, round(quota * args.scale))
            if have[name] >= want:
                continue
            try:
                ds = load_dataset(repo, config, split="train", streaming=True)
                feats = ds.features
                n_seen = 0
                for row in buffered_shuffle(ds, random.Random(f"{args.seed}:{name}")):
                    if have[name] >= want or n_seen >= want * 30:
                        break
                    n_seen += 1
                    got = row_to_image_context(row, kind, feats)
                    if got is None:
                        drop["multi_image"] += 1
                        continue
                    img, ctx = got
                    if min(img.size) < 224:
                        drop["small"] += 1
                        continue
                    img = img.convert("RGB")
                    h = dhash(img)
                    if near(h, seen_hash):
                        drop["duplicate"] += 1
                        continue
                    if near(h, mmmu_hash):
                        drop["mmmu_overlap"] += 1
                        continue
                    seen_hash.append(h)
                    iid = f"{name}-{have[name]:05d}"
                    path = os.path.abspath(os.path.join(args.out, "images", iid + ".jpg"))
                    img.save(path, quality=95)
                    fout.write(json.dumps({"id": iid, "image": path, "source": name, "subjects": subjects,
                                           "context": ctx[:700]}, ensure_ascii=False) + "\n")
                    fout.flush()
                    have[name] += 1
            except Exception as e:   # one unavailable source must not stop the others
                print(f"FAILED source {name}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            print(f"{name}: {have[name]}/{want}", flush=True)
    info = {"images": sum(have.values()), "per_source": dict(have), "dropped": dict(drop),
            "mmmu_image_hashes": len(mmmu_hash), "scale": args.scale, "seed": args.seed}
    json.dump(info, open(os.path.join(args.out, "pool.info.json"), "w"), indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
