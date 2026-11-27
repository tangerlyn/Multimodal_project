"""Experiment H, step 1: sample N rationales per question with the base model (vLLM) and score each one.

  python step02_sample_answers.py --items build/items.jsonl --out build/gen.jsonl --n 8

items.jsonl rows: {"id", "image", "prompt", "gold", "kind": "mc"|"num", "category"}.
The prompt is already the MMMU-eval text (vlmevalkit style + Qwen CoT sentence + answer line); the image goes first.
Sampling = the eval recipe (T 0.7, top_p 0.8, top_k 20, presence_penalty 1.5). Appends one row per question
with all samples ({"text", "finish", "pred", "valid", "correct"}); a restarted run skips finished ids.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "MMD", "src"))
from prompt import render, resize_image  # noqa: E402


def local_image(path):
    """Image paths in the data files were written on Colab (/content/virl/...). Map them to $VIRL_DIR
    (default: <repo>/data/virl) when the original path does not exist."""
    if os.path.exists(path):
        return path
    root = os.environ.get("VIRL_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "virl")
    tail = path.split("/virl/", 1)[1] if "/virl/" in path else os.path.basename(path)
    return os.path.join(root, tail)

MODEL_REPO = "Qwen/Qwen3-VL-4B-Instruct"
MODEL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
RECIPE = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.0, "presence_penalty": 1.5}

ANSWER_LINE = re.compile(r"(?im)^\W{0,3}answer\W{0,3}\s*[:：]\s*(.+?)\s*$")
LETTER = re.compile(r"^\W{0,3}([A-J])(?![A-Za-z])")
NUM = re.compile(r"-?\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?")


def clean(s):
    s = re.sub(r"\\boxed\{(.*)\}", r"\1", s)
    s = re.sub(r"\\(?:text|mathrm|mathbf)\{([^}]*)\}", r"\1", s)
    s = s.replace("\\(", "").replace("\\)", "").replace("$", "").replace("\\%", "%").replace(",", "")
    return s.strip("*` ").rstrip(".").strip()


def to_float(tok):
    if "/" in tok:
        a, b = (float(x) for x in tok.split("/"))
        return a / b if b else None
    return float(tok)


def score(text, finish, gold, kind):
    """Returns (pred, valid, correct). A sample is valid only if it finished and has a parsable final answer."""
    lines = ANSWER_LINE.findall(text)
    if finish != "stop" or not lines:
        return None, False, False
    ans = clean(lines[-1])
    if kind == "mc":
        m = LETTER.match(ans)
        if not m:
            return None, False, False
        return m.group(1), True, m.group(1) == gold.strip().upper()
    nums = NUM.findall(ans)
    if len(nums) != 1:
        return None, False, False
    v, g = to_float(nums[0].replace(" ", "")), float(gold)
    if v is None:
        return None, False, False
    return nums[0], True, abs(v - g) <= max(1e-6, 1e-3 * abs(g))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--min_pixels", type=int, default=1003520)
    ap.add_argument("--max_pixels", type=int, default=2007040)
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    from PIL import Image
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    items = [json.loads(l) for l in open(args.items) if l.strip()]
    if args.limit:
        items = items[:args.limit]
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(l)["id"] for l in open(args.out) if l.strip()}
    todo = [it for it in items if it["id"] not in done]
    print(f"{len(items)} items, {len(done)} done, {len(todo)} to run", flush=True)
    if not todo:
        return

    model_dir = snapshot_download(repo_id=MODEL_REPO, revision=MODEL_REVISION)
    tok = AutoTokenizer.from_pretrained(model_dir)
    llm = LLM(model=model_dir, dtype="bfloat16", seed=args.seed, max_model_len=8192, max_num_seqs=64,
              gpu_memory_utilization=0.90, limit_mm_per_prompt={"image": 1})
    start, n_done = time.time(), 0
    with open(args.out, "a") as fout:
        for b in range(0, len(todo), args.batch):
            chunk = todo[b:b + args.batch]
            reqs, params = [], []
            for it in chunk:
                with Image.open(local_image(it["image"])) as im:
                    img = resize_image(im, args.max_pixels, args.min_pixels)
                prompt = render(tok, [{"type": "image"}, {"type": "text", "text": it["prompt"]}])
                reqs.append({"prompt": prompt, "multi_modal_data": {"image": [img]}})
                seed = int(hashlib.sha256(f"{args.seed}:{it['id']}".encode()).hexdigest()[:8], 16)
                params.append(SamplingParams(n=args.n, max_tokens=args.max_tokens, seed=seed, **RECIPE))
            outs = llm.generate(reqs, params)
            for it, o in zip(chunk, outs):
                samples = []
                for c in o.outputs:
                    pred, valid, correct = score(c.text, c.finish_reason, it["gold"], it["kind"])
                    samples.append({"text": c.text, "finish": c.finish_reason, "n_tokens": len(c.token_ids),
                                    "pred": pred, "valid": valid, "correct": correct})
                fout.write(json.dumps({**it, "samples": samples}, ensure_ascii=False) + "\n")
            fout.flush()
            n_done += len(chunk)
            print(f"[gen {n_done}/{len(todo)}] {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
