"""Experiment T, step 1: a larger teacher model writes one short solution per question (Transformers backend).

  python step13_t_teacher_solutions.py --items build/gen.jsonl --out buildT/teacher.jsonl --limit 300

items rows: {"id", "image", "prompt", "gold", "kind": "mc"|"num", ...}; a "samples" list (the 4B model's own
answers from step02_sample_answers.py) is used only to report the student's pass rate on the same questions.
The teacher sees the student's eval prompt plus one sentence asking for a short solution, decodes greedily, and
every answer is scored with the same rule as step02_sample_answers.py. Appends one row per question; a rerun skips finished ids.
Layout: this file in <root>/H, the MMMU evaluator in <root>/MMD (as on the lab server).
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "MMD", "src"))
sys.path.insert(0, HERE)

TEACHER = "Qwen/Qwen3-VL-32B-Instruct"
CONCISE = " Keep the solution short: only the key steps, at most about 200 words, then the final answer line."


def student_pass(item):
    s = item.get("samples") or []
    return sum(x["correct"] for x in s) / len(s) if s else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=TEACHER)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--min_pixels", type=int, default=1003520)
    ap.add_argument("--max_pixels", type=int, default=2007040)
    ap.add_argument("--dry_run", action="store_true", help="check the data and the scorer without loading a model")
    args = ap.parse_args()

    from PIL import Image
    from step02_sample_answers import score
    from prompt import resize_image

    items = [json.loads(l) for l in open(args.items, encoding="utf-8") if l.strip()]
    if args.limit:
        items = items[:args.limit]
    done = {}
    if os.path.exists(args.out):
        for l in open(args.out, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                done[r["id"]] = r
    todo = [it for it in items if it["id"] not in done]
    print(f"{len(items)} items, {len(done)} done, {len(todo)} to run", flush=True)

    def summary(rows):
        if not rows:
            return "no rows"
        t = [r["teacher"] for r in rows]
        sp = [r["student_pass"] for r in rows if r["student_pass"] is not None]
        ok = [x for x in t if x["correct"]]
        toks = sorted(x["n_tokens"] for x in ok) or [0]
        return (f"n {len(rows)} | teacher correct {len(ok) / len(rows):.3f} | truncated "
                f"{sum(x['finish'] != 'stop' for x in t) / len(rows):.3f} | student pass "
                f"{(sum(sp) / len(sp) if sp else float('nan')):.3f} | tokens of correct answers: median "
                f"{toks[len(toks) // 2]}, max {toks[-1]}")

    if args.dry_run:
        missing = [it["id"] for it in todo if not os.path.exists(it["image"])]
        print(f"missing images: {len(missing)} {missing[:3]}")
        for it in todo[:3]:
            print(repr((it["prompt"] + CONCISE)[-260:]))
            s = (it.get("samples") or [{}])[0]
            if s:
                print("scorer on the first student sample:", score(s["text"], s["finish"], it["gold"], it["kind"]))
        print(summary(list(done.values())))
        return
    if not todo:
        print(summary(list(done.values())))
        return

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.model)
    processor.tokenizer.padding_side = "left"
    # Images are resized here to a multiple of 32 inside [min, max]; keep the processor from changing them.
    processor.image_processor.size = {"shortest_edge": 32 * 32, "longest_edge": args.max_pixels + 32 * 32 * 64}
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    eos = {processor.tokenizer.convert_tokens_to_ids("<|im_end|>"), processor.tokenizer.eos_token_id}
    pad = processor.tokenizer.pad_token_id
    print(f"loaded {args.model}; GPU memory {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)

    def run(chunk):
        texts, images = [], []
        for it in chunk:
            with Image.open(it["image"]) as im:
                images.append(resize_image(im, args.max_pixels, args.min_pixels))
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": it["prompt"] + CONCISE}]}]
            texts.append(processor.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        batch = processor(text=texts, images=images, padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**batch, max_new_tokens=args.max_tokens, do_sample=False, repetition_penalty=1.0)
        res = []
        for seq in out[:, batch["input_ids"].shape[1]:].tolist():
            cut = next((i for i, t in enumerate(seq) if t in eos or t == pad), None)
            finish = "stop" if cut is not None else "length"
            ids = seq if cut is None else seq[:cut]
            res.append((processor.tokenizer.decode(ids, skip_special_tokens=True).strip(), finish, len(ids)))
        return res

    def run_safe(chunk):
        try:
            return run(chunk)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(chunk) == 1:
                print(f"OOM on {chunk[0]['id']}, skipped", flush=True)
                return [None]
            h = len(chunk) // 2
            return run_safe(chunk[:h]) + run_safe(chunk[h:])

    start, rows = time.time(), list(done.values())
    with open(args.out, "a", encoding="utf-8") as fout:
        for b in range(0, len(todo), args.batch):
            chunk = todo[b:b + args.batch]
            for it, r in zip(chunk, run_safe(chunk)):
                if r is None:
                    continue
                text, finish, n = r
                pred, valid, correct = score(text, finish, it["gold"], it["kind"])
                row = {k: v for k, v in it.items() if k != "samples"}
                row["student_pass"] = student_pass(it)
                row["teacher"] = {"model": args.model, "text": text, "finish": finish, "n_tokens": n,
                                  "pred": pred, "valid": valid, "correct": correct}
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows.append(row)
            fout.flush()
            n_run = min(b + args.batch, len(todo))
            if (b // args.batch) % 5 == 4 or n_run == len(todo):
                el = time.time() - start
                print(f"[teacher {n_run}/{len(todo)}] {el:.0f}s, {el / n_run:.1f} s/item | {summary(rows)}", flush=True)
    print("FINAL " + summary(rows), flush=True)


if __name__ == "__main__":
    main()
