"""Experiment S, step 1: the teacher writes one university-level multiple-choice question per pool image.

  python step16_s_write_questions.py --pool ../synth/pool.jsonl --out ../synth/items.jsonl

For each image (step15_s_collect_images.py) the teacher gets the image, the subjects the image can serve and the dataset's own
text as background, and returns JSON {subject, question, options[4], answer, explanation}. The options are then
shuffled with a per-id seed, so the gold letter does not follow the teacher's habit of where it puts the answer.
Output rows have the step02/step13 item format ({"id", "image", "prompt", "gold", "kind": "mc", "category", ...}),
with the MMMU eval prompt, so the next step is the blind check:
  python step13_t_teacher_solutions.py --items ../synth/items.jsonl --out ../synth/teacher.jsonl
which keeps a question only if the teacher, without seeing the answer, reaches the same gold letter.
A rerun skips ids that are already in --out or in <out>.rejected.jsonl.
"""
import argparse
import json
import os
import random
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "MMD", "src"))

TEACHER = "Qwen/Qwen3-VL-32B-Instruct"
INSTRUCTION = """You are writing an exam question for a university course. The course subject is one of: {subjects}.

Background text that came with the image (it may be simple; use it only to understand the image):
{context}

Write ONE multiple-choice question about the image. Requirements:
- It cannot be answered without looking at the image.
- It needs university-level knowledge of the subject or several reasoning steps, not just reading one label or number.
- Exactly 4 options. Exactly one is correct. The wrong options must be plausible and of similar length.
- The question must stand alone: do not mention the background text, and do not reveal the answer.
- If the answer is a number, compute it carefully and make the wrong options close to it.

Reply with JSON only, in this form:
{{"subject": "<one subject from the list>", "question": "...", "options": ["...", "...", "...", "..."], "answer": "<the correct option's text, copied exactly>", "explanation": "<2-4 sentences>"}}"""


def parse(text, subjects):
    """Teacher reply -> dict(subject, question, options, answer_index, explanation), or (None, reason)."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None, "no_json"
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, "bad_json"
    q, opts, ans = d.get("question"), d.get("options"), d.get("answer")
    if not isinstance(q, str) or len(q.strip()) < 15:
        return None, "no_question"
    if not isinstance(opts, list) or len(opts) != 4 or not all(isinstance(o, (str, int, float)) for o in opts):
        return None, "not_4_options"
    opts = [re.sub(r"^\(?[A-D][).:]\s+", "", str(o)).strip() for o in opts]
    if len({o.lower() for o in opts}) != 4 or not all(opts):
        return None, "duplicate_options"
    ans = str(ans).strip()
    if re.fullmatch(r"[A-D]", ans):
        idx = ord(ans) - 65
    else:
        a = re.sub(r"^\(?[A-D][).:]\s+", "", ans).strip().lower()
        hits = [i for i, o in enumerate(opts) if o.lower() == a]
        if len(hits) != 1:
            return None, "answer_not_an_option"
        idx = hits[0]
    subj = d.get("subject") if d.get("subject") in subjects else subjects[0]
    return {"subject": subj, "question": q.strip(), "options": opts, "answer_index": idx,
            "explanation": str(d.get("explanation", "")).strip()}, None


def to_item(row, p):
    from prompt import build_text
    order = list(range(4))
    random.Random(f"shuffle:{row['id']}").shuffle(order)
    options = [p["options"][i] for i in order]
    gold = chr(65 + order.index(p["answer_index"]))
    sample = {"question_type": "multiple-choice", "question": p["question"], "options": options}
    return {"id": "S-" + row["id"], "image": row["image"], "question": p["question"], "options": options,
            "gold": gold, "kind": "mc", "category": p["subject"], "source": "synth:" + row["source"],
            "gen_explanation": p["explanation"], "prompt": build_text(sample, "vlmevalkit", True, "qwen")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=TEACHER)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max_tokens", type=int, default=500)
    ap.add_argument("--min_pixels", type=int, default=1003520)
    ap.add_argument("--max_pixels", type=int, default=2007040)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--fake", default=None, help="test only: a file with one teacher reply, used for every image")
    args = ap.parse_args()

    pool = [json.loads(l) for l in open(args.pool, encoding="utf-8") if l.strip()]
    if args.limit:
        pool = pool[:args.limit]
    rej_path = args.out + ".rejected.jsonl"
    done = set()
    for p in (args.out, rej_path):
        if os.path.exists(p):
            done |= {json.loads(l)["id"].removeprefix("S-") for l in open(p, encoding="utf-8") if l.strip()}
    todo = [r for r in pool if r["id"] not in done]
    print(f"{len(pool)} images, {len(done)} done, {len(todo)} to run", flush=True)
    if not todo:
        return

    if args.fake:
        reply = open(args.fake, encoding="utf-8").read()

        def run(chunk):
            return [reply for _ in chunk]
    else:
        import torch
        from PIL import Image
        from prompt import resize_image
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        processor = AutoProcessor.from_pretrained(args.model)
        processor.tokenizer.padding_side = "left"
        processor.image_processor.size = {"shortest_edge": 32 * 32, "longest_edge": args.max_pixels + 32 * 32 * 64}
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
        torch.manual_seed(args.seed)
        print(f"loaded {args.model}; GPU memory {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)

        def run_once(chunk):
            texts, images = [], []
            for r in chunk:
                with Image.open(r["image"]) as im:
                    images.append(resize_image(im, args.max_pixels, args.min_pixels))
                ask = INSTRUCTION.format(subjects=", ".join(s.replace("_", " ") for s in r["subjects"]),
                                         context=r["context"] or "(none)")
                msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": ask}]}]
                texts.append(processor.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            batch = processor(text=texts, images=images, padding=True, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model.generate(**batch, max_new_tokens=args.max_tokens, do_sample=True, temperature=0.7,
                                     top_p=0.8, top_k=20, repetition_penalty=1.0)
            return processor.tokenizer.batch_decode(out[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)

        def run(chunk):
            try:
                return run_once(chunk)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if len(chunk) == 1:
                    return [""]
                h = len(chunk) // 2
                return run(chunk[:h]) + run(chunk[h:])

    start, n_ok, why = time.time(), 0, {}
    with open(args.out, "a", encoding="utf-8") as fout, open(rej_path, "a", encoding="utf-8") as frej:
        for b in range(0, len(todo), args.batch):
            chunk = todo[b:b + args.batch]
            for r, text in zip(chunk, run(chunk)):
                subjects = [s.replace("_", " ") for s in r["subjects"]]
                p, reason = parse(text, subjects)
                if p is None:
                    why[reason] = why.get(reason, 0) + 1
                    frej.write(json.dumps({"id": r["id"], "reason": reason, "text": text[:1500]}, ensure_ascii=False) + "\n")
                    continue
                p["subject"] = p["subject"].replace(" ", "_")
                fout.write(json.dumps(to_item(r, p), ensure_ascii=False) + "\n")
                n_ok += 1
            fout.flush(); frej.flush()
            n_run = min(b + args.batch, len(todo))
            if (b // args.batch) % 5 == 4 or n_run == len(todo):
                el = time.time() - start
                print(f"[synth {n_run}/{len(todo)}] {el:.0f}s, {el / n_run:.1f} s/item | usable {n_ok} | rejected {why}",
                      flush=True)
    print(f"FINAL usable {n_ok}/{len(todo)} | rejected {why}", flush=True)


if __name__ == "__main__":
    main()
