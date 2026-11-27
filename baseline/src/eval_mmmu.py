"""MMMU-val generation with vLLM. Writes one record per item to an append-only jsonl.

Scoring is a separate step; see docs/raw_schema.md for the output contract.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from importlib import metadata

from data import DATASET_REPO, DATASET_REVISION, MAX_IMAGES, SUBJECTS, load_subject
from prompt import (COT_MODES, IMAGE_POSITIONS, PROMPT_STYLES, build_content, build_text, fallback_suffix,
                    prompt_hash, render, trim_loop)

MODEL_REPO = "Qwen/Qwen3-VL-4B-Instruct"
MODEL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"

# Qwen3-VL-4B-Instruct model card @ MODEL_REVISION, "Generation Hyperparameters" > VL, and
# QwenLM/Qwen3-VL README "Evaluation Reproduction" > Instruct models (same six values, seed=3407,
# out_seq_length=32768). out_seq_length is replaced by --max_new_tokens (engineering choice).
SAMPLING_RECIPE = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "repetition_penalty": 1.0,
    "presence_penalty": 1.5,
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", default=MODEL_REPO,
                   help="HF repo id (downloaded at --model_revision) or a local checkpoint directory")
    p.add_argument("--model_revision", default=MODEL_REVISION)
    p.add_argument("--data_root", default=os.environ.get("HF_DATASETS_CACHE"),
                   help="HF datasets cache directory for MMMU/MMMU")
    p.add_argument("--out", default="outputs/raw.jsonl")
    p.add_argument("--subjects", default=None, help="comma-separated subset of the 30 configs")
    p.add_argument("--limit", type=int, default=None, help="first N items of each selected subject")
    p.add_argument("--max_new_tokens", type=int, default=8192)
    p.add_argument("--max_pixels", type=int, default=1280 * 28 * 28,
                   help="images with more pixels are downscaled before the processor sees them "
                        "(Qwen's MMMU script: 5120*28*28 = 4014080)")
    p.add_argument("--min_pixels", type=int, default=None,
                   help="images with fewer pixels are upscaled, qwen_vl_utils smart_resize rules "
                        "(Qwen's MMMU script: 1280*28*28 = 1003520). Default: no upscaling")
    p.add_argument("--prompt_style", choices=PROMPT_STYLES, default="mmmu",
                   help="mmmu: MMMU official repo text (baseline). vlmevalkit: Qwen/VLMEvalKit MMMU text")
    p.add_argument("--image_position", choices=IMAGE_POSITIONS, default="inline",
                   help="inline: image replaces its <image N> token (baseline). first: all images before the text")
    p.add_argument("--cot", choices=COT_MODES, default="none",
                   help="qwen: the conditional chain-of-thought sentence of Qwen's MMMU script (--use-cot default text). "
                        "forced: 'Think step by step and explain your reasoning before giving the answer.'")
    p.add_argument("--lora_path", default=None,
                   help="LoRA adapter directory applied on top of --model_path through vLLM (no merge needed). "
                        "A merged checkpoint can instead be passed as --model_path")
    p.add_argument("--max_lora_rank", type=int, default=64)
    p.add_argument("--answer_line", action="store_true",
                   help='append "End your response with a final line of the form Answer: X" to the prompt')
    p.add_argument("--fallback", action="store_true",
                   help="second greedy pass for responses that hit max_new_tokens or contain no option letter: "
                        "prompt + response (verbatim loop trimmed) + 'Therefore, the final answer is (' -> "
                        "<out stem>.fallback.jsonl. raw.jsonl is unchanged")
    p.add_argument("--fallback_max_tokens", type=int, default=16)
    p.add_argument("--seed", type=int, default=3407,
                   help="Qwen3-VL README Evaluation Reproduction value. The baseline run used 42")
    p.add_argument("--max_model_len", type=int, default=16384)
    p.add_argument("--max_num_seqs", type=int, default=32)
    p.add_argument("--batch_items", type=int, default=150,
                   help="items handed to one vLLM generate call (subjects are accumulated until this many are pending); "
                        "records are appended after each call, so a killed run keeps its finished batches")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    return p.parse_args()


def request_seed(seed, sample_id):
    """Per-request seed, so a sample's draw does not depend on batch order or restarts."""
    return int(hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()[:8], 16)


def resolve_model(model_path, revision):
    if os.path.isdir(model_path):
        return model_path, None
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=model_path, revision=revision), revision


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class VramMonitor(threading.Thread):
    """Peak device memory via nvidia-smi. vLLM runs the engine in another process, so
    torch.cuda.max_memory_allocated() here would read 0. The figure includes the KV cache
    vLLM preallocates up to --gpu_memory_utilization."""

    def __init__(self, interval=2.0):
        super().__init__(daemon=True)
        self.interval, self.peak_mib, self._stop_event = interval, 0, threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            out = run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"])
            if out and out.splitlines()[0].strip().isdigit():
                self.peak_mib = max(self.peak_mib, int(out.splitlines()[0]))
            self._stop_event.wait(self.interval)

    def stop(self):
        self._stop_event.set()


LETTER_TOKEN = re.compile(r"\(([A-I])\)|(?<![A-Za-z])([A-I])(?![A-Za-z])")


def needs_fallback(sample, completion):
    """Truncated, or a multiple-choice response in which no valid option letter appears at all."""
    if completion.finish_reason == "length":
        return True
    if sample["question_type"] != "multiple-choice":
        return False
    valid = {chr(ord("A") + i) for i in range(len(sample["options"]))}
    return not any((m.group(1) or m.group(2)) in valid for m in LETTER_TOKEN.finditer(completion.text))


def fit_fallback_prompt(tokenizer, prompt, response, suffix, max_model_len, reserve):
    """Drop text from the front of `response` until prompt+response+suffix leaves `reserve` tokens."""
    budget = max_model_len - reserve
    while True:
        text = prompt + response + suffix
        n = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        if n <= budget or len(response) < 200:
            return text, n
        response = response[max(200, len(response) // 4):]


def read_done_ids(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


def source_hash():
    """Identity of the code that ran, for machines where the repo was copied without .git."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    h = hashlib.sha256()
    for rel in ("src/data.py", "src/prompt.py", "src/eval_mmmu.py", "scripts/run_mmmu_eval.sh"):
        with open(os.path.join(root, rel), "rb") as f:
            h.update(rel.encode() + b"\x00" + f.read())
    return h.hexdigest()[:16]


def versions():
    out = {}
    for pkg in ("vllm", "torch", "transformers", "datasets", "huggingface_hub", "pillow"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def main():
    args = parse_args()
    subjects = args.subjects.split(",") if args.subjects else SUBJECTS
    unknown = [s for s in subjects if s not in SUBJECTS]
    if unknown:
        sys.exit(f"unknown subjects: {unknown}")
    full_run = args.subjects is None and args.limit is None

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    start = time.time()
    monitor = VramMonitor()
    monitor.start()

    model_dir, model_revision = resolve_model(args.model_path, args.model_revision)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    lora_kwargs, lora_request = {}, None
    if args.lora_path:
        from vllm.lora.request import LoRARequest
        lora_kwargs = {"enable_lora": True, "max_lora_rank": args.max_lora_rank}
        lora_request = LoRARequest("adapter", 1, os.path.abspath(args.lora_path))
    llm = LLM(
        model=model_dir,
        dtype="bfloat16",
        seed=args.seed,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": MAX_IMAGES},
        **lora_kwargs,
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = read_done_ids(args.out)
    resumed = bool(done)
    generated, finish_counts, warnings = 0, {}, []
    fb_counts = {"attempted": 0, "trimmed_loops": 0, "finish": {}}
    stem = os.path.splitext(os.path.basename(args.out))[0]
    fb_path = os.path.join(os.path.dirname(args.out) or ".", f"{stem}.fallback.jsonl")
    fb_out = open(fb_path, "a") if args.fallback else None

    def flush(pending, fout):
        """Build prompts, generate (one vLLM call), run the fallback pass, append records in subject order."""
        nonlocal generated
        requests, params = [], []
        for s in pending:
            content, images, warns = build_content(build_text(s, args.prompt_style, args.answer_line, args.cot),
                                                   s["images"], args.max_pixels, args.min_pixels,
                                                   args.image_position)
            warnings.extend(f"{s['id']}: {w}" for w in warns)
            s["prompt"], s["num_images"], s["_images"] = render(tokenizer, content), len(images), images
            request = {"prompt": s["prompt"]}
            if images:
                request["multi_modal_data"] = {"image": images}
            requests.append(request)
            params.append(SamplingParams(max_tokens=args.max_new_tokens,
                                         seed=request_seed(args.seed, s["id"]), **SAMPLING_RECIPE))
        outputs = llm.generate(requests, params, lora_request=lora_request)

        if fb_out is not None:
            fb_requests, fb_records = [], []
            for s, o in zip(pending, outputs):
                c = o.outputs[0]
                if not needs_fallback(s, c):
                    continue
                trimmed, removed = trim_loop(c.text)
                fb_counts["trimmed_loops"] += removed > 0
                text, n_tokens = fit_fallback_prompt(tokenizer, s["prompt"], trimmed, fallback_suffix(s),
                                                     args.max_model_len, args.fallback_max_tokens + 8)
                req = {"prompt": text}
                if s["_images"]:
                    req["multi_modal_data"] = {"image": s["_images"]}
                fb_requests.append(req)
                fb_records.append({"id": s["id"], "trigger": c.finish_reason if c.finish_reason == "length" else "no_letter",
                                   "loop_chars_removed": removed,
                                   "response_chars_used": len(text) - len(s["prompt"]) - len(fallback_suffix(s)),
                                   "fallback_prompt_tokens": n_tokens, "fallback_suffix": fallback_suffix(s)})
            if fb_requests:
                fb_params = SamplingParams(max_tokens=args.fallback_max_tokens, temperature=0.0, seed=args.seed)
                for rec, fo in zip(fb_records, llm.generate(fb_requests, fb_params, lora_request=lora_request)):
                    fc = fo.outputs[0]
                    rec["fallback_response"] = fc.text
                    rec["fallback_finish_reason"] = fc.finish_reason
                    fb_counts["attempted"] += 1
                    fb_counts["finish"][fc.finish_reason] = fb_counts["finish"].get(fc.finish_reason, 0) + 1
                    fb_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fb_out.flush()

        per_subject = {}
        for s, o in zip(pending, outputs):
            completion = o.outputs[0]
            finish_counts[completion.finish_reason] = finish_counts.get(completion.finish_reason, 0) + 1
            fout.write(json.dumps({
                "id": s["id"],
                "subject": s["subject"],
                "question_type": s["question_type"],
                "options": s["options"],
                "answer": s["answer"],
                "prompt": s["prompt"],
                "response": completion.text,
                "finish_reason": completion.finish_reason,
                "num_images": s["num_images"],
            }, ensure_ascii=False) + "\n")
            per_subject[s["subject"]] = per_subject.get(s["subject"], 0) + 1
        fout.flush()
        generated += len(pending)
        for subject, n in per_subject.items():
            print(f"[{subject}] +{n} ({generated} this run, {time.time() - start:.0f}s)", flush=True)

    with open(args.out, "a") as fout:
        pending = []
        for i, subject in enumerate(subjects):
            samples = load_subject(subject, args.data_root)
            if args.limit is not None:
                samples = samples[:args.limit]
            pending += [s for s in samples if s["id"] not in done]
            last = i == len(subjects) - 1
            if pending and (len(pending) >= args.batch_items or last):
                flush(pending, fout)
                pending = []

    monitor.stop()
    if fb_out is not None:
        fb_out.close()
    total = len(read_done_ids(args.out))
    meta = {
        "git_commit": run(["git", "rev-parse", "HEAD"]),
        "git_dirty": None if run(["git", "rev-parse", "HEAD"]) is None else bool(run(["git", "status", "--porcelain"])),
        "source_hash": source_hash(),
        "model_path": args.model_path,
        "model_revision": model_revision,
        "lora_path": args.lora_path,
        "dataset": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "dtype": "bfloat16",
        "backend": "vllm",
        "versions": versions(),
        "python": sys.version.split()[0],
        "gpu": run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
        "sampling": {**SAMPLING_RECIPE, "seed": args.seed,
                     "per_request_seed": "int(sha256(f'{seed}:{id}').hexdigest()[:8], 16)"},
        "max_new_tokens": args.max_new_tokens,
        "max_pixels": args.max_pixels,
        "min_pixels": args.min_pixels if args.min_pixels is not None else "processor default",
        "prompt_style": args.prompt_style,
        "image_position": args.image_position,
        "answer_line": args.answer_line,
        "cot": args.cot,
        "fallback": {"enabled": args.fallback, "max_tokens": args.fallback_max_tokens, "file": fb_path if args.fallback else None,
                     "trigger": "finish_reason=length, or multiple-choice with no valid option letter in the response",
                     "decoding": "greedy (temperature 0), same images, response with verbatim loop trimmed", **fb_counts},
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "batch_items": args.batch_items,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "prompt_hash": prompt_hash(args.prompt_style, args.image_position, args.answer_line, args.cot),
        "subjects": subjects,
        "limit": args.limit,
        "resumed": resumed,
        "generated_this_run": generated,
        "records_in_out": total,
        "finish_reason_counts_this_run": finish_counts,
        "wall_time_s_this_run": round(time.time() - start, 1),
        "peak_vram_mib": monitor.peak_mib,
        "peak_vram_method": "max of nvidia-smi memory.used (GPU 0) sampled every 2s; includes vLLM's preallocated KV cache",
        "prompt_warnings": warnings,
    }
    # outputs/raw.jsonl -> outputs/run_meta.json; any other --out keeps its own meta file.
    meta_name = "run_meta.json" if stem == "raw" else f"{stem}.run_meta.json"
    meta_path = os.path.join(os.path.dirname(args.out) or ".", meta_name)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"{total} records in {args.out}; meta in {meta_path}")

    if full_run and total != 30 * len(SUBJECTS):
        sys.exit(f"full run expected 900 records, found {total}")


if __name__ == "__main__":
    main()
