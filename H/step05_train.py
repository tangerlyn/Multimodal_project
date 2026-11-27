"""Experiment H: SFT vs DPO on the same self-generated rationales. ViT frozen, LLM attention LoRA only.

  python step05_train.py --mode sft --data build/pairs_train.jsonl --val build/pairs_val.jsonl --out runs/H-SFT
  python step05_train.py --mode dpo --data build/pairs_train.jsonl --val build/pairs_val.jsonl --out runs/H-DPO

Each row: {"id", "image", "prompt" (user text, image goes first), "pos", "neg"}.
  sft : cross-entropy on the distinct "pos" responses (rejection-sampling fine-tuning).
  dpo : -log sigmoid(beta * [(log pi(pos) - log ref(pos)) - (log pi(neg) - log ref(neg))]), where ref is the
        same model with the adapter disabled (LoRA B starts at zero, so ref == base model).
Both modes see the same questions, images, prompt format and number of epochs. Writes <out>/lora (PEFT adapter),
log.jsonl and training_run.json. Merge with step06_merge_model.py, then evaluate with run_mmmu_eval.sh --model_path.
"""
import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "MMD", "src"))
from prompt import resize_image  # noqa: E402


def local_image(path):
    """Image paths in the data files were written on Colab (/content/virl/...). Map them to $VIRL_DIR
    (default: <repo>/data/virl) when the original path does not exist."""
    if os.path.exists(path):
        return path
    root = os.environ.get("VIRL_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "virl")
    tail = path.split("/virl/", 1)[1] if "/virl/" in path else os.path.basename(path)
    return os.path.join(root, tail)

BASE_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
BASE_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
TARGETS = {"attn": r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj)$",
           "attn+mlp": r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"}
CONNECTOR_KEYS = ("visual.merger.", "visual.deepstack_merger_list.")   # same as step06_merge_model.py
AUTOCAST = False   # set when the connector is trained: its weights are float32, the rest of the model bf16
EOT = "<|im_end|>"


def load(path, limit=None):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return rows[:limit] if limit else rows


class Encoder:
    """prompt (+ response) -> model inputs, with the same image resize as the MMMU eval (prompt.resize_image)."""

    def __init__(self, processor, min_pixels, max_pixels):
        self.p, self.min_pixels, self.max_pixels = processor, min_pixels, max_pixels
        # The image is already resized to a multiple of 32 inside [min, max]; keep the processor from changing it.
        processor.image_processor.size = {"shortest_edge": 32 * 32, "longest_edge": max_pixels + 32 * 32 * 64}
        self.cache = {}

    def image(self, path):
        with Image.open(local_image(path)) as im:
            return resize_image(im, self.max_pixels, self.min_pixels)

    def prompt(self, text):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        return self.p.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def __call__(self, row, response):
        img = self.image(row["image"])
        prompt = self.prompt(row["prompt"])
        full = self.p(text=[prompt + response + EOT], images=[img], return_tensors="pt")
        n_prompt = self.p(text=[prompt], images=[img], return_tensors="pt")["input_ids"].shape[1]
        labels = full["input_ids"].clone()
        labels[:, :n_prompt] = -100
        return {k: v.cuda() for k, v in full.items()}, labels.cuda()


def response_logprob(model, batch, labels):
    """Sum of token log-probs of the response part. Returns (sum_logp, n_tokens)."""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AUTOCAST):
        logits = model(**batch).logits[:, :-1].float()
    tgt = labels[:, 1:]
    mask = tgt != -100
    lp = -F.cross_entropy(logits[mask], tgt[mask], reduction="sum")
    return lp, int(mask.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["sft", "dpo"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--val", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--targets", choices=sorted(TARGETS), default="attn", help="LLM layers that get LoRA")
    ap.add_argument("--connector", action="store_true",
                    help="also train the merger + DeepStack mergers (full weights, sft only); saved as connector.pt")
    ap.add_argument("--connector-lr", type=float, default=1e-5)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--min-pixels", type=int, default=1003520)
    ap.add_argument("--max-pixels", type=int, default=2007040)
    ap.add_argument("--max-steps", type=int, default=None, help="smoke test: stop after N examples")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if args.connector and args.mode != "sft":
        sys.exit("--connector is for --mode sft only (the DPO reference would not see the original connector)")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    if os.path.exists(os.path.join(args.out, "training_run.json")):
        sys.exit(f"already finished: {args.out}")

    rows = load(args.data, args.limit)
    val = load(args.val, args.limit) if args.val else []
    if args.mode == "sft":   # distinct positives only
        seen, uniq = set(), []
        for r in rows:
            if (r["id"], r["pos"]) not in seen:
                seen.add((r["id"], r["pos"]))
                uniq.append(r)
        rows = uniq
    print(f"[H-{args.mode}] train examples {len(rows)}  val pairs {len(val)}", flush=True)

    processor = AutoProcessor.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0, bias="none",
                                             target_modules=TARGETS[args.targets], task_type="CAUSAL_LM"))
    # Connector: PEFT names are "base_model.model.<plain name>"; step06_merge_model.py wants plain names.
    conn, conn_init = {}, {}
    if args.connector:
        global AUTOCAST
        AUTOCAST = True
        for n, p in model.named_parameters():
            if any(k in n for k in CONNECTOR_KEYS):
                name = n.replace("base_model.model.", "", 1)
                conn_init[name] = p.detach().cpu().clone()          # bf16, to tell which tensors really moved
                p.data = p.data.float()                              # bf16 cannot hold steps of size 1e-5
                p.requires_grad_(True)
                conn[name] = p
        if not conn:
            sys.exit("no connector parameters found")
    conn_ids = {id(p) for p in conn.values()}
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    lora = [(n, p) for n, p in trainable if id(p) not in conn_ids]
    bad = [n for n, _ in lora if "lora_" not in n or "language_model" not in n or "visual" in n]
    bad += [n for n in conn if "visual." not in n or not any(k in n for k in CONNECTOR_KEYS)]
    if not lora or bad:
        sys.exit(f"unexpected trainable params: {bad[:5]}")
    n_params = sum(p.numel() for _, p in trainable)
    n_conn = sum(p.numel() for p in conn.values())
    print(f"LoRA r={args.rank} {args.targets}, {len(lora) // 2} layers, {n_params - n_conn:,} params; "
          + (f"connector {len(conn)} tensors, {n_conn:,} params, lr {args.connector_lr}; ViT frozen" if conn
             else "ViT + connector frozen"), flush=True)

    def save_trained():
        model.save_pretrained(os.path.join(args.out, "lora"))
        if conn:   # only tensors whose bf16 value changed: step06_merge_model.py checks that every listed one differs
            sd = {k: p.detach().to(torch.bfloat16).cpu() for k, p in conn.items()}
            sd = {k: v for k, v in sd.items() if not torch.equal(v, conn_init[k])}
            torch.save(sd, os.path.join(args.out, "connector.pt"))
            return len(sd)

    enc = Encoder(processor, args.min_pixels, args.max_pixels)
    groups = [{"params": [p for _, p in lora], "lr": args.lr}]
    if conn:
        groups.append({"params": list(conn.values()), "lr": args.connector_lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    total = int(len(rows) * args.epochs) if args.max_steps is None else min(args.max_steps, len(rows))
    n_updates = max(1, math.ceil(total / args.grad_accum))
    warm = max(1, int(0.05 * n_updates))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, n_updates - warm))))

    def ref_logprob(row, text):
        with torch.no_grad(), model.disable_adapter():
            b, l = enc(row, text)
            return response_logprob(model, b, l)[0].item()

    def dpo_terms(row):
        bp, lp_ = enc(row, row["pos"])
        pos, _ = response_logprob(model, bp, lp_)
        bn, ln_ = enc(row, row["neg"])
        neg, _ = response_logprob(model, bn, ln_)
        margin = (pos - row["_ref_pos"]) - (neg - row["_ref_neg"])
        return -F.logsigmoid(args.beta * margin), margin

    @torch.no_grad()
    def evaluate():
        """Val pairs: mean SFT loss per token on pos, and how often log pi(pos)-log pi(neg) beats the reference gap."""
        if not val:
            return {}
        model.eval()
        nll = ntok = wins = 0
        gaps = []
        for r in val:
            b, l = enc(r, r["pos"]); p, n = response_logprob(model, b, l)
            nll -= p.item(); ntok += n
            if "neg" not in r:   # SFT-only data (step14_t_build_sft.py): no preference margin to report
                continue
            b, l = enc(r, r["neg"]); q, _ = response_logprob(model, b, l)
            if "_ref_pos" not in r:
                r["_ref_pos"], r["_ref_neg"] = ref_logprob(r, r["pos"]), ref_logprob(r, r["neg"])
            g = (p.item() - r["_ref_pos"]) - (q.item() - r["_ref_neg"])
            gaps.append(g); wins += g > 0
        model.train()
        if not gaps:
            return {"val_pos_nll_per_tok": nll / ntok}
        return {"val_pos_nll_per_tok": nll / ntok, "val_margin_mean": sum(gaps) / len(gaps), "val_margin_win": wins / len(gaps)}

    log = open(os.path.join(args.out, "log.jsonl"), "a")
    order = []
    while len(order) < total:
        ep = list(range(len(rows))); random.shuffle(ep); order += ep
    order = order[:total]

    model.train()
    start, run_loss, run_n, step = time.time(), 0.0, 0, 0
    m0 = evaluate()
    print(f"before training: {m0}", flush=True)
    log.write(json.dumps({"step": 0, **m0}) + "\n"); log.flush()
    opt.zero_grad(set_to_none=True)
    for i, idx in enumerate(order, 1):
        r = rows[idx]
        try:
            if args.mode == "sft":
                b, l = enc(r, r["pos"])
                lp, n = response_logprob(model, b, l)
                loss = -lp / n
            else:
                if "_ref_pos" not in r:
                    r["_ref_pos"], r["_ref_neg"] = ref_logprob(r, r["pos"]), ref_logprob(r, r["neg"])
                loss, _ = dpo_terms(r)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss")
            (loss / args.grad_accum).backward()
            run_loss += loss.item(); run_n += 1
        except torch.cuda.OutOfMemoryError:
            print(f"OOM on {r['id']}, skipped", flush=True)
            opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
            continue
        if i % args.grad_accum == 0 or i == total:
            torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            if step % 10 == 0 or i == total:
                rec = {"step": step, "examples": i, "loss": run_loss / max(run_n, 1), "lr": sched.get_last_lr()[0],
                       "elapsed_s": round(time.time() - start)}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n"); log.flush()
                run_loss, run_n = 0.0, 0
            if step % 100 == 0:
                save_trained()
    m1 = evaluate()
    print(f"after training: {m1}", flush=True)
    log.write(json.dumps({"step": step, **m1}) + "\n"); log.close()
    n_moved = save_trained()
    if conn:
        print(f"connector: {n_moved}/{len(conn)} tensors changed at bf16 precision", flush=True)
    json.dump({"condition": f"H-{args.mode.upper()}", "mode": args.mode, "data": args.data, "n_train_examples": len(rows),
               "examples_seen": total, "optimizer_steps": step, "lr": args.lr, "beta": args.beta if args.mode == "dpo" else None,
               "rank": args.rank, "alpha": args.alpha, "targets": args.targets, "trainable_parameters": n_params,
               "connector_parameters": n_conn, "connector_lr": args.connector_lr if conn else None,
               "min_pixels": args.min_pixels, "max_pixels": args.max_pixels, "grad_accum": args.grad_accum,
               "seed": args.seed, "before": m0, "after": m1, "wall_time_s": round(time.time() - start)},
              open(os.path.join(args.out, "training_run.json"), "w"), indent=2)
    print(f"saved {args.out}/lora", flush=True)


if __name__ == "__main__":
    main()
