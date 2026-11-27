# `best` preset run — 2026-09-23

One command, `bash scripts/run_mmmu_eval.sh --preset best --out outputs/best.jsonl`, on a Colab A100 40GB.
Files: `outputs/best.jsonl`, `outputs/best.fallback.jsonl`, `outputs/best.run_meta.json`, `results/best/`,
`requirements.lock.txt`. The committed baseline (`outputs/raw.jsonl`, 59.44%) is untouched.

## Settings (all recorded in `outputs/best.run_meta.json`)

| | baseline (`--preset baseline`) | best |
|---|---|---|
| sampling | temperature 0.7, top_p 0.8, top_k 20, repetition_penalty 1.0, presence_penalty 1.5 | same |
| seed | 42 | 3407 (Qwen3-VL README "Evaluation Reproduction") |
| prompt | MMMU official repo text, "Answer with the option's letter … directly." | VLMEvalKit / Qwen MMMU-script text + Qwen `--use-cot` sentence + our `Answer: X` final-line instruction |
| images | at their `<image N>` position | all before the text (Qwen script) |
| pixels | ≤ 1,003,520, no upscaling | 1,003,520 … 4,014,080 (`smart_resize`) |
| max_new_tokens | 8192 | 32768 |
| fallback | none | greedy second pass for truncated items |
| prompt hash | `01c35ec9907e6dd9` | `8aa03b32c6bba8ae` |
| wall time / peak VRAM | 2,196 s / 36.5 GB | 9,962 s / 37.6 GB |

## Result

| | baseline | best | official |
|---|---:|---:|---:|
| pipeline score (Answer-line / boxed first, fallback, then official parser) | 59.56 | **65.22** | |
| strict score (official MMMU parser only) | 59.44 | 63.00 | |
| Overall (macro = micro, 30 items per subject) | | 587 / 900 | 67.4 |
| Δ vs official, pipeline | −7.84 pp | **−2.18 pp** | |

Per-subject table: `results/best/results.md`. Side by side: `python scripts/compare_runs.py raw best`.

## What changed, with evidence

| | baseline | best |
|---|---:|---:|
| answer-only responses (≤ 60 chars) | 605 | **0** |
| reasoned and finished | 224 (80.8% correct) | 823 (68.3% correct) |
| truncated (`finish_reason=length`) | 71 (23 verbatim loops) | 77 (60 loops) |
| open questions correct | 6 / 53 | 22 / 53 |
| fallback pass | — | 77 items, 19 wrong→right, 12 right→wrong |
| explicit-answer rule vs official parser | +1 / −0 | +24 / −11 |

Item transitions baseline → best: 126 wrong→right, 75 right→wrong, 461 stay right, 238 stay wrong.
Largest gains: Pharmacy +8, Physics +7, Manage +7, Finance +6, Economics +5, Chemistry +5.
Losses: Clinical_Medicine −5, Music −4 (6 truncated), Biology −3, Diagnostics_and_Laboratory_Medicine −2.
Truncation stays concentrated in Tech & Engineering: Mechanical_Engineering 13, Architecture_and_Engineering 12,
Energy_and_Power 12, Materials 8.

## Which setting did it: 90-item A/B (3 per subject, 16384 tokens, same seed)

| | control: baseline prompt, official budget | qwen CoT (best) | forced CoT |
|---|---:|---:|---:|
| accuracy (pipeline) | 56.7 | 63.3 | 62.2 |
| answer-only responses | 62 | 0 | 0 |
| truncated / loops | 4 / 3 | 10 / 8 | 10 / 6 |
| median response chars | 24 | 2,208 | 3,474 |
| wall time | 357 s | 547 s | 721 s |

The token budget and image resolution alone changed nothing (control = the baseline run's 51/90 on the same
items). Removing "answer directly" and adding Qwen's conditional CoT sentence made every response a
reasoned one; a mandatory CoT sentence added length and time, not accuracy. Files: `outputs/ab_*.jsonl`,
`results/ab_*/`.

## Remaining gap and what it says for fine-tuning

- 77 truncated items, 60 of them verbatim loops that reach 32,768 tokens. The fallback pass recovers a
  third. Convergence, not knowledge, is the failure there; preference pairs (finished vs looping response
  to the same item) are the natural training signal.
- 823 finished reasoned responses are 68.3% correct. That 32% is the capability gap fine-tuning has to
  target; the responses now contain the reasoning needed to classify the causes.
- Sampling noise: 75 items flipped right→wrong between two runs of the same model. Per-subject differences
  under ±3 items are not evidence; compare disciplines (120–210 items) or repeat seeds.
