# MMD — MMMU-val evaluation pipeline for Qwen3-VL-4B-Instruct

One command runs generation, schema validation, scoring and the report tables. The raw model
responses (`outputs/<name>.jsonl`) follow [docs/raw_schema.md](docs/raw_schema.md); scoring is
`scripts/score_mmmu.py` (official MMMU parser, pinned). A fine-tuned model is evaluated by changing
only `--model_path` (merged checkpoint) or `--lora_path` (adapter).

```bash
bash scripts/run_mmmu_eval.sh --preset best
```

| | |
|---|---|
| Model | `Qwen/Qwen3-VL-4B-Instruct` @ `ebb281ec70b05090aa6165b016eac8ec08e71b17`, bf16 |
| Data | `MMMU/MMMU` @ `98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68`, `validation`, 30 configs, 900 items |
| Backend | vLLM offline API (needed for `presence_penalty`; batched generation) |
| Sampling | model card "Generation Hyperparameters" (VL) = Qwen3-VL README "Evaluation Reproduction" (Instruct): `temperature=0.7 top_p=0.8 top_k=20 repetition_penalty=1.0 presence_penalty=1.5`. Seed: 42 in the committed baseline, 3407 (README value) is now the default |
| Budget | `max_new_tokens=8192`, `max_pixels=1003520` (1280×28×28), `min_pixels` processor default |
| Prompt | MMMU official repo format, see [src/prompt.py](src/prompt.py) |

## Run

```bash
pip install -r requirements.txt
```

```bash
bash scripts/run_mmmu_eval.sh --preset best --model_path Qwen/Qwen3-VL-4B-Instruct --data_root "$HF_DATASETS_CACHE"
```

### Presets

| Preset | Settings | Why |
|---|---|---|
| `baseline` | seed 42, `max_new_tokens` 8192, MMMU-repo prompt, inline images, `max_pixels` 1003520, no upscaling | the committed first run (`outputs/raw.jsonl`, 59.44%) |
| `official` | seed 3407, 32768 tokens, `min/max_pixels` 1003520/4014080, VLMEvalKit prompt, images first | Qwen's own MMMU script (`QwenLM/Qwen3-VL` `evaluation/mmmu/run_mmmu.py`) and README "Evaluation Reproduction" |
| `best` | `official` + `--cot qwen` + `--answer_line` + `--fallback` | **measured 65.22% (587/900), strict 63.00%; see [docs/best_run.md](docs/best_run.md)**. Chosen because the baseline showed 80% accuracy when the model reasons and finishes vs 56% when it answers directly and 27% when it is cut off, so the extras push for reasoning, give it a termination target, and recover an answer when it still runs out |

Flags given after `--preset` override the preset. Every setting is recorded in `<out stem>.run_meta.json`.
The sampling recipe (`temperature 0.7, top_p 0.8, top_k 20, repetition_penalty 1.0, presence_penalty 1.5`)
is the same in every preset and is not a flag.

### What one run produces

| File | Contents |
|---|---|
| `outputs/<name>.jsonl` | one raw record per item (`docs/raw_schema.md`) |
| `outputs/<name>.fallback.jsonl` | with `--fallback`: second-pass continuations for truncated items |
| `outputs/<name>.run_meta.json` (`run_meta.json` for `raw`) | versions, GPU, every effective setting, prompt hash, timing, peak VRAM |
| `results/<name>/results.md` | the report tables: 30 subjects + macro average (with the formula), comparison to 67.4, response-group and discipline breakdowns, parser statistics |
| `results/<name>/by_subject.csv`, `summary.json`, `scored.jsonl` | the same numbers as data, and one scored line per item |

### Scoring (`scripts/score_mmmu.py`)

Parser: MMMU official `mmmu/utils/eval_utils.py` at commit `268471d`, vendored byte-identical as
`scripts/mmmu_official_eval_utils.py` (SHA256 checked at load), with one change made in memory: a parse
failure counts as wrong instead of a random guess. Two scores are always reported from the same responses:

- **strict**: the official parser only.
- **pipeline**: before the official rules, (1) for items in the fallback file, the second-pass letter
  replaces the response; (2) the last `Answer: X` line, else the last `\boxed{X}`, is taken as the answer
  (open questions: the last `Answer:` line is parsed instead of the whole text). Everything else is the
  official parser. On the baseline this changes one item (`validation_Math_30`, a correct boxed C that the
  last-bracket rule mis-read as E).

`results.md` states which score each table uses. Overall = mean of the 30 subject accuracies; with 30
items per subject this equals correct/900, and the file prints both.

`--model_path` is an HF repo id (downloaded at the pinned revision) or a local checkpoint
directory, which is how a fine-tuned model is re-evaluated with identical settings.
`--data_root` is the `datasets` cache directory; raw hub downloads follow `HF_HOME`.
The script runs generation, then `scripts/check_raw.py` on the output, and exits non-zero if a
full run does not contain exactly 900 valid records.

| Flag | Default | |
|---|---|---|
| `--out` | `outputs/raw.jsonl` | restart with the same `--out` skips ids already written |
| `--subjects` | all 30 | comma-separated config names |
| `--limit` | none | first N items of each selected subject |
| `--max_new_tokens` | 8192 | at 1024, 18% of responses were cut off (see report §3.2) |
| `--max_pixels` | 1003520 | larger images are downscaled (bicubic, aspect kept) before the processor |
| `--min_pixels` | none | with a value, images are brought into `[min_pixels, max_pixels]` by `qwen_vl_utils` `smart_resize` rules (sides rounded to multiples of 32); Qwen's MMMU script uses 1003520 |
| `--prompt_style` | `mmmu` | `mmmu`: MMMU official repo text (baseline). `vlmevalkit`: the `Question: / Options: / Please select…` text of Qwen's MMMU script and VLMEvalKit |
| `--image_position` | `inline` | `inline`: image replaces its `<image N>` token (baseline). `first`: all images before the text, tokens kept literal, as Qwen's script does |
| `--cot` | `none` | `qwen`: the sentence Qwen's MMMU script adds with `--use-cot` ("If you are uncertain or the problem is too complex, make a reasoned guess … Avoid repeating steps indefinitely … Determine whether to think step by step …"), verbatim; it leaves the decision to reason to the model. `forced`: "Think step by step and explain your reasoning before giving the answer." (ours), for runs whose errors must be diagnosable |
| `--lora_path` | none | LoRA adapter directory applied through vLLM (`enable_lora`, `--max_lora_rank` 64) on top of `--model_path`; a merged checkpoint can instead be passed as `--model_path` |
| `--no_score` | off | stop after generation and `check_raw.py` |
| `--answer_line` | off | appends `End your response with a final line of the form "Answer: X"` to the prompt: a termination target against looping and an anchor for the parser |
| `--fallback` | off | second pass for responses that hit the token limit or (multiple-choice) contain no valid option letter: original prompt + response with any verbatim loop cut off + `Therefore, the final answer is (`, greedy, `--fallback_max_tokens` (16). Written to `<out stem>.fallback.jsonl`; the raw file is untouched |
| `--seed` | 3407 | Qwen3-VL README "Evaluation Reproduction" value; the baseline run used 42. Per-item seed is derived from this and the item id |
| `--max_model_len`, `--max_num_seqs`, `--gpu_memory_utilization` | 16384, 32, 0.90 | vLLM engine |

Smoke test, one item from every subject (30 items, exercises all 30 configs):

```bash
bash scripts/run_mmmu_eval.sh --preset best --limit 1 --out outputs/smoke.jsonl
```

Full run with the best settings (default `--out outputs/raw.jsonl`; use another name to keep an
existing file):

```bash
bash scripts/run_mmmu_eval.sh --preset best --out outputs/best.jsonl
```

Fine-tuned model, identical settings:

```bash
bash scripts/run_mmmu_eval.sh --preset best --model_path /path/to/merged_checkpoint --out outputs/ft.jsonl
```

```bash
bash scripts/run_mmmu_eval.sh --preset best --lora_path /path/to/adapter --out outputs/ft_lora.jsonl
```

Re-score an existing raw file without generating:

```bash
python scripts/score_mmmu.py outputs/raw.jsonl
```

Ablation on a few subjects, kept apart from the baseline:

```bash
bash scripts/run_mmmu_eval.sh --subjects Math,Physics --max_pixels 4014080 --out outputs/ablation_pixels.jsonl
```

## Colab (A100)

```python
!git clone <repo-url> MMD && cd MMD && pip install -q -r requirements.txt
```

vLLM replaces Colab's torch 2.11 (CUDA 12.8) with torch 2.13 (CUDA 13.0) but leaves Colab's
`torchaudio==2.11.0+cu128` in place because the version number already matches, and
`transformers` then fails to import with a CUDA-version mismatch. Reinstall it from PyPI:

```python
!pip install -q --force-reinstall --no-deps torchaudio==2.11.0
```

Then:

```python
%cd MMD
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
!bash scripts/run_mmmu_eval.sh --preset best --limit 1 --out outputs/smoke.jsonl
!bash scripts/run_mmmu_eval.sh --preset best --out outputs/best.jsonl
!pip freeze > requirements.lock.txt
```

Run through `!bash`, not by importing the module in a cell: vLLM starts its engine in a
subprocess. Commit `outputs/best.jsonl`, `outputs/best.fallback.jsonl`, `outputs/best.run_meta.json`,
`results/best/` and `requirements.lock.txt` afterwards; Colab storage does not persist.

Budget for `best` on an A100 40GB: `max_model_len` 65536 needs about 9.6 GiB of KV cache (36 layers ×
8 KV heads × 128 dims × 2 bytes × 2), which fits next to the 8 GiB of weights on a 24 GB card as well.
Expect a longer run than the baseline's 37 minutes: images are up to 4× larger and a looping response
now consumes 32768 tokens before the fallback pass recovers its answer.

## Layout

```
scripts/run_mmmu_eval.sh          one-command entry point: presets, generation, check, scoring
scripts/check_raw.py              schema and count validation (stdlib only)
scripts/score_mmmu.py             strict + pipeline scoring, extended metrics, results.md
scripts/mmmu_official_eval_utils.py  MMMU official parser @ 268471d, byte-identical (Apache-2.0)
src/data.py                       30-config loader at the pinned dataset revision
src/prompt.py                     prompt styles, CoT / answer-line text, <image N> placement, smart_resize, loop trimming
src/eval_mmmu.py                  vLLM generation (+ LoRA), fallback pass, raw.jsonl + run_meta.json
docs/raw_schema.md                output contract for the raw and fallback files
```
