# `outputs/raw.jsonl` — contract between generation (A) and scoring (B)

One JSON object per line, UTF-8, one line per MMMU validation item. The full run has 900 lines:
30 subjects × 30 items. Lines are appended subject by subject in the order of the assignment's
subject list; do not rely on order, join on `id`. No other fields are written.

Validate any file with `python scripts/check_raw.py <file>` (add `--partial` for smoke or
ablation files that do not cover all 900 items).

## Fields

| Field | Type | Meaning |
|---|---|---|
| `id` | str | MMMU id, `validation_<subject>_<n>`. Unique in the file. |
| `subject` | str | One of the 30 config names, e.g. `Art_Theory`. |
| `question_type` | str | `multiple-choice` or `open`, copied from the dataset. |
| `options` | list[str] | Option texts in order; index 0 is (A). `[]` for `open`. An option may be a literal `<image N>` reference. |
| `answer` | str | Gold answer, copied verbatim from the dataset. Never shown to the model. |
| `prompt` | str | The exact string sent to the model, after the chat template. |
| `response` | str | The model's raw continuation. No prompt echo, no stripping, no post-processing. |
| `finish_reason` | str | `stop` (model ended the turn) or `length` (hit `max_new_tokens`, so the answer may be cut off). |
| `num_images` | int | Images actually passed to the model for this item. |

## What B needs to know

**Counts in the pinned validation split** (dataset revision `98e6ac0…`, measured): 847
`multiple-choice`, 53 `open`. Images per item: 1 → 857, 2 → 24, 3 → 5, 4 → 8, 5 → 6.
Multiple-choice items have 2 to 9 options, so valid letters go beyond A–D.

**`answer` format.** Multiple-choice: a single letter. Open: usually a plain string (`"1.06"`,
`"Transformation"`), but three items store several accepted answers as a Python-literal list in
a string: `validation_Chemistry_30` (`"['$MgS$', 'MgS']"`), `validation_Geography_4`,
`validation_Math_15`. Parse with `ast.literal_eval` when the string starts with `[`.

**Prompt text.** Copied from the MMMU official repo
(`MMMU-Benchmark/MMMU` @ `268471d`, `mmmu/configs/llava1.5.yaml` and `construct_prompt` in
`mmmu/utils/data_utils.py`), with an empty task instruction:

```
{question}

(A) {option_A}
(B) {option_B}
...


Answer with the option's letter from the given choices directly.
```

```
{question}

Answer the question using a single word or phrase.
```

The blank line doubled before "Answer with" is in the original (each option line ends in `\n`
and the format adds `\n\n`). Responses are therefore expected to be a bare letter or a short
phrase, which is what the official `parse_multi_choice_response` / `parse_open_response` assume.

**Images in `prompt`.** Each `<image N>` in the question or options is replaced, at its first
occurrence, by the image from column `image_N`; in `prompt` it shows as
`<|vision_start|><|image_pad|><|vision_end|>`. A repeated reference to the same image stays as
the literal text `<image N>`. Four items have images that the text never references
(`validation_Agriculture_26`, `validation_Materials_15`, `validation_Pharmacy_4`,
`validation_Pharmacy_22`); those images are placed at the start of the user turn. This differs
from the official MMMU script, which passes only `image_1`.

**Sampling is on** (temperature 0.7), so `response` is one draw, not a deterministic output.
Each item's seed is derived from the global seed and its `id`, so a rerun with the same
settings, versions and hardware targets the same draw regardless of batch order.

## `<stem>.fallback.jsonl` (only with `--fallback`)

One line per item that hit `max_new_tokens` or, for multiple-choice, whose response contains no
valid option letter. `response` in the main file is still the raw first pass. Fields: `id`,
`trigger` (`length` or `no_letter`), `loop_chars_removed` (verbatim loop cut from the tail before
the second pass, 0 if none), `response_chars_used`, `fallback_prompt_tokens`, `fallback_suffix`
(`\n\nTherefore, the final answer is (` for multiple-choice, `...is:` for open),
`fallback_response` (greedy continuation, at most `--fallback_max_tokens`), `fallback_finish_reason`.
For a multiple-choice item the parser input is `"(" + fallback_response`. Score with and without
the substitution and report both.

## `results/<stem>/` (written by `scripts/score_mmmu.py`)

`scored.jsonl` has one line per item: `id, subject, question_type, answer, finish_reason, num_images,
response_chars, response_group (answer_only | reasoned | truncated), loop_chars_removed, strict_pred,
strict_method, strict_correct, pipeline_pred, pipeline_method, pipeline_correct, used_fallback`.
`by_subject.csv`, `summary.json` and `results.md` aggregate it. Parser and the two scores are described
in the README.

## `outputs/run_meta.json`

Written next to the jsonl (for another `--out` name, `<name>.run_meta.json`). Holds the git
commit, model and dataset revisions, package versions, GPU and driver, the effective sampling
parameters, `max_new_tokens`, `max_pixels`, prompt hash, wall time, peak VRAM with its
measurement method, `finish_reason` counts, and any prompt-building warnings.

## Files B will receive

| File | When | Check mode |
|---|---|---|
| `outputs/smoke.jsonl` | optional: smoke run (one item per subject, 30 lines) | `--partial` |
| `outputs/raw.jsonl` + `outputs/run_meta.json` | after the full run | full |
| `outputs/ablation_*.jsonl` | on request, never overwriting `raw.jsonl` | `--partial` |
