#!/usr/bin/env bash
# One-command MMMU-val evaluation: generation -> schema check -> scoring -> results tables.
# All unknown arguments are forwarded to src/eval_mmmu.py.
#
#   bash scripts/run_mmmu_eval.sh --preset best                       # full 900-item run, best settings
#   bash scripts/run_mmmu_eval.sh --preset best --limit 1 --out outputs/smoke.jsonl
#   bash scripts/run_mmmu_eval.sh --preset best --model_path /path/to/finetuned   # or --lora_path DIR
#   bash scripts/run_mmmu_eval.sh --preset baseline --out outputs/raw.jsonl       # the committed baseline
#
# Presets (user flags after --preset override them):
#   baseline : seed 42, max_new_tokens 8192, max_model_len 16384, MMMU-repo prompt, inline images, no extras
#   official : Qwen's own MMMU script settings: 32768 tokens, min/max_pixels 1280*28^2 / 5120*28^2,
#              VLMEvalKit prompt, images first, seed 3407
#   best     : official + --cot qwen (Qwen --use-cot text) + --answer_line + --fallback
# Other flags: --out PATH  --subjects A,B  --limit N  --max_new_tokens N  --max_pixels N  --min_pixels N
#              --prompt_style mmmu|vlmevalkit  --image_position inline|first  --cot none|qwen|forced  --answer_line  --fallback
#              --seed N  --lora_path DIR  --no_score
set -euo pipefail
cd "$(dirname "$0")/.."

out="outputs/raw.jsonl"
partial=""
preset=""
score=1
fwd=()
while (($#)); do
  case "$1" in
    --preset) preset="$2"; shift 2 ;;
    --preset=*) preset="${1#--preset=}"; shift ;;
    --no_score) score=0; shift ;;
    --out) out="$2"; fwd+=("$1" "$2"); shift 2 ;;
    --out=*) out="${1#--out=}"; fwd+=("$1"); shift ;;
    --subjects | --limit) partial="--partial"; fwd+=("$1" "$2"); shift 2 ;;
    --subjects=* | --limit=*) partial="--partial"; fwd+=("$1"); shift ;;
    *) fwd+=("$1"); shift ;;
  esac
done

official=(--max_new_tokens 32768 --max_model_len 65536 --min_pixels 1003520 --max_pixels 4014080
          --prompt_style vlmevalkit --image_position first --seed 3407)
# ${fwd[@]+"${fwd[@]}"}: expand the array only when non-empty (bash 3.2 + set -u would otherwise abort).
case "$preset" in
  "") ;;
  baseline) fwd=(--seed 42 --max_new_tokens 8192 --max_model_len 16384 --prompt_style mmmu --image_position inline ${fwd[@]+"${fwd[@]}"}) ;;
  official) fwd=("${official[@]}" ${fwd[@]+"${fwd[@]}"}) ;;
  best) fwd=("${official[@]}" --cot qwen --answer_line --fallback ${fwd[@]+"${fwd[@]}"}) ;;
  *) echo "unknown preset: $preset (baseline|official|best)" >&2; exit 2 ;;
esac

python src/eval_mmmu.py ${fwd[@]+"${fwd[@]}"}
python scripts/check_raw.py "$out" $partial
if ((score)); then
  python scripts/score_mmmu.py "$out" $partial
fi
