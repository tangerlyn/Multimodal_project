#!/usr/bin/env bash
# One-command MMMU-val generation. All arguments are forwarded to src/eval_mmmu.py.
#
#   bash scripts/run_mmmu_eval.sh --model_path Qwen/Qwen3-VL-4B-Instruct --data_root "$HF_DATASETS_CACHE"
#
# Optional: --out PATH  --subjects A,B  --limit N  --max_new_tokens N  --max_pixels N  --seed N
# Without --subjects/--limit this is the full 900-item run and the output is checked as such.
set -euo pipefail
cd "$(dirname "$0")/.."

out="outputs/raw.jsonl"
partial=""
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  case "${args[i]}" in
    --out) out="${args[i + 1]}" ;;
    --out=*) out="${args[i]#--out=}" ;;
    --subjects | --subjects=* | --limit | --limit=*) partial="--partial" ;;
  esac
done

python src/eval_mmmu.py "$@"
python scripts/check_raw.py "$out" $partial
