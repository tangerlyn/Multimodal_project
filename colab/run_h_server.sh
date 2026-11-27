#!/usr/bin/env bash
# Experiment H on the lab server (one A100). Layout: <root>/H (this code + build/), <root>/MMD (MMMU evaluator), <root>/virl.
#   cd <root>/H && nohup bash run_h_server.sh > ../h.log 2>&1 &
# Uses the saved first-pass samples (build/gen.jsonl), skips the 2048-token re-sampling, then for each mode:
# train -> merge -> MMMU val. The base model is evaluated with the same settings so the comparison is like for like.
# Every stage skips when its output exists, so a rerun resumes.
#   TOK=2048        generation cap for the MMMU runs (guideline value; the old runs used 32768)
#   MODES="dpo sft" training modes, in order      LR=5e-5
#   EVAL_EXTRA=""   extra flags for run_mmmu_eval.sh (e.g. "--backend transformers")
set -uo pipefail
export PYTHONUNBUFFERED=1
H=$(cd "$(dirname "$0")" && pwd)
ROOT=$(dirname "$H")
MMD=$ROOT/MMD
TOK=${TOK:-2048}
LR=${LR:-5e-5}
MODES=${MODES:-"dpo sft"}
EVAL_EXTRA=${EVAL_EXTRA:-}
stage() { echo "=== $1 ($(date)) ==="; }
cd "$H"
mkdir -p build2 runs merged "$MMD/outputs" "$MMD/results"

stage paths
if [ ! -f .paths_done ]; then
  sed -i "s|/content/MMD|$MMD|g; s|/content/virl|$ROOT/virl|g" ./*.py build/*.jsonl || exit 1
  touch .paths_done
fi
echo "root $ROOT"

stage env
python -c "import torch, transformers, peft; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'transformers', transformers.__version__, 'peft', peft.__version__)" || exit 1
python -c "import vllm; print('vllm', vllm.__version__)" || echo "vllm missing (set EVAL_EXTRA='--backend transformers')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

stage virl
if [ ! -f "$ROOT/virl/.ready" ]; then
  mkdir -p "$ROOT/virl"
  python -c "from huggingface_hub import snapshot_download; snapshot_download('TIGER-Lab/ViRL39K', repo_type='dataset', local_dir='$ROOT/virl')" || exit 1
  (cd "$ROOT/virl" && unzip -q -o images.zip) || exit 1
  touch "$ROOT/virl/.ready"
fi

stage build
[ -f build2/pairs_train.jsonl ] || python step04_build_pairs.py --gen build/gen.jsonl --out build2 --max_tokens 1500 || exit 1
wc -l build2/*.jsonl

mmmu() {  # mmmu NAME [extra flags]: skipped when results/NAME/summary.json exists
  local name=$1; shift
  [ -f "$MMD/results/$name/summary.json" ] && { echo "$name already scored"; return 0; }
  (cd "$MMD" && bash scripts/run_mmmu_eval.sh --preset best --min_pixels 4014080 --max_new_tokens "$TOK" \
      --batch_items 50 --out "outputs/$name.jsonl" $EVAL_EXTRA "$@") || echo "FAILED mmmu $name"
}

names=("base_t$TOK")
first=1
for M in $MODES; do
  stage "train $M"
  [ -f "runs/H-$M/training_run.json" ] || python step05_train.py --mode "$M" --lr "$LR" \
      --data build2/pairs_train.jsonl --val build2/pairs_val.jsonl --out "runs/H-$M" || exit 1
  stage "merge $M"
  [ -f "merged/H-$M/merge_info.json" ] || { rm -rf "merged/H-$M"; python step06_merge_model.py --lora "runs/H-$M/lora" --out "merged/H-$M" || exit 1; }
  if ((first)); then stage "mmmu base"; mmmu "base_t$TOK"; first=0; fi
  stage "mmmu $M"
  mmmu "h_${M}_t$TOK" --model_path "$H/merged/H-$M"
  names+=("h_${M}_t$TOK")
  stage "compare"
  (cd "$MMD" && python scripts/compare_runs.py "${names[@]}")
done
echo "H done ($(date))"
