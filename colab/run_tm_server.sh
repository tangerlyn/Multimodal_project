#!/usr/bin/env bash
# Experiment T variants on the lab server: the same teacher data and settings as T-sft, with more of the model open.
#   TM   LoRA on LLM attention + MLP
#   TMC  LoRA on LLM attention + MLP, and the connector (merger + DeepStack mergers) trained in full
#   TH   attention LoRA as T-sft, but only on questions the student got right at most half the time
#   cd ~/mmdl/h_run/H && (nohup bash run_tm_server.sh > ../chainTM.log 2>&1 &)
#   VARIANTS="TM TMC" (order), DATA=<dir with pairs_train.jsonl> (default: the one T-sft used)
# Each variant: smoke test (16 examples, then a merge without the weight check: two steps are too small to show in
# bf16) -> train -> merge -> MMMU val (cap 2048).
# Every stage skips when its output exists, so a rerun resumes. Ends with "TM CHAIN EXIT <code>".
set -uo pipefail
export PYTHONUNBUFFERED=1
H=$(cd "$(dirname "$0")" && pwd)
cd "$H"
VARIANTS=${VARIANTS:-"TM TMC"}
stage() { echo "=== $1 ($(date)) ==="; }
finish() { echo "TM CHAIN EXIT $1"; exit "$1"; }

DATA=${DATA:-$(python -c "import json; print(json.load(open('runs/T-sft/training_run.json'))['data'].rsplit('/', 1)[0])")}
[ -f "$DATA/pairs_train.jsonl" ] && [ -f "$DATA/pairs_val.jsonl" ] || { echo "no teacher data in '$DATA'"; finish 1; }
echo "data $DATA: $(wc -l < "$DATA/pairs_train.jsonl") train, $(wc -l < "$DATA/pairs_val.jsonl") val"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

names=(base_t2048 t_sft_t2048)
for V in $VARIANTS; do
  D=$DATA
  case $V in
    TM)  extra=(--targets attn+mlp) ;;
    TMC) extra=(--targets attn+mlp --connector) ;;
    TH)  extra=(--targets attn); D=$DATA-hard
         [ -f "$D/pairs_train.jsonl" ] || python step14_t_build_sft.py --teacher "$DATA/teacher.jsonl" --out "$D" --max_student_pass 0.5 || { echo "FAILED build $V"; finish 1; } ;;
    *) echo "unknown variant $V"; finish 1 ;;
  esac
  common=(--mode sft --rank 64 --alpha 128 --lr 5e-5 --epochs 2 --data "$D/pairs_train.jsonl" --val "$D/pairs_val.jsonl")
  conn=()

  if [ ! -f "runs/$V-sft/training_run.json" ]; then
    stage "smoke $V"
    rm -rf "runs/smoke-$V" "merged/smoke-$V"
    python step05_train.py "${common[@]}" "${extra[@]}" --limit 24 --max-steps 16 --out "runs/smoke-$V" || { echo "FAILED smoke train $V"; finish 1; }
    [ -f "runs/smoke-$V/connector.pt" ] && conn=(--connector "runs/smoke-$V/connector.pt")
    python step06_merge_model.py --lora "runs/smoke-$V/lora" "${conn[@]}" --no-check --out "merged/smoke-$V" || { echo "FAILED smoke merge $V"; finish 1; }
    rm -rf "runs/smoke-$V" "merged/smoke-$V"
    stage "train $V"
    python step05_train.py "${common[@]}" "${extra[@]}" --out "runs/$V-sft" || { echo "FAILED train $V"; finish 1; }
  fi

  stage "merge $V"
  conn=()
  [ -f "runs/$V-sft/connector.pt" ] && conn=(--connector "runs/$V-sft/connector.pt")
  [ -f "merged/$V-sft/merge_info.json" ] || { rm -rf "merged/$V-sft"; python step06_merge_model.py --lora "runs/$V-sft/lora" "${conn[@]}" --out "merged/$V-sft" || { echo "FAILED merge $V"; finish 1; }; }

  stage "mmmu $V"
  name=$(echo "$V" | tr 'A-Z' 'a-z')_sft_t2048
  SHARDS=4 TOK=2048 bash eval_tf.sh "$name" "$H/merged/$V-sft" || { echo "FAILED eval $V"; finish 1; }
  names+=("$name")
  stage "compare"
  (cd ../MMD && python scripts/compare_runs.py "${names[@]}")
done
finish 0
