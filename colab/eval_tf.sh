#!/usr/bin/env bash
# MMMU val with the Transformers backend (for machines where vLLM cannot be installed), split by subject into
# parallel shards on one GPU. Per-item seeds do not depend on the split, so shards give the same answers as one run.
#   bash eval_tf.sh NAME [MODEL_PATH]        SHARDS=3  TOK=2048
# Writes MMD/outputs/NAME.jsonl (+ .fallback.jsonl) and MMD/results/NAME/. A rerun resumes unfinished shards.
set -uo pipefail
export PYTHONUNBUFFERED=1
H=$(cd "$(dirname "$0")" && pwd)
MMD=$(dirname "$H")/MMD
NAME=$1
MODEL=${2:-}
SHARDS=${SHARDS:-3}
TOK=${TOK:-2048}
cd "$MMD" || exit 1
mkdir -p outputs logs
[ -f "results/$NAME/summary.json" ] && { echo "$NAME already scored"; exit 0; }

mapfile -t CHUNKS < <(python - "$SHARDS" <<'EOF'
import sys
sys.path.insert(0, "src")
from data import SUBJECTS
n = int(sys.argv[1])
k = -(-len(SUBJECTS) // n)
for i in range(0, len(SUBJECTS), k):
    print(",".join(SUBJECTS[i:i + k]))
EOF
)
[ "${#CHUNKS[@]}" -gt 0 ] || { echo "FAILED $NAME: could not read the subject list"; exit 1; }
echo "=== $NAME: ${#CHUNKS[@]} shards, max_new_tokens $TOK ($(date)) ==="

pids=()
for i in "${!CHUNKS[@]}"; do
  args=(--preset best --no_score --backend transformers --min_pixels 4014080 --max_new_tokens "$TOK"
        --batch_items 10 --subjects "${CHUNKS[$i]}" --out "outputs/$NAME.part$i.jsonl")
  [ -n "$MODEL" ] && args+=(--model_path "$MODEL")
  bash scripts/run_mmmu_eval.sh "${args[@]}" > "logs/$NAME.part$i.log" 2>&1 &
  pids+=($!)
  sleep 20
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done

: > "outputs/$NAME.jsonl"
: > "outputs/$NAME.fallback.jsonl"
for i in "${!CHUNKS[@]}"; do
  [ -f "outputs/$NAME.part$i.jsonl" ] && cat "outputs/$NAME.part$i.jsonl" >> "outputs/$NAME.jsonl"
  [ -f "outputs/$NAME.part$i.fallback.jsonl" ] && cat "outputs/$NAME.part$i.fallback.jsonl" >> "outputs/$NAME.fallback.jsonl"
done
n=$(wc -l < "outputs/$NAME.jsonl")
echo "$NAME: $n records (shard failure flag $fail) ($(date))"
if [ "$n" -ne 900 ]; then
  echo "FAILED $NAME: expected 900 records"
  tail -n 6 logs/"$NAME".part*.log
  exit 1
fi
python scripts/check_raw.py "outputs/$NAME.jsonl" && python scripts/score_mmmu.py "outputs/$NAME.jsonl" || { echo "FAILED $NAME: scoring"; exit 1; }
echo "DONE $NAME ($(date))"
