#!/usr/bin/env bash
# Evaluate base and H-DPO on MMMU val with the Transformers backend, sharing the GPU with a running training job.
#   cd <root>/H && nohup bash run_eval_server.sh > ../ev.log 2>&1 &
# Base starts now; H-DPO starts when its merged checkpoint appears. The number of shards for each run is chosen
# from the GPU memory that is free at that moment (about 14 GB per shard, 10 GB kept in reserve).
#   MODES="dpo"   which merged models to evaluate after base (e.g. "dpo sft")
set -uo pipefail
H=$(cd "$(dirname "$0")" && pwd)
MMD=$(dirname "$H")/MMD
TOK=${TOK:-2048}
MODES=${MODES:-dpo}
cd "$H" || exit 1

shards() {
  local free
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1)
  local n=$(( (free - 10000) / 14000 ))
  [ "$n" -lt 1 ] && n=1
  [ "$n" -gt "${1:-3}" ] && n=${1:-3}
  echo "$n"
}

names=("base_t$TOK")
s=$(shards 3)
echo "=== base: $s shards, free GPU memory $(nvidia-smi --query-gpu=memory.free --format=csv,noheader) ($(date)) ==="
SHARDS=$s TOK=$TOK bash eval_tf.sh "base_t$TOK" > ../ev_base.log 2>&1 &
base_pid=$!

for M in $MODES; do
  echo "waiting for merged/H-$M ..."
  while [ ! -f "merged/H-$M/merge_info.json" ]; do
    [ -f "runs/H-$M/FAILED" ] && break
    sleep 60
  done
  sleep 30
  s=$(shards 3)
  echo "=== $M: $s shards, free GPU memory $(nvidia-smi --query-gpu=memory.free --format=csv,noheader) ($(date)) ==="
  SHARDS=$s TOK=$TOK bash eval_tf.sh "h_${M}_t$TOK" "$H/merged/H-$M" > "../ev_$M.log" 2>&1
  names+=("h_${M}_t$TOK")
done
wait "$base_pid"
echo "=== compare ($(date)) ==="
(cd "$MMD" && python scripts/compare_runs.py "${names[@]}")
echo "EVAL done ($(date))"
