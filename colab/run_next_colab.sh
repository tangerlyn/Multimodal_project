#!/usr/bin/env bash
# Queue behind run_hd_colab.sh on Colab: evaluates the server-trained LoRAs one after another on MMMU val with the
# same settings as best_r2u and h_dpo_s3407 (vLLM, preset best, min_pixels 4014080, seed 3407). No training here.
#   uploads needed: /content/next_bundle.tgz (H/runs/H-dpoT/lora, H/runs/H-sft/lora)
#   nohup bash run_next_colab.sh > /content/hs.log 2>&1 &
# Starts only if /content/hd.log ended with "HD done". Finished stages are skipped, so a rerun resumes.
#   MODELS="dpoT sft"   order of evaluation; run names are h_dpot_s3407, h_sft_s3407
set -uo pipefail
export PYTHONUNBUFFERED=1
BACKUP=${BACKUP:-/content/drive/MyDrive/mmdl_hd}
MODELS=${MODELS:-dpoT sft}
stage() { echo "=== $1 ($(date)) ==="; }

backup() {
  [ -d /content/drive/MyDrive ] || return 0
  mkdir -p "$BACKUP"
  rsync -a /content/MMD/outputs /content/MMD/results "$BACKUP/" 2>/dev/null
  cp /content/hs.log "$BACKUP/hs.log" 2>/dev/null
  date > "$BACKUP/last_backup.txt"
}

stage wait
until grep -aq '^EXIT' /content/hd.log 2>/dev/null; do sleep 60; done
grep -aq 'HD done' /content/hd.log || { echo "the H-DPO run did not finish cleanly, queue not started"; exit 1; }
sleep 20
( while true; do sleep 300; backup; done ) &
BK=$!
trap 'backup; kill $BK 2>/dev/null' EXIT

stage unpack
tar xzf /content/next_bundle.tgz -C /content 2>/dev/null || exit 1
ls /content/H/runs

names=(best_r2u h_dpo_s3407)
labels="base,h_dpo"
for M in $MODELS; do
  NAME="h_$(echo "$M" | tr 'A-Z' 'a-z')_s3407"
  stage "merge $M"
  cd /content/H
  if [ ! -f "merged/H-$M/merge_info.json" ] && [ ! -f "/content/MMD/results/$NAME/summary.json" ]; then
    rm -rf "merged/H-$M"
    python step06_merge_model.py --lora "runs/H-$M/lora" --out "merged/H-$M" || { echo "FAILED merge $M"; continue; }
  fi
  stage "mmmu $M"
  cd /content/MMD
  if [ ! -f "results/$NAME/summary.json" ]; then
    bash scripts/run_mmmu_eval.sh --preset best --min_pixels 4014080 --batch_items 50 \
        --model_path "/content/H/merged/H-$M" --out "outputs/$NAME.jsonl" || { echo "FAILED mmmu $M"; continue; }
  fi
  backup
  names+=("$NAME"); labels="$labels,h_$(echo "$M" | tr 'A-Z' 'a-z')"
  stage "compare after $M"
  python scripts/compare_runs.py --labels "$labels" "${names[@]}"
done
echo "NEXT done ($(date))"
