#!/usr/bin/env bash
# Visual-dependency diagnostic on Colab: MMMU val with the images removed, same recipe as best_r2u otherwise
# (vLLM, preset best, seed 3407). Runs the base model, then the merged H-SFT model.
#   nohup bash run_noimg_colab.sh > /content/noimg.log 2>&1 &
# Compare with best_r2u (65.56%) and h_sft_s3407 (66.22%), which saw the images.
set -uo pipefail
export PYTHONUNBUFFERED=1
BACKUP=${BACKUP:-/content/drive/MyDrive/mmdl_hd}
stage() { echo "=== $1 ($(date)) ==="; }
backup() {
  [ -d /content/drive/MyDrive ] || return 0
  rsync -a /content/MMD/outputs /content/MMD/results "$BACKUP/" 2>/dev/null
  cp /content/noimg.log "$BACKUP/noimg.log" 2>/dev/null
  date > "$BACKUP/last_backup.txt"
}
( while true; do sleep 300; backup; done ) &
BK=$!
trap 'backup; kill $BK 2>/dev/null' EXIT

stage patch
cd /content/MMD
python /content/step07_patch_noimg.py src/eval_mmmu.py || exit 1

run() {  # name, extra args
  local name=$1; shift
  stage "mmmu $name"
  if [ ! -f "results/$name/summary.json" ]; then
    MMMU_IMAGE_MODE=none bash scripts/run_mmmu_eval.sh --preset best --min_pixels 4014080 --batch_items 50 \
        "$@" --out "outputs/$name.jsonl" || { echo "FAILED $name"; return 1; }
  fi
  grep -a '"image_mode"' "outputs/$name.run_meta.json" | head -n 1 | cut -c1-80
  backup
}
run base_noimg_s3407
run h_sft_noimg_s3407 --model_path /content/H/merged/H-sft

stage compare
python scripts/compare_runs.py --labels base,base_noimg,h_sft,h_sft_noimg best_r2u base_noimg_s3407 h_sft_s3407 h_sft_noimg_s3407
echo "NOIMG done ($(date))"
