#!/usr/bin/env bash
# MMMU-Pro on Colab (test split, 1730 items per setting). Same recipe as best_r2u (vLLM, vlmevalkit prompt, images
# first, CoT sentence, answer line, greedy fallback, min/max_pixels 4014080, seed 3407) except max_new_tokens 16384
# (the model card's value for the VL model; the MMMU val runs used 32768).
#   nohup bash run_pro_colab.sh > /content/pro.log 2>&1 &
# Runs, in order: base standard-10, base standard-10 without images, base vision, H-SFT standard-10, H-SFT vision.
# Finished runs are skipped and unfinished ones resume from their output file.
set -uo pipefail
export PYTHONUNBUFFERED=1
BACKUP=${BACKUP:-/content/drive/MyDrive/mmdl_hd}
stage() { echo "=== $1 ($(date)) ==="; }
backup() {
  [ -d /content/drive/MyDrive ] || return 0
  rsync -a /content/MMD/outputs /content/MMD/results "$BACKUP/" 2>/dev/null
  cp /content/pro.log "$BACKUP/pro.log" 2>/dev/null
  date > "$BACKUP/last_backup.txt"
}
( while true; do sleep 300; backup; done ) &
BK=$!
trap 'backup; kill $BK 2>/dev/null' EXIT

stage patch
cd /content/MMD
python /content/step07_patch_noimg.py src/eval_mmmu.py || exit 1
python /content/step08_patch_pro.py /content/MMD || exit 1

run() {  # name, dataset, image mode, extra args
  local name=$1 dataset=$2 mode=$3; shift 3
  stage "pro $name"
  if [ ! -f "results/$name/summary.json" ]; then
    MMMU_DATASET=$dataset MMMU_IMAGE_MODE=$mode python src/eval_mmmu.py \
        --max_new_tokens 16384 --max_model_len 65536 --min_pixels 4014080 --max_pixels 4014080 \
        --prompt_style vlmevalkit --image_position first --seed 3407 --cot qwen --answer_line --fallback \
        --batch_items 100 "$@" --out "outputs/$name.jsonl" || { echo "FAILED $name"; return 1; }
    python scripts/score_mmmu.py "outputs/$name.jsonl" --partial || { echo "FAILED score $name"; return 1; }
  fi
  backup
}
SFT=/content/H/merged/H-sft
run pro10_base_s3407       pro10      normal
run pro10_base_noimg_s3407 pro10      none
run prov_base_s3407        pro_vision normal
run pro10_h_sft_s3407      pro10      normal --model_path $SFT
run prov_h_sft_s3407       pro_vision normal --model_path $SFT

stage compare
python scripts/compare_runs.py --labels base_std10,base_std10_noimg,base_vision,sft_std10,sft_vision \
    pro10_base_s3407 pro10_base_noimg_s3407 prov_base_s3407 pro10_h_sft_s3407 prov_h_sft_s3407
echo "PRO done ($(date))"
