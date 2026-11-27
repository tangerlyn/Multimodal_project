#!/usr/bin/env bash
# H-SFT on Colab, queued behind run_hd_colab.sh: waits for the H-DPO evaluation to finish, then builds the same
# pairs, trains H-SFT, merges it and runs MMMU val with the same settings as best_r2u and h_dpo_s3407.
#   uploads needed: /content/h_server_bundle.tgz (H code + build/gen.jsonl)
#   nohup bash run_hs_colab.sh > /content/hs.log 2>&1 &
# Starts only if /content/hd.log ended with "HD done". Stages skip when their output exists, so a rerun resumes.
set -uo pipefail
export PYTHONUNBUFFERED=1
BACKUP=${BACKUP:-/content/drive/MyDrive/mmdl_hd}
NAME=h_sft_s3407
LR=${LR:-5e-5}
stage() { echo "=== $1 ($(date)) ==="; }

backup() {
  [ -d /content/drive/MyDrive ] || return 0
  mkdir -p "$BACKUP/H"
  rsync -a /content/MMD/outputs /content/MMD/results "$BACKUP/" 2>/dev/null
  rsync -a --exclude 'merged' /content/H/runs /content/H/build2 "$BACKUP/H/" 2>/dev/null
  cp /content/hs.log "$BACKUP/hs.log" 2>/dev/null
  date > "$BACKUP/last_backup.txt"
}

stage wait
until grep -aq '^EXIT' /content/hd.log 2>/dev/null; do sleep 60; done
grep -aq 'HD done' /content/hd.log || { echo "the H-DPO run did not finish cleanly, H-SFT not started"; exit 1; }
sleep 20
( while true; do sleep 300; backup; done ) &
BK=$!
trap 'backup; kill $BK 2>/dev/null' EXIT

stage unpack
mkdir -p /content/hs /content/H/build
[ -d /content/hs/h_run ] || tar xzf /content/h_server_bundle.tgz -C /content/hs 2>/dev/null || exit 1
cp /content/hs/h_run/H/step04_build_pairs.py /content/hs/h_run/H/step05_train.py /content/hs/h_run/H/step00_common.py /content/H/ || exit 1
cp -n /content/hs/h_run/H/build/* /content/H/build/
if [ -d "$BACKUP/H/runs/H-sft" ] && [ ! -d /content/H/runs/H-sft ]; then
  mkdir -p /content/H/runs && rsync -a "$BACKUP/H/runs/H-sft" /content/H/runs/ && echo "restored H-sft from $BACKUP"
fi

stage virl
if [ ! -f /content/virl/.ready ]; then
  mkdir -p /content/virl
  python -c "from huggingface_hub import snapshot_download; snapshot_download('TIGER-Lab/ViRL39K', repo_type='dataset', local_dir='/content/virl')" || exit 1
  (cd /content/virl && unzip -q -o images.zip) || exit 1
  touch /content/virl/.ready
fi

stage build
cd /content/H
[ -f build2/pairs_train.jsonl ] || python step04_build_pairs.py --gen build/gen.jsonl --out build2 --max_tokens 1500 || exit 1
wc -l build2/*.jsonl

stage "train sft"
[ -f runs/H-sft/training_run.json ] || python step05_train.py --mode sft --lr "$LR" \
    --data build2/pairs_train.jsonl --val build2/pairs_val.jsonl --out runs/H-sft || exit 1
backup

stage "merge sft"
[ -f merged/H-sft/merge_info.json ] || { rm -rf merged/H-sft; python step06_merge_model.py --lora runs/H-sft/lora --out merged/H-sft || exit 1; }
rm -rf /content/H/merged/H-dpo

stage "mmmu sft"
cd /content/MMD
if [ ! -f "results/$NAME/summary.json" ]; then
  bash scripts/run_mmmu_eval.sh --preset best --min_pixels 4014080 --batch_items 50 \
      --model_path /content/H/merged/H-sft --out "outputs/$NAME.jsonl" || { echo "FAILED mmmu"; exit 1; }
fi
backup

stage compare
python scripts/compare_runs.py --labels base,h_dpo,h_sft best_r2u h_dpo_s3407 "$NAME"
echo "HS done ($(date))"
