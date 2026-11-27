#!/usr/bin/env bash
# Resume of run_pro_colab.sh on a fresh Colab VM: H-SFT on MMMU-Pro standard-10, then vision (same settings).
#   uploads needed in /content: g1_bundle.tgz, sft_bundle.tgz (H/runs/H-sft/lora, H/step06_merge_model.py, H/step00_common.py),
#                               step07_patch_noimg.py, step08_patch_pro.py
#   nohup bash run_pro2_colab.sh > /content/pro.log 2>&1 &
# Waits up to WAIT_DRIVE seconds for Google Drive before the evaluation, restores outputs/results from $BACKUP
# (partial runs resume from their output file) and backs up every 5 minutes.
set -uo pipefail
export PYTHONUNBUFFERED=1
BACKUP=${BACKUP:-/content/drive/MyDrive/mmdl_hd}
WAIT_DRIVE=${WAIT_DRIVE:-1800}
stage() { echo "=== $1 ($(date)) ==="; }
backup() {
  [ -d /content/drive/MyDrive ] || return 0
  rsync -a /content/MMD/outputs /content/MMD/results "$BACKUP/" 2>/dev/null
  cp /content/pro.log "$BACKUP/pro2.log" 2>/dev/null
  date > "$BACKUP/last_backup.txt"
}

stage unpack
mkdir -p /content/MMD /content/H
[ -d /content/MMD/src ] || tar xzf /content/g1_bundle.tgz -C /content/MMD || exit 1
[ -d /content/H/runs/H-sft/lora ] || tar xzf /content/sft_bundle.tgz -C /content || exit 1
mkdir -p /content/MMD/outputs /content/MMD/results

stage setup
if ! python -c "import vllm" 2>/dev/null; then
  pip install -q -r /content/MMD/requirements.txt > /content/pip.log 2>&1 || { echo "pip failed, see /content/pip.log"; exit 1; }
  pip install -q --force-reinstall --no-deps torchaudio==2.11.0 >> /content/pip.log 2>&1
fi
python -c "import peft" 2>/dev/null || pip install -q peft >> /content/pip.log 2>&1
pip uninstall -y -q torchao >> /content/pip.log 2>&1
python -c "import torch, transformers, vllm, peft; print('torch', torch.__version__, 'transformers', transformers.__version__, 'vllm', vllm.__version__, 'peft', peft.__version__)" || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

stage merge
cd /content/H
[ -f merged/H-sft/merge_info.json ] || { rm -rf merged/H-sft; python step06_merge_model.py --lora runs/H-sft/lora --out merged/H-sft || exit 1; }

stage "wait for Drive"
t=0
until [ -d /content/drive/MyDrive ] || [ $t -ge "$WAIT_DRIVE" ]; do sleep 20; t=$((t + 20)); done
if [ -d "$BACKUP/outputs" ]; then
  rsync -a --include 'pro*' --include 'prov*' --exclude '*' "$BACKUP/outputs/" /content/MMD/outputs/
  rsync -a --include 'pro*/***' --include 'prov*/***' --exclude '*' "$BACKUP/results/" /content/MMD/results/
  echo "restored from $BACKUP"; wc -l /content/MMD/outputs/pro*_s3407.jsonl
else
  echo "WARNING: Google Drive not mounted, starting H-SFT from zero and without backup"
fi
( while true; do sleep 300; backup; done ) &
BK=$!
trap 'backup; kill $BK 2>/dev/null' EXIT

stage patch
cd /content/MMD
python /content/step07_patch_noimg.py src/eval_mmmu.py || exit 1
python /content/step08_patch_pro.py /content/MMD || exit 1

run() {  # name, dataset, extra args
  local name=$1 dataset=$2; shift 2
  stage "pro $name"
  if [ ! -f "results/$name/summary.json" ]; then
    # eval_mmmu.py exits non-zero after a complete non-900 run ("full run expected 900"), so check the line count.
    MMMU_DATASET=$dataset python src/eval_mmmu.py \
        --max_new_tokens 16384 --max_model_len 65536 --min_pixels 4014080 --max_pixels 4014080 \
        --prompt_style vlmevalkit --image_position first --seed 3407 --cot qwen --answer_line --fallback \
        --batch_items 100 "$@" --out "outputs/$name.jsonl"
    [ "$(wc -l < "outputs/$name.jsonl")" -eq 1730 ] || { echo "FAILED $name"; return 1; }
    python scripts/score_mmmu.py "outputs/$name.jsonl" --partial || { echo "FAILED score $name"; return 1; }
  fi
  backup
}
SFT=/content/H/merged/H-sft
run pro10_h_sft_s3407 pro10      --model_path $SFT
run prov_h_sft_s3407  pro_vision --model_path $SFT

stage compare
python scripts/compare_runs.py --labels base_std10,base_std10_noimg,base_vision,sft_std10,sft_vision \
    pro10_base_s3407 pro10_base_noimg_s3407 prov_base_s3407 pro10_h_sft_s3407 prov_h_sft_s3407
echo "PRO done ($(date))"
