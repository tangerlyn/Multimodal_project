#!/usr/bin/env bash
# Evaluate the server-trained H-DPO LoRA on MMMU val on Colab, with exactly the settings of the base run best_r2u
# (vLLM, preset best, min_pixels 4014080, max_new_tokens 32768, seed 3407), so the two are compared directly.
#   uploads needed: /content/g1_bundle.tgz (evaluator + results/best_r2u), /content/t_bundle.tgz.part* (H/runs/T-sft/lora, step06_merge_model.py, step00_common.py; split because large uploads fail)
#   nohup bash run_hd_colab.sh > /content/t.log 2>&1 &
# With Google Drive mounted, outputs/results and the log are copied to $BACKUP every 5 minutes and restored at
# start, and the evaluator skips finished items, so a rerun on a new VM resumes.
set -uo pipefail
export PYTHONUNBUFFERED=1
BACKUP=${BACKUP:-/content/drive/MyDrive/mmdl_hd}
NAME=t_sft_s3407
stage() { echo "=== $1 ($(date)) ==="; }
mkdir -p /content/MMD /content/H

backup() {
  [ -d /content/drive/MyDrive ] || return 0
  mkdir -p "$BACKUP"
  rsync -a /content/MMD/outputs /content/MMD/results "$BACKUP/" 2>/dev/null
  cp /content/t.log "$BACKUP/t.log" 2>/dev/null
  date > "$BACKUP/last_backup.txt"
}

stage unpack
[ -d /content/MMD/src ] || tar xzf /content/g1_bundle.tgz -C /content/MMD || exit 1
[ -d /content/H/runs/T-sft/lora ] || { cat /content/t_bundle.tgz.part* | tar xzf - -C /content; } || exit 1
mkdir -p /content/MMD/outputs /content/MMD/results

stage setup
if ! python -c "import vllm" 2>/dev/null; then
  pip install -q -r /content/MMD/requirements.txt > /content/t_pip.log 2>&1 || { echo "pip failed, see /content/t_pip.log"; exit 1; }
  pip install -q --force-reinstall --no-deps torchaudio==2.11.0 >> /content/t_pip.log 2>&1
fi
python -c "import peft" 2>/dev/null || pip install -q peft >> /content/t_pip.log 2>&1
# Colab ships an old torchao; peft refuses to load a LoRA when it finds one below 0.16. It is not needed here.
pip uninstall -y -q torchao >> /content/t_pip.log 2>&1
python -c "import torch, transformers, vllm, peft; print('torch', torch.__version__, 'transformers', transformers.__version__, 'vllm', vllm.__version__, 'peft', peft.__version__)" || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

stage merge
cd /content/H
[ -f merged/T-sft/merge_info.json ] || { rm -rf merged/T-sft; python step06_merge_model.py --lora runs/T-sft/lora --out merged/T-sft || exit 1; }

stage "wait for Drive"
t=0
until [ -d /content/drive/MyDrive ] || [ $t -ge "${WAIT_DRIVE:-1800}" ]; do sleep 20; t=$((t + 20)); done
if [ -d "$BACKUP/outputs" ]; then
  rsync -a --exclude 'pro*' "$BACKUP/outputs" "$BACKUP/results" /content/MMD/ && echo "restored from $BACKUP"
  wc -l /content/MMD/outputs/$NAME.jsonl 2>/dev/null
elif [ -d /content/drive/MyDrive ]; then
  echo "no backup yet at $BACKUP (fresh run)"
else
  echo "WARNING: Google Drive is not mounted, nothing will be backed up"
fi
( while true; do sleep 300; backup; done ) &
BK=$!
trap 'backup; kill $BK 2>/dev/null' EXIT

stage mmmu
cd /content/MMD
if [ ! -f "results/$NAME/summary.json" ]; then
  bash scripts/run_mmmu_eval.sh --preset best --min_pixels 4014080 --batch_items 50 \
      --model_path /content/H/merged/T-sft --out "outputs/$NAME.jsonl" || { echo "FAILED mmmu"; exit 1; }
fi
backup

stage compare
r="best_r2u"; l="base"
[ -f results/h_sft_s3407/summary.json ] && r="$r h_sft_s3407" && l="$l,h_sft"
python scripts/compare_runs.py --labels "$l,t_sft" $r "$NAME"
echo "T done ($(date))"
