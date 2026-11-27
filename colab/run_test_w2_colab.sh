#!/usr/bin/env bash
# MMMU test, second worker on another Colab GPU session (A100 or H100) next to the A100 "train" session (run_final_colab.sh).
# Same model and settings as the train session (H-SFT, cap 32768, seed 3407, per-item seeds, best_r2u recipe).
# The train session goes through the 30 subjects from the front; this worker takes whole subjects from the back
# (Sociology, Public_Health, ...) and stops before a subject the train session has already started (read from its
# Drive backup). Per-item seeds make each answer independent of which worker or batch produced it.
#   uploads in /content: g1_bundle.tgz, sft_bundle.tgz, step07_patch_noimg.py, step08_patch_pro.py, step09_patch_test.py, step10_patch_chunk.py
#   nohup bash run_test_w2_colab.sh > /content/w2.log 2>&1 &
# Output: outputs/test_h_sft_s3407_w2.jsonl (+ .fallback.jsonl). Backs up only *_w2* files and w2.log to
# Drive, so it never overwrites what the train session writes there. A rerun on a new VM resumes from Drive.
set -uo pipefail
export PYTHONUNBUFFERED=1
BACKUP=${BACKUP:-/content/drive/MyDrive/mmdl_hd}
OUT=outputs/test_h_sft_s3407_w2.jsonl
TRAIN=$BACKUP/outputs/test_h_sft_s3407.jsonl
stage() { echo "=== $1 ($(date)) ==="; }
backup() {
  [ -d /content/drive/MyDrive ] || return 0
  rsync -a --include '*_w2*' --exclude '*' /content/MMD/outputs/ "$BACKUP/outputs/" 2>/dev/null
  cp /content/w2.log "$BACKUP/w2.log" 2>/dev/null
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
until [ -d /content/drive/MyDrive ] || [ $t -ge "${WAIT_DRIVE:-3600}" ]; do sleep 20; t=$((t + 20)); done
[ -f "$TRAIN" ] || { echo "train output not found at $TRAIN"; exit 1; }
rsync -a --include '*_w2*' --exclude '*' "$BACKUP/outputs/" /content/MMD/outputs/ 2>/dev/null
echo "restored: $(cat /content/MMD/$OUT 2>/dev/null | wc -l) items"
( while true; do sleep 300; backup; done ) &
BK=$!
trap 'backup; kill $BK 2>/dev/null' EXIT

stage patch
cd /content/MMD
python /content/step07_patch_noimg.py src/eval_mmmu.py || exit 1
python /content/step08_patch_pro.py /content/MMD || exit 1
python /content/step09_patch_test.py /content/MMD || exit 1
python /content/step10_patch_chunk.py src/eval_mmmu.py || exit 1

# MMMU test items per subject (MMMU/MMMU test split)
declare -A N=([Accounting]=380 [Agriculture]=287 [Architecture_and_Engineering]=551 [Art]=231 [Art_Theory]=429
  [Basic_Medical_Science]=326 [Biology]=345 [Chemistry]=603 [Clinical_Medicine]=325 [Computer_Science]=371
  [Design]=169 [Diagnostics_and_Laboratory_Medicine]=162 [Economics]=267 [Electronics]=256 [Energy_and_Power]=432
  [Finance]=355 [Geography]=565 [History]=278 [Literature]=112 [Manage]=245 [Marketing]=181 [Materials]=458
  [Math]=505 [Mechanical_Engineering]=429 [Music]=334 [Pharmacy]=430 [Physics]=408 [Psychology]=305
  [Public_Health]=509 [Sociology]=252)
REV=$(python -c "import sys; sys.path.insert(0, 'src'); from data import SUBJECTS; print(' '.join(reversed(SUBJECTS)))")
count() { grep -c "\"id\": \"test_$2_[0-9]*\"" "$1" 2>/dev/null; }

for s in $REV; do
  mine=$(count "$OUT" "$s"); mine=${mine:-0}
  [ "$mine" -ge "${N[$s]}" ] && { echo "done already: $s"; continue; }
  theirs=$(count "$TRAIN" "$s"); theirs=${theirs:-0}
  if [ "$mine" -eq 0 ] && [ "$theirs" -gt 0 ]; then
    echo "MEET: train session has started $s ($theirs items), stopping here"; break
  fi
  stage "subject $s (${N[$s]} items, $mine done)"
  MMMU_DATASET=mmmu_test python src/eval_mmmu.py \
      --max_new_tokens 32768 --max_model_len 65536 --min_pixels 4014080 --max_pixels 4014080 \
      --prompt_style vlmevalkit --image_position first --seed 3407 --cot qwen --answer_line --fallback \
      --model_path /content/H/merged/H-sft --batch_items 100 --subjects "$s" --out "$OUT"
  mine=$(count "$OUT" "$s")
  [ "${mine:-0}" -ge "${N[$s]}" ] || { echo "FAILED $s: ${mine:-0} of ${N[$s]}"; exit 1; }
  echo "SUBJECT DONE $s $(date +%s)"
  backup
done
echo "W2 done ($(date)): $(wc -l < "$OUT") items"
