#!/usr/bin/env bash
# Experiment H on Colab: nohup bash run_h.sh > /content/h.log 2>&1 &
# Every stage resumes / skips when its output exists.
set -uo pipefail
cd /content/H
LR=${LR:-5e-5}
stage() { echo "=== $1 ($(date)) ==="; }

stage prep;  [ -f build/items.info.json ] || python step01_pick_questions.py --out build/items.jsonl || exit 1
stage gen;   python step02_sample_answers.py --items build/items.jsonl --out build/gen.jsonl --n 8 || exit 1
stage build; python step04_build_pairs.py --gen build/gen.jsonl --out build || exit 1
for M in sft dpo; do
  stage "train $M"
  [ -f runs/H-$M/training_run.json ] || python step05_train.py --mode $M --lr $LR \
      --data build/pairs_train.jsonl --val build/pairs_val.jsonl --out runs/H-$M || exit 1
  stage "merge $M"
  [ -f merged/H-$M/merge_info.json ] || { rm -rf merged/H-$M; python step06_merge_model.py --lora runs/H-$M/lora --out merged/H-$M || exit 1; }
done
cd /content/MMD
for M in dpo sft; do
  stage "mmmu $M"
  bash scripts/run_mmmu_eval.sh --preset best --min_pixels 4014080 --batch_items 50 \
      --model_path /content/H/merged/H-$M --out outputs/h_${M}_s3407.jsonl
done
python scripts/vote_mmmu.py results/best_r2u --out results/h_base >/dev/null 2>&1
echo "H done ($(date))"
