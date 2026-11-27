#!/usr/bin/env bash
# Experiment H, second pass: re-sample token-limit casualties at 2048 tokens, rebuild (samples <= 1500 tokens),
# then train / merge / evaluate.  nohup bash run_h2.sh >> /content/h.log 2>&1 &
set -uo pipefail
cd /content/H
LR=${LR:-5e-5}
stage() { echo "=== $1 ($(date)) ==="; }

stage "redo select"; python step03_redo_truncated.py select --gen build/gen.jsonl --out build/items_redo.jsonl || exit 1
stage "redo gen";    python step02_sample_answers.py --items build/items_redo.jsonl --out build/gen_redo.jsonl --n 8 --max_tokens 2048 || exit 1
stage "redo merge";  python step03_redo_truncated.py merge --gen build/gen.jsonl --redo build/gen_redo.jsonl --out build/gen2.jsonl || exit 1
stage build2;        python step04_build_pairs.py --gen build/gen2.jsonl --out build2 --max_tokens 1500 || exit 1
for M in sft dpo; do
  stage "train $M"
  [ -f runs/H-$M/training_run.json ] || python step05_train.py --mode $M --lr $LR \
      --data build2/pairs_train.jsonl --val build2/pairs_val.jsonl --out runs/H-$M || exit 1
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
