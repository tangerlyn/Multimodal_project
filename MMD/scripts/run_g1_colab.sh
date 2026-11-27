#!/usr/bin/env bash
# Experiment G1 (self-consistency, no training) on a Colab A100. Run from the project root:
#   nohup bash scripts/run_g1_colab.sh all > /content/g1.log 2>&1 &
# Stages (each resumes: run_mmmu_eval.sh skips ids already in its output file):
#   setup   pip install + torchaudio fix
#   stage1  seeds S1,S2 on all 900 items (same config as outputs/best_r2u.jsonl, the seed-3407 vote)
#   vote1   3-vote majority; writes the non-unanimous ids
#   stage2  seeds S3,S4 only on those ids (adaptive self-consistency: unanimous items keep their answer)
#   vote    final 5-vote majority -> results/g1_vote/summary.json
# Every finished seed is copied to $BACKUP (Drive) when it is mounted.
set -uo pipefail
cd "$(dirname "$0")/.."
BACKUP=${BACKUP:-/content/drive/MyDrive/mmmu_g1}
S1=${S1:-1001}; S2=${S2:-1002}; S3=${S3:-1003}; S4=${S4:-1004}
R2U=(--preset best --min_pixels 4014080 --batch_items 50)

backup() {
  [ -d /content/drive/MyDrive ] || return 0
  mkdir -p "$BACKUP" && cp -r outputs results "$BACKUP"/ 2>/dev/null && echo "backup -> $BACKUP"
}

seed_run() {  # seed [extra args]
  local s=$1; shift
  echo "=== seed $s $* ($(date)) ==="
  bash scripts/run_mmmu_eval.sh "${R2U[@]}" --seed "$s" --out "outputs/g1_s$s.jsonl" "$@"
  backup
}

ST=${1:-all}
want() { [ "$ST" = all ] || [ "$ST" = "$1" ]; }
case "$ST" in all|setup|stage1|vote1|stage2|vote) ;; *) echo "usage: $0 [all|setup|stage1|vote1|stage2|vote]"; exit 2 ;; esac

if want setup; then
  pip install -q -r requirements.txt && pip install -q --force-reinstall --no-deps torchaudio==2.11.0
  nvidia-smi --query-gpu=name,memory.total --format=csv
fi
if want stage1; then
  seed_run "$S1"; seed_run "$S2"
fi
if want vote1; then
  python scripts/vote_mmmu.py results/best_r2u "results/g1_s$S1" "results/g1_s$S2" \
    --out results/g1_vote3 --emit_ids outputs/g1_stage2_ids.txt
  backup
fi
if want stage2; then
  seed_run "$S3" --ids_file outputs/g1_stage2_ids.txt
  seed_run "$S4" --ids_file outputs/g1_stage2_ids.txt
fi
if want vote; then
  python scripts/vote_mmmu.py results/best_r2u "results/g1_s$S1" "results/g1_s$S2" \
    "results/g1_s$S3" "results/g1_s$S4" --out results/g1_vote
  backup
  echo "G1 done ($(date))"
fi
