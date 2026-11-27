#!/usr/bin/env bash
# Inference-setting experiment on Colab, queued behind run_next_colab.sh. No training, base model only.
# Same settings as best_r2u (vLLM, preset best, min_pixels 4014080, seed 3407) except the sampling values given
# in MMMU_SAMPLING, so the run isolates that one change. Default: repetition_penalty 1.05 (best_r2u used 1.0),
# aimed at the verbatim loops that end in truncated answers.
#   nohup bash run_inf_colab.sh > /content/inf.log 2>&1 &
# Starts when /content/hs.log has its EXIT line (whatever the outcome; this run does not depend on it).
#   SAMPLING='{"repetition_penalty": 1.05}'   NAME=base_rp105_s3407
set -uo pipefail
export PYTHONUNBUFFERED=1
BACKUP=${BACKUP:-/content/drive/MyDrive/mmdl_hd}
SAMPLING=${SAMPLING:-'{"repetition_penalty": 1.05}'}
NAME=${NAME:-base_rp105_s3407}
stage() { echo "=== $1 ($(date)) ==="; }

backup() {
  [ -d /content/drive/MyDrive ] || return 0
  mkdir -p "$BACKUP"
  rsync -a /content/MMD/outputs /content/MMD/results "$BACKUP/" 2>/dev/null
  cp /content/inf.log "$BACKUP/inf.log" 2>/dev/null
  date > "$BACKUP/last_backup.txt"
}

stage wait
until grep -aq '^EXIT' /content/hs.log 2>/dev/null; do sleep 60; done
sleep 30
( while true; do sleep 300; backup; done ) &
BK=$!
trap 'backup; kill $BK 2>/dev/null' EXIT

stage patch
cd /content/MMD
python - <<'EOF' || exit 1
p = "src/eval_mmmu.py"
s = open(p).read()
mark = 'SAMPLING_RECIPE.update(json.loads(os.environ.get("MMMU_SAMPLING", "{}")))'
if mark not in s:
    i = s.index("SAMPLING_RECIPE = {")
    j = s.index("}\n", i) + 2
    s = s[:j] + "# Optional override for inference-setting experiments; recorded in run_meta.json with the rest.\n" + mark + "\n" + s[j:]
    open(p, "w").write(s)
print("patched" if mark in open(p).read() else "NOT patched")
EOF
MMMU_SAMPLING="$SAMPLING" python -c "
import sys; sys.argv=['x']; sys.path.insert(0,'src')
import importlib.util, os
src=open('src/eval_mmmu.py').read()
i=src.index('SAMPLING_RECIPE = {'); j=src.index('MMMU_SAMPLING', i); j=src.index('\n', j)
import json; ns={'json':json,'os':os}; exec(src[i:j], ns); print('sampling recipe:', ns['SAMPLING_RECIPE'])
" || exit 1

stage "mmmu $NAME"
if [ ! -f "results/$NAME/summary.json" ]; then
  MMMU_SAMPLING="$SAMPLING" bash scripts/run_mmmu_eval.sh --preset best --min_pixels 4014080 --batch_items 50 \
      --out "outputs/$NAME.jsonl" || { echo "FAILED mmmu"; exit 1; }
fi
backup
grep -a -A8 '"sampling"' "outputs/$NAME.run_meta.json" 2>/dev/null | head -n 10

stage compare
python scripts/compare_runs.py --labels base,base_rp105 best_r2u "$NAME"
echo "INF done ($(date))"
