#!/usr/bin/env bash
# Retry-on-truncation on MMMU-Pro (test, 1,730 items), queued behind run_retry_colab.sh.
# Same rule as on MMMU val: items that ended at the output cap are generated again with seed 1001, those still
# truncated once more with seed 1002; a retry replaces the original only if it finished.
# Settings are those of the original MMMU-Pro runs (cap 16384, seed 3407 for the first pass).
# Order: H-SFT standard-10, H-SFT vision, then base standard-10, base vision (the fair comparison).
#   uploads needed: /content/step07_patch_noimg.py, /content/step08_patch_pro.py; Drive mounted with the original runs in $BACKUP
#   nohup bash run_retry_pro_colab.sh > /content/retrypro.log 2>&1 &
set -uo pipefail
export PYTHONUNBUFFERED=1
BACKUP=${BACKUP:-/content/drive/MyDrive/mmdl_hd}
stage() { echo "=== $1 ($(date)) ==="; }
backup() {
  [ -d /content/drive/MyDrive ] || return 0
  rsync -a /content/MMD/outputs /content/MMD/results "$BACKUP/" 2>/dev/null
  cp /content/retrypro.log "$BACKUP/retrypro.log" 2>/dev/null
  date > "$BACKUP/last_backup.txt"
}

stage wait
[ -n "${NO_WAIT:-}" ] || { until grep -aqE '^EXIT' /content/retry.log 2>/dev/null; do sleep 30; done; sleep 20; }
( while true; do sleep 300; backup; done ) &
BK=$!
trap 'backup; kill $BK 2>/dev/null' EXIT

stage restore
cd /content/MMD
rsync -a --include 'pro*' --include 'prov*' --exclude '*' "$BACKUP/outputs/" outputs/ || exit 1
rsync -a --include 'pro*/***' --include 'prov*/***' --exclude '*' "$BACKUP/results/" results/ || exit 1
wc -l outputs/pro10_base_s3407.jsonl outputs/pro10_h_sft_s3407.jsonl outputs/prov_base_s3407.jsonl outputs/prov_h_sft_s3407.jsonl || exit 1
python /content/step07_patch_noimg.py src/eval_mmmu.py || exit 1
python /content/step08_patch_pro.py /content/MMD || exit 1

retry() {  # original run name, dataset, extra args (model)
  local base=$1 dataset=$2; shift 2
  local out="${base%_s3407}_retry"
  stage "retry $base"
  if [ ! -f "results/$out/summary.json" ]; then
    python - "$base" "$out" <<'EOF' || return 1
import json, sys
base, out = sys.argv[1:3]
ids = [json.loads(l)["id"] for l in open(f"outputs/{base}.jsonl") if json.loads(l)["finish_reason"] == "length"]
open(f"outputs/{out}_r1.ids", "w").write("".join(i + "\n" for i in ids)); print(f"{base}: truncated {len(ids)}")
EOF
    for k in 1 2; do
      local ids="outputs/${out}_r$k.ids" gen="outputs/${out}_r$k.jsonl" n seed=$((1000 + k))
      n=$(grep -c . "$ids")
      echo "--- round $k: seed $seed, $n items ($(date))"
      if [ "$n" -gt 0 ] && ! { [ -f "$gen" ] && [ "$(wc -l < "$gen")" -eq "$n" ]; }; then
        MMMU_DATASET=$dataset python src/eval_mmmu.py \
            --max_new_tokens 16384 --max_model_len 65536 --min_pixels 4014080 --max_pixels 4014080 \
            --prompt_style vlmevalkit --image_position first --seed "$seed" --cot qwen --answer_line --fallback \
            --batch_items 200 --ids_file "$ids" "$@" --out "$gen"
        [ "$(wc -l < "$gen")" -eq "$n" ] || { echo "FAILED $base round $k"; return 1; }
        backup
      fi
      [ "$k" -eq 1 ] && python - "$out" <<'EOF'
import json, os, sys
out = sys.argv[1]; p = f"outputs/{out}_r1.jsonl"
ids = [json.loads(l)["id"] for l in open(p) if json.loads(l)["finish_reason"] == "length"] if os.path.exists(p) else []
open(f"outputs/{out}_r2.ids", "w").write("".join(i + "\n" for i in ids)); print("still truncated after round 1:", len(ids))
EOF
    done
    python - "$base" "$out" <<'EOF' || return 1
import json, os, sys
base, out = sys.argv[1:3]
def load(p): return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []
rep, fb = {}, {}
for k in (1, 2):   # first finished retry wins
    f = {r["id"]: r for r in load(f"outputs/{out}_r{k}.fallback.jsonl")}
    for r in load(f"outputs/{out}_r{k}.jsonl"):
        if r["finish_reason"] != "length" and r["id"] not in rep:
            rep[r["id"]] = r
            if r["id"] in f: fb[r["id"]] = f[r["id"]]
rows = load(f"outputs/{base}.jsonl")
with open(f"outputs/{out}.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(rep.get(r["id"], r), ensure_ascii=False) + "\n")
with open(f"outputs/{out}.fallback.jsonl", "w") as f:
    for r in [r for r in load(f"outputs/{base}.fallback.jsonl") if r["id"] not in rep] + list(fb.values()):
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
n = sum(r["finish_reason"] == "length" for r in rows)
print(f"{base}: truncated {n}, replaced by a finished retry {len(rep)}, still truncated {n - len(rep)}")
EOF
    python scripts/score_mmmu.py "outputs/$out.jsonl" --partial || { echo "FAILED score $out"; return 1; }
  fi
  python scripts/compare_runs.py --labels original,with_retry "$base" "$out" 2>&1 | head -n 16
  backup
}
SFT=/content/H/merged/H-sft
retry pro10_h_sft_s3407 pro10      --model_path $SFT
retry prov_h_sft_s3407  pro_vision --model_path $SFT
retry pro10_base_s3407  pro10
retry prov_base_s3407   pro_vision

stage compare
r=""; l=""
for n in pro10_base:base_std pro10_h_sft:sft_std prov_base:base_vis prov_h_sft:sft_vis; do
  [ -f "results/${n%%:*}_retry/summary.json" ] && r="$r ${n%%:*}_retry" && l="$l,${n##*:}"
done
python scripts/compare_runs.py --labels ${l#,} $r
echo "RETRYPRO done ($(date))"
