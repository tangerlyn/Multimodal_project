#!/usr/bin/env bash
# Output-cap check for the MMMU-Pro runs: they used max_new_tokens 16384, Qwen's MMMU script uses 32768.
# For each run, the items that were truncated at 16384 WITHOUT a verbatim loop are generated again at 32768
# (same per-item seed, so the first 16384 tokens repeat up to numerics), merged into a copy of the run and scored.
# Loop items are kept as they are: on MMMU val (cap 32768) no item that looped at 16384 finished later.
#   nohup bash run_pro3_colab.sh > /content/pro3.log 2>&1 &
# Starts when /content/pro.log has "PRO done". Runs in priority order; each merged run is <name>_m32k.
set -uo pipefail
export PYTHONUNBUFFERED=1
BACKUP=${BACKUP:-/content/drive/MyDrive/mmdl_hd}
stage() { echo "=== $1 ($(date)) ==="; }
backup() {
  [ -d /content/drive/MyDrive ] || return 0
  rsync -a /content/MMD/outputs /content/MMD/results "$BACKUP/" 2>/dev/null
  cp /content/pro3.log "$BACKUP/pro3.log" 2>/dev/null
  date > "$BACKUP/last_backup.txt"
}

stage wait
until grep -aq 'PRO done' /content/pro.log 2>/dev/null; do sleep 60; done
sleep 30
( while true; do sleep 300; backup; done ) &
BK=$!
trap 'backup; kill $BK 2>/dev/null' EXIT
cd /content/MMD
rm -f /content/cap_fin.txt

rerun() {  # name, dataset, extra args
  local name=$1 dataset=$2; shift 2
  stage "cap32k $name"
  [ -f "results/$name/scored.jsonl" ] || { echo "SKIP $name: not scored"; return 0; }
  if [ ! -f "results/${name}_m32k/summary.json" ]; then
    python - "$name" <<'EOF' || return 1
import json, sys
name = sys.argv[1]
rows = [json.loads(l) for l in open(f"results/{name}/scored.jsonl")]
ids = [r["id"] for r in rows if r["finish_reason"] == "length" and r["loop_chars_removed"] == 0]
open(f"outputs/{name}_t32k.ids", "w").write("\n".join(ids) + "\n")
print(f"{name}: truncated {sum(r['finish_reason'] == 'length' for r in rows)}, without loop {len(ids)}")
EOF
    n=$(grep -c . "outputs/${name}_t32k.ids")
    if [ "$n" -gt 0 ]; then
      MMMU_DATASET=$dataset python src/eval_mmmu.py \
          --max_new_tokens 32768 --max_model_len 65536 --min_pixels 4014080 --max_pixels 4014080 \
          --prompt_style vlmevalkit --image_position first --seed 3407 --cot qwen --answer_line --fallback \
          --batch_items 100 --ids_file "outputs/${name}_t32k.ids" "$@" --out "outputs/${name}_t32k.jsonl"
      [ "$(wc -l < "outputs/${name}_t32k.jsonl")" -eq "$n" ] || { echo "FAILED $name"; return 1; }
    fi
    python - "$name" <<'EOF' || return 1
import json, os, sys
name = sys.argv[1]
def load(p): return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []
new = {r["id"]: r for r in load(f"outputs/{name}_t32k.jsonl")}
new_fb = load(f"outputs/{name}_t32k.fallback.jsonl")
with open(f"outputs/{name}_m32k.jsonl", "w") as f:
    for r in load(f"outputs/{name}.jsonl"):
        f.write(json.dumps(new.get(r["id"], r), ensure_ascii=False) + "\n")
with open(f"outputs/{name}_m32k.fallback.jsonl", "w") as f:
    for r in [r for r in load(f"outputs/{name}.fallback.jsonl") if r["id"] not in new] + new_fb:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
fin = sum(r["finish_reason"] == "stop" for r in new.values())
print(f"{name}: regenerated {len(new)}, finished within 32768: {fin}, still truncated: {len(new) - fin}")
open("/content/cap_fin.txt", "a").write(f"{name} {len(new)} {fin}\n")
EOF
    python scripts/score_mmmu.py "outputs/${name}_m32k.jsonl" --partial || { echo "FAILED score $name"; return 1; }
  fi
  python scripts/compare_runs.py --labels cap16k,cap32k_merged "$name" "${name}_m32k" 2>&1 | head -n 16
  backup
}
SFT=/content/H/merged/H-sft
rerun pro10_base_s3407       pro10
rerun pro10_h_sft_s3407      pro10      --model_path $SFT
# Stop rule (set before the run): if at most STOP_MAX of the standard-setting items finish within 32768,
# the cap does not matter and the vision re-runs are skipped.
fin=$(cat outputs/pro10_*_t32k.jsonl 2>/dev/null | grep -c '"finish_reason": "stop"')
echo "standard runs: $fin regenerated items finished within 32768 (stop rule: <= ${STOP_MAX:-3} skips vision)"
if [ "$fin" -gt "${STOP_MAX:-3}" ]; then
  rerun prov_base_s3407        pro_vision
  rerun prov_h_sft_s3407       pro_vision --model_path $SFT
else
  echo "SKIP vision re-runs by the stop rule"
fi

stage compare
r=""; l=""
for n in pro10_base:base_std10 pro10_h_sft:sft_std10 prov_base:base_vision prov_h_sft:sft_vision; do
  [ -f "results/${n%%:*}_s3407_m32k/summary.json" ] && r="$r ${n%%:*}_s3407_m32k" && l="$l,${n##*:}"
done
python scripts/compare_runs.py --labels ${l#,} $r
echo "CAP done ($(date))"
