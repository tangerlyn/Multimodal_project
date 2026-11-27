#!/usr/bin/env bash
# Retry-on-truncation for H-SFT on MMMU val (Colab, same settings as best_r2u / h_sft_s3407).
# Items of h_sft_s3407 that ended at the output cap are generated again with seed 1001; those still truncated are
# generated once more with seed 1002. A retry replaces the original only if it finished (same rule as the base
# result 66.67%, which used the full runs g1_s1001 / g1_s1002). Everything else stays as in h_sft_s3407.
#   uploads needed: /content/sft_bundle.tgz; /content/MMD with outputs/results of h_sft_s3407 and best_r2u
#   nohup bash run_retry_colab.sh > /content/retry.log 2>&1 &
set -uo pipefail
export PYTHONUNBUFFERED=1
BACKUP=${BACKUP:-/content/drive/MyDrive/mmdl_hd}
BASE=h_sft_s3407
OUT=h_sft_retry
stage() { echo "=== $1 ($(date)) ==="; }
backup() {
  [ -d /content/drive/MyDrive ] || return 0
  rsync -a /content/MMD/outputs /content/MMD/results "$BACKUP/" 2>/dev/null
  cp /content/retry.log "$BACKUP/retry.log" 2>/dev/null
  date > "$BACKUP/last_backup.txt"
}
trap backup EXIT

stage merge
[ -d /content/H/runs/H-sft/lora ] || tar xzf /content/sft_bundle.tgz -C /content || exit 1
cd /content/H
[ -f merged/H-sft/merge_info.json ] || { rm -rf merged/H-sft; python step06_merge_model.py --lora runs/H-sft/lora --out merged/H-sft || exit 1; }

cd /content/MMD
[ -f "outputs/$BASE.jsonl" ] && [ -f "results/$BASE/scored.jsonl" ] || { echo "missing $BASE outputs/results"; exit 1; }
python - "$BASE" <<'EOF' || exit 1
import json, sys
ids = [json.loads(l)["id"] for l in open(f"outputs/{sys.argv[1]}.jsonl") if json.loads(l)["finish_reason"] == "length"]
open("outputs/retry1.ids", "w").write("\n".join(ids) + "\n"); print("truncated in the original run:", len(ids))
EOF

gen() {  # round, seed
  local ids="outputs/retry$1.ids" out="outputs/${OUT}_r$1.jsonl" n
  n=$(grep -c . "$ids")
  stage "retry $1 (seed $2, $n items)"
  [ "$n" -gt 0 ] || return 0
  [ -f "$out" ] && [ "$(wc -l < "$out")" -eq "$n" ] && return 0
  bash scripts/run_mmmu_eval.sh --preset best --min_pixels 4014080 --batch_items 100 \
      --model_path /content/H/merged/H-sft --seed "$2" --ids_file "$ids" --no_score --out "$out"
  [ "$(wc -l < "$out")" -eq "$n" ] || { echo "FAILED retry $1"; return 1; }
  backup
}
gen 1 1001 || exit 1
python - "$OUT" <<'EOF' || exit 1
import json, sys
rows = [json.loads(l) for l in open(f"outputs/{sys.argv[1]}_r1.jsonl")]
ids = [r["id"] for r in rows if r["finish_reason"] == "length"]
open("outputs/retry2.ids", "w").write("".join(i + "\n" for i in ids)); print("still truncated after retry 1:", len(ids), "of", len(rows))
EOF
gen 2 1002 || exit 1

stage merge-results
python - "$BASE" "$OUT" <<'EOF' || exit 1
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
print(f"truncated {n}, replaced by a finished retry {len(rep)}, still truncated {n - len(rep)}")
EOF
python scripts/score_mmmu.py "outputs/$OUT.jsonl" || { echo "FAILED score"; exit 1; }
backup

stage compare
python scripts/compare_runs.py --labels base,h_sft,h_sft_retry best_r2u "$BASE" "$OUT"
echo "RETRY done ($(date))"
