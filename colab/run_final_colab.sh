#!/usr/bin/env bash
# Final evaluation on a fresh Colab VM.
#  1. Finish the MMMU-Pro retry runs (only base vision is left; finished runs are restored from Drive and skipped).
#  2. MMMU test (10,500 items) with the final model: H-SFT, best_r2u settings (cap 32768, seed 3407), then
#     retry-on-truncation (seeds 1001, 1002; a finished retry replaces the original), then the answer file for
#     the official evaluation server (test answers are hidden, so there is no local score).
#   TEST_CAP (default 32768) sets max_new_tokens for the MMMU test generation.
#   uploads in /content: step10_patch_chunk.py, g1_bundle.tgz, sft_bundle.tgz, step07_patch_noimg.py, step08_patch_pro.py, step09_patch_test.py,
#                        run_retry_pro_colab.sh, step12_make_submission.py
#   nohup bash run_final_colab.sh > /content/final.log 2>&1 &
# Every stage resumes from Drive ($BACKUP), so on a new VM: mount Drive, upload the same files, run again.
set -uo pipefail
export PYTHONUNBUFFERED=1
BACKUP=${BACKUP:-/content/drive/MyDrive/mmdl_hd}
NAME=test_h_sft_s3407
OUT=test_h_sft_retry
stage() { echo "=== $1 ($(date)) ==="; }
backup() {
  [ -d /content/drive/MyDrive ] || return 0
  rsync -a /content/MMD/outputs /content/MMD/results /content/MMD/submissions "$BACKUP/" 2>/dev/null
  cp /content/final.log "$BACKUP/final.log" 2>/dev/null
  date > "$BACKUP/last_backup.txt"
}

stage unpack
mkdir -p /content/MMD /content/H
[ -d /content/MMD/src ] || tar xzf /content/g1_bundle.tgz -C /content/MMD || exit 1
[ -d /content/H/runs/H-sft/lora ] || tar xzf /content/sft_bundle.tgz -C /content || exit 1
mkdir -p /content/MMD/outputs /content/MMD/results /content/MMD/submissions

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
until [ -d /content/drive/MyDrive ] || [ $t -ge "${WAIT_DRIVE:-1800}" ]; do sleep 20; t=$((t + 20)); done
[ -d "$BACKUP/outputs" ] || { echo "Drive backup not found at $BACKUP"; exit 1; }
rsync -a "$BACKUP/outputs" "$BACKUP/results" /content/MMD/ && echo "restored from $BACKUP"
[ -d "$BACKUP/submissions" ] && rsync -a "$BACKUP/submissions" /content/MMD/
( while true; do sleep 300; backup; done ) &
BK=$!
trap 'backup; kill $BK 2>/dev/null' EXIT

stage patch
cd /content/MMD
python /content/step07_patch_noimg.py src/eval_mmmu.py || exit 1
python /content/step08_patch_pro.py /content/MMD || exit 1
python /content/step09_patch_test.py /content/MMD || exit 1
python /content/step10_patch_chunk.py src/eval_mmmu.py || exit 1

stage "MMMU-Pro retry (base vision)"
NO_WAIT=1 bash /content/run_retry_pro_colab.sh > /content/retrypro.log 2>&1
grep -aE 'replaced|FAILED|RETRYPRO done' /content/retrypro.log | tail -n 6
backup

gen() {  # out file, seed, extra args
  local out=$1 seed=$2; shift 2
  MMMU_DATASET=mmmu_test python src/eval_mmmu.py \
      --max_new_tokens "${TEST_CAP:-32768}" --max_model_len 65536 --min_pixels 4014080 --max_pixels 4014080 \
      --prompt_style vlmevalkit --image_position first --seed "$seed" --cot qwen --answer_line --fallback \
      --model_path /content/H/merged/H-sft --batch_items 100 "$@" --out "$out"
}

stage "MMMU test: $NAME"
if [ "$(cat outputs/$NAME.jsonl 2>/dev/null | wc -l)" -lt 10500 ]; then
  gen "outputs/$NAME.jsonl" 3407
fi
[ "$(wc -l < "outputs/$NAME.jsonl")" -eq 10500 ] || { echo "FAILED $NAME: $(wc -l < outputs/$NAME.jsonl) of 10500"; exit 1; }
backup

stage "MMMU test retry"
python - "$NAME" "$OUT" <<'EOF' || exit 1
import json, sys
name, out = sys.argv[1:3]
ids = [json.loads(l)["id"] for l in open(f"outputs/{name}.jsonl") if json.loads(l)["finish_reason"] == "length"]
open(f"outputs/{out}_r1.ids", "w").write("".join(i + "\n" for i in ids)); print("truncated:", len(ids))
EOF
for k in 1 2; do
  ids="outputs/${OUT}_r$k.ids"; g="outputs/${OUT}_r$k.jsonl"; n=$(grep -c . "$ids")
  echo "--- round $k: seed $((1000 + k)), $n items ($(date))"
  if [ "$n" -gt 0 ] && [ "$(cat "$g" 2>/dev/null | wc -l)" -lt "$n" ]; then
    gen "$g" $((1000 + k)) --ids_file "$ids"
    [ "$(wc -l < "$g")" -eq "$n" ] || { echo "FAILED retry $k"; exit 1; }
    backup
  fi
  [ "$k" -eq 1 ] && python - "$OUT" <<'EOF'
import json, os, sys
out = sys.argv[1]; p = f"outputs/{out}_r1.jsonl"
ids = [json.loads(l)["id"] for l in open(p) if json.loads(l)["finish_reason"] == "length"] if os.path.exists(p) else []
open(f"outputs/{out}_r2.ids", "w").write("".join(i + "\n" for i in ids)); print("still truncated after round 1:", len(ids))
EOF
done
python - "$NAME" "$OUT" <<'EOF' || exit 1
import json, os, sys
base, out = sys.argv[1:3]
def load(p): return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []
rep, fb = {}, {}
for k in (1, 2):
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

stage "score (predictions only) and answer files"
for r in "$NAME" "$OUT"; do
  python scripts/score_mmmu.py "outputs/$r.jsonl" --partial > /dev/null || { echo "FAILED parse $r"; exit 1; }
  python /content/step12_make_submission.py "results/$r/scored.jsonl" "submissions/$r.json" || exit 1
done
python - "$NAME" "$OUT" <<'EOF'
import json, sys, collections
for r in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f"results/{r}/scored.jsonl")]
    print(r, "items", len(rows), "truncated", sum(x["finish_reason"] == "length" for x in rows),
          "no prediction", sum(not x.get("pipeline_pred") for x in rows),
          "types", dict(collections.Counter(x["question_type"] for x in rows)))
EOF
backup
echo "FINAL done ($(date))"
