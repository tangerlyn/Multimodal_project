#!/usr/bin/env bash
# MMMU-Pro output-cap check (method A), on a second Colab session next to the MMMU test run.
# The MMMU-Pro runs used max_new_tokens 16384. After retry-on-truncation, some items still end at the cap. Only those
# items can change with a larger cap, so each is generated again at 32768 with its original per-item seed (3407),
# which follows the same path as the 16384 generation and only lets it run longer. A finished 32768 answer replaces
# the truncated one; the rest stay. Result per run: <run>_c32 (scored), compared with <run> and base vs H-SFT.
#   uploads in /content: g1_bundle.tgz, sft_bundle.tgz, step07_patch_noimg.py, step08_patch_pro.py, step10_patch_chunk.py
#   nohup bash run_pro32_colab.sh > /content/pro32.log 2>&1 &
# Backs up only *_c32* files to Drive, so it never overwrites what the MMMU test session writes there.
set -uo pipefail
export PYTHONUNBUFFERED=1
BACKUP=${BACKUP:-/content/drive/MyDrive/mmdl_hd}
stage() { echo "=== $1 ($(date)) ==="; }
backup() {
  [ -d /content/drive/MyDrive ] || return 0
  rsync -a --include '*_c32*' --exclude '*' /content/MMD/outputs/ "$BACKUP/outputs/" 2>/dev/null
  rsync -a --include '*_c32/***' --exclude '*' /content/MMD/results/ "$BACKUP/results/" 2>/dev/null
  cp /content/pro32.log "$BACKUP/pro32.log" 2>/dev/null
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
[ -d "$BACKUP/outputs" ] || { echo "Drive backup not found at $BACKUP"; exit 1; }
rsync -a --include 'pro*' --include 'prov*' --exclude '*' "$BACKUP/outputs/" /content/MMD/outputs/ || exit 1
rsync -a --include 'pro*/***' --include 'prov*/***' --exclude '*' "$BACKUP/results/" /content/MMD/results/ || exit 1
echo "restored from $BACKUP"; wc -l /content/MMD/outputs/pro*_retry.jsonl /content/MMD/outputs/prov*_retry.jsonl
( while true; do sleep 300; backup; done ) &
BK=$!
trap 'backup; kill $BK 2>/dev/null' EXIT

stage patch
cd /content/MMD
python /content/step07_patch_noimg.py src/eval_mmmu.py || exit 1
python /content/step08_patch_pro.py /content/MMD || exit 1
python /content/step10_patch_chunk.py src/eval_mmmu.py || exit 1

cap32() {  # retry run name, dataset, extra args (model)
  local run=$1 dataset=$2; shift 2
  local out="${run}_c32"
  stage "cap 32768: $run"
  if [ ! -f "results/$out/summary.json" ]; then
    python - "$run" <<'EOF' || return 1
import json, sys
run = sys.argv[1]
ids = [json.loads(l)["id"] for l in open(f"outputs/{run}.jsonl") if json.loads(l)["finish_reason"] == "length"]
open(f"outputs/{run}_c32.ids", "w").write("".join(i + "\n" for i in ids)); print(f"{run}: still truncated {len(ids)}")
EOF
    local ids="outputs/${run}_c32.ids" gen="outputs/${run}_c32gen.jsonl" n
    n=$(grep -c . "$ids")
    if [ "$n" -gt 0 ] && [ "$(cat "$gen" 2>/dev/null | wc -l)" -lt "$n" ]; then
      MMMU_DATASET=$dataset python src/eval_mmmu.py \
          --max_new_tokens 32768 --max_model_len 65536 --min_pixels 4014080 --max_pixels 4014080 \
          --prompt_style vlmevalkit --image_position first --seed 3407 --cot qwen --answer_line --fallback \
          --batch_items 100 --ids_file "$ids" "$@" --out "$gen"
      [ "$(wc -l < "$gen")" -eq "$n" ] || { echo "FAILED $run"; return 1; }
      backup
    fi
    python - "$run" <<'EOF' || return 1
import json, os, sys
run = sys.argv[1]
def load(p): return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []
new = {r["id"]: r for r in load(f"outputs/{run}_c32gen.jsonl") if r["finish_reason"] != "length"}
newfb = {r["id"]: r for r in load(f"outputs/{run}_c32gen.fallback.jsonl") if r["id"] in new}
rows = load(f"outputs/{run}.jsonl")
with open(f"outputs/{run}_c32.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(new.get(r["id"], r), ensure_ascii=False) + "\n")
with open(f"outputs/{run}_c32.fallback.jsonl", "w") as f:
    for r in [r for r in load(f"outputs/{run}.fallback.jsonl") if r["id"] not in new] + list(newfb.values()):
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
n = sum(r["finish_reason"] == "length" for r in rows)
print(f"{run}: truncated at 16384 {n}, finished within 32768 {len(new)}, still truncated {n - len(new)}")
EOF
    python scripts/score_mmmu.py "outputs/$out.jsonl" --partial > /dev/null || { echo "FAILED score $out"; return 1; }
  fi
  python - "$run" <<'EOF'
import json, sys
from math import comb
run = sys.argv[1]
ld = lambda n: {json.loads(l)["id"]: json.loads(l) for l in open(f"results/{n}/scored.jsonl")}
a, b = ld(run), ld(run + "_c32")
x = sum(1 for i in a if a[i]["pipeline_correct"] and not b[i]["pipeline_correct"]); y = sum(1 for i in a if b[i]["pipeline_correct"] and not a[i]["pipeline_correct"])
ca, cb = sum(bool(v["pipeline_correct"]) for v in a.values()), sum(bool(v["pipeline_correct"]) for v in b.values())
print(f"RESULT {run}: 16384 {ca} ({100*ca/len(a):.2f}%) -> 32768 {cb} ({100*cb/len(b):.2f}%), lost {x} gained {y}")
EOF
  backup
}
SFT=/content/H/merged/H-sft
cap32 pro10_base_retry  pro10
cap32 pro10_h_sft_retry pro10      --model_path $SFT
cap32 prov_base_retry   pro_vision
cap32 prov_h_sft_retry  pro_vision --model_path $SFT

stage compare
python - <<'EOF'
import json
from math import comb
ld = lambda n: {json.loads(l)["id"]: json.loads(l) for l in open(f"results/{n}/scored.jsonl")}
def mc(a, b):
    x = sum(1 for i in a if a[i]["pipeline_correct"] and not b[i]["pipeline_correct"]); y = sum(1 for i in a if b[i]["pipeline_correct"] and not a[i]["pipeline_correct"]); n = x + y
    return x, y, round(min(1, 2*sum(comb(n, k) for k in range(min(x, y)+1))/2**n), 4)
for s, d in (("standard", "pro10"), ("vision", "prov")):
    A, B = ld(f"{d}_base_retry_c32"), ld(f"{d}_h_sft_retry_c32")
    ca, cb = sum(bool(v["pipeline_correct"]) for v in A.values()), sum(bool(v["pipeline_correct"]) for v in B.values())
    print(f"COMPARE {s} at 32768: base {ca} ({100*ca/1730:.2f}%), H-SFT {cb} ({100*cb/1730:.2f}%), McNemar {mc(A, B)}")
EOF
python scripts/compare_runs.py --labels base_std,sft_std,base_vis,sft_vis \
    pro10_base_retry_c32 pro10_h_sft_retry_c32 prov_base_retry_c32 prov_h_sft_retry_c32
echo "PRO32 done ($(date))"
