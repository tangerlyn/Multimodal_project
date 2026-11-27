#!/usr/bin/env bash
# Experiment H end to end on a fresh Colab A100 (code in /content/H, MMMU evaluator in /content/MMD).
#   nohup bash run_h_all.sh > /content/h.log 2>&1 &
# setup -> ViRL39K download -> prep -> gen (1024 tokens) -> run_h2.sh (re-sample at 2048, build2, train SFT + DPO,
# merge, MMMU eval of both). Every stage skips when its output exists, so a rerun resumes.
# Backup: when Google Drive is mounted, everything needed to resume (build/, build2/, runs/, MMMU outputs/results,
# the log) is copied to $BACKUP every 5 minutes and at exit, and copied back at start on a new VM.
# Merged checkpoints are not backed up (9 GB each, rebuilt from the LoRA in minutes).
set -uo pipefail
export PYTHONUNBUFFERED=1
BACKUP=${BACKUP:-/content/drive/MyDrive/mmdl_h}
mkdir -p /content/H/build /content/H/build2 /content/H/runs /content/MMD/outputs /content/MMD/results
cd /content/H
stage() { echo "=== $1 ($(date)) ==="; }

backup() {
  [ -d /content/drive/MyDrive ] || return 0
  mkdir -p "$BACKUP/H" "$BACKUP/MMD"
  rsync -a /content/H/build /content/H/build2 /content/H/runs "$BACKUP/H/" 2>/dev/null
  rsync -a /content/MMD/outputs /content/MMD/results "$BACKUP/MMD/" 2>/dev/null
  cp /content/h.log "$BACKUP/h.log" 2>/dev/null
  date > "$BACKUP/last_backup.txt"
}

stage restore
if [ -d "$BACKUP/H" ]; then
  rsync -a "$BACKUP/H/" /content/H/ && rsync -a "$BACKUP/MMD/" /content/MMD/ && echo "restored from $BACKUP"
  ls build build2 runs /content/MMD/outputs 2>/dev/null | head -40
elif [ -d /content/drive/MyDrive ]; then
  echo "no backup yet at $BACKUP (fresh run)"
else
  echo "WARNING: Google Drive is not mounted, nothing will be backed up"
fi
( while true; do sleep 300; backup; done ) &
BK=$!
trap 'backup; kill $BK 2>/dev/null' EXIT

stage setup
if ! python -c "import vllm" 2>/dev/null; then
  pip install -q -r /content/MMD/requirements.txt > /content/h_pip.log 2>&1 || { echo "pip failed, see /content/h_pip.log"; exit 1; }
  pip install -q --force-reinstall --no-deps torchaudio==2.11.0 >> /content/h_pip.log 2>&1
fi
python -c "import peft" 2>/dev/null || pip install -q peft >> /content/h_pip.log 2>&1
python -c "import torch, transformers, vllm, peft; print('torch', torch.__version__, 'transformers', transformers.__version__, 'vllm', vllm.__version__, 'peft', peft.__version__)" || exit 1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

stage virl
if [ ! -f /content/virl/.ready ]; then
  mkdir -p /content/virl
  python -c "from huggingface_hub import snapshot_download; snapshot_download('TIGER-Lab/ViRL39K', repo_type='dataset', local_dir='/content/virl')" || exit 1
  (cd /content/virl && unzip -q -o images.zip) || exit 1
  touch /content/virl/.ready
fi
ls /content/virl | head

stage prep;  [ -f build/items.info.json ] || python step01_pick_questions.py --out build/items.jsonl || exit 1
[ -s build/items.jsonl ] || { echo "prep wrote no items"; exit 1; }
stage gen;   python step02_sample_answers.py --items build/items.jsonl --out build/gen.jsonl --n 8 || exit 1
backup

bash run_h2.sh
