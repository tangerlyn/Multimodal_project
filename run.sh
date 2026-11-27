#!/usr/bin/env bash
# One entry point for the H experiment. Run from anywhere; paths are relative to this folder.
#
#   bash run.sh setup                         install vLLM 0.29 + PEFT (one NVIDIA GPU, CUDA 12.x/13.x, Python 3.10+)
#   bash run.sh check                         quick checks: imports, GPU, data files, a 1-item MMMU val run
#   bash run.sh reproduce                     released H-SFT LoRA -> merge -> MMMU val -> retry on truncation
#
#   bash run.sh merge [LORA_DIR] [OUT_DIR]    LoRA + base weights -> full checkpoint for vLLM
#   bash run.sh eval NAME MODEL [DATASET]     MODEL = base | checkpoint dir; DATASET = val (default) | pro10 | pro_vision | test
#   bash run.sh retry NAME MODEL [DATASET]    regenerate NAME's truncated answers (seeds 1001, 1002) -> NAME_retry
#   bash run.sh submission NAME               results/NAME/scored.jsonl -> submissions/NAME.json (MMMU test server)
#
#   bash run.sh build                         data/gen.jsonl -> data/build2 (training pairs; CPU, seconds)
#   bash run.sh virl                          download ViRL39K images to data/virl (needed for training / generation)
#   bash run.sh train [sft|dpo]               data/build2 -> work/runs/H-<mode>/lora   (one A100, about 40 min)
#   bash run.sh generate                      from scratch: ViRL39K -> work/data/items.jsonl -> gen.jsonl (GPU, hours)
#
# Environment: WORK (default ./work) holds merged checkpoints and training runs; VIRL_DIR (default ./data/virl);
# HF_HOME for the Hugging Face cache. Evaluation outputs go to MMD/outputs and scores to MMD/results.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
WORK=${WORK:-$ROOT/work}
export VIRL_DIR=${VIRL_DIR:-$ROOT/data/virl}
export PYTHONUNBUFFERED=1
mkdir -p "$WORK"

# The evaluation recipe behind every reported number (best_r2u): Qwen's MMMU settings + CoT sentence + answer line +
# fallback, images fixed at 4,014,080 pixels, cap 32,768 tokens, seed 3407 with per-item seeds.
EVAL_ARGS=(--max_new_tokens 32768 --max_model_len 65536 --min_pixels 4014080 --max_pixels 4014080
           --prompt_style vlmevalkit --image_position first --cot qwen --answer_line --fallback --batch_items 100)

die() { echo "error: $*" >&2; exit 1; }

patch_eval() {  # MMMU-Pro, MMMU test, no-image mode and chunked writing; idempotent, inactive unless selected
  (cd "$ROOT/MMD" &&
   python "$ROOT/H/step07_patch_noimg.py" src/eval_mmmu.py >/dev/null &&
   python "$ROOT/H/step08_patch_pro.py" "$ROOT/MMD" >/dev/null &&
   python "$ROOT/H/step09_patch_test.py" "$ROOT/MMD" >/dev/null &&
   python "$ROOT/H/step10_patch_chunk.py" src/eval_mmmu.py >/dev/null) || die "patching MMD/src failed"
}

dataset_env() {  # DATASET -> value of MMMU_DATASET
  case "$1" in
    val) echo "" ;;
    pro10|pro_vision) echo "$1" ;;
    test) echo "mmmu_test" ;;
    *) die "unknown dataset '$1' (val | pro10 | pro_vision | test)" ;;
  esac
}

model_args() {  # MODEL -> evaluator flags
  if [ "$1" = base ]; then echo ""; else [ -d "$1" ] || die "no checkpoint at $1 (run: bash run.sh merge)"; echo "--model_path $(cd "$1" && pwd)"; fi
}

generate() {  # NAME MODEL DATASET SEED [extra evaluator flags...]
  local name=$1 model=$2 ds=$3 seed=$4; shift 4
  local env; env=$(dataset_env "$ds")
  # shellcheck disable=SC2046
  (cd "$ROOT/MMD" && MMMU_DATASET=$env python src/eval_mmmu.py "${EVAL_ARGS[@]}" --seed "$seed" \
      $(model_args "$model") "$@" --out "outputs/$name.jsonl")
}

score() {  # NAME DATASET
  local partial=""
  [ "$2" = val ] || partial="--partial"   # MMMU-Pro / test do not have the 30 x 30 val layout
  (cd "$ROOT/MMD" && python scripts/score_mmmu.py "outputs/$1.jsonl" $partial)
}

cmd=${1:-help}; shift || true
case "$cmd" in
  setup)
    python -m pip install -r "$ROOT/MMD/requirements.txt"
    python -m pip install --force-reinstall --no-deps torchaudio==2.11.0   # vLLM's torch 2.13 needs the matching torchaudio
    python -m pip install peft
    python -m pip uninstall -y torchao 2>/dev/null || true                # an incompatible torchao breaks transformers import
    python -c "import torch, transformers, vllm, peft; print('torch', torch.__version__, '| transformers', transformers.__version__, '| vllm', vllm.__version__, '| peft', peft.__version__)"
    ;;

  check)
    python -c "import torch; assert torch.cuda.is_available(), 'no CUDA GPU'; print('GPU:', torch.cuda.get_device_name(0))"
    for f in data/gen.jsonl data/build2/pairs_train.jsonl release/H-sft-lora/adapter_model.safetensors MMD/src/eval_mmmu.py; do
      [ -s "$ROOT/$f" ] && echo "ok  $f" || die "missing $f"
    done
    patch_eval
    generate check_base base val 3407 --limit 1
    score check_base val || true
    echo "check passed: the evaluator ran one item per subject with the base model"
    ;;

  merge)
    lora=${1:-$ROOT/release/H-sft-lora}; out=${2:-$WORK/merged/H-sft}
    [ -f "$out/merge_info.json" ] && { echo "already merged: $out"; exit 0; }
    rm -rf "$out"; python "$ROOT/H/step06_merge_model.py" --lora "$lora" --out "$out"
    ;;

  eval)
    [ $# -ge 2 ] || die "usage: bash run.sh eval NAME MODEL [DATASET]"
    name=$1 model=$2 ds=${3:-val}
    patch_eval
    if [ ! -f "$ROOT/MMD/results/$name/summary.json" ]; then
      generate "$name" "$model" "$ds" 3407   # resumes from outputs/NAME.jsonl if it exists
      score "$name" "$ds"
    fi
    cat "$ROOT/MMD/results/$name/results.md" 2>/dev/null | head -20 || true
    ;;

  retry)
    [ $# -ge 2 ] || die "usage: bash run.sh retry NAME MODEL [DATASET]"
    name=$1 model=$2 ds=${3:-val}; out=${name}_retry
    [ -f "$ROOT/MMD/outputs/$name.jsonl" ] || die "run 'bash run.sh eval $name $model $ds' first"
    patch_eval
    cd "$ROOT/MMD"
    python "$ROOT/H/step11_retry.py" select "outputs/$name.jsonl" "outputs/${out}_r1.ids"
    for k in 1 2; do
      ids="outputs/${out}_r$k.ids"
      n=$(grep -c . "$ids" || true)
      echo "retry round $k (seed $((1000 + k))): $n items"
      if [ "$n" -gt 0 ]; then
        generate "${out}_r$k" "$model" "$ds" $((1000 + k)) --ids_file "$ROOT/MMD/$ids"
      fi
      if [ "$k" = 1 ]; then
        if [ -f "outputs/${out}_r1.jsonl" ]; then
          python "$ROOT/H/step11_retry.py" select "outputs/${out}_r1.jsonl" "outputs/${out}_r2.ids"
        else
          : > "outputs/${out}_r2.ids"
        fi
      fi
    done
    python "$ROOT/H/step11_retry.py" merge "outputs/$name.jsonl" "outputs/$out"
    score "$out" "$ds"
    ;;

  submission)
    [ $# -ge 1 ] || die "usage: bash run.sh submission NAME"
    mkdir -p "$ROOT/MMD/submissions"
    python "$ROOT/H/step12_make_submission.py" "$ROOT/MMD/results/$1/scored.jsonl" "$ROOT/MMD/submissions/$1.json"
    ;;

  build)
    python "$ROOT/H/step04_build_pairs.py" --gen "$ROOT/data/gen.jsonl" --out "$ROOT/data/build2" --max_tokens 1500
    ;;

  virl)
    [ -f "$VIRL_DIR/.ready" ] && { echo "ViRL39K already at $VIRL_DIR"; exit 0; }
    mkdir -p "$VIRL_DIR"
    python -c "import sys; from huggingface_hub import snapshot_download; snapshot_download('TIGER-Lab/ViRL39K', repo_type='dataset', local_dir=sys.argv[1])" "$VIRL_DIR"
    (cd "$VIRL_DIR" && unzip -q -o images.zip) && touch "$VIRL_DIR/.ready"
    ;;

  train)
    mode=${1:-sft}; out=$WORK/runs/H-$mode
    [ -d "$VIRL_DIR/images" ] || die "ViRL39K images not found at $VIRL_DIR (run: bash run.sh virl)"
    [ -f "$out/training_run.json" ] && { echo "already trained: $out"; exit 0; }
    python "$ROOT/H/step05_train.py" --mode "$mode" --lr 5e-5 \
        --data "$ROOT/data/build2/pairs_train.jsonl" --val "$ROOT/data/build2/pairs_val.jsonl" --out "$out"
    echo "next: bash run.sh merge $out/lora $WORK/merged/H-$mode"
    ;;

  generate)
    [ -d "$VIRL_DIR/images" ] || die "ViRL39K images not found at $VIRL_DIR (run: bash run.sh virl)"
    mkdir -p "$WORK/data"
    python "$ROOT/H/step01_pick_questions.py" --virl "$VIRL_DIR" --out "$WORK/data/items.jsonl"
    python "$ROOT/H/step02_sample_answers.py" --items "$WORK/data/items.jsonl" --out "$WORK/data/gen.jsonl" --n 8 --limit 3000
    python "$ROOT/H/step04_build_pairs.py" --gen "$WORK/data/gen.jsonl" --out "$WORK/data/build2" --max_tokens 1500
    echo "built $WORK/data/build2; sampling differs slightly across GPUs, so counts can differ from data/build2"
    ;;

  reproduce)
    bash "$ROOT/run.sh" merge
    bash "$ROOT/run.sh" eval h_sft_s3407 "$WORK/merged/H-sft" val
    bash "$ROOT/run.sh" retry h_sft_s3407 "$WORK/merged/H-sft" val
    echo "reported: h_sft_s3407 66.22% (47 truncated), h_sft_s3407_retry 66.44% (MMMU val 900, Pipeline)"
    ;;

  help|-h|--help) sed -n '2,22p' "$ROOT/run.sh" ;;
  *) die "unknown command '$cmd' (bash run.sh help)" ;;
esac
