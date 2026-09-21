# MMMU-val Baseline Evaluation Report — Qwen3-VL-4B-Instruct

- **팀명**: _(기입)_
- **팀원**: _(기입)_
- **작성일**: 2026-09-21
- **재현 커맨드**: `bash scripts/run_mmmu_eval.sh --model_path Qwen/Qwen3-VL-4B-Instruct --data_root "$HF_DATASETS_CACHE"`

---

## 1. 환경 / 재현성

| 항목 | 값 |
|---|---|
| 모델 checkpoint | `Qwen/Qwen3-VL-4B-Instruct` (ebb281ec70b05090aa6165b016eac8ec08e71b17), bf16, downloaded with `snapshot_download(revision=...)` |
| 데이터 | `MMMU/MMMU` (98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68), `validation`, 30 configs loaded separately, 900 items, no filtering |
| 추론 백엔드 | vLLM 0.29.0 offline `LLM.generate` (torch 2.13.0+cu130, transformers 5.16.1, datasets 5.0.1, Python 3.13.15). Chosen because the official recipe sets `presence_penalty=1.5`, which `transformers.generate()` does not implement, and because batched generation finishes 900 items in well under an hour. |
| 사용 GPU | NVIDIA A100-SXM4-40GB (Google Colab), driver 580.82.07, CUDA 13.0 |
| 실측 peak VRAM | 35.6 GiB (36,470 MiB): maximum of `nvidia-smi memory.used`, sampled every 2 s. This is device-level and includes the KV cache vLLM preallocates at `gpu_memory_utilization=0.90`. vLLM's start-up log for this configuration reports 8.91 GiB for weights plus non-torch memory; the remainder is KV cache (24.57 GiB available). |
| 총 소요 시간 | 2,196 s (36.6 min) for 900 items, end to end on a fresh VM, including model and dataset download and engine start-up |
| 의존성 | [`requirements.txt`](../requirements.txt); Colab-specific torchaudio fix in [`README.md`](../README.md) |
| 실행 커맨드 | `bash scripts/run_mmmu_eval.sh --model_path <HF repo id or local checkpoint dir> --data_root <HF datasets cache dir>` |

The command runs generation and then `scripts/check_raw.py`, which fails unless the output holds
exactly 900 unique ids, 30 per subject, with every field of [`docs/raw_schema.md`](../docs/raw_schema.md).
Output: [`outputs/raw.jsonl`](../outputs/raw.jsonl); run record: [`outputs/run_meta.json`](../outputs/run_meta.json)
(versions, GPU, effective parameters, prompt hash, source hash `429fea80a5d6c769`, timing, VRAM).
To re-evaluate a fine-tuned checkpoint, only `--model_path` changes.

## 2. 프롬프트

**실제 모델에 들어간 프롬프트 전문** (변수 부분은 `{}`로 표시). Single user turn, no system
prompt, rendered with the model's own chat template:

Multiple-choice (847 items):

```
<|im_start|>user
{question}

(A) {option_A}
(B) {option_B}
...


Answer with the option's letter from the given choices directly.<|im_end|>
<|im_start|>assistant
```

Open (53 items):

```
<|im_start|>user
{question}

Answer the question using a single word or phrase.<|im_end|>
<|im_start|>assistant
```

Each `<image N>` token inside `{question}` or an option is replaced, at its first occurrence, by
the image from column `image_N` (`<|vision_start|><|image_pad|><|vision_end|>` in the rendered
string). A repeated reference stays as the literal text. Four items carry an image the text
never references (`validation_Agriculture_26`, `validation_Materials_15`, `validation_Pharmacy_4`,
`validation_Pharmacy_22`); those images are placed at the start of the turn. The exact string for
every item is stored in the `prompt` field of `outputs/raw.jsonl`.

- **출처**: Text format copied verbatim from the MMMU official repository,
  [MMMU-Benchmark/MMMU @ 268471d](https://github.com/MMMU-Benchmark/MMMU/tree/268471d0d488258990025331c7528359c324aa25):
  `mmmu/configs/llava1.5.yaml` (the two format strings, empty task instruction) and
  `construct_prompt` in `mmmu/utils/data_utils.py` (the `(A) option` rendering; the doubled blank
  line before "Answer with" comes from that code). Image interleaving is our own: the official
  script passes only `image_1`, which would drop images from the 43 multi-image items.
- **선택 이유**: It is the benchmark authors' own prompt, so the official parser's assumptions
  (a bare letter or a short phrase) hold, and it needs no tuning that would have to be repeated
  identically after fine-tuning.

## 3. 생성(Decoding) 설정

### 3.1 Sampling recipe

| 파라미터 | 값 |
|---|---|
| `do_sample` | True (`greedy='false'`) |
| `temperature` | 0.7 |
| `top_p` | 0.8 |
| `top_k` | 20 |
| `repetition_penalty` | 1.0 |
| `presence_penalty` | 1.5 |
| `seed` | 42; each request uses `int(sha256(f"42:{id}").hexdigest()[:8], 16)` |

- **출처**: The model card at the pinned revision, section "Generation Hyperparameters" > "VL":
  <https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/blob/ebb281ec70b05090aa6165b016eac8ec08e71b17/README.md#generation-hyperparameters>.
  All six values are used unchanged. The seed is ours (the card gives none); deriving it per item
  makes a sample's draw independent of batch composition and of restarts.

### 3.2 생성 예산 / 이미지 해상도

| 파라미터 | 값 |
|---|---|
| `max_new_tokens` | 8192 (the card's `out_seq_length` is 16384) |
| 이미지 해상도 처리 (`min_pixels`/`max_pixels` 등) | `max_pixels = 1,003,520` (1280×28×28): an image with more pixels is downscaled before the processor (bicubic, aspect ratio kept, never upscaled). `min_pixels`: processor default. Transparent images are flattened onto white. |

**선택 근거**

`max_new_tokens`. Despite "answer directly", the model often writes out its working. In the
baseline run the median response is 7 tokens and 602 of 900 are at most 16 tokens, but 170 exceed
1024 tokens. An earlier full run at 1024 cut off 166 of 900 responses (18.4%; count read from that run's log, its output file was lost with the Colab VM), so 1024 was rejected. At
8192, 99 responses that needed more than 1024 tokens finished normally (30 of them needed more
than 4096). 71 (7.9%) still hit the limit, all multiple-choice and concentrated in
Architecture_and_Engineering (14), Materials (11), Energy_and_Power (10) and
Mechanical_Engineering (9). At least 26 of the 71 end in a verbatim loop (their last 150
characters occur three or more times); the rest are long reconsiderations without a final answer.
We did not go to 16384: it would double the worst-case KV cache per sequence, and a looping
response does not end with more budget. These 71 stay in the denominator.

`max_pixels`. 98 of the 982 validation images exceed 1.0 M pixels (largest 5.46 M). Capping at
1.0 M bounds one image at about 980 visual tokens, so the largest item (5 images) stays near 5k
prompt tokens and fits `max_model_len=16384` together with 8192 output tokens. 546 of the 982
images have an alpha channel or palette; a plain RGB conversion can turn a transparent background
black, so they are composited onto white, as `qwen-vl-utils` does.

RTX 4090 (24 GB): not run. The model needs about 9 GiB and the settings above do not depend on
the 40 GB card; on 24 GB the same command leaves roughly 10 GiB of KV cache, which lowers
concurrency and speed but not the outputs' settings.

## 4. 채점(파싱) 방식

- 사용한 파서/로직: _(자체 구현 / 차용 도구명 + 링크)_
- 동작 방식 요약: _(예: 어떤 순서로 규칙을 적용하는지, 실패 시 fallback은 무엇인지)_

## 5. 결과

| No. | Subject | Data Num | Acc |
|---|---|---|---|
| 1 | Accounting | 30 | |
| 2 | Agriculture | 30 | |
| 3 | Architecture_and_Engineering | 30 | |
| 4 | Art | 30 | |
| 5 | Art_Theory | 30 | |
| 6 | Basic_Medical_Science | 30 | |
| 7 | Biology | 30 | |
| 8 | Chemistry | 30 | |
| 9 | Clinical_Medicine | 30 | |
| 10 | Computer_Science | 30 | |
| 11 | Design | 30 | |
| 12 | Diagnostics_and_Laboratory_Medicine | 30 | |
| 13 | Economics | 30 | |
| 14 | Electronics | 30 | |
| 15 | Energy_and_Power | 30 | |
| 16 | Finance | 30 | |
| 17 | Geography | 30 | |
| 18 | History | 30 | |
| 19 | Literature | 30 | |
| 20 | Manage | 30 | |
| 21 | Marketing | 30 | |
| 22 | Materials | 30 | |
| 23 | Math | 30 | |
| 24 | Mechanical_Engineering | 30 | |
| 25 | Music | 30 | |
| 26 | Pharmacy | 30 | |
| 27 | Physics | 30 | |
| 28 | Psychology | 30 | |
| 29 | Public_Health | 30 | |
| 30 | Sociology | 30 | |
| | **Overall (macro avg)** | **900** | |

계산식: `Overall = mean(30개 과목 accuracy)` _(다른 방식을 썼다면 명시)_

## 6. 공식 수치와의 비교

| | Overall (MMMU val) |
|---|---|
| 공식 (Qwen3-VL Technical Report) | 67.4 |
| 우리 재현 결과 | |
| 차이 (Δ) | |

## 7. 격차 분석

_(1000 char 이내로 작성 - Official 성능과 차이가 발생하는지, 그렇다면 그 이유를 서술. 길게 쓴다고 credit이 느는 게
아니라, 근거의 질이 핵심입니다. 레포트는 짧을수록 좋습니다.)_


## 8. 기타 특이사항 / 한계 (Optional)

_(재현 중 겪은 문제, 시간 관계상 못 해본 것, 다음에 시도해보고 싶은 것 등. 자유롭게)_
