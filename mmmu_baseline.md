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

- **공식 파서**: MMMU의 [`eval_utils.py`](https://github.com/MMMU-Benchmark/MMMU/blob/268471d0d488258990025331c7528359c324aa25/mmmu/utils/eval_utils.py)를 고정 버전으로 사용했다. 원본은 [`scripts/eval_utils_official.py`](scripts/eval_utils_official.py)에 보관했다.
- **추가한 코드**: [`scripts/score_mmmu.py`](scripts/score_mmmu.py)는 저장된 `outputs/raw.jsonl`을 읽어 답 추출, 정답 비교, 과목별 집계와 결과 저장을 수행한다. 모델을 다시 실행하거나 원본 `response`를 수정하지 않는다.
- **공식 함수의 변경점**: 객관식에서 후보를 찾지 못했을 때 `random.choice(all_choices)`로 찍던 한 줄만 `None`으로 바꿨다. 스크립트는 공식 원본의 SHA256을 확인한 뒤 수정본 `eval_utils_no_random.py`를 출력 디렉터리에 생성한다. 나머지 공식 답 추출·비교 규칙은 유지했다.

**객관식**

1. 응답의 `(A)`, `(B)` 같은 괄호 문자를 찾는다.
2. 괄호 후보가 없으면 공백으로 구분된 ` A `, ` B ` 등을 찾는다.
3. 위 후보가 없고 응답이 5단어를 초과하면 선택지 내용과 대소문자를 무시한 문자열 매칭을 시도한다.
4. 후보가 여러 개이면 해당 매칭 방식에서 가장 뒤에 등장한 후보를 선택한다. 후보가 없으면 `parsed_pred=null`, `parse_failed=True`, `correct=False`로 기록한다.
5. 추출한 선택지 문자와 정답 문자가 일치하면 정답이다. 보기 나열이나 수식·도형 표지가 답으로 잡힐 수 있으며, 추출 성공이 최종 답의 완결을 의미하지는 않는다.

**주관식**

공식 `parse_open_response`로 핵심 구절과 숫자 후보를 추출하고 `eval_open`으로 비교한다. 문자열은 소문자로 정규화하고, 숫자는 예측과 정답 양쪽을 소수점 둘째 자리로 반올림한다. 정규화된 숫자의 일치 또는 공식 문자열 포함 규칙으로 정답을 판정한다. 정답 필드에 복수 허용 답이 문자열 목록으로 저장된 경우 목록으로 복원해 비교한다. 빈 응답·후보 없음은 추출 실패로 처리하고, 후보 목록은 저장 시 일관된 순서로 정렬한다.

**채점 재현** — 저장소 루트에서 Python 3.11 이상과 `uv`를 사용한다. 필요한 패키지 버전은 스크립트의 인라인 의존성에 고정되어 있다.

```bash
uv run --script scripts/score_mmmu.py outputs/raw.jsonl results
```

`results/`에 `scored.jsonl`, `scored.csv`, `errors.csv`, `by_subject.csv`, `by_question_type.csv`, `by_finish_reason.csv`, `by_num_images.csv`, `summary.json`과 수정된 파서를 생성한다. 생성 길이 제한에 걸린 응답도 동일한 규칙으로 채점하고 900문항의 분모에 포함한다. 수동 정답 보정이나 별도의 허용 오차는 추가하지 않았다.

## 5. 결과

Acc는 정답률(%)이며 소수점 둘째 자리까지 표시했다.

| No. | Subject | Data Num | Acc |
|---|---|---|---|
| 1 | Accounting | 30 | 73.33 |
| 2 | Agriculture | 30 | 60.00 |
| 3 | Architecture_and_Engineering | 30 | 36.67 |
| 4 | Art | 30 | 63.33 |
| 5 | Art_Theory | 30 | 73.33 |
| 6 | Basic_Medical_Science | 30 | 66.67 |
| 7 | Biology | 30 | 60.00 |
| 8 | Chemistry | 30 | 30.00 |
| 9 | Clinical_Medicine | 30 | 70.00 |
| 10 | Computer_Science | 30 | 53.33 |
| 11 | Design | 30 | 73.33 |
| 12 | Diagnostics_and_Laboratory_Medicine | 30 | 43.33 |
| 13 | Economics | 30 | 70.00 |
| 14 | Electronics | 30 | 40.00 |
| 15 | Energy_and_Power | 30 | 56.67 |
| 16 | Finance | 30 | 60.00 |
| 17 | Geography | 30 | 53.33 |
| 18 | History | 30 | 70.00 |
| 19 | Literature | 30 | 76.67 |
| 20 | Manage | 30 | 36.67 |
| 21 | Marketing | 30 | 86.67 |
| 22 | Materials | 30 | 53.33 |
| 23 | Math | 30 | 60.00 |
| 24 | Mechanical_Engineering | 30 | 50.00 |
| 25 | Music | 30 | 33.33 |
| 26 | Pharmacy | 30 | 63.33 |
| 27 | Physics | 30 | 56.67 |
| 28 | Psychology | 30 | 73.33 |
| 29 | Public_Health | 30 | 76.67 |
| 30 | Sociology | 30 | 63.33 |
| | **Overall (macro avg)** | **900** | **59.44** |

계산식: `Overall = mean(30개 과목 accuracy)`. 모든 과목이 30문항이므로 전체 문항 기준 정확도 `535 / 900 × 100 = 59.4444...%`와 같다.

| 문항 유형 | 문항 수 | 정답 수 | 정답률 (%) |
|---|---:|---:|---:|
| 객관식 | 847 | 529 | 62.46 |
| 주관식 | 53 | 6 | 11.32 |
| 전체 | 900 | 535 | 59.44 |

오답은 365문항이다. 답 추출 실패는 6문항이며 모두 `finish_reason=length`였다. 길이 제한에 도달한 71문항도 제외하지 않고 채점했으며, 이 중 19문항은 추출 답이 정답 키와 일치하고 52문항은 오답이었다. 이 수치는 풀이의 완결 여부를 별도로 보정하지 않은 채점 결과다.

## 6. 공식 수치와의 비교

| | Overall accuracy (%) |
|---|---:|
| 공식 (Qwen3-VL Technical Report의 MMMU 보고 수치) | 67.40 |
| 우리 재현 결과 (MMMU validation, 900문항) | 59.44 |
| 차이 (Δ, 우리 결과 − 공식 수치) | **−7.96%p** |

차이는 반올림 전 정확도로 계산했다: `535 / 900 × 100 − 67.4 = −7.9556...%p`.
공식 수치와 이번 실행의 평가 split·생성 설정이 동일하다고 확인한 것은 아니므로, 위 표는 보고 수치의 비교다. 원인 해석은 7절에서 다룬다.

## 7. 격차 분석

_(1000 char 이내로 작성 - Official 성능과 차이가 발생하는지, 그렇다면 그 이유를 서술. 길게 쓴다고 credit이 느는 게
아니라, 근거의 질이 핵심입니다. 레포트는 짧을수록 좋습니다.)_


## 8. 기타 특이사항 / 한계 (Optional)

_(재현 중 겪은 문제, 시간 관계상 못 해본 것, 다음에 시도해보고 싶은 것 등. 자유롭게)_
