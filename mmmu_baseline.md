# MMMU-val Baseline Evaluation Report — Qwen3-VL-4B-Instruct

- **팀명**: _(기입)_
- **팀원**: _(기입)_
- **작성일**: 2026-09-25
- **재현 커맨드**: `bash scripts/run_mmmu_eval.sh --preset best --out outputs/best.jsonl --model_path Qwen/Qwen3-VL-4B-Instruct --data_root "$HF_DATASETS_CACHE"`

제출 실행은 `best` 프리셋(2026-09-23)입니다. 첫 실행(베이스라인, 2026-09-21, 59.44%)과 시드 재현 실행(2026-09-25)은 §8과 [docs/analysis_report.md](../docs/analysis_report.md)에 비교 자료로 남겼습니다.

---

## 1. 환경 / 재현성

| 항목 | 값 |
|---|---|
| 모델 checkpoint | `Qwen/Qwen3-VL-4B-Instruct` (ebb281ec70b05090aa6165b016eac8ec08e71b17), bf16, `snapshot_download(revision=...)`로 고정 |
| 데이터 | `MMMU/MMMU` (98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68), `validation`, 30개 config 각각 로드, 900문항, 필터링 없음 |
| 추론 백엔드 | vLLM 0.29.0 offline `LLM.generate` (torch 2.13.0+cu130, transformers 5.16.1, datasets 5.0.1, Python 3.13.15). 공식 레시피의 `presence_penalty=1.5`를 `transformers.generate()`가 구현하지 않고, 900문항 배치 생성이 빠르기 때문에 선택 |
| 사용 GPU | NVIDIA A100-SXM4-40GB, 40960 MiB, 580.82.07 (Google Colab) |
| 실측 peak VRAM | 36.7 GiB (37,614 MiB): `nvidia-smi memory.used`를 2초마다 샘플링한 장치 기준 최댓값. vLLM이 `gpu_memory_utilization=0.9`로 미리 잡는 KV 캐시 포함 |
| 총 소요 시간 | 9,962 s (2.77시간), 900문항 + fallback 2차 패스, 엔진 기동 포함. 설치와 다운로드 약 5분 별도 |
| 의존성 | [`requirements.txt`](../requirements.txt), 실행 환경 전체는 [`requirements.lock.txt`](../requirements.lock.txt). Colab의 torchaudio 수정은 [`README.md`](../README.md) 참고 |
| 실행 커맨드 | `bash scripts/run_mmmu_eval.sh --preset best --out outputs/best.jsonl` — 생성 → `check_raw.py` 검증 → fallback → `score_mmmu.py` 채점 → `results/best/results.md` 표 생성을 한 번에 수행. 파인튜닝 모델은 `--model_path <dir>` 또는 `--lora_path <adapter>`만 바꿈 |

출력: [`outputs/best.jsonl`](../outputs/best.jsonl) (원본 응답 900건), [`outputs/best.fallback.jsonl`](../outputs/best.fallback.jsonl) (2차 패스 77건), [`outputs/best.run_meta.json`](../outputs/best.run_meta.json) (버전, GPU, 모든 설정, 프롬프트 해시 `8aa03b32c6bba8ae`, 소스 해시 `b633ca2c8995dc02`, 시간, VRAM). 채점 결과: [`results/best/`](../results/best/).

## 2. 프롬프트

**실제 모델에 들어간 프롬프트 전문.** 시스템 프롬프트 없이 user 턴 하나를 모델의 chat template로 렌더링했으며, 아래는 출력 파일 `prompt` 필드의 실제 문자열입니다. 이미지는 `<|vision_start|><|image_pad|><|vision_end|>`로 나타나고 모든 이미지가 텍스트 앞에 열 순서로 놓입니다. 본문의 `<image N>` 토큰은 그대로 둡니다.

Multiple-choice (847 items, 예: `validation_Accounting_1`):

```
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Question: <image 1> Baxter Company has a relevant range of production between 15,000 and 30,000 units. The following cost data represents average variable costs per unit for 25,000 units of production. If 30,000 units are produced, what are the per unit manufacturing overhead costs incurred?
Options:
A. $6
B. $7
C. $8
D. $9
Please select the correct answer from the options above. If you are uncertain or the problem is too complex, make a reasoned guess based on the information provided. Avoid repeating steps indefinitely—provide your best guess even if unsure. Determine whether to think step by step based on the difficulty of the question, considering all relevant information before answering.
End your response with a final line of the form "Answer: X", where X is the option's letter.<|im_end|>
<|im_start|>assistant
```

Open (53 items, 예: `validation_Architecture_and_Engineering_14`):

```
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Question: Using a finite summation, compute the  initial deflection at midspan for the beam in  Figure P8.42. Given: E = 3000 kips/in.2 .  Use 3-ft segments. Assume I = 0.5IG. <image 1> If you are uncertain or the problem is too complex, make a reasoned guess based on the information provided. Avoid repeating steps indefinitely—provide your best guess even if unsure. Determine whether to think step by step based on the difficulty of the question, considering all relevant information before answering.
End your response with a final line of the form "Answer: <your answer>".<|im_end|>
<|im_start|>assistant
```

템플릿으로 쓰면 객관식은 `Question: {question}\nOptions:\nA. {option_A}\nB. {option_B}\n…\nPlease select the correct answer from the options above. {CoT 문장}\nEnd your response with a final line of the form "Answer: X", where X is the option's letter.`, 주관식은 `Question: {question} {CoT 문장}\nEnd your response with a final line of the form "Answer: <your answer>".`입니다.

- **출처**:
  - `Question: / Options: / A. … / Please select the correct answer from the options above.` — Qwen 공식 MMMU 평가 스크립트 [`QwenLM/Qwen3-VL` `evaluation/mmmu/run_mmmu.py`](https://github.com/QwenLM/Qwen3-VL/tree/main/evaluation/mmmu)의 `build_mmmu_prompt`. VLMEvalKit `vlmeval/vlm/qwen3_vl/prompt.py`와 동일. HF 데이터셋에 hint 열이 없어 "Hint:" 줄은 없음.
  - CoT 문장 `If you are uncertain or the problem is too complex, … before answering.` — 같은 스크립트의 `--use-cot` 기본 문장을 글자 그대로 복사.
  - 종결 지시 `End your response with a final line of the form "Answer: X"…` — 직접 설계. 루프 방지용 종결 목표이자 파서 앵커.
  - 이미지 선배치 — Qwen 스크립트 방식.
- **선택 이유**: 베이스라인(MMMU 공식 저장소 프롬프트, "Answer with the option's letter … directly.")에서는 900문항 중 605건이 한 글자 응답이었고 그 정확도가 55.5%인 반면 풀이를 쓴 응답은 80.8%였습니다. 90문항 대조 실험에서 예산·해상도만 바꾼 조건은 베이스라인과 같았고 이 프롬프트가 56.7→63.3%로 올렸으며, 강제 CoT("Think step by step")는 더 길고 느릴 뿐 이득이 없었습니다. 제작사가 평가에 쓴 형식이라 파인튜닝 후에도 그대로 재사용합니다.

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
| `seed` | 3407; 문항별로 `int(sha256(f"3407:{id}").hexdigest()[:8], 16)` |

- **출처**: 모델 카드 "Generation Hyperparameters" > "VL" (<https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/blob/ebb281ec70b05090aa6165b016eac8ec08e71b17/README.md#generation-hyperparameters>)와 Qwen3-VL README "Evaluation Reproduction" > Instruct models (<https://github.com/QwenLM/Qwen3-VL#evaluation-reproduction>). 여섯 값 모두 그대로 사용. 시드 3407은 README 값이며, Qwen의 MMMU 스크립트는 42를 하드코딩해 두 출처가 다릅니다(베이스라인은 42). 문항별 시드 파생은 우리 설계로, 배치 구성과 재시작에 무관하게 같은 문항이 같은 시드를 받게 합니다.

### 3.2 생성 예산 / 이미지 해상도

| 파라미터 | 값 |
|---|---|
| `max_new_tokens` | 32,768 (공식 `out_seq_length`) |
| `max_model_len` | 65,536 |
| 이미지 해상도 처리 | `min_pixels=1,003,520`, `max_pixels=4,014,080` (Qwen 스크립트의 1280·28², 5120·28²). qwen-vl-utils `smart_resize`와 같은 규칙으로 32의 배수, 범위 안으로 조정. 투명 이미지는 흰 배경에 합성 |
| fallback 2차 패스 | `finish_reason=length`이거나 선택지 문자가 없는 객관식 응답에 대해, 응답 끝 반복 구간을 잘라내고 원 프롬프트+응답+`Therefore, the final answer is (`를 이어 greedy(temperature 0) 16토큰 생성. 원본은 수정하지 않고 `best.fallback.jsonl`에 별도 기록 |

**선택 근거**

`max_new_tokens`. 베이스라인 시점에 1,024는 18.4%, 8,192는 7.9%가 잘렸습니다. 공식값 32,768로 올린 best에서도 77건(8.6%)이 잘렸고 그중 60건은 같은 문단을 글자 그대로 반복하는 루프입니다. 42건은 8,192 실행에서도 잘렸으므로 잘림은 예산이 아니라 답을 확정하지 못하는 문항의 특성입니다. 그래서 예산을 더 늘리는 대신 fallback 2차 패스를 두었고, 77건 중 19건을 회수했습니다(12건은 손실). `max_model_len` 65,536은 이미지 5장(최대 약 2만 토큰)과 출력 32,768이 함께 들어가는 값이며 A100 40GB에서 KV 캐시 약 9.6 GiB를 씁니다.

해상도. Qwen 스크립트 값을 그대로 썼습니다. 90문항 대조 실험에서 해상도와 예산만 공식값으로 바꾼 조건은 베이스라인과 정확히 같은 점수(51/90)여서, 해상도 변경 단독의 효과는 관찰되지 않았습니다.

## 4. 채점(파싱) 방식

- **사용한 파서**: MMMU 공식 저장소 [`MMMU-Benchmark/MMMU` 커밋 `268471d`](https://github.com/MMMU-Benchmark/MMMU/blob/268471d0d488258990025331c7528359c324aa25/mmmu/utils/eval_utils.py)의 `eval_utils.py`를 [`scripts/mmmu_official_eval_utils.py`](../scripts/mmmu_official_eval_utils.py)로 바이트 동일하게 포함하고, 로드 시 SHA256(`cc1a89b4…`)을 검증합니다. `parse_multi_choice_response`, `eval_multi_choice`, `parse_open_response`, `eval_open`을 그대로 호출합니다. 채점 스크립트는 [`scripts/score_mmmu.py`](../scripts/score_mmmu.py)입니다.
- **공식 코드의 유일한 변경**: 후보를 찾지 못했을 때 `random.choice(all_choices)`로 찍던 한 줄을 "답 없음(오답)"으로 바꿨습니다. 찍어서 맞힌 것을 점수에 넣지 않기 위함입니다.
- **동작 순서 (객관식)**: ① 응답에서 `(A)` 형태의 괄호 문자를 찾는다 → ② 없으면 공백으로 둘러싸인 ` A `를 찾는다 → ③ 없고 응답이 5단어를 넘으면 보기 본문을 대소문자 무시로 찾는다 → ④ 후보가 여럿이면 가장 뒤에 나온 것을 택한다. **주관식**: 숫자·핵심 구절을 추출해 소수 둘째 자리 반올림, 소문자 정규화, 포함 여부로 비교. 정답이 리스트 문자열이면 `ast.literal_eval`로 복원.
- **두 점수를 냅니다.** *엄격 점수*는 위 공식 규칙만 적용한 값입니다. *파이프라인 점수*(§5의 Acc)는 공식 규칙 앞에 두 단계를 둡니다: ⓐ fallback 파일에 있는 문항은 2차 패스 이어쓰기의 첫 문자를 답으로 사용, ⓑ 그 외에는 마지막 `Answer: X` 줄, 없으면 마지막 `\boxed{X}`를 유효한 보기 문자일 때 채택(주관식은 마지막 `Answer:` 줄만 공식 파서에 넣음). 둘 다 없으면 공식 규칙으로 내려갑니다. 이 순서가 필요한 이유는 공식 규칙의 "가장 뒤 괄호 문자"가 정답을 명시한 뒤 보기를 재나열한 응답에서 실패하기 때문입니다(`validation_Math_30`: `\boxed{C}`가 정답인데 E 추출).
- **효과**: 추출 방법 분포(best) — `Answer:` 줄 822, fallback 문자 68, 공식 규칙 1, 추출 실패 9. 명시 답 규칙 +24/−11, fallback +19/−12, 합계 +20문항(엄격 63.00% → 파이프라인 65.22%). 문항별 판정과 추출 방법은 [`results/best/scored.jsonl`](../results/best/scored.jsonl)의 `pipeline_method`에 기록됩니다.

## 5. 결과

Acc는 파이프라인 점수(%)이며 소수점 둘째 자리까지 표시했습니다. 엄격 점수와 다른 실행은 §8의 표 참고.

| No. | Subject | Data Num | Acc |
|---|---|---|---|
| 1 | Accounting | 30 | 76.67 |
| 2 | Agriculture | 30 | 56.67 |
| 3 | Architecture_and_Engineering | 30 | 43.33 |
| 4 | Art | 30 | 60.00 |
| 5 | Art_Theory | 30 | 80.00 |
| 6 | Basic_Medical_Science | 30 | 73.33 |
| 7 | Biology | 30 | 50.00 |
| 8 | Chemistry | 30 | 46.67 |
| 9 | Clinical_Medicine | 30 | 53.33 |
| 10 | Computer_Science | 30 | 60.00 |
| 11 | Design | 30 | 83.33 |
| 12 | Diagnostics_and_Laboratory_Medicine | 30 | 36.67 |
| 13 | Economics | 30 | 86.67 |
| 14 | Electronics | 30 | 50.00 |
| 15 | Energy_and_Power | 30 | 70.00 |
| 16 | Finance | 30 | 80.00 |
| 17 | Geography | 30 | 60.00 |
| 18 | History | 30 | 73.33 |
| 19 | Literature | 30 | 76.67 |
| 20 | Manage | 30 | 60.00 |
| 21 | Marketing | 30 | 86.67 |
| 22 | Materials | 30 | 53.33 |
| 23 | Math | 30 | 63.33 |
| 24 | Mechanical_Engineering | 30 | 56.67 |
| 25 | Music | 30 | 20.00 |
| 26 | Pharmacy | 30 | 90.00 |
| 27 | Physics | 30 | 80.00 |
| 28 | Psychology | 30 | 80.00 |
| 29 | Public_Health | 30 | 83.33 |
| 30 | Sociology | 30 | 66.67 |
| | **Overall (macro avg)** | **900** | **65.22** |

계산식: `Overall = mean(30개 과목 accuracy)` = 65.2222%. 과목당 문항 수가 30으로 같으므로 전체 정답 비율 `587 / 900 × 100 = 65.2222%`와 정확히 같습니다. 과목당 문항 수가 다른 split에서는 두 값이 달라지므로, 이 보고서의 종합 점수는 macro average임을 명시합니다.

| 문항 유형 | 문항 수 | 정답 수 | 정답률 (%) |
|---|---:|---:|---:|
| 객관식 | 847 | 565 | 66.71 |
| 주관식 | 53 | 22 | 41.51 |
| 전체 | 900 | 587 | 65.22 |

잘림(`finish_reason=length`) 77문항도 제외하지 않고 같은 규칙으로 채점했으며 그중 25문항이 정답 처리됐습니다. 추출 실패 9문항은 오답입니다.

## 6. 공식 수치와의 비교

| | Overall (MMMU val) |
|---|---:|
| 공식 (Qwen3-VL Technical Report) | 67.40 |
| 우리 재현 결과 (best, 파이프라인 점수) | 65.22 |
| 차이 (Δ) | **-2.18%p** |
| 참고: best 엄격 점수 (공식 파서만) | 63.00 (-4.40%p) |
| 참고: 베이스라인 (09-21, 시드 42, 8192 토큰) | 59.56 (-7.84%p) |
| 참고: best 시드 재현 (09-25, 시드 1234) | 65.89 (-1.51%p) |

공식 수치와 이번 실행의 채점 방식(공식은 judge 모델 사용)이 같지 않으므로 보고 수치의 비교입니다.

## 7. 격차 분석

공식 67.4 대비 best는 −2.18%p, 베이스라인은 −7.96%p다. 원인은 셋으로 확인됐다. (1) 프롬프트: 베이스라인의 "바로 답하라" 지시로 900문항 중 605건이 한 글자 응답이었고 정확도 55.5%였다. 풀이를 쓰고 정상 종료한 224건은 80.8%였다. 과목당 3문항 90문항 대조 실험에서 토큰 예산과 해상도만 공식값으로 바꾼 조건은 베이스라인과 동일(51/90)했고, Qwen 공식 CoT 문구를 넣자 57/90으로 올랐다. 격차의 대부분은 지시 문구가 만든 응답 형식 차이다. (2) 수렴 실패: best에서도 77건이 32,768 토큰까지 답을 확정하지 못했고 60건은 같은 문단을 반복하는 루프다. 예산을 4배로 늘려도 잘림 수가 줄지 않았으므로 문항 특성이며, 이 77건을 정상 종료 정확도 68%로 환산하면 약 +3%p로 남은 격차와 같은 크기다. (3) 채점: 공식 파서는 가장 뒤의 괄호 문자를 택해, 정답 C를 boxed로 낸 뒤 보기를 재나열한 응답(Math_30)에서 E를 추출한다. 명시 답 우선 규칙과 fallback으로 +20문항(엄격 63.00→65.22)이고, Qwen 공식 평가는 judge 모델로 답을 뽑으므로 채점 방식 차이도 격차의 일부다. 나머지는 샘플링 노이즈다. 설정을 고정하고 시드만 1234로 바꾼 재실행은 593/900 = 65.89%로 +0.67%p 차이였고, 두 실행 사이 134문항이 뒤집혔으며 과목 단위 표준편차는 2.4문항이었다. −2.18%p는 이 실측 노이즈 범위에 근접한다.

(760자)

## 8. 기타 특이사항 / 한계

**세 실행 비교.** 같은 모델·데이터·샘플링에서 프롬프트, 예산, 해상도, 채점 규칙만 다른 베이스라인과, best 설정에서 시드만 바꾼 재현 실행입니다.

| No. | Subject | Data Num | best (파이프라인) | best (엄격) | 베이스라인 | best 시드 1234 |
|---|---|---|---|---|---|---|
| 1 | Accounting | 30 | 76.67 | 80.00 | 73.33 | 73.33 |
| 2 | Agriculture | 30 | 56.67 | 50.00 | 60.00 | 46.67 |
| 3 | Architecture_and_Engineering | 30 | 43.33 | 40.00 | 36.67 | 50.00 |
| 4 | Art | 30 | 60.00 | 63.33 | 63.33 | 66.67 |
| 5 | Art_Theory | 30 | 80.00 | 76.67 | 73.33 | 76.67 |
| 6 | Basic_Medical_Science | 30 | 73.33 | 80.00 | 66.67 | 66.67 |
| 7 | Biology | 30 | 50.00 | 46.67 | 60.00 | 56.67 |
| 8 | Chemistry | 30 | 46.67 | 50.00 | 30.00 | 50.00 |
| 9 | Clinical_Medicine | 30 | 53.33 | 53.33 | 70.00 | 73.33 |
| 10 | Computer_Science | 30 | 60.00 | 53.33 | 53.33 | 63.33 |
| 11 | Design | 30 | 83.33 | 83.33 | 73.33 | 83.33 |
| 12 | Diagnostics_and_Laboratory_Medicine | 30 | 36.67 | 36.67 | 43.33 | 36.67 |
| 13 | Economics | 30 | 86.67 | 86.67 | 70.00 | 86.67 |
| 14 | Electronics | 30 | 50.00 | 56.67 | 40.00 | 50.00 |
| 15 | Energy_and_Power | 30 | 70.00 | 66.67 | 56.67 | 60.00 |
| 16 | Finance | 30 | 80.00 | 66.67 | 60.00 | 66.67 |
| 17 | Geography | 30 | 60.00 | 60.00 | 53.33 | 63.33 |
| 18 | History | 30 | 73.33 | 63.33 | 70.00 | 76.67 |
| 19 | Literature | 30 | 76.67 | 76.67 | 76.67 | 83.33 |
| 20 | Manage | 30 | 60.00 | 56.67 | 36.67 | 73.33 |
| 21 | Marketing | 30 | 86.67 | 83.33 | 86.67 | 86.67 |
| 22 | Materials | 30 | 53.33 | 46.67 | 53.33 | 53.33 |
| 23 | Math | 30 | 63.33 | 60.00 | 63.33 | 70.00 |
| 24 | Mechanical_Engineering | 30 | 56.67 | 40.00 | 50.00 | 50.00 |
| 25 | Music | 30 | 20.00 | 23.33 | 33.33 | 36.67 |
| 26 | Pharmacy | 30 | 90.00 | 86.67 | 63.33 | 80.00 |
| 27 | Physics | 30 | 80.00 | 83.33 | 56.67 | 73.33 |
| 28 | Psychology | 30 | 80.00 | 70.00 | 73.33 | 73.33 |
| 29 | Public_Health | 30 | 83.33 | 83.33 | 76.67 | 90.00 |
| 30 | Sociology | 30 | 66.67 | 66.67 | 63.33 | 60.00 |
| | **Overall** | **900** | **65.22** | **63.00** | **59.56** | **65.89** |

- **실행 간 노이즈(실측).** best와 시드 1234 재현 사이에 134문항이 뒤집혔고(3407만 정답 64, 1234만 정답 70) 전체 점수 차이는 +0.67%p입니다. 과목 단위 변동은 표준편차 2.4문항, 최대 6문항(Clinical_Medicine 16→22, Music 6→11)이라 과목 단위 차이는 결론의 근거가 되지 못하며, 분과(120~210문항) 단위도 ±3~5%p는 노이즈입니다. 파인튜닝 전후 비교는 시드 2~3개 평균으로 해야 합니다.
- **남은 격차의 성격.** 잘림 77건(루프 60)은 예산 무관한 수렴 실패로 학습 대상이고, 정상 종료 823건 중 오답 261건(32%)은 지식·판독·추론의 한계입니다. 분과별로 Tech & Engineering 55.7%(잘림 51건 집중), Health & Medicine 67.3%가 개선 여지가 큽니다. 시드 간 잘림 겹침이 41/77에 그쳐 루프는 확률적 사건이며, 같은 문항의 정상 종료 응답과 루프 응답을 짝지은 선호 학습 데이터 구성이 가능합니다.
- **주관식 채점 한계.** 공식 `parse_open_response`는 단위·표기 차이("551" 대 "100kΩ")를 오답 처리합니다. 주관식 53문항은 22/53이며 Tech & Engineering 18문항 중 4만 정답입니다.
- **하지 않은 것.** 다수결 샘플링(비용 5배, 공식 프로토콜과 비교 불가), RTX 4090 실행(모델 8 GiB + KV 캐시 9.6 GiB로 24 GB에서도 설정 변경 없이 동작할 것으로 계산되나 실측하지 않음).
- **재현 중 겪은 문제.** Colab CLI 세션은 런타임 토큰이 60분에 만료되어 CLI가 세션을 잃은 것으로 표시하지만 VM은 살아 있으며, 할당 목록에서 새 토큰을 받아 복구했습니다. 시드 재현 실행은 VM이 회수되어 600문항 지점의 백업에서 새 세션으로 이어 돌렸습니다(`resumed: true`, 두 세션 합산 약 2시간 50분). 150문항 배치마다 파일에 기록하는 구조 덕분에 손실이 없었습니다.
- **상세 분석.** 조건·과정·카테고리별(분과, 과목, 문항 유형, 이미지 수, 보기 수, 응답 길이, 잘림) 비교는 [docs/analysis_report.md](../docs/analysis_report.md)에 있습니다.
