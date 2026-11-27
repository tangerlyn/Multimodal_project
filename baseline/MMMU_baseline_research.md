# Qwen3-VL-4B-Instruct MMMU Baseline 자료조사

## 1. 과제 기본 조건

과제 가이드에서 지정된 모델과 데이터셋 조건을 먼저 정리했습니다.

  항목                내용
  ------------------- --------------------------------------------
  모델                `Qwen/Qwen3-VL-4B-Instruct`
  모델 revision       `ebb281ec70b05090aa6165b016eac8ec08e71b17`
  dtype               `bf16`
  데이터셋            `MMMU/MMMU`
  데이터셋 revision   `98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68`
  split               `validation`
  평가 문항           30개 과목 × 30문제 = 총 900문제
  공식 비교 점수      67.4

과목별 문제 수가 모두 30개이기 때문에 전체 점수는 30개 과목 정확도의
평균으로 계산하면 됩니다.

------------------------------------------------------------------------

## 2. Generation 설정

과제 가이드에서 sampling 값은 임의로 정하지 말고 Qwen에서 공개한
benchmark 재현 설정을 찾아 사용하라고 되어 있어서 Qwen3-VL 공식
repository의 Evaluation Reproduction 부분을 확인했습니다.

Instruct 모델 설정은 다음과 같습니다.

  파라미터                  값
  ---------------------- --------
  `do_sample`             `true`
  `temperature`           `0.7`
  `top_p`                 `0.8`
  `top_k`                  `20`
  `repetition_penalty`    `1.0`
  `presence_penalty`      `1.5`
  `seed`                  `3407`

공식 문서에는 `greedy=false`로 되어 있어 sampling을 사용하는 설정으로
확인했습니다. `out_seq_length`는 32768로 제시되어 있습니다.

① [Qwen3-VL 공식 Evaluation
    Reproduction](https://github.com/QwenLM/Qwen3-VL#evaluation-reproduction)
① [Qwen3-VL MMMU
    Evaluation](https://github.com/QwenLM/Qwen3-VL/tree/main/evaluation/mmmu)

특별히 다른 설정을 사용할 이유가 없다면 위 값을 기준으로 적용하면 될 것
같습니다. 최종 보고서 작성 전에는 실제 코드에 적용된 값과 다시 대조하면
될 것 같습니다.

------------------------------------------------------------------------

## 3. MMMU Prompt

프롬프트는 Qwen3-VL 공식 평가에서 사용한다고 안내된 VLMEvalKit 쪽 구현을
확인해봤습니다.

VLMEvalKit의 Qwen3-VL prompt 코드에서는 MMMU validation/test에 대해 별도
prompt를 적용하고 있습니다.

형식은 대략 다음과 같습니다.

``` text
Hint: {hint}
Question: {question}
Options:
A. {option_A}
B. {option_B}
C. {option_C}
D. {option_D}
...
Please select the correct answer from the options above.
```

`Hint`는 해당 값이 있을 때 포함되고, 선택지 수에 따라 A/B/C/D 이후
선택지가 추가될 수 있습니다. 이미지는 Qwen3-VL의 multimodal 입력으로
같이 전달됩니다.

① [VLMEvalKit Qwen3-VL prompt
    코드](https://github.com/open-compass/VLMEvalKit/blob/main/vlmeval/vlm/qwen3_vl/prompt.py)
① [VLMEvalKit
    Quickstart](https://github.com/open-compass/VLMEvalKit/blob/main/docs/en/Quickstart.md)

코드팀에서 별도로 만든 prompt가 있다면 최종 보고서에는 참고용 prompt가
아니라 실제 코드에서 사용한 prompt 전문을 기준으로 작성하면 될 것
같습니다.

가능하면 별도의 `Think step by step`이나 CoT 문구를 임의로 추가하지 않고
한 가지 prompt를 고정해서 baseline과 이후 fine-tuned model 평가에
동일하게 사용하는 방향이 좋을 것 같습니다.

------------------------------------------------------------------------

## 4. max_new_tokens / 이미지 설정

Qwen에서 공개한 MMMU evaluation 자료도 같이 확인했습니다.

참고할 수 있는 값은 다음과 같습니다.

  항목                       값
  ------------------ ------------------
  `max_new_tokens`        `32768`
  `min_pixels`        `1280 * 28 * 28`
  `max_pixels`        `5120 * 28 * 28`

① [Qwen3-VL MMMU Evaluation
    README](https://github.com/QwenLM/Qwen3-VL/blob/main/evaluation/mmmu/README.md)

다만 이 부분은 과제 가이드에서 팀이 직접 정할 수 있게 되어 있습니다. GPU
메모리와 실행시간, 답변이 중간에 잘리는지 등을 확인한 뒤 최종값을 정하면
될 것 같습니다.

실행 후에는 아래 값들도 같이 기록해두면 보고서 작성할 때 사용할 수 있을
것 같습니다.

① 실제 `max_new_tokens`
② 실제 `min_pixels`, `max_pixels`
③ peak VRAM
④ 전체 900문제 실행시간
⑤ OOM 발생 여부
⑥ 답변이 길이 제한 때문에 잘린 경우가 있었는지

------------------------------------------------------------------------

## 5. 답변 파싱 / 채점

MMMU는 모델이 항상 `A` 하나만 출력한다고 보장할 수 없기 때문에 모델
응답에서 최종 답을 추출하는 과정도 확인해봤습니다.

Qwen에서 공개한 MMMU evaluation 설명을 보면 먼저 rule-based 방식으로
답을 추출하고, 명확하게 판단하기 어려운 응답은 model-based extraction을
사용하는 구조로 되어 있습니다.

관련 함수로는 다음 항목들이 안내되어 있습니다.

① `can_infer_option()` : 선택지 추출

① `can_infer_text()` : 텍스트 기반 매칭

① `build_prompt()` : 추가 판정을 위한 prompt 생성

① [Qwen3-VL MMMU Evaluation
    README](https://github.com/QwenLM/Qwen3-VL/blob/main/evaluation/mmmu/README.md)

공개 evaluation 예시에서는 애매한 응답을 처리할 judge model을 별도로
지정할 수 있게 되어 있습니다.

우리 팀에서 parser를 직접 구현한다면 외부 judge를 사용하지 않고 아래처럼
단순하게 처리하는 방법도 가능할 것 같습니다.

``` text
모델 응답
  ↓
A/B/C/D 등 선택지 확인
  ↓
하나로 명확하게 추출되면 해당 답 사용
  ↓
추출 실패/충돌 → invalid 처리
```

이 방식은 구현과 재현은 단순하지만, 모델이 실제로 정답을 말했는데 표현
방식 때문에 parser가 놓치는 경우가 생길 수 있습니다. 반대로 공개
evaluation처럼 judge model까지 사용하면 공식 구현과 더 비슷하게 맞출 수
있지만 외부 모델에 대한 의존성이 추가됩니다.

코드팀에서 이미 parser를 구현했다면 실제 사용한 방식을 확인해서 보고서에
적으면 될 것 같고, 아직 정하지 않았다면 Qwen 공개 구현을 참고해서
결정하면 될 것 같습니다.

------------------------------------------------------------------------

## 6. 공식 점수와 비교할 때 확인할 부분

과제에서 제시된 Qwen3-VL Technical Report 기준 MMMU 점수는 67.4입니다.

다만 이번 과제는 67.4를 그대로 맞추는 것이 목적은 아니고, 동일한 평가
pipeline을 이후 fine-tuning 모델에도 다시 적용할 수 있게 만드는 것이
핵심이라고 되어 있습니다.

실제 점수가 67.4와 차이가 난다면 아래 항목들을 우선 확인해보면 될 것
같습니다.

1.  실제 prompt가 공식/공개 구현과 다른지
2.  sampling 값이 Qwen 공식 설정과 같은지
3.  `max_new_tokens` 차이가 있는지
4.  이미지 해상도 설정이 다른지
5.  parser 방식 때문에 정답을 놓친 경우가 있는지
6.  평가 backend나 구현 방식에 차이가 있는지

특히 parser에서 실패한 응답은 따로 저장해두면 좋을 것 같습니다. 모델 raw
response에는 정답이 있는데 parser만 실패한 경우가 있다면 나중에 gap
analysis의 근거로 사용할 수 있습니다.

------------------------------------------------------------------------

## 7. 결과가 나오면 추가로 확인할 내용

코드 실행 결과가 나오면 아래 항목들을 받아서 최종 보고서 내용과 대조하면
될 것 같습니다.

### 실행 환경

① inference backend와 버전
② GPU / VRAM
③ peak VRAM
④ 전체 실행시간
⑤ requirements 또는 environment 파일
⑥ 실제 실행 command

### 실제 평가 설정

① 실제 prompt 전문
② 실제 sampling 값
③ `max_new_tokens`
④ 이미지 해상도 설정
⑤ parser 방식 및 fallback

### 결과

① 30개 과목별 accuracy
② overall accuracy
③ 가능하면 raw prediction 결과
④ parsing failure가 발생한 sample

결과를 받으면 우선 900문제가 모두 실행되었는지 확인하고, overall 계산을
다시 검산한 뒤 공식 67.4와 차이를 계산하면 될 것 같습니다.

------------------------------------------------------------------------

## 참고 자료

① [Qwen3-VL 공식 Evaluation
    Reproduction](https://github.com/QwenLM/Qwen3-VL#evaluation-reproduction)
① [Qwen3-VL 공식 MMMU
    Evaluation](https://github.com/QwenLM/Qwen3-VL/tree/main/evaluation/mmmu)
① [Qwen3-VL MMMU Evaluation
    README](https://github.com/QwenLM/Qwen3-VL/blob/main/evaluation/mmmu/README.md)
① [VLMEvalKit Qwen3-VL
    Prompt](https://github.com/open-compass/VLMEvalKit/blob/main/vlmeval/vlm/qwen3_vl/prompt.py)
① [VLMEvalKit
    Quickstart](https://github.com/open-compass/VLMEvalKit/blob/main/docs/en/Quickstart.md)
