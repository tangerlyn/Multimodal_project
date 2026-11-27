# Results for `prov_base_s3407.jsonl`

Acc = pipeline score (%, two decimals). Strict = official parser only. See `summary.json`.

| No. | Subject | Data Num | Acc | Strict | length | loops |
|---|---|---|---|---|---|---|
| 1 | Accounting | 58 | 77.59 | 72.41 | 5 | 4 |
| 2 | Agriculture | 60 | 16.67 | 20.00 | 0 | 0 |
| 3 | Architecture_and_Engineering | 60 | 46.67 | 45.00 | 23 | 16 |
| 4 | Art | 53 | 49.06 | 37.74 | 5 | 5 |
| 5 | Art_Theory | 55 | 52.73 | 41.82 | 2 | 2 |
| 6 | Basic_Medical_Science | 52 | 44.23 | 32.69 | 0 | 0 |
| 7 | Biology | 59 | 38.98 | 35.59 | 1 | 1 |
| 8 | Chemistry | 60 | 48.33 | 41.67 | 8 | 6 |
| 9 | Clinical_Medicine | 59 | 35.59 | 25.42 | 1 | 1 |
| 10 | Computer_Science | 60 | 53.33 | 46.67 | 7 | 7 |
| 11 | Design | 60 | 56.67 | 51.67 | 2 | 2 |
| 12 | Diagnostics_and_Laboratory_Medicine | 60 | 21.67 | 16.67 | 1 | 1 |
| 13 | Economics | 59 | 76.27 | 76.27 | 1 | 0 |
| 14 | Electronics | 60 | 53.33 | 48.33 | 13 | 9 |
| 15 | Energy_and_Power | 58 | 48.28 | 43.10 | 24 | 11 |
| 16 | Finance | 60 | 76.67 | 75.00 | 6 | 5 |
| 17 | Geography | 52 | 38.46 | 30.77 | 2 | 0 |
| 18 | History | 56 | 51.79 | 41.07 | 0 | 0 |
| 19 | Literature | 52 | 71.15 | 57.69 | 0 | 0 |
| 20 | Manage | 50 | 62.00 | 52.00 | 3 | 2 |
| 21 | Marketing | 59 | 72.88 | 67.80 | 1 | 1 |
| 22 | Materials | 60 | 41.67 | 40.00 | 14 | 10 |
| 23 | Math | 60 | 53.33 | 50.00 | 5 | 3 |
| 24 | Mechanical_Engineering | 59 | 37.29 | 35.59 | 17 | 9 |
| 25 | Music | 60 | 30.00 | 23.33 | 20 | 19 |
| 26 | Pharmacy | 57 | 63.16 | 63.16 | 7 | 5 |
| 27 | Physics | 60 | 61.67 | 60.00 | 2 | 2 |
| 28 | Psychology | 60 | 43.33 | 35.00 | 1 | 1 |
| 29 | Public_Health | 58 | 68.97 | 68.97 | 3 | 2 |
| 30 | Sociology | 54 | 51.85 | 40.74 | 0 | 0 |
| | **Overall (macro avg)** | **1730** | **51.45** | **45.87** | 174 | 124 |

Overall = mean of the 30 subject accuracies = 51.4534. Micro = 888 / 1730 × 100 = 51.3295.

| | Overall (MMMU val) |
|---|---:|
| Official (Qwen3-VL Technical Report) | 67.40 |
| Ours, pipeline | 51.33 |
| Ours, strict official parser | 45.90 |
| Δ pipeline − official | -16.07 pp |

## Generation behaviour

| Response group | n | Acc | Strict |
|---|---:|---:|---:|
| answer_only | 0 | n/a | n/a |
| reasoned | 1556 | 54.43 | 49.55 |
| truncated | 174 | 23.56 | 13.22 |

Truncated (finish_reason=length): 174. Verbatim loops among them: 124. Mean response length: 10109 chars, median 2911. Answer-only responses (≤60 chars, one line): 0.

## By MMMU discipline

| Discipline | n | Acc | Strict | length | loops | mean chars |
|---|---:|---:|---:|---:|---:|---:|
| Art & Design | 228 | 46.93 | 38.60 | 29 | 28 | 9206 |
| Business | 286 | 73.43 | 69.23 | 16 | 12 | 7903 |
| Science | 291 | 48.45 | 43.99 | 18 | 12 | 8786 |
| Health & Medicine | 286 | 46.50 | 41.26 | 12 | 9 | 5667 |
| Humanities & Social Science | 222 | 54.05 | 43.24 | 1 | 1 | 3252 |
| Tech & Engineering | 417 | 42.45 | 39.81 | 98 | 62 | 19734 |

## By question type

| Type | n | Acc | Strict |
|---|---:|---:|---:|
| multiple-choice | 1730 | 51.33 | 45.90 |
| open | 0 | n/a | n/a |

## Parser

Extraction methods (pipeline): {'answer_line': 1545, 'fallback_letter': 171, 'official': 11, 'fallback+none': 3}. Explicit-answer rule changed 99 wrong→right and 23 right→wrong. Fallback covered 174 items: 27 wrong→right, 9 right→wrong. Parse failures: strict 6, pipeline 3.
