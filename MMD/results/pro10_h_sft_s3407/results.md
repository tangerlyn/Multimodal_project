# Results for `pro10_h_sft_s3407.jsonl`

Acc = pipeline score (%, two decimals). Strict = official parser only. See `summary.json`.

| No. | Subject | Data Num | Acc | Strict | length | loops |
|---|---|---|---|---|---|---|
| 1 | Accounting | 58 | 75.86 | 74.14 | 3 | 2 |
| 2 | Agriculture | 60 | 33.33 | 30.00 | 0 | 0 |
| 3 | Architecture_and_Engineering | 60 | 41.67 | 43.33 | 19 | 8 |
| 4 | Art | 53 | 60.38 | 54.72 | 1 | 1 |
| 5 | Art_Theory | 55 | 65.45 | 63.64 | 0 | 0 |
| 6 | Basic_Medical_Science | 52 | 51.92 | 46.15 | 0 | 0 |
| 7 | Biology | 59 | 42.37 | 37.29 | 2 | 1 |
| 8 | Chemistry | 60 | 45.00 | 43.33 | 4 | 3 |
| 9 | Clinical_Medicine | 59 | 40.68 | 37.29 | 0 | 0 |
| 10 | Computer_Science | 60 | 55.00 | 53.33 | 2 | 1 |
| 11 | Design | 60 | 68.33 | 65.00 | 0 | 0 |
| 12 | Diagnostics_and_Laboratory_Medicine | 60 | 26.67 | 25.00 | 0 | 0 |
| 13 | Economics | 59 | 74.58 | 72.88 | 0 | 0 |
| 14 | Electronics | 60 | 73.33 | 66.67 | 13 | 4 |
| 15 | Energy_and_Power | 58 | 48.28 | 46.55 | 14 | 2 |
| 16 | Finance | 60 | 78.33 | 78.33 | 4 | 1 |
| 17 | Geography | 52 | 51.92 | 50.00 | 3 | 1 |
| 18 | History | 56 | 66.07 | 67.86 | 0 | 0 |
| 19 | Literature | 52 | 69.23 | 67.31 | 0 | 0 |
| 20 | Manage | 50 | 64.00 | 60.00 | 0 | 0 |
| 21 | Marketing | 59 | 74.58 | 66.10 | 1 | 0 |
| 22 | Materials | 60 | 50.00 | 43.33 | 10 | 3 |
| 23 | Math | 60 | 63.33 | 61.67 | 3 | 1 |
| 24 | Mechanical_Engineering | 59 | 49.15 | 45.76 | 11 | 4 |
| 25 | Music | 60 | 28.33 | 31.67 | 17 | 13 |
| 26 | Pharmacy | 57 | 63.16 | 61.40 | 3 | 3 |
| 27 | Physics | 60 | 48.33 | 46.67 | 0 | 0 |
| 28 | Psychology | 60 | 43.33 | 38.33 | 1 | 1 |
| 29 | Public_Health | 58 | 74.14 | 75.86 | 3 | 1 |
| 30 | Sociology | 54 | 51.85 | 51.85 | 0 | 0 |
| | **Overall (macro avg)** | **1730** | **55.95** | **53.52** | 114 | 50 |

Overall = mean of the 30 subject accuracies = 55.9541. Micro = 965 / 1730 × 100 = 55.7803.

| | Overall (MMMU val) |
|---|---:|
| Official (Qwen3-VL Technical Report) | 67.40 |
| Ours, pipeline | 55.78 |
| Ours, strict official parser | 53.35 |
| Δ pipeline − official | -11.62 pp |

## Generation behaviour

| Response group | n | Acc | Strict |
|---|---:|---:|---:|
| answer_only | 0 | n/a | n/a |
| reasoned | 1616 | 58.54 | 56.37 |
| truncated | 114 | 16.67 | 10.53 |

Truncated (finish_reason=length): 114. Verbatim loops among them: 50. Mean response length: 8232 chars, median 2270. Answer-only responses (≤60 chars, one line): 0.

## By MMMU discipline

| Discipline | n | Acc | Strict | length | loops | mean chars |
|---|---:|---:|---:|---:|---:|---:|
| Art & Design | 228 | 55.26 | 53.51 | 18 | 14 | 6061 |
| Business | 286 | 73.78 | 70.63 | 8 | 3 | 5431 |
| Science | 291 | 50.17 | 47.77 | 12 | 6 | 7443 |
| Health & Medicine | 286 | 51.05 | 48.95 | 6 | 4 | 4014 |
| Humanities & Social Science | 222 | 57.21 | 55.86 | 1 | 1 | 2280 |
| Tech & Engineering | 417 | 50.12 | 47.00 | 69 | 22 | 17952 |

## By question type

| Type | n | Acc | Strict |
|---|---:|---:|---:|
| multiple-choice | 1730 | 55.78 | 53.35 |
| open | 0 | n/a | n/a |

## Parser

Extraction methods (pipeline): {'answer_line': 1611, 'fallback_letter': 101, 'fallback+answer_line': 2, 'fallback+none': 10, 'official': 5, 'fallback+official': 1}. Explicit-answer rule changed 47 wrong→right and 12 right→wrong. Fallback covered 114 items: 17 wrong→right, 10 right→wrong. Parse failures: strict 6, pipeline 10.
