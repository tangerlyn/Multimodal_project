# Results for `pro10_base_s3407.jsonl`

Acc = pipeline score (%, two decimals). Strict = official parser only. See `summary.json`.

| No. | Subject | Data Num | Acc | Strict | length | loops |
|---|---|---|---|---|---|---|
| 1 | Accounting | 58 | 75.86 | 75.86 | 7 | 6 |
| 2 | Agriculture | 60 | 26.67 | 23.33 | 0 | 0 |
| 3 | Architecture_and_Engineering | 60 | 41.67 | 30.00 | 26 | 14 |
| 4 | Art | 53 | 58.49 | 52.83 | 2 | 2 |
| 5 | Art_Theory | 55 | 67.27 | 65.45 | 1 | 1 |
| 6 | Basic_Medical_Science | 52 | 55.77 | 53.85 | 0 | 0 |
| 7 | Biology | 59 | 45.76 | 45.76 | 5 | 3 |
| 8 | Chemistry | 60 | 50.00 | 46.67 | 8 | 7 |
| 9 | Clinical_Medicine | 59 | 38.98 | 33.90 | 2 | 2 |
| 10 | Computer_Science | 60 | 60.00 | 60.00 | 5 | 5 |
| 11 | Design | 60 | 68.33 | 63.33 | 1 | 1 |
| 12 | Diagnostics_and_Laboratory_Medicine | 60 | 21.67 | 16.67 | 1 | 1 |
| 13 | Economics | 59 | 79.66 | 74.58 | 1 | 1 |
| 14 | Electronics | 60 | 66.67 | 51.67 | 22 | 13 |
| 15 | Energy_and_Power | 58 | 46.55 | 43.10 | 26 | 17 |
| 16 | Finance | 60 | 83.33 | 80.00 | 7 | 5 |
| 17 | Geography | 52 | 40.38 | 38.46 | 6 | 4 |
| 18 | History | 56 | 55.36 | 53.57 | 0 | 0 |
| 19 | Literature | 52 | 75.00 | 76.92 | 0 | 0 |
| 20 | Manage | 50 | 60.00 | 60.00 | 0 | 0 |
| 21 | Marketing | 59 | 77.97 | 69.49 | 2 | 1 |
| 22 | Materials | 60 | 40.00 | 33.33 | 19 | 11 |
| 23 | Math | 60 | 50.00 | 51.67 | 4 | 4 |
| 24 | Mechanical_Engineering | 59 | 35.59 | 33.90 | 18 | 14 |
| 25 | Music | 60 | 31.67 | 16.67 | 18 | 17 |
| 26 | Pharmacy | 57 | 64.91 | 56.14 | 4 | 4 |
| 27 | Physics | 60 | 55.00 | 55.00 | 0 | 0 |
| 28 | Psychology | 60 | 41.67 | 41.67 | 2 | 2 |
| 29 | Public_Health | 58 | 72.41 | 77.59 | 4 | 4 |
| 30 | Sociology | 54 | 53.70 | 51.85 | 0 | 0 |
| | **Overall (macro avg)** | **1730** | **54.68** | **51.11** | 191 | 139 |

Overall = mean of the 30 subject accuracies = 54.6784. Micro = 943 / 1730 × 100 = 54.5087.

| | Overall (MMMU val) |
|---|---:|
| Official (Qwen3-VL Technical Report) | 67.40 |
| Ours, pipeline | 54.51 |
| Ours, strict official parser | 50.87 |
| Δ pipeline − official | -12.89 pp |

## Generation behaviour

| Response group | n | Acc | Strict |
|---|---:|---:|---:|
| answer_only | 0 | n/a | n/a |
| reasoned | 1539 | 58.28 | 56.08 |
| truncated | 191 | 24.08 | 8.90 |

Truncated (finish_reason=length): 191. Verbatim loops among them: 139. Mean response length: 9954 chars, median 2372. Answer-only responses (≤60 chars, one line): 0.

## By MMMU discipline

| Discipline | n | Acc | Strict | length | loops | mean chars |
|---|---:|---:|---:|---:|---:|---:|
| Art & Design | 228 | 56.14 | 49.12 | 22 | 21 | 6798 |
| Business | 286 | 75.87 | 72.38 | 17 | 13 | 6669 |
| Science | 291 | 48.45 | 47.77 | 23 | 18 | 9768 |
| Health & Medicine | 286 | 50.35 | 47.20 | 11 | 11 | 4875 |
| Humanities & Social Science | 222 | 55.86 | 55.41 | 2 | 2 | 2597 |
| Tech & Engineering | 417 | 45.32 | 39.33 | 116 | 74 | 21465 |

## By question type

| Type | n | Acc | Strict |
|---|---:|---:|---:|
| multiple-choice | 1730 | 54.51 | 50.87 |
| open | 0 | n/a | n/a |

## Parser

Extraction methods (pipeline): {'fallback_letter': 160, 'answer_line': 1534, 'fallback+none': 27, 'official': 5, 'fallback+official': 4}. Explicit-answer rule changed 48 wrong→right and 14 right→wrong. Fallback covered 191 items: 39 wrong→right, 10 right→wrong. Parse failures: strict 7, pipeline 27.
