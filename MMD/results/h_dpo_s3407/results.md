# Results for `h_dpo_s3407.jsonl`

Acc = pipeline score (%, two decimals). Strict = official parser only. See `summary.json`.

| No. | Subject | Data Num | Acc | Strict | length | loops |
|---|---|---|---|---|---|---|
| 1 | Accounting | 30 | 73.33 | 73.33 | 6 | 5 |
| 2 | Agriculture | 30 | 50.00 | 53.33 | 0 | 0 |
| 3 | Architecture_and_Engineering | 30 | 43.33 | 36.67 | 9 | 5 |
| 4 | Art | 30 | 73.33 | 73.33 | 0 | 0 |
| 5 | Art_Theory | 30 | 80.00 | 73.33 | 1 | 1 |
| 6 | Basic_Medical_Science | 30 | 76.67 | 80.00 | 0 | 0 |
| 7 | Biology | 30 | 50.00 | 40.00 | 2 | 2 |
| 8 | Chemistry | 30 | 50.00 | 46.67 | 6 | 5 |
| 9 | Clinical_Medicine | 30 | 73.33 | 70.00 | 0 | 0 |
| 10 | Computer_Science | 30 | 63.33 | 56.67 | 6 | 6 |
| 11 | Design | 30 | 76.67 | 73.33 | 0 | 0 |
| 12 | Diagnostics_and_Laboratory_Medicine | 30 | 36.67 | 36.67 | 0 | 0 |
| 13 | Economics | 30 | 80.00 | 76.67 | 0 | 0 |
| 14 | Electronics | 30 | 43.33 | 53.33 | 6 | 4 |
| 15 | Energy_and_Power | 30 | 56.67 | 46.67 | 16 | 15 |
| 16 | Finance | 30 | 66.67 | 66.67 | 3 | 2 |
| 17 | Geography | 30 | 56.67 | 56.67 | 1 | 1 |
| 18 | History | 30 | 70.00 | 66.67 | 1 | 1 |
| 19 | Literature | 30 | 80.00 | 80.00 | 0 | 0 |
| 20 | Manage | 30 | 66.67 | 63.33 | 0 | 0 |
| 21 | Marketing | 30 | 76.67 | 73.33 | 1 | 1 |
| 22 | Materials | 30 | 60.00 | 50.00 | 14 | 7 |
| 23 | Math | 30 | 70.00 | 73.33 | 4 | 3 |
| 24 | Mechanical_Engineering | 30 | 53.33 | 43.33 | 13 | 9 |
| 25 | Music | 30 | 36.67 | 23.33 | 12 | 12 |
| 26 | Pharmacy | 30 | 86.67 | 73.33 | 2 | 2 |
| 27 | Physics | 30 | 80.00 | 76.67 | 1 | 1 |
| 28 | Psychology | 30 | 83.33 | 76.67 | 1 | 1 |
| 29 | Public_Health | 30 | 86.67 | 90.00 | 2 | 2 |
| 30 | Sociology | 30 | 60.00 | 60.00 | 0 | 0 |
| | **Overall (macro avg)** | **900** | **65.33** | **62.11** | 107 | 85 |

Overall = mean of the 30 subject accuracies = 65.3333. Micro = 588 / 900 × 100 = 65.3333 (equal because every subject has 30 items).

| | Overall (MMMU val) |
|---|---:|
| Official (Qwen3-VL Technical Report) | 67.40 |
| Ours, pipeline | 65.33 |
| Ours, strict official parser | 62.11 |
| Δ pipeline − official | -2.07 pp |

## Generation behaviour

| Response group | n | Acc | Strict |
|---|---:|---:|---:|
| answer_only | 0 | n/a | n/a |
| reasoned | 793 | 69.10 | 67.21 |
| truncated | 107 | 37.38 | 24.30 |

Truncated (finish_reason=length): 107. Verbatim loops among them: 85. Mean response length: 16264 chars, median 1682. Answer-only responses (≤60 chars, one line): 0.

## By MMMU discipline

| Discipline | n | Acc | Strict | length | loops | mean chars |
|---|---:|---:|---:|---:|---:|---:|
| Art & Design | 120 | 66.67 | 60.83 | 13 | 13 | 12730 |
| Business | 150 | 72.67 | 70.67 | 10 | 8 | 9811 |
| Science | 150 | 61.33 | 58.67 | 14 | 12 | 16045 |
| Health & Medicine | 150 | 72.00 | 70.00 | 4 | 4 | 5334 |
| Humanities & Social Science | 120 | 73.33 | 70.83 | 2 | 2 | 3771 |
| Tech & Engineering | 210 | 52.86 | 48.57 | 64 | 46 | 37997 |

## By question type

| Type | n | Acc | Strict |
|---|---:|---:|---:|
| multiple-choice | 847 | 66.59 | 63.28 |
| open | 53 | 45.28 | 43.40 |

## Parser

Extraction methods (pipeline): {'answer_line': 791, 'fallback_letter': 88, 'fallback+none': 9, 'fallback+official': 8, 'official': 1, 'fallback+answer_line': 3}. Explicit-answer rule changed 27 wrong→right and 11 right→wrong. Fallback covered 108 items: 23 wrong→right, 10 right→wrong. Parse failures: strict 23, pipeline 9.
