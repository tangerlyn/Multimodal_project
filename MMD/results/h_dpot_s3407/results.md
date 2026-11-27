# Results for `h_dpot_s3407.jsonl`

Acc = pipeline score (%, two decimals). Strict = official parser only. See `summary.json`.

| No. | Subject | Data Num | Acc | Strict | length | loops |
|---|---|---|---|---|---|---|
| 1 | Accounting | 30 | 80.00 | 66.67 | 8 | 8 |
| 2 | Agriculture | 30 | 50.00 | 56.67 | 0 | 0 |
| 3 | Architecture_and_Engineering | 30 | 56.67 | 30.00 | 15 | 11 |
| 4 | Art | 30 | 63.33 | 66.67 | 0 | 0 |
| 5 | Art_Theory | 30 | 76.67 | 73.33 | 0 | 0 |
| 6 | Basic_Medical_Science | 30 | 63.33 | 70.00 | 0 | 0 |
| 7 | Biology | 30 | 53.33 | 50.00 | 1 | 1 |
| 8 | Chemistry | 30 | 43.33 | 46.67 | 9 | 9 |
| 9 | Clinical_Medicine | 30 | 70.00 | 60.00 | 0 | 0 |
| 10 | Computer_Science | 30 | 63.33 | 56.67 | 6 | 4 |
| 11 | Design | 30 | 83.33 | 86.67 | 0 | 0 |
| 12 | Diagnostics_and_Laboratory_Medicine | 30 | 33.33 | 33.33 | 0 | 0 |
| 13 | Economics | 30 | 80.00 | 80.00 | 0 | 0 |
| 14 | Electronics | 30 | 46.67 | 50.00 | 6 | 6 |
| 15 | Energy_and_Power | 30 | 76.67 | 66.67 | 11 | 9 |
| 16 | Finance | 30 | 73.33 | 66.67 | 3 | 3 |
| 17 | Geography | 30 | 50.00 | 56.67 | 1 | 1 |
| 18 | History | 30 | 70.00 | 73.33 | 1 | 1 |
| 19 | Literature | 30 | 80.00 | 76.67 | 0 | 0 |
| 20 | Manage | 30 | 63.33 | 53.33 | 1 | 1 |
| 21 | Marketing | 30 | 80.00 | 83.33 | 1 | 1 |
| 22 | Materials | 30 | 56.67 | 46.67 | 11 | 7 |
| 23 | Math | 30 | 60.00 | 60.00 | 2 | 2 |
| 24 | Mechanical_Engineering | 30 | 43.33 | 40.00 | 11 | 9 |
| 25 | Music | 30 | 33.33 | 33.33 | 10 | 10 |
| 26 | Pharmacy | 30 | 80.00 | 76.67 | 3 | 3 |
| 27 | Physics | 30 | 73.33 | 73.33 | 2 | 1 |
| 28 | Psychology | 30 | 83.33 | 86.67 | 0 | 0 |
| 29 | Public_Health | 30 | 86.67 | 86.67 | 2 | 2 |
| 30 | Sociology | 30 | 63.33 | 63.33 | 0 | 0 |
| | **Overall (macro avg)** | **900** | **64.56** | **62.33** | 104 | 89 |

Overall = mean of the 30 subject accuracies = 64.5556. Micro = 581 / 900 × 100 = 64.5556 (equal because every subject has 30 items).

| | Overall (MMMU val) |
|---|---:|
| Official (Qwen3-VL Technical Report) | 67.40 |
| Ours, pipeline | 64.56 |
| Ours, strict official parser | 62.33 |
| Δ pipeline − official | -2.84 pp |

## Generation behaviour

| Response group | n | Acc | Strict |
|---|---:|---:|---:|
| answer_only | 0 | n/a | n/a |
| reasoned | 796 | 69.10 | 67.96 |
| truncated | 104 | 29.81 | 19.23 |

Truncated (finish_reason=length): 104. Verbatim loops among them: 89. Mean response length: 15729 chars, median 1592. Answer-only responses (≤60 chars, one line): 0.

## By MMMU discipline

| Discipline | n | Acc | Strict | length | loops | mean chars |
|---|---:|---:|---:|---:|---:|---:|
| Art & Design | 120 | 64.17 | 65.00 | 10 | 10 | 11065 |
| Business | 150 | 75.33 | 70.00 | 13 | 13 | 12716 |
| Science | 150 | 56.00 | 57.33 | 15 | 14 | 15600 |
| Health & Medicine | 150 | 66.67 | 65.33 | 5 | 5 | 5817 |
| Humanities & Social Science | 120 | 74.17 | 75.00 | 1 | 1 | 2817 |
| Tech & Engineering | 210 | 56.19 | 49.52 | 60 | 46 | 35097 |

## By question type

| Type | n | Acc | Strict |
|---|---:|---:|---:|
| multiple-choice | 847 | 65.88 | 63.64 |
| open | 53 | 43.40 | 41.51 |

## Parser

Extraction methods (pipeline): {'answer_line': 796, 'fallback+none': 12, 'fallback_letter': 85, 'fallback+official': 6, 'fallback+answer_line': 1}. Explicit-answer rule changed 22 wrong→right and 13 right→wrong. Fallback covered 104 items: 21 wrong→right, 10 right→wrong. Parse failures: strict 23, pipeline 12.
