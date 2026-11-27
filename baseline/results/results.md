# Results for `best.jsonl`

Acc = pipeline score (%, two decimals). Strict = official parser only. See `summary.json`.

| No. | Subject | Data Num | Acc | Strict | length | loops |
|---|---|---|---|---|---|---|
| 1 | Accounting | 30 | 76.67 | 80.00 | 4 | 2 |
| 2 | Agriculture | 30 | 56.67 | 50.00 | 0 | 0 |
| 3 | Architecture_and_Engineering | 30 | 43.33 | 40.00 | 12 | 10 |
| 4 | Art | 30 | 60.00 | 63.33 | 1 | 1 |
| 5 | Art_Theory | 30 | 80.00 | 76.67 | 1 | 1 |
| 6 | Basic_Medical_Science | 30 | 73.33 | 80.00 | 0 | 0 |
| 7 | Biology | 30 | 50.00 | 46.67 | 1 | 1 |
| 8 | Chemistry | 30 | 46.67 | 50.00 | 2 | 2 |
| 9 | Clinical_Medicine | 30 | 53.33 | 53.33 | 0 | 0 |
| 10 | Computer_Science | 30 | 60.00 | 53.33 | 1 | 1 |
| 11 | Design | 30 | 83.33 | 83.33 | 1 | 1 |
| 12 | Diagnostics_and_Laboratory_Medicine | 30 | 36.67 | 36.67 | 0 | 0 |
| 13 | Economics | 30 | 86.67 | 86.67 | 0 | 0 |
| 14 | Electronics | 30 | 50.00 | 56.67 | 5 | 5 |
| 15 | Energy_and_Power | 30 | 70.00 | 66.67 | 12 | 5 |
| 16 | Finance | 30 | 80.00 | 66.67 | 1 | 1 |
| 17 | Geography | 30 | 60.00 | 60.00 | 3 | 3 |
| 18 | History | 30 | 73.33 | 63.33 | 1 | 1 |
| 19 | Literature | 30 | 76.67 | 76.67 | 0 | 0 |
| 20 | Manage | 30 | 60.00 | 56.67 | 1 | 1 |
| 21 | Marketing | 30 | 86.67 | 83.33 | 0 | 0 |
| 22 | Materials | 30 | 53.33 | 46.67 | 8 | 8 |
| 23 | Math | 30 | 63.33 | 60.00 | 3 | 2 |
| 24 | Mechanical_Engineering | 30 | 56.67 | 40.00 | 13 | 9 |
| 25 | Music | 30 | 20.00 | 23.33 | 6 | 5 |
| 26 | Pharmacy | 30 | 90.00 | 86.67 | 0 | 0 |
| 27 | Physics | 30 | 80.00 | 83.33 | 0 | 0 |
| 28 | Psychology | 30 | 80.00 | 70.00 | 0 | 0 |
| 29 | Public_Health | 30 | 83.33 | 83.33 | 1 | 1 |
| 30 | Sociology | 30 | 66.67 | 66.67 | 0 | 0 |
| | **Overall (macro avg)** | **900** | **65.22** | **63.00** | 77 | 60 |

Overall = mean of the 30 subject accuracies = 65.2222. Micro = 587 / 900 × 100 = 65.2222 (equal because every subject has 30 items).

| | Overall (MMMU val) |
|---|---:|
| Official (Qwen3-VL Technical Report) | 67.40 |
| Ours, pipeline | 65.22 |
| Ours, strict official parser | 63.00 |
| Δ pipeline − official | -2.18 pp |

## Generation behaviour

| Response group | n | Acc | Strict |
|---|---:|---:|---:|
| answer_only | 0 | n/a | n/a |
| reasoned | 823 | 68.29 | 66.71 |
| truncated | 77 | 32.47 | 23.38 |

Truncated (finish_reason=length): 77. Verbatim loops among them: 60. Mean response length: 12756 chars, median 1854. Answer-only responses (≤60 chars, one line): 0.

## By MMMU discipline

| Discipline | n | Acc | Strict | length | loops | mean chars |
|---|---:|---:|---:|---:|---:|---:|
| Art & Design | 120 | 60.83 | 61.67 | 9 | 8 | 10565 |
| Business | 150 | 78.00 | 74.67 | 6 | 4 | 7506 |
| Science | 150 | 60.00 | 60.00 | 9 | 8 | 10793 |
| Health & Medicine | 150 | 67.33 | 68.00 | 1 | 1 | 3251 |
| Humanities & Social Science | 120 | 74.17 | 69.17 | 1 | 1 | 2999 |
| Tech & Engineering | 210 | 55.71 | 50.48 | 51 | 38 | 31523 |

## By question type

| Type | n | Acc | Strict |
|---|---:|---:|---:|
| multiple-choice | 847 | 66.71 | 64.34 |
| open | 53 | 41.51 | 41.51 |

## Parser

Extraction methods (pipeline): {'answer_line': 822, 'fallback_letter': 61, 'fallback+official': 5, 'fallback+none': 9, 'official': 1, 'fallback+answer_line': 2}. Explicit-answer rule changed 24 wrong→right and 11 right→wrong. Fallback covered 77 items: 19 wrong→right, 12 right→wrong. Parse failures: strict 13, pipeline 9.
