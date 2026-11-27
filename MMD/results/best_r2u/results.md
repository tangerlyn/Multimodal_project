# Results for `best_r2u.jsonl`

Acc = pipeline score (%, two decimals). Strict = official parser only. See `summary.json`.

| No. | Subject | Data Num | Acc | Strict | length | loops |
|---|---|---|---|---|---|---|
| 1 | Accounting | 30 | 80.00 | 80.00 | 3 | 2 |
| 2 | Agriculture | 30 | 53.33 | 50.00 | 0 | 0 |
| 3 | Architecture_and_Engineering | 30 | 53.33 | 36.67 | 11 | 5 |
| 4 | Art | 30 | 60.00 | 60.00 | 1 | 1 |
| 5 | Art_Theory | 30 | 83.33 | 80.00 | 1 | 1 |
| 6 | Basic_Medical_Science | 30 | 63.33 | 66.67 | 0 | 0 |
| 7 | Biology | 30 | 46.67 | 46.67 | 0 | 0 |
| 8 | Chemistry | 30 | 46.67 | 40.00 | 6 | 5 |
| 9 | Clinical_Medicine | 30 | 50.00 | 50.00 | 0 | 0 |
| 10 | Computer_Science | 30 | 73.33 | 70.00 | 4 | 4 |
| 11 | Design | 30 | 80.00 | 73.33 | 0 | 0 |
| 12 | Diagnostics_and_Laboratory_Medicine | 30 | 30.00 | 30.00 | 0 | 0 |
| 13 | Economics | 30 | 86.67 | 86.67 | 0 | 0 |
| 14 | Electronics | 30 | 50.00 | 63.33 | 4 | 4 |
| 15 | Energy_and_Power | 30 | 56.67 | 40.00 | 15 | 11 |
| 16 | Finance | 30 | 76.67 | 66.67 | 2 | 1 |
| 17 | Geography | 30 | 60.00 | 60.00 | 1 | 0 |
| 18 | History | 30 | 76.67 | 73.33 | 0 | 0 |
| 19 | Literature | 30 | 83.33 | 83.33 | 0 | 0 |
| 20 | Manage | 30 | 63.33 | 63.33 | 0 | 0 |
| 21 | Marketing | 30 | 83.33 | 86.67 | 2 | 1 |
| 22 | Materials | 30 | 63.33 | 56.67 | 11 | 8 |
| 23 | Math | 30 | 66.67 | 66.67 | 2 | 1 |
| 24 | Mechanical_Engineering | 30 | 53.33 | 46.67 | 12 | 8 |
| 25 | Music | 30 | 43.33 | 30.00 | 9 | 7 |
| 26 | Pharmacy | 30 | 76.67 | 76.67 | 1 | 1 |
| 27 | Physics | 30 | 76.67 | 73.33 | 1 | 1 |
| 28 | Psychology | 30 | 80.00 | 76.67 | 1 | 1 |
| 29 | Public_Health | 30 | 90.00 | 90.00 | 1 | 0 |
| 30 | Sociology | 30 | 60.00 | 60.00 | 0 | 0 |
| | **Overall (macro avg)** | **900** | **65.56** | **62.78** | 88 | 62 |

Overall = mean of the 30 subject accuracies = 65.5556. Micro = 590 / 900 × 100 = 65.5556 (equal because every subject has 30 items).

| | Overall (MMMU val) |
|---|---:|
| Official (Qwen3-VL Technical Report) | 67.40 |
| Ours, pipeline | 65.56 |
| Ours, strict official parser | 62.78 |
| Δ pipeline − official | -1.84 pp |

## Generation behaviour

| Response group | n | Acc | Strict |
|---|---:|---:|---:|
| answer_only | 0 | n/a | n/a |
| reasoned | 812 | 68.84 | 67.00 |
| truncated | 88 | 35.23 | 23.86 |

Truncated (finish_reason=length): 88. Verbatim loops among them: 62. Mean response length: 14107 chars, median 1805. Answer-only responses (≤60 chars, one line): 0.

## By MMMU discipline

| Discipline | n | Acc | Strict | length | loops | mean chars |
|---|---:|---:|---:|---:|---:|---:|
| Art & Design | 120 | 66.67 | 60.83 | 11 | 9 | 12536 |
| Business | 150 | 78.00 | 76.67 | 7 | 4 | 8806 |
| Science | 150 | 59.33 | 57.33 | 10 | 7 | 11770 |
| Health & Medicine | 150 | 62.00 | 62.67 | 2 | 1 | 3858 |
| Humanities & Social Science | 120 | 75.00 | 73.33 | 1 | 1 | 2987 |
| Tech & Engineering | 210 | 57.62 | 51.90 | 57 | 40 | 34136 |

## By question type

| Type | n | Acc | Strict |
|---|---:|---:|---:|
| multiple-choice | 847 | 66.82 | 63.99 |
| open | 53 | 45.28 | 43.40 |

## Parser

Extraction methods (pipeline): {'answer_line': 809, 'fallback_letter': 76, 'fallback+none': 9, 'fallback+answer_line': 1, 'official': 3, 'fallback+official': 2}. Explicit-answer rule changed 25 wrong→right and 10 right→wrong. Fallback covered 88 items: 23 wrong→right, 13 right→wrong. Parse failures: strict 15, pipeline 9.
