# Results for `t_sft_s3407.jsonl`

Acc = pipeline score (%, two decimals). Strict = official parser only. See `summary.json`.

| No. | Subject | Data Num | Acc | Strict | length | loops |
|---|---|---|---|---|---|---|
| 1 | Accounting | 30 | 70.00 | 70.00 | 0 | 0 |
| 2 | Agriculture | 30 | 53.33 | 50.00 | 0 | 0 |
| 3 | Architecture_and_Engineering | 30 | 56.67 | 50.00 | 3 | 2 |
| 4 | Art | 30 | 66.67 | 60.00 | 0 | 0 |
| 5 | Art_Theory | 30 | 83.33 | 66.67 | 0 | 0 |
| 6 | Basic_Medical_Science | 30 | 73.33 | 73.33 | 0 | 0 |
| 7 | Biology | 30 | 43.33 | 36.67 | 0 | 0 |
| 8 | Chemistry | 30 | 50.00 | 50.00 | 0 | 0 |
| 9 | Clinical_Medicine | 30 | 63.33 | 56.67 | 0 | 0 |
| 10 | Computer_Science | 30 | 66.67 | 63.33 | 0 | 0 |
| 11 | Design | 30 | 83.33 | 70.00 | 0 | 0 |
| 12 | Diagnostics_and_Laboratory_Medicine | 30 | 33.33 | 33.33 | 0 | 0 |
| 13 | Economics | 30 | 86.67 | 86.67 | 0 | 0 |
| 14 | Electronics | 30 | 53.33 | 53.33 | 1 | 0 |
| 15 | Energy_and_Power | 30 | 50.00 | 46.67 | 3 | 2 |
| 16 | Finance | 30 | 63.33 | 53.33 | 0 | 0 |
| 17 | Geography | 30 | 46.67 | 50.00 | 0 | 0 |
| 18 | History | 30 | 66.67 | 53.33 | 1 | 1 |
| 19 | Literature | 30 | 83.33 | 73.33 | 0 | 0 |
| 20 | Manage | 30 | 56.67 | 50.00 | 0 | 0 |
| 21 | Marketing | 30 | 86.67 | 83.33 | 0 | 0 |
| 22 | Materials | 30 | 40.00 | 40.00 | 1 | 0 |
| 23 | Math | 30 | 76.67 | 76.67 | 0 | 0 |
| 24 | Mechanical_Engineering | 30 | 50.00 | 46.67 | 3 | 3 |
| 25 | Music | 30 | 30.00 | 23.33 | 1 | 1 |
| 26 | Pharmacy | 30 | 70.00 | 63.33 | 0 | 0 |
| 27 | Physics | 30 | 70.00 | 70.00 | 0 | 0 |
| 28 | Psychology | 30 | 73.33 | 66.67 | 0 | 0 |
| 29 | Public_Health | 30 | 86.67 | 86.67 | 1 | 1 |
| 30 | Sociology | 30 | 50.00 | 50.00 | 0 | 0 |
| | **Overall (macro avg)** | **900** | **62.78** | **58.44** | 14 | 10 |

Overall = mean of the 30 subject accuracies = 62.7778. Micro = 565 / 900 × 100 = 62.7778 (equal because every subject has 30 items).

| | Overall (MMMU val) |
|---|---:|
| Official (Qwen3-VL Technical Report) | 67.40 |
| Ours, pipeline | 62.78 |
| Ours, strict official parser | 58.44 |
| Δ pipeline − official | -4.62 pp |

## Generation behaviour

| Response group | n | Acc | Strict |
|---|---:|---:|---:|
| answer_only | 0 | n/a | n/a |
| reasoned | 886 | 63.09 | 59.26 |
| truncated | 14 | 42.86 | 7.14 |

Truncated (finish_reason=length): 14. Verbatim loops among them: 10. Mean response length: 3594 chars, median 1204. Answer-only responses (≤60 chars, one line): 0.

## By MMMU discipline

| Discipline | n | Acc | Strict | length | loops | mean chars |
|---|---:|---:|---:|---:|---:|---:|
| Art & Design | 120 | 65.83 | 55.00 | 1 | 1 | 1882 |
| Business | 150 | 72.67 | 68.67 | 0 | 0 | 1919 |
| Science | 150 | 57.33 | 56.67 | 0 | 0 | 1965 |
| Health & Medicine | 150 | 65.33 | 62.67 | 1 | 1 | 1760 |
| Humanities & Social Science | 120 | 68.33 | 60.83 | 1 | 1 | 2112 |
| Tech & Engineering | 210 | 52.86 | 50.00 | 11 | 7 | 9088 |

## By question type

| Type | n | Acc | Strict |
|---|---:|---:|---:|
| multiple-choice | 847 | 64.11 | 60.09 |
| open | 53 | 41.51 | 32.08 |

## Parser

Extraction methods (pipeline): {'answer_line': 885, 'fallback_letter': 13, 'official': 1, 'fallback+none': 1}. Explicit-answer rule changed 41 wrong→right and 7 right→wrong. Fallback covered 14 items: 5 wrong→right, 0 right→wrong. Parse failures: strict 5, pipeline 1.
