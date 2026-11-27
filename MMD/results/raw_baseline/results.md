# Results for `raw.jsonl`

Acc = pipeline score (%, two decimals). Strict = official parser only. See `summary.json`.

| No. | Subject | Data Num | Acc | Strict | length | loops |
|---|---|---|---|---|---|---|
| 1 | Accounting | 30 | 73.33 | 73.33 | 6 | 2 |
| 2 | Agriculture | 30 | 60.00 | 60.00 | 0 | 0 |
| 3 | Architecture_and_Engineering | 30 | 36.67 | 36.67 | 14 | 2 |
| 4 | Art | 30 | 63.33 | 63.33 | 0 | 0 |
| 5 | Art_Theory | 30 | 73.33 | 73.33 | 0 | 0 |
| 6 | Basic_Medical_Science | 30 | 66.67 | 66.67 | 0 | 0 |
| 7 | Biology | 30 | 60.00 | 60.00 | 0 | 0 |
| 8 | Chemistry | 30 | 30.00 | 30.00 | 1 | 1 |
| 9 | Clinical_Medicine | 30 | 70.00 | 70.00 | 0 | 0 |
| 10 | Computer_Science | 30 | 53.33 | 53.33 | 1 | 0 |
| 11 | Design | 30 | 73.33 | 73.33 | 0 | 0 |
| 12 | Diagnostics_and_Laboratory_Medicine | 30 | 43.33 | 43.33 | 0 | 0 |
| 13 | Economics | 30 | 70.00 | 70.00 | 0 | 0 |
| 14 | Electronics | 30 | 40.00 | 40.00 | 6 | 2 |
| 15 | Energy_and_Power | 30 | 56.67 | 56.67 | 10 | 2 |
| 16 | Finance | 30 | 60.00 | 60.00 | 3 | 2 |
| 17 | Geography | 30 | 53.33 | 53.33 | 2 | 1 |
| 18 | History | 30 | 70.00 | 70.00 | 0 | 0 |
| 19 | Literature | 30 | 76.67 | 76.67 | 0 | 0 |
| 20 | Manage | 30 | 36.67 | 36.67 | 1 | 1 |
| 21 | Marketing | 30 | 86.67 | 86.67 | 1 | 0 |
| 22 | Materials | 30 | 53.33 | 53.33 | 11 | 3 |
| 23 | Math | 30 | 63.33 | 60.00 | 4 | 1 |
| 24 | Mechanical_Engineering | 30 | 50.00 | 50.00 | 9 | 5 |
| 25 | Music | 30 | 33.33 | 33.33 | 0 | 0 |
| 26 | Pharmacy | 30 | 63.33 | 63.33 | 0 | 0 |
| 27 | Physics | 30 | 56.67 | 56.67 | 0 | 0 |
| 28 | Psychology | 30 | 73.33 | 73.33 | 0 | 0 |
| 29 | Public_Health | 30 | 76.67 | 76.67 | 2 | 1 |
| 30 | Sociology | 30 | 63.33 | 63.33 | 0 | 0 |
| | **Overall (macro avg)** | **900** | **59.56** | **59.44** | 71 | 23 |

Overall = mean of the 30 subject accuracies = 59.5556. Micro = 536 / 900 × 100 = 59.5556 (equal because every subject has 30 items).

| | Overall (MMMU val) |
|---|---:|
| Official (Qwen3-VL Technical Report) | 67.40 |
| Ours, pipeline | 59.56 |
| Ours, strict official parser | 59.44 |
| Δ pipeline − official | -7.84 pp |

## Generation behaviour

| Response group | n | Acc | Strict |
|---|---:|---:|---:|
| answer_only | 605 | 55.54 | 55.54 |
| reasoned | 224 | 80.80 | 80.36 |
| truncated | 71 | 26.76 | 26.76 |

Truncated (finish_reason=length): 71. Verbatim loops among them: 23. Mean response length: 3095 chars, median 22. Answer-only responses (≤60 chars, one line): 605.

## By MMMU discipline

| Discipline | n | Acc | Strict | length | loops | mean chars |
|---|---:|---:|---:|---:|---:|---:|
| Art & Design | 120 | 60.83 | 60.83 | 0 | 0 | 17 |
| Business | 150 | 65.33 | 65.33 | 11 | 5 | 3729 |
| Science | 150 | 52.67 | 52.00 | 7 | 3 | 2610 |
| Health & Medicine | 150 | 64.00 | 64.00 | 2 | 1 | 844 |
| Humanities & Social Science | 120 | 70.83 | 70.83 | 0 | 0 | 48 |
| Tech & Engineering | 210 | 50.00 | 50.00 | 51 | 14 | 8098 |

## By question type

| Type | n | Acc | Strict |
|---|---:|---:|---:|
| multiple-choice | 847 | 62.57 | 62.46 |
| open | 53 | 11.32 | 11.32 |

## Parser

Extraction methods (pipeline): {'official': 818, 'answer_line': 64, 'boxed': 12, 'none': 6}. Explicit-answer rule changed 1 wrong→right and 0 right→wrong. Fallback covered 0 items: 0 wrong→right, 0 right→wrong. Parse failures: strict 6, pipeline 6.
