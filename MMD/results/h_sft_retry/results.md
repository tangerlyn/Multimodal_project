# Results for `h_sft_retry.jsonl`

Acc = pipeline score (%, two decimals). Strict = official parser only. See `summary.json`.

| No. | Subject | Data Num | Acc | Strict | length | loops |
|---|---|---|---|---|---|---|
| 1 | Accounting | 30 | 83.33 | 83.33 | 0 | 0 |
| 2 | Agriculture | 30 | 40.00 | 43.33 | 0 | 0 |
| 3 | Architecture_and_Engineering | 30 | 70.00 | 50.00 | 1 | 1 |
| 4 | Art | 30 | 70.00 | 66.67 | 0 | 0 |
| 5 | Art_Theory | 30 | 86.67 | 80.00 | 0 | 0 |
| 6 | Basic_Medical_Science | 30 | 66.67 | 66.67 | 0 | 0 |
| 7 | Biology | 30 | 43.33 | 46.67 | 0 | 0 |
| 8 | Chemistry | 30 | 56.67 | 46.67 | 0 | 0 |
| 9 | Clinical_Medicine | 30 | 76.67 | 76.67 | 0 | 0 |
| 10 | Computer_Science | 30 | 56.67 | 56.67 | 0 | 0 |
| 11 | Design | 30 | 80.00 | 80.00 | 0 | 0 |
| 12 | Diagnostics_and_Laboratory_Medicine | 30 | 36.67 | 40.00 | 0 | 0 |
| 13 | Economics | 30 | 86.67 | 86.67 | 0 | 0 |
| 14 | Electronics | 30 | 53.33 | 56.67 | 0 | 0 |
| 15 | Energy_and_Power | 30 | 63.33 | 60.00 | 3 | 2 |
| 16 | Finance | 30 | 80.00 | 73.33 | 0 | 0 |
| 17 | Geography | 30 | 53.33 | 56.67 | 0 | 0 |
| 18 | History | 30 | 73.33 | 73.33 | 0 | 0 |
| 19 | Literature | 30 | 76.67 | 76.67 | 0 | 0 |
| 20 | Manage | 30 | 60.00 | 53.33 | 0 | 0 |
| 21 | Marketing | 30 | 83.33 | 80.00 | 0 | 0 |
| 22 | Materials | 30 | 60.00 | 46.67 | 3 | 0 |
| 23 | Math | 30 | 73.33 | 70.00 | 0 | 0 |
| 24 | Mechanical_Engineering | 30 | 60.00 | 60.00 | 2 | 1 |
| 25 | Music | 30 | 33.33 | 30.00 | 3 | 2 |
| 26 | Pharmacy | 30 | 73.33 | 70.00 | 1 | 1 |
| 27 | Physics | 30 | 80.00 | 76.67 | 0 | 0 |
| 28 | Psychology | 30 | 80.00 | 73.33 | 0 | 0 |
| 29 | Public_Health | 30 | 80.00 | 80.00 | 0 | 0 |
| 30 | Sociology | 30 | 56.67 | 53.33 | 0 | 0 |
| | **Overall (macro avg)** | **900** | **66.44** | **63.78** | 13 | 7 |

Overall = mean of the 30 subject accuracies = 66.4444. Micro = 598 / 900 × 100 = 66.4444 (equal because every subject has 30 items).

| | Overall (MMMU val) |
|---|---:|
| Official (Qwen3-VL Technical Report) | 67.40 |
| Ours, pipeline | 66.44 |
| Ours, strict official parser | 63.78 |
| Δ pipeline − official | -0.96 pp |

## Generation behaviour

| Response group | n | Acc | Strict |
|---|---:|---:|---:|
| answer_only | 0 | n/a | n/a |
| reasoned | 887 | 66.85 | 64.60 |
| truncated | 13 | 38.46 | 7.69 |

Truncated (finish_reason=length): 13. Verbatim loops among them: 7. Mean response length: 6615 chars, median 1723. Answer-only responses (≤60 chars, one line): 0.

## By MMMU discipline

| Discipline | n | Acc | Strict | length | loops | mean chars |
|---|---:|---:|---:|---:|---:|---:|
| Art & Design | 120 | 67.50 | 64.17 | 3 | 2 | 4543 |
| Business | 150 | 78.67 | 75.33 | 0 | 0 | 4201 |
| Science | 150 | 61.33 | 59.33 | 0 | 0 | 5898 |
| Health & Medicine | 150 | 66.67 | 66.67 | 1 | 1 | 3082 |
| Humanities & Social Science | 120 | 71.67 | 69.17 | 0 | 0 | 1750 |
| Tech & Engineering | 210 | 57.62 | 53.33 | 9 | 4 | 15340 |

## By question type

| Type | n | Acc | Strict |
|---|---:|---:|---:|
| multiple-choice | 847 | 67.77 | 65.05 |
| open | 53 | 45.28 | 43.40 |

## Parser

Extraction methods (pipeline): {'answer_line': 884, 'official': 3, 'fallback_letter': 10, 'fallback+none': 2, 'fallback+official': 1}. Explicit-answer rule changed 28 wrong→right and 8 right→wrong. Fallback covered 13 items: 5 wrong→right, 1 right→wrong. Parse failures: strict 3, pipeline 2.
