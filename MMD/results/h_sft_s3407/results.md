# Results for `h_sft_s3407.jsonl`

Acc = pipeline score (%, two decimals). Strict = official parser only. See `summary.json`.

| No. | Subject | Data Num | Acc | Strict | length | loops |
|---|---|---|---|---|---|---|
| 1 | Accounting | 30 | 83.33 | 76.67 | 3 | 3 |
| 2 | Agriculture | 30 | 40.00 | 43.33 | 0 | 0 |
| 3 | Architecture_and_Engineering | 30 | 70.00 | 43.33 | 7 | 3 |
| 4 | Art | 30 | 70.00 | 66.67 | 0 | 0 |
| 5 | Art_Theory | 30 | 86.67 | 80.00 | 0 | 0 |
| 6 | Basic_Medical_Science | 30 | 66.67 | 66.67 | 0 | 0 |
| 7 | Biology | 30 | 43.33 | 50.00 | 1 | 0 |
| 8 | Chemistry | 30 | 53.33 | 46.67 | 1 | 1 |
| 9 | Clinical_Medicine | 30 | 76.67 | 76.67 | 0 | 0 |
| 10 | Computer_Science | 30 | 53.33 | 53.33 | 1 | 1 |
| 11 | Design | 30 | 80.00 | 80.00 | 0 | 0 |
| 12 | Diagnostics_and_Laboratory_Medicine | 30 | 36.67 | 40.00 | 0 | 0 |
| 13 | Economics | 30 | 86.67 | 86.67 | 0 | 0 |
| 14 | Electronics | 30 | 53.33 | 60.00 | 1 | 1 |
| 15 | Energy_and_Power | 30 | 63.33 | 56.67 | 9 | 3 |
| 16 | Finance | 30 | 80.00 | 73.33 | 0 | 0 |
| 17 | Geography | 30 | 50.00 | 53.33 | 1 | 0 |
| 18 | History | 30 | 73.33 | 70.00 | 1 | 1 |
| 19 | Literature | 30 | 76.67 | 76.67 | 0 | 0 |
| 20 | Manage | 30 | 60.00 | 53.33 | 0 | 0 |
| 21 | Marketing | 30 | 83.33 | 80.00 | 0 | 0 |
| 22 | Materials | 30 | 60.00 | 43.33 | 7 | 1 |
| 23 | Math | 30 | 73.33 | 70.00 | 0 | 0 |
| 24 | Mechanical_Engineering | 30 | 60.00 | 56.67 | 5 | 2 |
| 25 | Music | 30 | 36.67 | 30.00 | 5 | 3 |
| 26 | Pharmacy | 30 | 73.33 | 73.33 | 2 | 2 |
| 27 | Physics | 30 | 80.00 | 76.67 | 1 | 0 |
| 28 | Psychology | 30 | 80.00 | 73.33 | 0 | 0 |
| 29 | Public_Health | 30 | 80.00 | 80.00 | 2 | 0 |
| 30 | Sociology | 30 | 56.67 | 53.33 | 0 | 0 |
| | **Overall (macro avg)** | **900** | **66.22** | **63.00** | 47 | 21 |

Overall = mean of the 30 subject accuracies = 66.2222. Micro = 596 / 900 × 100 = 66.2222 (equal because every subject has 30 items).

| | Overall (MMMU val) |
|---|---:|
| Official (Qwen3-VL Technical Report) | 67.40 |
| Ours, pipeline | 66.22 |
| Ours, strict official parser | 63.00 |
| Δ pipeline − official | -1.18 pp |

## Generation behaviour

| Response group | n | Acc | Strict |
|---|---:|---:|---:|
| answer_only | 0 | n/a | n/a |
| reasoned | 853 | 67.41 | 65.06 |
| truncated | 47 | 44.68 | 25.53 |

Truncated (finish_reason=length): 47. Verbatim loops among them: 21. Mean response length: 9957 chars, median 1728. Answer-only responses (≤60 chars, one line): 0.

## By MMMU discipline

| Discipline | n | Acc | Strict | length | loops | mean chars |
|---|---:|---:|---:|---:|---:|---:|
| Art & Design | 120 | 68.33 | 64.17 | 5 | 3 | 6415 |
| Business | 150 | 78.67 | 74.00 | 3 | 3 | 5918 |
| Science | 150 | 60.00 | 59.33 | 4 | 1 | 8943 |
| Health & Medicine | 150 | 66.67 | 67.33 | 4 | 2 | 5056 |
| Humanities & Social Science | 120 | 71.67 | 68.33 | 1 | 1 | 2668 |
| Tech & Engineering | 210 | 57.14 | 50.95 | 30 | 11 | 23256 |

## By question type

| Type | n | Acc | Strict |
|---|---:|---:|---:|
| multiple-choice | 847 | 67.65 | 64.23 |
| open | 53 | 43.40 | 43.40 |

## Parser

Extraction methods (pipeline): {'answer_line': 850, 'fallback_letter': 40, 'official': 3, 'fallback+answer_line': 2, 'fallback+none': 4, 'fallback+official': 1}. Explicit-answer rule changed 28 wrong→right and 8 right→wrong. Fallback covered 47 items: 17 wrong→right, 8 right→wrong. Parse failures: strict 7, pipeline 4.
