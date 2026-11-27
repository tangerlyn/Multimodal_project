# Results for `base_noimg_s3407.jsonl`

Acc = pipeline score (%, two decimals). Strict = official parser only. See `summary.json`.

| No. | Subject | Data Num | Acc | Strict | length | loops |
|---|---|---|---|---|---|---|
| 1 | Accounting | 30 | 20.00 | 20.00 | 1 | 1 |
| 2 | Agriculture | 30 | 46.67 | 40.00 | 0 | 0 |
| 3 | Architecture_and_Engineering | 30 | 40.00 | 36.67 | 3 | 2 |
| 4 | Art | 30 | 40.00 | 36.67 | 1 | 1 |
| 5 | Art_Theory | 30 | 50.00 | 50.00 | 0 | 0 |
| 6 | Basic_Medical_Science | 30 | 70.00 | 66.67 | 0 | 0 |
| 7 | Biology | 30 | 53.33 | 53.33 | 0 | 0 |
| 8 | Chemistry | 30 | 30.00 | 30.00 | 1 | 0 |
| 9 | Clinical_Medicine | 30 | 63.33 | 60.00 | 0 | 0 |
| 10 | Computer_Science | 30 | 36.67 | 33.33 | 0 | 0 |
| 11 | Design | 30 | 46.67 | 46.67 | 0 | 0 |
| 12 | Diagnostics_and_Laboratory_Medicine | 30 | 36.67 | 36.67 | 0 | 0 |
| 13 | Economics | 30 | 40.00 | 40.00 | 0 | 0 |
| 14 | Electronics | 30 | 36.67 | 36.67 | 1 | 1 |
| 15 | Energy_and_Power | 30 | 53.33 | 50.00 | 9 | 5 |
| 16 | Finance | 30 | 16.67 | 16.67 | 0 | 0 |
| 17 | Geography | 30 | 60.00 | 60.00 | 1 | 1 |
| 18 | History | 30 | 66.67 | 63.33 | 0 | 0 |
| 19 | Literature | 30 | 83.33 | 83.33 | 0 | 0 |
| 20 | Manage | 30 | 36.67 | 36.67 | 1 | 0 |
| 21 | Marketing | 30 | 53.33 | 53.33 | 0 | 0 |
| 22 | Materials | 30 | 20.00 | 20.00 | 1 | 1 |
| 23 | Math | 30 | 66.67 | 73.33 | 0 | 0 |
| 24 | Mechanical_Engineering | 30 | 50.00 | 43.33 | 4 | 2 |
| 25 | Music | 30 | 26.67 | 26.67 | 2 | 2 |
| 26 | Pharmacy | 30 | 50.00 | 53.33 | 0 | 0 |
| 27 | Physics | 30 | 53.33 | 56.67 | 0 | 0 |
| 28 | Psychology | 30 | 43.33 | 40.00 | 0 | 0 |
| 29 | Public_Health | 30 | 46.67 | 46.67 | 1 | 0 |
| 30 | Sociology | 30 | 50.00 | 50.00 | 0 | 0 |
| | **Overall (macro avg)** | **900** | **46.22** | **45.33** | 26 | 16 |

Overall = mean of the 30 subject accuracies = 46.2222. Micro = 416 / 900 × 100 = 46.2222 (equal because every subject has 30 items).

| | Overall (MMMU val) |
|---|---:|
| Official (Qwen3-VL Technical Report) | 67.40 |
| Ours, pipeline | 46.22 |
| Ours, strict official parser | 45.33 |
| Δ pipeline − official | -21.18 pp |

## Generation behaviour

| Response group | n | Acc | Strict |
|---|---:|---:|---:|
| answer_only | 0 | n/a | n/a |
| reasoned | 874 | 46.68 | 46.00 |
| truncated | 26 | 30.77 | 23.08 |

Truncated (finish_reason=length): 26. Verbatim loops among them: 16. Mean response length: 6102 chars, median 1722. Answer-only responses (≤60 chars, one line): 0.

## By MMMU discipline

| Discipline | n | Acc | Strict | length | loops | mean chars |
|---|---:|---:|---:|---:|---:|---:|
| Art & Design | 120 | 40.83 | 40.00 | 3 | 3 | 4707 |
| Business | 150 | 33.33 | 33.33 | 2 | 1 | 4596 |
| Science | 150 | 52.67 | 54.67 | 2 | 1 | 5212 |
| Health & Medicine | 150 | 53.33 | 52.67 | 1 | 0 | 2767 |
| Humanities & Social Science | 120 | 60.83 | 59.17 | 0 | 0 | 1696 |
| Tech & Engineering | 210 | 40.48 | 37.14 | 18 | 11 | 13509 |

## By question type

| Type | n | Acc | Strict |
|---|---:|---:|---:|
| multiple-choice | 847 | 48.64 | 47.23 |
| open | 53 | 7.55 | 15.09 |

## Parser

Extraction methods (pipeline): {'answer_line': 872, 'fallback_letter': 17, 'fallback+none': 6, 'fallback+official': 3, 'official': 2}. Explicit-answer rule changed 12 wrong→right and 6 right→wrong. Fallback covered 26 items: 4 wrong→right, 2 right→wrong. Parse failures: strict 6, pipeline 6.
