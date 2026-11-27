# Results for `h_sft_noimg_s3407.jsonl`

Acc = pipeline score (%, two decimals). Strict = official parser only. See `summary.json`.

| No. | Subject | Data Num | Acc | Strict | length | loops |
|---|---|---|---|---|---|---|
| 1 | Accounting | 30 | 40.00 | 40.00 | 0 | 0 |
| 2 | Agriculture | 30 | 53.33 | 50.00 | 0 | 0 |
| 3 | Architecture_and_Engineering | 30 | 40.00 | 36.67 | 2 | 0 |
| 4 | Art | 30 | 40.00 | 40.00 | 1 | 1 |
| 5 | Art_Theory | 30 | 50.00 | 46.67 | 0 | 0 |
| 6 | Basic_Medical_Science | 30 | 70.00 | 73.33 | 0 | 0 |
| 7 | Biology | 30 | 43.33 | 46.67 | 0 | 0 |
| 8 | Chemistry | 30 | 36.67 | 40.00 | 0 | 0 |
| 9 | Clinical_Medicine | 30 | 60.00 | 56.67 | 0 | 0 |
| 10 | Computer_Science | 30 | 30.00 | 23.33 | 0 | 0 |
| 11 | Design | 30 | 50.00 | 46.67 | 0 | 0 |
| 12 | Diagnostics_and_Laboratory_Medicine | 30 | 33.33 | 33.33 | 1 | 1 |
| 13 | Economics | 30 | 53.33 | 53.33 | 1 | 0 |
| 14 | Electronics | 30 | 33.33 | 36.67 | 0 | 0 |
| 15 | Energy_and_Power | 30 | 53.33 | 56.67 | 5 | 3 |
| 16 | Finance | 30 | 36.67 | 33.33 | 0 | 0 |
| 17 | Geography | 30 | 66.67 | 63.33 | 0 | 0 |
| 18 | History | 30 | 66.67 | 63.33 | 0 | 0 |
| 19 | Literature | 30 | 80.00 | 80.00 | 0 | 0 |
| 20 | Manage | 30 | 26.67 | 26.67 | 0 | 0 |
| 21 | Marketing | 30 | 56.67 | 53.33 | 0 | 0 |
| 22 | Materials | 30 | 30.00 | 26.67 | 0 | 0 |
| 23 | Math | 30 | 70.00 | 70.00 | 0 | 0 |
| 24 | Mechanical_Engineering | 30 | 56.67 | 53.33 | 3 | 1 |
| 25 | Music | 30 | 43.33 | 43.33 | 0 | 0 |
| 26 | Pharmacy | 30 | 46.67 | 50.00 | 0 | 0 |
| 27 | Physics | 30 | 53.33 | 53.33 | 0 | 0 |
| 28 | Psychology | 30 | 43.33 | 36.67 | 0 | 0 |
| 29 | Public_Health | 30 | 50.00 | 46.67 | 0 | 0 |
| 30 | Sociology | 30 | 53.33 | 53.33 | 0 | 0 |
| | **Overall (macro avg)** | **900** | **48.89** | **47.78** | 13 | 6 |

Overall = mean of the 30 subject accuracies = 48.8889. Micro = 440 / 900 × 100 = 48.8889 (equal because every subject has 30 items).

| | Overall (MMMU val) |
|---|---:|
| Official (Qwen3-VL Technical Report) | 67.40 |
| Ours, pipeline | 48.89 |
| Ours, strict official parser | 47.78 |
| Δ pipeline − official | -18.51 pp |

## Generation behaviour

| Response group | n | Acc | Strict |
|---|---:|---:|---:|
| answer_only | 0 | n/a | n/a |
| reasoned | 887 | 49.15 | 47.91 |
| truncated | 13 | 30.77 | 38.46 |

Truncated (finish_reason=length): 13. Verbatim loops among them: 6. Mean response length: 4713 chars, median 1665. Answer-only responses (≤60 chars, one line): 0.

## By MMMU discipline

| Discipline | n | Acc | Strict | length | loops | mean chars |
|---|---:|---:|---:|---:|---:|---:|
| Art & Design | 120 | 45.83 | 44.17 | 1 | 1 | 2651 |
| Business | 150 | 42.67 | 41.33 | 1 | 0 | 3727 |
| Science | 150 | 54.00 | 54.67 | 0 | 0 | 3200 |
| Health & Medicine | 150 | 52.00 | 52.00 | 1 | 1 | 3044 |
| Humanities & Social Science | 120 | 60.83 | 58.33 | 0 | 0 | 1612 |
| Tech & Engineering | 210 | 42.38 | 40.48 | 10 | 4 | 10639 |

## By question type

| Type | n | Acc | Strict |
|---|---:|---:|---:|
| multiple-choice | 847 | 51.24 | 49.59 |
| open | 53 | 11.32 | 18.87 |

## Parser

Extraction methods (pipeline): {'answer_line': 885, 'fallback_letter': 11, 'fallback+none': 2, 'official': 2}. Explicit-answer rule changed 19 wrong→right and 8 right→wrong. Fallback covered 13 items: 1 wrong→right, 2 right→wrong. Parse failures: strict 3, pipeline 2.
