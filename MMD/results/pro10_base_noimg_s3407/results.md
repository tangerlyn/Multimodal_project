# Results for `pro10_base_noimg_s3407.jsonl`

Acc = pipeline score (%, two decimals). Strict = official parser only. See `summary.json`.

| No. | Subject | Data Num | Acc | Strict | length | loops |
|---|---|---|---|---|---|---|
| 1 | Accounting | 58 | 8.62 | 6.90 | 0 | 0 |
| 2 | Agriculture | 60 | 16.67 | 16.67 | 0 | 0 |
| 3 | Architecture_and_Engineering | 60 | 25.00 | 25.00 | 5 | 3 |
| 4 | Art | 53 | 5.66 | 7.55 | 0 | 0 |
| 5 | Art_Theory | 55 | 18.18 | 20.00 | 0 | 0 |
| 6 | Basic_Medical_Science | 52 | 28.85 | 26.92 | 0 | 0 |
| 7 | Biology | 59 | 16.95 | 15.25 | 1 | 0 |
| 8 | Chemistry | 60 | 26.67 | 25.00 | 0 | 0 |
| 9 | Clinical_Medicine | 59 | 32.20 | 27.12 | 0 | 0 |
| 10 | Computer_Science | 60 | 26.67 | 25.00 | 1 | 0 |
| 11 | Design | 60 | 28.33 | 25.00 | 0 | 0 |
| 12 | Diagnostics_and_Laboratory_Medicine | 60 | 16.67 | 11.67 | 0 | 0 |
| 13 | Economics | 59 | 28.81 | 28.81 | 1 | 1 |
| 14 | Electronics | 60 | 43.33 | 45.00 | 4 | 1 |
| 15 | Energy_and_Power | 58 | 25.86 | 25.86 | 12 | 10 |
| 16 | Finance | 60 | 18.33 | 16.67 | 0 | 0 |
| 17 | Geography | 52 | 23.08 | 21.15 | 1 | 1 |
| 18 | History | 56 | 25.00 | 21.43 | 0 | 0 |
| 19 | Literature | 52 | 59.62 | 53.85 | 0 | 0 |
| 20 | Manage | 50 | 30.00 | 26.00 | 0 | 0 |
| 21 | Marketing | 59 | 25.42 | 25.42 | 0 | 0 |
| 22 | Materials | 60 | 18.33 | 16.67 | 2 | 0 |
| 23 | Math | 60 | 50.00 | 50.00 | 2 | 1 |
| 24 | Mechanical_Engineering | 59 | 44.07 | 40.68 | 7 | 2 |
| 25 | Music | 60 | 28.33 | 28.33 | 2 | 2 |
| 26 | Pharmacy | 57 | 33.33 | 29.82 | 1 | 1 |
| 27 | Physics | 60 | 30.00 | 31.67 | 0 | 0 |
| 28 | Psychology | 60 | 35.00 | 33.33 | 0 | 0 |
| 29 | Public_Health | 58 | 20.69 | 22.41 | 1 | 1 |
| 30 | Sociology | 54 | 27.78 | 25.93 | 0 | 0 |
| | **Overall (macro avg)** | **1730** | **27.25** | **25.84** | 40 | 23 |

Overall = mean of the 30 subject accuracies = 27.2485. Micro = 471 / 1730 × 100 = 27.2254.

| | Overall (MMMU val) |
|---|---:|
| Official (Qwen3-VL Technical Report) | 67.40 |
| Ours, pipeline | 27.23 |
| Ours, strict official parser | 25.84 |
| Δ pipeline − official | -40.17 pp |

## Generation behaviour

| Response group | n | Acc | Strict |
|---|---:|---:|---:|
| answer_only | 0 | n/a | n/a |
| reasoned | 1690 | 27.63 | 25.92 |
| truncated | 40 | 10.00 | 22.50 |

Truncated (finish_reason=length): 40. Verbatim loops among them: 23. Mean response length: 4641 chars, median 2032. Answer-only responses (≤60 chars, one line): 0.

## By MMMU discipline

| Discipline | n | Acc | Strict | length | loops | mean chars |
|---|---:|---:|---:|---:|---:|---:|
| Art & Design | 228 | 20.61 | 20.61 | 2 | 2 | 2018 |
| Business | 286 | 22.03 | 20.63 | 1 | 1 | 4071 |
| Science | 291 | 29.55 | 28.87 | 4 | 2 | 4329 |
| Health & Medicine | 286 | 26.22 | 23.43 | 2 | 2 | 3221 |
| Humanities & Social Science | 222 | 36.49 | 33.33 | 0 | 0 | 2089 |
| Tech & Engineering | 417 | 28.54 | 27.82 | 31 | 16 | 9017 |

## By question type

| Type | n | Acc | Strict |
|---|---:|---:|---:|
| multiple-choice | 1730 | 27.23 | 25.84 |
| open | 0 | n/a | n/a |

## Parser

Extraction methods (pipeline): {'answer_line': 1687, 'fallback_letter': 31, 'official': 3, 'fallback+none': 8, 'fallback+official': 1}. Explicit-answer rule changed 43 wrong→right and 14 right→wrong. Fallback covered 40 items: 3 wrong→right, 8 right→wrong. Parse failures: strict 0, pipeline 8.
