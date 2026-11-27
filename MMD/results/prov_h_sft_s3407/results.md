# Results for `prov_h_sft_s3407.jsonl`

Acc = pipeline score (%, two decimals). Strict = official parser only. See `summary.json`.

| No. | Subject | Data Num | Acc | Strict | length | loops |
|---|---|---|---|---|---|---|
| 1 | Accounting | 58 | 77.59 | 74.14 | 1 | 0 |
| 2 | Agriculture | 60 | 25.00 | 25.00 | 0 | 0 |
| 3 | Architecture_and_Engineering | 60 | 50.00 | 46.67 | 12 | 2 |
| 4 | Art | 53 | 47.17 | 45.28 | 2 | 1 |
| 5 | Art_Theory | 55 | 60.00 | 54.55 | 1 | 1 |
| 6 | Basic_Medical_Science | 52 | 51.92 | 40.38 | 0 | 0 |
| 7 | Biology | 59 | 42.37 | 32.20 | 3 | 3 |
| 8 | Chemistry | 60 | 46.67 | 45.00 | 4 | 3 |
| 9 | Clinical_Medicine | 59 | 37.29 | 16.95 | 1 | 1 |
| 10 | Computer_Science | 60 | 56.67 | 50.00 | 3 | 1 |
| 11 | Design | 60 | 58.33 | 51.67 | 1 | 1 |
| 12 | Diagnostics_and_Laboratory_Medicine | 60 | 18.33 | 16.67 | 0 | 0 |
| 13 | Economics | 59 | 69.49 | 69.49 | 0 | 0 |
| 14 | Electronics | 60 | 58.33 | 58.33 | 10 | 2 |
| 15 | Energy_and_Power | 58 | 50.00 | 46.55 | 12 | 4 |
| 16 | Finance | 60 | 78.33 | 78.33 | 2 | 2 |
| 17 | Geography | 52 | 32.69 | 23.08 | 1 | 1 |
| 18 | History | 56 | 57.14 | 46.43 | 0 | 0 |
| 19 | Literature | 52 | 67.31 | 42.31 | 0 | 0 |
| 20 | Manage | 50 | 60.00 | 52.00 | 0 | 0 |
| 21 | Marketing | 59 | 72.88 | 67.80 | 1 | 1 |
| 22 | Materials | 60 | 51.67 | 50.00 | 7 | 4 |
| 23 | Math | 60 | 61.67 | 58.33 | 3 | 3 |
| 24 | Mechanical_Engineering | 59 | 33.90 | 32.20 | 11 | 4 |
| 25 | Music | 60 | 28.33 | 25.00 | 11 | 9 |
| 26 | Pharmacy | 57 | 70.18 | 66.67 | 2 | 1 |
| 27 | Physics | 60 | 63.33 | 60.00 | 0 | 0 |
| 28 | Psychology | 60 | 46.67 | 38.33 | 2 | 1 |
| 29 | Public_Health | 58 | 67.24 | 60.34 | 1 | 0 |
| 30 | Sociology | 54 | 50.00 | 46.30 | 0 | 0 |
| | **Overall (macro avg)** | **1730** | **53.02** | **47.33** | 91 | 45 |

Overall = mean of the 30 subject accuracies = 53.0168. Micro = 916 / 1730 × 100 = 52.9480.

| | Overall (MMMU val) |
|---|---:|
| Official (Qwen3-VL Technical Report) | 67.40 |
| Ours, pipeline | 52.95 |
| Ours, strict official parser | 47.40 |
| Δ pipeline − official | -14.45 pp |

## Generation behaviour

| Response group | n | Acc | Strict |
|---|---:|---:|---:|
| answer_only | 0 | n/a | n/a |
| reasoned | 1639 | 54.97 | 49.48 |
| truncated | 91 | 16.48 | 9.89 |

Truncated (finish_reason=length): 91. Verbatim loops among them: 45. Mean response length: 7939 chars, median 2640. Answer-only responses (≤60 chars, one line): 0.

## By MMMU discipline

| Discipline | n | Acc | Strict | length | loops | mean chars |
|---|---:|---:|---:|---:|---:|---:|
| Art & Design | 228 | 48.25 | 43.86 | 15 | 12 | 6078 |
| Business | 286 | 72.03 | 68.88 | 4 | 3 | 5560 |
| Science | 291 | 49.83 | 44.33 | 11 | 10 | 6828 |
| Health & Medicine | 286 | 48.60 | 39.86 | 4 | 2 | 4349 |
| Humanities & Social Science | 222 | 54.95 | 43.24 | 2 | 1 | 3321 |
| Tech & Engineering | 417 | 46.52 | 44.12 | 55 | 17 | 16282 |

## By question type

| Type | n | Acc | Strict |
|---|---:|---:|---:|
| multiple-choice | 1730 | 52.95 | 47.40 |
| open | 0 | n/a | n/a |

## Parser

Extraction methods (pipeline): {'answer_line': 1633, 'fallback_letter': 91, 'official': 5, 'fallback+none': 1}. Explicit-answer rule changed 111 wrong→right and 20 right→wrong. Fallback covered 92 items: 10 wrong→right, 5 right→wrong. Parse failures: strict 2, pipeline 1.
