---
base_model: Qwen/Qwen3-VL-4B-Instruct
library_name: peft
tags:
- base_model:adapter:Qwen/Qwen3-VL-4B-Instruct
- lora
- mmmu
---

# H-SFT LoRA for Qwen3-VL-4B-Instruct

LoRA adapter behind every H-SFT number in this repository. Base model: `Qwen/Qwen3-VL-4B-Instruct` at revision
`ebb281ec70b05090aa6165b016eac8ec08e71b17`.

| | |
|---|---|
| Targets | language-model attention q, k, v, o (vision encoder and connector untouched) |
| Rank / alpha | 16 / 32, 11,796,480 trainable parameters |
| Data | 637 correct solutions written by the base model on ViRL39K questions (`data/build2/pairs_train.jsonl`) |
| Training | lr 5e-5, 1 epoch (80 optimizer steps), gradient accumulation 8, seed 3407, images 1.0–2.0 M pixels, 2,513 s on one A100 40 GB |
| Validation NLL per token (correct solutions) | 0.275 → 0.229 |
| PEFT | 0.17.1 |

Full record: [`training_run.json`](training_run.json). Use: `bash run.sh merge` (from the repository root) merges it into
a full checkpoint for vLLM. Results and method: [repository README](../../README.md).
