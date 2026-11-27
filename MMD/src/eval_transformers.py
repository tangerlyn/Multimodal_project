"""Serial Transformers generation for controlled A/B/C comparisons."""

from dataclasses import dataclass
from types import SimpleNamespace

import torch
from transformers import AutoProcessor, LogitsProcessor, Qwen3VLForConditionalGeneration


@dataclass
class SamplingParams:
    max_tokens: int
    seed: int
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0


class PresencePenalty(LogitsProcessor):
    def __init__(self, prompt_length, penalty):
        self.prompt_length = prompt_length
        self.penalty = penalty

    def __call__(self, input_ids, scores):
        generated = input_ids[0, self.prompt_length:]
        scores[0, generated.unique()] -= self.penalty
        return scores


class TransformersEngine:
    def __init__(self, model_dir, args, revision):
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Transformers evaluation requires a BF16-capable CUDA GPU")
        self.max_model_len = args.max_model_len
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_dir, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
        if args.residual_adapter:
            from residual_merger import load_residual

            load_residual(self.model, args.residual_adapter, args.model_path, revision)

    @torch.inference_mode()
    def generate(self, requests, params, lora_request=None):
        # Existing MMMU code supplies one shared parameter object for greedy fallback.
        if isinstance(params, SamplingParams):
            params = [params] * len(requests)
        return [self._generate_one(request, recipe) for request, recipe in zip(requests, params)]

    def _generate_one(self, request, params):
        images = request.get("multi_modal_data", {}).get("image")
        inputs = self.processor(text=request["prompt"], images=images, return_tensors="pt",
                                add_special_tokens=False).to(self.model.device)
        prompt_length = inputs.input_ids.shape[1]
        budget = min(params.max_tokens, self.max_model_len - prompt_length)
        if budget <= 0:
            raise ValueError(f"prompt ({prompt_length} tokens) exhausts --max_model_len")
        sample = params.temperature > 0
        torch.manual_seed(params.seed)
        logits_processors = []
        if params.presence_penalty:
            logits_processors.append(PresencePenalty(prompt_length, params.presence_penalty))
        tokens = self.model.generate(
            **inputs, use_cache=True, max_new_tokens=budget, do_sample=sample,
            temperature=params.temperature if sample else None,
            top_k=params.top_k if sample else None, top_p=params.top_p if sample else None,
            repetition_penalty=params.repetition_penalty, logits_processor=logits_processors,
        )[0, prompt_length:].tolist()
        eos = self.model.generation_config.eos_token_id
        stop_ids = eos if isinstance(eos, list) else [eos]
        finish = "length" if len(tokens) == budget and tokens[-1] not in stop_ids else "stop"
        completion = SimpleNamespace(text=self.processor.tokenizer.decode(tokens, skip_special_tokens=True),
                                     finish_reason=finish)
        return SimpleNamespace(outputs=[completion])
