"""Train Qwen3-VL-4B's basic merger (B), output adapter (C), or spatial input adapter (D)."""

import argparse
import hashlib
import json
import math
import os
import tempfile
from io import BytesIO
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from PIL import Image, ImageOps

# Same model snapshot as the MMMU evaluator, with a separate training environment.
BASE_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
BASE_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
MERGER_PREFIX = "model.visual.merger."
MERGER_PARAMETERS = 27_271_680
IMAGE_TAG = "<image>"
IGNORE_INDEX = -100


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def positive_float(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("B", "C", "D"), default="B")
    parser.add_argument("--dataset", type=Path, required=True,
                        help="dataset ZIP or the extracted train.jsonl next to images/")
    parser.add_argument("--output-dir", type=Path,
                        help="new directory for the B checkpoint or C/D adapter")
    parser.add_argument("--model-dir", type=Path,
                        help="optional local copy of the same pinned Qwen3-VL snapshot")
    parser.add_argument("--validate-only", action="store_true",
                        help="verify annotations and decode referenced images without Torch or a model")
    parser.add_argument("--limit", type=positive_int, help="only read the first N samples for a smoke run")
    parser.add_argument("--max-steps", type=positive_int, help="override epochs with this many optimizer steps")
    parser.add_argument("--epochs", type=positive_int, default=1)
    parser.add_argument("--learning-rate", type=positive_float, default=1e-5)
    parser.add_argument("--gradient-accumulation-steps", type=positive_int, default=8)
    parser.add_argument("--max-pixels", type=positive_int, default=1_003_520)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    if args.output_dir is None and not args.validate_only:
        parser.error("--output-dir is required for training")
    if args.max_pixels < 32 * 32:
        parser.error("--max-pixels must be at least 1024")
    return args


class SftDataset:
    """Read the official single-image ShareGPT format without extracting the ZIP."""

    def __init__(self, path, archive=None, limit=None):
        self.archive = archive
        if archive is None:
            self.root = path.resolve().parent
            annotations = path.read_bytes()
        else:
            filenames = [info.filename for info in archive.infolist() if not info.is_dir()]
            self.members = set(filenames)
            candidates = [name for name in filenames if PurePosixPath(name).name == "train.jsonl"]
            if len(self.members) != len(filenames) or len(candidates) != 1:
                raise ValueError("ZIP must contain one train.jsonl and no duplicate paths")
            self.root = PurePosixPath(candidates[0]).parent
            annotations = archive.read(candidates[0])

        self.annotation_sha256 = hashlib.sha256(annotations).hexdigest()
        self.samples = []
        for line_number, line in enumerate(annotations.decode("utf-8").split("\n"), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                media = record["image"]
                conversations = record["conversations"]
                human, assistant = conversations
                question, answer = human["value"], assistant["value"]
                if human["from"] != "human" or assistant["from"] != "gpt":
                    raise ValueError("expected human followed by gpt")
                if (question.count(IMAGE_TAG) != 1 or not question.replace(IMAGE_TAG, "").strip()
                        or not answer.strip() or IMAGE_TAG in answer or "<|" in question + answer):
                    raise ValueError("expected one image tag, a question, and a plain assistant response")
                relative = PurePosixPath(media)
                if relative.is_absolute() or ".." in relative.parts or not relative.name:
                    raise ValueError("expected a relative image path within the dataset")
                image = self.root / relative
                if archive is None:
                    image = image.resolve()
                    if self.root not in image.parents or not image.is_file():
                        raise ValueError(f"image missing or outside the dataset: {media}")
                elif image.as_posix() not in self.members:
                    raise ValueError(f"image missing from ZIP: {media}")
            except (KeyError, TypeError, AttributeError, ValueError) as error:
                raise ValueError(f"train.jsonl line {line_number}: {error}") from error
            self.samples.append((image, question, answer))
            if limit is not None and len(self.samples) == limit:
                break
        if not self.samples:
            raise ValueError("dataset contains no training records")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, question, answer = self.samples[index]
        media = path if self.archive is None else BytesIO(self.archive.read(path.as_posix()))
        try:
            with Image.open(media) as image:
                image = ImageOps.exif_transpose(image).convert("RGBA")
                rgb = Image.new("RGB", image.size, "white")
                rgb.paste(image, mask=image.getchannel("A"))
        except OSError as error:
            raise ValueError(f"cannot decode image: {path}") from error
        return rgb, question, answer


class AssistantCollator:
    def __init__(self, processor, max_pixels):
        self.processor = processor
        self.processor.tokenizer.padding_side = "right"
        self.processor.image_processor.size = {"shortest_edge": 32 * 32, "longest_edge": max_pixels}
        self.eos_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")

    def __call__(self, examples):
        images, prompts, transcripts = [], [], []
        for image, question, answer in examples:
            before, after = question.split(IMAGE_TAG)
            content = [{"type": "text", "text": before}] if before else []
            content.append({"type": "image"})
            if after:
                content.append({"type": "text", "text": after})
            user = [{"role": "user", "content": content}]
            assistant = {"role": "assistant", "content": [{"type": "text", "text": answer}]}
            images.append(image)
            prompts.append(self.processor.apply_chat_template(user, tokenize=False, add_generation_prompt=True))
            transcripts.append(self.processor.apply_chat_template(
                user + [assistant], tokenize=False, add_generation_prompt=False))

        batch = self.processor(text=transcripts, images=images, padding=True,
                               return_tensors="pt", add_special_tokens=False)
        # Count the prompt after the processor expands image placeholders.
        prefix = self.processor(text=prompts, images=images, padding=True,
                                return_tensors="pt", add_special_tokens=False)
        labels = batch["input_ids"].clone()
        labels[~batch["attention_mask"].bool()] = IGNORE_INDEX
        for i, mask in enumerate(prefix["attention_mask"]):
            length = int(mask.sum())
            if not prefix["input_ids"][i, :length].equal(batch["input_ids"][i, :length]):
                raise ValueError("chat template produced a different user prefix in the full transcript")
            eos = (labels[i, length:] == self.eos_id).nonzero(as_tuple=True)[0]
            if len(eos) != 1 or int(eos[0]) < 1:
                raise ValueError("assistant turn must have an answer followed by one end-of-turn token")
            labels[i, :length] = IGNORE_INDEX
            labels[i, length + int(eos[0]) + 1:] = IGNORE_INDEX
        batch["labels"] = labels
        batch["use_cache"] = False
        return batch


def train(args, dataset):
    import torch
    import transformers
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Trainer, TrainerCallback, TrainingArguments, set_seed

    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or torch.cuda.device_count() != 1:
        raise SystemExit("Use one visible CUDA GPU, for example CUDA_VISIBLE_DEVICES=0.")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("Experiment B requires a CUDA GPU with BF16 support.")

    destination = args.output_dir.resolve()
    if destination.exists():
        raise SystemExit(f"Checkpoint directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = BASE_MODEL
    revision = BASE_REVISION
    if args.model_dir is not None:
        if not args.model_dir.is_dir():
            raise SystemExit(f"Local model directory does not exist: {args.model_dir}")
        source = str(args.model_dir.resolve())
        revision = None

    set_seed(args.seed)
    processor = AutoProcessor.from_pretrained(source, revision=revision)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        source, revision=revision, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.requires_grad_(False)
    if args.experiment in ("C", "D"):
        from residual_merger import ADAPTER_PREFIX, base_merger_digest, save_residual

        base_merger_sha = base_merger_digest(model)
        prefix = ADAPTER_PREFIX
        if args.experiment == "C":
            from residual_merger import ADAPTER_PARAMETERS, EXPERIMENT_C, attach_residual

            experiment, expected_count = EXPERIMENT_C, ADAPTER_PARAMETERS
            trainable_module = attach_residual(model, dtype=torch.float32)
        else:
            from spatial_merger import EXPERIMENT_D, SPATIAL_PARAMETERS, attach_spatial

            experiment, expected_count = EXPERIMENT_D, SPATIAL_PARAMETERS
            trainable_module = attach_spatial(model, dtype=torch.float32)
    else:
        experiment = "B_basic_merger_only"
        trainable_module = model.model.visual.merger
        expected_count, prefix = MERGER_PARAMETERS, MERGER_PREFIX

    original_state = {name: value.detach().cpu().clone() for name, value in trainable_module.state_dict().items()}
    trainable_module.float().requires_grad_(True)
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    count = sum(p.numel() for p in trainable.values())
    if count != expected_count or any(not name.startswith(prefix) for name in trainable):
        raise RuntimeError(f"unexpected trainable parameters ({count:,}): {list(trainable)}")
    print(f"Experiment {args.experiment}: training {count:,} parameters in {prefix.rstrip('.')}", flush=True)

    class CheckMergerGradient(TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            if state.global_step == 0:
                gradients = [p.grad for p in trainable.values()]
                if any(g is None for g in gradients):
                    raise RuntimeError("a trainable parameter is disconnected from the loss")
                if not all(bool(torch.isfinite(g).all()) for g in gradients):
                    raise RuntimeError("non-finite gradients")
                if not any(bool(torch.count_nonzero(g)) for g in gradients):
                    raise RuntimeError("trainable module received only zero gradients")

    with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as temp:
        training_args = TrainingArguments(
            output_dir=str(Path(temp) / "training"),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps or -1,
            learning_rate=args.learning_rate,
            warmup_ratio=0.0 if args.max_steps == 1 else 0.03,
            weight_decay=0.01,
            optim="adamw_torch",
            bf16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            remove_unused_columns=False,
            dataloader_num_workers=0,
            logging_first_step=True,
            logging_steps=10,
            save_strategy="no",
            report_to="none",
            seed=args.seed,
            data_seed=args.seed,
        )
        trainer = Trainer(model=model, args=training_args, train_dataset=dataset,
                          data_collator=AssistantCollator(processor, args.max_pixels),
                          callbacks=[CheckMergerGradient()])
        result = trainer.train()
        if not math.isfinite(result.training_loss):
            raise RuntimeError("training loss is non-finite")
        # Validate the actual exported values after casting back to evaluation precision.
        trainable_module.bfloat16()
        if all(torch.equal(value.cpu(), original_state[name].bfloat16())
               for name, value in trainable_module.state_dict().items()):
            raise RuntimeError("training did not change any weights at the saved BF16 precision")
        checkpoint = Path(temp) / "checkpoint"
        if args.experiment in ("C", "D"):
            checkpoint.mkdir()
            save_residual(trainable_module, checkpoint)
        else:
            trainer.save_model(str(checkpoint))
            # Training-only resize settings must not change the B evaluation processor.
            AutoProcessor.from_pretrained(source, revision=revision).save_pretrained(checkpoint)
        meta = {
            "experiment": experiment,
            "base_model": source,
            "base_revision": revision,
            "dataset": str(args.dataset.resolve()),
            "annotation_sha256": dataset.annotation_sha256,
            "examples": len(dataset),
            "trainable_parameters": count,
            "trainable_names": list(trainable),
            "seed": args.seed,
            "max_pixels": args.max_pixels,
            "learning_rate": args.learning_rate,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "epochs_requested": args.epochs,
            "max_steps_requested": args.max_steps,
            "optimizer_steps": result.global_step,
            "metrics": result.metrics,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        }
        if args.experiment in ("C", "D"):
            meta["base_merger_sha256"] = base_merger_sha
        (checkpoint / "training_run.json").write_text(json.dumps(meta, indent=2) + "\n")
        checkpoint.rename(destination)
    print(f"Checkpoint ready: {destination}")


def main():
    args = parse_args()

    def run(dataset):
        print(f"Loaded {len(dataset):,} samples from {args.dataset}", flush=True)
        if args.validate_only:
            for index in range(len(dataset)):
                dataset[index]
            print(f"Validated all {len(dataset):,} referenced images. JSONL SHA-256: {dataset.annotation_sha256}")
        else:
            train(args, dataset)

    if args.dataset.suffix.lower() == ".zip":
        with ZipFile(args.dataset) as archive:
            run(SftDataset(args.dataset, archive=archive, limit=args.limit))
    else:
        run(SftDataset(args.dataset, limit=args.limit))


if __name__ == "__main__":
    main()
