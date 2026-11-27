"""Shared data loading, prompt building and condition definitions for Experiment B."""

import json
from pathlib import Path

from PIL import Image, ImageOps

BASE_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
BASE_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
IMAGE_TAG = "<image>"
IGNORE_INDEX = -100

MAIN_PREFIX = "model.visual.merger."
DEEPSTACK_PREFIX = "model.visual.deepstack_merger_list."

# Trainable modules and their expected parameter counts for Qwen3-VL-4B.
# Main merger: LayerNorm(1024) + Linear(4096->4096) + Linear(4096->2560) = 27,271,680.
# Each DeepStack merger uses a post-shuffle LayerNorm(4096) instead: 27,277,824 (x3).
CONDITIONS = {
    "B1": {"prefixes": (MAIN_PREFIX,), "expected": 27_271_680,
           "description": "main merger only"},
    "B2": {"prefixes": (MAIN_PREFIX, DEEPSTACK_PREFIX), "expected": 27_271_680 + 3 * 27_277_824,
           "description": "main merger + 3 DeepStack mergers"},
}


def load_records(jsonl_path, limit=None):
    """Read ShareGPT-style single-image records. Image paths resolve from the JSONL folder."""
    jsonl_path = Path(jsonl_path).resolve()
    records = []
    for n, line in enumerate(jsonl_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        human, assistant = row["conversations"]
        if human["from"] != "human" or assistant["from"] != "gpt":
            raise ValueError(f"{jsonl_path.name}:{n}: expected human then gpt")
        question, answer = human["value"], assistant["value"]
        if question.count(IMAGE_TAG) != 1 or not answer.strip():
            raise ValueError(f"{jsonl_path.name}:{n}: expected one <image> tag and an answer")
        image = (jsonl_path.parent / row["image"]).resolve()
        if not image.is_file():
            raise FileNotFoundError(f"{jsonl_path.name}:{n}: missing image {image}")
        records.append({"image": image, "question": question, "answer": answer.strip(),
                        "source": row.get("source", "unknown"), "index": row.get("index", n - 1)})
        if limit is not None and len(records) == limit:
            break
    if not records:
        raise ValueError(f"no records in {jsonl_path}")
    return records


def open_rgb(path, blind=False):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGBA")
        rgb = Image.new("RGB", image.size, "white")
        rgb.paste(image, mask=image.getchannel("A"))
    if blind:
        # Same size (same number of visual tokens), no visual content.
        rgb = Image.new("RGB", rgb.size, (128, 128, 128))
    return rgb


def user_message(question):
    before, after = question.split(IMAGE_TAG)
    content = [{"type": "text", "text": before}] if before.strip() else []
    content.append({"type": "image"})
    if after.strip():
        content.append({"type": "text", "text": after})
    return [{"role": "user", "content": content}]


def set_max_pixels(processor, max_pixels):
    processor.image_processor.size = {"shortest_edge": 32 * 32, "longest_edge": max_pixels}
