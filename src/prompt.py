"""Prompt construction.

Text format: MMMU official repo (MMMU-Benchmark/MMMU @ 268471d0d488258990025331c7528359c324aa25),
`mmmu/configs/llava1.5.yaml` + `construct_prompt` in `mmmu/utils/data_utils.py`, with an empty
task instruction. The format strings and the "(A) option\\n" rendering are copied verbatim.

Image placement is ours: the official code feeds only `image_1`; we replace each `<image N>`
token with the actual image at that position so multi-image items keep all their images.
"""
import hashlib
import math
import re

from PIL import Image

MC_FORMAT = "{}\n\n{}\n\nAnswer with the option's letter from the given choices directly."
OPEN_FORMAT = "{}\n\nAnswer the question using a single word or phrase."

IMAGE_TOKEN = re.compile(r"<image (\d+)>")


def prompt_hash():
    return hashlib.sha256((MC_FORMAT + "\x00" + OPEN_FORMAT).encode()).hexdigest()[:16]


def build_text(sample):
    if sample["question_type"] == "multiple-choice":
        example = ""
        for i, option in enumerate(sample["options"]):
            example += f"({chr(ord('A') + i)}) {option}\n"
        return MC_FORMAT.format(sample["question"], example)
    return OPEN_FORMAT.format(sample["question"])


def resize_image(img, max_pixels):
    """RGB; downscale (bicubic, aspect kept) only when width*height exceeds max_pixels.

    Transparent pixels are flattened onto white (as qwen-vl-utils does); a plain
    convert("RGB") would turn a transparent diagram background black.
    """
    if img.mode in ("RGBA", "LA", "P") or "transparency" in img.info:
        rgba = img.convert("RGBA")
        img = Image.new("RGB", rgba.size, (255, 255, 255))
        img.paste(rgba, mask=rgba.split()[3])
    else:
        img = img.convert("RGB")
    w, h = img.size
    if w * h > max_pixels:
        scale = math.sqrt(max_pixels / (w * h))
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)
    return img


def build_content(text, images, max_pixels):
    """Split `text` on `<image N>` tokens into chat content items.

    Each image is inserted once, at its first reference. A repeated reference, or one to a
    missing column, stays as literal text. Images never referenced are put first.
    Returns (content, ordered_images, warnings).
    """
    content, ordered, used, warnings = [], [], set(), []

    def add_text(s):
        if not s:
            return
        if content and content[-1]["type"] == "text":
            content[-1]["text"] += s
        else:
            content.append({"type": "text", "text": s})

    def add_image(n):
        used.add(n)
        ordered.append(resize_image(images[n], max_pixels))
        content.append({"type": "image"})

    referenced = {int(m.group(1)) for m in IMAGE_TOKEN.finditer(text)}
    for n in sorted(set(images) - referenced):
        warnings.append(f"image_{n} never referenced; placed first")
        add_image(n)

    pos = 0
    for m in IMAGE_TOKEN.finditer(text):
        add_text(text[pos:m.start()])
        n = int(m.group(1))
        if n in images and n not in used:
            add_image(n)
        else:
            if n not in images:
                warnings.append(f"<image {n}> referenced but image_{n} is null")
            add_text(m.group(0))
        pos = m.end()
    add_text(text[pos:])
    return content, ordered, warnings


def render(tokenizer, content):
    """Exact string sent to the model: single user turn, model chat template, no system prompt."""
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
