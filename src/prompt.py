"""Prompt construction.

Two text styles, selected with --prompt_style:

* ``mmmu`` (baseline): MMMU official repo (MMMU-Benchmark/MMMU @ 268471d0d488258990025331c7528359c324aa25),
  `mmmu/configs/llava1.5.yaml` + `construct_prompt` in `mmmu/utils/data_utils.py`, empty task
  instruction. Format strings and the "(A) option\\n" rendering are copied verbatim.
* ``vlmevalkit``: the prompt Qwen's own MMMU evaluation uses (QwenLM/Qwen3-VL `evaluation/mmmu/run_mmmu.py`
  `build_mmmu_prompt`, identical to VLMEvalKit `vlmeval/vlm/qwen3_vl/prompt.py` `_build_mmmu_prompt`).
  The HF dataset has no `hint` column, so the "Hint:" line never appears.

Two image placements, selected with --image_position:

* ``inline`` (baseline): each `<image N>` token is replaced by the image at that position.
* ``first``: all images precede the text, in column order, and `<image N>` stays as literal text.
  This is what the Qwen / VLMEvalKit scripts do.

--cot qwen appends the default chain-of-thought sentence of Qwen's own MMMU script
(QwenLM/Qwen3-VL `evaluation/mmmu/run_mmmu.py`, `--use-cot` with no custom prompt), verbatim. It leaves
the decision to reason to the model. --cot forced appends our own sentence that requires the reasoning.

--answer_line appends one sentence asking for a final ``Answer: X`` line. It gives the model a
termination target (looping responses are ones that never commit) and a parser anchor.

Resizing follows qwen_vl_utils `smart_resize` (factor 32 = patch 16 x merge 2 for Qwen3-VL): both
sides are rounded to multiples of the factor and the pixel count is brought into [min_pixels,
max_pixels]. With min_pixels=None (baseline) no upscaling happens and only the max is enforced.
"""
import hashlib
import math
import re

from PIL import Image

PROMPT_STYLES = ("mmmu", "vlmevalkit")
IMAGE_POSITIONS = ("inline", "first")
IMAGE_FACTOR = 32  # Qwen3-VL: patch_size 16 * spatial_merge_size 2

MC_FORMAT = "{}\n\n{}\n\nAnswer with the option's letter from the given choices directly."
OPEN_FORMAT = "{}\n\nAnswer the question using a single word or phrase."
# QwenLM/Qwen3-VL evaluation/mmmu/run_mmmu.py, run_inference(): the text appended when --use-cot is
# given without --cot-prompt. Copied verbatim, including the leading space and the em dash.
COT_PROMPT = (" If you are uncertain or the problem is too complex, make a reasoned guess based on the "
              "information provided. Avoid repeating steps indefinitely\u2014provide your best guess even if "
              "unsure. Determine whether to think step by step based on the difficulty of the question, "
              "considering all relevant information before answering.")
# Ours: mandatory reasoning, for runs whose errors must be diagnosable from the response text.
COT_FORCED = " Think step by step and explain your reasoning before giving the answer."
COT_MODES = ("none", "qwen", "forced")
ANSWER_LINE_MC = "\nEnd your response with a final line of the form \"Answer: X\", where X is the option's letter."
ANSWER_LINE_OPEN = "\nEnd your response with a final line of the form \"Answer: <your answer>\"."

IMAGE_TOKEN = re.compile(r"<image (\d+)>")


def prompt_hash(style="mmmu", image_position="inline", answer_line=False, cot="none"):
    """Identity of the text format. Baseline value (mmmu/inline, no extras) is unchanged."""
    if style == "mmmu" and image_position == "inline" and not answer_line and cot == "none":
        payload = MC_FORMAT + "\x00" + OPEN_FORMAT
    else:
        payload = f"{style}\x00{image_position}\x00{int(answer_line)}\x00{cot}\x00" + build_text(
            {"question_type": "multiple-choice", "question": "Q", "options": ["x", "y"]}, style, answer_line, cot
        ) + "\x00" + build_text({"question_type": "open", "question": "Q", "options": []}, style, answer_line, cot)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def build_text(sample, style="mmmu", answer_line=False, cot="none"):
    mc = sample["question_type"] == "multiple-choice"
    if style == "mmmu":
        if mc:
            example = ""
            for i, option in enumerate(sample["options"]):
                example += f"({chr(ord('A') + i)}) {option}\n"
            text = MC_FORMAT.format(sample["question"], example)
        else:
            text = OPEN_FORMAT.format(sample["question"])
    elif style == "vlmevalkit":
        text = f"Question: {sample['question']}\n"
        if sample["options"]:
            text += "Options:\n"
            for i, option in enumerate(sample["options"]):
                text += f"{chr(ord('A') + i)}. {option}\n"
            text += "Please select the correct answer from the options above. \n"
        text = text.rstrip()
    else:
        raise ValueError(f"unknown prompt style {style!r}")
    if cot == "qwen":
        text += COT_PROMPT
    elif cot == "forced":
        text += COT_FORCED
    elif cot != "none":
        raise ValueError(f"unknown cot mode {cot!r}")
    if answer_line:
        text += ANSWER_LINE_MC if mc else ANSWER_LINE_OPEN
    return text


def trim_loop(text, min_unit=20, max_unit=1000, min_repeats=3, window=2000):
    """Cut a verbatim loop off the end of a response.

    Looks in the last `window` chars for the shortest tail unit (min_unit..max_unit chars) that
    occurs at least `min_repeats` times there; if found, keeps the text up to the end of that
    unit's first occurrence in the whole response. Returns (trimmed_text, removed_chars).
    """
    tail = text[-window:]
    for k in range(min_unit, min(max_unit, len(tail) // min_repeats) + 1):
        unit = tail[-k:]
        if tail.count(unit) >= min_repeats:
            cut = text.find(unit) + k
            return text[:cut], len(text) - cut
    return text, 0


FALLBACK_SUFFIX_MC = "\n\nTherefore, the final answer is ("
FALLBACK_SUFFIX_OPEN = "\n\nTherefore, the final answer is:"


def fallback_suffix(sample):
    return FALLBACK_SUFFIX_MC if sample["question_type"] == "multiple-choice" else FALLBACK_SUFFIX_OPEN


def _round(n, factor):
    return round(n / factor) * factor


def smart_resize(height, width, factor, min_pixels, max_pixels):
    """qwen_vl_utils.vision_process.smart_resize, verbatim logic."""
    h_bar = max(factor, _round(height, factor))
    w_bar = max(factor, _round(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif min_pixels is not None and h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return max(factor, h_bar), max(factor, w_bar)


def resize_image(img, max_pixels, min_pixels=None):
    """RGB; transparent pixels flattened onto white (as qwen-vl-utils does).

    min_pixels=None keeps the baseline behaviour: downscale (bicubic, aspect kept) only when
    width*height exceeds max_pixels, never upscale, no rounding. With min_pixels set, apply
    smart_resize so the image lands in [min_pixels, max_pixels] on multiples of IMAGE_FACTOR.
    """
    if img.mode in ("RGBA", "LA", "P") or "transparency" in img.info:
        rgba = img.convert("RGBA")
        img = Image.new("RGB", rgba.size, (255, 255, 255))
        img.paste(rgba, mask=rgba.split()[3])
    else:
        img = img.convert("RGB")
    w, h = img.size
    if min_pixels is None:
        if w * h > max_pixels:
            scale = math.sqrt(max_pixels / (w * h))
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)
        return img
    new_h, new_w = smart_resize(h, w, IMAGE_FACTOR, min_pixels, max_pixels)
    if (new_w, new_h) != (w, h):
        img = img.resize((new_w, new_h), Image.BICUBIC)
    return img


def build_content(text, images, max_pixels, min_pixels=None, image_position="inline"):
    """Split `text` into chat content items.

    inline: each image is inserted once, at its first `<image N>` reference. A repeated
    reference, or one to a missing column, stays as literal text. Images never referenced go first.
    first: all images go first in column order; the text is kept whole, tokens included.
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
        ordered.append(resize_image(images[n], max_pixels, min_pixels))
        content.append({"type": "image"})

    referenced = {int(m.group(1)) for m in IMAGE_TOKEN.finditer(text)}
    for n in sorted(referenced - set(images)):
        warnings.append(f"<image {n}> referenced but image_{n} is null")

    if image_position == "first":
        for n in sorted(images):
            add_image(n)
        add_text(text)
        return content, ordered, warnings
    if image_position != "inline":
        raise ValueError(f"unknown image position {image_position!r}")

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
            add_text(m.group(0))
        pos = m.end()
    add_text(text[pos:])
    return content, ordered, warnings


def render(tokenizer, content):
    """Exact string sent to the model: single user turn, model chat template, no system prompt."""
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
