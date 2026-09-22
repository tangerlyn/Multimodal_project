#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.2.6", "polars==1.32.3", "pydantic==2.11.7"]
# ///
# How to run:
# uv run score_mmmu.py outputs/raw.jsonl outputs/scored_official_no_random
# Place eval_utils_official.py from MMMU commit 268471d beside this script.
"""Score saved MMMU responses without generation or random fallback."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Final, Literal, assert_never

import polars as pl
from pydantic import BaseModel, ConfigDict, TypeAdapter

UPSTREAM_SHA: Final = "268471d0d488258990025331c7528359c324aa25"
UPSTREAM_HASH: Final = "cc1a89b4a27f697702d340bc6a7578b5237135754ddaa9ef105239a3d9847367"
ANSWERS: Final = TypeAdapter(list[str], config=ConfigDict(strict=True))


class RawRow(BaseModel):
    """A saved generation record; answers are used only after parsing."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")
    id: str
    subject: str
    question_type: Literal["multiple-choice", "open"]
    options: list[str]
    answer: str
    prompt: str
    response: str
    finish_reason: Literal["stop", "length"]
    num_images: int


def main() -> None:
    """Keep all 900 rows, score them and export auditable results."""
    if len(sys.argv) != 3:
        raise SystemExit("Usage: score_mmmu.py RAW_JSONL OUTPUT_DIRECTORY")
    raw_path, out = map(Path, sys.argv[1:])
    source_path = Path(__file__).with_name("eval_utils_official.py")
    source = source_path.read_bytes()
    if hashlib.sha256(source).hexdigest() != UPSTREAM_HASH:
        raise SystemExit("Official parser hash mismatch; use the pinned source.")
    replacement = "pred_index = random.choice(all_choices)"
    original = source.decode("utf-8")
    if original.count(replacement) != 1:
        raise SystemExit("Expected exactly one random fallback.")
    modified = original.replace(replacement, "pred_index = None")
    out.mkdir(parents=True, exist_ok=True)
    parser_path = out / "eval_utils_no_random.py"
    parser_path.write_text(modified, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("mmmu_no_random", parser_path)
    if spec is None or spec.loader is None:
        raise SystemExit("Cannot load the parser.")
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    rows = [RawRow.model_validate_json(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 900 or len({row.id for row in rows}) != 900:
        raise SystemExit("Expected 900 unique records; run check_raw.py first.")
    records = []
    for row in rows:
        match row.question_type:
            case "multiple-choice":
                choices = [chr(65 + i) for i in range(len(row.options))]
                prediction = official.parse_multi_choice_response(
                    row.response, choices, dict(zip(choices, row.options, strict=True)),
                )
                parse_failed = prediction is None
                correct = bool(official.eval_multi_choice(row.answer, prediction))
            case "open":
                gold = row.answer
                if row.answer.strip().startswith("["):
                    gold = ANSWERS.validate_python(ast.literal_eval(row.answer))
                prediction = sorted(official.parse_open_response(row.response), key=str)
                parse_failed = not row.response.strip() or not prediction
                correct = not parse_failed and bool(official.eval_open(gold, prediction))
            case unreachable:
                assert_never(unreachable)
        records.append({
            **row.model_dump(), "parsed_pred": prediction,
            "parse_failed": parse_failed, "correct": correct,
        })
    with (out / "scored.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    # CSV stores heterogeneous predictions as JSON text to retain their types.
    table = pl.DataFrame([
        {**record, "parsed_pred": json.dumps(record["parsed_pred"], ensure_ascii=False),
         "options": json.dumps(record["options"], ensure_ascii=False)}
        for record in records
    ])
    table.write_csv(out / "scored.csv", include_bom=True)
    table.filter(~pl.col("correct")).write_csv(out / "errors.csv", include_bom=True)
    aggregates = [
        pl.len().alias("n"), pl.col("correct").sum().alias("correct"),
        (pl.col("correct").mean() * 100).alias("accuracy_pct"),
        pl.col("parse_failed").sum().alias("parse_failed"),
        (pl.col("finish_reason") == "length").sum().alias("truncated"),
    ]
    subjects = table.group_by("subject").agg(aggregates).sort("subject")
    if subjects.height != 30 or subjects["n"].to_list() != [30] * 30:
        raise SystemExit("Expected 30 subjects with 30 records each.")
    subjects.write_csv(out / "by_subject.csv", include_bom=True)
    for group in ("question_type", "finish_reason", "num_images"):
        table.group_by(group).agg(aggregates).sort(group).write_csv(out / f"by_{group}.csv", include_bom=True)
    n_correct = sum(record["correct"] for record in records)
    accuracy = 100 * n_correct / len(rows)
    macro = sum(subjects["accuracy_pct"].to_list()) / 30
    if not math.isclose(accuracy, macro, abs_tol=1e-10):
        raise SystemExit("Macro/micro mismatch.")
    summary = {
        "parser": "MMMU official, random fallback replaced with None only",
        "upstream_commit": UPSTREAM_SHA,
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "upstream_sha256": UPSTREAM_HASH,
        "parser_sha256": hashlib.sha256(modified.encode()).hexdigest(),
        "total": len(rows), "correct": n_correct,
        "accuracy_pct": accuracy, "macro_accuracy_pct": macro,
        "parse_failed": sum(record["parse_failed"] for record in records),
        "truncated": sum(row.finish_reason == "length" for row in rows),
        "delta_vs_67_4_pp": accuracy - 67.4,
        "note": "Truncated responses use the same parser; no manual corrections. Open parsing success does not imply a meaningful answer was extracted.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    with pl.Config(tbl_rows=30):
        print(subjects)


if __name__ == "__main__":
    main()
