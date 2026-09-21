#!/usr/bin/env python3
"""Validate a raw.jsonl against docs/raw_schema.md. Not a parser or scorer.

    python scripts/check_raw.py outputs/raw.jsonl            # full run: 900 ids, 30 per subject
    python scripts/check_raw.py outputs/smoke.jsonl --partial  # smoke/ablation: schema only

Standard library only. Exits 1 on any violation.
"""
import argparse
import collections
import json
import sys

SUBJECTS = [
    "Accounting", "Agriculture", "Architecture_and_Engineering", "Art", "Art_Theory",
    "Basic_Medical_Science", "Biology", "Chemistry", "Clinical_Medicine", "Computer_Science",
    "Design", "Diagnostics_and_Laboratory_Medicine", "Economics", "Electronics",
    "Energy_and_Power", "Finance", "Geography", "History", "Literature", "Manage",
    "Marketing", "Materials", "Math", "Mechanical_Engineering", "Music", "Pharmacy",
    "Physics", "Psychology", "Public_Health", "Sociology",
]
FIELDS = {
    "id": str, "subject": str, "question_type": str, "options": list, "answer": str,
    "prompt": str, "response": str, "finish_reason": str, "num_images": int,
}
QUESTION_TYPES = {"multiple-choice", "open"}
FINISH_REASONS = {"stop", "length"}
PER_SUBJECT = 30
MAX_ERRORS_SHOWN = 20


def check_record(rec, where):
    errors = []
    for field, typ in FIELDS.items():
        if field not in rec:
            errors.append(f"{where}: missing field '{field}'")
        elif not isinstance(rec[field], typ) or isinstance(rec[field], bool):
            errors.append(f"{where}: '{field}' should be {typ.__name__}, got {type(rec[field]).__name__}")
    extra = set(rec) - set(FIELDS)
    if extra:
        errors.append(f"{where}: unexpected fields {sorted(extra)}")
    if errors:
        return errors

    if rec["subject"] not in SUBJECTS:
        errors.append(f"{where}: unknown subject '{rec['subject']}'")
    elif not rec["id"].startswith(f"validation_{rec['subject']}_"):
        errors.append(f"{where}: id '{rec['id']}' does not match subject '{rec['subject']}'")
    if rec["question_type"] not in QUESTION_TYPES:
        errors.append(f"{where}: question_type '{rec['question_type']}'")
    if not all(isinstance(o, str) for o in rec["options"]):
        errors.append(f"{where}: options must be a list of strings")
    if rec["question_type"] == "multiple-choice" and len(rec["options"]) < 2:
        errors.append(f"{where}: multiple-choice with {len(rec['options'])} options")
    if rec["question_type"] == "open" and rec["options"]:
        errors.append(f"{where}: open question with non-empty options")
    if rec["finish_reason"] not in FINISH_REASONS:
        errors.append(f"{where}: finish_reason '{rec['finish_reason']}'")
    if not 0 <= rec["num_images"] <= 7:
        errors.append(f"{where}: num_images {rec['num_images']}")
    if not rec["prompt"].strip():
        errors.append(f"{where}: empty prompt")
    if rec["answer"] and rec["question_type"] == "multiple-choice":
        idx = ord(rec["answer"][0]) - ord("A")
        if len(rec["answer"]) != 1 or not 0 <= idx < max(len(rec["options"]), 1):
            errors.append(f"{where}: gold answer '{rec['answer']}' outside the option range")
    return errors


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--partial", action="store_true", help="skip the 900 / 30-per-subject checks")
    args = ap.parse_args()

    errors, records, seen = [], [], set()
    with open(args.path) as f:
        for lineno, line in enumerate(f, 1):
            where = f"line {lineno}"
            if not line.strip():
                errors.append(f"{where}: blank line")
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"{where}: invalid JSON ({e})")
                continue
            if not isinstance(rec, dict):
                errors.append(f"{where}: not a JSON object")
                continue
            rec_errors = check_record(rec, where)
            errors += rec_errors
            if rec_errors:
                continue
            if rec["id"] in seen:
                errors.append(f"{where}: duplicate id '{rec['id']}'")
            seen.add(rec["id"])
            records.append(rec)

    per_subject = collections.Counter(r["subject"] for r in records)
    if not args.partial:
        if len(seen) != PER_SUBJECT * len(SUBJECTS):
            errors.append(f"expected {PER_SUBJECT * len(SUBJECTS)} unique ids, found {len(seen)}")
        for s in SUBJECTS:
            if per_subject[s] != PER_SUBJECT:
                errors.append(f"subject {s}: {per_subject[s]} records, expected {PER_SUBJECT}")

    n = len(records)
    length = sum(r["finish_reason"] == "length" for r in records)
    empty = sum(not r["response"].strip() for r in records)
    types = collections.Counter(r["question_type"] for r in records)
    images = collections.Counter(r["num_images"] for r in records)
    print(f"file            {args.path}")
    print(f"valid records   {n}  ({len(per_subject)} subjects)")
    print(f"question types  {dict(types)}")
    print(f"num_images      {dict(sorted(images.items()))}")
    print(f"finish=length   {length}  ({100 * length / n:.2f}%)" if n else "finish=length   n/a")
    print(f"empty responses {empty}")

    if errors:
        print(f"\nFAIL: {len(errors)} problem(s)")
        for e in errors[:MAX_ERRORS_SHOWN]:
            print(f"  {e}")
        if len(errors) > MAX_ERRORS_SHOWN:
            print(f"  ... {len(errors) - MAX_ERRORS_SHOWN} more")
        sys.exit(1)
    print("\nOK" + (" (partial: counts not enforced)" if args.partial else ""))


if __name__ == "__main__":
    main()
