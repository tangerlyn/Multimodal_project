"""MMMU validation loader: 30 subject configs at a pinned dataset revision."""
import ast

from datasets import load_dataset

DATASET_REPO = "MMMU/MMMU"
DATASET_REVISION = "98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68"
SPLIT = "validation"
MAX_IMAGES = 7

SUBJECTS = [
    "Accounting", "Agriculture", "Architecture_and_Engineering", "Art", "Art_Theory",
    "Basic_Medical_Science", "Biology", "Chemistry", "Clinical_Medicine", "Computer_Science",
    "Design", "Diagnostics_and_Laboratory_Medicine", "Economics", "Electronics",
    "Energy_and_Power", "Finance", "Geography", "History", "Literature", "Manage",
    "Marketing", "Materials", "Math", "Mechanical_Engineering", "Music", "Pharmacy",
    "Physics", "Psychology", "Public_Health", "Sociology",
]


def load_subject(subject, data_root=None, revision=DATASET_REVISION):
    """Return the validation items of one subject, in dataset order.

    `images` maps the N of `image_N` to a PIL image for every non-null column.
    """
    ds = load_dataset(DATASET_REPO, subject, split=SPLIT, revision=revision, cache_dir=data_root)
    return rows_to_samples(ds, subject)


def rows_to_samples(ds, subject):
    samples = []
    for row in ds:
        # `options` is stored as a Python-literal string, "[]" for open questions.
        options = ast.literal_eval(row["options"])
        images = {n: row[f"image_{n}"] for n in range(1, MAX_IMAGES + 1) if row[f"image_{n}"] is not None}
        samples.append({
            "id": row["id"],
            "subject": subject,
            "question_type": row["question_type"],
            "question": row["question"],
            "options": [str(o) for o in options],
            "answer": row["answer"],
            "images": images,
        })
    return samples
