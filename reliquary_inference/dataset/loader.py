from __future__ import annotations

import hashlib
import random
from typing import Any

from ..constants import (
    DEFAULT_DATASET_NAME,
    DEFAULT_DATASET_SPLIT,
    DEFAULT_FALLBACK_PROMPTS,
)


def load_dataset_cached(dataset_name: str = DEFAULT_DATASET_NAME, split: str = DEFAULT_DATASET_SPLIT):
    try:
        from datasets import load_dataset

        return load_dataset(dataset_name, split=split)
    except Exception:
        return [{"text": prompt} for prompt in DEFAULT_FALLBACK_PROMPTS]


def get_prompt_by_index(dataset: Any, index: int) -> str | None:
    if index < 0 or index >= len(dataset):
        return None
    try:
        row = dataset[index]
        text = row.get("text") if isinstance(row, dict) else None
        return text or None
    except Exception:
        return None


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def deterministic_indices(*, seed_material: str, dataset_size: int, count: int) -> list[int]:
    rng = random.Random(int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16))
    if dataset_size <= 0:
        return []
    count = min(count, dataset_size)
    return sorted(rng.sample(range(dataset_size), count))
