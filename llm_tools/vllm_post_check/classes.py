"""Class-name helpers."""

from __future__ import annotations

from pathlib import Path


def load_classes(path: str | Path) -> list[str]:
    """Load one class name per line."""
    items: list[str] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            value = line.strip()
            if value and not value.startswith("#"):
                items.append(value)
    if not items:
        raise ValueError(f"No classes found in {path}")
    return items


def class_index(classes: list[str]) -> dict[str, int]:
    return {name.lower(): idx for idx, name in enumerate(classes)}


def resolve_class_id(name: str, classes: list[str]) -> int | None:
    lookup = class_index(classes)
    key = str(name or "").strip().lower()
    if key in lookup:
        return lookup[key]
    normalized = key.replace("_", " ").replace("-", " ")
    for class_name, idx in lookup.items():
        if class_name.replace("_", " ").replace("-", " ") == normalized:
            return idx
    return None
