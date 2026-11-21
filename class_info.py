"""Canonical class metadata shared by training, evaluation, and the web app.

The Kaggle Chest CT-Scan Images dataset uses short class names in ``test/``
(``adenocarcinoma``, ``normal``, ...) but appends patient/stage information to
the folder names in ``train/`` and ``valid/`` (for example
``adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib``). Both forms map to the same
canonical class below so the model and the UI agree on class order.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

BASE_DIR = Path(__file__).resolve().parent
CLASSES_PATH = BASE_DIR / "model_classes.json"
MODEL_INFO_PATH = BASE_DIR / "model_info.json"

CANONICAL_CLASSES = [
    "adenocarcinoma",
    "large.cell.carcinoma",
    "normal",
    "squamous.cell.carcinoma",
]

CLASS_LABELS = {
    "adenocarcinoma": "Adenocarcinoma",
    "large.cell.carcinoma": "Large Cell Carcinoma",
    "normal": "Normal",
    "squamous.cell.carcinoma": "Squamous Cell Carcinoma",
}

CLASS_DESCRIPTIONS = {
    "adenocarcinoma": (
        "Adenocarcinoma: A type of non-small cell lung cancer that starts in "
        "glandular cells."
    ),
    "large.cell.carcinoma": (
        "Large Cell Carcinoma: A type of non-small cell lung cancer made up of "
        "large, abnormal cells."
    ),
    "normal": "Normal: No cancer class detected by this model.",
    "squamous.cell.carcinoma": (
        "Squamous Cell Carcinoma: A type of non-small cell lung cancer that "
        "starts in the flat cells lining the airways."
    ),
}


def canonical_class_name(folder_name: str) -> str:
    """Map a dataset folder name to its canonical class name.

    Unknown folder names are returned unchanged so custom datasets still work.
    """
    candidate = folder_name.strip().lower()
    for canonical in CANONICAL_CLASSES:
        if candidate == canonical or candidate.startswith(canonical + "_"):
            return canonical
    return folder_name


def display_label(class_name: str) -> str:
    """Return a short, human-friendly label for a class name."""
    canonical = canonical_class_name(class_name)
    if canonical in CLASS_LABELS:
        return CLASS_LABELS[canonical]
    return class_name.replace("_", " ").replace(".", " ").strip().title()


def describe_class(class_name: str) -> str:
    """Return a description, generating one for unknown/custom classes."""
    canonical = canonical_class_name(class_name)
    if canonical in CLASS_DESCRIPTIONS:
        return CLASS_DESCRIPTIONS[canonical]
    return f"{display_label(class_name)}: Class from the training dataset."


def _classes_from_payload(payload) -> list[str] | None:
    if isinstance(payload, list) and payload and all(
        isinstance(item, str) for item in payload
    ):
        return list(payload)
    if isinstance(payload, dict):
        classes = payload.get("classes")
        if isinstance(classes, list) and classes and all(
            isinstance(item, str) for item in classes
        ):
            return list(classes)
    return None


def load_class_names(
    classes_path: Path | str = CLASSES_PATH,
    info_path: Path | str = MODEL_INFO_PATH,
) -> list[str]:
    """Load the class order, preferring model_classes.json then model_info.json."""
    for path in (Path(classes_path), Path(info_path)):
        if not path.exists():
            continue
        try:
            names = _classes_from_payload(
                json.loads(path.read_text(encoding="utf-8"))
            )
        except (json.JSONDecodeError, OSError):
            names = None
        if names:
            return names
    return list(CANONICAL_CLASSES)


def class_metadata(class_names: Iterable[str]) -> dict:
    """Return the class name/label/description payload used by the API."""
    names = list(class_names)
    return {
        "class_names": names,
        "class_labels": {name: display_label(name) for name in names},
        "class_descriptions": {name: describe_class(name) for name in names},
    }
