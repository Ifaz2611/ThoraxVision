"""Dataset discovery, image loading, and evaluation helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from class_info import canonical_class_name

BASE_DIR = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SPLIT_NAMES = ("train", "valid", "test")
VALID_NAMES = ("valid", "val", "validation")


def _looks_like_split_dir(path: Path) -> bool:
    """True when *path* contains class folders with images inside."""
    if not path.is_dir():
        return False
    for class_dir in path.iterdir():
        if not class_dir.is_dir():
            continue
        try:
            if any(
                file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
                for file in class_dir.iterdir()
            ):
                return True
        except OSError:
            continue
    return False


def _find_split(root: Path, names: Iterable[str]) -> Path | None:
    for name in names:
        candidate = root / name
        if _looks_like_split_dir(candidate):
            return candidate
    return None


def resolve_dataset_splits(
    dataset: str | Path | None = None,
) -> dict[str, Path | None]:
    """Resolve the ``train``/``valid``/``test`` split directories.

    ``dataset`` may point at:

    * the train split itself (siblings ``valid``/``test`` are auto-detected),
    * a directory containing ``train``/``valid``/``test`` (for example
      ``.../Data``),
    * the Kaggle dataset root containing ``Data/train``,
    * a repository root containing ``dataset/Data/train`` (or the nested
      ``dataset/dataset/Data/train`` layout).

    When omitted, ``CTSCAN_DATASET_DIR`` and common local locations are tried.
    """
    if dataset:
        start = Path(dataset).expanduser().resolve()
        candidates = [start]
        if start.name.lower() in (*SPLIT_NAMES, *VALID_NAMES):
            candidates.insert(0, start.parent)

        # Kaggle layout: <root>/Data/train, plus nested variants such as
        # <root>/dataset/Data/train and <root>/dataset/dataset/Data/train.
        for base in list(candidates):
            candidates.extend(
                [
                    base / "Data",
                    base / "dataset" / "Data",
                    base / "dataset" / "dataset" / "Data",
                ]
            )

        for base in candidates:
            train = _find_split(base, ("train",))
            if train:
                return {
                    "train": train,
                    "valid": _find_split(base, VALID_NAMES),
                    "test": _find_split(base, ("test",)),
                }

        # The path itself may be an arbitrary split directory.
        if _looks_like_split_dir(start):
            return {
                "train": start,
                "valid": _find_split(start.parent, VALID_NAMES),
                "test": _find_split(start.parent, ("test",)),
            }

        raise FileNotFoundError(
            f"Could not find a train split under {start}. Expected a folder "
            "containing class subfolders with images."
        )

    env_path = os.environ.get("CTSCAN_DATASET_DIR")
    if env_path:
        return resolve_dataset_splits(env_path)

    for candidate in (
        BASE_DIR / "dataset" / "Data",
        BASE_DIR / "dataset" / "dataset" / "Data",
        BASE_DIR / "dataset",
        BASE_DIR / "data",
    ):
        try:
            return resolve_dataset_splits(candidate)
        except FileNotFoundError:
            continue

    raise FileNotFoundError(
        "Dataset not found. Download the Chest CT-Scan Images dataset and pass "
        "--dataset <path>, or set CTSCAN_DATASET_DIR."
    )


def iter_class_images(split_dir: Path | str) -> Iterable[tuple[Path, str]]:
    """Yield ``(image_path, canonical_class_name)`` for every image in a split."""
    split_dir = Path(split_dir)
    for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
        label = canonical_class_name(class_dir.name)
        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                yield image_path, label


def discover_classes(split_dir: Path | str) -> list[str]:
    classes = sorted({label for _, label in iter_class_images(split_dir)})
    if not classes:
        raise ValueError(f"No images found in {split_dir}")
    return classes


def load_split(
    split_dir: Path | str,
    image_size: int = 128,
    classes: Iterable[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    """Load a split into flattened, normalised grayscale arrays.

    Returns ``(X, y, classes, skipped)``. Images that cannot be decoded are
    skipped and counted instead of aborting the whole run.
    """
    classes = list(classes) if classes is not None else discover_classes(split_dir)
    class_to_index = {name: index for index, name in enumerate(classes)}
    features: list[np.ndarray] = []
    labels: list[int] = []
    skipped = 0

    for image_path, label in iter_class_images(split_dir):
        if label not in class_to_index:
            skipped += 1
            continue
        try:
            with Image.open(image_path) as image:
                array = (
                    np.asarray(
                        image.convert("L").resize((image_size, image_size)),
                        dtype=np.float32,
                    )
                    / 255.0
                )
        except (OSError, ValueError):
            skipped += 1
            continue
        features.append(array.reshape(-1))
        labels.append(class_to_index[label])

    if not features:
        raise ValueError(f"No usable images found in {split_dir}")

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        classes,
        skipped,
    )


def compute_metrics(
    y_true: Iterable[int], y_pred: Iterable[int], class_names: Iterable[str]
) -> dict:
    """Compute accuracy, macro/weighted metrics, per-class scores, and a matrix."""
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    class_names = list(class_names)
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))
    labels = list(range(len(class_names)))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    weighted_f1 = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )[2]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(class_names)
        },
        "confusion_matrix": matrix.tolist(),
        "labels": class_names,
        "n_samples": int(len(y_true)),
    }


def format_metrics(metrics: dict, title: str | None = None) -> str:
    lines = []
    if title:
        lines.append(title)
    lines.append(f"  accuracy: {metrics['accuracy']:.4f}")
    lines.append(
        f"  macro F1: {metrics['macro_f1']:.4f}  "
        f"weighted F1: {metrics['weighted_f1']:.4f}"
    )
    lines.append(f"  samples: {metrics['n_samples']}")
    lines.append("  per class:")
    for name, values in metrics["per_class"].items():
        lines.append(
            f"    {name:<28} precision={values['precision']:.3f} "
            f"recall={values['recall']:.3f} f1={values['f1']:.3f} "
            f"n={values['support']}"
        )
    return "\n".join(lines)
