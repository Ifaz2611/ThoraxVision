"""Evaluate a saved model on one of the dataset splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as `python scripts/evaluate.py` from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from thoraxvision.dataset_utils import (
    compute_metrics,
    format_metrics,
    iter_class_images,
    resolve_dataset_splits,
)
from thoraxvision.paths import MODELS_DIR
from thoraxvision.predictor import load_predictor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate model.pkl (or a CNN checkpoint) on a dataset split."
    )
    parser.add_argument("--model", default=str(MODELS_DIR / "model.pkl"))
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset path. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--split", choices=["train", "valid", "test"], default="test"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for the JSON metrics report.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        splits = resolve_dataset_splits(args.dataset)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    split_dir = splits.get(args.split)
    if split_dir is None:
        available = [name for name, path in splits.items() if path is not None]
        raise SystemExit(
            f"Split '{args.split}' not found. Available: {available}"
        )

    predictor = load_predictor(args.model)
    y_true: list[int] = []
    y_pred: list[int] = []

    for image_path, label in iter_class_images(split_dir):
        if label not in predictor.classes:
            continue
        result = predictor.predict(image_path)
        y_true.append(predictor.classes.index(label))
        y_pred.append(result["index"])

    if not y_true:
        raise SystemExit(f"No images evaluated in {split_dir}")

    metrics = compute_metrics(y_true, y_pred, predictor.classes)
    print(format_metrics(metrics, title=f"{args.split} metrics ({split_dir})"))

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Saved metrics to {output}")


if __name__ == "__main__":
    main()
