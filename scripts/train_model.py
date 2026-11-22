import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn
from sklearn.ensemble import RandomForestClassifier

from dataset_utils import (
    compute_metrics,
    format_metrics,
    load_split,
    resolve_dataset_splits,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
CLASSES_PATH = BASE_DIR / "model_classes.json"
INFO_PATH = BASE_DIR / "model_info.json"
REPORTS_DIR = BASE_DIR / "reports"
IMG_SIZE = 128


def _display_path(path: Path | None) -> str | None:
    """Store dataset paths relative to the repo when possible."""
    if path is None:
        return None
    try:
        return path.relative_to(BASE_DIR).as_posix()
    except ValueError:
        return path.as_posix()


def parse_args():
    parser = argparse.ArgumentParser(description="Train the CT scan classifier.")
    parser.add_argument(
        "--dataset",
        default=os.environ.get("CTSCAN_DATASET_DIR"),
        help=(
            "Path to the dataset: a split folder (e.g. dataset/Data/train), the "
            "folder containing train/valid/test, a Kaggle dataset root "
            "containing Data/train, or the repository root. Defaults to the "
            "first dataset location found under the repo, or the "
            "CTSCAN_DATASET_DIR environment variable."
        ),
    )
    parser.add_argument("--image-size", type=int, default=IMG_SIZE)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        splits = resolve_dataset_splits(args.dataset)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    train_dir = splits["train"]
    print(f"Train split: {train_dir}")

    try:
        X_train, y_train, classes, skipped = load_split(
            train_dir, image_size=args.image_size
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print("Classes:", classes)
    print(f"X shape: {X_train.shape}, y shape: {y_train.shape} (skipped {skipped})")

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print(f"Train accuracy: {model.score(X_train, y_train):.4f}")

    metrics = {
        "train": compute_metrics(y_train, model.predict(X_train), classes),
    }

    for split_name in ("valid", "test"):
        split_dir = splits.get(split_name)
        if split_dir is None:
            print(f"{split_name}: not found, skipping evaluation")
            continue
        try:
            X_split, y_split, _, skipped = load_split(
                split_dir, image_size=args.image_size, classes=classes
            )
        except ValueError as exc:
            print(f"{split_name}: {exc}")
            continue
        metrics[split_name] = compute_metrics(
            y_split, model.predict(X_split), classes
        )
        print(
            format_metrics(
                metrics[split_name],
                title=f"{split_name} metrics ({split_dir}, skipped {skipped})",
            )
        )

    joblib.dump(model, MODEL_PATH, compress=3)
    CLASSES_PATH.write_text(json.dumps(classes, indent=2), encoding="utf-8")

    info = {
        "model_type": "random_forest",
        "model_file": MODEL_PATH.name,
        "sklearn_version": sklearn.__version__,
        "image_size": args.image_size,
        "n_estimators": args.n_estimators,
        "random_state": args.random_state,
        "classes": classes,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            name: _display_path(path) for name, path in splits.items()
        },
        "metrics": metrics,
    }
    INFO_PATH.write_text(json.dumps(info, indent=2), encoding="utf-8")

    REPORTS_DIR.mkdir(exist_ok=True)
    for split_name, split_metrics in metrics.items():
        (REPORTS_DIR / f"rf_{split_name}.json").write_text(
            json.dumps(split_metrics, indent=2), encoding="utf-8"
        )

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved class order to {CLASSES_PATH}")
    print(f"Saved model info to {INFO_PATH}")
    print(f"Saved metrics to {REPORTS_DIR}")


if __name__ == "__main__":
    main()
