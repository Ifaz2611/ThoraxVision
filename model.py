import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "dataset" / "Data" / "train"
MODEL_PATH = BASE_DIR / "model.pkl"
CLASSES_PATH = BASE_DIR / "model_classes.json"
IMG_SIZE = 128
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def resolve_data_dir(arg_path):
    """Accept a path to the train dir itself or to the dataset root."""
    p = Path(arg_path).expanduser().resolve() if arg_path else DEFAULT_DATA_DIR
    # Accept either the train folder itself or a dataset root containing it
    for candidate in (p / "Data" / "train", p / "train"):
        if candidate.is_dir():
            return candidate
    return p


def main():
    parser = argparse.ArgumentParser(description="Train the CT scan classifier.")
    parser.add_argument(
        "--dataset",
        default=os.environ.get("CTSCAN_DATASET_DIR"),
        help=(
            "Path to the training data: either the folder containing the "
            "class subfolders (e.g. dataset/Data/train) or a dataset root "
            "containing Data/train. Defaults to ./dataset/Data/train, or the "
            "CTSCAN_DATASET_DIR environment variable."
        ),
    )
    args = parser.parse_args()

    DATA_DIR = resolve_data_dir(args.dataset)

    if not DATA_DIR.is_dir():
        raise SystemExit(
            f"Dataset not found at {DATA_DIR}.\n"
            "Download the Chest CT-Scan Images dataset from Kaggle and unzip it "
            "there, or pass --dataset <path-to-train-folder>."
        )

    # sorted() keeps the class order deterministic so the app and model agree
    classes = sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir())
    if not classes:
        raise SystemExit(f"No class folders found in {DATA_DIR}.")

    print("Classes:", classes)

    X = []
    y = []
    for label_idx, label in enumerate(classes):
        class_dir = DATA_DIR / label
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            img = Image.open(img_path).convert("L").resize((IMG_SIZE, IMG_SIZE))
            arr = np.array(img).flatten() / 255.0
            X.append(arr)
            y.append(label_idx)

    X = np.array(X)
    y = np.array(y)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    CLASSES_PATH.write_text(json.dumps(classes, indent=2), encoding="utf-8")

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved class order to {CLASSES_PATH}")
    print("Train accuracy:", model.score(X_train, y_train))
    print("Test accuracy:", model.score(X_test, y_test))


if __name__ == "__main__":
    main()
