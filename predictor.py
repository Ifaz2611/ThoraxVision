"""Model loading and prediction for the RandomForest and optional CNN models."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

from class_info import (
    CLASSES_PATH,
    MODEL_INFO_PATH,
    describe_class,
    display_label,
    load_class_names,
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "model.pkl"
DEFAULT_CNN_PATH = BASE_DIR / "model_cnn.pt"


class PredictorUnavailableError(RuntimeError):
    """Raised when the configured model cannot be loaded."""


def _load_model_info(path: Path | str = MODEL_INFO_PATH) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _result_payload(classes: list[str], index: int, probabilities) -> dict:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    order = np.argsort(probabilities)[::-1]
    return {
        "index": int(index),
        "class_name": classes[index],
        "label": display_label(classes[index]),
        "description": describe_class(classes[index]),
        "probabilities": [
            {
                "index": int(position),
                "class_name": classes[position],
                "label": display_label(classes[position]),
                "probability": float(probabilities[position]),
            }
            for position in order
        ],
    }


class RandomForestPredictor:
    """Predictor backed by the scikit-learn RandomForest in ``model.pkl``."""

    model_type = "random_forest"

    def __init__(
        self,
        model_path: Path | str = DEFAULT_MODEL_PATH,
        classes_path: Path | str = CLASSES_PATH,
        info_path: Path | str = MODEL_INFO_PATH,
    ):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise PredictorUnavailableError(f"Model file not found: {self.model_path}")

        self.model = joblib.load(self.model_path)
        # The web app predicts one image at a time; avoid joblib worker overhead
        # (and the related scikit-learn parallel warning) on every request.
        if hasattr(self.model, "n_jobs"):
            self.model.n_jobs = 1
        self.info = _load_model_info(info_path)
        self.classes = self._resolve_classes(classes_path, info_path)
        self.image_size = self._resolve_image_size()
        self.sklearn_version = self.info.get("sklearn_version")

    def _resolve_classes(self, classes_path, info_path) -> list[str]:
        classes = load_class_names(classes_path, info_path)
        n_classes = int(getattr(self.model, "n_classes_", len(classes)))
        if len(classes) != n_classes:
            raise PredictorUnavailableError(
                f"Class list has {len(classes)} entries but the model has "
                f"{n_classes} classes. Retrain with python model.py."
            )
        return classes

    def _resolve_image_size(self) -> int:
        # The model's feature count is authoritative; model_info.json is only a
        # fallback for models whose input is not a square flattened image.
        n_features = int(getattr(self.model, "n_features_in_", 0))
        root = int(round(math.sqrt(n_features))) if n_features > 0 else 0
        if root > 0 and root * root == n_features:
            return root
        size = self.info.get("image_size")
        if isinstance(size, int) and size > 0:
            return size
        return 128

    def preprocess(self, image_path: Path | str) -> np.ndarray:
        with Image.open(image_path) as image:
            array = (
                np.asarray(
                    image.convert("L").resize((self.image_size, self.image_size)),
                    dtype=np.float32,
                )
                / 255.0
            )
        return array.reshape(1, -1)

    def predict(self, image_path: Path | str) -> dict:
        features = self.preprocess(image_path)
        raw_prediction = self.model.predict(features)[0]
        if isinstance(raw_prediction, (int, np.integer)):
            index = int(raw_prediction)
        else:
            index = self.classes.index(str(raw_prediction))

        raw_probabilities = self.model.predict_proba(features)[0]
        probabilities = np.zeros(len(self.classes), dtype=np.float64)
        for column, model_class in enumerate(getattr(self.model, "classes_", [])):
            if isinstance(model_class, (int, np.integer)):
                position = int(model_class)
            else:
                position = self.classes.index(str(model_class))
            if 0 <= position < len(probabilities):
                probabilities[position] = raw_probabilities[column]

        return _result_payload(self.classes, index, probabilities)


class TorchPredictor:
    """Predictor backed by an optional transfer-learning CNN checkpoint."""

    model_type = "cnn"

    def __init__(self, model_path: Path | str = DEFAULT_CNN_PATH):
        try:
            import torch
            import torchvision  # noqa: F401
        except ImportError as exc:
            raise PredictorUnavailableError(
                "CNN models require torch and torchvision. Install them with: "
                "pip install -r requirements-cnn.txt"
            ) from exc

        self.torch = torch
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise PredictorUnavailableError(
                f"CNN model file not found: {self.model_path}"
            )

        checkpoint = torch.load(
            self.model_path, map_location="cpu", weights_only=False
        )
        self.classes = list(checkpoint["classes"])
        self.image_size = int(checkpoint.get("image_size", 224))
        self.arch = checkpoint.get("arch", "mobilenet_v2")
        normalization = checkpoint.get("normalization", {})
        self.mean = normalization.get("mean", [0.485, 0.456, 0.406])
        self.std = normalization.get("std", [0.229, 0.224, 0.225])
        self.model = self._build_model(len(self.classes))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.model_type = f"cnn_{self.arch}"

        from torchvision import transforms

        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

    def _build_model(self, num_classes: int):
        from torchvision import models

        if self.arch == "mobilenet_v2":
            model = models.mobilenet_v2(weights=None)
            model.classifier[1] = self.torch.nn.Linear(
                model.last_channel, num_classes
            )
        elif self.arch == "resnet18":
            model = models.resnet18(weights=None)
            model.fc = self.torch.nn.Linear(model.fc.in_features, num_classes)
        else:
            raise PredictorUnavailableError(
                f"Unsupported CNN architecture: {self.arch}"
            )
        return model

    def predict(self, image_path: Path | str) -> dict:
        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB")).unsqueeze(0)

        with self.torch.no_grad():
            logits = self.model(tensor)
            probabilities = (
                self.torch.softmax(logits, dim=1)[0].cpu().numpy()
            )
        index = int(np.argmax(probabilities))
        return _result_payload(self.classes, index, probabilities)


def load_predictor(model_path: Path | str | None = None):
    """Load the configured predictor.

    Selection order:

    1. An explicit *model_path* (``.pt``/``.pth`` selects the CNN, anything
       else selects the RandomForest).
    2. ``CTSCAN_MODEL_PATH`` if set.
    3. ``CTSCAN_USE_CNN=1`` and ``model_cnn.pt`` exists.
    4. ``model.pkl``.
    """
    if model_path is not None:
        path = Path(model_path).expanduser()
        if path.suffix.lower() in {".pt", ".pth"}:
            return TorchPredictor(path)
        return RandomForestPredictor(path)

    env_path = os.environ.get("CTSCAN_MODEL_PATH")
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if path.suffix.lower() in {".pt", ".pth"}:
            return TorchPredictor(path)
        return RandomForestPredictor(path)

    use_cnn = os.environ.get("CTSCAN_USE_CNN", "").lower() in {"1", "true", "yes"}
    if use_cnn and DEFAULT_CNN_PATH.exists():
        return TorchPredictor(DEFAULT_CNN_PATH)

    return RandomForestPredictor(DEFAULT_MODEL_PATH)
