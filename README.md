<div align="center">

# ThoraxVision

**Chest CT-scan image classification in the browser — Flask + scikit-learn, DICOM-aware, with class probabilities and image statistics. Built as a learning project.**

[![Tests](https://img.shields.io/badge/tests-20%20passing-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

</div>

> [!IMPORTANT]
> ThoraxVision is **not** a medical device and must **not** be used for diagnosis. Always consult a qualified radiologist.

---

## What is this?

ThoraxVision is a web app for **chest CT-scan image classification**, built as a learning/educational project. A Flask app where you upload a chest CT scan (PNG, JPEG, BMP, TIFF, or DICOM `.dcm`), and it predicts which class the scan belongs to along with per-class probabilities and basic image statistics.

It performs a **4-way classification**:

- `adenocarcinoma`
- `large.cell.carcinoma`
- `squamous.cell.carcinoma`
- `normal` (no cancer)

**Use case:** Educational/demo purposes — a doctor-style "upload a scan, get a prediction with confidence bars" experience. It is explicitly **not** a medical tool. The README is candid about performance: the RandomForest baseline gets ~99.8% train accuracy but only ~47.6% test accuracy, and even the transfer-learning CNN only reaches ~61.6% test accuracy with poor recall on `large.cell.carcinoma`.

## Tech stack

- **Flask** backend (`thoraxvision/app.py`) with a browser UI (HTML/CSS/JS templates)
- **RandomForest** classifier as the default model (`models/model.pkl`, ~250 KB, committed to the repo) — classifies on flattened 128×128 grayscale pixels
- **Optional MobileNetV2 / ResNet18 CNN** via transfer learning (`torch` / `torchvision`, not in default requirements)
- **pydicom + OpenCV** for DICOM detection, windowing, and PNG conversion
- **waitress / gunicorn** production entrypoints, plus a **Dockerfile** and **pytest** suite (20 tests)
- **Training/evaluation scripts** in `scripts/` (`train_model.py`, `evaluate.py`, `train_cnn.py`)

## Contents

- [Features](#features)
- [Quick start](#quick-start)
- [Project structure](#project-structure)
- [Dataset](#dataset)
- [Production deployment](#production-deployment)
  - [Docker](#docker)
- [Training and evaluation](#training-and-evaluation)
  - [RandomForest baseline](#randomforest-baseline)
  - [Evaluate a saved model](#evaluate-a-saved-model)
  - [Transfer-learning CNN (optional)](#transfer-learning-cnn-optional)
- [Results](#results)
- [API](#api)
- [Tests](#tests)
- [Model distribution](#model-distribution)
- [Maintainer notes](#maintainer-notes)
- [License](#license)

## Features

- **Image upload** — PNG, JPEG, BMP, TIFF (max 25 MB)
- **DICOM support** — `.dcm` / `.dicom` files are auto-detected, windowed, and converted to PNG before analysis
- **ML classification** — RandomForest by default; optional MobileNetV2 / ResNet18 CNN via transfer learning
- **Class probabilities** — full `predict_proba` breakdown rendered as bars in the UI
- **Human-readable labels** — friendly class names, labels, and descriptions instead of raw folder names
- **Image statistics** — mean intensity, standard deviation, edge count
- **Health endpoint** — `/health` reports model status for deployment checks and Docker healthchecks
- **Production servers** — gunicorn and waitress entrypoints included
- **Test suite** — 20 pytest tests covering the API, class mapping, and dataset utilities

## Quick start

Requires **Python 3.11+**.

```bash
git clone https://github.com/Ifaz2611/ThoraxVision
cd ThoraxVision

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
python -m thoraxvision.app
```

Open <http://127.0.0.1:5000> and upload a scan. The app works out of the box — `models/model.pkl` and `models/model_info.json` are committed, so no dataset download is needed just to try it.

> [!TIP]
> New here? [GET-STARTED.md](GET-STARTED.md) has a guided walkthrough.

## Project structure

<details>
<summary>Expand the full tree</summary>

```text
.
├── thoraxvision/           # Application package
│   ├── app.py              # Flask app: routes, upload handling, analysis
│   ├── serve.py            # Cross-platform waitress production entrypoint
│   ├── predictor.py        # Model loading (RandomForest / CNN) + prediction
│   ├── class_info.py       # Canonical class names, labels, descriptions
│   ├── dataset_utils.py    # Dataset discovery, loading, metrics
│   ├── dicom_utils.py      # DICOM detection, windowing, PNG conversion
│   ├── paths.py            # Canonical repo paths (models/, dataset/, ...)
│   ├── static/             # style.css, script.js
│   └── templates/          # index.html
├── scripts/                # Training & evaluation CLIs
│   ├── train_model.py      # Trains models/model.pkl from the dataset
│   ├── evaluate.py         # Evaluates a saved model on a dataset split
│   └── train_cnn.py        # Optional transfer-learning experiment
├── models/                 # Model artifacts
│   ├── model.pkl           # Trained RandomForest model (~250 KB, committed)
│   ├── model_info.json     # Model metadata + baseline metrics (committed)
│   ├── model_classes.json  # Class order written by train_model.py (git-ignored)
│   ├── model_cnn.pt        # Optional CNN checkpoint (git-ignored)
│   └── legacy/             # Old model experiments (git-ignored)
├── dataset/                # Training data (git-ignored)
│   └── Data/{train,valid,test}
├── reports/                # Generated evaluation metrics
├── tests/                  # pytest suite
├── requirements.txt        # Runtime dependencies
├── requirements-dev.txt    # Runtime + pytest
├── requirements-cnn.txt    # Runtime + torch/torchvision
├── Dockerfile
├── .dockerignore
└── uploads/                # Created at runtime, git-ignored
```

</details>

## Dataset

The training images (~120 MB) are **not** part of this repository. Download the [Chest CT-Scan Images dataset](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images) from Kaggle and unzip it to:

```text
dataset/
└── Data/
    ├── train/    # long folder names incl. patient/stage info
    ├── valid/
    └── test/     # short folder names (adenocarcinoma, normal, ...)
```

The code maps both naming schemes to four canonical classes — `adenocarcinoma`, `large.cell.carcinoma`, `normal`, `squamous.cell.carcinoma` — so class order stays stable across splits. It also accepts the nested `dataset/dataset/Data/...` layout and auto-detects `valid/` / `test/` siblings.

Keep the dataset outside the repo (recommended) and point scripts at it with `--dataset` or `CTSCAN_DATASET_DIR`.

## Production deployment

> [!WARNING]
> Flask's built-in dev server is not meant for production. Use one of the following instead.

```bash
# waitress (Windows / macOS / Linux)
python -m thoraxvision.serve    # -> http://0.0.0.0:8000

# gunicorn (Linux / macOS)
gunicorn --bind 0.0.0.0:8000 --workers 2 --timeout 120 thoraxvision.app:app
```

`thoraxvision/serve.py` respects `HOST`, `PORT` (default `8000`), and `WEB_THREADS` (default `4`).

### Docker

```bash
docker build -t ctscan-analyzer .
docker run --rm -p 8000:8000 ctscan-analyzer
```

The image runs as a non-root user, serves via gunicorn, and healthchecks `/health` (building requires a running Docker daemon):

```bash
curl http://127.0.0.1:8000/health
```

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_type": "random_forest",
  "classes": ["adenocarcinoma", "large.cell.carcinoma", "normal", "squamous.cell.carcinoma"],
  "sklearn_version": "1.8.0"
}
```

Returns `503` when the model can't be loaded.

## Training and evaluation

### RandomForest baseline

```bash
python scripts/train_model.py                                  # auto-detects ./dataset
python scripts/train_model.py --dataset "C:/path/to/dataset"   # or set CTSCAN_DATASET_DIR
```

`--dataset` accepts the class-subfolder dir (`.../Data/train`), the dir containing `train/valid/test`, a Kaggle root containing `Data/train`, or the repo root. Training runs on `train/`, evaluates on `valid/` and `test/` when present, and writes:

- `models/model.pkl` — compressed RandomForest classifier
- `models/model_classes.json` — canonical class order
- `models/model_info.json` — sklearn version, image size, dataset paths, metrics
- `reports/rf_{train,valid,test}.json` — per-split metrics

### Evaluate a saved model

```bash
python scripts/evaluate.py --split test --output reports/rf_test.json
python scripts/evaluate.py --model models/model_cnn.pt --split test   # works for RF or CNN
```

### Transfer-learning CNN (optional)

```bash
pip install -r requirements-cnn.txt
python scripts/train_cnn.py --dataset dataset/Data --epochs 8 --freeze-epochs 3 --image-size 160
```

Fine-tunes a pretrained MobileNetV2 (or ResNet18): freezes the backbone for `--freeze-epochs`, keeps the best validation macro-F1 checkpoint, and writes `models/model_cnn.pt` (git-ignored) plus `reports/cnn_metrics.json`.

Serve the CNN instead of the RandomForest:

```bash
CTSCAN_USE_CNN=1 python -m thoraxvision.app                        # opt in
CTSCAN_MODEL_PATH=models/model_cnn.pt python -m thoraxvision.app   # or point at a checkpoint
```

Windows PowerShell: `$env:CTSCAN_USE_CNN="1"; python -m thoraxvision.app`

CNN dependencies are not in the default Docker image (torch adds several hundred MB) — build a custom image if you need them.

## Results

RandomForest baseline: flattened 128×128 grayscale pixels (16,384 features), 100 trees, `random_state=42`.

| Model | Split | Accuracy | Macro F1 |
| --- | --- | ---: | ---: |
| RandomForest | train | 99.84% | 0.9981 |
| RandomForest | valid | 72.22% | 0.7409 |
| RandomForest | test | 47.62% | 0.4862 |
| MobileNetV2 (160 px) | valid | 65.28% | 0.6624 |
| MobileNetV2 (160 px) | test | **61.59%** | **0.5803** |

Transfer learning lifts test accuracy by ~14 points (47.6% → 61.6%), but `large.cell.carcinoma` recall remains poor (0.10–0.17) — one more reason this is a demo, not a diagnostic tool. Full per-class metrics: [reports/](reports/).

## API

`POST /upload` accepts a multipart form field `image` (PNG / JPEG / BMP / TIFF / DICOM). Files are validated by extension, stored under a random UUID during processing, and always deleted afterwards.

<details>
<summary>Example response</summary>

```json
{
  "model_prediction": "adenocarcinoma",
  "model_prediction_index": 0,
  "predicted_label": "Adenocarcinoma",
  "prediction_description": "Adenocarcinoma: A type of non-small cell lung cancer that starts in glandular cells.",
  "probabilities": [
    { "index": 0, "class_name": "adenocarcinoma", "label": "Adenocarcinoma", "probability": 0.42 }
  ],
  "model_type": "random_forest",
  "input_type": "image",
  "class_names": ["adenocarcinoma", "large.cell.carcinoma", "normal", "squamous.cell.carcinoma"],
  "class_labels": { "...": "..." },
  "class_descriptions": { "...": "..." },
  "analysis": {
    "mean_intensity": 0,
    "std_intensity": 0,
    "edge_count": 0,
    "findings": [],
    "disclaimer": "FOR EDUCATIONAL PURPOSES ONLY - NOT FOR MEDICAL USE"
  }
}
```

</details>

- `model_prediction` is the **class name**; `model_prediction_index` is its index into `class_names`.
- `probabilities` is sorted descending and sums to 1.
- `input_type` is `"image"` or `"dicom"`.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest
```

The 20 tests cover the index page, `/health`, `/upload` (PNG + DICOM), invalid and unsupported uploads, probability/index consistency, upload cleanup, class-name mapping, dataset discovery, and metrics.

## Model distribution

<details>
<summary>What's committed, what's not, and why</summary>

- `models/model.pkl` (~250 KB compressed) stays **in the repo** — it's required at startup and avoids a release-artifact dependency.
- `models/model_info.json` is committed alongside it as the class-order/metadata fallback when `model_classes.json` is absent.
- `models/model_cnn.pt` (~9 MB) is **not** committed; it's reproducible via `scripts/train_cnn.py`.
- If the RF model ever grows past a few MB, move it to a release artifact or Git LFS.

</details>

## Maintainer notes

<details>
<summary>Operational notes for maintainers</summary>

- `models/model.pkl` was trained with scikit-learn 1.8.0; `requirements.txt` pins `scikit-learn>=1.8,<2.0`. Retrain with `python scripts/train_model.py` after a major sklearn upgrade.
- The model loads once at import. If loading fails, the app still starts, `/health` returns `503`, and `/upload` returns a clear error.
- Uploaded files and DICOM conversions are deleted in `finally` blocks.
- If the dataset ever landed in git history, purge it with `git filter-repo --path dataset --path test_images --invert-paths`.

</details>

## License

Released under the Apache-2.0 license — see [LICENSE](LICENSE) for the full text.
