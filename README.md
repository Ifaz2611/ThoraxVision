# ML-Based CT Scan Image Analyzer

A small Flask web app that runs basic image statistics and a machine-learning classifier on uploaded CT scan images. It was built as a learning/demo project, not as a medical tool.

> **Disclaimer:** This is **not** a medical device and must **not** be used for diagnosis. Always consult a qualified radiologist.

## What it does

- Upload a CT scan image (PNG, JPEG, BMP, TIFF) or a DICOM file (`.dcm` / `.dicom`)
- Shows basic image statistics: mean intensity, standard deviation, and edge count
- Predicts a class label using the trained RandomForest model
- Shows the predicted class name and the full `predict_proba` class-probability breakdown
- Converts DICOM pixel data to an 8-bit PNG before analysis
- Exposes a `/health` endpoint for deployment checks
- Always displays an educational-use disclaimer

## Project structure

```
.
├── app.py                 # Flask app: routes, upload handling, analysis
├── predictor.py           # RandomForest / optional CNN loading + prediction
├── class_info.py          # Canonical class names, labels, descriptions
├── dataset_utils.py       # Dataset discovery, loading, metrics
├── dicom_utils.py         # DICOM detection and PNG conversion
├── model.py               # Trains model.pkl from the dataset
├── evaluate.py            # Evaluates a saved model on a dataset split
├── train_cnn.py           # Optional transfer-learning experiment
├── serve.py               # Cross-platform waitress production entrypoint
├── model.pkl              # Trained RandomForest model (included)
├── model_info.json        # Model metadata and baseline metrics (included)
├── model_classes.json     # Class order written by model.py (git-ignored)
├── model_cnn.pt           # Optional CNN checkpoint (git-ignored)
├── requirements.txt       # Runtime dependencies
├── requirements-dev.txt   # Runtime + pytest
├── requirements-cnn.txt   # Runtime + torch/torchvision
├── reports/               # Generated evaluation metrics
├── tests/                 # pytest suite
├── static/
│   ├── script.js
│   └── style.css
├── templates/
│   └── index.html
├── Dockerfile
├── .dockerignore
└── uploads/               # Created at runtime, git-ignored
```

## Dataset (not included)

The training images (~120 MB) are **not** part of this repository — they are large and must never be committed. Download the Chest CT-Scan Images dataset from Kaggle:

- <https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images>

Unzip it so the folder layout is:

```
dataset/
└── Data/
    ├── train/
    │   ├── adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib/
    │   ├── large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa/
    │   ├── normal/
    │   └── squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa/
    ├── valid/
    └── test/
```

`train/` and `valid/` use long folder names that include patient/stage information; `test/` uses the short names (`adenocarcinoma`, `normal`, ...). The training/evaluation code maps both forms to canonical class names (`adenocarcinoma`, `large.cell.carcinoma`, `normal`, `squamous.cell.carcinoma`), so class order is stable across splits. A folder whose name starts with a canonical class name followed by `_` is treated as that class (this is how the Kaggle patient/stage suffixes are handled).

The repository also accepts the nested `dataset/dataset/Data/...` layout and auto-detects `valid/` and `test/` siblings.

If you keep the dataset outside the repo (recommended), point the scripts at it with `--dataset` or the `CTSCAN_DATASET_DIR` environment variable — see below.

## Setup

Python **3.11+** is required (scikit-learn 1.8 requires it).

```bash
git clone <your-repo-url>
cd CT-Scan-Image-Analyzer_Using_ML

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

For tests:

```bash
pip install -r requirements-dev.txt
```

## Run

Development server:

```bash
python app.py
```

Then open <http://127.0.0.1:5000> and upload an image. Debug mode is off by default; set `FLASK_DEBUG=1` to enable it.

`app.py` loads `model.pkl` at startup and needs no dataset — it only requires the model files, which are committed to the repo.

### Production server

Flask's development server is not for production. Use one of:

```bash
# Windows / macOS / Linux (waitress)
python serve.py
# -> http://0.0.0.0:8000
```

```bash
# Linux / macOS (gunicorn)
gunicorn --bind 0.0.0.0:8000 --workers 2 --timeout 120 app:app
```

`serve.py` respects `HOST`, `PORT` (default `8000`), and `WEB_THREADS` (default `4`). For gunicorn, use `--bind` and `--workers` as shown above.

### Docker

```bash
docker build -t ctscan-analyzer .
docker run --rm -p 8000:8000 ctscan-analyzer
```

The image runs as a non-root user, uses gunicorn, and includes a `HEALTHCHECK` against `/health`.

### Health check

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

Returns `503` when the model cannot be loaded.

## Train your own model

```bash
python model.py                                  # auto-detects the dataset under ./dataset
python model.py --dataset "C:/path/to/dataset"   # dataset root, Data/ folder, or train folder
```

`--dataset` accepts either the folder containing the class subfolders (e.g. `dataset/Data/train`), the folder containing `train/valid/test`, a Kaggle dataset root containing `Data/train`, or the repository root. You can also set `CTSCAN_DATASET_DIR` instead of passing the flag.

This reads images from the `train/` split, evaluates on `valid/` and `test/` when present, and writes:

- `model.pkl` — the trained RandomForest classifier (compressed)
- `model_classes.json` — the canonical class order used by the app
- `model_info.json` — scikit-learn version, image size, dataset paths, and metrics
- `reports/rf_{train,valid,test}.json` — per-split metrics

### Evaluate a saved model

```bash
python evaluate.py --split test --output reports/rf_test.json
python evaluate.py --model model_cnn.pt --split test
```

`evaluate.py` works with both the RandomForest and the optional CNN checkpoint.

## Baseline metrics

The RandomForest uses flattened 128×128 grayscale pixels (16,384 features), 100 trees, and `random_state=42`. These numbers come from the included `model.pkl` and the Kaggle dataset's own splits:

| Model | Split | Accuracy | Macro F1 | Weighted F1 | Samples |
| --- | --- | ---: | ---: | ---: | ---: |
| RandomForest | train | 99.84% | 0.9981 | 0.9984 | 613 |
| RandomForest | valid | 72.22% | 0.7409 | 0.7192 | 72 |
| RandomForest | test | 47.62% | 0.4862 | 0.4580 | 315 |
| MobileNetV2 (transfer learning, 160 px) | valid | 65.28% | 0.6624 | 0.6385 | 72 |
| MobileNetV2 (transfer learning, 160 px) | test | **61.59%** | **0.5803** | **0.5869** | 315 |

RandomForest test per-class scores:

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| adenocarcinoma | 0.419 | 0.325 | 0.366 | 120 |
| large.cell.carcinoma | 0.294 | 0.098 | 0.147 | 51 |
| normal | 1.000 | 1.000 | 1.000 | 54 |
| squamous.cell.carcinoma | 0.344 | 0.578 | 0.432 | 90 |

MobileNetV2 test per-class scores:

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| adenocarcinoma | 0.542 | 0.700 | 0.611 | 120 |
| large.cell.carcinoma | 0.714 | 0.098 | 0.172 | 51 |
| normal | 1.000 | 0.981 | 0.991 | 54 |
| squamous.cell.carcinoma | 0.520 | 0.578 | 0.547 | 90 |

The large train/test gap is expected for a flattened-pixel model on a small, visually similar dataset. Transfer learning improves test accuracy by ~14 percentage points (47.6% → 61.6%), but `large.cell.carcinoma` remains hard to recall, so this is still a demo, not a diagnostic tool.


## Transfer-learning experiment (optional)

`train_cnn.py` fine-tunes a pretrained MobileNetV2 (or ResNet18) on the same splits. It is deliberately separate from the default deployment path so the Docker image stays small.

```bash
pip install -r requirements-cnn.txt
python train_cnn.py --dataset dataset/dataset --epochs 8 --freeze-epochs 3 --image-size 160
```

This writes `model_cnn.pt` (git-ignored) and `reports/cnn_metrics.json`. The script freezes the backbone for the first `--freeze-epochs`, then unfreezes it for fine-tuning, keeps the best validation macro-F1 checkpoint, and reports test metrics at the end. The included run reached 61.6% test accuracy / 0.580 macro F1, versus 47.6% / 0.486 for the RandomForest baseline.

To serve the CNN instead of the RandomForest:

```bash
# either opt in explicitly...
CTSCAN_USE_CNN=1 python app.py

# ...or point at a specific checkpoint
CTSCAN_MODEL_PATH=model_cnn.pt python app.py
```

On Windows PowerShell, use `$env:CTSCAN_USE_CNN="1"; python app.py`.

CNN results from this repo's run are recorded in `reports/cnn_metrics.json`. Transfer learning is not included in the default Docker image because torch/torchvision add several hundred MB; build a custom image with `requirements-cnn.txt` and the checkpoint if you need it in production.

## API

`POST /upload` with a multipart form field named `image`. Uploaded files are validated by extension, stored under a random UUID name during processing, and deleted afterwards. DICOM files are converted to a temporary PNG first. The maximum upload size is 25 MB.

Example response:

```json
{
  "analysis": {
    "mean_intensity": 0,
    "std_intensity": 0,
    "edge_count": 0,
    "findings": [],
    "disclaimer": "FOR EDUCATIONAL PURPOSES ONLY - NOT FOR MEDICAL USE"
  },
  "model_prediction": "adenocarcinoma",
  "model_prediction_index": 0,
  "predicted_label": "Adenocarcinoma",
  "prediction_description": "Adenocarcinoma: A type of non-small cell lung cancer that starts in glandular cells.",
  "probabilities": [
    {
      "index": 0,
      "class_name": "adenocarcinoma",
      "label": "Adenocarcinoma",
      "probability": 0.42
    },
    {
      "index": 3,
      "class_name": "squamous.cell.carcinoma",
      "label": "Squamous Cell Carcinoma",
      "probability": 0.31
    }
  ],
  "model_type": "random_forest",
  "input_type": "image",
  "class_names": [
    "adenocarcinoma",
    "large.cell.carcinoma",
    "normal",
    "squamous.cell.carcinoma"
  ],
  "class_labels": {
    "adenocarcinoma": "Adenocarcinoma",
    "large.cell.carcinoma": "Large Cell Carcinoma",
    "normal": "Normal",
    "squamous.cell.carcinoma": "Squamous Cell Carcinoma"
  },
  "class_descriptions": {
    "adenocarcinoma": "Adenocarcinoma: A type of non-small cell lung cancer that starts in glandular cells.",
    "large.cell.carcinoma": "Large Cell Carcinoma: A type of non-small cell lung cancer made up of large, abnormal cells.",
    "normal": "Normal: No cancer class detected by this model.",
    "squamous.cell.carcinoma": "Squamous Cell Carcinoma: A type of non-small cell lung cancer that starts in the flat cells lining the airways."
  }
}
```

`model_prediction` is now the predicted **class name**; use `model_prediction_index` if you need the index into `class_names`. `probabilities` is sorted from highest to lowest and sums to 1. `input_type` is `"image"` or `"dicom"`.

## Tests

```bash
python -m pytest
```

The suite covers the index page, `/health`, `/upload` with PNG and DICOM files, unsupported/invalid uploads, probability/index consistency, upload cleanup, class-name mapping, dataset discovery, and metric computation.

## Model distribution decision

`model.pkl` stays **in the repository**. After compression it is ~250 KB, it is required for the app to work out of the box, and it avoids a network fetch or release-artifact dependency at startup. `model_info.json` is committed alongside it so the app knows the class order, image size, and scikit-learn version even when the generated `model_classes.json` is absent.

The optional CNN checkpoint (`model_cnn.pt`, ~9 MB) is **not** committed; it is reproducible with `train_cnn.py` and is excluded by `.dockerignore`/`.gitignore`. If the RandomForest ever grows beyond a few MB, move it to a release artifact or Git LFS instead of committing it.

## Notes for maintainers

- `model.pkl` was retrained with scikit-learn 1.8.0; `requirements.txt` pins `scikit-learn>=1.8,<2.0` to avoid `InconsistentVersionWarning` on load. Retrain with `python model.py` after a major scikit-learn upgrade.
- `model_classes.json` is generated and git-ignored; `model_info.json` is committed and acts as a fallback source of class order.
- The app loads the model once at import. If loading fails, the process still starts, `/health` returns `503`, and `/upload` returns a clear error instead of crashing at boot.
- Uploaded files are deleted in a `finally` block; DICOM conversions are deleted too.
- If you push this repo somewhere that already has the dataset in its git history, remove it with:

  ```bash
  git filter-repo --path dataset --path "test images" --invert-paths
  ```

## License

MIT — see [LICENSE](LICENSE).
