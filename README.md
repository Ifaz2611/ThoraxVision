# ML-Based CT Scan Image Analyzer

A small Flask web app that runs basic image statistics and a RandomForest classifier on uploaded CT scan images. It was built as a learning/demo project, not as a medical tool.

> **Disclaimer:** This is **not** a medical device and must **not** be used for diagnosis. Always consult a qualified radiologist.

## What it does

- Upload a CT scan image (PNG, JPEG, BMP, or TIFF)
- Shows basic image statistics: mean intensity, standard deviation, and edge count
- Predicts a class label using a trained RandomForest model
- Always displays an educational-use disclaimer

## Project structure

```
.
├── app.py                 # Flask app: upload, analysis, prediction
├── model.py               # Trains model.pkl from the dataset
├── model.pkl              # Trained RandomForest model (included, ~1 MB)
├── model_classes.json     # Class order written by model.py (git-ignored, optional)
├── requirements.txt       # Python dependencies
├── static/
│   ├── script.js
│   └── style.css
├── templates/
│   └── index.html
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

The class labels come from the **folder names** inside `train/`. `dataset/`, `test images/`, and `_dataset_backup/` are all listed in `.gitignore`.

If you keep the dataset outside the repo (recommended), point `model.py` at it with `--dataset` or the `CTSCAN_DATASET_DIR` environment variable — see below.

## Setup

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

## Run

```bash
python app.py
```

Then open <http://127.0.0.1:5000> and upload an image. The server runs with debug mode off by default; set `FLASK_DEBUG=1` to enable it.

`app.py` loads `model.pkl` at startup and needs no dataset — it only requires the model file, which is committed to the repo.

## Train your own model

```bash
python model.py                                  # uses ./dataset/Data/train
python model.py --dataset "C:/path/to/dataset"   # or a dataset root / train folder
```

`--dataset` accepts either the folder containing the class subfolders (e.g. `dataset/Data/train`) or a dataset root that contains `Data/train`. You can also set `CTSCAN_DATASET_DIR` instead of passing the flag.

This reads images from `<dataset>/train/<class>/` and writes:

- `model.pkl` — the trained RandomForest classifier
- `model_classes.json` — the class order used by the app (sorted folder names)

The app falls back to its built-in class list if `model_classes.json` is missing, and `CLASS_DESCRIPTIONS` only covers the default class names — custom folder names still work, they just won't have friendly descriptions.

## API

`POST /upload` with a multipart form field named `image`. Uploaded files are validated by extension, stored under a random UUID name during processing, and deleted afterwards.

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
  "model_prediction": "0",
  "class_names": [
    "adenocarcinoma",
    "large.cell.carcinoma",
    "normal",
    "squamous.cell.carcinoma"
  ],
  "class_descriptions": {
    "adenocarcinoma": "Adenocarcinoma: A type of lung cancer.",
    "large.cell.carcinoma": "Large Cell Carcinoma: A type of lung cancer.",
    "normal": "Normal: No cancer detected.",
    "squamous.cell.carcinoma": "Squamous Cell Carcinoma: A type of lung cancer."
  }
}
```

`model_prediction` is the predicted class **index**; look it up in `class_names`.

## Notes for maintainers

- `model.pkl` was trained with scikit-learn 1.6.x; newer versions may emit an `InconsistentVersionWarning` on load. Retrain with `python model.py` if you hit issues.
- If you push this repo somewhere that already has the dataset in its git history, remove it with:

  ```bash
  git filter-repo --path dataset --path "test images" --invert-paths
  ```

## License

MIT — see [LICENSE](LICENSE).
