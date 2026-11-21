# ML-Based CT Scan Image Analyzer

A small Flask web app that runs basic image statistics and a RandomForest classifier on uploaded CT scan images. It was built as a learning/demo project, not as a medical tool.

> **Disclaimer:** This is **not** a medical device and must **not** be used for diagnosis. Always consult a qualified radiologist.

## What it does

- Upload a CT scan image (PNG, JPEG, BMP, or TIFF)
- Shows basic image statistics: mean intensity, standard deviation, and edge count
- Predicts one of four classes: `adenocarcinoma`, `large.cell.carcinoma`, `normal`, `squamous.cell.carcinoma`
- Always displays an educational-use disclaimer

## Project structure

```
.
├── app.py                 # Flask app: upload, analysis, prediction
├── model.py               # Trains and saves model.pkl
├── model.pkl              # Trained RandomForest model (included)
├── requirements.txt       # Python dependencies
├── static/
│   ├── script.js
│   └── style.css
├── templates/
│   └── index.html
├── uploads/               # Created at runtime, git-ignored
└── legacy/                # Old experiments, git-ignored
```

## Dataset (not included)

The training images are **not** part of this repository — they are large and should never be committed. Download the Chest CT-Scan Images dataset from Kaggle:

- <https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images>

Unzip it so the folder layout is:

```
dataset/
└── Data/
    └── train/
        ├── adenocarcinoma/
        ├── large.cell.carcinoma/
        ├── normal/
        └── squamous.cell.carcinoma/
```

`dataset/` and `test images/` are listed in `.gitignore`, so they will not be committed again.

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

## Train your own model

```bash
python model.py
```

This reads images from `dataset/Data/train/<class>/` and writes:

- `model.pkl` — the trained RandomForest classifier
- `model_classes.json` — the class order used by the app

The app falls back to its built-in class list if `model_classes.json` is missing.

## API

`POST /upload` with a multipart form field named `image`.

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

## License

MIT — see [LICENSE](LICENSE).
