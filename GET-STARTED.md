# Get Started

New here? This guide gets the app running in ~5 minutes. For full details see [README.md](README.md).

> **Reminder:** This is an educational demo, **not** a medical tool. Do not use it for diagnosis.

## 1. Install Python

You need Python 3.10 or newer. Check with:

```bash
python --version
```

## 2. Clone and set up

```bash
git clone https://github.com/Ifaz2611/CT-Scan-Image-Analyzer_Using_ML
cd CT-Scan-Image-Analyzer_Using_ML

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## 3. Run the app

```bash
python app.py
```

Open <http://127.0.0.1:5000> in your browser and upload a CT scan image (PNG, JPEG, BMP, or TIFF).

That's it — the app works out of the box because `model.pkl` is included in the repo. No dataset download needed just to try it.

## Optional: retrain the model yourself

Only needed if you want to train on your own copy of the data:

1. Download the [Chest CT-Scan Images dataset](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images) from Kaggle.
2. Unzip it anywhere (it does **not** need to go inside this repo).
3. Train, pointing at the data:

   ```bash
   python model.py --dataset "C:/path/to/dataset"
   ```

   `--dataset` accepts the folder containing the class subfolders (e.g. `.../Data/train`) or the dataset root. You can also set the `CTSCAN_DATASET_DIR` environment variable instead.

This rewrites `model.pkl` and creates `model_classes.json`.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `No module named 'flask'` | You forgot to activate the venv or install requirements — redo step 2 |
| `Dataset not found at ...` when training | Pass `--dataset <path>` or check the folder layout in README.md |
| `InconsistentVersionWarning` on startup | `model.pkl` was trained on scikit-learn 1.6.x — retrain with `python model.py` or pin `scikit-learn==1.6.*` |
| Upload fails with "Unsupported file type" | Only PNG/JPEG/BMP/TIFF are accepted |
