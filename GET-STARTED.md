# Get Started

New here? This guide gets the app running in ~5 minutes. For full details see [README.md](README.md).

> **Reminder:** This is an educational demo, **not** a medical tool. Do not use it for diagnosis.

## 1. Install Python

You need Python **3.11 or newer** (scikit-learn 1.8 requires it). Check with:

```bash
python --version
```

## 2. Clone and set up

```bash
git clone https://github.com/Ifaz2611/ThoraxVision
cd ThoraxVision

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## 3. Run the app

```bash
python -m thoraxvision.app
```

Open <http://127.0.0.1:5000> in your browser and upload a CT scan image (PNG, JPEG, BMP, TIFF, or DICOM `.dcm`). DICOM files are converted to PNG automatically.

That's it — the app works out of the box because `models/model.pkl` and `models/model_info.json` are included in the repo. No dataset download needed just to try it.

The results show the predicted class name, its description, the per-class probabilities, and the basic image statistics.

## 4. Production server (optional)

Flask's development server is not for production.

```bash
# Windows / macOS / Linux (waitress)
python -m thoraxvision.serve
# -> http://0.0.0.0:8000

# Linux / macOS (gunicorn)
gunicorn --bind 0.0.0.0:8000 --workers 2 --timeout 120 thoraxvision.app:app
```

Or use Docker:

```bash
docker build -t ctscan-analyzer .
docker run --rm -p 8000:8000 ctscan-analyzer
```

Check it with <http://127.0.0.1:8000/health>.

## 5. Run the tests (optional)

```bash
pip install -r requirements-dev.txt
python -m pytest
```

## Optional: retrain the model yourself

Only needed if you want to train on your own copy of the data:

1. Download the [Chest CT-Scan Images dataset](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images) from Kaggle.
2. Unzip it anywhere (it does **not** need to go inside this repo).
3. Train, pointing at the data:

   ```bash
   python scripts/train_model.py --dataset "C:/path/to/dataset"
   ```

   `--dataset` accepts the folder containing the class subfolders (e.g. `.../Data/train`), the folder containing `train/valid/test`, or the dataset root. You can also set the `CTSCAN_DATASET_DIR` environment variable instead.

This rewrites `models/model.pkl`, creates `models/model_classes.json` and `models/model_info.json`, and saves valid/test metrics under `reports/`.

## Optional: try the transfer-learning CNN

```bash
pip install -r requirements-cnn.txt
python scripts/train_cnn.py --dataset dataset/Data --epochs 8 --freeze-epochs 3 --image-size 160
CTSCAN_USE_CNN=1 python -m thoraxvision.app
# Windows PowerShell: $env:CTSCAN_USE_CNN="1"; python -m thoraxvision.app
```

This produces `models/model_cnn.pt` and `reports/cnn_metrics.json`. See README.md for the measured comparison.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `No module named 'flask'` | You forgot to activate the venv or install requirements — redo step 2 |
| `Dataset not found at ...` when training | Pass `--dataset <path>` or check the folder layout in README.md |
| `InconsistentVersionWarning` on startup | Retrain with `python scripts/train_model.py` (the included model was trained with scikit-learn 1.8) |
| Upload fails with "Unsupported file type" | Only PNG/JPEG/BMP/TIFF/DICOM are accepted |
| DICOM upload says pydicom is missing | `pip install pydicom` (already in requirements.txt) |
| CNN training says torch is missing | `pip install -r requirements-cnn.txt` |
| Port 5000 already in use | Set `PORT=5001` before running `python -m thoraxvision.app` (PowerShell: `$env:PORT="5001"`) |
