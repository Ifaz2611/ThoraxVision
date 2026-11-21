# TODO

All items below are implemented. See README.md for usage and measured metrics.

## Code quality
- [x] Retrain `model.pkl` on the current scikit-learn version to remove the `InconsistentVersionWarning`
  - Retrained with scikit-learn 1.8.0 and pinned `scikit-learn>=1.8,<2.0` in `requirements.txt`.
- [x] Map `model_prediction` index to the actual class name in the API response instead of returning a bare index
  - `model_prediction` is now the class name; `model_prediction_index`, `predicted_label`, and `prediction_description` are also returned.
- [x] Update `CLASS_NAMES` / `CLASS_DESCRIPTIONS` fallback in `app.py` to match the real dataset folder names (or generate descriptions from `model_classes.json`)
  - `class_info.py` canonicalises the long `train/`/`valid/` folder names and the short `test/` names, and generates labels/descriptions for custom classes. `model_classes.json` and `model_info.json` are both used as fallbacks.
- [x] Add a basic test (e.g. `tests/test_app.py`) covering the `/upload` endpoint with a sample image
  - Added `tests/test_app.py` (PNG + DICOM uploads, errors, cleanup) plus `tests/test_class_info.py` and `tests/test_dataset_utils.py`.

## Features
- [x] Show class probabilities (`model.predict_proba`) alongside the top prediction
  - `/upload` returns a sorted `probabilities` array; the UI renders probability bars.
- [x] Accept DICOM files or convert them to PNG before analysis
  - `dicom_utils.py` detects DICOM by extension/magic bytes, applies rescale/windowing, and converts the first frame to an 8-bit PNG before analysis. Requires `pydicom`.
- [x] Add a `/health` endpoint for deployment checks
  - Returns model status, type, class list, and scikit-learn version (`200`/`503`); used by the Docker `HEALTHCHECK`.

## Deployment
- [x] Add a `Dockerfile` and `.dockerignore`
  - Python 3.12 slim image, non-root user, gunicorn, `/health` healthcheck.
- [x] Add `gunicorn`/`waitress` production server instructions (Flask's dev server is not for production)
  - Added `serve.py` (waitress, cross-platform) and documented gunicorn; both are in `requirements.txt`.
- [x] Decide how `model.pkl` is distributed — keep in repo, or fetch from a release artifact
  - Kept in the repo (~250 KB compressed). `model_info.json` is committed alongside it; the optional `model_cnn.pt` is git-ignored and reproducible with `train_cnn.py`.

## Dataset / experiments
- [x] Try transfer learning (e.g. a small CNN) instead of flattened-pixel RandomForest — accuracy will likely improve a lot
  - Added `train_cnn.py` (MobileNetV2/ResNet18 fine-tuning) and optional app support via `CTSCAN_USE_CNN=1` or `CTSCAN_MODEL_PATH=model_cnn.pt`. The included run improved test accuracy from 47.6% (RandomForest) to 61.6% (MobileNetV2), and macro F1 from 0.486 to 0.580.
- [x] Evaluate on the dataset's `test/` split and record baseline metrics in README
  - Added `evaluate.py` and `reports/`; baseline RandomForest metrics are recorded in README.md.
- [x] Handle `valid/` and `test/` splits in `model.py` (currently only `train/` is used)
  - `dataset_utils.resolve_dataset_splits()` auto-detects `train`/`valid`/`test` (including nested Kaggle layouts); `model.py` evaluates all available splits.
