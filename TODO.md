# TODO

## Code quality
- [ ] Retrain `model.pkl` on the current scikit-learn version to remove the `InconsistentVersionWarning` (or pin `scikit-learn==1.6.*` in requirements.txt)
- [ ] Map `model_prediction` index to the actual class name in the API response instead of returning a bare index
- [ ] Update `CLASS_NAMES` / `CLASS_DESCRIPTIONS` fallback in `app.py` to match the real dataset folder names (or generate descriptions from `model_classes.json`)
- [ ] Add a basic test (e.g. `tests/test_app.py`) covering the `/upload` endpoint with a sample image

## Features
- [ ] Show class probabilities (`model.predict_proba`) alongside the top prediction
- [ ] Accept DICOM files or convert them to PNG before analysis
- [ ] Add a `/health` endpoint for deployment checks

## Deployment
- [ ] Add a `Dockerfile` and `.dockerignore`
- [ ] Add `gunicorn`/`waitress` production server instructions (Flask's dev server is not for production)
- [ ] Decide how `model.pkl` is distributed — keep in repo, or fetch from a release artifact

## Dataset / experiments
- [ ] Try transfer learning (e.g. a small CNN) instead of flattened-pixel RandomForest — accuracy will likely improve a lot
- [ ] Evaluate on the dataset's `test/` split and record baseline metrics in README
- [ ] Handle `valid/` and `test/` splits in `model.py` (currently only `train/` is used)
