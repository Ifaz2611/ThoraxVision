# Contributing to ThoraxVision

Thanks for your interest in contributing. This project is a learning/educational
project — contributions that improve code quality, tests, docs, or the training
pipeline are all welcome.

## Getting started

1. Fork and clone the repo.
2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   # source venv/bin/activate   # macOS/Linux

   pip install -r requirements.txt
   pip install -r requirements-dev.txt   # pytest and friends
   ```

3. Run the app locally:

   ```bash
   python -m thoraxvision.app
   ```

## Running tests

```bash
pytest
```

Or run a subset:

```bash
pytest tests/test_app.py -v
```

All tests must pass before submitting a PR.

## Code style

- **Python 3.11+**
- Follow existing conventions: compact code, minimal comments, no unnecessary
  error handling. Keep Flask handlers thin and put logic in the package
  (`thoraxvision/`).
- Keep `requirements.txt` lean — heavy optional deps (torch/torchvision) stay in
  `requirements-cnn.txt`.
- Add a test under `tests/` for any behavior change.

## Reporting issues

Please include:

- Python version and OS
- The exact error or unexpected behavior
- Steps to reproduce (or a small example image/DICOM file)
- Relevant logs or screenshots of the browser UI

## Pull requests

1. Branch off `main` (e.g. `fix/upload-validation`, `feat/metrics-page`).
2. Keep PRs focused — one logical change per PR.
3. Update the README if your change affects setup, usage, or the API.
4. Ensure `pytest` passes and the app still runs end-to-end.

## Medical disclaimer

This is **not** a medical device. Do not submit changes that could be interpreted
as improving diagnostic accuracy claims, and do not use real patient data in
issues, tests, or PRs. The `dataset/` directory should only ever contain
publicly available research data.

## Code of Conduct

All contributors are expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
