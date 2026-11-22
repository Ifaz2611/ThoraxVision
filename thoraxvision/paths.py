"""Canonical filesystem locations for the ThoraxVision project."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = Path(__file__).resolve().parent

MODELS_DIR = REPO_ROOT / "models"
DATASET_DIR = REPO_ROOT / "dataset"
DATA_DIR = REPO_ROOT / "data"
REPORTS_DIR = REPO_ROOT / "reports"
UPLOADS_DIR = REPO_ROOT / "uploads"
