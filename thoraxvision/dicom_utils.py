"""Optional DICOM support: convert a DICOM image to an 8-bit PNG."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

try:
    import pydicom
except ImportError:  # pragma: no cover - exercised when pydicom is absent
    pydicom = None

DICOM_EXTENSIONS = {".dcm", ".dicom"}


class DicomError(ValueError):
    """Raised when a DICOM file cannot be read or decoded."""


def is_dicom_file(path: Path | str) -> bool:
    """Detect DICOM by extension or by the ``DICM`` magic marker."""
    path = Path(path)
    if path.suffix.lower() in DICOM_EXTENSIONS:
        return True
    try:
        with path.open("rb") as handle:
            handle.seek(128)
            return handle.read(4) == b"DICM"
    except OSError:
        return False


def _first(value):
    if isinstance(value, (str, bytes)):
        return value
    if hasattr(value, "__iter__"):
        try:
            return next(iter(value), None)
        except TypeError:
            pass
    return value


def dicom_to_png(source: Path | str, destination: Path | str) -> Path:
    """Convert the first frame of a DICOM file to an 8-bit grayscale PNG.

    Applies rescale slope/intercept and the window center/width when present,
    then normalises the result to 0-255.
    """
    if pydicom is None:
        raise DicomError(
            "DICOM support requires pydicom. Install it with: pip install pydicom"
        )

    source = Path(source)
    destination = Path(destination)

    try:
        dataset = pydicom.dcmread(str(source))
    except Exception as exc:  # pydicom raises several different exception types
        raise DicomError(f"Could not read DICOM file: {exc}") from exc

    try:
        pixels = np.asarray(dataset.pixel_array)
    except Exception as exc:
        raise DicomError(f"Could not decode DICOM pixel data: {exc}") from exc

    if pixels.ndim == 4:  # (frames, rows, columns, channels)
        pixels = pixels[0]
    elif pixels.ndim == 3:
        # Multi-frame grayscale keeps only the first frame. Single-frame RGB
        # (rows, columns, channels) is converted to grayscale below.
        number_of_frames = int(getattr(dataset, "NumberOfFrames", 1) or 1)
        if number_of_frames > 1 or getattr(dataset, "SamplesPerPixel", 1) == 1:
            pixels = pixels[0]

    if pixels.ndim == 3 and pixels.shape[-1] in (3, 4):
        image = Image.fromarray(pixels.astype(np.uint8)).convert("L")
        image.save(destination)
        return destination

    pixels = pixels.astype(np.float32)
    slope = float(_first(getattr(dataset, "RescaleSlope", 1)) or 1)
    intercept = float(_first(getattr(dataset, "RescaleIntercept", 0)) or 0)
    pixels = pixels * slope + intercept

    center = _first(getattr(dataset, "WindowCenter", None))
    width = _first(getattr(dataset, "WindowWidth", None))
    if center is not None and width is not None:
        center = float(center)
        width = float(width)
        if width > 1:
            pixels = np.clip(pixels, center - width / 2.0, center + width / 2.0)
        elif width == 1:
            pixels = np.where(pixels > center, 1.0, 0.0)

    low = float(np.min(pixels))
    high = float(np.max(pixels))
    if high > low:
        pixels = (pixels - low) / (high - low) * 255.0
    else:
        pixels = np.zeros_like(pixels)

    if getattr(dataset, "PhotometricInterpretation", "") == "MONOCHROME1":
        pixels = 255.0 - pixels

    image = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
    image.save(destination)
    return destination
