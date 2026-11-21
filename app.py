import os
import uuid
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from class_info import class_metadata
from dicom_utils import DicomError, dicom_to_png, is_dicom_file
from predictor import PredictorUnavailableError, load_predictor

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "bmp",
    "tif",
    "tiff",
    "dcm",
    "dicom",
}
MAX_CONTENT_LENGTH = 25 * 1024 * 1024  # 25 MB

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# Load the model once at startup. The app stays up (and /health reports the
# failure) even when the model file is missing so deployment checks are useful.
try:
    predictor = load_predictor()
    model_error = None
except Exception as exc:  # noqa: BLE001 - startup should not crash the process
    predictor = None
    model_error = str(exc)
    app.logger.warning("Model could not be loaded: %s", exc)


def basic_ct_analysis(image_path):
    """
    EDUCATIONAL PURPOSES ONLY - NOT FOR MEDICAL DIAGNOSIS
    This is a basic image analysis demonstration
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Could not read the uploaded image.")

    # Basic image statistics
    mean_intensity = np.mean(image)
    std_intensity = np.std(image)

    # Edge detection for structural analysis
    edges = cv2.Canny(image, 100, 200)
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Simple brightness/darkness analysis
    dark_regions = np.sum(image < 50)  # Very dark pixels
    bright_regions = np.sum(image > 200)  # Very bright pixels

    # Basic "analysis" (NOT medical diagnosis)
    findings = []

    if std_intensity > 60:
        findings.append(
            "High contrast regions detected (could indicate structural variations)"
        )

    if len(contours) > 100:
        findings.append("Multiple edge structures detected")

    if dark_regions > (image.size * 0.3):
        findings.append("Significant dark regions present")

    if bright_regions > (image.size * 0.1):
        findings.append(
            "High-density regions detected (possible bone/contrast material)"
        )

    # Always include disclaimer
    findings.append("⚠️ IMPORTANT: This is NOT a medical diagnosis!")
    findings.append(
        "⚠️ Consult a qualified radiologist for proper interpretation!"
    )

    return {
        "mean_intensity": float(mean_intensity),
        "std_intensity": float(std_intensity),
        "edge_count": len(contours),
        "findings": findings,
        "disclaimer": "FOR EDUCATIONAL PURPOSES ONLY - NOT FOR MEDICAL USE",
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    payload = {
        "status": "ok" if predictor is not None else "error",
        "model_loaded": predictor is not None,
        "model_type": getattr(predictor, "model_type", None),
        "classes": getattr(predictor, "classes", []),
        "sklearn_version": getattr(predictor, "sklearn_version", None),
    }
    if model_error:
        payload["error"] = model_error
    return jsonify(payload), (200 if predictor is not None else 503)


@app.errorhandler(413)
def file_too_large(_error):
    limit_mb = MAX_CONTENT_LENGTH // (1024 * 1024)
    return (
        jsonify({"error": f"File too large. Maximum size is {limit_mb} MB."}),
        413,
    )


@app.route("/upload", methods=["POST"])
def upload_image():
    if predictor is None:
        return (
            jsonify(
                {
                    "error": "Model is not available.",
                    "detail": model_error,
                }
            ),
            503,
        )

    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["image"]
    if not file or not file.filename:
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(file.filename)
    if not filename:
        return jsonify({"error": "Invalid filename"}), 400

    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension not in ALLOWED_EXTENSIONS:
        return (
            jsonify(
                {
                    "error": "Unsupported file type. Allowed: PNG, JPEG, BMP, "
                    "TIFF, DICOM."
                }
            ),
            400,
        )

    token = uuid.uuid4().hex
    filepath = UPLOAD_FOLDER / f"{token}_{filename}"
    converted_path = None

    try:
        file.save(filepath)

        if is_dicom_file(filepath):
            converted_path = UPLOAD_FOLDER / f"{token}_converted.png"
            try:
                dicom_to_png(filepath, converted_path)
            except DicomError as exc:
                return jsonify({"error": str(exc)}), 400
            processed_path = converted_path
            input_type = "dicom"
        else:
            processed_path = filepath
            input_type = "image"

        # Basic CT analysis (educational only)
        analysis = basic_ct_analysis(processed_path)

        # Model prediction + per-class probabilities
        prediction = predictor.predict(processed_path)
        metadata = class_metadata(predictor.classes)

        return jsonify(
            {
                "analysis": analysis,
                "model_prediction": prediction["class_name"],
                "model_prediction_index": prediction["index"],
                "predicted_label": prediction["label"],
                "prediction_description": prediction["description"],
                "probabilities": prediction["probabilities"],
                "model_type": predictor.model_type,
                "input_type": input_type,
                **metadata,
            }
        )

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except PredictorUnavailableError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception:  # noqa: BLE001 - return JSON instead of an HTML error page
        app.logger.exception("Failed to process upload")
        return jsonify({"error": "Could not process the uploaded image."}), 500

    finally:
        filepath.unlink(missing_ok=True)
        if converted_path is not None:
            converted_path.unlink(missing_ok=True)


if __name__ == "__main__":
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5000")),
        debug=os.environ.get("FLASK_DEBUG", "0") == "1",
    )
