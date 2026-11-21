import json
import os
import uuid
from pathlib import Path

import cv2
import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
MODEL_PATH = BASE_DIR / "model.pkl"
CLASSES_PATH = BASE_DIR / "model_classes.json"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}

# Default class order (matches the sorted folder names used by model.py).
# model.py writes model_classes.json, which takes priority when present.
CLASS_NAMES = [
    "adenocarcinoma",
    "large.cell.carcinoma",
    "normal",
    "squamous.cell.carcinoma",
]

CLASS_DESCRIPTIONS = {
    "adenocarcinoma": "Adenocarcinoma: A type of lung cancer.",
    "large.cell.carcinoma": "Large Cell Carcinoma: A type of lung cancer.",
    "normal": "Normal: No cancer detected.",
    "squamous.cell.carcinoma": "Squamous Cell Carcinoma: A type of lung cancer.",
}


def load_class_names():
    if CLASSES_PATH.exists():
        try:
            names = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
            if isinstance(names, list) and names:
                return names
        except (json.JSONDecodeError, OSError):
            pass
    return CLASS_NAMES


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# Load the trained RandomForest model
model = joblib.load(MODEL_PATH)
class_names = load_class_names()


def preprocess_image_for_model(image_path):
    # For RandomForest: grayscale, resize, flatten, normalize
    img = Image.open(image_path).convert("L")
    img = img.resize((128, 128))
    arr = np.array(img).flatten() / 255.0
    return arr.reshape(1, -1)


def basic_ct_analysis(image_path):
    """
    EDUCATIONAL PURPOSES ONLY - NOT FOR MEDICAL DIAGNOSIS
    This is a basic image analysis demonstration
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Basic image statistics
    mean_intensity = np.mean(image)
    std_intensity = np.std(image)

    # Edge detection for structural analysis
    edges = cv2.Canny(image, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Simple brightness/darkness analysis
    dark_regions = np.sum(image < 50)  # Very dark pixels
    bright_regions = np.sum(image > 200)  # Very bright pixels

    # Basic "analysis" (NOT medical diagnosis)
    findings = []

    if std_intensity > 60:
        findings.append("High contrast regions detected (could indicate structural variations)")

    if len(contours) > 100:
        findings.append("Multiple edge structures detected")

    if dark_regions > (image.size * 0.3):
        findings.append("Significant dark regions present")

    if bright_regions > (image.size * 0.1):
        findings.append("High-density regions detected (possible bone/contrast material)")

    # Always include disclaimer
    findings.append("⚠️ IMPORTANT: This is NOT a medical diagnosis!")
    findings.append("⚠️ Consult a qualified radiologist for proper interpretation!")

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


@app.route("/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(file.filename)
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Unsupported file type"}), 400

    filepath = UPLOAD_FOLDER / f"{uuid.uuid4().hex}_{filename}"

    try:
        file.save(filepath)

        # Basic CT analysis (educational only)
        analysis = basic_ct_analysis(str(filepath))

        # Model prediction
        img_arr = preprocess_image_for_model(filepath)
        prediction = model.predict(img_arr)
        prediction_label = str(prediction[0])

        return jsonify(
            {
                "analysis": analysis,
                "model_prediction": prediction_label,
                "class_names": class_names,
                "class_descriptions": CLASS_DESCRIPTIONS,
            }
        )

    except Exception as e:
        import traceback

        print(traceback.format_exc())
        return jsonify({"error": f"Error: {str(e)}"}), 500

    finally:
        filepath.unlink(missing_ok=True)


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG", "0") == "1")
