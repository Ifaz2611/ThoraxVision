import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app import UPLOAD_FOLDER, app


@pytest.fixture()
def client():
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        yield test_client


def png_bytes(size=(64, 64), value=128):
    image = Image.new("L", size, color=value)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def dicom_bytes():
    pydicom = pytest.importorskip("pydicom")
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import (
        ExplicitVRLittleEndian,
        SecondaryCaptureImageStorage,
        generate_uid,
    )

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    dataset = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.Modality = "CT"
    dataset.PatientName = "Test^Patient"
    dataset.PatientID = "TEST001"
    dataset.StudyInstanceUID = generate_uid()
    dataset.SeriesInstanceUID = generate_uid()
    dataset.StudyDate = "20260101"
    dataset.StudyTime = "120000"
    dataset.ContentDate = "20260101"
    dataset.ContentTime = "120000"
    dataset.Rows = 64
    dataset.Columns = 64
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.RescaleSlope = 1
    dataset.RescaleIntercept = -1024
    dataset.WindowCenter = 40
    dataset.WindowWidth = 400
    dataset.PixelData = np.arange(64 * 64, dtype=np.uint16).tobytes()

    buffer = io.BytesIO()
    dataset.save_as(buffer, enforce_file_format=True)
    return buffer.getvalue()


def post_file(client, filename, data, field="image"):
    return client.post(
        "/upload",
        data={field: (io.BytesIO(data), filename)},
        content_type="multipart/form-data",
    )


def test_health_reports_loaded_model(client):
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
    assert payload["model_type"]
    assert payload["classes"]


def test_index_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"CT Scan Image Analyzer" in response.data


def test_upload_png_returns_class_name_and_probabilities(client):
    response = post_file(client, "scan.png", png_bytes())
    assert response.status_code == 200, response.get_data(as_text=True)

    payload = response.get_json()
    assert payload["input_type"] == "image"
    assert payload["model_prediction"] in payload["class_names"]
    index = payload["model_prediction_index"]
    assert payload["class_names"][index] == payload["model_prediction"]
    assert payload["predicted_label"]
    assert payload["prediction_description"]
    assert payload["class_labels"][payload["model_prediction"]]

    probabilities = payload["probabilities"]
    assert len(probabilities) == len(payload["class_names"])
    assert sum(item["probability"] for item in probabilities) == pytest.approx(
        1.0, abs=1e-6
    )
    assert [item["probability"] for item in probabilities] == sorted(
        (item["probability"] for item in probabilities), reverse=True
    )
    assert probabilities[0]["class_name"] == payload["model_prediction"]


def test_upload_dicom_is_converted_and_analyzed(client):
    response = post_file(client, "scan.dcm", dicom_bytes())
    assert response.status_code == 200, response.get_data(as_text=True)
    payload = response.get_json()
    assert payload["input_type"] == "dicom"
    assert payload["model_prediction"] in payload["class_names"]


def test_upload_rejects_unsupported_extension(client):
    response = post_file(client, "scan.gif", b"GIF89a")
    assert response.status_code == 400
    assert "Unsupported" in response.get_json()["error"]


def test_upload_rejects_missing_file(client):
    response = client.post("/upload", data={}, content_type="multipart/form-data")
    assert response.status_code == 400
    assert "No file" in response.get_json()["error"]


def test_upload_rejects_invalid_image(client):
    response = post_file(client, "not-an-image.png", b"this is not a PNG")
    assert response.status_code == 400
    assert "Could not read" in response.get_json()["error"]


def test_uploaded_files_are_cleaned_up(client):
    before = {path.name for path in Path(UPLOAD_FOLDER).iterdir()}
    response = post_file(client, "scan.png", png_bytes())
    assert response.status_code == 200
    after = {path.name for path in Path(UPLOAD_FOLDER).iterdir()}
    assert after == before
