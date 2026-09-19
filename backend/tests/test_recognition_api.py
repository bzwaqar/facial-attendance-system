"""Unit and integration tests for Face Recognition API endpoints."""

import base64
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _make_sample_image_bytes():
    """Create a valid PNG image byte array with synthetic face."""
    img = np.ones((240, 240, 3), dtype=np.uint8) * 128
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def test_predict_recognition_file_empty_payload():
    """Verify empty image submission returns HTTP 400."""
    response = client.post(
        "/api/v1/recognition/predict",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_predict_recognition_file_valid_image():
    """Verify recognition endpoint runs on valid image bytes."""
    img_bytes = _make_sample_image_bytes()
    response = client.post(
        "/api/v1/recognition/predict",
        files={"file": ("test.png", img_bytes, "image/png")},
        data={"threshold": 0.60, "check_liveness": False},
    )
    assert response.status_code == 200
    data = response.json()
    assert "total_faces_detected" in data
    assert "faces" in data
    assert "inference_time_ms" in data
    assert data["inference_time_ms"] >= 0


def test_predict_recognition_base64_endpoint():
    """Verify base64 recognition endpoint parses and evaluates correctly."""
    img_bytes = _make_sample_image_bytes()
    b64_str = base64.b64encode(img_bytes).decode("utf-8")
    payload = {
        "image_base64": f"data:image/png;base64,{b64_str}",
        "threshold": 0.65,
        "check_liveness": True,
    }

    response = client.post("/api/v1/recognition/predict-base64", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "total_faces_detected" in data
    assert "faces" in data


def test_predict_recognition_invalid_base64():
    """Verify malformed base64 input returns HTTP 400."""
    payload = {
        "image_base64": "not_valid_base64_!@#$",
        "threshold": 0.60,
    }
    response = client.post("/api/v1/recognition/predict-base64", json=payload)
    assert response.status_code == 400
