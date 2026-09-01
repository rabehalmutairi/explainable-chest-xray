"""Thin HTTP client for the chest X-ray FastAPI backend.

No ML logic here - this module only calls the API and returns plain data
(dicts, bytes, headers). All preprocessing/inference/Grad-CAM lives in the
backend; the frontend consumes it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests

API_BASE_URL = os.environ.get("CHEST_XRAY_API_URL", "http://127.0.0.1:8123")
_TIMEOUT_HEALTH = 5
_TIMEOUT_PREDICT = 30
_TIMEOUT_GRADCAM = 30


class ApiError(RuntimeError):
    """Raised for any backend failure the UI should show to the user."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class GradCamResult:
    image_bytes: bytes
    pathology: str
    probability: float
    threshold: float
    above_threshold: bool
    target_layer: str
    note: str


def _raise_for_backend_error(resp: requests.Response) -> None:
    if resp.ok:
        return
    detail = resp.reason
    try:
        body = resp.json()
        detail = body.get("detail", detail)
    except ValueError:
        pass
    raise ApiError(str(detail), status_code=resp.status_code)


def check_health() -> dict[str, Any] | None:
    """Returns the /health payload, or None if the backend is unreachable."""
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=_TIMEOUT_HEALTH)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return None


def get_model_info() -> dict[str, Any]:
    resp = requests.get(f"{API_BASE_URL}/model", timeout=_TIMEOUT_HEALTH)
    _raise_for_backend_error(resp)
    return resp.json()


def predict(image_bytes: bytes, filename: str) -> dict[str, Any]:
    """POST /predict. Returns the full JSON response (all 14 predictions)."""
    files = {"file": (filename, image_bytes, "application/octet-stream")}
    try:
        resp = requests.post(f"{API_BASE_URL}/predict", files=files, timeout=_TIMEOUT_PREDICT)
    except requests.RequestException as exc:
        raise ApiError(f"Could not reach the analysis backend: {exc}") from exc
    _raise_for_backend_error(resp)
    return resp.json()


def get_gradcam(image_bytes: bytes, filename: str, pathology: str) -> GradCamResult:
    """POST /gradcam. Returns the PNG overlay plus the metadata carried in headers."""
    files = {"file": (filename, image_bytes, "application/octet-stream")}
    data = {"pathology": pathology}
    try:
        resp = requests.post(
            f"{API_BASE_URL}/gradcam", files=files, data=data, timeout=_TIMEOUT_GRADCAM
        )
    except requests.RequestException as exc:
        raise ApiError(f"Could not reach the analysis backend: {exc}") from exc
    _raise_for_backend_error(resp)

    h = resp.headers
    return GradCamResult(
        image_bytes=resp.content,
        pathology=h.get("X-Pathology", pathology),
        probability=float(h.get("X-Probability", 0.0)),
        threshold=float(h.get("X-Threshold", 0.0)),
        above_threshold=h.get("X-Above-Threshold", "False") == "True",
        target_layer=h.get("X-Target-Layer", ""),
        note=h.get("X-Explain-Note", ""),
    )
