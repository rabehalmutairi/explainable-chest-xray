"""FastAPI service for the ConvNeXt-Tiny 320 chest X-ray classifier.

Run:
    set CHEST_XRAY_CHECKPOINT=models\\convnext_tiny_320_inference.pth
    uvicorn app.main:app --reload

All ML logic (model loading, preprocessing, inference, thresholds, Grad-CAM)
lives in src/inference.py and src/gradcam.py. This file is HTTP only: the
frontend consumes these responses rather than reimplementing any of it.

Uploads are held in memory for the duration of one request and never written
to disk or cached server-side.

Research prototype. Not for clinical use.
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.gradcam import EXPLAIN_NOTE, TARGET_LAYER_NAME, GradCAM  # noqa: E402
from src.inference import (  # noqa: E402
    LABELS,
    MAX_UPLOAD_BYTES,
    MODEL_VERSION,
    ChestXrayClassifier,
    InvalidImageError,
)

from app.schemas import HealthResponse, ModelInfoResponse, PredictResponse  # noqa: E402

CHECKPOINT_ENV = "CHEST_XRAY_CHECKPOINT"
DEFAULT_CHECKPOINT = "models/convnext_tiny_320_inference.pth"

state: dict[str, Any] = {"classifier": None, "gradcam": None, "error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the checkpoint once at startup, not per request."""
    path = os.environ.get(CHECKPOINT_ENV, DEFAULT_CHECKPOINT)
    try:
        started = time.perf_counter()
        clf = ChestXrayClassifier(path)
        state["classifier"] = clf
        state["gradcam"] = GradCAM(clf)
        print(f"[startup] loaded {path} in {time.perf_counter() - started:.1f}s")
    except Exception as exc:  # keep the API up so /health can explain why
        state["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[startup] FAILED to load checkpoint: {state['error']}", file=sys.stderr)
    yield
    state.clear()


app = FastAPI(
    title="Chest X-Ray Multi-Label Classifier",
    description=(
        "14-pathology chest radiograph classification (ConvNeXt-Tiny 320) "
        "with Grad-CAM explanations. Research prototype - not validated for "
        "clinical use."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local demo; tighten before any non-local deployment
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


def _require_model() -> ChestXrayClassifier:
    if state.get("classifier") is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model unavailable: {state.get('error') or 'not loaded'}. "
                f"Set {CHECKPOINT_ENV} to the checkpoint path and restart."
            ),
        )
    return state["classifier"]


def _require_gradcam() -> GradCAM:
    _require_model()
    return state["gradcam"]


async def _read_upload(file: UploadFile) -> bytes:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
        )
    return data


@app.get("/health", response_model=HealthResponse)
def health() -> dict[str, Any]:
    clf = state.get("classifier")
    return {
        "status": "ok" if clf else "degraded",
        "model_loaded": clf is not None,
        "error": state.get("error"),
        "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
    }


@app.get("/model", response_model=ModelInfoResponse)
def model_info() -> dict[str, Any]:
    clf = _require_model()
    return {
        "model_version": MODEL_VERSION,
        "architecture": "convnext_tiny",
        "image_size": 320,
        "target_layer": TARGET_LAYER_NAME,
        "labels": LABELS,
        "thresholds": clf.thresholds,
        "checkpoint_epoch": clf.source_epoch,
        "val_macro_auc": clf.val_macro_auc,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    positives_only: bool = False,
) -> dict[str, Any]:
    """Classify one chest radiograph.

    Returns all 14 probabilities with their per-class thresholds by default;
    pass positives_only=true to return only findings above threshold.
    """
    clf = _require_model()
    data = await _read_upload(file)

    try:
        started = time.perf_counter()
        result = await run_in_threadpool(clf.predict, data, positives_only)
        result["inference_ms"] = round((time.perf_counter() - started) * 1000, 1)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    result["filename"] = file.filename
    return result


@app.post("/gradcam", response_class=Response, responses={200: {"content": {"image/png": {}}}})
async def gradcam(
    file: UploadFile = File(...),
    pathology: str = Form(...),
) -> Response:
    """Grad-CAM overlay for one pathology, returned as a PNG image.

    The frontend already has probability/threshold from /predict; this
    endpoint's only job is the heatmap itself, so it returns raw image
    bytes rather than JSON wrapping a base64 string.
    """
    gc = _require_gradcam()
    if pathology not in LABELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown pathology {pathology!r}. Must be one of: {LABELS}",
        )
    data = await _read_upload(file)

    try:
        explanation = await run_in_threadpool(gc.explain, data, pathology)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Grad-CAM failed: {exc}") from exc

    buf = BytesIO()
    explanation.overlay_image.save(buf, format="PNG")
    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={
            "X-Pathology": explanation.pathology,
            "X-Probability": str(explanation.probability),
            "X-Threshold": str(explanation.threshold),
            "X-Above-Threshold": str(explanation.above_threshold),
            "X-Target-Layer": explanation.target_layer,
            "X-Explain-Note": EXPLAIN_NOTE,
        },
    )
