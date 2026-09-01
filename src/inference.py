"""Reusable inference for the ConvNeXt-Tiny 320 chest X-ray model.

Single source of truth for architecture, checkpoint loading, preprocessing,
label order, thresholds and result formatting. FastAPI, Streamlit and any
notebook should import from here rather than re-implementing the pipeline.

The model emits LOGITS. Sigmoid is applied here, never inside the network.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, UnidentifiedImageError
from torchvision.models import convnext_tiny

# Output order is bound to the checkpoint. Changing it silently breaks
# every saved model and every threshold file.
LABELS: list[str] = [
    "Atelectasis",
    "Consolidation",
    "Infiltration",
    "Pneumothorax",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Effusion",
    "Pneumonia",
    "Pleural_Thickening",
    "Cardiomegaly",
    "Nodule",
    "Mass",
    "Hernia",
]

# Validation-optimized, from results/convnext_thresholds.json. Used only when
# the checkpoint does not carry its own thresholds.
DEFAULT_THRESHOLDS: dict[str, float] = {
    "Atelectasis": 0.34,
    "Consolidation": 0.12,
    "Infiltration": 0.17,
    "Pneumothorax": 0.29,
    "Edema": 0.18,
    "Emphysema": 0.30,
    "Fibrosis": 0.18,
    "Effusion": 0.22,
    "Pneumonia": 0.05,
    "Pleural_Thickening": 0.23,
    "Cardiomegaly": 0.67,
    "Nodule": 0.20,
    "Mass": 0.19,
    "Hernia": 0.50,
}

MODEL_VERSION = "ConvNeXt-Tiny-320-BCE"

IMAGE_SIZE = 320
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
ACCEPTED_FORMATS = {"PNG", "JPEG", "BMP", "TIFF"}


class InvalidImageError(ValueError):
    """Raised when an upload is not a usable image."""


def build_model(num_classes: int = len(LABELS)) -> nn.Module:
    """ConvNeXt-Tiny with the project's 14-output head. Weights are not loaded."""
    model = convnext_tiny(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    return model


def build_transform() -> T.Compose:
    """Exactly the eval transform used in training. No crop, no flip."""
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_image(data: bytes | str | Path | Image.Image) -> Image.Image:
    """Accept raw bytes, a path or a PIL image; always return RGB."""
    if isinstance(data, Image.Image):
        return data.convert("RGB")
    try:
        if isinstance(data, (str, Path)):
            image = Image.open(data)
        else:
            if len(data) > MAX_UPLOAD_BYTES:
                raise InvalidImageError(
                    f"Image exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit."
                )
            image = Image.open(io.BytesIO(data))
        image.load()
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError(f"Not a readable image file: {exc}") from exc

    if image.format and image.format.upper() not in ACCEPTED_FORMATS:
        raise InvalidImageError(
            f"Unsupported format {image.format}. Accepted: {sorted(ACCEPTED_FORMATS)}"
        )
    return image.convert("RGB")


class ChestXrayClassifier:
    """Loads the checkpoint once and serves predictions.

    Construct a single instance at process start and reuse it; loading is the
    expensive step, a forward pass on CPU is roughly 0.3-1.0 s.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device | None = None,
        thresholds: dict[str, float] | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.device = torch.device(device) if device else torch.device("cpu")
        self.transform = build_transform()

        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        state, meta = self._unwrap(ckpt)

        self.labels: list[str] = [str(x) for x in meta.get("labels", LABELS)]
        if self.labels != LABELS:
            raise ValueError(
                f"Checkpoint label order does not match project order.\n"
                f"  checkpoint: {self.labels}\n  expected:   {LABELS}"
            )

        out_dim = state["classifier.2.weight"].shape[0]
        if out_dim != len(LABELS):
            raise ValueError(
                f"Checkpoint has {out_dim} outputs; project standard is {len(LABELS)}. "
                "'No Finding' is the absence of the 14, not a class."
            )

        self.model = build_model()
        result = self.model.load_state_dict(state, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise ValueError(
                f"Checkpoint did not load cleanly. "
                f"missing={result.missing_keys} unexpected={result.unexpected_keys}"
            )
        self.model.to(self.device).eval()

        self.thresholds = dict(
            thresholds or meta.get("thresholds") or DEFAULT_THRESHOLDS
        )
        missing = set(LABELS) - set(self.thresholds)
        if missing:
            raise ValueError(f"No threshold for: {sorted(missing)}")

        self.source_epoch = meta.get("source_epoch", meta.get("epoch"))
        self.val_macro_auc = meta.get("val_macro_auc")

    @staticmethod
    def _unwrap(ckpt: Any) -> tuple[dict, dict]:
        """Handle both the training checkpoint (dict with model_state_dict plus
        optimizer state) and a bare state_dict."""
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            meta = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
            return ckpt["model_state_dict"], meta
        if isinstance(ckpt, dict) and "classifier.2.weight" in ckpt:
            return ckpt, {}
        raise ValueError(
            "Unrecognized checkpoint layout; expected 'model_state_dict' or a bare state_dict."
        )

    @torch.no_grad()
    def predict(
        self,
        image: bytes | str | Path | Image.Image,
        positives_only: bool = False,
    ) -> dict[str, Any]:
        """Run one image. Returns JSON-serializable plain Python types only.

        Field names follow the project's reporting vocabulary: this is a
        model probability crossing a per-class threshold, not "accuracy",
        and 'above_threshold' is the model's finding, not a diagnosis.
        """
        pil = load_image(image)
        tensor = self.transform(pil).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = torch.sigmoid(logits)[0].cpu().tolist()

        predictions = []
        for label, prob in zip(LABELS, probs):
            threshold = float(self.thresholds[label])
            predictions.append({
                "pathology": label,
                "probability": round(float(prob), 4),
                "threshold": threshold,
                "above_threshold": bool(prob >= threshold),
            })

        positives = [p["pathology"] for p in predictions if p["above_threshold"]]
        predictions.sort(key=lambda p: p["probability"], reverse=True)

        return {
            "predictions": [p for p in predictions if p["above_threshold"]]
            if positives_only
            else predictions,
            "positive_findings": positives,
            "no_finding": not positives,
            "model_version": MODEL_VERSION,
            "checkpoint_epoch": self.source_epoch,
            "val_macro_auc": self.val_macro_auc,
        }

    def predict_batch(
        self, images: Iterable[bytes | str | Path | Image.Image]
    ) -> list[dict[str, Any]]:
        return [self.predict(img) for img in images]


def load_thresholds(path: str | Path) -> dict[str, float]:
    """Read a thresholds JSON and validate it covers all 14 classes."""
    with open(path) as fh:
        data = json.load(fh)
    missing = set(LABELS) - set(data)
    if missing:
        raise ValueError(f"Threshold file missing: {sorted(missing)}")
    return {k: float(v) for k, v in data.items()}
