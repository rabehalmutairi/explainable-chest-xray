"""Response models for the chest X-ray API.

Field names follow the project's reporting vocabulary: "probability" and
"above_threshold", never "accuracy" for a single prediction. "no_finding" is
derived (none of the 14 crossed threshold), never a 15th model output.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionItem(BaseModel):
    pathology: str
    probability: float = Field(..., ge=0.0, le=1.0, description="Model probability (sigmoid output).")
    threshold: float = Field(..., description="Validation-tuned decision threshold for this class.")
    above_threshold: bool = Field(..., description="Whether probability crossed this class's threshold.")


class PredictResponse(BaseModel):
    predictions: list[PredictionItem]
    positive_findings: list[str] = Field(..., description="Pathologies above their threshold.")
    no_finding: bool = Field(..., description="True only when no pathology crossed its threshold.")
    model_version: str
    checkpoint_epoch: int | None = None
    val_macro_auc: float | None = None
    inference_ms: float
    filename: str | None = None


class ModelInfoResponse(BaseModel):
    model_version: str
    architecture: str
    image_size: int
    target_layer: str
    labels: list[str]
    thresholds: dict[str, float]
    checkpoint_epoch: int | None = None
    val_macro_auc: float | None = None
    note: str = "Research prototype. Not intended, validated, or suitable for clinical use."


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    error: str | None = None
    max_upload_mb: int
