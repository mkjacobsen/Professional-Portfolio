"""
FastAPI prediction service for retail demand forecasting.

Endpoints:
- POST /predict: Accept a single prediction request, return a demand forecast
- GET /health: Return service health, current model version, and champion MAE
- POST /monitor: Accept a batch of recent observations, run drift detection, return a MonitoringReport serialised as a dict

The champion model is loaded at application startup from the ArtifactStore.  The store path
is configured via the ``ARTIFACT_STORE_PATH`` environment variable (Defaults to ``./artifacts``)
"""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .monitoring import DriftDetector
from .store import ArtifactStore

# Schemas

class PredictionRequest(BaseModel):
    """Input schema for single demand-forecast request"""

    store_id: int
    product_id: int
    date: str
    price: float
    is_weekend: bool
    is_holiday: bool
    lag_1_demand: float
    lag_7_demand: float
    rolling_7_mean: float

class PredictionResponse(BaseModel):
    """Output schema for single demand-forecast response"""

    store_id: int
    product_id: int
    date: str
    predicted_demand: float
    model_version: int

class MonitorRequest(BaseModel):
    """Schema for batch of production observations sent to /monitor"""

    observations: list[dict]
    include_labels: bool = False

# Application

app = FastAPI(
    title="Demand Forecasting Service",
    description=(
        "End-to-end MLOps demo: real-time demand forecasting with "
        "production drift monitoring"
    ),
    version = "0.1.0",
)

# Module state populated at startup
_model = None
_model_version: int = 0
_champion_mae: Optional[float] = None
_reference_features: Optional[pd.DataFrame] = None
_artifact_store: Optional[ArtifactStore] = None

@app.on_event("startup")
def _load_champion() -> None:
    """Load champion model from ArtifactStore at startup"""
    global _model, _model_version, _champion_mae, _artifact_store
    store_path = os.environ.get("ARTIFACT_STORE_PATH", "./artifacts")
    _artifact_store = ArtifactStore(store_path)

    result = _artifact_store.load_champion()
    if result is not None:
        _model, metrics = result
        _model_version = metrics.get("version", 0)
        _champion_mae = metrics.get("mae")

# Route Handlers

@app.get("/health")
def health() -> dict:
    """
    Return service health data

    Returns:
    - dict with keys [status, model_loaded, model_version, and champion_mae]
    """

    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "model_version": _model_version,
        "champion_mae": _champion_mae,
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Generate demand forecast for single store/product/date

    Request must include pre-computed lag features (lag_1_demand, lag_7_demand, rolling_7_mean) because the
    service does not have access to the full historical time series at inference time
    """

    if _model is None:
        raise HTTPException(status_code = 503, detail="No champion model loaded")

    # Build feature row - must match FEATURE_COLUMNS order
    from .features import FEATURE_COLUMNS
    import math

    date = pd.to_datetime(request.date)
    dow = date.dayofweek
    month = date.month

    row = {
        "price": request.price,
        "date_of_week_sin": math.sin(2 * math.pi * dow / 7),
        "date_of_week_cos": math.cos(2 * math.pi * dow / 7),
        "month_sin": math.sin(2 * math.pi * (month - 1) / 12),
        "month_cos": math.cos(2 * math.pi * (month - 1) / 12),
        "is_weekend": int(request.is_weekend),
        "is_holiday": int(request.is_holiday),
        "lag_1_demand": request.lag_1_demand,
        "lag_7_demand": request.lag_7_demand,
        "lag_14_demand": request.lag_7_demand, # fallback if lag_14 not provided
        "rolling_7_mean": request.rolling_7_mean,
        "rolling_14_mean": request.rolling_7_mean, #fallback
        "store_size_encoded": request.store_id % 3,
        "product_category_encoded": request.product_id % 3,
    }

    # Keep columns used for training
    feature_names = getattr(_model, "feature_names_in_", None) or FEATURE_COLUMNS
    X = pd.DataFrame([{k: row.get(k, 0.0) for k in feature_names}])

    prediction = float(_model.predict(X)[0])
    prediction = max(0.0, round(prediction, 2))

    return PredictionResponse(
        store_id = request.store_id,
        product_id = request.product_id,
        date = request.date,
        predicted_Demand = prediction,
        model_version = _model_version,
    )

@app.post("/monitor")
def monitor(request: MonitorRequest) -> dict:
    """
    Run drift detection on a batch of recent production observations

    Parameters:
    - request.observations: List of feature dicts - same schema as training feature DataFrame
    - request_include_labels: True indicates a demand key in each observation dict and include label drift in report

    Returns:
    - dict: serialised :class:`monitoring.MonitoringReport`
    """

    if _reference_features is None:
        raise HTTPException(
            status_code = 503,
            detail = (
                "Reference feature snapshot not loaded. "
                "Run the training pipeline first."
            ),
        )

    current_df = pd.DataFrame(request.observations)
    current_labels = None
    if request.include_labels and "demand" in current_df.columns:
        current_labels = current_df["demand"].to_numpy(dtype=float)

    detector = DriftDetector(_reference_features)
    report =detector.generate_report(current_df, current_labels)
    return asdict(report)