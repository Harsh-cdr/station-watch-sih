"""
Station Watch — FastAPI backend.

The same app serves:
  GET  /          -> dashboard
  GET  /health    -> health check
  GET  /stream    -> five scripted demo readings
  POST /predict   -> one custom sensor reading
  GET  /docs      -> interactive API documentation

The backend loads the supplied Isolation Forest model artifact when possible.
If the artifact cannot be loaded, it retrains the deterministic demo model.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd
import shap
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from shap_explainer_demo import (
    FEATURES,
    compute_confidence,
    generate_normal_data,
    get_demo_readings,
    train_model,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("station-watch")

BASE_DIR = Path(__file__).resolve().parent
DASHBOARD_PATH = BASE_DIR / "dashboard.html"
MODEL_PATH = BASE_DIR / "isolation_forest_edge_model.pkl"

SCORE_THRESHOLD = 0.05


def load_model():
    """
    Load the committed model artifact.

    If the artifact was generated with an incompatible local library version,
    fall back to the deterministic training routine so the demo can still run.
    """
    if MODEL_PATH.exists():
        try:
            model = joblib.load(MODEL_PATH)
            logger.info("Loaded model artifact: %s", MODEL_PATH.name)
            return model
        except Exception as exc:
            logger.warning(
                "Could not load %s (%s). Retraining the deterministic demo model.",
                MODEL_PATH.name,
                exc,
            )

    logger.info("Training deterministic demo model...")
    return train_model(generate_normal_data(n_samples=2000))


_model = load_model()
_explainer = shap.TreeExplainer(_model)
_normal_data = generate_normal_data(n_samples=2000)
_training_scores = _model.decision_function(_normal_data)
_SCORE_MIN = float(_training_scores.min())
_SCORE_MAX = float(_training_scores.max())


app = FastAPI(
    title="Station Watch — AWS Sensor Anomaly Detector API",
    version="1.0.0",
)


# The dashboard is served by this same FastAPI process, so production use does
# not require a separately hosted frontend or broad cross-origin access.
# Keeping localhost in the allow-list also supports local development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class Reading(BaseModel):
    temperature: float = Field(..., description="Temperature in °C")
    pressure: float = Field(..., description="Pressure in hPa")
    humidity: float = Field(..., description="Relative humidity in %")


def analyze_reading(row: pd.DataFrame) -> dict:
    """Run one reading through the model + SHAP and return JSON-safe values."""
    row = row[FEATURES].astype(float)

    score = float(_model.decision_function(row)[0])
    shap_values = _explainer.shap_values(row)
    shap_values = shap_values[0]

    status = "anomaly" if score < SCORE_THRESHOLD else "normal"
    confidence = compute_confidence(
        score,
        SCORE_THRESHOLD,
        _SCORE_MIN,
        _SCORE_MAX,
    )

    top_feature = None
    top_value = None

    if status == "anomaly":
        contributions = sorted(
            zip(FEATURES, shap_values),
            key=lambda item: abs(float(item[1])),
            reverse=True,
        )
        top_feature, top_value = contributions[0]

    return {
        "temperature": float(row.iloc[0]["temperature"]),
        "pressure": float(row.iloc[0]["pressure"]),
        "humidity": float(row.iloc[0]["humidity"]),
        "status": status,
        "score": round(score, 3),
        "confidence": round(confidence, 1),
        "top_driver": top_feature,
        "top_driver_contribution": (
            round(float(top_value), 3) if top_value is not None else None
        ),
    }


@app.get("/", include_in_schema=False)
def dashboard():
    """Serve the dashboard from the same origin as the API."""
    return FileResponse(DASHBOARD_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stream")
def stream():
    """Return the five scripted demo readings in their fixed order."""
    readings = get_demo_readings()
    return {
        "readings": [
            analyze_reading(readings.iloc[[i]])
            for i in range(len(readings))
        ]
    }


@app.post("/predict")
def predict(reading: Reading):
    """Analyze a custom sensor reading submitted by the dashboard."""
    row = pd.DataFrame(
        [reading.model_dump()],
        columns=FEATURES,
    )
    return analyze_reading(row)
