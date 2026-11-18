"""FastAPI service for bike sharing demand prediction.

POST /predict with hourly weather/calendar features returns the predicted
rental count. GET /health and GET /model-info expose service metadata.
"""

from __future__ import annotations

import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Bike Sharing Demand API",
    description="Predict hourly bike rental demand (UCI Bike Sharing dataset)",
    version="1.0.0",
)

MODEL_PATH = os.path.join("models", "best_model.pkl")
FEATURES = [
    "season", "yr", "mnth", "hr", "holiday", "weekday", "workingday",
    "weathersit", "temp", "atemp", "hum", "windspeed",
    "hr_sin", "hr_cos", "dow_sin", "dow_cos", "mnth_sin", "mnth_cos",
    "rush_hour", "temp_hum",
]

_model = None


def get_model():
    """Load the model once, lazily, and cache it for the process lifetime."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"{MODEL_PATH} not found. Train the model first: python src/train.py"
            )
        _model = joblib.load(MODEL_PATH)
    return _model


class PredictionInput(BaseModel):
    """One hourly record. Defaults match a normal Wednesday afternoon hour."""

    season: int = Field(ge=1, le=4, description="1=winter ... 4=fall")
    yr: int = Field(ge=0, le=1, description="0=2011, 1=2012")
    mnth: int = Field(ge=1, le=12)
    hr: int = Field(ge=0, le=23)
    holiday: int = Field(ge=0, le=1)
    weekday: int = Field(ge=0, le=6)
    workingday: int = Field(ge=0, le=1)
    weathersit: int = Field(ge=1, le=4, description="1=clear ... 4=heavy rain")
    temp: float = Field(ge=0, le=1, description="normalized Celsius / 41")
    atemp: float = Field(ge=0, le=1, description="normalized feels-like / 50")
    hum: float = Field(ge=0, le=1, description="normalized humidity")
    windspeed: float = Field(ge=0, le=1, description="normalized wind speed")


def build_features(payload: PredictionInput) -> pd.DataFrame:
    """Apply the same cyclical/interaction engineering as training."""
    d = payload.model_dump()
    d["hr_sin"] = np.sin(2 * np.pi * d["hr"] / 24)
    d["hr_cos"] = np.cos(2 * np.pi * d["hr"] / 24)
    d["dow_sin"] = np.sin(2 * np.pi * d["weekday"] / 7)
    d["dow_cos"] = np.cos(2 * np.pi * d["weekday"] / 7)
    d["mnth_sin"] = np.sin(2 * np.pi * d["mnth"] / 12)
    d["mnth_cos"] = np.cos(2 * np.pi * d["mnth"] / 12)
    d["rush_hour"] = int(d["hr"] in [7, 8, 17, 18, 19])
    d["temp_hum"] = d["temp"] * d["hum"]
    return pd.DataFrame([{f: d[f] for f in FEATURES}])


class PredictionOutput(BaseModel):
    predicted_cnt: float
    model: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "bike-sharing-demand"}


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    try:
        get_model()
        model_name = open(os.path.join("models", "best_model_name.txt")).read().strip()
    except FileNotFoundError as err:
        raise HTTPException(status_code=503, detail="Model not trained yet") from err
    return {"model": model_name, "features": FEATURES}


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: PredictionInput) -> PredictionOutput:
    try:
        model = get_model()
    except FileNotFoundError as err:
        raise HTTPException(status_code=503, detail="Model not trained yet") from err
    X = build_features(payload)
    pred = float(model.predict(X)[0])
    return PredictionOutput(predicted_cnt=max(0.0, pred), model="best_model")
