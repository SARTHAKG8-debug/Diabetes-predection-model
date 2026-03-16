"""
FastAPI application for DDoS attack detection.

Run locally:
    uvicorn app:app --reload

Swagger docs:
    http://127.0.0.1:8000/docs
"""

import os
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

# ── Load saved artifacts ──────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

try:
    model = joblib.load(os.path.join(MODEL_DIR, "ddos_model.joblib"))
    le_protocol = joblib.load(os.path.join(MODEL_DIR, "label_encoder.joblib"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.joblib"))
except FileNotFoundError:
    raise RuntimeError(
        "Model files not found!  Run  python train_model.py  first."
    )

LABEL_MAP = {0: "Benign", 1: "Malicious"}

# ── Pydantic schemas ──────────────────────────────────────────

class RawPredictRequest(BaseModel):
    """Accept human-readable features (Protocol as string like 'UDP', 'TCP', etc.)"""
    dt: float = Field(..., description="Timestamp delta")
    switch: float = Field(..., description="Switch ID")
    pktcount: float = Field(..., description="Packet count")
    bytecount: float = Field(..., description="Byte count")
    dur: float = Field(..., description="Duration (seconds)")
    dur_nsec: float = Field(..., description="Duration (nanoseconds)")
    tot_dur: float = Field(..., description="Total duration")
    flows: float = Field(..., description="Number of flows")
    packetins: float = Field(..., description="Packet-ins")
    pktperflow: float = Field(..., description="Packets per flow")
    byteperflow: float = Field(..., description="Bytes per flow")
    pktrate: float = Field(..., description="Packet rate")
    Pairflow: float = Field(..., description="Pair flow")
    Protocol: str = Field(..., description="Protocol name, e.g. 'UDP', 'TCP', 'ICMP'")
    port_no: float = Field(..., description="Port number")
    tx_bytes: float = Field(..., description="Transmitted bytes")
    rx_bytes: float = Field(..., description="Received bytes")
    tx_kbps: float = Field(..., description="Transmit kbps")
    rx_kbps: Optional[float] = Field(None, description="Receive kbps (nullable)")
    tot_kbps: Optional[float] = Field(None, description="Total kbps (nullable)")

class NumericPredictRequest(BaseModel):
    """Accept a list of already-encoded numeric feature values (in the same order as training)."""
    features: list[float] = Field(
        ..., description="List of numeric feature values in training order"
    )

class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    label: int

# ── App ───────────────────────────────────────────────────────
app = FastAPI(
    title="DDoS Attack Detection API",
    description="Predict whether network traffic is **Benign** or a **DDoS attack** using a trained Random Forest model.",
    version="1.0.0",
)


@app.get("/", tags=["Health"])
def health_check():
    """Simple health-check endpoint."""
    return {
        "status": "ok",
        "model": "RandomForestClassifier",
        "features": feature_names,
    }


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict_numeric(req: NumericPredictRequest):
    """
    Predict from a pre-encoded numeric feature vector.
    The list must match the training feature order exactly.
    """
    if len(req.features) != len(feature_names):
        raise HTTPException(
            status_code=422,
            detail=f"Expected {len(feature_names)} features, got {len(req.features)}. "
                   f"Required order: {feature_names}",
        )

    X = np.array(req.features).reshape(1, -1)
    pred = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][pred])

    return PredictResponse(
        prediction=LABEL_MAP.get(pred, "Unknown"),
        confidence=round(proba, 4),
        label=pred,
    )


@app.post("/predict_raw", response_model=PredictResponse, tags=["Prediction"])
def predict_raw(req: RawPredictRequest):
    """
    Predict from raw/human-readable features.
    Protocol should be a string (e.g. 'UDP', 'TCP').
    Missing rx_kbps / tot_kbps will be filled with 0.
    """
    data = req.model_dump()

    # Encode protocol
    protocol_str = data.pop("Protocol")
    try:
        protocol_encoded = le_protocol.transform([protocol_str])[0]
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown Protocol '{protocol_str}'. "
                   f"Known values: {list(le_protocol.classes_)}",
        )
    data["Protocol"] = float(protocol_encoded)

    # Handle nullables
    if data.get("rx_kbps") is None:
        data["rx_kbps"] = 0.0
    if data.get("tot_kbps") is None:
        data["tot_kbps"] = 0.0

    # Build feature vector in the correct order
    try:
        row = [data[f] for f in feature_names]
    except KeyError as e:
        raise HTTPException(status_code=422, detail=f"Missing feature: {e}")

    X = np.array(row).reshape(1, -1)
    pred = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][pred])

    return PredictResponse(
        prediction=LABEL_MAP.get(pred, "Unknown"),
        confidence=round(proba, 4),
        label=pred,
    )
