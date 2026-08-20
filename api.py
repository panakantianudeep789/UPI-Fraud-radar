"""
UPI Fraud Radar — FastAPI Real-Time Scoring Service
====================================================
Run with:  uvicorn api:app --host 0.0.0.0 --port 8000 --reload
Docs at:   http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import List
import numpy as np
import pandas as pd
import joblib
import shap
import time

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class TransactionRequest(BaseModel):
    amount: float = Field(..., gt=0, description="Transaction amount in INR", examples=[9500.0])
    avg_user_amount: float = Field(..., gt=0, description="User's historical average transaction amount (INR)", examples=[800.0])
    hour: int = Field(..., ge=0, le=23, description="Hour of the day (0-23)", examples=[3])
    is_new_device: int = Field(..., ge=0, le=1, description="1 = unrecognized device, 0 = known device", examples=[1])
    is_new_city: int = Field(..., ge=0, le=1, description="1 = transaction from new city, 0 = home city", examples=[1])
    minutes_since_last_txn: float = Field(..., ge=0, description="Minutes elapsed since the user's last transaction", examples=[2.0])

    model_config = {
        "json_schema_extra": {
            "example": {
                "amount": 9500.0,
                "avg_user_amount": 800.0,
                "hour": 3,
                "is_new_device": 1,
                "is_new_city": 1,
                "minutes_since_last_txn": 2.0,
            }
        }
    }


class ShapContribution(BaseModel):
    feature: str
    shap_value: float
    direction: str  # "toward_fraud" | "toward_legit"


class ScoreResponse(BaseModel):
    fraud_score: float = Field(..., description="Fraud probability (0.0 - 1.0)")
    risk_level: str = Field(..., description="LOW | MEDIUM | HIGH")
    recommendation: str = Field(..., description="Suggested action for this transaction")
    shap_contributions: List[ShapContribution] = Field(..., description="Feature-level SHAP explanations (sorted by |impact|)")
    latency_ms: float = Field(..., description="End-to-end scoring latency in milliseconds")


class BatchRequest(BaseModel):
    transactions: List[TransactionRequest]


class BatchResponse(BaseModel):
    results: List[ScoreResponse]
    total_latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model: str
    feature_count: int
    uptime_seconds: float


# ---------------------------------------------------------------------------
# App lifecycle - load model once at startup
# ---------------------------------------------------------------------------

_state: dict = {}
_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML artifacts once at startup; release on shutdown."""
    print("Loading fraud model ...")
    _state["model"] = joblib.load("fraud_model.pkl")
    _state["feature_cols"] = joblib.load("feature_cols.pkl")
    _state["explainer"] = shap.TreeExplainer(_state["model"])
    print(f"Model ready - {len(_state['feature_cols'])} features")
    yield
    _state.clear()
    print("Model unloaded")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="UPI Fraud Radar API",
    description=(
        "Real-time UPI transaction fraud scoring powered by XGBoost + SHAP. "
        "Each response includes a fraud probability, risk classification, "
        "human-readable recommendation, and per-feature SHAP explanations "
        "required for RBI audit compliance."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Feature engineering (mirrors train_model.py exactly)
# ---------------------------------------------------------------------------

def _engineer_features(req: TransactionRequest) -> pd.DataFrame:
    amount = req.amount
    avg = req.avg_user_amount
    hour = req.hour
    mins_since = req.minutes_since_last_txn
    return pd.DataFrame([{
        "amount": amount,
        "log_amount": np.log1p(amount),
        "amount_to_avg_ratio": amount / (avg + 1e-6),
        "hour": hour,
        "is_odd_hour": 1 if (hour < 5 or hour > 22) else 0,
        "is_new_device": req.is_new_device,
        "is_new_city": req.is_new_city,
        "minutes_since_last_txn": mins_since,
        "is_rapid_txn": 1 if mins_since < 5 else 0,
        "is_round_or_micro": 1 if (amount < 15 or amount % 100 == 0) else 0,
    }])


def _score_one(req: TransactionRequest) -> ScoreResponse:
    """Core scoring logic - called by both /score and /score/batch."""
    t0 = time.perf_counter()

    model = _state["model"]
    feature_cols = _state["feature_cols"]
    explainer = _state["explainer"]

    X = _engineer_features(req)[feature_cols]
    fraud_score = float(model.predict_proba(X)[0, 1])
    shap_vals = explainer.shap_values(X)[0]

    # Risk level & recommendation
    if fraud_score > 0.5:
        risk_level = "HIGH"
        recommendation = "Block transaction - require step-up authentication"
    elif fraud_score > 0.15:
        risk_level = "MEDIUM"
        recommendation = "Soft challenge - OTP or PIN re-entry"
    else:
        risk_level = "LOW"
        recommendation = "Allow transaction"

    # SHAP contributions (sorted by absolute impact)
    shap_contribs = [
        ShapContribution(
            feature=feat,
            shap_value=round(float(val), 4),
            direction="toward_fraud" if val > 0 else "toward_legit",
        )
        for feat, val in sorted(zip(feature_cols, shap_vals), key=lambda x: -abs(x[1]))
    ]

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    return ScoreResponse(
        fraud_score=round(fraud_score, 4),
        risk_level=risk_level,
        recommendation=recommendation,
        shap_contributions=shap_contribs,
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["Info"])
def root():
    return {
        "service": "UPI Fraud Radar API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "score": "POST /score",
        "batch": "POST /score/batch",
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
def health():
    """Liveness check - returns model info and server uptime."""
    return HealthResponse(
        status="ok",
        model="XGBoost v3 (fraud_model.pkl)",
        feature_count=len(_state.get("feature_cols", [])),
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.post("/score", response_model=ScoreResponse, tags=["Scoring"])
def score(req: TransactionRequest):
    """
    Score a single UPI transaction.

    Returns fraud probability (0-1), risk level, recommended action,
    and SHAP feature contributions explaining the decision.
    Target latency: <100ms.
    """
    try:
        return _score_one(req)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/score/batch", response_model=BatchResponse, tags=["Scoring"])
def score_batch(req: BatchRequest):
    """
    Score a batch of UPI transactions in a single request.

    Useful for bulk re-scoring or back-testing.
    """
    if not req.transactions:
        raise HTTPException(status_code=400, detail="transactions list cannot be empty")
    try:
        t0 = time.perf_counter()
        results = [_score_one(txn) for txn in req.transactions]
        total_ms = round((time.perf_counter() - t0) * 1000, 2)
        return BatchResponse(results=results, total_latency_ms=total_ms)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
