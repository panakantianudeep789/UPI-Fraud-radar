# UPI Fraud Radar — REST API

Real-time fraud scoring REST API built with FastAPI + XGBoost + SHAP.
Every response includes a fraud probability, risk level, recommendation, and
per-feature SHAP explanations — meeting RBI audit/compliance requirements.

## Start the API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Interactive Swagger docs: http://localhost:8000/docs

---

## Endpoints

### `GET /health`
Liveness check.

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "ok",
  "model": "XGBoost v3 (fraud_model.pkl)",
  "feature_count": 10,
  "uptime_seconds": 42.3
}
```

---

### `POST /score`
Score a single transaction. Target latency: **<100ms**.

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 9500,
    "avg_user_amount": 800,
    "hour": 3,
    "is_new_device": 1,
    "is_new_city": 1,
    "minutes_since_last_txn": 2.0
  }'
```

Response:
```json
{
  "fraud_score": 0.9731,
  "risk_level": "HIGH",
  "recommendation": "Block transaction - require step-up authentication",
  "shap_contributions": [
    {"feature": "is_new_device",        "shap_value": 0.4821, "direction": "toward_fraud"},
    {"feature": "is_rapid_txn",         "shap_value": 0.3104, "direction": "toward_fraud"},
    {"feature": "amount_to_avg_ratio",  "shap_value": 0.2917, "direction": "toward_fraud"},
    {"feature": "is_odd_hour",          "shap_value": 0.1832, "direction": "toward_fraud"},
    {"feature": "is_new_city",          "shap_value": 0.1201, "direction": "toward_fraud"}
  ],
  "latency_ms": 8.43
}
```

---

### `POST /score/batch`
Score multiple transactions in one call.

```bash
curl -X POST http://localhost:8000/score/batch \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {"amount": 9500, "avg_user_amount": 800, "hour": 3,  "is_new_device": 1, "is_new_city": 1, "minutes_since_last_txn": 2},
      {"amount": 500,  "avg_user_amount": 450, "hour": 14, "is_new_device": 0, "is_new_city": 0, "minutes_since_last_txn": 240}
    ]
  }'
```

---

## Run Alongside the Dashboard

Open **two terminals**:

```bash
# Terminal 1 — API
uvicorn api:app --host 0.0.0.0 --port 8000

# Terminal 2 — Dashboard
streamlit run app.py
```

Then open the **🔌 Live API Demo** tab in the dashboard to see the two services
communicating in real time.

---

## Architecture

```
Streamlit dashboard (port 8501)
        │   HTTP POST /score
        ▼
FastAPI service (port 8000)
        │
        ├── load_model()  ← fraud_model.pkl  (loaded once at startup)
        ├── engineer_features()
        ├── XGBoost.predict_proba()
        └── shap.TreeExplainer()  → SHAP contributions in response
```
