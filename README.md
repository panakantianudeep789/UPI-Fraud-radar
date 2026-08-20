# 🛡️ UPI Fraud Radar — Real-Time Transaction Fraud Detection

A real-time fraud detection system for UPI-style digital payments, built to mirror
how risk teams at Paytm / Razorpay / Navi / PhonePe / MobiKwik approach transaction
fraud: supervised classification + unsupervised anomaly detection + explainable
per-transaction decisions.

## Problem
UPI processes 15+ billion transactions/month in India. Even a 0.1% fraud rate
represents massive absolute losses. Fraud is also **extremely imbalanced**
(~1% of transactions), so naive accuracy-optimized models fail in production.

## What this project demonstrates
- **Realistic synthetic UPI dataset** (60k transactions) with 4 injected fraud
  patterns: account takeover, money-mule bursts, geographic impossibility, and
  card-testing/probing fraud
- **Feature engineering** for fraud signals: transaction velocity, amount-vs-history
  ratio, new-device/new-city flags, odd-hour flags
- **Class-imbalance handling** via `scale_pos_weight` (avoids synthetic-oversampling
  artifacts from SMOTE)
- **Model comparison**: Logistic Regression (baseline) → Random Forest → XGBoost →
  Isolation Forest (unsupervised anomaly detector)
- **Explainability**: SHAP values explain *why* each transaction was flagged —
  critical for RBI audit/compliance requirements, not just a black-box score
- **Live dashboard**: Streamlit app simulating a real-time transaction feed with
  fraud alerts, plus a manual transaction scorer and model performance view

## Results (held-out test set, 15k transactions)

| Model | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.59 | 1.00 | 0.74 | 0.976 |
| Random Forest | 0.86 | 0.98 | 0.91 | 0.988 |
| **XGBoost (chosen)** | **0.91** | **0.95** | **0.93** | **0.989** |
| Isolation Forest (unsupervised) | 0.71 | 0.67 | 0.69 | — |

**Why PR-AUC, not accuracy:** with 1.2% fraud rate, predicting "not fraud" always
gives ~99% accuracy while catching zero fraud. PR-AUC and recall are the metrics
that matter for imbalanced fraud detection.

**Top fraud signals (SHAP global importance):** `is_new_device` >> `is_rapid_txn`
> `amount_to_avg_ratio` > `minutes_since_last_txn`. This matches real-world fraud
literature: device fingerprinting and transaction velocity are the strongest
account-takeover signals.

## Project structure
```
generate_data.py   - synthetic UPI transaction generator (fraud pattern injection)
train_model.py     - feature engineering + model training + evaluation
explain_model.py   - SHAP explainability analysis
app.py             - Streamlit real-time dashboard
fraud_model.pkl    - trained XGBoost model
```



