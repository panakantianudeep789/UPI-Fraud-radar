import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import time
import random
import json
try:
    import requests as _requests
    _requests_ok = True
except ImportError:
    _requests_ok = False

st.set_page_config(page_title="UPI Fraud Radar", layout="wide", page_icon="🛡️")

@st.cache_resource
def load_model():
    model = joblib.load("fraud_model.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    explainer = shap.TreeExplainer(model)
    return model, feature_cols, explainer

@st.cache_data
def load_data():
    return pd.read_csv("upi_transactions.csv")

model, feature_cols, explainer = load_model()
df = load_data()

st.title("🛡️ UPI Fraud Radar — Real-Time Transaction Risk Scoring")
st.caption("XGBoost fraud classifier · 91% precision / 95% recall on held-out data · SHAP-explained decisions")

def engineer_features(row):
    amount = row["amount"]
    avg = row["avg_user_amount"]
    hour = row["hour"]
    mins_since = row["minutes_since_last_txn"]
    return pd.DataFrame([{
        "amount": amount,
        "log_amount": np.log1p(amount),
        "amount_to_avg_ratio": amount / (avg + 1e-6),
        "hour": hour,
        "is_odd_hour": 1 if (hour < 5 or hour > 22) else 0,
        "is_new_device": row["is_new_device"],
        "is_new_city": row["is_new_city"],
        "minutes_since_last_txn": mins_since,
        "is_rapid_txn": 1 if mins_since < 5 else 0,
        "is_round_or_micro": 1 if (amount < 15 or amount % 100 == 0) else 0,
    }])

tab1, tab2, tab3, tab4 = st.tabs(["🔴 Live Feed Simulation", "🧮 Score a Custom Transaction", "📊 Model Performance", "🔌 Live API Demo"])

# ---------------- TAB 1: Live feed ----------------
with tab1:
    st.subheader("Simulated live UPI transaction stream")
    n_txns = st.slider("Number of transactions to stream", 5, 40, 15)
    if st.button("▶️ Start stream", type="primary"):
        sample = df.sample(n_txns, random_state=random.randint(0, 9999)).reset_index(drop=True)
        placeholder = st.empty()
        alert_box = st.container()
        alerts = []
        for i, row in sample.iterrows():
            X_row = engineer_features(row)[feature_cols]
            proba = model.predict_proba(X_row)[0, 1]
            risk = "🔴 HIGH RISK" if proba > 0.5 else ("🟡 WATCH" if proba > 0.15 else "🟢 OK")
            with placeholder.container():
                c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 2, 2])
                c1.metric("Txn", row["txn_id"])
                c2.metric("Amount", f"₹{row['amount']:.0f}")
                c3.metric("City", row["city"])
                c4.metric("Fraud Score", f"{proba:.1%}")
                c5.metric("Status", risk)
            if proba > 0.5:
                shap_vals = explainer.shap_values(X_row)[0]
                top_reasons = sorted(zip(feature_cols, shap_vals), key=lambda x: -x[1])[:3]
                reason_str = ", ".join([f"{f}" for f, v in top_reasons if v > 0])
                alerts.append(f"🚨 **{row['txn_id']}** (₹{row['amount']:.0f}) flagged — top signals: *{reason_str}*")
            time.sleep(0.15)
        if alerts:
            with alert_box:
                st.error(f"### {len(alerts)} fraud alerts raised")
                for a in alerts:
                    st.markdown(a)
        else:
            st.success("No high-risk transactions in this stream.")

# ---------------- TAB 2: Custom scoring ----------------
with tab2:
    st.subheader("Score a transaction manually")
    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("Amount (₹)", min_value=1.0, value=500.0)
        avg_amount = st.number_input("User's average transaction amount (₹)", min_value=1.0, value=450.0)
        hour = st.slider("Hour of day (0-23)", 0, 23, 14)
    with col2:
        is_new_device = st.checkbox("New/unrecognized device")
        is_new_city = st.checkbox("Transaction from a new city")
        mins_since_last = st.number_input("Minutes since user's last transaction", min_value=0.0, value=120.0)

    if st.button("Score this transaction", type="primary"):
        row = {"amount": amount, "avg_user_amount": avg_amount, "hour": hour,
               "is_new_device": int(is_new_device), "is_new_city": int(is_new_city),
               "minutes_since_last_txn": mins_since_last}
        X_row = engineer_features(row)[feature_cols]
        proba = model.predict_proba(X_row)[0, 1]
        shap_vals = explainer.shap_values(X_row)[0]

        st.metric("Fraud Probability", f"{proba:.1%}")
        if proba > 0.5:
            st.error("🔴 HIGH RISK — recommend blocking / step-up authentication")
        elif proba > 0.15:
            st.warning("🟡 MEDIUM RISK — recommend soft challenge (OTP/PIN re-entry)")
        else:
            st.success("🟢 LOW RISK — allow")

        st.markdown("**Why this score (SHAP contributions):**")
        contribs = sorted(zip(feature_cols, shap_vals), key=lambda x: -abs(x[1]))
        for feat, val in contribs:
            direction = "⬆️ toward fraud" if val > 0 else "⬇️ toward legit"
            st.write(f"- `{feat}`: {val:+.3f} ({direction})")

# ---------------- TAB 3: Model performance ----------------
with tab3:
    st.subheader("Model comparison")
    comp = pd.read_csv("model_comparison.csv", index_col=0)
    st.dataframe(comp.style.highlight_max(axis=0, color="lightgreen"), use_container_width=True)
    st.markdown("""
    **Why XGBoost was chosen:** best precision/recall balance and PR-AUC (0.989) on a highly imbalanced
    dataset (1.2% fraud rate), using `scale_pos_weight` instead of oversampling to avoid synthetic-noise
    artifacts. PR-AUC is reported instead of plain accuracy because accuracy is misleading on imbalanced
    fraud data (predicting "no fraud" always would give ~99% accuracy).
    """)
    st.subheader("Global feature importance")
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    st.bar_chart(importances)

# ---------------- TAB 4: Live API Demo ----------------
with tab4:
    API_URL = "http://localhost:8000"

    st.subheader("🔌 Live FastAPI Scoring Demo")
    st.markdown(
        "This tab calls the **FastAPI `/score` endpoint** in real-time. "
        "Make sure the API server is running: `uvicorn api:app --port 8000`"
    )

    # --- API health status ---
    col_status, col_docs = st.columns([3, 1])
    with col_status:
        if _requests_ok:
            try:
                health_resp = _requests.get(f"{API_URL}/health", timeout=2)
                h = health_resp.json()
                st.success(
                    f"✅ API online · model: `{h['model']}` · "
                    f"{h['feature_count']} features · uptime: {h['uptime_seconds']}s"
                )
            except Exception:
                st.error("❌ API is **offline**. Start it with: `uvicorn api:app --port 8000`")
        else:
            st.warning("`requests` not installed — run `pip install requests`")
    with col_docs:
        st.link_button("📖 Swagger Docs", f"{API_URL}/docs")

    st.divider()

    # --- Transaction input form ---
    st.markdown("#### Build a Transaction Request")
    c1, c2, c3 = st.columns(3)
    with c1:
        api_amount = st.number_input("Amount (INR)", min_value=1.0, value=9500.0, key="api_amount")
        api_avg = st.number_input("Avg user amount (INR)", min_value=1.0, value=800.0, key="api_avg")
    with c2:
        api_hour = st.slider("Hour of day", 0, 23, 3, key="api_hour")
        api_mins = st.number_input("Mins since last txn", min_value=0.0, value=2.0, key="api_mins")
    with c3:
        api_new_device = st.checkbox("New / unrecognized device", value=True, key="api_dev")
        api_new_city = st.checkbox("New city", value=True, key="api_city")

    payload = {
        "amount": api_amount,
        "avg_user_amount": api_avg,
        "hour": api_hour,
        "is_new_device": int(api_new_device),
        "is_new_city": int(api_new_city),
        "minutes_since_last_txn": api_mins,
    }

    # Live JSON preview of the request
    st.markdown("**Request body preview (JSON):**")
    st.code(json.dumps(payload, indent=2), language="json")

    if st.button("🚀 POST /score", type="primary", key="api_btn"):
        if not _requests_ok:
            st.error("Install `requests`: `pip install requests`")
        else:
            try:
                t0 = time.perf_counter()
                resp = _requests.post(
                    f"{API_URL}/score",
                    json=payload,
                    timeout=10,
                )
                round_trip_ms = round((time.perf_counter() - t0) * 1000, 1)

                if resp.status_code == 200:
                    data = resp.json()

                    # --- Key metrics ---
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Fraud Score", f"{data['fraud_score']:.1%}")
                    m2.metric("Risk Level", data["risk_level"])
                    m3.metric("Model latency", f"{data['latency_ms']} ms")
                    m4.metric("Round-trip", f"{round_trip_ms} ms")

                    # Recommendation banner
                    if data["risk_level"] == "HIGH":
                        st.error(f"🔴 {data['recommendation']}")
                    elif data["risk_level"] == "MEDIUM":
                        st.warning(f"🟡 {data['recommendation']}")
                    else:
                        st.success(f"🟢 {data['recommendation']}")

                    # --- Request / Response side-by-side ---
                    st.markdown("#### Raw API Interaction")
                    left, right = st.columns(2)
                    with left:
                        st.markdown("**Request**")
                        st.code(
                            f"POST {API_URL}/score\n"
                            f"Content-Type: application/json\n\n"
                            + json.dumps(payload, indent=2),
                            language="json",
                        )
                    with right:
                        st.markdown("**Response**")
                        st.code(json.dumps(data, indent=2), language="json")

                    # --- SHAP waterfall (text) ---
                    st.markdown("#### SHAP Feature Contributions")
                    for c in data["shap_contributions"]:
                        direction = "⬆️ fraud" if c["direction"] == "toward_fraud" else "⬇️ legit"
                        bar_len = min(int(abs(c["shap_value"]) * 30), 30)
                        bar = ("█" * bar_len).ljust(30)
                        st.write(f"`{c['feature']:25s}` {bar} `{c['shap_value']:+.4f}` ({direction})")

                else:
                    st.error(f"API returned {resp.status_code}: {resp.text}")

            except _requests.exceptions.ConnectionError:
                st.error("Cannot reach API. Run: `uvicorn api:app --port 8000`")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
