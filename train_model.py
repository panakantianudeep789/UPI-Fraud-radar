import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              roc_auc_score, average_precision_score,
                              confusion_matrix, classification_report)

df = pd.read_csv("upi_transactions.csv")

# ---- Feature engineering ----
df["amount_to_avg_ratio"] = df["amount"] / (df["avg_user_amount"] + 1e-6)
df["is_odd_hour"] = df["hour"].apply(lambda h: 1 if (h < 5 or h > 22) else 0)
df["is_rapid_txn"] = (df["minutes_since_last_txn"] < 5).astype(int)
df["is_round_or_micro"] = df["amount"].apply(lambda a: 1 if (a < 15 or a % 100 == 0) else 0)
df["log_amount"] = np.log1p(df["amount"])

feature_cols = [
    "amount", "log_amount", "amount_to_avg_ratio", "hour", "is_odd_hour",
    "is_new_device", "is_new_city", "minutes_since_last_txn", "is_rapid_txn",
    "is_round_or_micro",
]

X = df[feature_cols]
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

results = {}

# ---- Baseline: Logistic Regression ----
lr = LogisticRegression(max_iter=1000, class_weight="balanced")
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_proba = lr.predict_proba(X_test)[:, 1]
results["Logistic Regression"] = {
    "precision": precision_score(y_test, lr_pred),
    "recall": recall_score(y_test, lr_pred),
    "f1": f1_score(y_test, lr_pred),
    "roc_auc": roc_auc_score(y_test, lr_proba),
    "pr_auc": average_precision_score(y_test, lr_proba),
}

# ---- Random Forest ----
rf = RandomForestClassifier(n_estimators=300, max_depth=10, class_weight="balanced",
                             random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]
results["Random Forest"] = {
    "precision": precision_score(y_test, rf_pred),
    "recall": recall_score(y_test, rf_pred),
    "f1": f1_score(y_test, rf_pred),
    "roc_auc": roc_auc_score(y_test, rf_proba),
    "pr_auc": average_precision_score(y_test, rf_proba),
}

# ---- XGBoost with scale_pos_weight for imbalance ----
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb = XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.08,
    scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
    random_state=42, n_jobs=-1
)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
xgb_proba = xgb.predict_proba(X_test)[:, 1]
results["XGBoost"] = {
    "precision": precision_score(y_test, xgb_pred),
    "recall": recall_score(y_test, xgb_pred),
    "f1": f1_score(y_test, xgb_pred),
    "roc_auc": roc_auc_score(y_test, xgb_proba),
    "pr_auc": average_precision_score(y_test, xgb_proba),
}

# ---- Isolation Forest (unsupervised anomaly detector, no labels used) ----
iso = IsolationForest(contamination=0.012, random_state=42, n_jobs=-1)
iso.fit(X_train)
iso_pred_raw = iso.predict(X_test)  # -1 = anomaly, 1 = normal
iso_pred = (iso_pred_raw == -1).astype(int)
results["Isolation Forest (unsupervised)"] = {
    "precision": precision_score(y_test, iso_pred),
    "recall": recall_score(y_test, iso_pred),
    "f1": f1_score(y_test, iso_pred),
    "roc_auc": np.nan,
    "pr_auc": np.nan,
}

print("\n=== MODEL COMPARISON ===")
res_df = pd.DataFrame(results).T.round(4)
print(res_df.to_string())

print("\n=== XGBoost Confusion Matrix (best model) ===")
cm = confusion_matrix(y_test, xgb_pred)
print(pd.DataFrame(cm, index=["Actual: Legit", "Actual: Fraud"],
                    columns=["Pred: Legit", "Pred: Fraud"]))

print("\n=== XGBoost Classification Report ===")
print(classification_report(y_test, xgb_pred, target_names=["Legit", "Fraud"]))

# Save best model (XGBoost) + feature list
joblib.dump(xgb, "fraud_model.pkl")
joblib.dump(feature_cols, "feature_cols.pkl")
res_df.to_csv("model_comparison.csv")
print("\nSaved: fraud_model.pkl, feature_cols.pkl, model_comparison.csv")
