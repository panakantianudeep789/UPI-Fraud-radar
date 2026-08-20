import pandas as pd
import joblib
import shap

model = joblib.load("fraud_model.pkl")
feature_cols = joblib.load("feature_cols.pkl")
df = pd.read_csv("upi_transactions.csv")

df["amount_to_avg_ratio"] = df["amount"] / (df["avg_user_amount"] + 1e-6)
df["is_odd_hour"] = df["hour"].apply(lambda h: 1 if (h < 5 or h > 22) else 0)
df["is_rapid_txn"] = (df["minutes_since_last_txn"] < 5).astype(int)
df["is_round_or_micro"] = df["amount"].apply(lambda a: 1 if (a < 15 or a % 100 == 0) else 0)
df["log_amount"] = pd.Series(df["amount"]).apply(lambda x: __import__("numpy").log1p(x))

X = df[feature_cols]

explainer = shap.TreeExplainer(model)

# Explain a few fraud examples
fraud_sample = df[df["is_fraud"] == 1].head(3)
sample_X = fraud_sample[feature_cols]
shap_values = explainer.shap_values(sample_X)

print("=== Sample fraud explanations ===\n")
for i, (idx, row) in enumerate(fraud_sample.iterrows()):
    print(f"Transaction {row['txn_id']} | fraud_type={row['fraud_type']} | amount=Rs.{row['amount']}")
    contribs = sorted(zip(feature_cols, shap_values[i]), key=lambda x: -abs(x[1]))
    for feat, val in contribs[:4]:
        direction = "pushes toward FRAUD" if val > 0 else "pushes toward LEGIT"
        print(f"   {feat:25s} = {row[feat]:<10} | SHAP={val:+.3f} ({direction})")
    print()

# Global feature importance
mean_abs_shap = pd.Series(
    abs(shap.TreeExplainer(model).shap_values(X.sample(2000, random_state=1))).mean(axis=0),
    index=feature_cols
).sort_values(ascending=False)
print("=== Global feature importance (mean |SHAP|) ===")
print(mean_abs_shap.round(4))
