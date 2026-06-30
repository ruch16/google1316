import numpy as np, pandas as pd, time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

np.random.seed(42)
N = 500_000

df = pd.DataFrame({
    "amount": np.random.exponential(150, N),
    "txn_hour": np.random.randint(0, 24, N),
    "merchant_risk": np.random.beta(2, 8, N),
    "velocity_1h": np.random.poisson(2, N),
    "is_foreign": np.random.binomial(1, 0.08, N),
    "device_trust": np.random.beta(5, 2, N),
})
risk_score = (
    0.004*df.amount + 0.6*df.merchant_risk + 0.15*df.velocity_1h
    + 1.2*df.is_foreign - 1.0*df.device_trust + np.random.normal(0,0.5,N)
)
df["is_fraud"] = (risk_score > risk_score.quantile(0.97)).astype(int)

t0 = time.time()
X = df.drop(columns="is_fraud"); y = df.is_fraud
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
clf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=1)
clf.fit(X_train, y_train)
preds = clf.predict_proba(X_test)[:,1]
cpu_time = time.time() - t0
auc = roc_auc_score(y_test, preds)

# scoring-only latency (inference on a single batch, mimics real-time decisioning)
t1 = time.time()
_ = clf.predict_proba(X_test)
cpu_infer_time = time.time() - t1

print(f"Rows: {N}")
print(f"CPU full pipeline (train+infer) time: {cpu_time:.2f}s")
print(f"CPU inference-only time on {len(X_test)} rows: {cpu_infer_time:.3f}s")
print(f"AUC: {auc:.4f}")
print(f"Throughput (CPU inference): {len(X_test)/cpu_infer_time:,.0f} txns/sec")

import json
json.dump({
    "rows": N, "cpu_pipeline_s": cpu_time, "cpu_infer_s": cpu_infer_time,
    "auc": auc, "cpu_throughput": len(X_test)/cpu_infer_time
}, open("bench_results.json","w"), indent=2)
