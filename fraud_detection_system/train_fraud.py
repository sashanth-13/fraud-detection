"""
=============================================================
 Phase 1 (Enhanced): Fraud Detection – XGBoost + SMOTE
=============================================================
 New features added over v1:
   hour_of_day            – fraudsters strike 1-4 AM
   is_weekend             – weekend fraud patterns
   is_international       – foreign IP flag
   distance_from_home_km  – geospatial anomaly
   time_since_last_txn_hrs– rapid-sequence attacks
   is_round_amount        – suspicious round sums
   failed_attempts_24h    – failed auth before success
"""

import os, sqlite3, joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, ConfusionMatrixDisplay,
)
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ─── 1. Load ─────────────────────────────────────────────
DB_PATH = os.path.join("data", "fraud_detection.db")
print("=" * 60)
print("  FRAUD DETECTION  (XGBoost + SMOTE) – Enhanced v2")
print("=" * 60)

print("\n[1/7] Loading transactions...")
conn = sqlite3.connect(DB_PATH)
df   = pd.read_sql("SELECT * FROM transactions", conn)
conn.close()

print(f"      Rows   : {len(df):,}")
print(f"      Fraud  : {df['is_fraud'].sum():,}  ({df['is_fraud'].mean()*100:.2f} %)")

# ─── 2. Feature Engineering ──────────────────────────────
print("\n[2/7] Engineering features...")

DROP_COLS = ["transaction_id", "user_id", "timestamp", "ip_address"]
df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

# Encode categorical features
CAT_COLS = ["merchant_category", "transaction_method", "device_type"]
le = LabelEncoder()
for col in CAT_COLS:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))

# Boolean columns → int
BOOL_COLS = ["is_weekend", "is_international", "is_round_amount"]
for col in BOOL_COLS:
    if col in df.columns:
        df[col] = df[col].astype(int)

X = df.drop(columns=["is_fraud"])
y = df["is_fraud"].astype(int)

print(f"      Total features : {len(X.columns)}")
print(f"      Features       : {list(X.columns)}")
print(f"      Class balance  : {dict(y.value_counts())}")

# ─── 3. Train/Test Split ─────────────────────────────────
print("\n[3/7] Train/Test split (80/20, stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"      Train : {len(X_train):,}   Test : {len(X_test):,}")

# ─── 4. SMOTE ────────────────────────────────────────────
print("\n[4/7] Applying SMOTE...")
print(f"      Before → {dict(y_train.value_counts())}")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"      After  → {dict(pd.Series(y_train_res).value_counts())}")

# ─── 5. Train XGBoost ────────────────────────────────────
print("\n[5/7] Training XGBClassifier (200 trees)...")
model = XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42, n_jobs=-1,
)
model.fit(X_train_res, y_train_res)
print("      Done ✓")

# ─── 6. Evaluate ─────────────────────────────────────────
print("\n[6/7] Evaluating...")
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n" + "─" * 55)
print("  CLASSIFICATION REPORT")
print("─" * 55)
print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))
roc = roc_auc_score(y_test, y_proba)
print(f"  ROC-AUC : {roc:.4f}")
print("─" * 55)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Legit", "Fraud"])
fig, ax = plt.subplots(figsize=(5, 5))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("XGBoost Fraud – Confusion Matrix")
plt.tight_layout()
fig.savefig("fraud_confusion_matrix.png", dpi=120)
plt.close(fig)
print("  Saved: fraud_confusion_matrix.png")

# Feature Importance
feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values()
fig2, ax2 = plt.subplots(figsize=(9, 6))
colors = ["#ef4444" if "fraud" in f or f in ["distance_from_home_km","failed_attempts_24h",
          "is_international","is_round_amount","hour_of_day","time_since_last_txn_hrs"]
          else "#6366f1" for f in feat_imp.index]
feat_imp.plot(kind="barh", ax=ax2, color=colors)
ax2.set_title("XGBoost Feature Importance (Enhanced v2)")
ax2.set_xlabel("Importance Score")
plt.tight_layout()
fig2.savefig("fraud_feature_importance.png", dpi=120)
plt.close(fig2)
print("  Saved: fraud_feature_importance.png")

# ─── 7. Save ─────────────────────────────────────────────
print("\n[7/7] Saving model...")
joblib.dump(model, "xgboost_fraud.pkl")
print("      xgboost_fraud.pkl  ✓")

print("\n" + "=" * 60)
print("  TRAINING COMPLETE!")
print("=" * 60)
