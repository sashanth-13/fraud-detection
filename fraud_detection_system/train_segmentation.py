"""
=============================================================
 Phase 2 (Enhanced): Customer Segmentation – K-Means RFM+
=============================================================
 New features added over v1:
   avg_transaction_amount   – value per visit
   online_purchase_ratio    – digital engagement
   night_owl_ratio          – behavioral timing signal
   unique_merchants_count   – breadth of spending
   churn_risk_score         – pre-computed composite risk
   preferred_payment_method – encoded payment behaviour
"""

import os, sqlite3, joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# ─── 1. Load ─────────────────────────────────────────────
DB_PATH = os.path.join("data", "fraud_detection.db")

print("=" * 60)
print("  CUSTOMER SEGMENTATION  (K-Means RFM+) – Enhanced v2")
print("=" * 60)

print("\n[1/6] Loading users...")
conn = sqlite3.connect(DB_PATH)
df   = pd.read_sql("SELECT * FROM users", conn)
conn.close()

print(f"      Users loaded : {len(df):,}")

# ─── 2. Feature Selection & Encoding ─────────────────────
print("\n[2/6] Preparing features...")

# Encode preferred_payment_method as ordinal integer
le = LabelEncoder()
if "preferred_payment_method" in df.columns:
    df["preferred_payment_method_enc"] = le.fit_transform(
        df["preferred_payment_method"].astype(str)
    )

# Drop identifier and string columns
DROP_COLS = ["user_id", "top_category_preference", "preferred_payment_method",
             "home_lat", "home_lon"]
df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

FEATURES = [
    # Original RFM
    "avg_monthly_spend",
    "recency_days",
    "frequency_per_month",
    "total_monetary_value",
    # New behavioural features
    "avg_transaction_amount",
    "online_purchase_ratio",
    "night_owl_ratio",
    "unique_merchants_count",
    "churn_risk_score",
    "preferred_payment_method_enc",
]
FEATURES = [f for f in FEATURES if f in df.columns]

print(f"      Features ({len(FEATURES)}) : {FEATURES}")

X = df[FEATURES].fillna(0)

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("      StandardScaler applied ✓")

# ─── 3. Elbow + Silhouette ───────────────────────────────
print("\n[3/6] Finding optimal K (1→10)...")
inertias    = []
silhouettes = []

for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    if k > 1:
        sil = silhouette_score(X_scaled, km.labels_)
        silhouettes.append(sil)
        print(f"      K={k}  Inertia={km.inertia_:,.1f}  Silhouette={sil:.4f}")
    else:
        print(f"      K=1  Inertia={km.inertia_:,.1f}  Silhouette=N/A")

# Plots
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(range(1, 11), inertias, "o-", color="#6366f1", lw=2, ms=7)
axes[0].axvline(4, color="red", ls="--", alpha=0.7, label="Chosen K=4")
axes[0].set_title("Elbow – Inertia vs K")
axes[0].set_xlabel("K"); axes[0].set_ylabel("Inertia")
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(range(2, 11), silhouettes, "s-", color="#22c55e", lw=2, ms=7)
axes[1].axvline(4, color="red", ls="--", alpha=0.7, label="Chosen K=4")
axes[1].set_title("Silhouette Score vs K")
axes[1].set_xlabel("K"); axes[1].set_ylabel("Silhouette (higher=better)")
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
fig.savefig("segmentation_elbow.png", dpi=120)
plt.close(fig)
print("  Saved: segmentation_elbow.png")

# ─── 4. Train Final K-Means ──────────────────────────────
OPTIMAL_K = 4
print(f"\n[4/6] Training KMeans (K={OPTIMAL_K})...")
kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=20, max_iter=500)
kmeans.fit(X_scaled)
print(f"      Inertia = {kmeans.inertia_:,.2f}  ✓")

# ─── 5. Assign & Profile Clusters ────────────────────────
print("\n[5/6] Profiling segments...")
df_out = df.copy()
df_out["cluster"] = kmeans.labels_

profile = df_out.groupby("cluster")[FEATURES].mean().round(3)
print("\n  Cluster Profiles:")
print(profile.to_string())

# Rank clusters to auto-assign business meaning
rank = (
    profile["total_monetary_value"].rank() +
    profile["frequency_per_month"].rank() +
    profile["avg_transaction_amount"].rank() -
    profile["recency_days"].rank() -
    profile["churn_risk_score"].rank()
)
order = rank.sort_values(ascending=False).index.tolist()

LABELS = {
    order[0]: "[1] Champions  - High-Value, Frequent, Recent",
    order[1]: "[2] Loyal Mid-Tier  - Steady, Moderate Spend",
    order[2]: "[3] At-Risk  - Declining Engagement (Churn Alert)",
    order[3]: "[4] Dormant  - Inactive / Low-Value Customers",
}

df_out["segment"] = df_out["cluster"].map(LABELS)

print("\n  Segment Summary:")
print("─" * 60)
for cid in order:
    cnt = (df_out["cluster"] == cid).sum()
    pct = cnt / len(df_out) * 100
    print(f"  {LABELS[cid]}")
    avg_spend = profile.loc[cid, "total_monetary_value"]
    avg_churn = profile.loc[cid, "churn_risk_score"]
    print(f"       Users: {cnt:,} ({pct:.1f}%)  |  Avg Lifetime Value: ${avg_spend:,.0f}  |  Churn Risk: {avg_churn:.2f}")
print("─" * 60)

# PCA 2-D Scatter
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_scaled)
palette = ["#6366f1", "#22c55e", "#f59e0b", "#ef4444"]

fig2, ax = plt.subplots(figsize=(10, 7))
for i, cid in enumerate(order):
    mask = df_out["cluster"] == cid
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
               c=palette[i], alpha=0.35, s=12,
               label=LABELS[cid].split("-")[0].strip())
ax.set_title("Customer Segments – PCA 2D (Enhanced RFM+)")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
ax.legend(loc="upper right", fontsize=8, framealpha=0.6)
ax.grid(alpha=0.2)
plt.tight_layout()
fig2.savefig("segmentation_clusters.png", dpi=120)
plt.close(fig2)
print("  Saved: segmentation_clusters.png")

# Feature Contribution Bar
fig3, ax3 = plt.subplots(figsize=(11, 5))
centroid_df = pd.DataFrame(kmeans.cluster_centers_, columns=FEATURES)
centroid_df.index = [LABELS[i].split("-")[0].strip() for i in range(OPTIMAL_K)]
centroid_df.T.plot(kind="bar", ax=ax3, colormap="tab10", width=0.7)
ax3.set_title("Cluster Centroids per Feature (Scaled)")
ax3.set_xlabel("Feature"); ax3.set_ylabel("Standardised Value")
ax3.axhline(0, color="white", lw=0.8, alpha=0.4)
ax3.legend(loc="upper right", fontsize=8)
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
fig3.savefig("segmentation_centroids.png", dpi=120)
plt.close(fig3)
print("  Saved: segmentation_centroids.png")

# ─── 6. Save ─────────────────────────────────────────────
print("\n[6/6] Saving...")
joblib.dump(kmeans, "kmeans_segmentation.pkl")
joblib.dump(scaler, "kmeans_scaler.pkl")
print("      kmeans_segmentation.pkl  ✓")
print("      kmeans_scaler.pkl        ✓")

print("\n" + "=" * 60)
print("  SEGMENTATION TRAINING COMPLETE!")
print("=" * 60)
