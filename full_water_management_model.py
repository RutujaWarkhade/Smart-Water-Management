"""
Smart Water Management System
1. Supervised Learning → Consumption Prediction
2. Unsupervised Learning → Consumer Segmentation
"""

import pandas as pd
import numpy as np
import os, joblib, json, warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─── CREATE FOLDERS ───────────────────────────────────────────────────────────
os.makedirs("D:/Water_Management/models", exist_ok=True)
os.makedirs("D:/Water_Management/outputs", exist_ok=True)

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
df = pd.read_csv("D:/Water_Management/water_consumption.csv")
print(f"Loaded {len(df):,} records")

# ─── ENCODING ────────────────────────────────────────────────────────────────
le = LabelEncoder()
df["income_enc"] = le.fit_transform(df["income_level"])
df["zone_enc"]   = le.fit_transform(df["zone"])

# ─────────────────────────────────────────────────────────────────────────────
# PART 1: SUPERVISED LEARNING (PREDICTION)
# ─────────────────────────────────────────────────────────────────────────────

FEATURES = [
    "temperature_c", "rainfall_mm", "household_size",
    "hist_avg_3d", "is_weekend", "month"
]

TARGET = "daily_usage_liters"

df[FEATURES] = df[FEATURES].fillna(df[FEATURES].median())

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

models = {
    "Linear": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=120, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=150, random_state=42)
}

best_model, best_r2 = None, -999

print("\n--- SUPERVISED LEARNING RESULTS ---")

for name, model in models.items():

    if name == "Linear":
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)
    else:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)

    print(f"{name} → R2: {r2:.4f}, MAE: {mae:.2f}")

    if r2 > best_r2:
        best_model = model
        best_r2 = r2
        best_name = name

print(f"\n Best Model: {best_name} (R2={best_r2:.4f})")

# SAVE MODEL
joblib.dump(best_model, "D:/Water_Management/models/best_model.pkl")
joblib.dump(scaler, "D:/Water_Management/models/scaler.pkl")

# ─────────────────────────────────────────────────────────────────────────────
#  PART 2: UNSUPERVISED LEARNING (CLUSTERING)
# ─────────────────────────────────────────────────────────────────────────────

print("\n--- UNSUPERVISED LEARNING (CLUSTERING) ---")

cluster_features = [
    "daily_usage_liters",
    "household_size",
    "hist_avg_3d",
    "month"
]

X_cluster = df[cluster_features]

scaler_c = StandardScaler()
X_scaled = scaler_c.fit_transform(X_cluster)

# KMeans clustering (3 groups)
kmeans = KMeans(n_clusters=3, random_state=42)
df["cluster"] = kmeans.fit_predict(X_scaled)

# Label clusters meaningfully
cluster_means = df.groupby("cluster")["daily_usage_liters"].mean()

cluster_labels = {}
sorted_clusters = cluster_means.sort_values()

cluster_labels[sorted_clusters.index[0]] = "Low Consumption"
cluster_labels[sorted_clusters.index[1]] = "Seasonal Users"
cluster_labels[sorted_clusters.index[2]] = "High Consumption"

df["segment"] = df["cluster"].map(cluster_labels)

print("\nCluster Summary:")
print(df.groupby("segment")["daily_usage_liters"].mean())

# SAVE CLUSTER MODEL
joblib.dump(kmeans, "D:/Water_Management/models/kmeans.pkl")
joblib.dump(scaler_c, "D:/Water_Management/models/cluster_scaler.pkl")

# SAVE OUTPUT DATA
df.to_csv("D:/Water_Management/outputs/segmented_consumers.csv", index=False)

print("\n Full system trained & saved successfully!")