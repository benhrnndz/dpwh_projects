import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Load Data ──────────────────────────────────────────────────────
df = pd.read_csv(r"d:\Users\Documents\PUP\IPT Project\dpwh_flood_correlation_by_district.csv")

# ── Define Features and Targets ────────────────────────────────────
features = [
    "n_projects_completed_before_flood",
    "n_flood_mitigation_completed",
    "n_drainage_completed",
    "n_slope_protection_completed",
    "n_revetment_completed",
    "n_dike_completed"
]

X  = df[features]
y1 = df["total_area_flooded_sq_km"]
y2 = df["pop_exposed"]

# ── Train/Test Split — Model 1 (Flood Area) ────────────────────────
X_train1, X_test1, y_train1, y_test1 = train_test_split(
    X, y1, test_size=0.2, random_state=42
)

# ── Train/Test Split — Model 2 (Pop Exposed) ──────────────────────
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X, y2, test_size=0.2, random_state=42
)

# ── Scale Features ─────────────────────────────────────────────────
scaler1 = StandardScaler()
X_train1_sc = scaler1.fit_transform(X_train1)
X_test1_sc  = scaler1.transform(X_test1)

scaler2 = StandardScaler()
X_train2_sc = scaler2.fit_transform(X_train2)
X_test2_sc  = scaler2.transform(X_test2)

# ── Confirm ────────────────────────────────────────────────────────
print("Train size:", X_train1.shape[0], "rows")
print("Test size :", X_test1.shape[0], "rows")
print("Features  :", list(X.columns))
print("Done — no errors!")