# ─────────────────────────────────────────────────────────────
# DPWH Flood Correlation Study — Multiple Linear Regression
# ─────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

# ── 1. Load Data ──────────────────────────────────────────────
df = pd.read_csv("dpwh_flood_correlation_by_district.csv")
print(f"Shape: {df.shape}")
print(df.isnull().sum())

# ── 2. Define Features and Targets ───────────────────────────
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

# ── 3. Correlation Matrix ─────────────────────────────────────
corr_vars = features + ["total_area_flooded_sq_km", "pop_exposed"]
corr_matrix = df[corr_vars].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, square=True)
plt.title("Pearson Correlation Matrix — Selected Variables")
plt.tight_layout()
plt.savefig("correlation_matrix.png", dpi=150)
plt.show()

# Pearson (already done)
corr_pearson = df[corr_vars].corr(method="pearson")

# Spearman
corr_spearman = df[corr_vars].corr(method="spearman")

# Plot Spearman
plt.figure(figsize=(11, 9))
sns.heatmap(corr_spearman, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, square=True, annot_kws={"size": 9})
plt.title("Spearman Correlation Matrix — DPWH Projects vs Flood Outcomes")
plt.tight_layout()
plt.savefig("correlation_matrix_spearman.png", dpi=150, bbox_inches='tight')
plt.show()

# ── 4. MODEL 1: Total Area Flooded ───────────────────────────
X_train1, X_test1, y_train1, y_test1 = train_test_split(
    X, y1, test_size=0.2, random_state=42
)

scaler1 = StandardScaler()
X_train1_sc = scaler1.fit_transform(X_train1)
X_test1_sc  = scaler1.transform(X_test1)

model1 = LinearRegression()
model1.fit(X_train1_sc, y_train1)

y_pred1 = model1.predict(X_test1_sc)
r2_1    = r2_score(y_test1, y_pred1)
rmse_1  = np.sqrt(mean_squared_error(y_test1, y_pred1))

cv1 = cross_val_score(LinearRegression(), X_train1_sc,
                      y_train1, cv=5, scoring="r2")

print("\n── Model 1: Total Area Flooded ──")
print(f"R²        : {r2_1:.4f}")
print(f"RMSE      : {rmse_1:.4f} sq km")
print(f"CV R² (5-fold): {cv1.mean():.3f} ± {cv1.std():.3f}")

coef_df1 = pd.DataFrame({
    "Feature": features,
    "Coefficient": model1.coef_
}).sort_values("Coefficient")
print(coef_df1.to_string(index=False))

# Statsmodels — p-values for Model 1
X_sm = sm.add_constant(X)
ols1 = sm.OLS(y1, X_sm).fit()
print(ols1.summary())

# ── 5. MODEL 2: Population Exposed ───────────────────────────
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X, y2, test_size=0.2, random_state=42
)

scaler2 = StandardScaler()
X_train2_sc = scaler2.fit_transform(X_train2)
X_test2_sc  = scaler2.transform(X_test2)

model2 = LinearRegression()
model2.fit(X_train2_sc, y_train2)

y_pred2 = model2.predict(X_test2_sc)
r2_2    = r2_score(y_test2, y_pred2)
rmse_2  = np.sqrt(mean_squared_error(y_test2, y_pred2))

cv2 = cross_val_score(LinearRegression(), X_train2_sc,
                      y_train2, cv=5, scoring="r2")

print("\n── Model 2: Population Exposed ──")
print(f"R²        : {r2_2:.4f}")
print(f"RMSE      : {rmse_2:,.0f} people")
print(f"CV R² (5-fold): {cv2.mean():.3f} ± {cv2.std():.3f}")

coef_df2 = pd.DataFrame({
    "Feature": features,
    "Coefficient": model2.coef_
}).sort_values("Coefficient")
print(coef_df2.to_string(index=False))

# Statsmodels — p-values for Model 2
ols2 = sm.OLS(y2, X_sm).fit()
print(ols2.summary())

# ── Actual vs Predicted — Model 1 ────────────────────────────
plt.figure(figsize=(7, 5))
plt.scatter(y_test1, y_pred1, alpha=0.7, color="#534AB7", edgecolors="white")
plt.plot([y_test1.min(), y_test1.max()],
         [y_test1.min(), y_test1.max()], "r--", label="Perfect Fit")
plt.xlabel("Actual Flood Area (sq km)")
plt.ylabel("Predicted Flood Area (sq km)")
plt.title("Model 1: Actual vs Predicted — Total Area Flooded")
plt.legend()
plt.tight_layout()
plt.savefig("model1_actual_vs_predicted.png", dpi=150)
plt.show()

# ── Actual vs Predicted — Model 2 ────────────────────────────
plt.figure(figsize=(7, 5))
plt.scatter(y_test2, y_pred2, alpha=0.7, color="#1D9E75", edgecolors="white")
plt.plot([y_test2.min(), y_test2.max()],
         [y_test2.min(), y_test2.max()], "r--", label="Perfect Fit")
plt.xlabel("Actual Population Exposed")
plt.ylabel("Predicted Population Exposed")
plt.title("Model 2: Actual vs Predicted — Population Exposed")
plt.legend()
plt.tight_layout()
plt.savefig("model2_actual_vs_predicted.png", dpi=150)
plt.show()

# ── Feature Importance Comparison ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, coef_df, title, color in zip(
    axes,
    [coef_df1, coef_df2],
    ["Standardized Coefficients — Flood Area",
     "Standardized Coefficients — Pop Exposed"],
    ["#534AB7", "#1D9E75"]
):
    ax.barh(coef_df["Feature"], coef_df["Coefficient"],
            color=color, edgecolor="white")
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Coefficient Value")

plt.suptitle("Feature Importance: DPWH Project Types vs Flood Outcomes",
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("feature_importance_comparison.png", dpi=150)
plt.show()

# ── Residuals Plot — Model 1 ──────────────────────────────────
residuals1 = y_test1 - y_pred1
plt.figure(figsize=(7, 4))
plt.scatter(y_pred1, residuals1, alpha=0.7, color="#534AB7")
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Predicted Flood Area (sq km)")
plt.ylabel("Residuals")
plt.title("Model 1: Residuals Plot")
plt.tight_layout()
plt.savefig("model1_residuals.png", dpi=150)
plt.show()