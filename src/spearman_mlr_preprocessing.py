"""
============================================================================
SULONG SA DALUYONG  |  Group 4
Mapping the Correlation Between Completed Public Works & Urban Flood
Resilience in NCR  --  PREPROCESSING for SPEARMAN + MLR
----------------------------------------------------------------------------
SKEWNESS CHECK RESULTS (run before this script):
  perc_total_area_flooded : +4.612  SKEWED  -> use Spearman, not Pearson
  pop_exposed             : +1.686  SKEWED  -> use Spearman, not Pearson
  project-type columns    : all < |1.0|  -> acceptable

FINAL 9-FIELD SET:
  SPATIAL : ncr_district
  CAUSE   : n_projects_completed_before_flood
  OUTCOME : perc_total_area_flooded
  OUTCOME : pop_exposed
  TYPE    : n_flood_mitigation_completed
  TYPE    : n_drainage_completed
  TYPE    : n_rehabilitation_completed
  TYPE    : n_slope_protection_completed
  TYPE    : n_revetment_completed

NOTE: No binning here. Spearman and MLR work on raw numbers.
      Columns are cleaned and standardized only.
============================================================================
"""
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False

SRC = "/mnt/user-data/uploads/dpwh_flood_correlation_by_district_csv.xlsx"
OUT = "/home/claude"

# 1. LOAD ------------------------------------------------------------------
df = pd.read_excel(SRC)
print(f"[load] raw shape: {df.shape}")

# 2. CLEAN -----------------------------------------------------------------
df = df.drop_duplicates().reset_index(drop=True)

district_map = {
    "NCR, City of Manila, First District": "Manila_1st",
    "NCR, Second District":               "NCR_2nd",
    "NCR, Third District":                "NCR_3rd",
    "NCR, Fourth District":               "NCR_4th",
}
df["ncr_district"] = df["ncr_district"].str.strip().replace(district_map)

# Keep only the 9 fields
keep = [
    "ncr_district",
    "n_projects_completed_before_flood",
    "perc_total_area_flooded",
    "pop_exposed",
    "n_flood_mitigation_completed",
    "n_drainage_completed",
    "n_rehabilitation_completed",
    "n_slope_protection_completed",
    "n_revetment_completed",
]
clean = df[keep].copy()

# Missing values (none here, but explicit)
clean = clean.dropna(subset=["perc_total_area_flooded"]).reset_index(drop=True)
num_cols = clean.select_dtypes(include="number").columns
clean[num_cols] = clean[num_cols].fillna(clean[num_cols].median())
print(f"[clean] shape: {clean.shape} | missing: {int(clean.isna().sum().sum())}")
print(f"[clean] districts: {clean['ncr_district'].value_counts().to_dict()}")

# Rename columns to readable names for the report
clean.columns = [
    "District",
    "Projects_Completed",
    "Flooding_Pct",
    "Pop_Exposed",
    "FloodMitigation_Done",
    "Drainage_Done",
    "Rehabilitation_Done",
    "SlopeProtection_Done",
    "Revetment_Done",
]

# 3. DESCRIPTIVE STATS -----------------------------------------------------
num = clean.select_dtypes(include="number")
desc = num.describe().round(2)
print(f"\n[stats] descriptive summary:")
print(desc)

# 4. SPEARMAN CORRELATION MATRIX -------------------------------------------
# Correlate each public-works column against each flood-outcome column
works_cols   = ["Projects_Completed", "FloodMitigation_Done", "Drainage_Done",
                "Rehabilitation_Done", "SlopeProtection_Done", "Revetment_Done"]
outcome_cols = ["Flooding_Pct", "Pop_Exposed"]

print("\n[spearman] Correlation vs flood outcomes (rho | p-value):")
spearman_rows = []
for w in works_cols:
    row = {"Variable": w}
    for o in outcome_cols:
        rho, p = stats.spearmanr(clean[w], clean[o])
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        print(f"  {w:30s} vs {o:15s}: rho={rho:+.3f}  p={p:.4f} {sig}")
        row[f"{o}_rho"] = round(rho, 3)
        row[f"{o}_p"]   = round(p, 4)
        row[f"{o}_sig"] = sig
    spearman_rows.append(row)
spearman_df = pd.DataFrame(spearman_rows)

# Full correlation matrix (all numeric columns)
full_corr = pd.DataFrame(index=works_cols, columns=outcome_cols)
for w in works_cols:
    for o in outcome_cols:
        rho, _ = stats.spearmanr(clean[w], clean[o])
        full_corr.loc[w, o] = round(rho, 3)

# 5. MLR -------------------------------------------------------------------
# Model 1: Flooding_Pct ~ all works columns
# Model 2: Pop_Exposed  ~ all works columns
X = clean[works_cols]

print("\n[mlr] Multiple Linear Regression results:")
mlr_results = []
for outcome in outcome_cols:
    y = clean[outcome]
    slope_dict = {}
    # Manual OLS via scipy for full stats (coefficients + p-values)
    X_with_const = np.column_stack([np.ones(len(X)), X.values])
    result = np.linalg.lstsq(X_with_const, y.values, rcond=None)
    coeffs = result[0]
    # Residuals and standard errors
    y_pred = X_with_const @ coeffs
    residuals = y.values - y_pred
    n, k = len(y), X_with_const.shape[1]
    mse = np.sum(residuals**2) / (n - k)
    var_b = mse * np.linalg.inv(X_with_const.T @ X_with_const).diagonal()
    se = np.sqrt(var_b)
    t_vals = coeffs / se
    p_vals = [2 * (1 - stats.t.cdf(abs(t), df=n-k)) for t in t_vals]
    ss_tot = np.sum((y.values - y.mean())**2)
    ss_res = np.sum(residuals**2)
    r2 = 1 - ss_res / ss_tot
    print(f"\n  Outcome: {outcome}  |  R² = {r2:.4f}")
    print(f"  {'Predictor':30s} {'Coeff':>10} {'p-value':>10} {'Sig':>5}")
    print(f"  {'Intercept':30s} {coeffs[0]:>10.4f} {p_vals[0]:>10.4f}")
    for i, col in enumerate(works_cols):
        sig = "***" if p_vals[i+1] < 0.001 else ("**" if p_vals[i+1] < 0.01 else ("*" if p_vals[i+1] < 0.05 else ""))
        print(f"  {col:30s} {coeffs[i+1]:>10.4f} {p_vals[i+1]:>10.4f} {sig:>5}")
        mlr_results.append({
            "Outcome": outcome, "Predictor": col,
            "Coefficient": round(coeffs[i+1], 4),
            "p_value": round(p_vals[i+1], 4),
            "Significant": sig,
            "R2": round(r2, 4)
        })

mlr_df = pd.DataFrame(mlr_results)

# 6. SAVE ------------------------------------------------------------------
clean.to_csv(f"{OUT}/flood_clean_data.csv", index=False)
spearman_df.to_csv(f"{OUT}/flood_spearman_results.csv", index=False)
full_corr.to_csv(f"{OUT}/flood_spearman_matrix.csv")
mlr_df.to_csv(f"{OUT}/flood_mlr_results.csv", index=False)
desc.to_csv(f"{OUT}/flood_descriptive_stats.csv")

print(f"\n[save] flood_clean_data.csv          -> clean 9-field dataset (raw numbers)")
print(f"[save] flood_spearman_results.csv    -> rho + p-values per variable pair")
print(f"[save] flood_spearman_matrix.csv     -> correlation matrix for heatmap")
print(f"[save] flood_mlr_results.csv         -> MLR coefficients + p-values")
print(f"[save] flood_descriptive_stats.csv   -> descriptive statistics")
print("\nDONE.")
