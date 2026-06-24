import matplotlib.pyplot as plt

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