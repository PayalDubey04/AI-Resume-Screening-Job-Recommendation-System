from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
CHARTS_DIR = RESULTS_DIR / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Model Comparison: Error Metrics (MAE & RMSE - Lower is Better)
comp_df = pd.read_csv(RESULTS_DIR / "model_comparison.csv")
comp_df.columns = [c.strip() for c in comp_df.columns]
model_col = comp_df.columns[0]

fig, ax = plt.subplots(figsize=(8, 4.8))
x = range(len(comp_df))
width = 0.35

bars1 = ax.bar([i - width/2 for i in x], comp_df["MAE"], width, label="MAE (Lower is Better)", color="#4C72B0")
bars2 = ax.bar([i + width/2 for i in x], comp_df["RMSE"], width, label="RMSE (Lower is Better)", color="#DD8452")

ax.set_ylabel("Error Score")
ax.set_title("Model Error Comparison on 1,909 Held-Out Test Pairs")
ax.set_xticks(list(x))
ax.set_xticklabels(comp_df[model_col], rotation=12, ha="right")
ax.set_ylim(0, max(comp_df["RMSE"]) * 1.18)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)

for bar in bars1:
    ax.annotate(f"{bar.get_height():.4f}", (bar.get_x() + bar.get_width()/2, bar.get_height()),
                ha="center", va="bottom", fontsize=8.5)
for bar in bars2:
    ax.annotate(f"{bar.get_height():.4f}", (bar.get_x() + bar.get_width()/2, bar.get_height()),
                ha="center", va="bottom", fontsize=8.5)

plt.tight_layout()
plt.savefig(CHARTS_DIR / "1_model_error_comparison.png", dpi=300)
plt.close()

# 2. Model Comparison: Correlation Metrics (Pearson & Spearman - Higher is Better)
fig, ax = plt.subplots(figsize=(8, 4.8))
bars1 = ax.bar([i - width/2 for i in x], comp_df["Pearson"], width, label="Pearson r (Higher is Better)", color="#55A868")
bars2 = ax.bar([i + width/2 for i in x], comp_df["Spearman"], width, label="Spearman ρ (Higher is Better)", color="#8172B3")

ax.set_ylabel("Correlation Coefficient")
ax.set_title("Ranking & Linear Correlation Comparison (n = 1,909)")
ax.set_xticks(list(x))
ax.set_xticklabels(comp_df[model_col], rotation=12, ha="right")
ax.set_ylim(0, max(comp_df["Spearman"]) * 1.22)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)

for bar in bars1:
    ax.annotate(f"{bar.get_height():.4f}", (bar.get_x() + bar.get_width()/2, bar.get_height()),
                ha="center", va="bottom", fontsize=8.5)
for bar in bars2:
    ax.annotate(f"{bar.get_height():.4f}", (bar.get_x() + bar.get_width()/2, bar.get_height()),
                ha="center", va="bottom", fontsize=8.5)

plt.tight_layout()
plt.savefig(CHARTS_DIR / "2_model_correlation_comparison.png", dpi=300)
plt.close()

# 3. Ablation Study: MAE Across Field Combinations
abl_df = pd.read_csv(RESULTS_DIR / "ablation_results.csv")
colors = ["#2E7D32" if "[Ours]" in cfg else "#5C768D" for cfg in abl_df["Configuration"]]

fig, ax = plt.subplots(figsize=(9, 5))
y_pos = range(len(abl_df))
bars = ax.barh(y_pos, abl_df["MAE"], color=colors)

ax.set_yticks(list(y_pos))
ax.set_yticklabels(abl_df["Configuration"])
ax.invert_yaxis()
ax.set_xlabel("Mean Absolute Error (MAE — Lower is Better)")
ax.set_title("Field-Aware Ablation Study & Weight Sensitivity (50/20/30 vs. Equal)")
ax.set_xlim(0, max(abl_df["MAE"]) * 1.18)
ax.grid(axis="x", linestyle="--", alpha=0.4)

for bar, (_, row) in zip(bars, abl_df.iterrows()):
    ax.annotate(f"MAE: {row['MAE']:.4f} (n={row['Evaluated_Samples']})",
                (bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2),
                va="center", fontsize=8.5)

plt.tight_layout()
plt.savefig(CHARTS_DIR / "3_ablation_study.png", dpi=300)
plt.close()

print("All 3 charts saved successfully to:", CHARTS_DIR)