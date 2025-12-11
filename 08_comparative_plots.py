#!/usr/bin/env python3
"""
06_comparative_plots.py

Comparative plots between decennial census changes (2010-2020) and ACS 2023 features at CA level.
- Scatter plots with correlations
- Correlation heatmap
- Enhanced bar plot for Hispanic increases

Author: Sai Shashank Satuluri
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# -----------------------------
# CONFIG
# -----------------------------
DECENNIAL_CSV = "decennial_2010_2020_comparison_ca.csv"
ACS_CSV = "acs2023_features_ca.csv"
OUT_DIR = "."

os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# Load and merge data
# -----------------------------
print("Loading data...")
df_dec = pd.read_csv(DECENNIAL_CSV, dtype={"ca_num": str})
df_acs = pd.read_csv(ACS_CSV, dtype={"ca_num": str})

# Merge on ca_num
df = df_dec.merge(df_acs, on="ca_num", how="inner")
print(f"Merged data: {len(df)} community areas")

# -----------------------------
# 1) Scatter: Hispanic pct change vs. pct_unemployed
# -----------------------------
print("Creating scatter: Hispanic change vs. Unemployment...")
x = df["hispanic_pct_change"]
y = df["pct_unemployed"]
valid = ~(x.isna() | y.isna())
x_clean = x[valid]
y_clean = y[valid]

plt.figure(figsize=(8, 6))
plt.scatter(x_clean, y_clean, alpha=0.7, edgecolors="k")
plt.xlabel("Hispanic Share Change (2020 - 2010, pp)")
plt.ylabel("Unemployment Rate (%)")
plt.title("Hispanic Share Change vs. Unemployment Rate (ACS 2023)")

# Add regression line
if len(x_clean) > 1:
    m, b = np.polyfit(x_clean, y_clean, 1)
    plt.plot(x_clean, m*x_clean + b, color="red", linewidth=2)
    corr, _ = pearsonr(x_clean, y_clean)
    plt.text(0.05, 0.95, f"Pearson r = {corr:.2f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment="top")

plt.grid(True, alpha=0.3)
plt.tight_layout()
out = os.path.join(OUT_DIR, "scatter_hispanic_change_vs_unemployment.png")
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# -----------------------------
# 2) Scatter: Population change vs. median_household_income
# -----------------------------
print("Creating scatter: Population change vs. Median Income...")
x = df["pop_total_change"]
y = df["median_household_income"]
valid = ~(x.isna() | y.isna())
x_clean = x[valid]
y_clean = y[valid]

plt.figure(figsize=(8, 6))
plt.scatter(x_clean, y_clean, alpha=0.7, edgecolors="k")
plt.xlabel("Population Change (2020 - 2010)")
plt.ylabel("Median Household Income ($)")
plt.title("Population Change vs. Median Household Income (ACS 2023)")

# Add regression line
if len(x_clean) > 1:
    m, b = np.polyfit(x_clean, y_clean, 1)
    plt.plot(x_clean, m*x_clean + b, color="red", linewidth=2)
    corr, _ = pearsonr(x_clean, y_clean)
    plt.text(0.05, 0.95, f"Pearson r = {corr:.2f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment="top")

plt.grid(True, alpha=0.3)
plt.tight_layout()
out = os.path.join(OUT_DIR, "scatter_pop_change_vs_income.png")
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# -----------------------------
# 3) Correlation heatmap for selected variables
# -----------------------------
print("Creating correlation heatmap...")
# Select key decennial changes and ACS features
dec_vars = ["pop_total_change", "white_pct_change", "black_pct_change", "asian_pct_change", "hispanic_pct_change"]
acs_vars = ["median_household_income", "pct_bachelors_or_higher", "pct_poverty", "population_density_km2"]
all_vars = dec_vars + acs_vars

corr_df = df[all_vars].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap: Decennial Changes vs. ACS Features")
plt.tight_layout()
out = os.path.join(OUT_DIR, "correlation_heatmap_decennial_acs.png")
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# -----------------------------
# 4) Enhanced bar plot: Top 10 Hispanic increases, colored by poverty
# -----------------------------
print("Creating enhanced bar plot: Top 10 Hispanic increases...")
top_hisp = df.sort_values("hispanic_pct_change", ascending=False).head(10).copy()
plt.figure(figsize=(12, 8))
bars = plt.barh(top_hisp["ca_num"], top_hisp["hispanic_pct_change"], color=plt.cm.viridis(top_hisp["pct_poverty"] / top_hisp["pct_poverty"].max()))
plt.gca().invert_yaxis()
plt.xlabel("Change in Hispanic Share (percentage points, 2020 - 2010)")
plt.title("Top 10 Community Areas — Largest Increase in Hispanic Share\nColored by Poverty Rate (darker = higher poverty)")

# Add colorbar
sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=top_hisp["pct_poverty"].min(), vmax=top_hisp["pct_poverty"].max()))
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca())
cbar.set_label("Poverty Rate (%)")

for i, v in enumerate(top_hisp["hispanic_pct_change"]):
    plt.text(v + (0.5 if v >= 0 else -0.5), i, f"{v:.1f} pp", va="center", ha="left" if v >= 0 else "right")

plt.tight_layout()
out = os.path.join(OUT_DIR, "top10_hispanic_increase_colored_by_poverty.png")
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

print("\n✅ Comparative plots completed.")
