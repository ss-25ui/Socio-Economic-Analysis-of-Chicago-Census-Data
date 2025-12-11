#!/usr/bin/env python3
# 04_visualize_demographics_better.py
# Uses YOUR CSV (decennial_2010_2020_comparison.csv) and produces clearer,
# more insightful demographic visuals.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ====== CONFIG ======
IN_CSV  = "decennial_2010_2020_comparison_ca.csv"
OUT_DIR = "."
# ====================

os.makedirs(OUT_DIR, exist_ok=True)
df = pd.read_csv(IN_CSV, dtype={"ca_num": str})

need = [
    "ca_num",
    "pop_total_2010","pop_total_2020","pop_total_change",
    "white_pct_2010","black_pct_2010","asian_pct_2010","hispanic_pct_2010",
    "white_pct_2020","black_pct_2020","asian_pct_2020","hispanic_pct_2020",
    "hispanic_pct_change"
]
missing = [c for c in need if c not in df.columns]
if missing:
    raise ValueError(f"CSV is missing required columns: {missing}")

# ---------- Helpers ----------
def counts_from_shares(prefix):
    pop = df[f"pop_total_{prefix}"].astype(float)
    return {
        "pop": pop,
        "white":    (df[f"white_pct_{prefix}"].astype(float)    /100.0) * pop,
        "black":    (df[f"black_pct_{prefix}"].astype(float)    /100.0) * pop,
        "asian":    (df[f"asian_pct_{prefix}"].astype(float)    /100.0) * pop,
        "hispanic": (df[f"hispanic_pct_{prefix}"].astype(float) /100.0) * pop,
    }

def weighted_share(total_counts, key):
    denom = total_counts["pop"]
    return 100.0 * (total_counts[key] / denom) if denom > 0 else np.nan

def diversity_index(shares_pct):
    # Simpson diversity: 1 - Σ p_i^2, with p in proportions (0..1)
    p = np.clip(np.array(shares_pct) / 100.0, 0, 1)
    return float(1.0 - np.sum(p**2))

# ---------- Reconstruct population-weighted totals ----------
c10 = counts_from_shares("2010")
c20 = counts_from_shares("2020")
tot10 = {k: np.nansum(v) for k, v in c10.items()}
tot20 = {k: np.nansum(v) for k, v in c20.items()}

# Population totals
total_pop_2010 = tot10["pop"]
total_pop_2020 = tot20["pop"]

# Weighted shares overall
shares10 = {
    "White":    weighted_share(tot10, "white"),
    "Black":    weighted_share(tot10, "black"),
    "Asian":    weighted_share(tot10, "asian"),
    "Hispanic": weighted_share(tot10, "hispanic"),
}
shares20 = {
    "White":    weighted_share(tot20, "white"),
    "Black":    weighted_share(tot20, "black"),
    "Asian":    weighted_share(tot20, "asian"),
    "Hispanic": weighted_share(tot20, "hispanic"),
}

# ---------- 1) Total population bar (KEEP) ----------
plt.figure(figsize=(6,5))
vals = [total_pop_2010, total_pop_2020]
labels = ["2010", "2020"]
bars = plt.bar(labels, vals)
for b, v in zip(bars, vals):
    plt.text(b.get_x()+b.get_width()/2, v + max(vals)*0.01, f"{int(round(v)):,}",
             ha="center", va="bottom")
plt.ylabel("Total Population (persons)")
plt.title("Total Population — All Chicago Community Areas")
plt.tight_layout()
out = os.path.join(OUT_DIR, "dem_total_pop_bar.png")
plt.savefig(out, dpi=300); plt.close(); print("Saved:", out)

# ---------- 2) Race/Ethnicity pies (KEEP, with clear labels) ----------
labels_r = ["White","Black","Asian","Hispanic"]
r2010 = [shares10[k] for k in labels_r]
r2020 = [shares20[k] for k in labels_r]

fig, ax = plt.subplots(1,2, figsize=(12,6))
ax[0].pie(r2010, labels=labels_r, autopct="%1.1f%%", startangle=90)
ax[0].set_title("2010 Composition (population-weighted)")
ax[1].pie(r2020, labels=labels_r, autopct="%1.1f%%", startangle=90)
ax[1].set_title("2020 Composition (population-weighted)")
fig.suptitle("Race/Ethnicity Composition — All Chicago Community Areas\nValues are shares of total population",
             fontsize=13, y=0.98)
plt.tight_layout()
out = os.path.join(OUT_DIR, "dem_race_pies_2010_2020.png")
plt.savefig(out, dpi=300, bbox_inches="tight"); plt.close(); print("Saved:", out)

# ---------- 3) Non-White share (2010 vs 2020) ----------
nonwhite_2010 = 100.0 - shares10["White"]
nonwhite_2020 = 100.0 - shares20["White"]
plt.figure(figsize=(6,5))
bars = plt.bar(["2010", "2020"], [nonwhite_2010, nonwhite_2020])
for i,(b,v) in enumerate(zip(bars, [nonwhite_2010, nonwhite_2020])):
    plt.text(b.get_x()+b.get_width()/2, v+0.5, f"{v:.1f}%", ha="center", va="bottom")
plt.ylim(0, 100)
plt.ylabel("Percent of Total Population")
plt.title("Non-White Share of Population — 2010 vs 2020 (weighted)")
plt.tight_layout()
out = os.path.join(OUT_DIR, "dem_nonwhite_share_bar.png")
plt.savefig(out, dpi=300); plt.close(); print("Saved:", out)

# ---------- 4) Diversity Index (Simpson 1−Σp²), weighted ----------
div10 = diversity_index([shares10[k] for k in labels_r])
div20 = diversity_index([shares20[k] for k in labels_r])
delta_div = div20 - div10

plt.figure(figsize=(6,5))
bars = plt.bar(["2010","2020"], [div10, div20])
for b, v in zip(bars, [div10, div20]):
    plt.text(b.get_x()+b.get_width()/2, v + 0.01, f"{v:.3f}", ha="center", va="bottom")
plt.ylim(0, 1.0)
plt.ylabel("Diversity (Simpson 1−Σp², 0 to 1)")
plt.title(f"Diversity Index — 2010 vs 2020 (weighted)\nΔ = {delta_div:.3f}")
plt.tight_layout()
out = os.path.join(OUT_DIR, "dem_diversity_index_bar.png")
plt.savefig(out, dpi=300); plt.close(); print("Saved:", out)

# ---------- 5) Top 10 CAs by Hispanic share increase (pp) ----------
if "hispanic_pct_change" in df.columns:
    top_hisp = df.sort_values("hispanic_pct_change", ascending=False).head(10).copy()
    plt.figure(figsize=(10,6))
    plt.barh(top_hisp["ca_num"], top_hisp["hispanic_pct_change"])
    plt.gca().invert_yaxis()
    plt.xlabel("Change in Hispanic Share (percentage points, 2020 − 2010)")
    plt.title("Top 10 Community Areas — Largest Increase in Hispanic Share")
    for i,v in enumerate(top_hisp["hispanic_pct_change"]):
        plt.text(v + (0.5 if v>=0 else -0.5), i, f"{v:.1f} pp",
                 va="center", ha="left" if v>=0 else "right")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "dem_top10_hispanic_share_increase.png")
    plt.savefig(out, dpi=300); plt.close(); print("Saved:", out)

# ---------- 6) Majority group counts (2010 vs 2020) ----------
def majority_group(row, year):
    vals = {
        "White":    row[f"white_pct_{year}"],
        "Black":    row[f"black_pct_{year}"],
        "Asian":    row[f"asian_pct_{year}"],
        "Hispanic": row[f"hispanic_pct_{year}"],
    }
    # handle NaNs by treating as very small
    for k in vals:
        if pd.isna(vals[k]): vals[k] = -1
    return max(vals, key=vals.get)

maj2010 = df.apply(lambda r: majority_group(r, "2010"), axis=1)
maj2020 = df.apply(lambda r: majority_group(r, "2020"), axis=1)

counts2010 = maj2010.value_counts().reindex(labels_r, fill_value=0)
counts2020 = maj2020.value_counts().reindex(labels_r, fill_value=0)

x = np.arange(len(labels_r))
w = 0.35
plt.figure(figsize=(9,5))
plt.bar(x - w/2, counts2010.values, width=w, label="2010")
plt.bar(x + w/2, counts2020.values, width=w, label="2020")
plt.xticks(x, labels_r)
plt.ylabel("Number of Community Areas")
plt.title("Majority Group by Block Group — Counts (2010 vs 2020)")
plt.legend()
plt.tight_layout()
out = os.path.join(OUT_DIR, "dem_majority_group_counts.png")
plt.savefig(out, dpi=300); plt.close(); print("Saved:", out)

print("\n✅ Figures in folder")