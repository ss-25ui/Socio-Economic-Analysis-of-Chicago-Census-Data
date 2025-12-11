#!/usr/bin/env python3
"""
aggregate_acs_to_ca.py

Aggregate ACS 2023 block group features to Community Area level using population-weighted averages.

Author: Sai Shashank Satuluri
"""

import pandas as pd
import numpy as np

# -----------------------------
# CONFIGURATION
# -----------------------------
ACS_BG_CSV = "acs2023_extended_bg_all.csv"
CA_BG_MAPPING = "output/ca_bg_mapping.csv"
OUT_CA_CSV = "acs2023_features_ca.csv"

# Features to aggregate (population-weighted)
WEIGHTED_FEATURES = [
    "median_age", "pct_male", "pct_female", "pct_under_18", "pct_65plus",
    "pct_white", "pct_black", "pct_asian", "pct_other_race", "pct_hispanic",
    "median_rent", "median_household_income", "pct_bachelors_or_higher",
    "pct_drove_alone", "pct_carpooled", "pct_public_transit", "pct_walk_bike", "pct_worked_from_home",
    "pct_poverty", "pct_unemployed", "pct_owner_occupied", "pct_renter_occupied",
    "median_home_value", "gini_index", "pct_foreign_born", "pct_english_only",
    "pct_has_computer_broadband", "pct_no_vehicle", "population_density_km2"
]

# -----------------------------
# Main
# -----------------------------
def main():
    print("Loading ACS BG features...")
    acs_bg = pd.read_csv(ACS_BG_CSV, dtype={"GEOID": str})

    print("Loading CA-BG mapping...")
    ca_bg = pd.read_csv(CA_BG_MAPPING, dtype={"GEOID": str})

    print("Merging ACS with CA mapping...")
    acs_ca = acs_bg.merge(ca_bg, on="GEOID", how="inner")

    print("Aggregating to CA level...")
    ca_agg = acs_ca.groupby("ca_num").apply(
        lambda g: pd.Series({
            "total_pop": g["total_pop"].sum(),
            **{feat: (g[feat] * g["total_pop"]).sum() / g["total_pop"].sum() for feat in WEIGHTED_FEATURES if feat != "population_density_km2"},
            "population_density_km2": (g["population_density_km2"] * g["total_pop"]).sum() / g["total_pop"].sum()  # Weighted by pop
        }),
        include_groups=False
    ).reset_index()

    print(f"Aggregated to {len(ca_agg)} CAs")

    print("Saving CA-level features...")
    ca_agg.to_csv(OUT_CA_CSV, index=False)
    print(f"Saved to {OUT_CA_CSV}")

    print("Nulls check:")
    print(ca_agg.isna().sum().sort_values(ascending=False))

if __name__ == "__main__":
    main()
