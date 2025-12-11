#!/usr/bin/env python3
"""
01_acs_2023_extended_features.py

Pull ACS 2023 (acs/acs5) BLOCK GROUP data for:
- population & demographics
- housing (median rent)
- economics (median household income)
- derived population density (people per km²)

Author: <your name>
"""

import os
import requests
import pandas as pd
import numpy as np
from typing import List
import geopandas as gpd

# -----------------------------
# CONFIGURATION (EDIT THESE)
# -----------------------------
BG_ID_PATH = "output/selected_bg_ids.txt"
BG_SHP_PATH = "tl_2023_17_bg/tl_2023_17_bg.shp"
OUT_CSV    = "acs2023_extended_bg_all.csv"

# Optional Census API key
CENSUS_API_KEY = os.getenv("CENSUS_API_KEY", "")

YEAR = 2023
API_BASE = "https://api.census.gov/data"
ENDPOINT = f"{API_BASE}/{YEAR}/acs/acs5"
STATE_FIPS  = "17"   # Illinois
COUNTY_FIPS = "031"  # Cook

# -----------------------------
# Helper: section header printing
# -----------------------------
def header(msg: str):
    print("\n" + msg)
    print("-" * len(msg))

# -----------------------------
# 1️⃣ Load selected BG GEOIDs
# -----------------------------
def load_bg_ids(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        ids = [ln.strip() for ln in f if ln.strip()]
    ids = [s[:12] for s in ids]  # normalize to 12-digit
    return sorted(set(ids))

# -----------------------------
# 2️⃣ Fetch ACS data
# -----------------------------
def fetch_vars(var_list: List[str]) -> pd.DataFrame:
    params = {
        "get": ",".join(var_list),
        "for": "block group:*",
        "in":  f"state:{STATE_FIPS} county:{COUNTY_FIPS}",
    }
    if CENSUS_API_KEY:
        params["key"] = CENSUS_API_KEY
    r = requests.get(ENDPOINT, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    cols, rows = data[0], data[1:]
    df = pd.DataFrame(rows, columns=cols)
    for c in df.columns:
        if c not in ("state","county","tract","block group"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["GEOID"] = (
        df["state"].astype(str).str.zfill(2)
        + df["county"].astype(str).str.zfill(3)
        + df["tract"].astype(str).str.zfill(6)
        + df["block group"].astype(str)
    )
    return df

def fetch_union(var_list: List[str], chunk: int = 40) -> pd.DataFrame:
    frames = []
    for i in range(0, len(var_list), chunk):
        sub = var_list[i:i+chunk]
        frames.append(fetch_vars(sub))
    out = frames[0]
    for df in frames[1:]:
        out = out.merge(
            df,
            on=["state","county","tract","block group","GEOID"],
            how="outer",
            validate="one_to_one"
        )
    return out

# -----------------------------
# 3️⃣ Variables (Population, Demographics, Rent, Income)
# -----------------------------
BASE_VARS = [
    "B01003_001E",  # total population
    "B01002_001E",  # median age
    "B02001_001E",  # race total
    "B02001_002E",  # White alone
    "B02001_003E",  # Black alone
    "B02001_005E",  # Asian alone
    "B03003_001E",  # Hispanic total
    "B03003_003E",  # Hispanic count
    "B25064_001E",  # median rent
    "B19013_001E",  # median household income
    # for % under 18, %65+, %male we need B01001
    "B01001_001E","B01001_002E","B01001_026E",
    "B01001_003E","B01001_004E","B01001_005E","B01001_006E",
    "B01001_027E","B01001_028E","B01001_029E","B01001_030E",
    "B01001_020E","B01001_021E","B01001_022E","B01001_023E","B01001_024E","B01001_025E",
    "B01001_044E","B01001_045E","B01001_046E","B01001_047E","B01001_048E","B01001_049E",
    # Education
    "B15003_001E",  # education total
    "B15003_017E","B15003_018E","B15003_021E","B15003_022E",  # bachelor's or higher
    # Transportation
    "B08301_001E",  # means of transportation to work total
    "B08301_003E",  # drove alone
    "B08301_004E",  # carpooled
    "B08301_010E",  # public transit
    "B08301_016E","B08301_017E",  # walk/bike
    "B08301_019E",  # worked from home
    # Poverty
    "B17001_001E",  # poverty status total
    "B17001_002E",  # below poverty level
    # Unemployment
    "B23025_001E",  # employment status total
    "B23025_005E",  # unemployed
    # Housing tenure
    "B25003_001E",  # tenure total
    "B25003_002E",  # owner occupied
    "B25003_003E",  # renter occupied
    # Median home value
    "B25077_001E",  # median home value
    # Gini index
    "B19083_001E",  # Gini index
    # Foreign-born
    "B05012_001E",  # nativity total
    "B05012_003E",  # foreign-born
    # Language
    "B16001_001E",  # language spoken at home total
    "B16001_002E",  # English only
    # Internet access
    "B28002_001E",  # presence of computer total
    "B28002_013E",  # has computer and broadband
    # Vehicles
    "B08201_001E",  # household size by vehicles total
    "B08201_002E",  # no vehicle
]

# -----------------------------
# 4️⃣ Build feature dataframe
# -----------------------------
def build_features(df: pd.DataFrame, gdf_bg: gpd.GeoDataFrame) -> pd.DataFrame:
    header("Building derived features")
    out = pd.DataFrame({"GEOID": df["GEOID"]})

    # --- population and age ---
    out["total_pop"]  = df["B01003_001E"]
    out["median_age"] = df["B01002_001E"]

    # --- gender composition ---
    tot = df["B01001_001E"].replace(0, np.nan)
    out["pct_male"]   = df["B01001_002E"] / tot
    out["pct_female"] = df["B01001_026E"] / tot

    # --- age structure ---
    u18_m = df[["B01001_003E","B01001_004E","B01001_005E","B01001_006E"]].sum(axis=1)
    u18_f = df[["B01001_027E","B01001_028E","B01001_029E","B01001_030E"]].sum(axis=1)
    out["pct_under_18"] = (u18_m + u18_f) / tot

    a65_m = df[["B01001_020E","B01001_021E","B01001_022E","B01001_023E","B01001_024E","B01001_025E"]].sum(axis=1)
    a65_f = df[["B01001_044E","B01001_045E","B01001_046E","B01001_047E","B01001_048E","B01001_049E"]].sum(axis=1)
    out["pct_65plus"] = (a65_m + a65_f) / tot

    # --- race and ethnicity ---
    race_tot = df["B02001_001E"].replace(0, np.nan)
    out["pct_white"] = df["B02001_002E"] / race_tot
    out["pct_black"] = df["B02001_003E"] / race_tot
    out["pct_asian"] = df["B02001_005E"] / race_tot
    summed = df[["B02001_002E","B02001_003E","B02001_005E"]].fillna(0).sum(axis=1)
    out["pct_other_race"] = np.clip(1 - summed / race_tot, 0, 1)

    hisp_tot = df["B03003_001E"].replace(0, np.nan)
    out["pct_hispanic"] = df["B03003_003E"] / hisp_tot

    # --- housing & income ---
    out["median_rent"] = df["B25064_001E"]
    out["median_household_income"] = df["B19013_001E"]

    # --- education ---
    edu_tot = df["B15003_001E"].replace(0, np.nan)
    out["pct_bachelors_or_higher"] = (df["B15003_017E"] + df["B15003_018E"] + df["B15003_021E"] + df["B15003_022E"]) / edu_tot

    # --- transportation ---
    trans_tot = df["B08301_001E"].replace(0, np.nan)
    out["pct_drove_alone"] = df["B08301_003E"] / trans_tot
    out["pct_carpooled"] = df["B08301_004E"] / trans_tot
    out["pct_public_transit"] = df["B08301_010E"] / trans_tot
    out["pct_walk_bike"] = (df["B08301_016E"] + df["B08301_017E"]) / trans_tot
    out["pct_worked_from_home"] = df["B08301_019E"] / trans_tot

    # --- poverty ---
    pov_tot = df["B17001_001E"].replace(0, np.nan)
    out["pct_poverty"] = df["B17001_002E"] / pov_tot

    # --- unemployment ---
    emp_tot = df["B23025_001E"].replace(0, np.nan)
    out["pct_unemployed"] = df["B23025_005E"] / emp_tot

    # --- housing tenure ---
    ten_tot = df["B25003_001E"].replace(0, np.nan)
    out["pct_owner_occupied"] = df["B25003_002E"] / ten_tot
    out["pct_renter_occupied"] = df["B25003_003E"] / ten_tot

    # --- median home value ---
    out["median_home_value"] = df["B25077_001E"]

    # --- inequality (Gini index) ---
    out["gini_index"] = df["B19083_001E"]

    # --- foreign-born ---
    nat_tot = df["B05012_001E"].replace(0, np.nan)
    out["pct_foreign_born"] = df["B05012_003E"] / nat_tot

    # --- language ---
    lang_tot = df["B16001_001E"].replace(0, np.nan)
    out["pct_english_only"] = df["B16001_002E"] / lang_tot

    # --- internet access ---
    comp_tot = df["B28002_001E"].replace(0, np.nan)
    out["pct_has_computer_broadband"] = df["B28002_013E"] / comp_tot

    # --- vehicles ---
    veh_tot = df["B08201_001E"].replace(0, np.nan)
    out["pct_no_vehicle"] = df["B08201_002E"] / veh_tot

    # --- population density (per km²) ---
    gdf_bg = gdf_bg.to_crs(3857)  # project to meters for area calc
    area_km2 = gdf_bg.area / 1e6
    area_df = pd.DataFrame({"GEOID": gdf_bg["GEOID"], "area_km2": area_km2.values})
    out = out.merge(area_df, on="GEOID", how="left")
    out["population_density_km2"] = np.where(
        (out["area_km2"] > 0) & out["total_pop"].notna(),
        out["total_pop"] / out["area_km2"],
        np.nan,
    )

    cols = [
        "GEOID","total_pop","median_age","pct_male","pct_female",
        "pct_under_18","pct_65plus",
        "pct_white","pct_black","pct_asian","pct_other_race","pct_hispanic",
        "median_rent","median_household_income","pct_bachelors_or_higher",
        "pct_drove_alone","pct_carpooled","pct_public_transit","pct_walk_bike","pct_worked_from_home",
        "pct_poverty","pct_unemployed","pct_owner_occupied","pct_renter_occupied",
        "median_home_value","gini_index","pct_foreign_born","pct_english_only",
        "pct_has_computer_broadband","pct_no_vehicle","population_density_km2"
    ]
    return out[cols]

# -----------------------------
# 5️⃣ Main
# -----------------------------
def main():
    header("Loading selected GEOIDs")
    geoids = load_bg_ids(BG_ID_PATH)
    print(f"Loaded {len(geoids)} BGs")

    header("Downloading ACS 2023 variables")
    df_all = fetch_union(BASE_VARS)

    header("Filtering to selected BGs")
    df_sel = df_all[df_all["GEOID"].isin(geoids)].copy()
    print(f"After filter: {df_sel['GEOID'].nunique()} BGs (expected ~{len(geoids)})")

    header("Loading BG shapefile for area calculations")
    gdf_bg = gpd.read_file(BG_SHP_PATH)
    gdf_bg = gdf_bg[gdf_bg["COUNTYFP"] == "031"].copy()
    gdf_bg["GEOID"] = gdf_bg["GEOID"].astype(str)

    feats = build_features(df_sel, gdf_bg)

    header("Writing CSV")
    if os.path.dirname(OUT_CSV):
        os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    feats.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(feats)} rows to: {OUT_CSV}")

    header("Nulls check (counts by column)")
    print(feats.isna().sum().sort_values(ascending=False))

if __name__ == "__main__":
    main()