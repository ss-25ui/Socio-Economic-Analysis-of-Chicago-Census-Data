import pandas as pd
import geopandas as gpd
import requests, io

# --- Configuration ---
BG_2010 = r"tl_2010_17031_bg10/tl_2010_17031_bg10.shp"
BG_2020 = r"tl_2020_17_bg/tl_2020_17_bg.shp"
COMMUNITY_URL = "https://data.cityofchicago.org/resource/igwz-8jzy.geojson"

# 1️⃣ Load Community Areas and select all 77
ca = gpd.read_file(COMMUNITY_URL).to_crs(4326)
ca["ca_num"] = ca["area_numbe"].astype(int)
# Select all 77 CAs

# 2️⃣ Load 2010 + 2020 block groups for Cook County
bg10 = gpd.read_file(BG_2010)
bg20 = gpd.read_file(BG_2020)
bg10 = bg10[bg10["COUNTYFP10"]=="031"].to_crs(4326)
bg20 = bg20[bg20["COUNTYFP"]=="031"].to_crs(4326)

# 3️⃣ Spatial filter (intersection with all CA)
sel10 = gpd.sjoin(bg10, ca[["ca_num","geometry"]], predicate="intersects")
sel20 = gpd.sjoin(bg20, ca[["ca_num","geometry"]], predicate="intersects")

print("2010 BGs:", len(sel10), "  2020 BGs:", len(sel20))

# 4️⃣ Download Census API data (Decennial PL 94-171)
def fetch_decennial(year, vars):
    state, county = "17","031"
    base = f"https://api.census.gov/data/{year}/dec/pl"
    get = ",".join(vars)
    url = f"{base}?get={get}&for=block%20group:*&in=state:{state}%20county:{county}"
    r = requests.get(url)
    r.raise_for_status()
    df = pd.DataFrame(r.json()[1:], columns=r.json()[0])
    df["GEOID"] = df["state"]+df["county"]+df["tract"]+df["block group"]
    return df

vars2010 = ["P001001","P001003","P001004","P001005","P001006","P002002"]
vars2020 = ["P1_001N","P1_003N","P1_004N","P1_005N","P1_006N","P2_002N"]

df10 = fetch_decennial(2010, vars2010)
df20 = fetch_decennial(2020, vars2020)

# 5️⃣ Rename + compute race percentages
df10 = df10.rename(columns={
    "P001001":"pop_total","P001003":"white","P001004":"black",
    "P001005":"asian","P001006":"other","P002002":"hispanic"})
df20 = df20.rename(columns={
    "P1_001N":"pop_total","P1_003N":"white","P1_004N":"black",
    "P1_005N":"asian","P1_006N":"other","P2_002N":"hispanic"})

for df in [df10,df20]:
    df[["pop_total","white","black","asian","other","hispanic"]] = \
        df[["pop_total","white","black","asian","other","hispanic"]].astype(int)
    for col in ["white","black","asian","other","hispanic"]:
        df[col+"_pct"] = 100*df[col]/df["pop_total"]

# 6️⃣ Filter to your block groups and merge by GEOID
sel10["GEOID"] = sel10["GEOID10"].astype(str)
sel20["GEOID"] = sel20["GEOID"].astype(str)

m10 = sel10.merge(df10, on="GEOID", how="left")
m20 = sel20.merge(df20, on="GEOID", how="left")

# 7️⃣ Compare % change between 2010 and 2020
compare = m20[["GEOID","pop_total","white_pct","black_pct","asian_pct","hispanic_pct"]].copy()
compare = compare.rename(columns={c:c+"_2020" for c in compare.columns if c!="GEOID"})

m10_cols = m10[["GEOID","pop_total","white_pct","black_pct","asian_pct","hispanic_pct"]].rename(
    columns={c:c+"_2010" for c in m10.columns if c!="GEOID"})
compare = compare.merge(m10_cols, on="GEOID", how="inner")

for col in ["pop_total","white_pct","black_pct","asian_pct","hispanic_pct"]:
    compare[col+"_change"] = compare[f"{col}_2020"] - compare[f"{col}_2010"]

# 8️⃣ Aggregate to CA level using population-weighted averages
ca_compare = []
for ca_num in ca["ca_num"]:
    ca_bgs = compare[compare["GEOID"].isin(sel10[sel10["ca_num"] == ca_num]["GEOID"])]
    if not ca_bgs.empty:
        # Population-weighted averages
        total_pop_2010 = ca_bgs["pop_total_2010"].sum()
        total_pop_2020 = ca_bgs["pop_total_2020"].sum()
        weighted_white_2010 = (ca_bgs["white_pct_2010"] * ca_bgs["pop_total_2010"]).sum() / total_pop_2010 if total_pop_2010 > 0 else 0
        weighted_white_2020 = (ca_bgs["white_pct_2020"] * ca_bgs["pop_total_2020"]).sum() / total_pop_2020 if total_pop_2020 > 0 else 0
        # Similarly for other races
        weighted_black_2010 = (ca_bgs["black_pct_2010"] * ca_bgs["pop_total_2010"]).sum() / total_pop_2010 if total_pop_2010 > 0 else 0
        weighted_black_2020 = (ca_bgs["black_pct_2020"] * ca_bgs["pop_total_2020"]).sum() / total_pop_2020 if total_pop_2020 > 0 else 0
        weighted_asian_2010 = (ca_bgs["asian_pct_2010"] * ca_bgs["pop_total_2010"]).sum() / total_pop_2010 if total_pop_2010 > 0 else 0
        weighted_asian_2020 = (ca_bgs["asian_pct_2020"] * ca_bgs["pop_total_2020"]).sum() / total_pop_2020 if total_pop_2020 > 0 else 0
        weighted_hispanic_2010 = (ca_bgs["hispanic_pct_2010"] * ca_bgs["pop_total_2010"]).sum() / total_pop_2010 if total_pop_2010 > 0 else 0
        weighted_hispanic_2020 = (ca_bgs["hispanic_pct_2020"] * ca_bgs["pop_total_2020"]).sum() / total_pop_2020 if total_pop_2020 > 0 else 0

        ca_compare.append({
            "ca_num": ca_num,
            "pop_total_2010": total_pop_2010,
            "pop_total_2020": total_pop_2020,
            "white_pct_2010": weighted_white_2010,
            "white_pct_2020": weighted_white_2020,
            "black_pct_2010": weighted_black_2010,
            "black_pct_2020": weighted_black_2020,
            "asian_pct_2010": weighted_asian_2010,
            "asian_pct_2020": weighted_asian_2020,
            "hispanic_pct_2010": weighted_hispanic_2010,
            "hispanic_pct_2020": weighted_hispanic_2020,
            "pop_total_change": total_pop_2020 - total_pop_2010,
            "white_pct_change": weighted_white_2020 - weighted_white_2010,
            "black_pct_change": weighted_black_2020 - weighted_black_2010,
            "asian_pct_change": weighted_asian_2020 - weighted_asian_2010,
            "hispanic_pct_change": weighted_hispanic_2020 - weighted_hispanic_2010,
        })

ca_compare_df = pd.DataFrame(ca_compare)

# 9️⃣ Save outputs
compare.to_csv("decennial_2010_2020_comparison.csv", index=False)
ca_compare_df.to_csv("decennial_2010_2020_comparison_ca.csv", index=False)
print("Saved decennial_2010_2020_comparison.csv successfully.")
print("Saved decennial_2010_2020_comparison_ca.csv successfully.")
