#!/usr/bin/env python3
"""
ca_selection.py (NO HARDCODED PATHS)

Select all 77 Chicago Community Areas and their intersecting block groups.
Regenerate selected_bg_ids.txt for citywide analysis.

Author: Sai Shashank Satuluri
"""

import geopandas as gpd
import os
import json
from tkinter import Tk
from tkinter.filedialog import askopenfilename


# ----------------------------------------------------
# Helper: always ask user to select inputs
# ----------------------------------------------------
def get_paths():
    cfg_file = "config.json"

    print("\nSelect the Illinois 2023 Block Group SHP file:")
    Tk().withdraw()
    shp_path = askopenfilename(
        title="Select tl_2023_17_bg.shp",
        filetypes=[("Shapefile", "*.shp")]
    )

    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)

    cfg = {
        "blockgroup_shapefile": shp_path,
        "output_dir": output_dir
    }

    with open(cfg_file, "w") as f:
        json.dump(cfg, f, indent=4)

    print("\nSaved configuration to config.json")
    return cfg


# ----------------------------------------------------
# Load config / paths
# ----------------------------------------------------
paths = get_paths()
BG_SHP = paths["blockgroup_shapefile"]
OUT_DIR = paths["output_dir"]

print(f"\nUsing Block Groups file: {BG_SHP}")
print(f"Outputs will be stored in: {OUT_DIR}\n")


# ----------------------------------------------------
# 1) Load Community Areas (CAs)
# ----------------------------------------------------
ca = gpd.read_file("https://data.cityofchicago.org/resource/igwz-8jzy.geojson")
ca["ca_num"] = ca["area_numbe"].astype(int)
ca_sel = ca[["ca_num", "community", "geometry"]].copy()


# ----------------------------------------------------
# 2) Load Block Groups (Cook County only)
# ----------------------------------------------------
bg = gpd.read_file(BG_SHP)
bg = bg[bg["COUNTYFP"] == "031"].copy()


# ----------------------------------------------------
# 3) Reproject to equal-area CRS
# ----------------------------------------------------
EA = "EPSG:26916"
ca_ea = ca_sel.to_crs(EA)
bg_ea = bg.to_crs(EA)


# ----------------------------------------------------
# 4) Intersections
# ----------------------------------------------------
inter = gpd.overlay(bg_ea, ca_ea, how="intersection", keep_geom_type=True)
if inter.empty:
    raise RuntimeError("No intersections found—check inputs/CRS and Cook filter.")

inter["inter_area"] = inter.geometry.area


# ----------------------------------------------------
# 5) Assign each BG to CA with max overlap
# ----------------------------------------------------
idx = inter.groupby("GEOID")["inter_area"].idxmax()
inter_max = inter.loc[idx, ["GEOID", "ca_num", "community", "inter_area"]].copy()


# ----------------------------------------------------
# 6) Counts
# ----------------------------------------------------
counts = inter_max.groupby("ca_num")["GEOID"].nunique()
total_bgs = counts.sum()

print(f"Total Community Areas: {len(ca_sel)}")
print(f"Total Block Groups assigned: {total_bgs}")


# ----------------------------------------------------
# 7) Save outputs (NO HARDCODED PATHS)
# ----------------------------------------------------
bg_id_out = os.path.join(OUT_DIR, "selected_bg_ids.txt")
with open(bg_id_out, "w") as f:
    f.write("\n".join(inter_max["GEOID"].unique().tolist()))
print(f"Saved selected BG IDs → {bg_id_out}")

mapping_out = os.path.join(OUT_DIR, "ca_bg_mapping.csv")
inter_max[["GEOID", "ca_num", "community"]].to_csv(mapping_out, index=False)
print(f"Saved CA–BG mapping → {mapping_out}")
