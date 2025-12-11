#!/usr/bin/env python3
"""
02_build_features_and_network.py

Socioeconomic + spatial clustering for selected Chicago block groups (CA 23 + CA 26):
 - Load ACS features (2023) and standardize
 - Build feature-space KNN graph
 - Build spatial adjacency graph from BG shapefile
 - Combine graphs and run Leiden clustering
 - Compute silhouette score
 - Save cluster CSV and cluster map (PNG)

Author: Sai Shashank Satuluri
"""

import os
import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.neighbors import kneighbors_graph
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import igraph as ig
import leidenalg as la

# -----------------------------
# CONFIG (edit paths as needed)
# -----------------------------
DATA_CSV       = "acs2023_features_ca.csv"
BG_SHP_PATH    = "tl_2023_17_bg/tl_2023_17_bg.shp"
OUT_CLUSTER_CSV= "acs2023_ca_clusters.csv"
OUT_PLOT_PATH  = "acs2023_ca_clusters_map.png"

K_NEIGHBORS = 8          # K for feature KNN
RESOLUTION  = 0.8        # Leiden resolution (tune if too many/few clusters)

# -----------------------------
# 1) Load data and standardize
# -----------------------------
print("\nLoading ACS feature data…")
# Force ca_num to string for node labels
df = pd.read_csv(DATA_CSV, dtype={"ca_num": str})

feature_cols = [
    "median_age","pct_male","pct_female","pct_under_18","pct_65plus",
    "pct_white","pct_black","pct_asian","pct_other_race","pct_hispanic",
    "median_rent","median_household_income","pct_bachelors_or_higher",
    "pct_drove_alone","pct_carpooled","pct_public_transit","pct_walk_bike","pct_worked_from_home",
    "population_density_km2"
]

X = df[feature_cols].copy()
print(f"Standardizing {len(feature_cols)} features…")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols, index=df.index)

df_scaled = pd.concat([df[["ca_num"]], X_scaled_df], axis=1)

# -----------------------------
# 2) Feature-space KNN graph
# -----------------------------
print("\nBuilding feature-space KNN graph…")
knn_graph = kneighbors_graph(X_scaled, n_neighbors=K_NEIGHBORS, mode="connectivity", include_self=False)
G_knn = nx.from_scipy_sparse_array(knn_graph)
G_knn = nx.relabel_nodes(G_knn, {i: df["ca_num"].iloc[i] for i in range(len(df))})
print(f"KNN graph → nodes: {G_knn.number_of_nodes()}, edges: {G_knn.number_of_edges()}")

# -----------------------------
# 3) Spatial adjacency graph
# -----------------------------
print("\nBuilding spatial adjacency graph…")
gdf = gpd.read_file(BG_SHP_PATH)
gdf = gdf[gdf["COUNTYFP"] == "031"].copy()
# ensure 12-char GEOIDs
gdf["GEOID"] = gdf["GEOID"].astype(str).str.zfill(12)

# keep only our BGs - but since we're working at CA level, we don't need BG selection here
# gdf_sel = gdf[gdf["GEOID"].isin(df["GEOID"])].copy().reset_index(drop=True)

# Build adjacency via CA touches (queen)
ca_gdf = gpd.read_file("https://data.cityofchicago.org/resource/igwz-8jzy.geojson").to_crs(4326)
ca_gdf["ca_num"] = ca_gdf["area_numbe"].astype(str)
ca_gdf = ca_gdf[ca_gdf["ca_num"].isin(df["ca_num"])].copy()

spatial_edges = []
for i, geom_i in enumerate(ca_gdf.geometry):
    neighbors = ca_gdf[ca_gdf.geometry.touches(geom_i)].index.tolist()
    u = ca_gdf.loc[i, "ca_num"]
    for j in neighbors:
        v = ca_gdf.loc[j, "ca_num"]
        if u != v:
            spatial_edges.append((u, v))

G_spatial = nx.Graph()
G_spatial.add_nodes_from(df["ca_num"].tolist())
G_spatial.add_edges_from(spatial_edges)
print(f"Spatial graph → nodes: {G_spatial.number_of_nodes()}, edges: {G_spatial.number_of_edges()}")

# -----------------------------
# 4) Combine graphs
# -----------------------------
print("\nCombining spatial + feature graphs (union)…")
G_combined = nx.compose(G_knn, G_spatial)
print(f"Combined graph → nodes: {G_combined.number_of_nodes()}, edges: {G_combined.number_of_edges()}")

# -----------------------------
# 5) Leiden clustering
# -----------------------------
print("\nRunning Leiden clustering…")
g_ig = ig.Graph.from_networkx(G_combined)
partition = la.find_partition(g_ig, la.RBConfigurationVertexPartition, resolution_parameter=RESOLUTION)
labels = np.array(partition.membership)

df_labeled = df.copy()
df_labeled["cluster"] = labels
n_clusters = int(df_labeled["cluster"].nunique())
print(f"Detected {n_clusters} clusters.")

# -----------------------------
# 6) Silhouette + summary
# -----------------------------
print("\nComputing silhouette score…")
sil = silhouette_score(X_scaled, labels)
print(f"Silhouette score: {sil:.3f}")

summary = df_labeled.groupby("cluster")[feature_cols].mean().round(2)
print("\nCluster summary (scaled means):")
print(summary)

# Save labeled CSV
df_labeled.to_csv(OUT_CLUSTER_CSV, index=False)
print(f"\nClustered CSV saved → {OUT_CLUSTER_CSV}")

# -----------------------------
# 7) Visualization (t-SNE + Map)
# -----------------------------
print("\nCreating t-SNE plot…")
tsne = TSNE(n_components=2, random_state=42, perplexity=10)
tsne_xy = tsne.fit_transform(X_scaled)
df_labeled["tsne_x"], df_labeled["tsne_y"] = tsne_xy[:,0], tsne_xy[:,1]

plt.figure(figsize=(7,6))
for c in sorted(df_labeled["cluster"].unique()):
    sub = df_labeled[df_labeled["cluster"] == c]
    plt.scatter(sub["tsne_x"], sub["tsne_y"], alpha=0.75, label=f"Cluster {c}")
plt.title("t-SNE Projection of Community Areas (standardized features)")
plt.legend()
plt.tight_layout()
tsne_out = os.path.join(os.path.dirname(OUT_PLOT_PATH), "acs2023_ca_tsne_clusters.png")
plt.savefig(tsne_out, dpi=300, bbox_inches="tight")
plt.close()
print(f"t-SNE plot saved → {tsne_out}")

# ---- Map: CA-level clusters ----
print("\nPreparing CA map join…")
# Use CA boundaries for mapping
ca_gdf = gpd.read_file("https://data.cityofchicago.org/resource/igwz-8jzy.geojson").to_crs(4326)
ca_gdf["ca_num"] = ca_gdf["area_numbe"].astype(str)
# left-join clusters
ca_join = ca_gdf.merge(df_labeled[["ca_num","cluster"]], on="ca_num", how="left", indicator=True)
print(ca_join["_merge"].value_counts())   # should be 'both' for all

# Plot map
print("Plotting CA cluster map…")
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ca_join.plot(
    column="cluster",
    categorical=True,
    cmap="tab20",
    legend=True,
    linewidth=0.5,
    edgecolor="black",
    ax=ax,
    missing_kwds={"color": "lightgrey", "hatch": "///", "label": "No cluster"}
)
ax.set_title("Leiden Clusters of Chicago Community Areas (2023)")
ax.axis("off")
plt.savefig(OUT_PLOT_PATH, dpi=300, bbox_inches="tight")
plt.close()
print(f"CA cluster map saved → {OUT_PLOT_PATH}")

print("\n✅ Done.")