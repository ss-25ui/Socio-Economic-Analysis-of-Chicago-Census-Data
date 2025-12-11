#!/usr/bin/env python3
"""
05_additional_visualizations.py

Additional visualizations for citywide socio-spatial clustering:
- Feature heatmap by cluster
- Network graph of hybrid similarity
- Choropleth maps for key ACS features
- Box plots for feature distributions by cluster

Author: Sai Shashank Satuluri
"""

import os
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.preprocessing import StandardScaler

# -----------------------------
# CONFIG
# -----------------------------
FEATURES_CSV = "acs2023_features_ca.csv"
CLUSTERS_CSV = "acs2023_ca_clusters.csv"
CA_GEOJSON = "https://data.cityofchicago.org/resource/igwz-8jzy.geojson"
OUT_DIR = "."

os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
print("Loading data...")
df_features = pd.read_csv(FEATURES_CSV, dtype={"ca_num": str})
df_clusters = pd.read_csv(CLUSTERS_CSV, dtype={"ca_num": str})
ca_gdf = gpd.read_file(CA_GEOJSON).to_crs(4326)
ca_gdf["ca_num"] = ca_gdf["area_numbe"].astype(str)

# Merge
df = df_features.merge(df_clusters[["ca_num", "cluster"]], on="ca_num", how="left")
df = ca_gdf[["ca_num", "community", "geometry"]].merge(df, on="ca_num", how="left")

# Feature columns (exclude ca_num, geometry, community, cluster)
feature_cols = [c for c in df.columns if c not in ["ca_num", "geometry", "community", "cluster"]]

# -----------------------------
# 1) Feature Heatmap by Cluster
# -----------------------------
print("Creating feature heatmap...")
cluster_means = df.groupby("cluster")[feature_cols].mean()
plt.figure(figsize=(12, 8))
sns.heatmap(cluster_means.T, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Mean Feature Values by Cluster (Standardized)")
plt.tight_layout()
out = os.path.join(OUT_DIR, "cluster_feature_heatmap.png")
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# -----------------------------
# 2) Network Graph (simplified, using feature similarity)
# -----------------------------
print("Creating network graph...")
# Build simple KNN graph for visualization (top 5 neighbors)
from sklearn.neighbors import kneighbors_graph
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[feature_cols])
knn_graph = kneighbors_graph(X_scaled, n_neighbors=5, mode="connectivity", include_self=False)
G = nx.from_scipy_sparse_array(knn_graph)

# Add node attributes
for i, row in df.iterrows():
    G.nodes[i]["cluster"] = row["cluster"]
    G.nodes[i]["community"] = row["community"]

# Plot
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, seed=42)
colors = [G.nodes[n]["cluster"] for n in G.nodes]
nx.draw(G, pos, node_color=colors, with_labels=False, node_size=100, cmap=plt.cm.tab10, edge_color="gray", alpha=0.7)
plt.title("Hybrid Feature-Spatial Similarity Network (KNN Approximation)")
plt.axis("off")
out = os.path.join(OUT_DIR, "ca_similarity_network.png")
plt.savefig(out, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# -----------------------------
# 3) Choropleth Maps for Key Features
# -----------------------------
key_features = ["median_household_income", "pct_bachelors_or_higher", "population_density_km2"]
for feat in key_features:
    if feat in df.columns:
        print(f"Creating choropleth for {feat}...")
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        df.plot(column=feat, ax=ax, legend=True, cmap="viridis", edgecolor="black", linewidth=0.5)
        ax.set_title(f"Choropleth: {feat.replace('_', ' ').title()}")
        ax.axis("off")
        out = os.path.join(OUT_DIR, f"choropleth_{feat}.png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out}")

# -----------------------------
# 4) Box Plots for Feature Distributions by Cluster
# -----------------------------
print("Creating box plots...")
for feat in key_features:
    if feat in df.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df, x="cluster", y=feat, hue="cluster", palette="Set2", legend=False)
        plt.title(f"Distribution of {feat.replace('_', ' ').title()} by Cluster")
        plt.tight_layout()
        out = os.path.join(OUT_DIR, f"boxplot_{feat}_by_cluster.png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out}")

print("\n✅ Additional visualizations completed.")
