#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot heatmap of correlations with:
- left Method color bar
- left Phenotype color bar
- within-heatmap significance stars (p < 0.05)

Input CSV must contain:
    method, phenotype, celltype, r, p
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from _paths import FULLDATA_ROOT

# ========== 1. Load data ==========
df = pd.read_csv(FULLDATA_ROOT / "correlation_analysis.csv")

# Ensure ordering (customize)
method_order = ["scMD", "EpiSCORE", "HiBED"]
phenotype_order = ["Age", "CDR", "CERAD", "Braak Score"]
celltype_order = ["Astro", "Micro", "Endo", "Oligo", "OPC", "Inh", "Exc"]

df["method"] = pd.Categorical(df["method"], method_order)
df["phenotype"] = pd.Categorical(df["phenotype"], phenotype_order)
df["celltype"] = pd.Categorical(df["celltype"], celltype_order)

# pivot: rows = phenotype × method; columns = cell types
row_keys = []
for m in method_order:
    for ph in phenotype_order:
        row_keys.append((m, ph))

rows = pd.MultiIndex.from_tuples(row_keys, names=["method", "phenotype"])
heat = df.pivot_table(index=["method", "phenotype"], columns="celltype", values="r").reindex(rows)
pvals = df.pivot_table(index=["method", "phenotype"], columns="celltype", values="p").reindex(rows)

# ========== 2. Color maps ==========
cmap = mpl.cm.get_cmap("coolwarm")  # red→blue

method_colors = {
    "scMD": "#d44b4b",
    "EpiSCORE": "#48a86f",
    "HiBED": "#3458bc"
}

phenotype_colors = {
    "Age": "#4daf4a",
    "CDR": "#ff7f00",
    "CERAD": "#e41a1c",
    "Braak Score": "#377eb8"
}

# ========== 3. Draw figure ==========
fig = plt.figure(figsize=(10, 10))
gs = fig.add_gridspec(nrows=len(rows), ncols=3, width_ratios=[0.2, 0.2, 1.0])

# Left method bar
ax_m = fig.add_subplot(gs[:, 0])
ax_m.imshow(
    np.array([[method_colors[m]] for m, ph in rows]).reshape(-1, 1, 3),
    aspect="auto"
)
ax_m.set_xticks([])
ax_m.set_yticks([])

# Left phenotype bar
ax_p = fig.add_subplot(gs[:, 1])
ax_p.imshow(
    np.array([[phenotype_colors[ph]] for m, ph in rows]).reshape(-1, 1, 3),
    aspect="auto"
)
ax_p.set_xticks([])
ax_p.set_yticks([])

# Main heatmap
ax = fig.add_subplot(gs[:, 2])
im = ax.imshow(heat, cmap=cmap, vmin=-0.2, vmax=0.2, aspect="auto")

ax.set_xticks(np.arange(len(celltype_order)))
ax.set_xticklabels(celltype_order, rotation=45, ha="right")

ax.set_yticks(np.arange(len(rows)))
ax.set_yticklabels([f"{m} | {ph}" for m, ph in rows], fontsize=7)

# Add stars
for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        if not np.isnan(pvals.iloc[i, j]) and pvals.iloc[i, j] < 0.05:
            ax.text(j, i, "*", ha="center", va="center", fontsize=12, color="black")

# Colorbar
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Correlation (Spearman r)")

plt.tight_layout()
plt.show()
