"""
config.py

Shared configuration for evaluation and visualization scripts.
Edit paths here instead of in each individual script.
"""

from pathlib import Path

from _paths import DATA_ROOT, IFIMAGE_ROOT

# ============================================================================
# PATHS
# ============================================================================

# Base directory containing GT data
DATASET_DIR = IFIMAGE_ROOT / "00_dataset_withoutpecam"

# Where evaluation results are saved/loaded
RESULTS_DIR = IFIMAGE_ROOT / "evaluation_results"

# Where plots are saved
PLOTS_DIR = IFIMAGE_ROOT / "plots"

# ============================================================================
# ALGORITHM DIRECTORIES
# ============================================================================

# Cell segmentation (2-channel: DAPI + marker)
CELL_2CH_ALGOS = {
    "CellposeSAM": IFIMAGE_ROOT / "01_cellpose_benchmark" / "refilter_outputs" / "feat-mean_thr-otsu_area-100_gate-off",
    "CellposeSAM_Unrefined": IFIMAGE_ROOT / "01_cellpose_benchmark" / "cyto_prediction",
    "StarDist": IFIMAGE_ROOT / "02_stardist_benchmark" / "refilter_outputs" / "feat-mean_thr-otsu_area-100_gate-off",
    "CellSAM": IFIMAGE_ROOT / "03_cellsam_benchmark" / "refilter_outputs" / "feat-mean_thr-otsu_area-100_gate-off",
    "MESMER": IFIMAGE_ROOT / "04_mesmer_benchmark" / "refilter_outputs" / "feat-mean_thr-otsu_area-100_gate-off",
    "Watershed": IFIMAGE_ROOT / "06_watershed_benchmark" / "refilter_outputs" / "feat-mean_thr-otsu_area-100_gate-off",
    "Omnipose": IFIMAGE_ROOT / "07_omnipose_benchmark" / "refilter_outputs" / "feat-mean_thr-otsu_area-100_gate-off",
    "LACSS": IFIMAGE_ROOT / "011_lacss" / "refilter_outputs" / "feat-mean_thr-otsu_area-100_gate-off",
    "SplineDist": IFIMAGE_ROOT / "08_splinedist_benchmark" / "refilter_outputs" / "feat-mean_thr-otsu_area-100_gate-off",
    "MicroSAM": IFIMAGE_ROOT / "012_microsam_benchmark" / "refilter_outputs" / "feat-mean_thr-otsu_area-100_gate-off",
    "InstanSeg": IFIMAGE_ROOT / "013_ins" / "refilter_outputs" / "feat-mean_thr-otsu_area-100_gate-off",
}

# Cell segmentation (marker-only)
CELL_MARKER_ALGOS = {
    "CellposeSAM": IFIMAGE_ROOT / "01_cellpose_benchmark" / "markeronly",
    "StarDist": IFIMAGE_ROOT / "02_stardist_benchmark" / "markeronly",
    "CellSAM": IFIMAGE_ROOT / "03_cellsam_benchmark" / "markeronly",
    "MESMER": IFIMAGE_ROOT / "04_mesmer_benchmark" / "markeronly",
    "Watershed": IFIMAGE_ROOT / "06_watershed_benchmark" / "markeronly",
    "Omnipose": IFIMAGE_ROOT / "07_omnipose_benchmark" / "markeronly",
    "LACSS": IFIMAGE_ROOT / "011_lacss" / "markeronly",
    "SplineDist": IFIMAGE_ROOT / "08_splinedist_benchmark" / "markeronly",
    "MicroSAM": IFIMAGE_ROOT / "012_microsam_benchmark" / "markeronly",
    "InstanSeg": IFIMAGE_ROOT / "013_ins" / "markeronly",
}

# Nuclei segmentation
NUCLEI_ALGOS = {
    "CellposeSAM": IFIMAGE_ROOT / "01_cellpose_benchmark" / "nuclei_prediction",
    "StarDist": IFIMAGE_ROOT / "02_stardist_benchmark" / "nuclei_prediction",
    "CellSAM": IFIMAGE_ROOT / "03_cellsam_benchmark" / "nuclei_prediction",
    "MESMER": IFIMAGE_ROOT / "04_mesmer_benchmark" / "nuclei_prediction",
    "Watershed": IFIMAGE_ROOT / "06_watershed_benchmark" / "nuclei_prediction",
    "Omnipose": IFIMAGE_ROOT / "07_omnipose_benchmark" / "nuclei_prediction",
    "SplineDist": IFIMAGE_ROOT / "08_splinedist_benchmark" / "nuclei_prediction",
    "LACSS": IFIMAGE_ROOT / "011_lacss" / "nuclei_prediction",
    "MicroSAM": IFIMAGE_ROOT / "012_microsam_benchmark" / "nuclei_prediction",
    "InstanSeg": IFIMAGE_ROOT / "013_ins" / "nuclei_prediction",
}

# ============================================================================
# EVALUATION PARAMETERS
# ============================================================================

# IoU thresholds for AP calculation
AP_THRESHOLDS = tuple([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])

# Boundary F-score scales
BOUNDARY_SCALES = (1.0, 2.0)

# Parallel workers for evaluation
MAX_WORKERS = 8

# ============================================================================
# PLOT PARAMETERS
# ============================================================================

# Cell type grouping patterns (regex patterns for filtering images by cell type)
CELL_TYPE_GROUPS = {
    "OLIG2": r"OLIG2",  # Oligodendrocyte marker
    "NEUN": r"NEUN",    # Neuronal marker
    "IBA1": r"IBA1",    # Microglia marker
    "GFAP": r"GFAP",    # Astrocyte marker
}

# Fixed colors for each algorithm (consistent across all plots)
# Using professional color palette with good contrast and colorblind-safe colors
ALGORITHM_COLORS = {
    # Primary methods (darker, more saturated colors)
    "CellposeSAM": "#1f77b4",           # Professional blue
    "CellposeSAM_Unrefined": "#aec7e8", # Light blue (for comparison)
    "StarDist": "#ff7f0e",              # Orange
    "CellSAM": "#2ca02c",               # Green
    "MESMER": "#d62728",                # Red
    
    # Secondary methods (complementary colors)
    "Watershed": "#9467bd",             # Purple
    "Omnipose": "#8c564b",              # Brown
    "LACSS": "#e377c2",                 # Pink
    "SplineDist": "#7f7f7f",            # Gray
    "MicroSAM": "#bcbd22",              # Yellow-green
    "InstanSeg": "#17becf",             # Cyan
    
    # Alternative names (same colors as base algorithms)
    "Cellpose": "#1f77b4",              # Same as CellposeSAM
    "Cellpose Unrefined": "#aec7e8",    # Same as CellposeSAM_Unrefined
}

# Line styles for different algorithm variants
# Format: "-" solid, "--" dashed, "-." dash-dot, ":" dotted
ALGORITHM_LINESTYLES = {
    "CellposeSAM": "-",                 # Solid
    "CellposeSAM_Unrefined": "--",      # Dashed (to distinguish from refined)
    "StarDist": "-",
    "CellSAM": "-",
    "MESMER": "-",
    "Watershed": "-",
    "Omnipose": "-",
    "LACSS": "-",
    "SplineDist": "-",
    "MicroSAM": "-",
    "InstanSeg": "-",
}

# Marker styles for each algorithm
# Common markers: "o" circle, "s" square, "^" triangle, "D" diamond, "*" star
ALGORITHM_MARKERS = {
    "CellposeSAM": "o",                 # Circle
    "CellposeSAM_Unrefined": "s",       # Square
    "StarDist": "^",                    # Triangle up
    "CellSAM": "v",                     # Triangle down
    "MESMER": "D",                      # Diamond
    "Watershed": "p",                   # Pentagon
    "Omnipose": "h",                    # Hexagon
    "LACSS": "*",                       # Star
    "SplineDist": "X",                  # X
    "MicroSAM": "P",                    # Plus (filled)
    "InstanSeg": "d",                   # Thin diamond
}

# Figure DPI for saved images
FIGURE_DPI = 300

# Whether to use transparent background
TRANSPARENT_BG = True

# Default figure size (width, height) in inches
DEFAULT_FIGSIZE = (7, 5)

# Font sizes for different plot elements
FONT_SIZES = {
    "title": 11,    # Figure title
    "label": 10,    # Axis labels
    "legend": 9,    # Legend text
    "tick": 9,      # Tick labels
}
