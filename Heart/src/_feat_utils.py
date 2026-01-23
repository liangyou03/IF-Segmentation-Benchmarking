"""Utilities shared by training scripts."""
import pandas as pd

FEAT_MINIMAL = [
    "area_px2","perimeter_px","major_axis_px","minor_axis_px",
    "eccentricity","solidity","extent","feret_max_px","aspect_ratio","circularity",
]

def pick_auto_features(df: pd.DataFrame) -> list:
    """Pick numeric columns, exclude IDs/QC and spatially leaky coords, drop all-NaN columns."""
    exclude_prefix = ("centroid_", "bbox_")
    exclude_exact  = {"section_id","image_id","roi_id","unified_label","qc_pass","qc_reason"}
    feats = []
    for c in df.columns:
        if c in exclude_exact: 
            continue
        if any(c.startswith(p) for p in exclude_prefix):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if df[c].notna().any():  # avoid all-NaN
            feats.append(c)
    return sorted(feats)
