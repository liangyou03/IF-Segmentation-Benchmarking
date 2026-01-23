#!/usr/bin/env python3
"""Extended morphology features via scikit-image regionprops_table.

python src/compute_morphology_max.py \
  --manifest data/processed/manifest.csv \
  --out data/processed/features_morphology_full.csv
  
"""
import argparse, math
from pathlib import Path
import numpy as np, pandas as pd
from roifile import ImagejRoi
from skimage.draw import polygon as draw_polygon
from skimage.measure import label, regionprops_table

def load_xy(roi_path: Path):
    ij = ImagejRoi.fromfile(str(roi_path))
    yx = np.array(list(ij.coordinates()), float)
    return np.c_[yx[:,1], yx[:,0]]

def rasterize_xy(xy):
    x, y = xy[:,0], xy[:,1]
    x0, x1 = int(np.floor(x.min()))-1, int(np.ceil(x.max()))+1
    y0, y1 = int(np.floor(y.min()))-1, int(np.ceil(y.max()))+1
    h, w = max(1, y1-y0+1), max(1, x1-x0+1)
    rr, cc = draw_polygon(y-y0, x-x0, (h, w))
    m = np.zeros((h, w), bool); m[rr, cc] = True
    return m

def circ(area, perim): return np.nan if perim<=0 else 4*math.pi*area/(perim*perim)
def roundness(area, maj): return np.nan if maj<=0 else 4*area/(math.pi*maj*maj)

PROPS = [
    "area","perimeter","convex_area","eccentricity","equivalent_diameter_area",
    "euler_number","extent","feret_diameter_max","filled_area",
    "major_axis_length","minor_axis_length","orientation","solidity",
    "bbox","centroid","moments_hu","inertia_tensor_eigvals"
]

def main():
    ap = argparse.ArgumentParser(description="Extended morphology with scikit-image.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_area_um2", type=float, default=5.0, help="QC threshold on area (um^2 if available).")
    args = ap.parse_args()

    man = pd.read_csv(args.manifest)
    rows = []
    for _, r in man.iterrows():
        rp = Path(str(r["roi_abspath"]))
        if not rp.exists():
            rows.append({**r, "qc_pass":0, "qc_reason":"missing_roi"}); continue
        try:
            mask = rasterize_xy(load_xy(rp))
            lab  = label(mask.astype(np.uint8))
            tbl  = regionprops_table(lab, properties=PROPS)
            if len(tbl["area"]) == 0:
                rows.append({**r, "qc_pass":0, "qc_reason":"empty"}); continue

            # Flatten regionprops_table (first region only)
            rec = {k: (v[0] if isinstance(v, (list, np.ndarray, pd.Series)) else v) for k,v in tbl.items()}
            # Derived scalars
            Apx  = float(rec["area"]); Ppx = float(rec["perimeter"])
            Maj  = float(rec["major_axis_length"]); Min = float(rec["minor_axis_length"])
            rec["aspect_ratio"] = Maj/Min if Min>0 else np.nan
            rec["circularity"]  = circ(Apx, Ppx)
            rec["roundness"]    = roundness(Apx, Maj)
            # bbox area
            y0,x0,y1,x1 = int(rec.get("bbox-0",0)), int(rec.get("bbox-1",0)), int(rec.get("bbox-2",0)), int(rec.get("bbox-3",0))
            rec["bbox_area_px2"] = max(0,(y1-y0))*max(0,(x1-x0))

            # Micron area if pixel size provided
            px_um = float(r.get("pixel_size_x_um", np.nan))
            py_um = float(r.get("pixel_size_y_um", np.nan))
            rec["area_um2"] = Apx*(px_um*py_um) if np.isfinite(px_um) and np.isfinite(py_um) else np.nan

            area_qc = rec["area_um2"] if np.isfinite(rec["area_um2"]) else Apx
            qc_pass = 1 if area_qc >= args.min_area_um2 else 0
            rec["qc_pass"], rec["qc_reason"] = qc_pass, ("" if qc_pass else f"small_area<{args.min_area_um2}")

            # Attach IDs
            rec.update({
                "section_id": r.get("section_id",""),
                "image_id": r.get("image_id",""),
                "unified_label": r.get("unified_label",""),
                "roi_id": r.get("roi_id",""),
                "roi_abspath": str(rp)
            })
            rows.append(rec)
        except Exception as e:
            rows.append({**r, "qc_pass":0, "qc_reason":f"error:{type(e).__name__}"})

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[OK] wrote {len(rows)} rows -> {out}")

if __name__ == "__main__":
    main()
