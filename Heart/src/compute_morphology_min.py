#!/usr/bin/env python3
"""Minimal nuclear morphology features using scikit-image."""
import argparse, math
from pathlib import Path
import numpy as np, pandas as pd
from roifile import ImagejRoi
from skimage.draw import polygon as draw_polygon
from skimage.measure import label, regionprops_table

def load_xy(roi_path: Path):
    """Return Nx2 (x,y) integer coordinates from an ImageJ ROI."""
    ij = ImagejRoi.fromfile(str(roi_path))
    yx = np.array(list(ij.coordinates()), float)  # (y,x)
    return np.c_[yx[:,1], yx[:,0]]               # (x,y)

def rasterize_xy(xy):
    """Rasterize polygon to a tight binary mask around the ROI."""
    x, y = xy[:,0], xy[:,1]
    x0, x1 = int(np.floor(x.min()))-1, int(np.ceil(x.max()))+1
    y0, y1 = int(np.floor(y.min()))-1, int(np.ceil(y.max()))+1
    h, w = max(1, y1-y0+1), max(1, x1-x0+1)
    rr, cc = draw_polygon(y-y0, x-x0, (h, w))
    m = np.zeros((h, w), bool); m[rr, cc] = True
    return m

def circ(area, perim):  # 4πA/P²
    return np.nan if perim <= 0 else 4*math.pi*area/(perim*perim)

def main():
    ap = argparse.ArgumentParser(description="Minimal morphology with scikit-image.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_area_um2", type=float, default=5.0, help="QC threshold on area (um^2 if available).")
    args = ap.parse_args()

    man = pd.read_csv(args.manifest)
    out_rows = []

    for _, r in man.iterrows():
        rp = Path(str(r["roi_abspath"]))
        if not rp.exists(): 
            out_rows.append({**r, "qc_pass":0, "qc_reason":"missing_roi"}); continue
        try:
            xy = load_xy(rp); mask = rasterize_xy(xy)
            lab = label(mask.astype(np.uint8))
            props = regionprops_table(lab, properties=[
                "area","perimeter","major_axis_length","minor_axis_length",
                "eccentricity","solidity","extent","feret_diameter_max",
            ])
            if len(props["area"]) == 0:
                out_rows.append({**r, "qc_pass":0, "qc_reason":"empty"}); continue

            Apx  = float(props["area"][0]); Ppx  = float(props["perimeter"][0])
            Maj  = float(props["major_axis_length"][0]); Min  = float(props["minor_axis_length"][0])
            Ecc  = float(props["eccentricity"][0]);      Sol  = float(props["solidity"][0])
            Ext  = float(props["extent"][0]);            Fmax = float(props["feret_diameter_max"][0])

            Aspect = Maj/Min if Min>0 else np.nan
            Circ   = circ(Apx, Ppx)

            # Optional micron conversion
            px_um = float(r.get("pixel_size_x_um", np.nan))
            py_um = float(r.get("pixel_size_y_um", np.nan))
            Aum2  = Apx*(px_um*py_um) if np.isfinite(px_um) and np.isfinite(py_um) else np.nan

            area_qc  = Aum2 if np.isfinite(Aum2) else Apx
            qc_pass  = 1 if area_qc >= args.min_area_um2 else 0
            qc_reason= "" if qc_pass else f"small_area<{args.min_area_um2}"

            out_rows.append({
                "section_id": r.get("section_id",""),
                "image_id": r.get("image_id",""),
                "unified_label": r.get("unified_label",""),
                "roi_id": r.get("roi_id",""),
                "roi_abspath": str(rp),
                "area_px2": Apx, "perimeter_px": Ppx,
                "major_axis_px": Maj, "minor_axis_px": Min,
                "eccentricity": Ecc, "solidity": Sol, "extent": Ext,
                "feret_max_px": Fmax, "aspect_ratio": Aspect, "circularity": Circ,
                "area_um2": Aum2, "qc_pass": qc_pass, "qc_reason": qc_reason
            })
        except Exception as e:
            out_rows.append({**r, "qc_pass":0, "qc_reason":f"error:{type(e).__name__}"})

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(out, index=False)
    print(f"[OK] wrote {len(out_rows)} rows -> {out}")

if __name__ == "__main__":
    main()
