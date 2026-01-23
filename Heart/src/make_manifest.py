#!/usr/bin/env python3
"""Scan ROI folders and build a manifest CSV.

Folder convention:
data/raw/
  01_section/<class>/*.roi
  02_section/<class>/*.roi
  (optionally a single image file per section: *.tif or *.ome.tif)


python src/make_manifest.py   --raw_dir data/raw   --out data/processed/manifest.csv   --px_um 0.5 --py_um 0.5


Output columns:
section_id,image_id,unified_label,roi_id,roi_abspath,pixel_size_x_um,pixel_size_y_um
"""
import argparse, re
from pathlib import Path
import pandas as pd

LABEL_MAP = {
    "mural_cell":"mural", "mural":"mural",
    "immune_cell":"immune", "immune":"immune",
    "epi":"epi", "ec":"ec", "cm":"cm", "fb":"fb"
}

def norm_label(name: str) -> str:
    key = name.strip().lower()
    return LABEL_MAP.get(key, key)

def guess_image_id(section_dir: Path) -> str:
    # Try to select a single image file name as image_id, else fallback to section folder name
    imgs = sorted([*section_dir.glob("*.ome.tif"), *section_dir.glob("*.tif"), *section_dir.glob("*.tiff")])
    return imgs[0].stem.split(".")[0] if imgs else section_dir.name

def main():
    ap = argparse.ArgumentParser(description="Build manifest from ROI tree.")
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--px_um", type=float, default=float("nan"), help="Pixel size X (microns).")
    ap.add_argument("--py_um", type=float, default=float("nan"), help="Pixel size Y (microns).")
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    rows = []
    for sec_dir in sorted(raw.glob("*_section")):
        m = re.match(r"(\d+)_section", sec_dir.name)
        section_id = m.group(1) if m else sec_dir.name
        image_id = guess_image_id(sec_dir)
        for cls_dir in sorted([p for p in sec_dir.iterdir() if p.is_dir()]):
            ulabel = norm_label(cls_dir.name)
            for roi in sorted(cls_dir.glob("*.roi")):
                rows.append({
                    "section_id": section_id,
                    "image_id": image_id,
                    "unified_label": ulabel,
                    "roi_id": roi.stem,
                    "roi_abspath": str(roi.resolve()),
                    "pixel_size_x_um": args.px_um,
                    "pixel_size_y_um": args.py_um,
                })
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[OK] wrote {len(rows)} rows -> {out}")

if __name__ == "__main__":
    main()
