# heart/run_splinedist.py
"""
SplineDist inference script for all channels.
Environment: ifimage_splinedist
"""
import numpy as np
from pathlib import Path
import tifffile
from tqdm import tqdm
from csbdeep.utils import normalize
from splinedist.models import SplineDist2D

from _paths import HEART_DATA_ROOT, IFIMAGE_ROOT

PROCESSED_DIR = HEART_DATA_ROOT / "processed"
OUTPUT_DIR = HEART_DATA_ROOT / "benchmark_results" / "splinedist_predictions"
PRETRAINED_ROOT = IFIMAGE_ROOT / "08_splinedist_benchmark" / "splinedist_models" / "bbbc038_8"

# Normalization percentiles
P_LOWER, P_UPPER = 1, 99.8

def _pick_sd_model_dir(root):
    """Locate a SplineDist model directory."""
    if not root.exists():
        return None
    # Allow the root itself or any subdirectory to contain the model files
    for p in [root] + list(root.rglob("*")):
        if p.is_dir():
            cfg = p / "config.json"
            has_w = any(p.glob("weights*.h5"))
            if cfg.exists() and has_w:
                return p.parent, p.name
    return None

def _load_model():
    """Load the SplineDist model."""
    picked = _pick_sd_model_dir(PRETRAINED_ROOT)
    if picked is None:
        raise FileNotFoundError(
            f"No SplineDist model directory containing config.json and weights_*.h5 found under {PRETRAINED_ROOT.resolve()}"
        )
    basedir, name = picked
    return SplineDist2D(None, name=name, basedir=str(basedir))

print("=" * 70)
print("🚀 SplineDist Segmentation")
print("=" * 70)

# Load model
print("Loading SplineDist model...")
model = _load_model()
print(f"Model loaded from: {model.basedir}/{model.name}")

tif_files = list(PROCESSED_DIR.glob('*/*.tif'))
print(f"Found {len(tif_files)} TIF images")
print(f"Output: {OUTPUT_DIR}")
print("=" * 70)

for tif_path in tqdm(tif_files, desc="SplineDist", unit="img"):
    image = tifffile.imread(tif_path)
    
    # Ensure 2D grayscale images
    if image.ndim == 3:
        image = image[..., 0]
    
    # Normalize
    image_norm = normalize(image, P_LOWER, P_UPPER)
    
    # SplineDist segmentation
    labels, _ = model.predict_instances(image_norm)
    labels = labels.astype(np.int32, copy=False)
    
    region = tif_path.parent.name
    filename = tif_path.stem
    output_path = OUTPUT_DIR / region / f"{filename}_pred.npy"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, labels)

print("\n" + "=" * 70)
print(f"✅ Done! Results saved to: {OUTPUT_DIR}")
print("=" * 70)
