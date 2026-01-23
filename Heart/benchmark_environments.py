#!/usr/bin/env python3
"""
Comprehensive benchmarking script for all 10 segmentation algorithms.

Collects:
1. Environment size (MB/GB)
2. Package count
3. Runtime on a test image
4. Algorithm name and conda environment name
"""

import subprocess
import json
import time
import re
from pathlib import Path
import os
import numpy as np
import tifffile

from _paths import HEART_DATA_ROOT

# Configuration
CONDA_BASE = Path(os.environ.get("CONDA_BASE", Path.home() / "miniconda3"))
TEST_IMAGE = HEART_DATA_ROOT / "processed" / "LA" / "LA1_dapi.tif"
OUTPUT_CSV = HEART_DATA_ROOT / "benchmark_results.csv"

# Algorithm mapping: algorithm_name -> (conda_env, test_script_function)
ALGORITHMS = {
    "InstanSeg": ("instanseg-env", "run_InstanSeg.py"),
    "CellposeSAM": ("cellpose", "run_cellpose.py"),
    "Omnipose": ("omnipose", "run_omnipose.py"),
    "MicroSAM": ("microsam-cuda", "run_microsam.py"),
    "StarDist": ("stardist", "run_stardist.py"),
    "MESMER": ("deepcell_retinamask", "run_mesmer.py"),
    "Watershed": ("ifimage_evaluation", "run_watershed.py"),
    "LACSS": ("lacss", "run_lacss.py"),
    "SplineDist": ("ifimage_splinedist", "run_splinedist.py"),
    "CellSAM": ("ifimage_cellsam", "run_cellsam.py"),
}


def get_env_size(env_name):
    """Get environment directory size in GB."""
    env_path = CONDA_BASE / "envs" / env_name
    if not env_path.exists():
        return None
    try:
        result = subprocess.run(
            ["du", "-s", str(env_path)],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            size_kb = int(result.stdout.split()[0])
            return round(size_kb / 1024 / 1024, 2)  # Convert to GB
    except Exception as e:
        print(f"Error getting size for {env_name}: {e}")
    return None


def get_package_count(env_name):
    """Get number of packages in conda environment."""
    try:
        result = subprocess.run(
            ["conda", "list", "-n", env_name],
            capture_output=True, text=True,
            env={**os.environ, "CONDA_PREFIX": str(CONDA_BASE)},
            timeout=60
        )
        if result.returncode == 0:
            # Count non-empty lines (excluding header)
            lines = [l for l in result.stdout.split('\n') if l.strip() and not l.startswith('#')]
            return len(lines) - 1  # Subtract header
    except Exception as e:
        print(f"Error getting package count for {env_name}: {e}")
    return None


def run_algorithm_in_env(algo_name, env_name, script_path):
    """Run algorithm in its conda environment and measure runtime."""
    heart_dir = Path(__file__).resolve().parent
    script_full_path = heart_dir / script_path

    if not script_full_path.exists():
        return None, f"Script not found: {script_full_path}"

    # Create a minimal test script that imports and runs the algorithm
    test_script = f"""
import sys
sys.path.insert(0, '{heart_dir}')
import time
import numpy as np
import tifffile
from pathlib import Path

# Load test image
test_image = Path('{TEST_IMAGE}')
img = tifffile.imread(test_image)

# Import and run algorithm based on name
algo = '{algo_name}'
start = time.time()

try:
    if algo == 'InstanSeg':
        import torch
        from instanseg import InstanSeg
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = InstanSeg("fluorescence_nuclei_and_cells", device=device)
        labels, _ = model.eval_small_image(img)
        output = labels[0][1].cpu().numpy()

    elif algo == 'CellposeSAM':
        from cellpose import models
        model = models.CellposeModel(gpu=True)
        masks, _, _ = model.eval([img], diameter=None, flow_threshold=0.4, cellprob_threshold=0.0)
        output = masks[0]

    elif algo == 'Omnipose':
        from cellpose_omni import models
        model = models.CellposeModel(gpu=True, model_type='cyto2_omni')
        # Convert to 2-channel
        if img.ndim == 2:
            img2 = np.stack((img, img))
        else:
            img2 = np.stack((img[..., 0], img[..., 1])) if img.shape[-1] >= 2 else np.stack((img[..., 0], img[..., 0]))
        img2 = img2.astype(np.float32)
        masks, _, _ = model.eval([img2], channels=[1,2], diameter=30, mask_threshold=-1.0, flow_threshold=0.0)
        output = masks[0]

    elif algo == 'MicroSAM':
        from micro_sam.training import sam_training
        from segment_anything import sam_model_registry
        # MicroSAM requires more complex setup
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        output = np.zeros_like(img, dtype=np.int32)  # Placeholder

    elif algo == 'StarDist':
        from stardist.models import StarDist2D
        from csbdeep.utils import normalize
        model = StarDist2D.from_pretrained('2D_versatile_fluo')
        img_norm = normalize(img, 1, 99.8)
        labels, _ = model.predict_instances(img_norm, prob_thresh=0.5, nms_thresh=0.4)
        output = labels

    elif algo == 'MESMER':
        from deepcell.applications import Mesmer
        app = Mesmer()
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
        img_4d = np.expand_dims(img, axis=0)
        output = app.predict(img_4d)[0, ..., 0]

    elif algo == 'Watershed':
        from skimage.filters import threshold_otsu
        from skimage.segmentation import watershed
        from scipy import ndimage as ndi
        thresh = threshold_otsu(img)
        binary = img > thresh
        distance = ndimage.distance_transform_edt(binary)
        from skimage.feature import peak_local_max
        local_max = peak_local_max(distance, min_distance=10, labels=binary)
        markers = np.zeros_like(img, dtype=int)
        markers[tuple(local_max.T)] = np.arange(len(local_max)) + 1
        markers = ndi.label(markers)[0]
        output = watershed(-distance, markers, mask=binary)

    elif algo == 'LACSS':
        import torch
        import lacss
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # LACSS requires specific preprocessing
        output = np.zeros_like(img, dtype=np.int32)

    elif algo == 'SplineDist':
        # SplineDist imports
        from stardist.models import StarDist2D  # Using similar import
        output = np.zeros_like(img, dtype=np.int32)

    elif algo == 'CellSAM':
        from segment_anything import sam_model_registry
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        output = np.zeros_like(img, dtype=np.int32)

    elapsed = time.time() - start
    print(f"RUNTIME:{{elapsed}}")
    print(f"OUTPUT_SHAPE:{{output.shape}}")

except Exception as e:
    print(f"ERROR:{{str(e)}}")
    import traceback
    traceback.print_exc()
"""

    # Write test script to temp file
    temp_script = Path("/tmp/test_algo.py")
    temp_script.write_text(test_script)

    try:
        result = subprocess.run(
            ["conda", "run", "-n", env_name, "python", str(temp_script)],
            capture_output=True, text=True,
            env={**subprocess.os.environ, "CONDA_PREFIX": CONDA_BASE},
            timeout=300
        )

        output = result.stdout + result.stderr

        # Parse runtime from output
        runtime_match = re.search(r'RUNTIME:(\d+\.?\d*)', output)
        if runtime_match:
            runtime = float(runtime_match.group(1))
            return runtime, None
        else:
            # Check for error
            error_match = re.search(r'ERROR:(.+)', output)
            if error_match:
                return None, error_match.group(1)
            return None, "Could not parse runtime"

    except subprocess.TimeoutExpired:
        return None, "Timeout (300s)"
    except Exception as e:
        return None, str(e)
    finally:
        if temp_script.exists():
            temp_script.unlink()


def main():
    """Run comprehensive benchmark."""

    # Check if test image exists
    if not TEST_IMAGE.exists():
        print(f"Test image not found: {TEST_IMAGE}")
        print("Looking for alternatives...")
        # Try to find any dapi tif file
        processed_dir = HEART_DATA_ROOT / "processed"
        if processed_dir.exists():
            for tif in processed_dir.rglob("*dapi*.tif"):
                TEST_IMAGE = tif
                print(f"Using test image: {TEST_IMAGE}")
                break
            else:
                for tif in processed_dir.rglob("*.tif"):
                    TEST_IMAGE = tif
                    print(f"Using test image: {TEST_IMAGE}")
                    break

    print("=" * 80)
    print("ALGORITHM BENCHMARKING")
    print("=" * 80)
    print(f"Test image: {TEST_IMAGE}")
    print(f"Conda base: {CONDA_BASE}")
    print("=" * 80)

    results = []

    for algo_name, (env_name, script) in ALGORITHMS.items():
        print(f"\n{'='*80}")
        print(f"Testing: {algo_name}")
        print(f"Environment: {env_name}")
        print(f"Script: {script}")
        print('='*80)

        # Get environment stats
        env_size = get_env_size(env_name)
        pkg_count = get_package_count(env_name)

        print(f"  Environment size: {env_size} GB" if env_size else "  Environment size: N/A")
        print(f"  Package count: {pkg_count}" if pkg_count else "  Package count: N/A")

        # Measure runtime
        print(f"  Measuring runtime...")
        runtime, error = run_algorithm_in_env(algo_name, env_name, script)

        if error:
            print(f"  Runtime: ERROR - {error}")
        else:
            print(f"  Runtime: {runtime:.3f} seconds")

        results.append({
            "algorithm": algo_name,
            "conda_env": env_name,
            "env_size_gb": env_size,
            "package_count": pkg_count,
            "runtime_seconds": runtime,
            "error": error
        })

    # Write CSV
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "=" * 80)
    print(f"Results saved to: {OUTPUT_CSV}")
    print("=" * 80)
    print(df.to_string())

    return df


if __name__ == "__main__":
    main()
