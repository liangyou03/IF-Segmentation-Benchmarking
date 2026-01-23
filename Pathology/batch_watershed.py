"""
Batch Watershed segmentation for all markers (PARALLEL VERSION)
GFAP, iba1, NeuN, Olig2, PECAM

Features:
- Multiprocessing for speed
- Checkpoint/resume support
- Real-time progress tracking
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tifffile import imread
from tqdm import tqdm
import json
import time
from datetime import datetime, timedelta
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import filelock

from _paths import RAW_DATA_ROOT, SEGMENTATION_WATERSHED_ROOT

from scipy import ndimage as ndi
from scipy.ndimage import binary_closing, gaussian_filter
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.morphology import disk, remove_small_objects, remove_small_holes
from skimage.measure import label
from skimage.filters import threshold_otsu

# ============================================================
# Parameters
# ============================================================
NUC_SIGMA        = 1.0
NUC_MIN_SIZE     = 60
NUC_MIN_DISTANCE = 10

CYTO_SIGMA       = 1.5
FG_CLOSE_RADIUS  = 3
FG_MIN_HOLE_AREA = 64
COMPACTNESS      = 0.001

N_WORKERS = max(1, mp.cpu_count() - 2)  # leave 2 cores for system

# ============================================================
# Paths
# ============================================================
base_path = RAW_DATA_ROOT
out_base = SEGMENTATION_WATERSHED_ROOT
out_base.mkdir(parents=True, exist_ok=True)

checkpoint_file = out_base / 'checkpoint.json'
checkpoint_lock = out_base / 'checkpoint.lock'
error_log = out_base / 'errors.log'

markers = ['GFAP', 'iba1', 'NeuN', 'Olig2', 'PECAM']

# ============================================================
# Checkpoint (thread-safe)
# ============================================================
def load_checkpoint():
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            return set(json.load(f)['completed'])
    return set()

def save_checkpoint_atomic(completed_set):
    """Thread-safe checkpoint save."""
    lock = filelock.FileLock(str(checkpoint_lock))
    with lock:
        with open(checkpoint_file, 'w') as f:
            json.dump({'completed': list(completed_set), 'last_update': str(datetime.now())}, f)

def add_to_checkpoint(file_path):
    """Add single file to checkpoint (thread-safe)."""
    lock = filelock.FileLock(str(checkpoint_lock))
    with lock:
        completed = load_checkpoint()
        completed.add(file_path)
        with open(checkpoint_file, 'w') as f:
            json.dump({'completed': list(completed), 'last_update': str(datetime.now())}, f)

def log_error(msg):
    lock = filelock.FileLock(str(error_log) + '.lock')
    with lock:
        with open(error_log, 'a') as f:
            f.write(f"{datetime.now()} - {msg}\n")

# ============================================================
# Segmentation Functions
# ============================================================
def segment_nuclei(img_nuc):
    """Watershed segmentation for nuclei."""
    if img_nuc.ndim == 3:
        img_nuc = img_nuc[:, :, 0]
    
    smoothed = gaussian_filter(img_nuc.astype(np.float32), sigma=NUC_SIGMA)
    
    try:
        thresh = threshold_otsu(smoothed)
    except ValueError:
        thresh = smoothed.mean()
    
    binary = smoothed > thresh
    binary = remove_small_objects(binary, min_size=NUC_MIN_SIZE)
    binary = remove_small_holes(binary, area_threshold=FG_MIN_HOLE_AREA)
    
    distance = ndi.distance_transform_edt(binary)
    coords = peak_local_max(distance, min_distance=NUC_MIN_DISTANCE, labels=binary)
    
    mask_peaks = np.zeros(distance.shape, dtype=bool)
    mask_peaks[tuple(coords.T)] = True
    markers = label(mask_peaks)
    
    labels = watershed(-distance, markers, mask=binary, compactness=COMPACTNESS)
    return labels.astype(np.int32)


def segment_marker(img_marker):
    """Watershed segmentation for marker (simplified, no nuclear seeds)."""
    if img_marker.ndim == 3:
        img_marker = img_marker[:, :, 0]
    
    smoothed = gaussian_filter(img_marker.astype(np.float32), sigma=CYTO_SIGMA)
    
    try:
        thresh = threshold_otsu(smoothed)
    except ValueError:
        thresh = smoothed.mean()
    
    binary = smoothed > thresh
    selem = disk(FG_CLOSE_RADIUS)
    binary = binary_closing(binary, structure=selem)
    binary = remove_small_holes(binary, area_threshold=FG_MIN_HOLE_AREA)
    binary = remove_small_objects(binary, min_size=NUC_MIN_SIZE)
    
    distance = ndi.distance_transform_edt(binary)
    coords = peak_local_max(distance, min_distance=NUC_MIN_DISTANCE, labels=binary)
    
    mask_peaks = np.zeros(distance.shape, dtype=bool)
    mask_peaks[tuple(coords.T)] = True
    markers = label(mask_peaks)
    
    labels = watershed(-distance, markers, mask=binary, compactness=COMPACTNESS)
    return labels.astype(np.int32)


# ============================================================
# Worker Function
# ============================================================
def find_c1_file(c0_path):
    """Find matching c1 file in same directory."""
    c0_path = Path(c0_path)
    parent = c0_path.parent
    # Look for *c1* file in same directory
    c1_files = list(parent.glob('*c1*-1040.tiff'))
    if c1_files:
        return c1_files[0]
    return None


def process_single_file(args):
    """Process a single file pair. Returns (success, file_path, error_msg)."""
    marker, c0_path, _, out_dir = args  # ignore passed c1_path, find it ourselves
    c0_path = Path(c0_path)
    out_dir = Path(out_dir)
    
    try:
        # Skip if output exists
        if (out_dir / 'mask_nuc.npy').exists() and (out_dir / 'mask_marker.npy').exists():
            return (True, str(c0_path), None)
        
        # Find c1 file
        c1_path = find_c1_file(c0_path)
        if c1_path is None or not c1_path.exists():
            return (False, str(c0_path), "No c1 pair")
        
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Read images
        img_nuc = imread(c0_path)
        img_marker = imread(c1_path)
        
        # Segment
        mask_nuc = segment_nuclei(img_nuc)
        mask_marker = segment_marker(img_marker)
        
        # Save (skip merged to save time)
        np.save(out_dir / 'mask_nuc.npy', mask_nuc)
        np.save(out_dir / 'mask_marker.npy', mask_marker)
        
        return (True, str(c0_path), None)
        
    except Exception as e:
        return (False, str(c0_path), str(e))


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("="*60)
    print(f"Parallel Watershed Segmentation ({N_WORKERS} workers)")
    print("="*60)
    
    # Load checkpoint
    completed = load_checkpoint()
    print(f"✓ Checkpoint: {len(completed)} files already done")
    
    # Pre-run check
    print("\n📊 Pre-run file discovery:")
    for marker in markers:
        marker_path = base_path / marker
        c0_files = list(marker_path.glob('**/*c0*-1040.tiff'))
        pids = set()
        for c0 in c0_files:
            for p in c0.parts:
                if p.isdigit() and len(p) >= 6:
                    pids.add(p)
                    break
        print(f"  {marker}: {len(c0_files)} files, {len(pids)} participants")
    
    # Gather all tasks
    all_tasks = []
    
    for marker in markers:
        marker_path = base_path / marker
        c0_files = list(marker_path.glob('**/*c0*-1040.tiff'))
        
        for c0 in c0_files:
            if str(c0) in completed:
                continue
            
            c1 = Path(str(c0).replace('c0', 'c1'))
            
            # Extract participant ID
            participant_id = 'unknown'
            for p in c0.parts:
                if p.isdigit() and len(p) >= 6:
                    participant_id = p
                    break
            
            sample_name = c0.parent.name.replace('.tiff_files', '')
            out_dir = out_base / marker / participant_id / sample_name
            
            all_tasks.append((marker, str(c0), str(c1), str(out_dir)))
    
    total_files = len(completed) + len(all_tasks)
    print(f"\n{'='*60}")
    print(f"Total: {total_files} | Done: {len(completed)} | Remaining: {len(all_tasks)}")
    print(f"{'='*60}\n")
    
    if not all_tasks:
        print("✓ All files already processed!")
        exit(0)
    
    # Process in parallel
    start_time = time.time()
    processed = 0
    errors = 0
    
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(process_single_file, task): task for task in all_tasks}
        
        pbar = tqdm(as_completed(futures), total=len(all_tasks), 
                    desc="Segmenting", 
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')
        
        for future in pbar:
            success, file_path, error_msg = future.result()
            
            if success:
                add_to_checkpoint(file_path)
                processed += 1
            else:
                log_error(f"{file_path}: {error_msg}")
                errors += 1
            
            pbar.set_postfix({'done': processed, 'err': errors})
    
    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✓ Completed!")
    print(f"  Processed: {processed}")
    print(f"  Errors: {errors}")
    print(f"  Time: {timedelta(seconds=int(elapsed))}")
    print(f"  Speed: {processed/elapsed:.1f} files/sec")
    print(f"  Output: {out_base}")
    print(f"{'='*60}")
