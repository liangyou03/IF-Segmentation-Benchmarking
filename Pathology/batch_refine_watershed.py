"""
Batch refinement for Watershed segmentation (PARALLEL VERSION)

Logic:
1. Load mask (mask_merged or mask_marker)
2. Load marker intensity image (c1)
3. For each cell, compute mean intensity
4. Compute Otsu threshold on the distribution of mean intensities (mask-wise)
5. Keep only cells with mean intensity >= Otsu threshold
6. Save refined mask and statistics
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime, timedelta
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import filelock

from _paths import (
    RAW_DATA_ROOT,
    SEGMENTATION_WATERSHED_ROOT,
    SEGMENTATION_WATERSHED_REFINED_ROOT,
)

from skimage.filters import threshold_otsu
from scipy import ndimage as ndi
from tifffile import imread

# ============================================================
# Parameters
# ============================================================
MIN_CELL_AREA = 50
N_WORKERS = min(60, mp.cpu_count() - 2)

# ============================================================
# Paths
# ============================================================
seg_base = SEGMENTATION_WATERSHED_ROOT
raw_base = RAW_DATA_ROOT
refined_base = SEGMENTATION_WATERSHED_REFINED_ROOT
refined_base.mkdir(parents=True, exist_ok=True)

checkpoint_file = refined_base / 'checkpoint.json'
checkpoint_lock = refined_base / 'checkpoint.lock'
error_log = refined_base / 'errors.log'
raw_mapping_file = refined_base / 'raw_image_mapping.csv'

markers = ['GFAP', 'iba1', 'NeuN', 'Olig2', 'PECAM']

# ============================================================
# Checkpoint (thread-safe)
# ============================================================
def load_checkpoint():
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            return set(json.load(f)['completed'])
    return set()

def add_to_checkpoint(file_path):
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
# Build raw image mapping (one-time)
# ============================================================
def build_raw_mapping():
    """Build mapping from (marker, participant_id, sample_name) -> (c0_path, c1_path)"""
    
    if raw_mapping_file.exists():
        print(f"Loading existing raw mapping from {raw_mapping_file}...")
        df = pd.read_csv(raw_mapping_file)
        mapping = {}
        for _, row in df.iterrows():
            key = (row['marker'], str(row['participant_id']), row['sample_name'])
            mapping[key] = (row['c0_path'], row['c1_path'] if pd.notna(row['c1_path']) else None)
        print(f"  Loaded {len(mapping)} mappings")
        return mapping
    
    print("Building raw image mapping (one-time operation)...")
    records = []
    
    for marker in markers:
        marker_path = raw_base / marker
        print(f"  Scanning {marker}...")
        c0_files = list(marker_path.glob('**/*c0*-1040.tiff'))
        
        for c0 in tqdm(c0_files, desc=f"  {marker}", leave=False):
            # Extract participant ID
            participant_id = 'unknown'
            for p in c0.parts:
                if p.isdigit() and len(p) >= 6:
                    participant_id = p
                    break
            
            sample_name = c0.parent.name.replace('.tiff_files', '')
            
            # Find c1 in same directory
            c1_files = list(c0.parent.glob('*c1*-1040.tiff'))
            c1_path = str(c1_files[0]) if c1_files else None
            
            records.append({
                'marker': marker,
                'participant_id': participant_id,
                'sample_name': sample_name,
                'c0_path': str(c0),
                'c1_path': c1_path
            })
    
    df = pd.DataFrame(records)
    df.to_csv(raw_mapping_file, index=False)
    print(f"  Saved mapping to {raw_mapping_file}")
    
    mapping = {}
    for _, row in df.iterrows():
        key = (row['marker'], str(row['participant_id']), row['sample_name'])
        mapping[key] = (row['c0_path'], row['c1_path'])
    
    print(f"  Total mappings: {len(mapping)}")
    return mapping

# ============================================================
# Refinement Function
# ============================================================
def refine_mask_by_intensity(mask, intensity_img, min_area=MIN_CELL_AREA):
    """
    Refine mask using intensity-based Otsu filtering.
    
    Steps:
    1. For each cell, compute mean intensity
    2. Compute Otsu threshold on the distribution of mean intensities
    3. Keep cells with mean intensity >= threshold
    
    Args:
        mask: labeled segmentation mask
        intensity_img: intensity image (same shape as mask)
        min_area: minimum cell area to consider
    
    Returns:
        n_total: total cells (area >= min_area)
        n_positive: cells passing Otsu threshold
        refined_mask: mask with only positive cells
    """
    # Handle multi-channel images
    if intensity_img.ndim == 3:
        intensity_img = intensity_img[:, :, 0].astype(np.float32)
    else:
        intensity_img = intensity_img.astype(np.float32)
    
    # Get unique cell IDs
    cell_ids = np.unique(mask)
    cell_ids = cell_ids[cell_ids > 0]
    
    if cell_ids.size == 0:
        return 0, 0, np.zeros_like(mask)
    
    # Compute mean intensity for each cell
    cell_intensities = []
    valid_cells = []
    
    for cell_id in cell_ids:
        cell_mask = (mask == cell_id)
        area = cell_mask.sum()
        
        if area < min_area:
            continue
        
        mean_intensity = intensity_img[cell_mask].mean()
        cell_intensities.append(mean_intensity)
        valid_cells.append(cell_id)
    
    n_total = len(valid_cells)
    
    if n_total == 0:
        return 0, 0, np.zeros_like(mask)
    
    # Compute Otsu threshold on mean intensities (mask-wise)
    cell_intensities = np.array(cell_intensities)
    
    try:
        # Otsu on the distribution of cell mean intensities
        thr = threshold_otsu(cell_intensities)
    except:
        # Fallback to median if Otsu fails
        thr = np.median(cell_intensities)
    
    # Select cells with mean intensity >= threshold
    positive_ids = [cid for cid, intensity in zip(valid_cells, cell_intensities) 
                    if intensity >= thr]
    
    n_positive = len(positive_ids)
    
    # Create refined mask with relabeled IDs
    refined_mask = np.zeros_like(mask, dtype=np.int32)
    for new_label, old_id in enumerate(positive_ids, start=1):
        refined_mask[mask == old_id] = new_label
    
    return n_total, n_positive, refined_mask


# ============================================================
# Worker Function
# ============================================================
def process_single_sample(args):
    """Process a single sample. Returns (success, input_dir, error_msg)."""
    input_dir, output_dir, c0_path, c1_path, marker, pid, sample_name = args
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    try:
        # Skip if already exists
        if (output_dir / 'mask_refined.npy').exists() and \
           (output_dir / 'statistics.csv').exists():
            return (True, str(input_dir), None)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load merged mask (preferred) or marker mask
        mask_file = input_dir / 'mask_merged.npy'
        if not mask_file.exists():
            mask_file = input_dir / 'mask_marker.npy'
        if not mask_file.exists():
            return (False, str(input_dir), "No mask file found")
        
        mask = np.load(mask_file)
        
        # Load marker intensity image (c1)
        if not c1_path or not Path(c1_path).exists():
            return (False, str(input_dir), "No c1 image found")
        
        marker_img = imread(c1_path)
        
        # Refine using marker intensity
        n_total, n_positive, refined_mask = refine_mask_by_intensity(
            mask, marker_img, min_area=MIN_CELL_AREA
        )
        
        # Calculate ratio
        ratio = n_positive / n_total if n_total > 0 else 0.0
        
        # Save refined mask
        np.save(output_dir / 'mask_refined.npy', refined_mask)
        
        # Save statistics
        stats = {
            'marker': marker,
            'participant_id': pid,
            'sample_name': sample_name,
            'n_total_cells': n_total,
            'n_marker_positive': n_positive,
            'marker_positive_ratio': ratio
        }
        pd.DataFrame([stats]).to_csv(output_dir / 'statistics.csv', index=False)
        
        return (True, str(input_dir), None)
        
    except Exception as e:
        return (False, str(input_dir), str(e))


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("="*60)
    print(f"Parallel Watershed Refinement ({N_WORKERS} workers)")
    print("="*60)
    print(f"\nLogic:")
    print(f"  1. Load mask (merged or marker)")
    print(f"  2. Load marker intensity image (c1)")
    print(f"  3. Compute mean intensity per cell")
    print(f"  4. Otsu threshold on mean intensities (mask-wise)")
    print(f"  5. Keep cells with intensity >= threshold")
    print(f"\nMin cell area: {MIN_CELL_AREA} px")
    
    # Build raw image mapping first
    raw_mapping = build_raw_mapping()
    
    completed = load_checkpoint()
    print(f"\n✓ Checkpoint: {len(completed)} samples already refined")
    
    # Gather all samples
    print("\n📊 Gathering samples...")
    all_tasks = []
    
    for marker in markers:
        marker_seg_dir = seg_base / marker
        
        if not marker_seg_dir.exists():
            print(f"  ⚠️  {marker}: not found")
            continue
        
        count = 0
        for pid_dir in marker_seg_dir.iterdir():
            if not pid_dir.is_dir():
                continue
            pid = pid_dir.name
            
            for sample_dir in pid_dir.iterdir():
                if not sample_dir.is_dir():
                    continue
                
                if str(sample_dir) in completed:
                    continue
                
                # Check if any mask exists
                has_mask = (sample_dir / 'mask_merged.npy').exists() or \
                           (sample_dir / 'mask_marker.npy').exists()
                
                if not has_mask:
                    continue
                
                sample_name = sample_dir.name
                output_dir = refined_base / marker / pid / sample_name
                
                # Look up raw images from pre-built mapping
                key = (marker, pid, sample_name)
                c0_path, c1_path = raw_mapping.get(key, (None, None))
                
                all_tasks.append((
                    str(sample_dir), str(output_dir), c0_path, c1_path,
                    marker, pid, sample_name
                ))
                count += 1
        
        print(f"  {marker}: {count} samples to refine")
    
    print(f"\n{'='*60}")
    print(f"Total remaining: {len(all_tasks)}")
    print(f"{'='*60}\n")
    
    if not all_tasks:
        print("✓ All samples already refined!")
        exit(0)
    
    # Process
    start_time = time.time()
    processed = 0
    errors = 0
    
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(process_single_sample, task): task for task in all_tasks}
        
        pbar = tqdm(as_completed(futures), total=len(all_tasks),
                    desc="Refining",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')
        
        for future in pbar:
            success, input_dir, error_msg = future.result()
            
            if success:
                add_to_checkpoint(input_dir)
                processed += 1
            else:
                log_error(f"{input_dir}: {error_msg}")
                errors += 1
            
            pbar.set_postfix({'done': processed, 'err': errors})
    
    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✓ Completed!")
    print(f"  Processed: {processed}")
    print(f"  Errors: {errors}")
    print(f"  Time: {timedelta(seconds=int(elapsed))}")
    if elapsed > 0:
        print(f"  Speed: {processed/elapsed:.1f} samples/sec")
    print(f"{'='*60}")
    
    # Verify
    print("\n📊 Final verification:")
    for marker in markers:
        marker_dir = refined_base / marker
        if marker_dir.exists():
            pids = [d.name for d in marker_dir.iterdir() if d.is_dir()]
            n_samples = sum(len(list((marker_dir / p).iterdir())) for p in pids)
            print(f"  {marker}: {n_samples} samples, {len(pids)} participants")
