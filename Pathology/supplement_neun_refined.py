"""
Supplement missing 27 NeuN participants - refined version.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tifffile import imread
from skimage.filters import threshold_otsu
from scipy import ndimage as ndi
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from _paths import RAW_DATA_ROOT, SEGMENTATION_ROOT, SEGMENTATION_REFINED_ROOT

# Paths
base_path = RAW_DATA_ROOT / 'NeuN'
seg_base = SEGMENTATION_ROOT / 'NeuN'
refined_base = SEGMENTATION_REFINED_ROOT / 'NeuN'
refined_base.mkdir(parents=True, exist_ok=True)

# Missing participants
missing_participants = [
    '10208143', '10253148', '10271474', '10290427', '10394182',
    '10405008', '10436131', '10460587', '10473384', '10478041',
    '10510090', '10536568', '10551157', '10557081', '10577337',
    '10929637', '11165535', '11200645', '11259716', '11326252',
    '11345331', '11409232', '11430815', '11444465', '11453772',
    '11455530', '11460357'
]

print(f"Missing participants: {len(missing_participants)}")

# ============================================================
# Refinement function
# ============================================================
def refine_mask(mask, marker_img, min_area=100):
    ids = np.unique(mask)
    ids = ids[ids > 0]
    
    if ids.size == 0:
        return 0, 0, mask * 0
    
    union = marker_img[mask > 0]
    if union.size == 0:
        return 0, 0, mask * 0
    
    try:
        thr = threshold_otsu(union)
    except:
        thr = np.median(union)
    
    n_total = 0
    positive_ids = []
    
    for k in ids:
        cell_mask = (mask == k)
        area = int(cell_mask.sum())
        
        if area < min_area:
            continue
        
        n_total += 1
        mean_intensity = float(marker_img[cell_mask].mean())
        
        if mean_intensity >= thr:
            positive_ids.append(k)
    
    n_positive = len(positive_ids)
    
    refined = np.zeros_like(mask, dtype=np.int32)
    if positive_ids:
        out_mask = np.isin(mask, positive_ids)
        refined[out_mask] = ndi.label(out_mask)[0][out_mask]
    
    return n_total, n_positive, refined

# ============================================================
# Collect samples
# ============================================================
print("\nCollecting samples...")
all_samples = []

for pid in tqdm(missing_participants, desc="Scanning"):
    participant_seg_dir = seg_base / pid
    
    if not participant_seg_dir.exists():
        continue
    
    sample_dirs = [d for d in participant_seg_dir.iterdir() if d.is_dir()]
    
    for sample_dir in sample_dirs:
        mask_merged = sample_dir / 'mask_merged.npy'
        
        if not mask_merged.exists():
            continue
        
        sample_name = sample_dir.name
        
        # Find the corresponding raw files
        # Expected location: raw/NeuN/NeuN/{pid}/Gray|White/{sample_name}.tiff_files/
        # Try both Gray and White
        for subdir in ['Gray', 'White']:
            tiff_files_dir = base_path / 'NeuN' / pid / subdir / f'{sample_name}.tiff_files'
            
            if not tiff_files_dir.exists():
                continue
            
            # Locate c1 files (marker channel)
            # File pattern: {sample_name}_b0c1*-1040.tiff or {sample_name}_NeuN*-1040.tiff
            c1_files = list(tiff_files_dir.glob(f'{sample_name}_*c1*-1040.tiff'))
            neun_files = list(tiff_files_dir.glob(f'{sample_name}_NeuN*-1040.tiff'))
            
            c1_path = None
            if c1_files:
                c1_path = c1_files[0]
            elif neun_files:
                c1_path = neun_files[0]
            
            if c1_path:
                all_samples.append({
                    'participant_id': pid,
                    'sample_name': sample_name,
                    'mask_merged_path': mask_merged,
                    'c1_path': c1_path,
                    'output_dir': refined_base / pid / sample_name
                })
                break  # Found, no need to check other subdirs

print(f"\nTotal samples found: {len(all_samples)}")

# Statistics
participant_counts = {}
for s in all_samples:
    pid = s['participant_id']
    participant_counts[pid] = participant_counts.get(pid, 0) + 1

print("\nSamples per participant:")
found_count = 0
for pid in sorted(missing_participants):
    count = participant_counts.get(pid, 0)
    if count > 0:
        print(f"  {pid}: {count} samples ✓")
        found_count += 1
    else:
        print(f"  {pid}: 0 samples ✗")

print(f"\nParticipants with data: {found_count}/27")

if len(all_samples) == 0:
    print("\n⚠️  No samples found!")
    exit(1)

# ============================================================
# Process
# ============================================================
print("\n" + "="*60)
print("Starting refinement...")
print("="*60)

results = []
errors = []

pbar = tqdm(all_samples, desc="Refining")

for sample_info in pbar:
    try:
        output_dir = sample_info['output_dir']
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Skip if exists
        if (output_dir / 'mask_refined.npy').exists() and \
           (output_dir / 'statistics.csv').exists():
            df_stats = pd.read_csv(output_dir / 'statistics.csv')
            results.append({
                'marker': 'NeuN',
                'participant_id': sample_info['participant_id'],
                'sample_name': sample_info['sample_name'],
                'c0_path': '',
                'c1_path': str(sample_info['c1_path']),
                'out_dir': str(output_dir),
                'n_total_cells': df_stats['n_total_cells'].iloc[0],
                'n_marker_positive': df_stats['n_marker_positive'].iloc[0],
                'marker_positive_ratio_refined': df_stats['marker_positive_ratio_refined'].iloc[0]
            })
            pbar.set_postfix({'status': 'exists', 'pid': sample_info['participant_id']})
            continue
        
        # Load
        mask_merged = np.load(sample_info['mask_merged_path'])
        marker_img = imread(sample_info['c1_path'])
        
        if marker_img.ndim == 3:
            marker_img = marker_img[:, :, 0].astype(np.float32)
        else:
            marker_img = marker_img.astype(np.float32)
        
        # Refine
        n_total, n_positive, refined_mask = refine_mask(mask_merged, marker_img, min_area=100)
        
        ratio = n_positive / n_total if n_total > 0 else np.nan
        
        # Save
        np.save(output_dir / 'mask_refined.npy', refined_mask)
        
        stats = {
            'n_total_cells': n_total,
            'n_marker_positive': n_positive,
            'marker_positive_ratio_refined': ratio
        }
        df_stats = pd.DataFrame([stats])
        df_stats.to_csv(output_dir / 'statistics.csv', index=False)
        
        results.append({
            'marker': 'NeuN',
            'participant_id': sample_info['participant_id'],
            'sample_name': sample_info['sample_name'],
            'c0_path': '',
            'c1_path': str(sample_info['c1_path']),
            'out_dir': str(output_dir),
            'n_total_cells': n_total,
            'n_marker_positive': n_positive,
            'marker_positive_ratio_refined': ratio
        })
        
        pbar.set_postfix({
            'pid': sample_info['participant_id'],
            'n': len(results),
            'err': len(errors)
        })
        
    except Exception as e:
        errors.append({
            'sample': f"{sample_info['participant_id']}/{sample_info['sample_name']}",
            'error': str(e)
        })
        pbar.set_postfix({'status': 'ERROR', 'err': len(errors)})
        continue

# Save
df_results = pd.DataFrame(results)
results_file = refined_base.parent / 'file_mapping_neun_supplement_refined.csv'
df_results.to_csv(results_file, index=False)

print("\n" + "="*60)
print(f"✓ Completed!")
print("="*60)
print(f"  Processed: {len(df_results)}")
print(f"  Errors: {len(errors)}")
print(f"  Output: {results_file}")

if errors:
    print(f"\nErrors: {len(errors)}")
    for err in errors[:5]:
        print(f"  {err['sample']}: {err['error']}")

# Verify
refined_participants = set([d.name for d in refined_base.iterdir() if d.is_dir()])
print(f"\n✓ Participants in refined: {len(refined_participants)}")
print(f"  Newly added: {len(set(missing_participants) & refined_participants)}/27")

# Summary
print(f"\n✓ Refined ratio stats:")
if len(df_results) > 0:
    print(f"  Mean: {df_results['marker_positive_ratio_refined'].mean():.3f}")
    print(f"  Std: {df_results['marker_positive_ratio_refined'].std():.3f}")

print("="*60)
