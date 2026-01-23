"""
Add the missing 27 NeuN participants to the refined dataset.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

from _paths import SEGMENTATION_ROOT, SEGMENTATION_REFINED_ROOT

# Paths
seg_base = SEGMENTATION_ROOT / 'NeuN'
refined_base = SEGMENTATION_REFINED_ROOT / 'NeuN'
refined_base.mkdir(parents=True, exist_ok=True)

# Checkpoint
checkpoint_file = refined_base.parent / 'checkpoint_neun_supplement.json'

def load_checkpoint():
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            return set(json.load(f)['completed'])
    return set()

def save_checkpoint(completed):
    with open(checkpoint_file, 'w') as f:
        json.dump({'completed': list(completed), 'last_update': str(datetime.now())}, f)

# Missing 27 participants
missing_participants = [
    '10208143', '10253148', '10271474', '10290427', '10394182',
    '10405008', '10436131', '10460587', '10473384', '10478041',
    '10510090', '10536568', '10551157', '10557081', '10577337',
    '10929637', '11165535', '11200645', '11259716', '11326252',
    '11345331', '11409232', '11430815', '11444465', '11453772',
    '11455530', '11460357'
]

print(f"Missing participants to process: {len(missing_participants)}")

# Collect all samples that require processing
all_samples = []
for pid in missing_participants:
    participant_dir = seg_base / pid
    if not participant_dir.exists():
        print(f"⚠️  Participant {pid} not found in segmentation!")
        continue
    
    sample_dirs = [d for d in participant_dir.iterdir() if d.is_dir()]
    for sample_dir in sample_dirs:
        # Check required files
        mask_nuc = sample_dir / 'mask_nuc.npy'
        mask_marker = sample_dir / 'mask_marker.npy'
        mask_merged = sample_dir / 'mask_merged.npy'
        
        if all([mask_nuc.exists(), mask_marker.exists(), mask_merged.exists()]):
            all_samples.append({
                'participant_id': pid,
                'sample_name': sample_dir.name,
                'input_dir': sample_dir,
                'output_dir': refined_base / pid / sample_dir.name
            })

print(f"Total samples to refine: {len(all_samples)}")

# Count samples per participant
participant_counts = {}
for s in all_samples:
    pid = s['participant_id']
    participant_counts[pid] = participant_counts.get(pid, 0) + 1

print("\nSamples per participant:")
for pid, count in sorted(participant_counts.items()):
    print(f"  {pid}: {count} samples")

# Load checkpoint
completed = load_checkpoint()
remaining = [s for s in all_samples if str(s['input_dir']) not in completed]

print(f"\nProcessing status:")
print(f"  Total: {len(all_samples)}")
print(f"  Completed: {len(all_samples) - len(remaining)}")
print(f"  Remaining: {len(remaining)}")

if len(remaining) == 0:
    print("\n✓ All samples already refined!")
    exit(0)

# ============================================================
# Refinement Functions
# ============================================================
def refine_mask(mask):
    """Remove tiny noisy regions."""
    from scipy import ndimage
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels > 0]
    
    refined_mask = np.zeros_like(mask)
    min_size = 50  # Minimum cell size in pixels
    
    for label in unique_labels:
        cell_mask = (mask == label)
        if cell_mask.sum() >= min_size:
            refined_mask[cell_mask] = label
    
    return refined_mask

def compute_statistics(mask_nuc_refined, mask_marker_refined):
    """Compute statistics for refined masks."""
    n_total_cells = len(np.unique(mask_nuc_refined)) - 1  # Exclude background
    
    # Count marker-positive cells
    marker_positive_labels = set()
    for cell_label in np.unique(mask_nuc_refined):
        if cell_label == 0:
            continue
        cell_region = (mask_nuc_refined == cell_label)
        marker_overlap = mask_marker_refined[cell_region]
        if np.any(marker_overlap > 0):
            marker_positive_labels.add(cell_label)
    
    n_marker_positive = len(marker_positive_labels)
    
    return {
        'n_total_cells': n_total_cells,
        'n_marker_positive': n_marker_positive,
        'marker_positive_ratio': n_marker_positive / n_total_cells if n_total_cells > 0 else 0
    }

# ============================================================
# Process
# ============================================================
print("\n" + "="*60)
print("Starting refinement...")
print("="*60)

processed = 0
errors = []

pbar = tqdm(remaining, desc="Refining NeuN")

for sample_info in pbar:
    try:
        input_dir = sample_info['input_dir']
        output_dir = sample_info['output_dir']
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Skip if already exists
        if (output_dir / 'mask_nuc_refined.npy').exists() and \
           (output_dir / 'mask_marker_refined.npy').exists() and \
           (output_dir / 'statistics.csv').exists():
            completed.add(str(input_dir))
            pbar.set_postfix({'status': 'exists', 'errors': len(errors)})
            continue
        
        # Load masks
        mask_nuc = np.load(input_dir / 'mask_nuc.npy')
        mask_marker = np.load(input_dir / 'mask_marker.npy')
        
        # Refine
        mask_nuc_refined = refine_mask(mask_nuc)
        mask_marker_refined = refine_mask(mask_marker)
        
        # Compute statistics
        stats = compute_statistics(mask_nuc_refined, mask_marker_refined)
        
        # Save
        np.save(output_dir / 'mask_nuc_refined.npy', mask_nuc_refined)
        np.save(output_dir / 'mask_marker_refined.npy', mask_marker_refined)
        
        df_stats = pd.DataFrame([stats])
        df_stats.to_csv(output_dir / 'statistics.csv', index=False)
        
        # Update checkpoint
        completed.add(str(input_dir))
        processed += 1
        
        if processed % 100 == 0:
            save_checkpoint(completed)
        
        pbar.set_postfix({
            'pid': sample_info['participant_id'],
            'status': 'done',
            'errors': len(errors)
        })
        
    except Exception as e:
        errors.append({'sample': str(input_dir), 'error': str(e)})
        pbar.set_postfix({'status': 'error', 'errors': len(errors)})
        continue

# Final checkpoint
save_checkpoint(completed)

# ============================================================
# Summary
# ============================================================
print("\n" + "="*60)
print("✓ Refinement completed!")
print("="*60)
print(f"  Processed: {processed}")
print(f"  Errors: {len(errors)}")

if errors:
    print("\nErrors encountered:")
    for err in errors[:5]:
        print(f"  {err['sample']}: {err['error']}")
    if len(errors) > 5:
        print(f"  ... and {len(errors) - 5} more")

# Verify results
refined_participants = set([d.name for d in refined_base.iterdir() if d.is_dir()])
print(f"\n✓ Participants in refined directory now: {len(refined_participants)}")
print(f"  Expected: 40 + 27 = 67")

newly_added = set(missing_participants) & refined_participants
print(f"  Newly added: {len(newly_added)}/27")

if len(newly_added) < 27:
    still_missing = set(missing_participants) - refined_participants
    print(f"\n⚠️  Still missing {len(still_missing)} participants:")
    for pid in sorted(list(still_missing)):
        print(f"    {pid}")

print("="*60)
