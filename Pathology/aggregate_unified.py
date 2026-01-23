"""
Fast Aggregation v2 - Optimized
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

from _paths import FULLDATA_ROOT

# ============================================================
# Configuration
# ============================================================
BASE_PATH = FULLDATA_ROOT
RAW_BASE = BASE_PATH / 'raw'
CLINICAL_FILE = BASE_PATH / 'ROSMAP_clinical_n69.csv'
OUTPUT_FILE = BASE_PATH / 'full_aggregation.csv'

MARKERS = ['GFAP', 'iba1', 'NeuN', 'Olig2', 'PECAM']
N_WORKERS = mp.cpu_count()  # Use all CPU cores

# ============================================================
# Fast helpers
# ============================================================
def detect_region(path_str):
    p = path_str.lower()
    return 'grey' if '/grey/' in p or '/gray/' in p else ('white' if '/white/' in p else 'unknown')

def fast_count(path):
    """Fastest possible cell count."""
    try:
        m = np.load(path, mmap_mode='r')  # Memory-mapped to avoid loading entire array
        return int(m.max())
    except:
        return None

def process_sample(args):
    marker, pid, sample, raw, refined, region, algo = args
    nuc = fast_count(f"{raw}/mask_nuc.npy") if raw else None
    pos = fast_count(f"{refined}/mask_refined.npy") if refined else None
    if nuc is None and pos is None:
        return None
    return (marker, pid, sample, region, algo, nuc, pos)

# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    # Region map
    print("Building region map...")
    region_map = {}
    for marker in MARKERS:
        for c0 in (RAW_BASE / marker).glob('**/*c0*-1040.tiff'):
            pid = next((p for p in c0.parts if p.isdigit() and len(p) >= 6), 'unknown')
            region_map[(marker, pid, c0.parent.name.replace('.tiff_files', ''))] = detect_region(str(c0))
    print(f"  {len(region_map)} mapped")

    # Collect tasks
    print("Collecting samples...")
    tasks = []
    for algo in ['cellpose', 'watershed']:
        raw_base = BASE_PATH / ('segmentation' if algo == 'cellpose' else 'segmentation_watershed')
        ref_base = BASE_PATH / ('segmentation_refined' if algo == 'cellpose' else 'segmentation_watershed_refined')
        
        samples = {}
        for base, idx in [(raw_base, 0), (ref_base, 1)]:
            if not base.exists():
                continue
            for m in MARKERS:
                md = base / m
                if not md.exists():
                    continue
                for pd_ in md.iterdir():
                    if not pd_.is_dir():
                        continue
                    for sd in pd_.iterdir():
                        if sd.is_dir():
                            k = (m, pd_.name, sd.name)
                            if k not in samples:
                                samples[k] = [None, None]
                            samples[k][idx] = str(sd)
        
        for (m, pid, sn), (raw, ref) in samples.items():
            tasks.append((m, pid, sn, raw, ref, region_map.get((m, pid, sn), 'unknown'), algo))
    
    print(f"  {len(tasks)} samples")

    # Process
    print(f"Processing ({N_WORKERS} workers)...")
    with mp.Pool(N_WORKERS) as pool:
        results = list(tqdm(pool.imap_unordered(process_sample, tasks, chunksize=100), total=len(tasks)))
    
    # Filter & convert
    data = [r for r in results if r]
    df = pd.DataFrame(data, columns=['marker', 'participant_id', 'sample_name', 'region', 'algorithm', 'nuc_count', 'marker_count'])
    print(f"Valid: {len(df)}")

    # Aggregate
    print("Aggregating...")
    rows = []
    for algo in ['cellpose', 'watershed']:
        da = df[df['algorithm'] == algo]
        for region in ['grey', 'white', 'pooled']:
            dr = da if region == 'pooled' else da[da['region'] == region]
            for pid, gp in dr.groupby('participant_id'):
                row = {'participant_id': pid, 'algorithm': algo, 'region': region}
                for m in MARKERS:
                    mg = gp[gp['marker'] == m]
                    if len(mg):
                        nuc = mg['nuc_count'].sum()
                        pos = mg['marker_count'].sum()
                        row[f'{m}_nuc_sum'] = nuc
                        row[f'{m}_positive_sum'] = pos
                        row[f'{m}_n_samples'] = len(mg)
                        if pd.notna(pos) and nuc > 0:
                            row[f'{m}_ratio'] = pos / nuc
                rows.append(row)
    
    df_agg = pd.DataFrame(rows)

    # Clinical merge
    if CLINICAL_FILE.exists():
        clin = pd.read_csv(CLINICAL_FILE)
        clin['participant_id'] = clin['projid'].astype(str)
        df_agg['participant_id'] = df_agg['participant_id'].astype(str)
        df_agg = df_agg.merge(clin, on='participant_id', how='left')

    df_agg.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ {OUTPUT_FILE}")
    print(f"   {df_agg.shape[0]} rows, {df_agg['participant_id'].nunique()} donors")
    print(df_agg.groupby(['algorithm', 'region']).size().unstack(fill_value=0))
