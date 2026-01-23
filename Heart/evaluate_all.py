# heart/evaluate_all.py
"""
Evaluate all algorithms at the object and pixel level using ``stardist.matching``.
Computes recall, pixel recall, and missing rate.
"""
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from stardist.matching import matching

from _paths import HEART_DATA_ROOT

# Configuration
GT_DIR = HEART_DATA_ROOT / "ground_truth_masks"
RESULTS_BASE = HEART_DATA_ROOT / "benchmark_results"
OUTPUT_CSV = HEART_DATA_ROOT / "evaluation_results.csv"

# Algorithm list
ALGORITHMS = [
    'cellpose',
    'cellpose_sam',
    'stardist',
    'omnipose',
    'watershed',
    'mesmer',
    'lacss',
    'microsam',
    'cellsam',
    'splinedist',
    'instanseg',  # Newly added
]

def calculate_pixel_recall(gt_mask, pred_mask):
    """
    Compute pixel-level recall.

    Pixel Recall = TP pixels / (TP pixels + FN pixels)
    """
    gt_binary = (gt_mask > 0).astype(bool)
    pred_binary = (pred_mask > 0).astype(bool)
    
    tp_pixels = np.logical_and(gt_binary, pred_binary).sum()
    fn_pixels = np.logical_and(gt_binary, ~pred_binary).sum()
    
    if (tp_pixels + fn_pixels) == 0:
        return 0.0
    
    pixel_recall = tp_pixels / (tp_pixels + fn_pixels)
    return pixel_recall

def evaluate_single(gt_mask, pred_mask, iou_threshold=0.5):
    """
    Evaluate a single image pair.

    Returns a dict with:
        - n_gt: number of GT objects
        - n_pred: number of predicted objects
        - n_matched: number of matched objects
        - n_undetected: number of missing GT objects
        - object_recall: object-level recall
        - pixel_recall: pixel-level recall
        - missing_rate: missing rate
    """
    n_gt = len(np.unique(gt_mask)) - 1  # Exclude background label 0
    n_pred = len(np.unique(pred_mask)) - 1
    
    if n_gt == 0:
        return {
            'n_gt': 0,
            'n_pred': n_pred,
            'n_matched': 0,
            'n_undetected': 0,
            'object_recall': 0.0,
            'pixel_recall': 0.0,
            'missing_rate': 0.0
        }
    
    # Instance matching via stardist.matching
    matched = matching(gt_mask, pred_mask, thresh=iou_threshold)
    
    # matched includes tp/fp/fn statistics
    n_matched = matched.tp  # True positives (matched GT objects)
    n_undetected = matched.fn  # False negatives (missed GT objects)
    
    # Object-level Recall
    object_recall = n_matched / n_gt if n_gt > 0 else 0.0
    
    # Missing Rate
    missing_rate = n_undetected / n_gt if n_gt > 0 else 0.0
    
    # Pixel-level Recall
    pixel_recall = calculate_pixel_recall(gt_mask, pred_mask)
    
    return {
        'n_gt': n_gt,
        'n_pred': n_pred,
        'n_matched': n_matched,
        'n_undetected': n_undetected,
        'object_recall': object_recall,
        'pixel_recall': pixel_recall,
        'missing_rate': missing_rate
    }

def find_predictions(algo_name, region, area, channel):
    """
    Locate a prediction file for a specific region/area/channel.
    Handles multiple filename layouts.
    """
    pred_dir = RESULTS_BASE / f"{algo_name}_predictions" / region
    
    possible_names = [
        f"{area}_{channel}_pred.npy",      # LA1_dapi_pred.npy
        f"{channel}-{area}_pred.npy",      # dapi-LA1_pred.npy
        f"{area}_pred.npy"
    ]
    
    for name in possible_names:
        pred_path = pred_dir / name
        if pred_path.exists():
            return pred_path
    
    return None

def main():
    print("=" * 70)
    print("📊 Heart Dataset Evaluation")
    print("=" * 70)
    
    gt_mapping = pd.read_csv(GT_DIR / 'file_mapping.csv')
    
    print(f"\n📂 Ground Truth: {len(gt_mapping)} annotations")
    print(f"📂 Algorithms: {len(ALGORITHMS)}")
    print(f"📂 Algorithms: {', '.join(ALGORITHMS)}")
    
    all_results = []
    
    for algo_name in ALGORITHMS:
        print(f"\n{'='*70}")
        print(f"🔬 Evaluating: {algo_name}")
        print(f"{'='*70}")
        
        algo_dir = RESULTS_BASE / f"{algo_name}_predictions"
        if not algo_dir.exists():
            print(f"  ⚠️  Prediction directory not found, skipping...")
            continue
        
        n_evaluated = 0
        n_missing = 0
        
        for idx, row in tqdm(gt_mapping.iterrows(), 
                            total=len(gt_mapping),
                            desc=f"{algo_name}"):
            
            region = row['region']
            area = row['area']
            cell_type = row['cell_type']
            gt_path = Path(row['mask_absolute_path'])
            
            channel = f"{cell_type.lower()}"  # Customize if another mapping is needed
            
            # Find the corresponding prediction file
            pred_path = find_predictions(algo_name, region, area, 'dapi')
            
            if pred_path is None:
                n_missing += 1
                continue
            
            try:
                # Load masks
                gt_mask = np.load(gt_path)
                pred_mask = np.load(pred_path)
                
                # Evaluate
                metrics = evaluate_single(gt_mask, pred_mask, iou_threshold=0.5)
                
                # Attach metadata
                metrics.update({
                    'algorithm': algo_name,
                    'region': region,
                    'area': area,
                    'cell_type': cell_type,
                    'gt_path': str(gt_path),
                    'pred_path': str(pred_path)
                })
                
                all_results.append(metrics)
                n_evaluated += 1
                
            except Exception as e:
                print(f"\n  ✗ Failed {region}/{area}-{cell_type}: {e}")
                continue
        
        print(f"  ✓ Evaluated: {n_evaluated}/{len(gt_mapping)}")
        if n_missing > 0:
            print(f"  ⚠️  Missing predictions: {n_missing}")
    
    # Save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(OUTPUT_CSV, index=False)
        
        print("\n" + "=" * 70)
        print("📊 EVALUATION SUMMARY")
        print("=" * 70)
        
        # Summary by algorithm
        print("\n🔬 By Algorithm:")
        algo_summary = results_df.groupby('algorithm').agg({
            'object_recall': ['mean', 'std'],
            'pixel_recall': ['mean', 'std'],
            'missing_rate': ['mean', 'std'],
            'n_gt': 'sum',
            'n_matched': 'sum',
            'n_undetected': 'sum'
        }).round(4)
        print(algo_summary)
        
        # Summary by region
        print("\n🫀 By Region:")
        region_summary = results_df.groupby('region')[
            ['object_recall', 'pixel_recall', 'missing_rate']
        ].mean().round(4)
        print(region_summary)
        
        # Summary by cell type
        print("\n🧬 By Cell Type:")
        celltype_summary = results_df.groupby('cell_type')[
            ['object_recall', 'pixel_recall', 'missing_rate']
        ].mean().round(4)
        print(celltype_summary)
        
        print(f"\n💾 Results saved to: {OUTPUT_CSV}")
        print("=" * 70)
    else:
        print("\n❌ No results to save!")

if __name__ == "__main__":
    main()
