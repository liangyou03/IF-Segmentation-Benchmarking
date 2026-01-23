# heart/run_heart_benchmark.py
"""
Multi-algorithm benchmarking for the heart dataset leveraging the shared evaluation framework.
"""

import numpy as np
from pathlib import Path
import pandas as pd
import json
from tqdm import tqdm
import sys

# Add the Brain directory to sys.path to reuse evaluation utilities
REPO_ROOT = Path(__file__).resolve().parents[1]
BRAIN_DIR = REPO_ROOT / "Brain"
if str(BRAIN_DIR) not in sys.path:
    sys.path.append(str(BRAIN_DIR))
from evaluation_core import evaluate_segmentation
from config import METRICS

from _paths import HEART_DATA_ROOT

class HeartBenchmark:
    def __init__(self, raw_dir, gt_dir, output_base):
        self.raw_dir = Path(raw_dir)
        self.gt_dir = Path(gt_dir)
        self.output_base = Path(output_base)
        
        # Load ground-truth mapping
        self.mapping_df = pd.read_csv(gt_dir / 'file_mapping.csv')
        
        # Algorithms to benchmark
        self.algorithms = [
            'cellpose',
            'stardist', 
            'omnipose',
            'watershed',
            'mesmer',
            'cellsam'
        ]
        
    def get_output_dir(self, algo_name):
        """Return the output directory for a given algorithm."""
        algo_dir = self.output_base / f"heart_{algo_name}_benchmark"
        algo_dir.mkdir(parents=True, exist_ok=True)
        return algo_dir
    
    def load_image(self, image_path):
        """Load an image (supports multi-channel TIFF)."""
        from PIL import Image
        img = Image.open(image_path)
        
        # Convert multi-channel inputs to numpy arrays
        img_array = np.array(img)
        
        # Return the DAPI channel (assume first channel or grayscale)
        if img_array.ndim == 2:
            return img_array
        elif img_array.ndim == 3:
            return img_array  # Could be RGB or multi-channel
        else:
            raise ValueError(f"Unexpected image dimension: {img_array.ndim}")
    
    def run_cellpose(self, image, diameter=15):
        """Run the Cellpose nuclei model."""
        from cellpose import models
        
        model = models.Cellpose(gpu=True, model_type='nuclei')
        
        # Handle multi-channel inputs
        if image.ndim == 3:
            # Use the first channel (DAPI)
            image = image[..., 0] if image.shape[2] > 1 else image
        
        masks, flows, styles, diams = model.eval(
            image,
            diameter=diameter,
            channels=[0, 0],  # grayscale
            flow_threshold=0.4,
            cellprob_threshold=0.0
        )
        return masks
    
    def run_stardist(self, image):
        """Run StarDist."""
        from stardist.models import StarDist2D
        
        if image.ndim == 3:
            image = image[..., 0]
        
        model = StarDist2D.from_pretrained('2D_versatile_fluo')
        labels, _ = model.predict_instances(image, prob_thresh=0.5, nms_thresh=0.4)
        return labels
    
    def run_omnipose(self, image, diameter=15):
        """Run Omnipose."""
        from cellpose import models
        
        if image.ndim == 3:
            image = image[..., 0]
        
        model = models.Cellpose(gpu=True, model_type='bact_phase_omni')
        masks, flows, styles, diams = model.eval(
            image,
            diameter=diameter,
            channels=[0, 0],
            omni=True,
            flow_threshold=0.4,
            cellprob_threshold=0.0
        )
        return masks
    
    def run_watershed(self, image):
        """Run Watershed segmentation."""
        from skimage.filters import threshold_otsu
        from skimage.segmentation import watershed
        from skimage.feature import peak_local_max
        from scipy import ndimage as ndi
        
        if image.ndim == 3:
            image = image[..., 0]
        
        # Threshold the image
        thresh = threshold_otsu(image)
        binary = image > thresh
        
        # Distance transform
        distance = ndi.distance_transform_edt(binary)
        
        # Find peaks to use as markers
        local_max = peak_local_max(distance, min_distance=10, labels=binary)
        markers = np.zeros_like(image, dtype=int)
        markers[tuple(local_max.T)] = np.arange(len(local_max)) + 1
        markers = ndi.label(markers)[0]
        
        # Watershed segmentation
        labels = watershed(-distance, markers, mask=binary)
        return labels
    
    def run_mesmer(self, image):
        """Run Mesmer (DeepCell)."""
        try:
            from deepcell.applications import NuclearSegmentation
            
            if image.ndim == 2:
                image = np.expand_dims(image, axis=-1)
            if image.ndim == 3 and image.shape[2] > 1:
                image = image[..., 0:1]
            
            # Mesmer requires 4D input [batch, height, width, channels]
            image_4d = np.expand_dims(image, axis=0)
            
            app = NuclearSegmentation()
            masks = app.predict(image_4d, image_mpp=0.5)
            return masks[0, ..., 0]
        except Exception as e:
            print(f"    Mesmer failed: {e}")
            return None
    
    def run_cellsam(self, image):
        """Placeholder for CellSAM (add implementation if available)."""
        # Implementation depends on the available CellSAM version
        print("    CellSAM not implemented yet")
        return None
    
    def run_algorithm(self, algo_name, image, **kwargs):
        """Run the specified algorithm."""
        algo_map = {
            'cellpose': self.run_cellpose,
            'stardist': self.run_stardist,
            'omnipose': self.run_omnipose,
            'watershed': self.run_watershed,
            'mesmer': self.run_mesmer,
            'cellsam': self.run_cellsam
        }
        
        if algo_name not in algo_map:
            raise ValueError(f"Unknown algorithm: {algo_name}")
        
        return algo_map[algo_name](image, **kwargs)
    
    def run_predictions(self, algorithms=None):
        """Run predictions for all configured algorithms."""
        if algorithms is None:
            algorithms = self.algorithms
        
        results = []
        
        print("=" * 60)
        print("🔬 Running Heart Dataset Benchmark")
        print("=" * 60)
        
        for algo_name in algorithms:
            print(f"\n{'='*60}")
            print(f"🚀 Running {algo_name.upper()}")
            print(f"{'='*60}")
            
            algo_dir = self.get_output_dir(algo_name)
            pred_dir = algo_dir / 'predictions'
            
            # Create region subdirectories
            for region in ['LA', 'RA', 'LV', 'RV', 'SEP']:
                (pred_dir / region).mkdir(parents=True, exist_ok=True)
            
            for idx, row in tqdm(self.mapping_df.iterrows(), 
                                total=len(self.mapping_df),
                                desc=f"{algo_name}"):
                
                region = row['region']
                area = row['area']
                cell_type = row['cell_type']
                image_path = Path(row['image_absolute_path'])
                
                try:
                    # Load image
                    image = self.load_image(image_path)
                    
                    # Run algorithm
                    pred_mask = self.run_algorithm(algo_name, image)
                    
                    if pred_mask is None:
                        continue
                    
                    # Save prediction
                    output_path = pred_dir / region / f"{cell_type}-{area}_pred.npy"
                    np.save(output_path, pred_mask)
                    
                    results.append({
                        'algorithm': algo_name,
                        'region': region,
                        'area': area,
                        'cell_type': cell_type,
                        'image_path': str(image_path),
                        'gt_mask_path': row['mask_absolute_path'],
                        'pred_mask_path': str(output_path),
                        'n_gt_nuclei': row['n_nuclei'],
                        'n_pred_nuclei': len(np.unique(pred_mask)) - 1
                    })
                    
                except Exception as e:
                    print(f"\n  ✗ Failed: {region}/{area}-{cell_type}: {e}")
                    continue
        
        # Save prediction summary
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.output_base / 'heart_predictions_all.csv', index=False)
        
        print(f"\n✅ Predictions complete!")
        print(f"📊 Total predictions: {len(results_df)}")
        print(f"\n📈 Predictions by algorithm:")
        print(results_df.groupby('algorithm').size())
        
        return results_df
    
    def evaluate_all(self, predictions_csv=None):
        """Evaluate all prediction results."""
        if predictions_csv is None:
            predictions_csv = self.output_base / 'heart_predictions_all.csv'
        
        pred_df = pd.read_csv(predictions_csv)
        
        print("\n" + "=" * 60)
        print("📊 Evaluating Predictions")
        print("=" * 60)
        
        all_metrics = []
        
        for idx, row in tqdm(pred_df.iterrows(), total=len(pred_df)):
            try:
                # Load masks
                gt_mask = np.load(row['gt_mask_path'])
                pred_mask = np.load(row['pred_mask_path'])
                
                # Compute metrics using evaluation_core
                metrics = evaluate_segmentation(gt_mask, pred_mask, iou_threshold=0.5)
                
                metrics.update({
                    'algorithm': row['algorithm'],
                    'region': row['region'],
                    'area': row['area'],
                    'cell_type': row['cell_type'],
                    'n_gt_nuclei': row['n_gt_nuclei'],
                    'n_pred_nuclei': row['n_pred_nuclei']
                })
                
                all_metrics.append(metrics)
                
            except Exception as e:
                print(f"\n  ✗ Evaluation failed: {row['algorithm']}/{row['region']}/{row['area']}: {e}")
                continue
        
        # Save detailed metrics
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_csv(self.output_base / 'heart_evaluation_metrics.csv', index=False)
        
        # Generate summary report
        self.print_summary(metrics_df)
        
        return metrics_df
    
    def print_summary(self, metrics_df):
        """Print a high-level evaluation summary."""
        print("\n" + "=" * 60)
        print("📊 HEART DATASET BENCHMARK SUMMARY")
        print("=" * 60)
        
        print("\n🔬 Overall Performance by Algorithm:")
        algo_summary = metrics_df.groupby('algorithm')[
            ['precision', 'recall', 'f1_score', 'avg_iou']
        ].agg(['mean', 'std'])
        print(algo_summary)
        
        print("\n🫀 Performance by Region:")
        region_summary = metrics_df.groupby('region')[
            ['precision', 'recall', 'f1_score']
        ].mean()
        print(region_summary)
        
        print("\n🧬 Performance by Cell Type:")
        celltype_summary = metrics_df.groupby('cell_type')[
            ['precision', 'recall', 'f1_score']
        ].mean()
        print(celltype_summary)
        
        print("\n📈 Best Algorithm per Metric:")
        best_f1 = metrics_df.groupby('algorithm')['f1_score'].mean().idxmax()
        best_iou = metrics_df.groupby('algorithm')['avg_iou'].mean().idxmax()
        best_recall = metrics_df.groupby('algorithm')['recall'].mean().idxmax()
        
        print(f"  • Best F1-Score: {best_f1}")
        print(f"  • Best IoU: {best_iou}")
        print(f"  • Best Recall: {best_recall}")


def main():
    raw_dir = HEART_DATA_ROOT / "raw"
    gt_dir = HEART_DATA_ROOT / "ground_truth_masks"
    output_base = HEART_DATA_ROOT / "benchmark_results"
    benchmark = HeartBenchmark(raw_dir, gt_dir, output_base)
    algorithms = ['cellpose', 'stardist', 'omnipose', 'watershed']
    pred_df = benchmark.run_predictions(algorithms=algorithms)
    metrics_df = benchmark.evaluate_all()
    print("\n✅ Heart benchmark complete!")

if __name__ == "__main__":
    main()
