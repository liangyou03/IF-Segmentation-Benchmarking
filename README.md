# IF-Segmentation-Benchmarking

Tools and reproducible scripts for benchmarking instance segmentation pipelines on large immunofluorescence (IF) datasets.  
The repository collects end-to-end workflows for:

- **Brain cohort** – multi-algorithm experiments (Cellpose, StarDist, MicroSAM, etc.), refinement utilities, and visualization notebooks.
- **Heart cohort** – data preparation, algorithm runners, morphology-based classification, and publication-ready figures.
- **Pathology cohort** – NeuN supplementation/aggregation, batch processing, and clinical correlation analyses.

---

## Repository Layout

```
Brain/        # Scripts to preprocess, segment, and evaluate the brain IF dataset
Heart/        # Heart pipeline (data prep, segmentation, evaluation, visualization)
Pathology/    # Pathology-specific aggregation and analysis scripts
Data/         # Expected location for raw and processed data assets (download through link)
Usability/    # Utility scripts for environment profiling
Python.gitignore
```

Each top-level module now contains a `_paths.py` helper that discovers the repository root (via `.git`) and exposes path constants such as:

```python
from _paths import DATA_ROOT, IFIMAGE_ROOT, HEART_DATA_ROOT
```

---

All data used in this benchmarking can be found at [this link](https://drive.google.com/file/d/18X-1QAe5xseo5wJZaTeKlYdwkG3d7AXc/view?usp=drive_link).

```
Data/
├── ifimage/                   # shared IF dataset root
│   ├── heart/...
│   ├── 00_dataset_withoutpecam/...
│   └── evaluation_results/...
└── ... (custom folders per cohort)
```

Adjust the subdirectories to mirror the original storage layout. The helper modules assume `Data/ifimage/...`.

---

## Typical Workflows

### Brain
1. Configure paths in `Brain/config.py` (already pointing to `Data/ifimage/...`).
2. Run segmentation for a method, e.g.:

   ```bash
   python Brain/cellpose/prediction_cyto.py
   ```

3. Evaluate and visualize:

   ```bash
   python Brain/run_evaluation.py
   python Brain/panel_ploting.py
   ```

### Heart

1. Prepare channel-separated TIFFs:

   ```bash
   python Heart/prepare_chan_data.py
   ```

2. Launch any segmentation runner (`run_cellpose.py`, `run_mesmer.py`, etc.).
3. Evaluate and generate plots:

   ```bash
   python Heart/evaluate_all.py
   python Heart/visualize_results.py
   python Heart/visualize_academic_figure.py
   ```

### Pathology
1. Batch process NeuN refinements or watershed runs:

   ```bash
   python Pathology/batch_cellpose.py
   python Pathology/batch_watershed.py
   ```

2. Merge/aggregate donor-level statistics:

   ```bash
   python Pathology/merge_and_aggregate.py
   python Pathology/aggregate_unified.py
   ```
