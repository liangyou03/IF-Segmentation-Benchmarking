# IF-Segmentation-Benchmarking

Tools and reproducible scripts for benchmarking instance segmentation pipelines on large immunofluorescence (IF) datasets.  
The repository collects end-to-end workflows for:

- **Brain cohort** – multi-algorithm experiments (Cellpose, StarDist, MicroSAM, etc.), refinement utilities, and visualization notebooks.
- **Heart cohort** – data preparation, algorithm runners, morphology-based classification, and publication-ready figures.
- **Pathology cohort** – NeuN supplementation/aggregation, batch processing, and clinical correlation analyses.

All projects share the same `Data/` root so that absolute paths are no longer required when sharing or publishing the code.

---

## Repository Layout

```
Brain/        # Scripts to preprocess, segment, and evaluate the brain IF dataset
Heart/        # Heart pipeline (data prep, segmentation, evaluation, visualization)
Pathology/    # Pathology-specific aggregation and analysis scripts
Data/         # Expected location for raw and processed data assets (not tracked)
Usability/    # Utility scripts for environment profiling
Python.gitignore
```

Each top-level module now contains a `_paths.py` helper that discovers the repository root (via `.git`) and exposes path constants such as:

```python
from _paths import DATA_ROOT, IFIMAGE_ROOT, HEART_DATA_ROOT
```

Any new script should import these helpers instead of hard-coding `/ihome/...` or other machine-specific directories.

---

## Getting Started

1. **Clone the repository**

   ```bash
   git clone git@github.com:liangyou03/IF-Segmentation-Benchmarking.git
   cd IF-Segmentation-Benchmarking
   ```

2. **Create a Python environment**

   ```bash
   conda create -n if-benchmark python=3.10
   conda activate if-benchmark
   pip install -r requirements.txt  # create this file to match your setup
   ```

3. **Organize data under `Data/`**

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
   python Brain/cellsam/prediction_cyto.py
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

---

## Contributing

1. Fork and create a feature branch.
2. Add or update scripts using `_paths.py` for any filesystem references.
3. Run `python -m compileall <script>` or unit tests if provided.
4. Submit a pull request detailing the cohort(s) touched and any data assumptions.

---

## License

This project aggregates research utilities; please consult the LICENSE file (or project owners) before using the code in external products. Data under `Data/` is not tracked by Git and must be sourced according to your institutional agreements.
