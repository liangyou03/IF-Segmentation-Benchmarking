# Algorithm Environment Analysis

## Overview

This document provides a comprehensive analysis of conda environments used for segmentation algorithms in the ifimage project.

## Environment to Algorithm Mapping

Based on script analysis and slurm job configurations:

| Environment | Algorithm | Directory | GPU Required | Notes |
|-------------|-----------|-----------|--------------|-------|
| `ifimage-cpsam-gpu` | Cellpose-SAM | `01_cellpose_benchmark/` | Yes | PyTorch + CUDA 12.1 |
| `ifimage_stardist` | StarDist | `02_stardist_benchmark/` | Yes | TensorFlow based |
| `ifimage_cellsam` | CellSAM | `03_cellsam_benchmark/` | Yes | SAM-based |
| `deepcell_retinamask` | MESMER | `04_mesmer_benchmark/` | Yes | DeepCell/MESMER |
| `ifimage` | Watershed | `06_watershed_benchmark/` | No | Pure scikit-image |
| `cellpose` | Omnipose | `07_omnipose_benchmark/` | Yes | Cellpose variant |
| `lacss` | LACSS | `011_lacss/` | Yes | Learning-based |
| `micro-sam` | MicroSAM | `012_microsam_benchmark/` | Yes | Segment Anything |
| `ifimage` | InstanSeg | `013_ins/` | Yes | Fluorescence model |
| `ifimage_evaluation` | Evaluation | - | No | Metrics calculation |

## Environment Details

### 1. ifimage-cpsam-gpu (Cellpose-SAM)

**Purpose**: Cellpose-SAM segmentation (primary method)

**Dependencies** (from env.yml):
```yaml
- python=3.11
- pytorch + torchvision + pytorch-cuda=12.1
- numpy, scipy, scikit-image, numba
- tifffile, imagecodecs, opencv
- natsort, fastremap, tqdm
- pip: cellpose>=4.0, huggingface_hub
```

**Estimated size**: ~5-8 GB (includes PyTorch + CUDA)

**Algorithms**:
- Cellpose-SAM (2-channel nuclei + cell)
- Cellpose (nuclei only)
- Cellpose (marker only)

---

### 2. ifimage_stardist (StarDist)

**Purpose**: StarDist segmentation with star-convex polygons

**Dependencies**:
```python
- stardist (TensorFlow based)
- csbdeep
- numpy, scipy, scikit-image
- tifffile
```

**Estimated size**: ~3-5 GB (includes TensorFlow)

**Algorithms**:
- StarDist (2-channel nuclei + cell)
- StarDist (nuclei only)
- StarDist (marker only)

---

### 3. ifimage_cellsam (CellSAM)

**Purpose**: CellSAM (SAM-based cell segmentation)

**Dependencies**:
- PyTorch based
- cellpose/segment-anything

**Estimated size**: ~4-6 GB

**Algorithms**:
- CellSAM (2-channel nuclei + cell)
- CellSAM (marker only)

---

### 4. deepcell_retinamask (MESMER)

**Purpose**: MESMER from DeepCell for multiplexed images

**Dependencies**:
```python
- deepcell
- tensorflow
```

**Estimated size**: ~3-5 GB

**Algorithms**:
- MESMER (2-channel nuclei + cell)
- MESMER (marker only)

---

### 5. ifimage (Watershed, InstanSeg)

**Purpose**: Classical Watershed and InstanSeg

**Dependencies**:
```python
- scikit-image
- numpy, scipy
- torch (for InstanSeg)
- instanseg
```

**Estimated size**: ~2-4 GB

**Algorithms**:
- Watershed (classical, no ML)
- InstanSeg (PyTorch based)

---

### 6. cellpose (Omnipose)

**Purpose**: Omnipose (improved Cellpose for diverse shapes)

**Dependencies**:
- cellpose
- omnipose

**Estimated size**: ~3-5 GB

**Algorithms**:
- Omnipose (nuclei + cell)
- Omnipose (marker only)

---

### 7. lacss (LACSS)

**Purpose**: Learning-based cell segmentation

**Dependencies**:
```python
- lacss
- tensorflow/pytorch
```

**Estimated size**: ~3-5 GB

**Algorithms**:
- LACSS (2-channel nuclei + cell)
- LACSS (marker only)

---

### 8. micro-sam (MicroSAM)

**Purpose**: Segment Anything Model for microscopy

**Dependencies**:
```python
- segment-anything
- torch
- micro-sam
```

**Estimated size**: ~5-8 GB (includes SAM models)

**Algorithms**:
- MicroSAM (nuclei + cell)
- MicroSAM (marker only)

---

### 9. ifimage_evaluation (Evaluation)

**Purpose**: Metrics calculation and evaluation

**Dependencies**:
```python
- numpy, scipy, pandas
- scikit-image
- tqdm
```

**Estimated size**: ~1-2 GB

**Usage**: All evaluation scripts

---

## Disk Usage Summary

| Environment | Estimated Size | With Models | Notes |
|-------------|---------------|-------------|-------|
| ifimage-cpsam-gpu | 5-8 GB | 6-10 GB | Largest (PyTorch + CUDA) |
| ifimage_stardist | 3-5 GB | 4-6 GB | TensorFlow based |
| ifimage_cellsam | 4-6 GB | 5-7 GB | SAM-based |
| deepcell_retinamask | 3-5 GB | 4-6 GB | MESMER |
| ifimage | 2-4 GB | 3-5 GB | Watershed + InstanSeg |
| cellpose | 3-5 GB | 4-6 GB | Omnipose |
| lacss | 3-5 GB | 4-6 GB | LACSS |
| micro-sam | 5-8 GB | 7-12 GB | SAM models are large |
| ifimage_evaluation | 1-2 GB | 1-2 GB | Evaluation only |

**Total estimated**: ~30-55 GB for all environments

---

## Command Reference

### Get disk usage of each environment:
```bash
# For conda environments
du -sh ~/miniconda3/envs/ifimage-*

# Or for mamba
mamba env list
du -sh ~/micromamba/envs/*
```

### List packages in each environment:
```bash
conda activate ifimage-cpsam-gpu
conda list --export > packages_ifimage_cpsam_gpu.txt

conda activate ifimage_stardist
conda list --export > packages_ifimage_stardist.txt
```
