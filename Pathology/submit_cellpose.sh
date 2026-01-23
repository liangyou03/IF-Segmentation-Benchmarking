#!/bin/bash
#SBATCH --job-name=cellpose_seg
#SBATCH --output=cellpose_%j.log
#SBATCH --error=cellpose_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cluster=gpu
#SBATCH --partition=a100

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cellpose

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"
python batch_cellpose.py
