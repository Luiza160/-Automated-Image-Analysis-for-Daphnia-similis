#!/bin/bash
#SBATCH --job-name=yolo_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# --- ambiente ---
module load cuda/12.4
source ~/miniconda3/etc/profile.d/conda.sh
conda activate yolo_env

# --- paths ---
CVAT_ZIP="$HOME/datasets/dataset.zip"
WORKDIR="$HOME/daphnia_runs/yolo_run1"

cd "$HOME/daphnia-repo/training/yolo"

python train_yolo.py \
    --cvat-zip "$CVAT_ZIP" \
    --format coco \
    --workdir "$WORKDIR" \
    --epochs 200 \
    --patience 20 \
    --imgsz 640 \
    --batch 16 \
    --device 0 \
    --run-name yolo_run \
    --lr0 0.001 \
    --n-augmentations 2 \
    --val-fraction 0.15 \
    --test-fraction 0.15
