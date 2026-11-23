#!/bin/bash
#SBATCH --job-name=yolo_train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# --- ambiente ---

# --- paths ---
CVAT_ZIP="${CVAT_ZIP:?Defina CVAT_ZIP com um caminho absoluto}"
WORKDIR="${WORKDIR:?Defina WORKDIR com um caminho absoluto}"

cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/training/yolo"

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
