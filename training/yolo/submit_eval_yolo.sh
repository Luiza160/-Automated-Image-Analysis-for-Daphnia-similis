#!/bin/bash
#SBATCH --job-name=yolo_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate yolo_env

# --- paths (devem bater com o submit_train_yolo.sh) ---
CVAT_ZIP="$HOME/datasets/dataset.zip"
WORKDIR="$HOME/daphnia_runs/yolo_run1"

cd "$HOME/daphnia-repo/training/yolo"

python evaluate_yolo.py \
    --workdir "$WORKDIR" \
    --checkpoint "$WORKDIR/yolo_run/weights/best_real.pt" \
    --cvat-zip "$CVAT_ZIP" \
    --iou-threshold 0.5 \
    --score-threshold 0.5
