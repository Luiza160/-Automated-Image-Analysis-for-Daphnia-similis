#!/bin/bash
#SBATCH --job-name=fasterrcnn_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate fasterrcnn_env

# --- caminhos (WORKDIR deve ser o mesmo do submit_train_fasterrcnn.sh) ---
CVAT_ZIP="$HOME/datasets/dataset.zip"
WORKDIR="$HOME/daphnia_runs/fasterrcnn_run1"

cd "$HOME/daphnia-repo/training/faster_rcnn"

python evaluate_fasterrcnn.py \
    --workdir "$WORKDIR" \
    --checkpoint "$WORKDIR/fasterrcnn_run/best_model.pth" \
    --cvat-zip "$CVAT_ZIP" \
    --iou-threshold 0.5 \
    --score-threshold 0.5
