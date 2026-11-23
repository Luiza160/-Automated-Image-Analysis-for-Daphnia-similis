#!/bin/bash
#SBATCH --job-name=fasterrcnn_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail


# --- caminhos (WORKDIR deve ser o mesmo do submit_train_fasterrcnn.sh) ---
CVAT_ZIP="${CVAT_ZIP:?Defina CVAT_ZIP com um caminho absoluto}"
WORKDIR="${WORKDIR:?Defina WORKDIR com um caminho absoluto}"

cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/training/faster_rcnn"

python evaluate_fasterrcnn.py \
    --workdir "$WORKDIR" \
    --checkpoint "$WORKDIR/fasterrcnn_run/best_model.pth" \
    --cvat-zip "$CVAT_ZIP" \
    --iou-threshold 0.5 \
    --score-threshold 0.5
