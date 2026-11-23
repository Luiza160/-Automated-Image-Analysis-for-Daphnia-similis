#!/bin/bash
#SBATCH --job-name=yolo_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail


# --- paths (devem bater com o submit_train_yolo.sh) ---
CVAT_ZIP="${CVAT_ZIP:?Defina CVAT_ZIP com um caminho absoluto}"
WORKDIR="${WORKDIR:?Defina WORKDIR com um caminho absoluto}"

cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/training/yolo"

python evaluate_yolo.py \
    --workdir "$WORKDIR" \
    --checkpoint "$WORKDIR/yolo_run/weights/best_real.pt" \
    --cvat-zip "$CVAT_ZIP" \
    --iou-threshold 0.5 \
    --score-threshold 0.5
