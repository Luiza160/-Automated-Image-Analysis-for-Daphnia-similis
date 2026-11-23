#!/bin/bash
#SBATCH --job-name=fasterrcnn_train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# --- ambiente ---

# --- caminhos (WORKDIR deve ser o mesmo do submit_eval_fasterrcnn.sh) ---
CVAT_ZIP="${CVAT_ZIP:?Defina CVAT_ZIP com um caminho absoluto}"
WORKDIR="${WORKDIR:?Defina WORKDIR com um caminho absoluto}"

cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/training/faster_rcnn"

python train_fasterrcnn.py \
    --cvat-zip "$CVAT_ZIP" \
    --format coco \
    --workdir "$WORKDIR" \
    --epochs 200 \
    --patience 20 \
    --lr 0.001 \
    --batch-size 2 \
    --run-name fasterrcnn_run \
    --val-fraction 0.15 \
    --test-fraction 0.15
