#!/bin/bash
#SBATCH --job-name=fasterrcnn_train
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
conda activate fasterrcnn_env

# --- caminhos (WORKDIR deve ser o mesmo do submit_eval_fasterrcnn.sh) ---
CVAT_ZIP="$HOME/datasets/dataset.zip"
WORKDIR="$HOME/daphnia_runs/fasterrcnn_run1"

cd "$HOME/daphnia-repo/training/faster_rcnn"

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
