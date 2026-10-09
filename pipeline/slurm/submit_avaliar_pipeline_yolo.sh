#!/bin/bash
#SBATCH --job-name=pipeline_yolo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
 
set -euo pipefail
 
source ~/miniconda3/etc/profile.d/conda.sh
conda activate yolo_env
 
# --- caminhos (mesmos do submit_train_yolo.sh) ---
REPO="$HOME/daphnia-repo"
CVAT_ZIP="$HOME/datasets/dataset.zip"
TRAIN_WORKDIR="$HOME/daphnia_runs/yolo_run1"
OUT_WORKDIR="$HOME/daphnia_runs/pipeline_yolo"
 
cd "$REPO/pipeline"
 
python avaliar_pipeline_completo.py \
    --model-type yolo \
    --checkpoint "$TRAIN_WORKDIR/yolo_run/weights/best_real.pt" \
    --cvat-zip "$CVAT_ZIP" \
    --format coco \
    --workdir "$OUT_WORKDIR" \
    --score-threshold 0.5 \
    --crop-scale-factor 1.6 \
    --min-proporcao 0.5 \
    --max-proporcao 1.2
