#!/bin/bash
#SBATCH --job-name=pipeline_faster
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate fasterrcnn_env

cd "$HOME/ValidacaoConfianca"

python avaliar_pipeline_completo.py \
    --model-type fasterrcnn \
    --checkpoint "$HOME/Faster R-CNN/run1/fasterrcnn_run/best_model.pth" \
    --cvat-zip "$HOME/datasets/dataset.zip" \
    --format coco \
    --workdir avaliacao_faster \
    --score-threshold 0.5 \
    --crop-scale-factor 1.6 \
    --min-proporcao 0.5 \
    --max-proporcao 1.2
