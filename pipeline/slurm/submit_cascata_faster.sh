#!/bin/bash
#SBATCH --job-name=cascata_filtros
#SBATCH --output=cascata_%j.out
#SBATCH --error=cascata_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

set -euo pipefail

# Variáveis configuráveis de caminhos
REPO="$HOME/daphnia-repo/pipeline"
CVAT_ZIP="$HOME/datasets/dataset.zip"
CHECKPOINT="$HOME/daphnia_runs/fasterrcnn_run1/fasterrcnn_run/best_model.pth"
OUT_WORKDIR="$HOME/daphnia_runs/avaliacao_cascata"
REALESRGAN_WEIGHTS="$REPO/weights/RealESRGAN_x4plus.pth"

echo "=== Iniciando Pipeline em Cascata (Faster R-CNN + Real-ESRGAN) ==="
echo "Data: $(date)"
echo "Node: $(hostname)"

source ~/.bashrc
conda activate pipeline_env

cd "$REPO"

export PYTHONUNBUFFERED=1

python avaliar_pipeline_cascata.py \
    --model-type fasterrcnn \
    --checkpoint "$CHECKPOINT" \
    --cvat-zip "$CVAT_ZIP" \
    --format coco \
    --workdir "$OUT_WORKDIR" \
    --crop-scale-factor 1.6 \
    --filtros sem_filtro realesrgan \
    --realesrgan-weights "$REALESRGAN_WEIGHTS" \
    --realesrgan-tile 256

echo "=== Finalizado com sucesso! ==="
