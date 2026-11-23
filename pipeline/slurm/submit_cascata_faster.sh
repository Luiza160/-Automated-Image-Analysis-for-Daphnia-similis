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
CVAT_ZIP="${CVAT_ZIP:?Defina CVAT_ZIP com um caminho absoluto}"
CHECKPOINT="${CHECKPOINT:?Defina CHECKPOINT com um caminho absoluto}"
OUT_WORKDIR="${WORKDIR:?Defina WORKDIR com um caminho absoluto}"
REALESRGAN_WEIGHTS="${REALESRGAN_WEIGHTS:?Defina REALESRGAN_WEIGHTS com um caminho absoluto}"

echo "=== Iniciando Pipeline em Cascata (Faster R-CNN + Real-ESRGAN) ==="
echo "Data: $(date)"
echo "Node: $(hostname)"


cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/pipeline"

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
