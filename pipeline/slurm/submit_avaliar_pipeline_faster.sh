#!/bin/bash
#SBATCH --job-name=eval_pipe_faster
#SBATCH --output=pipeline_faster_%j.out
#SBATCH --error=pipeline_faster_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

set -e

# Variáveis configuráveis de caminhos
REPO="$HOME/daphnia-repo/pipeline"
CVAT_ZIP="$HOME/datasets/dataset.zip"
CHECKPOINT="$HOME/daphnia_runs/fasterrcnn_run1/fasterrcnn_run/best_model.pth"
OUT_WORKDIR="$HOME/daphnia_runs/pipeline_faster"

echo "=== Iniciando Avaliação do Pipeline (Faster R-CNN) ==="
echo "Data: $(date)"
echo "Node: $(hostname)"

# Ativação do ambiente único do pipeline
source ~/.bashrc
conda activate pipeline_env

cd "$REPO"

python avaliar_pipeline_completo.py \
    --cvat-zip "$CVAT_ZIP" \
    --format coco \
    --model-type fasterrcnn \
    --checkpoint "$CHECKPOINT" \
    --workdir "$OUT_WORKDIR"

echo "=== Finalizado com sucesso! ==="
