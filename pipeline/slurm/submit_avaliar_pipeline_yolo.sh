#!/bin/bash
#SBATCH --job-name=eval_pipe_yolo
#SBATCH --output=pipeline_yolo_%j.out
#SBATCH --error=pipeline_yolo_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

set -e

# Variáveis configuráveis de caminhos
REPO="$HOME/daphnia-repo/pipeline"
CVAT_ZIP="$HOME/datasets/dataset.zip"
CHECKPOINT="$HOME/daphnia_runs/yolo_run1/yolo_run/weights/best_real.pt"
OUT_WORKDIR="$HOME/daphnia_runs/pipeline_yolo"

echo "=== Iniciando Avaliação do Pipeline (YOLO) ==="
echo "Data: $(date)"
echo "Node: $(hostname)"

# Ativação do ambiente único do pipeline
source ~/.bashrc
conda activate pipeline_env

cd "$REPO"

python avaliar_pipeline_completo.py \
    --cvat_zip "$CVAT_ZIP" \
    --model_type yolo \
    --checkpoint "$CHECKPOINT" \
    --output_dir "$OUT_WORKDIR"

echo "=== Finalizado com sucesso! ==="
