#!/bin/bash
#SBATCH --job-name=daphnia
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out

set -euo pipefail

cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
python direct/main.py \
    --entrada "${ENTRADA:?Defina a entrada com um caminho absoluto}" \
    --workdir "${WORKDIR:?Defina a pasta de trabalho com um caminho absoluto}" \
    --model-type "${MODELO:-yolo}" \
    --checkpoint "${CHECKPOINT:?Defina o caminho absoluto do checkpoint}" \
    "$@"
