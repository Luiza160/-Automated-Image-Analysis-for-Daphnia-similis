"""
Linha de comando do pipeline da Daphnia.

Exemplos:
    python main.py --entrada fotos.zip --workdir resultados
    python main.py --entrada "Cont Ap@GO" --workdir resultados --px-por-mm 145
    python main.py --model-type fasterrcnn --checkpoint fasterrcnn_best.pth --entrada fotos.zip --workdir res
"""
import argparse
from pathlib import Path

import pipeline

BASE = Path(__file__).resolve().parent


def achar_pesos(nome):
    """Procura o arquivo de pesos nesta pasta e na pasta de cima."""
    for pasta in (BASE.parent / 'weights', BASE):
        if (pasta / nome).is_file():
            return pasta / nome
    return BASE / nome


def parse_args():
    p = argparse.ArgumentParser(description="Detecta e mede Daphnias em fotos.")
    p.add_argument("--entrada", "--dataset-zip", dest="entrada", type=Path, required=True,
                   help="arquivo .zip, pasta com as fotos, ou uma única foto")
    p.add_argument("--workdir", type=Path, required=True, help="pasta onde salvar os resultados")
    p.add_argument("--model-type", choices=["yolo", "fasterrcnn"], default="yolo")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="pesos do detector (padrão: yolo11n_best.pt ou fasterrcnn_best.pth)")
    p.add_argument("--score-threshold", type=float, default=None,
                   help="confiança mínima da detecção (padrão: 0.5 YOLO, 0.8 Faster R-CNN)")
    p.add_argument("--crop-scale-factor", type=float, default=1.6,
                   help="margem do recorte em volta da caixa (não altere sem revalidar)")
    p.add_argument("--px-por-mm", type=float, default=None,
                   help="escala da foto em pixels por mm, para a planilha sair também em mm")
    p.add_argument("--sem-imagens", action="store_true",
                   help="não salva as imagens anotadas (um pouco mais rápido)")
    p.add_argument("--salvar-etapas", action="store_true",
                   help="salva uma imagem de cada etapa para cada Daphnia")
    a = p.parse_args()
    if a.checkpoint is None:
        a.checkpoint = achar_pesos("yolo11n_best.pt" if a.model_type == "yolo" else "fasterrcnn_best.pth")
    if a.score_threshold is None:
        a.score_threshold = 0.5 if a.model_type == "yolo" else 0.8
    a.salvar_imagens = not a.sem_imagens
    return a


if __name__ == "__main__":
    pipeline.executar_pipeline(parse_args())
