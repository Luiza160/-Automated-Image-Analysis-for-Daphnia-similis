import argparse
from pathlib import Path
import sys
import pipeline


def parse_args():
    p = argparse.ArgumentParser(description="Interface CLI para o Pipeline Cascata")
    p.add_argument("--model-type", choices=["yolo", "fasterrcnn"], required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dataset-zip", type=Path, required=True)
    p.add_argument("--workdir", type=Path, required=True)
    p.add_argument("--score-threshold", type=float, default=0.5)
    p.add_argument("--crop-scale-factor", type=float, default=1.6)
    p.add_argument("--daphruler-python", type=Path, default=Path(sys.executable))
    p.add_argument("--reusar-recortes", action="store_true")
    p.add_argument("--filtros", nargs="+", default=["sem_filtro"])
    p.add_argument("--nome-execucao", default=None)
    p.add_argument("--min-proporcao", type=float, default=0.5)
    p.add_argument("--max-proporcao", type=float, default=1.0)
    p.add_argument("--aumento-objetiva", type=int, default=None,
                   help="Aumento da objetiva (ex.: 10, 20 ou 40). Apenas metadado.")
    p.add_argument("--px-por-mm", type=float, default=0.0,
                   help="Calibração em pixels por milímetro; 0 desativa conversão.")

    # Paralelismo (CPU)
    p.add_argument("--device", default="cpu",
                   help="dispositivo do Real-ESRGAN: cpu (padrão), xpu (GPU Intel) ou cuda")
    p.add_argument("--device-deteccao", default="cpu",
                   help="dispositivo da detecção (padrão: cpu)")
    p.add_argument("--cpus", type=int, default=None,
                   help="núcleos a usar (padrão: todos)")
    p.add_argument("--workers-deteccao", type=int, default=2,
                   help="processos para a detecção")
    p.add_argument("--workers-esrgan", type=int, default=2,
                   help="processos para o Real-ESRGAN")
    p.add_argument("--ruler-workers", type=int, default=2,
                   help="lotes paralelos do Daphnia Ruler")

    # Parametros padrão dos filtros
    p.add_argument("--clahe-clip", type=float, default=3.0)
    p.add_argument("--clahe-tile", type=int, default=8)
    p.add_argument("--sharpness-amount", type=float, default=1.0)
    p.add_argument("--sharpness-sigma", type=float, default=1.0)
    p.add_argument("--deblur-sigma", type=float, default=1.5)
    p.add_argument("--deblur-iterations", type=int, default=10)
    p.add_argument("--morph-kernel", type=int, default=15)
    p.add_argument("--median-kernel", type=int, default=3)
    p.add_argument("--bilateral-diameter", type=int, default=7)
    p.add_argument("--bilateral-sigma-color", type=float, default=50.0)
    p.add_argument("--bilateral-sigma-space", type=float, default=50.0)
    p.add_argument("--realesrgan-weights", type=Path, default=Path("weights/RealESRGAN_x4plus.pth"))
    p.add_argument("--realesrgan-outscale", type=float, default=4.0)
    p.add_argument("--realesrgan-tile", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pipeline.executar_pipeline_cascata(args)