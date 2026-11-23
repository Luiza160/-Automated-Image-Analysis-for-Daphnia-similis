"""
Pipeline em cascata: detecção -> recorte -> [filtro] -> Daphnia Ruler.

Etapa 0 mede todos os recortes sem filtro. Os recortes com medida plausível
são salvos e saem da fila; os restantes (implausível, sem medida ou erro no
daphruler) seguem para o próximo filtro, e assim por diante.

Os filtros são CUMULATIVOS: a etapa k recebe a imagem já filtrada pela
etapa k-1. Um recorte resolvido na etapa 3 passou pelos filtros 1, 2 e 3.

Filtros que mudam o tamanho da imagem (ex: Real-ESRGAN x4) são tratados
automaticamente: a medida do Ruler é dividida pela escala real de cada
recorte antes de calcular a proporção com a caixa detectada.

Uso:
    python avaliar_pipeline_cascata.py \
        --model-type fasterrcnn \
        --checkpoint best_model.pth \
        --cvat-zip dataset.zip --format coco \
        --workdir avaliacao_cascata \
        --crop-scale-factor 1.6 \
        --filtros sem_filtro clahe sharpness deblur \
        --realesrgan-weights weights/RealESRGAN_x4plus.pth \
        --hat-repo ~/HAT \
        --hat-weights weights/HAT_SRx4_ImageNet-pretrain.pth

Filtros disponíveis:
    sem_filtro, clahe, sharpness, deblur, top_hat, black_hat,
    median, bilateral, realesrgan, hat.

Os filtros são cumulativos e podem ser combinados livremente com --filtros.
Real-ESRGAN e HAT são super-resolução x4 (por padrão) e a escala é
corrigida automaticamente antes do cálculo da proporção.
"""

import argparse
import csv
import math
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from PIL import Image

from common import read_cvat


# --------------------------------------------------------------------------
# Filtros
# --------------------------------------------------------------------------
class SemFiltro:
    nome = "sem_filtro"

    def aplicar(self, img):
        return img


class CLAHEFiltro:
    """CLAHE aplicado no canal L do espaço LAB."""

    nome = "clahe"

    def __init__(self, clip_limit=3.0, tile_size=8):
        self.clip_limit = clip_limit
        self.tile_size = tile_size

    def aplicar(self, img):
        import cv2
        import numpy as np

        arr = np.asarray(img.convert("RGB"))
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=(self.tile_size, self.tile_size),
        )
        l = clahe.apply(l)

        out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
        return Image.fromarray(out)


class SharpnessFiltro:
    """Unsharp mask: realça bordas sem alterar a resolução."""

    nome = "sharpness"

    def __init__(self, amount=1.0, sigma=1.0):
        self.amount = amount
        self.sigma = sigma

    def aplicar(self, img):
        import cv2
        import numpy as np

        arr = np.asarray(img.convert("RGB"))
        blur = cv2.GaussianBlur(arr, (0, 0), self.sigma)
        out = cv2.addWeighted(
            arr, 1.0 + self.amount,
            blur, -self.amount,
            0,
        )
        out = np.clip(out, 0, 255).astype(np.uint8)
        return Image.fromarray(out)


class DeblurFiltro:
    """
    Restauração por Richardson-Lucy com PSF gaussiana.

    Não altera a resolução. Os parâmetros sigma e iterations controlam
    respectivamente a largura estimada do desfoque e o número de iterações.
    """

    nome = "deblur"

    def __init__(self, sigma=1.5, iterations=10):
        self.sigma = sigma
        self.iterations = iterations

    def aplicar(self, img):
        import numpy as np
        from skimage.restoration import richardson_lucy

        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0

        # PSF gaussiana.
        radius = max(1, int(np.ceil(3 * self.sigma)))
        size = 2 * radius + 1
        axis = np.arange(-radius, radius + 1, dtype=np.float32)
        xx, yy = np.meshgrid(axis, axis)
        psf = np.exp(-(xx**2 + yy**2) / (2 * self.sigma**2))
        psf /= psf.sum()

        channels = []
        for c in range(3):
            restored = richardson_lucy(
                arr[:, :, c],
                psf,
                num_iter=self.iterations,
                clip=False,
            )
            channels.append(restored)

        out = np.clip(np.stack(channels, axis=2), 0.0, 1.0)
        return Image.fromarray((out * 255.0).astype(np.uint8))


class MorphologyFiltro:
    """
    Top-hat ou black-hat morfológico.

    top_hat: destaca estruturas claras menores que o elemento estruturante.
    black_hat: destaca estruturas escuras menores que o elemento estruturante.
    """

    def __init__(self, mode="top_hat", kernel_size=15):
        if mode not in ("top_hat", "black_hat"):
            raise ValueError("mode deve ser 'top_hat' ou 'black_hat'")
        self.mode = mode
        self.kernel_size = kernel_size
        self.nome = mode

    def aplicar(self, img):
        import cv2
        import numpy as np

        arr = np.asarray(img.convert("RGB"))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.kernel_size, self.kernel_size),
        )

        if self.mode == "top_hat":
            out = cv2.morphologyEx(arr, cv2.MORPH_TOPHAT, kernel)
        else:
            out = cv2.morphologyEx(arr, cv2.MORPH_BLACKHAT, kernel)

        # A operação morfológica retorna a diferença. Para não descartar
        # completamente a imagem original, somamos a resposta ao original.
        if self.mode == "top_hat":
            out = cv2.add(arr, out)
        else:
            out = cv2.subtract(arr, out)

        return Image.fromarray(out)


class MedianFiltro:
    """Redução de ruído impulsivo preservando relativamente as bordas."""

    nome = "median"

    def __init__(self, kernel_size=3):
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size deve ser ímpar e >= 3")
        self.kernel_size = kernel_size

    def aplicar(self, img):
        import cv2
        import numpy as np

        arr = np.asarray(img.convert("RGB"))
        out = cv2.medianBlur(arr, self.kernel_size)
        return Image.fromarray(out)


class BilateralFiltro:
    """Suavização com preservação de bordas."""

    nome = "bilateral"

    def __init__(self, diameter=7, sigma_color=50.0, sigma_space=50.0):
        self.diameter = diameter
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def aplicar(self, img):
        import cv2
        import numpy as np

        arr = np.asarray(img.convert("RGB"))
        out = cv2.bilateralFilter(
            arr,
            self.diameter,
            self.sigma_color,
            self.sigma_space,
        )
        return Image.fromarray(out)


class RealESRGANFiltro:
    """Super-resolução com Real-ESRGAN (modelo RealESRGAN_x4plus)."""

    def __init__(self, weights, outscale=4.0, tile=0):
        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        if not Path(weights).exists():
            raise FileNotFoundError(f"Pesos do Real-ESRGAN não encontrados: {weights}")

        rede = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        self.upsampler = RealESRGANer(
            scale=4,
            model_path=str(weights),
            model=rede,
            tile=tile,
            tile_pad=10,
            pre_pad=0,
            half=torch.cuda.is_available(),
        )
        self.outscale = outscale
        self.nome = f"realesrgan_x{outscale:g}"

    def aplicar(self, img):
        import cv2
        import numpy as np

        bgr = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        saida, _ = self.upsampler.enhance(bgr, outscale=self.outscale)
        return Image.fromarray(cv2.cvtColor(saida, cv2.COLOR_BGR2RGB))


class HATFiltro:
    """
    HAT (Hybrid Attention Transformer) para SR x4.

    Usa a arquitetura HAT-SRx4_ImageNet-pretrain oficial. O repositório
    XPixelGroup/HAT deve estar disponível em --hat-repo e os pesos em
    --hat-weights.

    A imagem é preenchida até múltiplos da janela (16x16), processada e
    recortada novamente. A saída tem 4x a resolução original.
    """

    nome = "hat_x4"

    def __init__(self, repo, weights, tile=0, device=None):
        import sys

        import torch

        self.repo = Path(repo).expanduser().resolve()
        self.weights = Path(weights).expanduser().resolve()

        if not self.repo.exists():
            raise FileNotFoundError(
                f"Repositório HAT não encontrado: {self.repo}\n"
                "Use --hat-repo apontando para o clone de XPixelGroup/HAT."
            )

        if not self.weights.exists():
            raise FileNotFoundError(f"Pesos do HAT não encontrados: {self.weights}")

        repo_str = str(self.repo)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        from hat.archs.hat_arch import HAT

        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.window_size = 16
        self.scale = 4

        self.model = HAT(
            upscale=4,
            in_chans=3,
            img_size=64,
            window_size=16,
            compress_ratio=3,
            squeeze_factor=30,
            conv_scale=0.01,
            overlap_ratio=0.5,
            img_range=1.0,
            depths=(6, 6, 6, 6, 6, 6),
            embed_dim=180,
            num_heads=(6, 6, 6, 6, 6, 6),
            mlp_ratio=2,
            upsampler="pixelshuffle",
            resi_connection="1conv",
        )

        checkpoint = torch.load(self.weights, map_location="cpu")
        if isinstance(checkpoint, dict):
            if "params_ema" in checkpoint:
                state_dict = checkpoint["params_ema"]
            elif "params" in checkpoint:
                state_dict = checkpoint["params"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Remove eventual DataParallel prefix.
        state_dict = {
            k.replace("module.", "", 1) if k.startswith("module.") else k: v
            for k, v in state_dict.items()
        }

        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _pad_to_window(arr, window_size):
        import cv2

        h, w = arr.shape[:2]
        pad_h = (window_size - h % window_size) % window_size
        pad_w = (window_size - w % window_size) % window_size

        if pad_h == 0 and pad_w == 0:
            return arr, h, w

        # BORDER_REFLECT_101 evita introduzir bordas pretas.
        padded = cv2.copyMakeBorder(
            arr,
            0, pad_h,
            0, pad_w,
            cv2.BORDER_REFLECT_101,
        )
        return padded, h, w

    def aplicar(self, img):
        import numpy as np
        import torch

        arr = np.asarray(img.convert("RGB"))
        padded, h, w = self._pad_to_window(arr, self.window_size)

        tensor = (
            torch.from_numpy(padded.transpose(2, 0, 1))
            .float()
            .unsqueeze(0)
            / 255.0
        ).to(self.device)

        with torch.inference_mode():
            output = self.model(tensor).clamp_(0.0, 1.0)

        out = (
            output.squeeze(0)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )

        # Remove a região gerada a partir do padding.
        out = out[:h * self.scale, :w * self.scale]
        out = (out * 255.0).round().astype(np.uint8)

        return Image.fromarray(out)


def criar_filtro(nome, args):
    if nome == "sem_filtro":
        return SemFiltro()

    if nome == "clahe":
        return CLAHEFiltro(
            clip_limit=args.clahe_clip,
            tile_size=args.clahe_tile,
        )

    if nome == "sharpness":
        return SharpnessFiltro(
            amount=args.sharpness_amount,
            sigma=args.sharpness_sigma,
        )

    if nome == "deblur":
        return DeblurFiltro(
            sigma=args.deblur_sigma,
            iterations=args.deblur_iterations,
        )

    if nome == "top_hat":
        return MorphologyFiltro(
            mode="top_hat",
            kernel_size=args.morph_kernel,
        )

    if nome == "black_hat":
        return MorphologyFiltro(
            mode="black_hat",
            kernel_size=args.morph_kernel,
        )

    if nome == "median":
        return MedianFiltro(args.median_kernel)

    if nome == "bilateral":
        return BilateralFiltro(
            diameter=args.bilateral_diameter,
            sigma_color=args.bilateral_sigma_color,
            sigma_space=args.bilateral_sigma_space,
        )

    if nome == "realesrgan":
        return RealESRGANFiltro(
            args.realesrgan_weights,
            args.realesrgan_outscale,
            args.realesrgan_tile,
        )

    if nome == "hat":
        return HATFiltro(
            repo=args.hat_repo,
            weights=args.hat_weights,
            tile=args.hat_tile,
        )

    raise ValueError(
        f"Filtro desconhecido: {nome}. "
        "Opções: sem_filtro, clahe, sharpness, deblur, top_hat, "
        "black_hat, median, bilateral, realesrgan, hat"
    )


# --------------------------------------------------------------------------
# Detecção
# --------------------------------------------------------------------------
def load_fasterrcnn(checkpoint_path, device):
    import torch
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def predict_fasterrcnn(model, image_path, device, score_threshold):
    import torch
    from torchvision.transforms.functional import pil_to_tensor

    image = Image.open(image_path).convert("RGB")
    tensor = pil_to_tensor(image).float().to(device) / 255
    with torch.no_grad():
        output = model([tensor])[0]
    boxes = output["boxes"].cpu().tolist()
    scores = output["scores"].cpu().tolist()
    keep = [i for i, s in enumerate(scores) if s >= score_threshold]
    return [boxes[i] for i in keep], [scores[i] for i in keep]


def predict_yolo(model, image_path, score_threshold):
    result = model.predict(source=str(image_path), conf=score_threshold, verbose=False)[0]
    return result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist()


META_FIELDS = [
    "crop_name", "image_name", "confianca_deteccao",
    "x1", "y1", "x2", "y2", "largura_caixa_px", "altura_caixa_px", "diagonal_caixa_px",
]


def detectar_e_recortar(args, originais_dir, meta_csv):
    """Roda a detecção em todas as imagens e salva um recorte por Daphnia."""
    cvat_dir = args.cvat_dir
    if args.cvat_zip is not None:
        cvat_dir = (args.workdir / "cvat_export_extracted").resolve()
        if cvat_dir.exists():
            shutil.rmtree(cvat_dir)
        cvat_dir.mkdir(parents=True)
        with zipfile.ZipFile(args.cvat_zip.resolve()) as archive:
            archive.extractall(cvat_dir)

    _, records = read_cvat(cvat_dir, args.format)
    image_paths = sorted({path for path, *_ in records})
    print(f"{len(image_paths)} imagens encontradas")

    device = None
    if args.model_type == "fasterrcnn":
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_fasterrcnn(args.checkpoint, device)
    else:
        from ultralytics import YOLO

        model = YOLO(str(args.checkpoint))

    if originais_dir.exists():
        shutil.rmtree(originais_dir)
    originais_dir.mkdir(parents=True)

    metadata = []
    for image_path in image_paths:
        if args.model_type == "fasterrcnn":
            boxes, scores = predict_fasterrcnn(model, image_path, device, args.score_threshold)
        else:
            boxes, scores = predict_yolo(model, image_path, args.score_threshold)

        image = Image.open(image_path).convert("RGB")
        for i, (box, score) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            extra_w = w * (args.crop_scale_factor - 1) / 2
            extra_h = h * (args.crop_scale_factor - 1) / 2
            cx1, cy1 = max(0, int(x1 - extra_w)), max(0, int(y1 - extra_h))
            cx2, cy2 = min(image.width, int(x2 + extra_w)), min(image.height, int(y2 + extra_h))

            crop_name = f"{len(metadata):04d}_{image_path.stem}_d{i}.png"
            image.crop((cx1, cy1, cx2, cy2)).save(originais_dir / crop_name)
            metadata.append({
                "crop_name": crop_name,
                "image_name": image_path.name,
                "confianca_deteccao": round(score, 4),
                "x1": cx1, "y1": cy1, "x2": cx2, "y2": cy2,
                "largura_caixa_px": w,
                "altura_caixa_px": h,
                "diagonal_caixa_px": math.hypot(w, h),
            })
        print(f"  {image_path.name}: {len(boxes)} detecção(ões)")

    with open(meta_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=META_FIELDS)
        writer.writeheader()
        writer.writerows(metadata)

    # libera a GPU para o Real-ESRGAN
    del model
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:
        pass

    return metadata


def carregar_metadata(meta_csv):
    with open(meta_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["diagonal_caixa_px"] = float(r["diagonal_caixa_px"])
    return rows


# --------------------------------------------------------------------------
# Daphnia Ruler
# --------------------------------------------------------------------------
def rodar_daphruler(pasta, workdir, coluna, daphruler_python):
    """
    Roda o daphruler na pasta. Se falhar no lote, testa um recorte por vez,
    remove os que quebram e roda de novo. Retorna (medidas, excluidos).
    """
    excluidos = []
    if not any(pasta.glob("*.png")):
        return {}, excluidos

    cmd = [str(daphruler_python), "-m", "daphruler", "-p", str(pasta)]
    if subprocess.run(cmd).returncode != 0:
        print("[aviso] daphruler falhou no lote; isolando recortes problemáticos...")
        teste = workdir / "_teste_isolamento"
        for crop in sorted(pasta.glob("*.png")):
            if teste.exists():
                shutil.rmtree(teste)
            teste.mkdir(parents=True)
            shutil.copy2(crop, teste / crop.name)
            r = subprocess.run(
                [str(daphruler_python), "-m", "daphruler", "-p", str(teste), "-n"], capture_output=True
            )
            if r.returncode != 0:
                excluidos.append(crop.name)
                crop.unlink()
                print(f"  [excluído] {crop.name}")
        shutil.rmtree(teste, ignore_errors=True)
        if any(pasta.glob("*.png")):
            subprocess.run(cmd, check=True)

    ruler_csv = pasta / "results" / f"measurement_results_{pasta.name}.csv"
    if not ruler_csv.exists():
        candidatos = list(pasta.rglob("measurement_results_*")) + list(pasta.parent.glob("measurement_results_*"))
        if not candidatos:
            print("[aviso] nenhum resultado do daphruler encontrado nesta etapa")
            return {}, excluidos
        ruler_csv = candidatos[0]

    medidas = {}
    with open(ruler_csv) as f:
        for row in csv.DictReader(f):
            nome = row["ID"].replace("\\", "/").split("/")[-1]
            valor = row.get(coluna, "NA")
            medidas[nome] = None if valor in ("NA", "", None) else float(valor)
    return medidas, excluidos


# --------------------------------------------------------------------------
# Informações dos filtros
# --------------------------------------------------------------------------
def parametros_filtro(nome, args):
    """Retorna uma descrição compacta dos parâmetros usados em cada etapa."""
    if nome == "sem_filtro":
        return "nenhum"

    if nome == "clahe":
        return f"clip={args.clahe_clip};tile={args.clahe_tile}"

    if nome == "sharpness":
        return f"amount={args.sharpness_amount};sigma={args.sharpness_sigma}"

    if nome == "deblur":
        return f"sigma={args.deblur_sigma};iterations={args.deblur_iterations}"

    if nome in ("top_hat", "black_hat"):
        return f"kernel={args.morph_kernel}"

    if nome == "median":
        return f"kernel={args.median_kernel}"

    if nome == "bilateral":
        return (
            f"diameter={args.bilateral_diameter};"
            f"sigma_color={args.bilateral_sigma_color};"
            f"sigma_space={args.bilateral_sigma_space}"
        )

    if nome == "realesrgan":
        return (
            f"outscale={args.realesrgan_outscale};"
            f"tile={args.realesrgan_tile};"
            f"weights={args.realesrgan_weights}"
        )

    if nome == "hat":
        return (
            f"scale=4;tile={args.hat_tile};"
            f"weights={args.hat_weights};repo={args.hat_repo}"
        )

    return ""


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-type", choices=["yolo", "fasterrcnn"], required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--cvat-zip", type=Path)
    src.add_argument("--cvat-dir", type=Path)
    p.add_argument("--format", choices=["coco", "yolo", "voc"], required=True)
    p.add_argument("--workdir", type=Path, required=True)
    p.add_argument("--score-threshold", type=float, default=0.8)
    p.add_argument("--crop-scale-factor", type=float, default=1.6)
    p.add_argument("--min-proporcao", type=float, default=0.5)
    p.add_argument("--max-proporcao", type=float, default=1.0)
    p.add_argument("--measurement-column", default="body.Length.h")
    p.add_argument(
        "--daphruler-python",
        type=Path,
        default=Path(sys.executable),
        help="Python usado para executar o Daphnia Ruler. "
             "Útil quando o pipeline e o Daphnia Ruler estão em ambientes Conda diferentes.",
    )
    p.add_argument(
        "--reusar-recortes", action="store_true",
        help="Pula a detecção se recortes_originais/ e crop_metadata.csv já existirem no workdir",
    )

    p.add_argument(
        "--filtros",
        nargs="+",
        default=["sem_filtro"],
        help=(
            "Ordem da cascata. Opções: sem_filtro, clahe, sharpness, deblur, "
            "top_hat, black_hat, median, bilateral, realesrgan, hat. "
            "Exemplo: --filtros sem_filtro clahe sharpness deblur"
        ),
    )

    p.add_argument(
        "--nome-execucao",
        default=None,
        help=(
            "Nome da subpasta de resultados dentro de <workdir>/execucoes/. "
            "Se omitido, usa os filtros unidos por '+' (ex.: sem_filtro+realesrgan). "
            "Os recortes originais e crop_metadata.csv continuam compartilhados no workdir."
        ),
    )

    # CLAHE
    p.add_argument("--clahe-clip", type=float, default=3.0)
    p.add_argument("--clahe-tile", type=int, default=8)

    # Unsharp mask
    p.add_argument("--sharpness-amount", type=float, default=1.0)
    p.add_argument("--sharpness-sigma", type=float, default=1.0)

    # Richardson-Lucy
    p.add_argument("--deblur-sigma", type=float, default=1.5)
    p.add_argument("--deblur-iterations", type=int, default=10)

    # Morphologia
    p.add_argument("--morph-kernel", type=int, default=15)

    # Denoising
    p.add_argument("--median-kernel", type=int, default=3)
    p.add_argument("--bilateral-diameter", type=int, default=7)
    p.add_argument("--bilateral-sigma-color", type=float, default=50.0)
    p.add_argument("--bilateral-sigma-space", type=float, default=50.0)

    # Real-ESRGAN
    p.add_argument(
        "--realesrgan-weights",
        type=Path,
        default=Path("weights/RealESRGAN_x4plus.pth"),
    )
    p.add_argument("--realesrgan-outscale", type=float, default=4.0)
    p.add_argument("--realesrgan-tile", type=int, default=0)

    # HAT SR x4
    p.add_argument(
        "--hat-repo",
        type=Path,
        default=Path.home() / "HAT",
        help="Clone do repositório oficial XPixelGroup/HAT.",
    )
    p.add_argument(
        "--hat-weights",
        type=Path,
        default=Path("weights/HAT_SRx4_ImageNet-pretrain.pth"),
        help="Pesos do HAT SRx4.",
    )
    p.add_argument(
        "--hat-tile",
        type=int,
        default=0,
        help="Reservado para futura implementação de tiling nativo no HAT.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)

    originais_dir = args.workdir / "recortes_originais"
    meta_csv = args.workdir / "crop_metadata.csv"

    # 1. Detecção + recortes (uma vez só)
    if args.reusar_recortes and meta_csv.exists() and originais_dir.exists():
        print("Reutilizando recortes existentes")
        metadata = carregar_metadata(meta_csv)
    else:
        metadata = detectar_e_recortar(args, originais_dir, meta_csv)
    meta_por_nome = {m["crop_name"]: m for m in metadata}
    print(f"\n{len(metadata)} Daphnias recortados")

    nome_execucao = args.nome_execucao or "+".join(args.filtros)
    run_dir = args.workdir / "execucoes" / nome_execucao
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Resultados desta execução: {run_dir}")

    todas_plaus_dir = run_dir / "plausiveis_todas_etapas"
    if todas_plaus_dir.exists():
        shutil.rmtree(todas_plaus_dir)
    todas_plaus_dir.mkdir(parents=True)

    # 2. Cascata de filtros
    pendentes = [m["crop_name"] for m in metadata]
    # imagem de entrada de cada recorte na próxima etapa (começa no original,
    # depois aponta para a versão com todos os filtros aplicados até ali)
    fonte = {nome: originais_dir / nome for nome in pendentes}
    resolvidos = {}   # crop_name -> linha do resultado
    resumo = []

    for k, nome_filtro in enumerate(args.filtros):
        if not pendentes:
            print("Nenhum recorte pendente; cascata encerrada.")
            break

        filtro = criar_filtro(nome_filtro, args)
        filtros_acumulados = "+".join(args.filtros[: k + 1])
        etapa_dir = run_dir / f"etapa_{k}_{filtro.nome}"
        if etapa_dir.exists():
            shutil.rmtree(etapa_dir)
        filtrados_dir = etapa_dir / "filtrados"   # cópia intocada (entrada da próxima etapa)
        recortes_dir = etapa_dir / "recortes"     # pasta onde o daphruler roda
        plaus_dir = etapa_dir / "plausiveis"
        for d in (filtrados_dir, recortes_dir, plaus_dir):
            d.mkdir(parents=True)

        print(f"\n=== Etapa {k}: {filtros_acumulados} ({len(pendentes)} recortes) ===")
        escalas = {}
        erros_filtro = {}
        for nome in pendentes:
            largura_original = Image.open(originais_dir / nome).width
            entrada = Image.open(fonte[nome]).convert("RGB")
            try:
                filtrado = filtro.aplicar(entrada)
                filtrado.save(filtrados_dir / nome)
                shutil.copy2(filtrados_dir / nome, recortes_dir / nome)
                # escala acumulada em relação ao recorte ORIGINAL
                escalas[nome] = filtrado.width / largura_original
                fonte[nome] = filtrados_dir / nome
            except Exception as exc:
                # Um filtro que falhe em um único recorte não derruba a cascata.
                # Mantemos o recorte fora do Daphnia Ruler nesta etapa para que
                # o próximo filtro possa tentar processá-lo.
                erros_filtro[nome] = f"{type(exc).__name__}: {exc}"
                print(f"  [erro filtro] {nome}: {erros_filtro[nome]}")

        medidas, excluidos = rodar_daphruler(
            recortes_dir,
            run_dir,
            args.measurement_column,
            args.daphruler_python,
        )

        linhas, novos_pendentes = [], []
        contagem = {
            "plausivel": 0,
            "implausivel": 0,
            "sem_medida": 0,
            "excluido_erro_daphruler": 0,
            "erro_filtro": 0,
        }
        for nome in pendentes:
            meta = meta_por_nome[nome]
            medida, proporcao = None, None

            if nome in erros_filtro:
                status = "erro_filtro"
                novos_pendentes.append(nome)
            elif nome in excluidos:
                status = "excluido_erro_daphruler"
                novos_pendentes.append(nome)
            elif medidas.get(nome) is None:
                status = "sem_medida"
                novos_pendentes.append(nome)
            else:
                medida = medidas[nome] / escalas[nome]  # volta para pixels do recorte original
                proporcao = medida / meta["diagonal_caixa_px"]
                ok = args.min_proporcao <= proporcao <= args.max_proporcao
                status = "plausivel" if ok else "implausivel"
                if status != "plausivel":
                    novos_pendentes.append(nome)
            contagem[status] += 1

            linha = {
                "crop_name": nome,
                "imagem": meta["image_name"],
                "confianca_deteccao": meta["confianca_deteccao"],
                "etapa": k,
                "filtro": filtro.nome,
                "parametros_filtro": parametros_filtro(nome_filtro, args),
                "filtros_acumulados": filtros_acumulados,
                "escala_filtro": round(escalas[nome], 3) if nome in escalas else None,
                "medida_ruler_px_original": round(medida, 2) if medida is not None else None,
                "diagonal_caixa_px": round(meta["diagonal_caixa_px"], 1),
                "proporcao": round(proporcao, 3) if proporcao is not None else None,
                "status": status,
            }
            linhas.append(linha)

            if status == "plausivel":
                shutil.copy2(filtrados_dir / nome, plaus_dir / nome)
                shutil.copy2(
                    filtrados_dir / nome,
                    todas_plaus_dir / f"etapa{k}_{filtros_acumulados}_{nome}",
                )
                resolvidos[nome] = linha

        with open(etapa_dir / "resultado_etapa.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
            writer.writeheader()
            writer.writerows(linhas)

        print(f"Etapa {k} ({filtros_acumulados}): " + ", ".join(f"{s}={n}" for s, n in contagem.items()))
        resumo.append((k, filtros_acumulados, len(pendentes), contagem["plausivel"]))
        pendentes = novos_pendentes

    # 3. Tabela final: um registro por recorte
    final_csv = run_dir / "avaliacao_cascata.csv"
    campos = [
        "crop_name", "imagem", "confianca_deteccao",
        "etapa", "filtro", "parametros_filtro", "filtros_acumulados",
        "escala_filtro", "medida_ruler_px_original",
        "diagonal_caixa_px", "proporcao", "status",
    ]
    with open(final_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        for m in metadata:
            nome = m["crop_name"]
            if nome in resolvidos:
                writer.writerow(resolvidos[nome])
            else:
                writer.writerow({
                    "crop_name": nome,
                    "imagem": m["image_name"],
                    "confianca_deteccao": m["confianca_deteccao"],
                    "diagonal_caixa_px": round(m["diagonal_caixa_px"], 1),
                    "status": "nao_resolvido",
                })

    total = len(metadata)
    acumulado = 0
    print("\n=== Resumo da cascata ===")
    for k, nome, entrada, plaus in resumo:
        acumulado += plaus
        print(f"Etapa {k} ({nome}): {plaus}/{entrada} novos plausíveis | acumulado {acumulado}/{total} "
              f"({100 * acumulado / total:.1f}%)")
    print(f"Não resolvidos: {len(pendentes)}")
    print(f"\nImagens plausíveis: {todas_plaus_dir}")
    print(f"Tabela final: {final_csv}")


if __name__ == "__main__":
    main()