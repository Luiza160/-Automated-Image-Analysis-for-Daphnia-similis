#!/usr/bin/env python
"""
Pipeline em cascata: detecção -> recorte -> [filtro] -> Daphnia Ruler.

Etapa 0 mede todos os recortes sem filtro. Os recortes com medida plausível
são salvos e saem da fila; os restantes (implausível, sem medida ou erro no
daphruler) seguem para o próximo filtro, e assim por diante.

Os filtros são cumulativos: a etapa k recebe a imagem já filtrada pela
etapa k-1. Um recorte resolvido na etapa 3 passou pelos filtros 1, 2 e 3.

Filtros que mudam o tamanho da imagem (como o Real-ESRGAN x4) são tratados
automaticamente: a medida do Ruler é dividida pela escala real de cada
recorte antes de calcular a proporção com a caixa detectada.

Uso:
    python avaliar_pipeline_cascata.py \
        --model-type fasterrcnn \
        --checkpoint $HOME/daphnia_runs/fasterrcnn_run1/fasterrcnn_run/best_model.pth \
        --cvat-zip $HOME/datasets/dataset.zip --format coco \
        --workdir $HOME/daphnia_runs/avaliacao_cascata \
        --crop-scale-factor 1.6 \
        --filtros sem_filtro realesrgan \
        --realesrgan-weights weights/RealESRGAN_x4plus.pth

Para adicionar um novo filtro: criar uma classe com `nome` e `aplicar(img) -> img`
e registre em criar_filtro().
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


class RealESRGANFiltro:
    """Super-resolução com Real-ESRGAN (modelo RealESRGAN_x4plus)."""

    def __init__(self, weights, outscale=4.0, tile=0):
        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        rede = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        self.upsampler = RealESRGANer(
            scale=4,
            model_path=str(weights),
            model=rede,
            tile=tile,          # >0 processa em blocos (use se faltar memória na GPU)
            tile_pad=10,
            pre_pad=0,
            half=torch.cuda.is_available(),
        )
        self.outscale = outscale
        self.nome = f"realesrgan_x{outscale:g}"

    def aplicar(self, img):
        import cv2
        import numpy as np

        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        saida, _ = self.upsampler.enhance(bgr, outscale=self.outscale)
        return Image.fromarray(cv2.cvtColor(saida, cv2.COLOR_BGR2RGB))


def criar_filtro(nome, args):
    if nome == "sem_filtro":
        return SemFiltro()
    if nome == "realesrgan":
        return RealESRGANFiltro(args.realesrgan_weights, args.realesrgan_outscale, args.realesrgan_tile)
    raise ValueError(f"Filtro desconhecido: {nome}")


# --------------------------------------------------------------------------
# Detecção
# --------------------------------------------------------------------------
def load_fasterrcnn(checkpoint_path, device):
    import torch
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    checkpoint = torch.load(checkpoint_path, map_location=device)
    num_classes = checkpoint.get("num_classes", 2)

    model = fasterrcnn_resnet50_fpn_v2(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

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

    # Libera a GPU para o Real-ESRGAN
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
def rodar_daphruler(pasta, workdir, coluna):
    """
    Roda o daphruler na pasta. Se falhar no lote, testa um recorte por vez,
    remove os que quebram e roda de novo. Retorna (medidas, excluidos).
    """
    excluidos = []
    cmd = [sys.executable, "-m", "daphruler", "-p", str(pasta)]
    if subprocess.run(cmd).returncode != 0:
        print("[aviso] daphruler falhou no lote; isolando recortes problemáticos...")
        teste = workdir / "_teste_isolamento"
        for crop in sorted(pasta.glob("*.png")):
            if teste.exists():
                shutil.rmtree(teste)
            teste.mkdir(parents=True)
            shutil.copy2(crop, teste / crop.name)
            r = subprocess.run(
                [sys.executable, "-m", "daphruler", "-p", str(teste), "-n"], capture_output=True
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
    p.add_argument("--score-threshold", type=float, default=0.5)
    p.add_argument("--crop-scale-factor", type=float, default=1.6)
    p.add_argument("--min-proporcao", type=float, default=0.5)
    p.add_argument("--max-proporcao", type=float, default=2.0)
    p.add_argument("--measurement-column", default="body.Length.h")
    p.add_argument(
        "--reusar-recortes", action="store_true",
        help="Pula a detecção se recortes_originais/ e crop_metadata.csv já existirem no workdir",
    )

    p.add_argument("--filtros", nargs="+", default=["sem_filtro", "realesrgan"],
                   help="Ordem da cascata, ex: sem_filtro realesrgan")
    p.add_argument("--realesrgan-weights", type=Path, default=Path("weights/RealESRGAN_x4plus.pth"))
    p.add_argument("--realesrgan-outscale", type=float, default=4.0)
    p.add_argument("--realesrgan-tile", type=int, default=0)
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

    todas_plaus_dir = args.workdir / "plausiveis_todas_etapas"
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
        etapa_dir = args.workdir / f"etapa_{k}_{filtro.nome}"
        if etapa_dir.exists():
            shutil.rmtree(etapa_dir)
        filtrados_dir = etapa_dir / "filtrados"   # cópia intocada (entrada da próxima etapa)
        recortes_dir = etapa_dir / "recortes"     # pasta onde o daphruler roda
        plaus_dir = etapa_dir / "plausiveis"
        for d in (filtrados_dir, recortes_dir, plaus_dir):
            d.mkdir(parents=True)

        print(f"\n=== Etapa {k}: {filtros_acumulados} ({len(pendentes)} recortes) ===")
        escalas = {}
        for nome in pendentes:
            largura_original = Image.open(originais_dir / nome).width
            entrada = Image.open(fonte[nome]).convert("RGB")
            filtrado = filtro.aplicar(entrada)
            filtrado.save(filtrados_dir / nome)
            shutil.copy2(filtrados_dir / nome, recortes_dir / nome)
            # escala acumulada em relação ao recorte ORIGINAL
            escalas[nome] = filtrado.width / largura_original
            fonte[nome] = filtrados_dir / nome

        medidas, excluidos = rodar_daphruler(recortes_dir, args.workdir, args.measurement_column)

        linhas, novos_pendentes = [], []
        contagem = {"plausivel": 0, "implausivel": 0, "sem_medida": 0, "excluido_erro_daphruler": 0}
        for nome in pendentes:
            meta = meta_por_nome[nome]
            medida, proporcao = None, None
            if nome in excluidos:
                status = "excluido_erro_daphruler"
            elif medidas.get(nome) is None:
                status = "sem_medida"
            else:
                medida = medidas[nome] / escalas[nome]  # volta para pixels do recorte original
                proporcao = medida / meta["diagonal_caixa_px"]
                ok = args.min_proporcao <= proporcao <= args.max_proporcao
                status = "plausivel" if ok else "implausivel"
            contagem[status] += 1

            linha = {
                "crop_name": nome,
                "imagem": meta["image_name"],
                "confianca_deteccao": meta["confianca_deteccao"],
                "etapa": k,
                "filtro": filtro.nome,
                "filtros_acumulados": filtros_acumulados,
                "escala_filtro": round(escalas[nome], 3),
                "medida_ruler_px_original": round(medida, 2) if medida is not None else None,
                "diagonal_caixa_px": round(meta["diagonal_caixa_px"], 1),
                "proporcao": round(proporcao, 3) if proporcao is not None else None,
                "status": status,
            }
            linhas.append(linha)

            if status == "plausivel":
                shutil.copy2(filtrados_dir / nome, plaus_dir / nome)
                shutil.copy2(filtrados_dir / nome, todas_plaus_dir / f"etapa{k}_{filtros_acumulados}_{nome}")
                resolvidos[nome] = linha
            else:
                novos_pendentes.append(nome)

        with open(etapa_dir / "resultado_etapa.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
            writer.writeheader()
            writer.writerows(linhas)

        print(f"Etapa {k} ({filtros_acumulados}): " + ", ".join(f"{s}={n}" for s, n in contagem.items()))
        resumo.append((k, filtros_acumulados, len(pendentes), contagem["plausivel"]))
        pendentes = novos_pendentes

    # 3. Tabela final: um registro por recorte
    final_csv = args.workdir / "avaliacao_cascata.csv"
    campos = ["crop_name", "imagem", "confianca_deteccao", "etapa", "filtro", "filtros_acumulados", "escala_filtro",
              "medida_ruler_px_original", "diagonal_caixa_px", "proporcao", "status"]
    with open(final_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        for m in metadata:
            nome = m["crop_name"]
            if nome in resolvidos:
                writer.writerow(resolvidos[nome])
            else:
                writer.writerow({"crop_name": nome, "imagem": m["image_name"],
                                 "confianca_deteccao": m["confianca_deteccao"],
                                 "diagonal_caixa_px": round(m["diagonal_caixa_px"], 1),
                                 "status": "nao_resolvido"})

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
