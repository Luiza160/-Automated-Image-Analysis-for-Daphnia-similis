#!/usr/bin/env python
"""
Roda o pipeline completo (detecção -> recorte -> Daphnia Ruler) em TODAS as
imagens de um dataset CVAT, sem nenhum filtro de pré-processamento, e
reporta quantos Daphnias foram medidos com um comprimento biologicamente
plausível vs. quantos parecem ter sido mal segmentados pelo Ruler (ex: o
Ruler mediu só o olho em vez do corpo inteiro, como já vimos acontecer).

Não usa gabarito manual — a checagem é de PLAUSIBILIDADE (a medida do
Ruler é proporcional ao tamanho da própria caixa detectada?), não de erro
exato contra um valor conhecido. Serve para ter um panorama geral: "de
zero a cem, quantos Daphnias esse pipeline consegue medir direito sem
nenhum processamento extra?"

Uso:
    python avaliar_pipeline_completo.py \
        --model-type fasterrcnn \
        --checkpoint "../Faster R-CNN/run1/fasterrcnn_run/best_model.pth" \
        --cvat-zip dataset.zip \
        --format coco \
        --workdir avaliacao_geral \
        --crop-scale-factor 1.2 \
        --min-proporcao 0.5 --max-proporcao 1.2
"""

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from PIL import Image

from common import read_cvat


# --------------------------------------------------------------------------
# Modelo de detecção
# --------------------------------------------------------------------------
def load_fasterrcnn(checkpoint_path, device):
    import torch
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint["class_names"]
    num_classes = checkpoint["num_classes"]

    model = fasterrcnn_resnet50_fpn_v2(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
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
    boxes = result.boxes.xyxy.cpu().tolist()
    scores = result.boxes.conf.cpu().tolist()
    return boxes, scores


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
    p.add_argument("--score-threshold", type=float, default=0.5, help="Confiança mínima de detecção para considerar um Daphnia")
    p.add_argument(
        "--crop-scale-factor",
        type=float,
        default=1.2,
        help="Fator de expansão da caixa detectada antes de recortar (1.2 = 20%% maior em cada dimensão, proporcional ao tamanho de cada Daphnia, não pixels fixos)",
    )

    # Faixa de plausibilidade agora é uma PROPORÇÃO (medida do Ruler /
    # diagonal da caixa detectada), não pixels fixos — assim se adapta
    # automaticamente a qualquer zoom, sem precisar de um valor global.
    p.add_argument("--min-proporcao", type=float, default=0.5, help="Proporção mínima (medida Ruler / diagonal da caixa) considerada plausível")
    p.add_argument("--max-proporcao", type=float, default=2.0, help="Proporção máxima considerada plausível")
    p.add_argument("--measurement-column", default="body.Length.h")

    return p.parse_args()


def main():
    args = parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)

    # 1. Extrair dataset (se necessário)
    if args.cvat_zip is not None:
        cvat_zip_path = args.cvat_zip.resolve()
        cvat_dir = (args.workdir / "cvat_export_extracted").resolve()
        if cvat_dir.exists():
            shutil.rmtree(cvat_dir)
        cvat_dir.mkdir(parents=True)
        print(f"Extraindo {cvat_zip_path} -> {cvat_dir}")
        with zipfile.ZipFile(cvat_zip_path) as archive:
            archive.extractall(cvat_dir)
    else:
        cvat_dir = args.cvat_dir

    # 2. Ler as imagens (usamos read_cvat só para ter a lista de imagens
    #    do dataset; não usamos as anotações aqui, é a detecção do MODELO
    #    que decide onde estão os Daphnias)
    print("Listando imagens do dataset...")
    classes, records = read_cvat(cvat_dir, args.format)
    image_paths = sorted({path for path, *_ in records})
    print(f"{len(image_paths)} imagens encontradas")

    # 3. Carregar modelo
    device = None
    if args.model_type == "fasterrcnn":
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device:", device)
        model = load_fasterrcnn(args.checkpoint, device)
    else:
        from ultralytics import YOLO

        model = YOLO(str(args.checkpoint))

    # 4. Rodar detecção em todas as imagens e recortar cada Daphnia
    #    encontrado, com margem PROPORCIONAL ao tamanho de cada caixa (não
    #    pixels fixos, se adapta a qualquer zoom), um arquivo por Daphnia
    #    (formato esperado pelo Ruler)
    crops_dir = args.workdir / "recortes_todas_imagens"
    if crops_dir.exists():
        shutil.rmtree(crops_dir)
    crops_dir.mkdir(parents=True)

    crop_metadata = []  # crop_name -> {image_name, confianca, indice_na_imagem}
    total_detections = 0
    for image_path in image_paths:
        if args.model_type == "fasterrcnn":
            boxes, scores = predict_fasterrcnn(model, image_path, device, args.score_threshold)
        else:
            boxes, scores = predict_yolo(model, image_path, args.score_threshold)

        image = Image.open(image_path).convert("RGB")
        for i, (box, score) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = box
            box_width = x2 - x1
            box_height = y2 - y1
            diagonal_caixa_px = math.hypot(box_width, box_height)
            # Expande a caixa proporcionalmente ao seu próprio tamanho (não
            # pixels fixos), para que Daphnias grandes e pequenos ganhem a
            # mesma % extra de espaço ao redor, independente do zoom da imagem.
            extra_w = box_width * (args.crop_scale_factor - 1) / 2
            extra_h = box_height * (args.crop_scale_factor - 1) / 2
            x1 = max(0, int(x1 - extra_w))
            y1 = max(0, int(y1 - extra_h))
            x2 = min(image.width, int(x2 + extra_w))
            y2 = min(image.height, int(y2 + extra_h))

            crop_name = f"{total_detections:04d}_{image_path.stem}_d{i}.png"
            image.crop((x1, y1, x2, y2)).save(crops_dir / crop_name)
            crop_metadata.append({
                    "crop_name": crop_name,
                    "image_name": image_path.name,
                    "confianca_deteccao": round(score, 4),
                    "diagonal_caixa_px": diagonal_caixa_px,
                    "largura_caixa_px": box_width, 
                    "altura_caixa_px": box_height
                    "x1": x1, 
                    "y1": y1, 
                    "x2": x2, 
                    "y2": y2
                })
            total_detections += 1

            csv_path = crops_dir / "crop_metadata.csv" 
            with open(csv_path, "w", newline="", encoding="utf-8") as csvfile: 
                fieldnames = [ "crop_name", "image_name", "confianca_deteccao", "x1", "y1", "x2", "y2", "largura_caixa_px", "altura_caixa_px", "diagonal_caixa_px"] 
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames) 
                writer.writeheader() 
                writer.writerows(crop_metadata) 
                print(f"CSV salvo em: {csv_path}") 
                print(f"Total de detecções: {len(crop_metadata)}")

        print(f"  {image_path.name}: {len(boxes)} detecção(ões)")

    print(f"\nTotal de {total_detections} Daphnias detectados e recortados em {len(image_paths)} imagens")

    # 5. Rodar o Daphnia Ruler em todos os recortes de uma vez. Se falhar
    #    (bug conhecido do daphruler: quebra quando UM item retorna None
    #    internamente, em vez de pular esse item), isola automaticamente
    #    o(s) recorte(s) problemático(s) testando um por vez, remove-os, e
    #    tenta de novo com o restante — sem intervenção manual.
    print(f"\nRodando python -m daphruler em {crops_dir} (pode demorar, são {total_detections} imagens)...")
    cmd = [sys.executable, "-m", "daphruler", "-p", str(crops_dir)]
    result = subprocess.run(cmd)

    excluded_crops = []
    if result.returncode != 0:
        print(
            "\n[aviso] python -m daphruler falhou processando a pasta inteira "
            "(bug conhecido: uma imagem retornando resultado inválido quebra a "
            "montagem da tabela final). Isolando o(s) recorte(s) problemático(s) "
            "testando um por vez..."
        )
        all_crop_paths = sorted(crops_dir.glob("*.png"))
        good_crops = []
        for crop_path in all_crop_paths:
            test_dir = args.workdir / "_teste_isolamento"
            if test_dir.exists():
                shutil.rmtree(test_dir)
            test_dir.mkdir(parents=True)
            shutil.copy2(crop_path, test_dir / crop_path.name)
            single_result = subprocess.run(
                [sys.executable, "-m", "daphruler", "-p", str(test_dir), "-n"],
                capture_output=True,
            )
            shutil.rmtree(test_dir)
            if single_result.returncode != 0:
                excluded_crops.append(crop_path.name)
                crop_path.unlink()  # remove da pasta principal, para a nova tentativa em lote
                print(f"  [excluído] {crop_path.name} — causa falha no daphruler, removido da análise")
            else:
                good_crops.append(crop_path.name)

        print(f"\n{len(excluded_crops)} recorte(s) excluído(s), {len(good_crops)} restantes. Rodando novamente em lote...")
        cmd = [sys.executable, "-m", "daphruler", "-p", str(crops_dir)]
        subprocess.run(cmd, check=True)

    if excluded_crops:
        print(
            f"\n[nota metodológica] {len(excluded_crops)} recorte(s) foram excluídos da análise por "
            "causarem falha interna no daphruler (não simplesmente 'medida ruim', mas erro de "
            "processamento): " + ", ".join(excluded_crops)
        )

    # 6. Ler o resultado do Ruler
    ruler_csv = crops_dir / "results" / f"measurement_results_{crops_dir.name}.csv"
    if not ruler_csv.exists():
        candidates = list(args.workdir.rglob("measurement_results_*"))
        if not candidates:
            sys.exit("Não encontrei o arquivo de resultado do Daphnia Ruler")
        ruler_csv = candidates[0]

    ruler_measurements = {}
    with open(ruler_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_id = row["ID"]
            crop_name = raw_id.replace("\\", "/").split("/")[-1]
            value = row.get(args.measurement_column, "NA")
            ruler_measurements[crop_name] = None if value == "NA" else float(value)

    # 7. Classificar cada Daphnia: medida plausível, implausível, ou
    #    não medida (NA) pelo Ruler — usando a PROPORÇÃO entre a medida do
    #    Ruler e a diagonal da própria caixa detectada (se adapta a
    #    qualquer zoom, em vez de um valor fixo em pixels).
    out_csv = args.workdir / "avaliacao_pipeline_completo.csv"
    n_plausivel, n_implausivel, n_sem_medida, n_excluido = 0, 0, 0, 0
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["imagem", "confianca_deteccao", "medida_ruler_px", "diagonal_caixa_px", "proporcao", "status"])
        for meta in crop_metadata:
            if meta["crop_name"] in excluded_crops:
                status = "excluido_erro_daphruler"
                n_excluido += 1
                writer.writerow([meta["image_name"], meta["confianca_deteccao"], None, meta["diagonal_caixa_px"], None, status])
                continue
            value = ruler_measurements.get(meta["crop_name"])
            if value is None:
                status = "sem_medida"
                proporcao = None
                n_sem_medida += 1
            else:
                proporcao = value / meta["diagonal_caixa_px"]
                if args.min_proporcao <= proporcao <= args.max_proporcao:
                    status = "plausivel"
                    n_plausivel += 1
                else:
                    status = "implausivel"
                    n_implausivel += 1
            writer.writerow(
                [
                    meta["image_name"],
                    meta["confianca_deteccao"],
                    value,
                    round(meta["diagonal_caixa_px"], 1),
                    round(proporcao, 3) if proporcao is not None else None,
                    status,
                ]
            )

    total = n_plausivel + n_implausivel + n_sem_medida + n_excluido
    print(f"\n=== Resumo (sem nenhum filtro de pré-processamento) ===")
    print(f"Total de Daphnias detectados: {total}")
    print(f"  Plausíveis (proporção Ruler/caixa entre {args.min_proporcao} e {args.max_proporcao}): {n_plausivel} ({100*n_plausivel/total:.1f}%)")
    print(f"  Implausíveis (fora da faixa, ex: mediu o olho): {n_implausivel} ({100*n_implausivel/total:.1f}%)")
    print(f"  Sem medida (Ruler retornou NA): {n_sem_medida} ({100*n_sem_medida/total:.1f}%)")
    if n_excluido:
        print(f"  Excluídos por erro interno do daphruler: {n_excluido} ({100*n_excluido/total:.1f}%)")
    print(f"\nTabela detalhada salva em: {out_csv}")


if __name__ == "__main__":
    main()
