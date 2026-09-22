#!/usr/bin/env python
"""
Avalia um checkpoint Faster R-CNN no conjunto de TESTE, que não foi usado
em nenhum momento do treino. O split é reproduzido a partir do
run_config.json salvo pelo train_fasterrcnn.py.

Métricas (mesmo formato de CSV do evaluate_yolo.py):
    - mAP50 e mAP50-95        (não dependem de score-threshold)
    - Precision, Recall e F1  (num score-threshold fixo)
    - IoU médio das detecções corretas
    - Tempo médio de inferência por imagem

Uso:
    python evaluate_fasterrcnn.py \\
        --workdir /home/$USER/fasterrcnn_run \\
        --checkpoint /home/$USER/fasterrcnn_run/fasterrcnn_run/best_model.pth \\
        --cvat-zip /caminho/para/dataset.zip
"""

import argparse
import csv
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

from common import read_cvat, split_records_3way, match_predictions


def load_run_config(workdir):
    config_path = workdir / "run_config.json"
    if not config_path.exists():
        sys.exit(
            f"Não encontrei {config_path}. Rode train_fasterrcnn.py primeiro (mesmo --workdir), "
            "ou passe --format/--val-fraction/--test-fraction/--seed manualmente."
        )
    return json.loads(config_path.read_text())


def load_model(checkpoint_path, device):
    import torch
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint["class_names"]
    num_classes = checkpoint["num_classes"]

    # Recria a mesma arquitetura do treino e carrega os pesos salvos
    model = fasterrcnn_resnet50_fpn_v2(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_names


def predict_fasterrcnn(model, image_path, device, score_threshold):
    """Retorna (caixas, scores) das detecções com score >= score_threshold."""
    import torch
    from PIL import Image
    from torchvision.transforms.functional import pil_to_tensor

    image = Image.open(image_path).convert("RGB")
    tensor = pil_to_tensor(image).float().to(device) / 255
    with torch.no_grad():
        output = model([tensor])[0]
    boxes = output["boxes"].cpu().tolist()
    scores = output["scores"].cpu().tolist()
    keep = [i for i, s in enumerate(scores) if s >= score_threshold]
    return [boxes[i] for i in keep], [scores[i] for i in keep]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workdir", type=Path, required=True, help="Mesmo --workdir usado no train_fasterrcnn.py")
    p.add_argument("--checkpoint", type=Path, required=True, help="Ex: .../fasterrcnn_run/best_model.pth")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--cvat-zip", type=Path)
    src.add_argument("--cvat-dir", type=Path)

    p.add_argument("--format", choices=["coco", "yolo", "voc"], default=None, help="Default: lido do run_config.json")
    p.add_argument("--val-fraction", type=float, default=None, help="Default: lido do run_config.json")
    p.add_argument("--test-fraction", type=float, default=None, help="Default: lido do run_config.json")
    p.add_argument("--seed", type=int, default=None, help="Default: lido do run_config.json")

    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--score-threshold", type=float, default=0.5, help="Usado só para Precision/Recall/F1")
    p.add_argument("--output-csv", type=Path, default=None)

    return p.parse_args()


def main():
    args = parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)

    # Parâmetros do split: da linha de comando, se passados; senão, do run_config.json
    run_config = load_run_config(args.workdir)
    fmt = args.format or run_config["format"]
    val_fraction = args.val_fraction if args.val_fraction is not None else run_config["val_fraction"]
    # run_config de versões antigas (sem conjunto de teste) não têm test_fraction
    if "test_fraction" not in run_config and args.test_fraction is None:
        sys.exit(
            "run_config.json não tem 'test_fraction' (foi gerado por uma versão antiga do "
            "train_fasterrcnn.py, com split de 2 vias). Re-treine com o script atualizado, "
            "ou passe --test-fraction manualmente (mas o resultado não será o mesmo conjunto "
            "de teste que teria sido reservado durante o treino)."
        )
    test_fraction = args.test_fraction if args.test_fraction is not None else run_config["test_fraction"]
    seed = args.seed if args.seed is not None else run_config["seed"]
    print(
        f"Usando format={fmt} val_fraction={val_fraction} test_fraction={test_fraction} "
        f"seed={seed} (do run_config.json, salvo no treino)"
    )

    # 1. Extrair o zip
    if args.cvat_zip is not None:
        cvat_zip_path = args.cvat_zip.resolve()
        cvat_dir = (args.workdir / "cvat_export_eval").resolve()
        if cvat_dir.exists():
            shutil.rmtree(cvat_dir)
        cvat_dir.mkdir(parents=True)
        print(f"Extraindo {cvat_zip_path} -> {cvat_dir}")
        with zipfile.ZipFile(cvat_zip_path) as archive:
            archive.extractall(cvat_dir)
    else:
        cvat_dir = args.cvat_dir
        if not cvat_dir.exists():
            sys.exit(f"--cvat-dir não existe: {cvat_dir}")

    # 2. Reproduzir o split do treino e ficar só com o teste
    print("Lendo anotações e reproduzindo o split treino/validação/teste do treino...")
    classes, records = read_cvat(cvat_dir, fmt)
    _, _, test_records = split_records_3way(records, val_fraction, test_fraction, seed)
    print(f"Avaliando em {len(test_records)} imagens de TESTE (nunca vistas no treino nem no early stopping)")

    # 3. Carregar o modelo
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model, model_classes = load_model(args.checkpoint, device)
    print("Classes do checkpoint:", model_classes)

    # 4. Precision / Recall / F1 no score-threshold escolhido
    total_tp, total_fp, total_fn = 0, 0, 0
    all_matched_ious = []
    total_time = 0.0

    for path, width, height, gt_boxes, gt_labels in test_records:
        start = time.time()
        pred_boxes, pred_scores = predict_fasterrcnn(model, path, device, args.score_threshold)
        total_time += time.time() - start

        tp, fp, fn, ious = match_predictions(pred_boxes, pred_scores, gt_boxes, args.iou_threshold)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        all_matched_ious.extend(ious)

    n_images = len(test_records)
    per_image_ms = (total_time / n_images) * 1000 if n_images else 0.0

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = sum(all_matched_ious) / len(all_matched_ious) if all_matched_ious else 0.0

    print(f"\n=== Métricas no conjunto de TESTE, score>={args.score_threshold}, IoU>={args.iou_threshold} ===")
    print(f"TP: {total_tp}  FP: {total_fp}  FN: {total_fn}")
    print(f"Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")
    print(f"IoU médio (detecções corretas): {mean_iou:.4f}")
    print(f"Tempo médio de inferência: {per_image_ms:.1f} ms/imagem")

    # 5. mAP: usa todas as predições (score >= 0), porque a métrica varre os thresholds.
    #    Labels todos 0 porque há uma classe só.
    map50 = map50_95 = None
    try:
        from torchmetrics.detection.mean_ap import MeanAveragePrecision

        metric = MeanAveragePrecision(iou_type="bbox")
        for path, width, height, gt_boxes, gt_labels in test_records:
            pred_boxes, pred_scores = predict_fasterrcnn(model, path, device, 0.0)
            pred = {
                "boxes": torch.tensor(pred_boxes).reshape(-1, 4) if pred_boxes else torch.zeros((0, 4)),
                "scores": torch.tensor(pred_scores) if pred_scores else torch.zeros((0,)),
                "labels": torch.zeros(len(pred_boxes), dtype=torch.int64),
            }
            gt = {
                "boxes": torch.tensor(gt_boxes).reshape(-1, 4) if gt_boxes else torch.zeros((0, 4)),
                "labels": torch.zeros(len(gt_boxes), dtype=torch.int64),
            }
            metric.update([pred], [gt])
        results = metric.compute()
        map50 = results["map_50"].item()
        map50_95 = results["map"].item()
        print(f"mAP50:     {map50:.4f}")
        print(f"mAP50-95:  {map50_95:.4f}")
    except ImportError:
        print("torchmetrics não instalado — pulando mAP (pip install torchmetrics pycocotools)")

    # 6. Salvar CSV (mesmo formato do evaluate_yolo.py)
    out_csv = args.output_csv or (args.workdir / "eval_fasterrcnn.csv")
    with out_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["model_type", "fasterrcnn"])
        writer.writerow(["eval_split", "test"])
        writer.writerow(["n_test_images", n_images])
        writer.writerow(["iou_threshold", args.iou_threshold])
        writer.writerow(["score_threshold", args.score_threshold])
        writer.writerow(["TP", total_tp])
        writer.writerow(["FP", total_fp])
        writer.writerow(["FN", total_fn])
        writer.writerow(["precision", precision])
        writer.writerow(["recall", recall])
        writer.writerow(["f1_score", f1])
        writer.writerow(["mean_iou_correct_detections", mean_iou])
        writer.writerow(["inference_ms_per_image", per_image_ms])
        if map50 is not None:
            writer.writerow(["mAP50", map50])
            writer.writerow(["mAP50_95", map50_95])
    print(f"\nMétricas salvas em: {out_csv}")


if __name__ == "__main__":
    main()
