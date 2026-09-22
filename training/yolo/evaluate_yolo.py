#!/usr/bin/env python
"""
Avalia um checkpoint YOLO treinado no conjunto de TESTE (não o de validação
usado durante o treino pelo Ultralytics) — reproduzido via seed +
val_fraction + test_fraction, lidos automaticamente do run_config.json
salvo pelo train_yolo.py (não precisa digitar de novo).

O conjunto de teste nunca influenciou o treino de forma alguma (ele fica
fora do data.yaml que o Ultralytics enxerga — nem para ajustar pesos, nem
para decidir quando parar via early stopping), o que torna esta avaliação
mais confiável do que reportar métricas sobre o conjunto de validação.

Métricas calculadas (mesmo conjunto usado no evaluate_fasterrcnn.py, para a
comparação final fazer sentido):
    - mAP50 e mAP50-95         (métricas principais — não dependem de threshold)
    - Precision / Recall / F1  (num score-threshold fixo, complementar)
    - IoU médio das detecções corretas
    - Tempo médio de inferência por imagem

Uso:
    python evaluate_yolo.py \
        --workdir /scratch/$USER/yolo_run \
        --checkpoint /scratch/$USER/yolo_run/yolo_run/weights/best.pt \
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
            f"Não encontrei {config_path}. Rode train_yolo.py primeiro (mesmo --workdir), "
            "ou passe --format/--val-fraction/--test-fraction/--seed manualmente."
        )
    return json.loads(config_path.read_text())


def predict_yolo(model, image_path, score_threshold):
    result = model.predict(source=str(image_path), conf=score_threshold, verbose=False)[0]
    boxes = result.boxes.xyxy.cpu().tolist()
    scores = result.boxes.conf.cpu().tolist()
    return boxes, scores


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workdir", type=Path, required=True, help="Mesmo --workdir usado no train_yolo.py")
    p.add_argument("--checkpoint", type=Path, required=True, help="Ex: .../weights/best.pt")

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

    run_config = load_run_config(args.workdir)
    fmt = args.format or run_config["format"]
    val_fraction = args.val_fraction if args.val_fraction is not None else run_config["val_fraction"]
    # run_config antigos (split de 2 vias) não têm test_fraction — nesse caso,
    # avisa claramente em vez de inventar um valor.
    if "test_fraction" not in run_config and args.test_fraction is None:
        sys.exit(
            "run_config.json não tem 'test_fraction' (foi gerado por uma versão antiga do "
            "train_yolo.py, com split de 2 vias). Re-treine com o script atualizado, "
            "ou passe --test-fraction manualmente (mas o resultado não será o mesmo conjunto "
            "de teste que teria sido reservado durante o treino)."
        )
    test_fraction = args.test_fraction if args.test_fraction is not None else run_config["test_fraction"]
    seed = args.seed if args.seed is not None else run_config["seed"]
    print(
        f"Usando format={fmt} val_fraction={val_fraction} test_fraction={test_fraction} "
        f"seed={seed} (do run_config.json, salvo no treino)"
    )

    # 1. Extrair (se necessário)
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

    # 2. Reproduzir o MESMO split usado no treino, e usar só o conjunto de TESTE
    print("Lendo anotações e reproduzindo o split treino/validação/teste do treino...")
    classes, records = read_cvat(cvat_dir, fmt)
    _, _, test_records = split_records_3way(records, val_fraction, test_fraction, seed)
    print(f"Avaliando em {len(test_records)} imagens de TESTE (nunca vistas no treino nem no early stopping)")

    # 3. Carregar modelo
    from ultralytics import YOLO

    model = YOLO(str(args.checkpoint))
    print("Classes do checkpoint:", list(model.names.values()))

    # 4. Inferência + matching em score-threshold fixo (para Precision/Recall/F1)
    total_tp, total_fp, total_fn = 0, 0, 0
    all_matched_ious = []
    total_time = 0.0

    for path, width, height, gt_boxes, gt_labels in test_records:
        start = time.time()
        pred_boxes, pred_scores = predict_yolo(model, path, args.score_threshold)
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

    # 5. mAP50 / mAP50-95 (métrica principal, não depende de threshold)
    map50 = map50_95 = None
    try:
        import torch
        from torchmetrics.detection.mean_ap import MeanAveragePrecision

        metric = MeanAveragePrecision(iou_type="bbox")
        for path, width, height, gt_boxes, gt_labels in test_records:
            pred_boxes, pred_scores = predict_yolo(model, path, 0.0)
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

    # 6. Salvar CSV — mesmo formato usado pelo evaluate_fasterrcnn.py
    out_csv = args.output_csv or (args.workdir / "eval_yolo.csv")
    with out_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["model_type", "yolo"])
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
