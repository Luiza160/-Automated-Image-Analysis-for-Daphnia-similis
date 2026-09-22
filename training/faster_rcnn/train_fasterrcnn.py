#!/usr/bin/env python
"""
Treina um Faster R-CNN (TorchVision) a partir de anotações exportadas do CVAT.

Etapas:
    1. Extrai o zip do CVAT (ou usa uma pasta já extraída).
    2. Separa as imagens em treino / validação / teste (common.py).
    3. Treina com augmentação só nas imagens de treino.
    4. Para quando a loss de validação deixa de melhorar (early stopping).
    5. Salva checkpoints, métricas por época, gráficos e um run_config.json,
       usado pelo evaluate_fasterrcnn.py para reproduzir o mesmo split.

Uso:
    python train_fasterrcnn.py \\
        --cvat-zip /caminho/para/dataset.zip \\
        --format coco \\
        --workdir /home/$USER/fasterrcnn_run \\
        --epochs 200 --patience 20 --lr 0.001
"""

import argparse
import csv
import json
import random
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
import albumentations as A

from common import read_cvat, split_records_3way


# --------------------------------------------------------------------------
# Augmentação (a mesma usada no train_yolo.py)
# --------------------------------------------------------------------------
def build_augmentation_pipeline():
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=7, p=1.0),
                    A.MedianBlur(blur_limit=7, p=1.0),
                    A.GaussianBlur(blur_limit=7, p=1.0),
                ],
                p=0.3,
            ),
            A.OneOf(
                [
                    A.GaussNoise(std_range=(0.0124, 0.0277), p=1.0),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                ],
                p=0.2,
            ),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.3, contrast_limit=0.3, brightness_by_max=True, p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5
            ),
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(8, 32),
                hole_width_range=(8, 32),
                fill=0,
                p=0.2,
            ),
        ],
        # Transforma as caixas junto com a imagem; descarta caixas com menos de 20% visível
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], min_visibility=0.2),
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--cvat-zip", type=Path, help="ZIP exportado do CVAT")
    src.add_argument("--cvat-dir", type=Path, help="Pasta já extraída do export CVAT")

    p.add_argument("--format", choices=["coco", "yolo", "voc"], required=True)
    p.add_argument("--workdir", type=Path, required=True, help="Diretório de trabalho (runs/checkpoints)")
    p.add_argument("--val-fraction", type=float, default=0.15, help="Fração usada só para early stopping")
    p.add_argument("--test-fraction", type=float, default=0.15, help="Fração reservada para avaliação final (nunca vista no treino)")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20, help="Épocas sem melhora na val_loss antes de parar")
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=0.0005)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--run-name", default="fasterrcnn_run")
    p.add_argument("--min-delta", type=float, default=0.001, help="Melhora mínima na val_loss para resetar patience")
    p.add_argument("--no-augmentation", action="store_true", help="Desativa augmentação no treino")

    p.add_argument("--skip-training", action="store_true", help="Só monta o split, não treina")

    return p.parse_args()


def main():
    args = parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)

    # 1. Extrair o zip numa pasta própria, sem tocar no arquivo original
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
        if not cvat_dir.exists():
            sys.exit(f"--cvat-dir não existe: {cvat_dir}")

    # 2. Ler anotações e separar treino / validação / teste
    print("Lendo anotações e criando split treino/validação/teste...")
    classes, records = read_cvat(cvat_dir, args.format)
    train_records, val_records, test_records = split_records_3way(
        records, args.val_fraction, args.test_fraction, args.seed
    )
    print("Classes:", classes)
    print(
        "Train images:", len(train_records),
        "Validation images:", len(val_records),
        "Test images:", len(test_records),
    )

    # 3. Salvar a configuração do split, lida depois pelo evaluate_fasterrcnn.py
    run_config = {
        "model_type": "fasterrcnn",
        "format": args.format,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "seed": args.seed,
        "classes": classes,
        "run_name": args.run_name,
        "n_train": len(train_records),
        "n_val": len(val_records),
        "n_test": len(test_records),
    }
    (args.workdir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    if args.skip_training:
        print("`--skip-training` ativo: split pronto, nada foi treinado")
        return

    # 4. Dataset e DataLoaders
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torchvision.transforms.functional import pil_to_tensor

    augmentation_pipeline = None if args.no_augmentation else build_augmentation_pipeline()

    class DaphniaDataset(Dataset):
        def __init__(self, records, pipeline=None):
            self.records = records
            self.pipeline = pipeline

        def __len__(self):
            return len(self.records)

        def __getitem__(self, i):
            path, width, height, boxes, labels = self.records[i]
            image_pil = Image.open(path).convert("RGB")

            if self.pipeline is not None and boxes:
                image_np = np.array(image_pil)
                h, w = image_np.shape[:2]
                # O Albumentations rejeita caixas fora da imagem; recorta nas bordas
                safe_boxes, safe_labels = [], []
                for (x1, y1, x2, y2), label in zip(boxes, labels):
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if x2 > x1 and y2 > y1:
                        safe_boxes.append([x1, y1, x2, y2])
                        safe_labels.append(label)
                transformed = self.pipeline(image=image_np, bboxes=safe_boxes, labels=safe_labels)
                image = pil_to_tensor(Image.fromarray(transformed["image"])).float() / 255
                boxes_t = torch.tensor(transformed["bboxes"], dtype=torch.float32).reshape(-1, 4)
                labels_t = torch.tensor(transformed["labels"], dtype=torch.int64) + 1  # 0 = background
                # Se a augmentação eliminou todas as caixas, usa a imagem original
                if boxes_t.numel() == 0:
                    image = pil_to_tensor(image_pil).float() / 255
                    boxes_t = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
                    labels_t = torch.tensor(labels, dtype=torch.int64) + 1
            else:
                image = pil_to_tensor(image_pil).float() / 255
                boxes_t = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
                labels_t = torch.tensor(labels, dtype=torch.int64) + 1  # 0 = background

            target = {"boxes": boxes_t, "labels": labels_t}
            return image, target

    def collate(batch):
        # Imagens podem ter tamanhos e números de caixas diferentes, então não empilha
        return tuple(zip(*batch))

    train_loader = DataLoader(
        DaphniaDataset(train_records, pipeline=augmentation_pipeline),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        DaphniaDataset(val_records, pipeline=None),  # validação nunca é augmentada
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    # test_records não é usado aqui: fica só para o evaluate_fasterrcnn.py

    # 5. Modelo pré-treinado no COCO, com a última camada trocada para as nossas classes
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    run_dir = args.workdir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(classes) + 1)  # +1 = background
    model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    def mean_loss(loader, training):
        # O Faster R-CNN só retorna as losses em modo train, mesmo na validação
        model.train()
        total = 0.0
        for images, targets in loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with torch.set_grad_enabled(training):
                loss = sum(model(images, targets).values())
                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            total += loss.item()
        return total / len(loader)

    # mAP a cada época na validação, só para acompanhar o treino.
    # As métricas finais vêm do evaluate_fasterrcnn.py, no conjunto de teste.
    try:
        from torchmetrics.detection.mean_ap import MeanAveragePrecision
    except ImportError:
        sys.exit("torchmetrics não está instalado. Rode: pip install torchmetrics pycocotools")

    def detection_metrics(loader):
        model.eval()
        metric = MeanAveragePrecision(iou_type="bbox")
        with torch.no_grad():
            for images, targets in loader:
                images_dev = [img.to(device) for img in images]
                outputs = model(images_dev)
                preds = [
                    {"boxes": out["boxes"].cpu(), "scores": out["scores"].cpu(), "labels": out["labels"].cpu()}
                    for out in outputs
                ]
                gts = [{"boxes": t["boxes"], "labels": t["labels"]} for t in targets]
                metric.update(preds, gts)
        model.train()
        results = metric.compute()
        return {
            "mAP50": results["map_50"].item(),
            "mAP50_95": results["map"].item(),
            "mAP75": results["map_75"].item(),
            "mAR_100": results["mar_100"].item(),
        }

    # 6. Loop de treino com early stopping pela val_loss
    best_loss, waiting = float("inf"), 0
    metrics_path = run_dir / "metrics.csv"
    fieldnames = [
        "epoch", "train_loss", "val_loss",
        "mAP50", "mAP50_95", "mAP75", "mAR_100", "learning_rate",
    ]
    with metrics_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            train_loss = mean_loss(train_loader, training=True)
            with torch.no_grad():
                val_loss = mean_loss(val_loader, training=False)
            det_metrics = detection_metrics(val_loader)
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "mAP50": det_metrics["mAP50"],
                    "mAP50_95": det_metrics["mAP50_95"],
                    "mAP75": det_metrics["mAP75"],
                    "mAR_100": det_metrics["mAR_100"],
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )
            handle.flush()  # grava a cada época, para acompanhar durante o job
            print(
                f"Epoch {epoch:03d}: train={train_loss:.4f} val={val_loss:.4f} "
                f"mAP50={det_metrics['mAP50']:.4f} mAP50-95={det_metrics['mAP50_95']:.4f}"
            )

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "class_names": classes,
                "num_classes": len(classes) + 1,
            }
            torch.save(checkpoint, run_dir / "last_model.pth")

            if val_loss < best_loss - args.min_delta:
                best_loss, waiting = val_loss, 0
                torch.save(checkpoint, run_dir / "best_model.pth")
            else:
                waiting += 1
                if waiting >= args.patience:
                    print(f"Early stopping na época {epoch} (sem melhora por {args.patience} épocas)")
                    break

    print("Best checkpoint:", run_dir / "best_model.pth")

    # 7. Gráficos
    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")  # o cluster não tem tela
    import matplotlib.pyplot as plt

    metrics = pd.read_csv(metrics_path)

    metrics.plot(x="epoch", y=["train_loss", "val_loss"], grid=True, figsize=(8, 4))
    plt.tight_layout()
    plt.savefig(run_dir / "loss_curve.png", dpi=150)

    # Painéis no mesmo formato do results.png do YOLO, para comparar os dois
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    panels = [
        ("train_loss", "train/loss"),
        ("val_loss", "val/loss"),
        ("learning_rate", "learning_rate"),
        ("mAP50", "metrics/mAP50"),
        ("mAP50_95", "metrics/mAP50-95"),
        ("mAR_100", "metrics/mAR_100"),
    ]
    for ax, (col, title) in zip(axes.flat, panels):
        ax.plot(metrics["epoch"], metrics[col], marker="o", linewidth=1.5)
        ax.set_title(title)
        ax.grid(True)
        ax.set_xlabel("epoch")
    plt.tight_layout()
    plt.savefig(run_dir / "results.png", dpi=150)

    print(metrics.tail())

    # 8. Registrar o caminho do melhor checkpoint no run_config
    run_config["best_checkpoint"] = str(run_dir / "best_model.pth")
    run_config["run_dir"] = str(run_dir)
    (args.workdir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    # 9. Compactar os resultados
    archive_path = shutil.make_archive(str(args.workdir / f"{args.run_name}_results"), "zip", run_dir)
    print("Resultados empacotados em:", archive_path)


if __name__ == "__main__":
    main()
