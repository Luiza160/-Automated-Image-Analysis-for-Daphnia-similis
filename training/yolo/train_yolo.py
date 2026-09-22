#!/usr/bin/env python
"""
Treina um YOLO (Ultralytics) a partir de anotações exportadas do CVAT.

Etapas:
    1. Extrai o zip do CVAT (ou usa uma pasta já extraída).
    2. Separa as imagens em treino / validação / teste (common.py).
    3. Monta o dataset no formato do Ultralytics (images/, labels/, data.yaml)
       só com treino e validação. O teste fica de fora.
    4. Gera cópias augmentadas das imagens de treino.
    5. Treina e salva best_real.pt (ver correção do EMA abaixo).
    6. Salva um run_config.json, usado pelo evaluate_yolo.py para
       reproduzir o mesmo split.

Uso:
    python train_yolo.py \\
        --cvat-zip /caminho/para/dataset.zip \\
        --format coco \\
        --workdir /home/$USER/yolo_run \\
        --epochs 200 --patience 20 --lr0 0.001 --batch 16 --imgsz 640
"""

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

from PIL import Image
import yaml
import numpy as np
import albumentations as A

from common import read_cvat, split_records_3way


# --------------------------------------------------------------------------
# Augmentação (a mesma usada no train_fasterrcnn.py)
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


def apply_augmentation(pipeline, image_path, boxes, labels, seed):
    """Aplica o pipeline uma vez numa imagem e retorna a imagem e as caixas transformadas."""
    rng_state = np.random.RandomState(seed)
    image = np.array(Image.open(image_path).convert("RGB"))
    # O Albumentations rejeita caixas fora da imagem; recorta nas bordas
    h, w = image.shape[:2]
    safe_boxes = []
    safe_labels = []
    for (x1, y1, x2, y2), label in zip(boxes, labels):
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            safe_boxes.append([x1, y1, x2, y2])
            safe_labels.append(label)
    transformed = pipeline(image=image, bboxes=safe_boxes, labels=safe_labels)
    return transformed["image"], transformed["bboxes"], transformed["labels"], rng_state


# --------------------------------------------------------------------------
# Dataset no formato do Ultralytics
# --------------------------------------------------------------------------
def build_yolo_dataset(cvat_dir, dataset_dir, fmt, val_fraction, test_fraction, seed, n_augmentations):
    classes, records = read_cvat(cvat_dir, fmt)
    train_records, val_records, test_records = split_records_3way(
        records, val_fraction, test_fraction, seed
    )
    # O teste não entra no data.yaml: o Ultralytics nunca o vê durante o treino
    splits = {"train": train_records, "val": val_records}

    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    pipeline = build_augmentation_pipeline() if n_augmentations > 0 else None

    counts = {}
    for split, items in splits.items():
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        def write_sample(name, image_array_or_path, width, height, boxes, labels):
            if isinstance(image_array_or_path, Path):
                shutil.copy2(image_array_or_path, img_dir / name)
            else:
                Image.fromarray(image_array_or_path).save(img_dir / name)
            # Formato YOLO: classe, centro x, centro y, largura, altura (normalizados de 0 a 1)
            rows = []
            for (x1, y1, x2, y2), label in zip(boxes, labels):
                rows.append(
                    f"{label} {(x1 + x2) / (2 * width):.8f} {(y1 + y2) / (2 * height):.8f} "
                    f"{(x2 - x1) / width:.8f} {(y2 - y1) / height:.8f}"
                )
            (lbl_dir / Path(name).with_suffix(".txt")).write_text("\n".join(rows))

        n_written = 0
        for i, (source, width, height, boxes, labels) in enumerate(items):
            base_name = f"{i:05d}_{source.name}"
            write_sample(base_name, source, width, height, boxes, labels)
            n_written += 1

            # Só no treino: salva n_augmentations cópias augmentadas além da original
            if split == "train" and pipeline is not None and boxes:
                for a in range(n_augmentations):
                    aug_image, aug_boxes, aug_labels, _ = apply_augmentation(
                        pipeline, source, boxes, labels, seed=seed * 100000 + i * 10 + a
                    )
                    if not aug_boxes:
                        continue  # a augmentação pode ter eliminado todas as caixas
                    aug_height, aug_width = aug_image.shape[0], aug_image.shape[1]
                    aug_name = f"{i:05d}_aug{a}_{Path(source.name).stem}.jpg"
                    write_sample(aug_name, aug_image, aug_width, aug_height, aug_boxes, aug_labels)
                    n_written += 1

        counts[split] = n_written

    config = {
        "path": str(dataset_dir),
        "train": "images/train",
        "val": "images/val",
        "names": dict(enumerate(classes)),
    }
    data_yaml = dataset_dir / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(config, sort_keys=False))

    print("Classes:", classes)
    print("Imagens por split (incluindo augmentadas no treino):", counts)
    print("Imagens de teste (fora do data.yaml, reservadas para avaliação final):", len(test_records))
    return data_yaml, classes


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--cvat-zip", type=Path, help="ZIP exportado do CVAT")
    src.add_argument("--cvat-dir", type=Path, help="Pasta já extraída do export CVAT")

    p.add_argument("--format", choices=["coco", "yolo", "voc"], required=True)
    p.add_argument("--workdir", type=Path, required=True, help="Diretório de trabalho (dataset + runs)")
    p.add_argument("--val-fraction", type=float, default=0.15, help="Fração usada para validação durante o treino")
    p.add_argument("--test-fraction", type=float, default=0.15, help="Fração reservada para avaliação final (nunca vista no treino)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--n-augmentations",
        type=int,
        default=2,
        help="Nº de cópias augmentadas geradas por imagem de TREINO (0 desativa augmentação)",
    )

    p.add_argument("--model", default="yolo11n.pt", help="Checkpoint base do Ultralytics")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0", help="ex: '0', '0,1', ou 'cpu'")
    p.add_argument("--run-name", default="yolo_run")
    p.add_argument("--lr0", type=float, default=0.001)

    p.add_argument("--skip-training", action="store_true", help="Só monta o dataset, não treina")

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

    # 2. Montar o dataset (split + augmentação)
    dataset_dir = args.workdir / "yolo_dataset"
    print("Lendo anotações, criando split treino/validação/teste e aplicando augmentação...")
    data_yaml, classes = build_yolo_dataset(
        cvat_dir, dataset_dir, args.format, args.val_fraction, args.test_fraction, args.seed, args.n_augmentations
    )

    # 3. Salvar a configuração do split, lida depois pelo evaluate_yolo.py
    run_config = {
        "model_type": "yolo",
        "format": args.format,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "seed": args.seed,
        "classes": classes,
        "run_name": args.run_name,
    }
    (args.workdir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    if args.skip_training:
        print("`--skip-training` ativo: dataset pronto em", dataset_dir)
        return

    # 4. Treinar a partir do modelo pré-treinado
    from ultralytics import YOLO

    model = YOLO(args.model)

    # ------------------------------------------------------------------
    # Correção do EMA
    # O Ultralytics salva no best.pt uma média móvel (EMA) dos pesos, e não
    # os pesos da melhor época. Essa média é calibrada para milhares de
    # atualizações; com poucas imagens ela não converge, e o best.pt fica
    # pior do que o modelo realmente foi. Por isso, ao fim de cada época,
    # salvamos nós mesmos os pesos reais em best_real.pt quando a métrica
    # (fitness) melhora.
    # ------------------------------------------------------------------
    best_state = {"fitness": None}

    def save_real_best_callback(trainer):
        import io
        from copy import deepcopy

        import torch

        fitness = trainer.fitness
        if fitness is None:
            return
        if best_state["fitness"] is None or fitness > best_state["fitness"]:
            best_state["fitness"] = fitness
            real_weights_dir = Path(trainer.save_dir) / "weights"
            real_weights_dir.mkdir(parents=True, exist_ok=True)

            # Mesmo formato do checkpoint do Ultralytics, mas com os pesos reais
            # (trainer.model) em vez do EMA. Assim YOLO("best_real.pt") carrega normalmente.
            real_model = deepcopy(trainer.model).half().to(memory_format=torch.contiguous_format)
            buffer = io.BytesIO()
            torch.save(
                {
                    "epoch": trainer.epoch,
                    "best_fitness": fitness,
                    "model": real_model,
                    "ema": None,
                    "updates": None,
                    "optimizer": None,
                    "train_args": vars(trainer.args),
                    "train_metrics": {**trainer.metrics, "fitness": fitness},
                    "train_results": {},
                    "date": None,
                    "version": None,
                },
                buffer,
            )
            (real_weights_dir / "best_real.pt").write_bytes(buffer.getvalue())
            print(f"[best_real] Novo melhor (fitness={fitness:.4f}) salvo em best_real.pt, sem usar EMA")

    model.add_callback("on_fit_epoch_end", save_real_best_callback)

    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.workdir),
        name=args.run_name,
        exist_ok=True,
        seed=args.seed,
        plots=True,
        lr0=args.lr0,
    )
    run_dir = Path(model.trainer.save_dir)
    print("Best checkpoint (Ultralytics, via EMA — pode estar degradado em datasets pequenos):", run_dir / "weights/best.pt")
    print("Best checkpoint (pesos reais, recomendado):", run_dir / "weights/best_real.pt")

    # 5. Métricas por época (geradas pelo Ultralytics, na validação)
    import pandas as pd

    metrics = pd.read_csv(run_dir / "results.csv")
    metrics.columns = metrics.columns.str.strip()
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    print(metrics.tail())

    # 6. Registrar os checkpoints no run_config (o recomendado é o best_real.pt)
    run_config["best_checkpoint"] = str(run_dir / "weights/best_real.pt")
    run_config["best_checkpoint_ema_ultralytics"] = str(run_dir / "weights/best.pt")
    run_config["run_dir"] = str(run_dir)
    (args.workdir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    # 7. Compactar os resultados
    archive_path = shutil.make_archive(str(args.workdir / f"{args.run_name}_results"), "zip", run_dir)
    print("Resultados empacotados em:", archive_path)


if __name__ == "__main__":
    main()
