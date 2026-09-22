"""
Funções compartilhadas pelos pipelines YOLO e Faster R-CNN:
leitura das anotações do CVAT, split treino/validação/teste e
matching entre predições e anotações.

Este arquivo deve ser idêntico em training/yolo e training/faster_rcnn,
para que os dois modelos usem os mesmos dados e a mesma avaliação.
"""

import json
import random
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image

IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# --------------------------------------------------------------------------
# Leitura das anotações do CVAT
# Todos os leitores retornam (classes, records), onde cada record é
# (caminho, largura, altura, caixas [x1, y1, x2, y2], labels).
# --------------------------------------------------------------------------
def images_in(root):
    return [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_TYPES]


def find_image(root, name):
    """Localiza a imagem pelo caminho da anotação ou, se não achar, pelo nome do arquivo."""
    direct = root / str(name).replace("\\", "/").lstrip("./")
    if direct.is_file():
        return direct
    matches = [p for p in images_in(root) if p.name == Path(name).name]
    if len(matches) != 1:
        raise FileNotFoundError(f"Cannot uniquely find image {name}")
    return matches[0]


def read_coco(root):
    found = []
    for path in root.rglob("*.json"):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if {"images", "annotations", "categories"} <= data.keys():
            found.append(data)
    if len(found) != 1:
        raise ValueError(
            f"Esperava exatamente um JSON COCO (images/annotations/categories); encontrei {len(found)}"
        )
    data = found[0]
    categories = sorted(data["categories"], key=lambda x: x["id"])
    class_id = {c["id"]: i for i, c in enumerate(categories)}  # ids do COCO -> 0, 1, 2...
    grouped = {}
    for ann in data["annotations"]:
        if not ann.get("iscrowd", 0):
            grouped.setdefault(ann["image_id"], []).append(ann)
    records = []
    for item in data["images"]:
        path = find_image(root, item["file_name"])
        width, height = item.get("width"), item.get("height")
        if not width or not height:
            width, height = Image.open(path).size
        boxes, labels = [], []
        for ann in grouped.get(item["id"], []):
            x, y, w, h = ann["bbox"]  # COCO usa [x, y, largura, altura]
            boxes.append([x, y, x + w, y + h])
            labels.append(class_id[ann["category_id"]])
        records.append((path, width, height, boxes, labels))
    return [c["name"] for c in categories], records


def read_voc(root):
    raw, names = [], set()
    for xml_path in root.rglob("*.xml"):
        try:
            node = ET.parse(xml_path).getroot()
        except ET.ParseError:
            continue
        if node.tag != "annotation":
            continue
        path = find_image(root, node.findtext("filename"))
        width, height = Image.open(path).size
        objects = []
        for obj in node.findall("object"):
            name, box = obj.findtext("name"), obj.find("bndbox")
            xyxy = [float(box.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
            objects.append((name, xyxy))
            names.add(name)
        raw.append((path, width, height, objects))
    classes = sorted(names)
    ids = {name: i for i, name in enumerate(classes)}
    records = [
        (p, w, h, [b for _, b in o], [ids[n] for n, _ in o]) for p, w, h, o in raw
    ]
    return classes, records


def read_yolo(root):
    names_file = next(iter(root.rglob("obj.names")), None) or next(
        iter(root.rglob("classes.txt")), None
    )
    if names_file is None:
        raise FileNotFoundError("Não encontrei obj.names nem classes.txt")
    classes = [line.strip() for line in names_file.read_text().splitlines() if line.strip()]
    records = []
    for path in images_in(root):
        width, height = Image.open(path).size
        # O .txt pode estar ao lado da imagem ou numa pasta labels/ paralela a images/
        label = path.with_suffix(".txt")
        if not label.exists() and "images" in path.parts:
            parts = list(path.parts)
            parts[parts.index("images")] = "labels"
            label = Path(*parts).with_suffix(".txt")
        boxes, labels = [], []
        if label.exists():
            for row in label.read_text().splitlines():
                if not row.strip():
                    continue
                # YOLO usa centro e tamanho normalizados (0 a 1); converte para pixels
                c, xc, yc, bw, bh = map(float, row.split()[:5])
                boxes.append(
                    [
                        (xc - bw / 2) * width,
                        (yc - bh / 2) * height,
                        (xc + bw / 2) * width,
                        (yc + bh / 2) * height,
                    ]
                )
                labels.append(int(c))
        records.append((path, width, height, boxes, labels))
    return classes, records


def read_cvat(root, fmt):
    """Lê a pasta extraída do CVAT no formato fmt ('coco', 'yolo' ou 'voc')."""
    reader = {"coco": read_coco, "voc": read_voc, "yolo": read_yolo}[fmt]
    return reader(root)


# --------------------------------------------------------------------------
# Split treino / validação / teste
# --------------------------------------------------------------------------
def split_records_3way(records, val_fraction, test_fraction, seed):
    """
    Separa as imagens em treino, validação (early stopping) e teste
    (avaliação final). Com o mesmo seed e as mesmas frações, o split é
    sempre o mesmo, o que permite reproduzi-lo na avaliação.
    """
    assert len(records) >= 3, "É preciso pelo menos três imagens (uma por conjunto)"
    assert val_fraction + test_fraction < 1.0, "val_fraction + test_fraction deve deixar imagens para o treino"

    records = list(records)
    random.Random(seed).shuffle(records)

    n_val = max(1, round(len(records) * val_fraction))
    n_test = max(1, round(len(records) * test_fraction))

    test_records = records[:n_test]
    val_records = records[n_test:n_test + n_val]
    train_records = records[n_test + n_val:]

    return train_records, val_records, test_records


def split_records(records, val_fraction, seed):
    """Split antigo, só treino/validação. Use split_records_3way."""
    assert len(records) >= 2, "É preciso pelo menos duas imagens"
    records = list(records)
    random.Random(seed).shuffle(records)
    n_val = max(1, round(len(records) * val_fraction))
    val_records, train_records = records[:n_val], records[n_val:]
    return train_records, val_records


# --------------------------------------------------------------------------
# IoU e matching entre predições e anotações
# --------------------------------------------------------------------------
def box_iou(box_a, box_b):
    """Intersection over Union entre duas caixas [x1, y1, x2, y2]."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def match_predictions(pred_boxes, pred_scores, gt_boxes, iou_threshold):
    """
    Pareia cada predição (da mais confiante para a menos) com a anotação
    livre de maior IoU. Acima do iou_threshold conta como acerto (TP);
    abaixo, como falso positivo (FP). Anotações sem par são FN.

    Retorna (tp, fp, fn, ious_dos_tp).
    """
    order = sorted(range(len(pred_boxes)), key=lambda i: pred_scores[i], reverse=True)
    used_gt = set()
    tp, fp = 0, 0
    matched_ious = []

    for i in order:
        best_iou, best_j = 0.0, -1
        for j, gt_box in enumerate(gt_boxes):
            if j in used_gt:
                continue
            iou = box_iou(pred_boxes[i], gt_box)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_threshold and best_j >= 0:
            tp += 1
            used_gt.add(best_j)
            matched_ious.append(best_iou)
        else:
            fp += 1

    fn = len(gt_boxes) - len(used_gt)
    return tp, fp, fn, matched_ious
