"""Carrega o detector treinado (YOLO ou Faster R-CNN) e encontra as Daphnias numa foto."""
from pathlib import Path

import numpy as np


def carregar_modelo(model_type, checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Arquivo de pesos não encontrado: {checkpoint_path}")
    if model_type == "fasterrcnn":
        import torch
        from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:  # torch >= 2.6 exige weights_only=False para checkpoints salvos como dict
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:  # torch antigo, sem esse argumento
            checkpoint = torch.load(checkpoint_path, map_location=device)
        model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, checkpoint.get("num_classes", 2))
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        model.to(device).eval()
        return {"type": "fasterrcnn", "model": model, "device": device}
    if model_type == "yolo":
        from ultralytics import YOLO
        return {"type": "yolo", "model": YOLO(str(checkpoint_path))}
    raise ValueError(f"Tipo de modelo desconhecido: {model_type} (use 'yolo' ou 'fasterrcnn')")


def predizer(modelo_dict, imagem_bgr, score_threshold):
    """
    imagem_bgr: foto já carregada (array numpy BGR, como o cv2 lê).
    Retorna lista de (caixa [x1, y1, x2, y2], confiança), só com confiança >= score_threshold.
    """
    if modelo_dict["type"] == "fasterrcnn":
        import torch
        rgb = np.ascontiguousarray(imagem_bgr[:, :, ::-1])
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(modelo_dict["device"])
        with torch.no_grad():
            saida = modelo_dict["model"]([tensor])[0]
        caixas = saida["boxes"].cpu().tolist()
        scores = saida["scores"].cpu().tolist()
        return [(b, float(s)) for b, s in zip(caixas, scores) if s >= score_threshold]
    resultado = modelo_dict["model"].predict(source=imagem_bgr, conf=score_threshold, verbose=False)[0]
    return [(b, float(s)) for b, s in zip(resultado.boxes.xyxy.cpu().tolist(),
                                          resultado.boxes.conf.cpu().tolist())]
