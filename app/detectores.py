import torch
from PIL import Image


def carregar_modelo(model_type, checkpoint_path, device="cpu"):
    device = torch.device(device)
    if model_type == "fasterrcnn":
        from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, checkpoint.get("num_classes", 2))
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        model.to(device).eval()
        return {"type": "fasterrcnn", "model": model, "device": device}
    else:
        from ultralytics import YOLO
        model = YOLO(str(checkpoint_path))
        return {"type": "yolo", "model": model, "device": device}


def predizer(modelo_dict, image_path, score_threshold):
    if modelo_dict["type"] == "fasterrcnn":
        from torchvision.transforms.functional import pil_to_tensor
        image = Image.open(image_path).convert("RGB")
        tensor = pil_to_tensor(image).float().to(modelo_dict["device"]) / 255
        with torch.no_grad():
            output = modelo_dict["model"]([tensor])[0]
        boxes = output["boxes"].cpu().tolist()
        scores = output["scores"].cpu().tolist()
        keep = [i for i, s in enumerate(scores) if s >= score_threshold]
        return [boxes[i] for i in keep]
    else:
        result = modelo_dict["model"].predict(
            source=str(image_path), conf=score_threshold,
            device=str(modelo_dict["device"]), verbose=False)[0]
        return result.boxes.xyxy.cpu().tolist()