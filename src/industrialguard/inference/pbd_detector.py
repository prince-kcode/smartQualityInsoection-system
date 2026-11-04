"""Inference helpers for the separate supervised plastic-bottle detector."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT = PROJECT_ROOT / "results/retrained/pbd_bottle_detector/pbd_bottle_defect_detector.pt"
CLASS_NAMES = [
    "background", "body_stain", "body_concave", "body_score", "body_squeeze",
    "shoulder_pipe", "shoulder_no_burr", "shoulder_stain",
]


def load_pbd_detector():
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=None, weights_backbone=None, min_size=checkpoint["min_size"],
        max_size=checkpoint["max_size"], box_detections_per_img=100,
    )
    features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(features, len(CLASS_NAMES))
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, device


def detect_bottle_defects(model, device: torch.device, rgb: np.ndarray, threshold: float = 0.35):
    tensor = torch.from_numpy(np.ascontiguousarray(rgb).copy()).permute(2, 0, 1).float().div(255.0)
    with torch.inference_mode():
        result = model([tensor.to(device)])[0]
    keep = result["scores"].detach().cpu() >= threshold
    boxes = result["boxes"].detach().cpu()[keep].numpy()
    labels = result["labels"].detach().cpu()[keep].numpy()
    scores = result["scores"].detach().cpu()[keep].numpy()
    return boxes, labels, scores


def draw_bottle_detections(rgb: np.ndarray, boxes, labels, scores) -> np.ndarray:
    image = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(image)
    for box, label, score in zip(boxes, labels, scores, strict=True):
        x1, y1, x2, y2 = (float(value) for value in box)
        name = CLASS_NAMES[int(label)].replace("_", " ")
        draw.rectangle((x1, y1, x2, y2), outline=(255, 55, 55), width=3)
        draw.text((x1, max(0, y1 - 18)), f"{name} {score:.2f}", fill=(255, 55, 55), stroke_width=2, stroke_fill="white")
    return np.asarray(image)
