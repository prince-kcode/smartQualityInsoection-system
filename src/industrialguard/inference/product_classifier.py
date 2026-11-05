"""Industrial Product Recognition & Classification Pipeline.

Architecture: EfficientNet-B0 transfer learning model trained on balanced industrial dataset.
Supports: plastic_bottle, electrical_cable, charger_cable, wooden_spoon, metal_spoon.
Includes confidence thresholding, out-of-distribution (OOD) rejection, and fallback.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = PROJECT_ROOT / "models" / "product_classifier_efficientnet.pt"
CENTROIDS_PATH = PROJECT_ROOT / "models" / "product_classifier_centroids.pt"

# Standard 224x224 RGB ImageNet normalization matching training exactly
IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGE_TRANSFORM = T.Compose([
    T.Resize(IMAGE_SIZE, interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Granular human-readable product descriptions
OBJECT_DESCRIPTIONS: dict[str, str] = {
    "plastic_bottle": "Plastic Bottle (PET Recyclable Packaging)",
    "electrical_cable": "Electrical Cable (Industrial Wiring Harness & Sheathed Core)",
    "charger_cable": "Charger Cable (USB-C / Peripheral Wire Cord)",
    "wooden_spoon": "Wooden Spoon (Natural Beechwood Cutlery Specimen)",
    "metal_spoon": "Metal Spoon (Precision Stainless Steel Tableware)",
    "bottle": "Bottle (Industrial Packaging)",
    "cable": "Cable (Industrial Wire / Conductor)",
    "wood": "Wood (Natural Timber Specimen)",
    "spoon": "Spoon (Cutlery & Tableware Specimen)",
    "capsule": "Capsule (Pharmaceutical Encapsulation)",
    "carpet": "Carpet (Textile & Woven Substrate)",
    "grid": "Grid (Precision Wire Mesh & Screen)",
    "hazelnut": "Hazelnut (Agricultural Specimen)",
    "leather": "Leather (Engineered Upholstery Material)",
    "metal_nut": "Metal Nut (Precision Machined Fastener)",
    "pill": "Pill (Pharmaceutical Compressed Tablet)",
    "screw": "Screw (Threaded Fastener & Hardware)",
    "tile": "Tile (Ceramic Structural Surface)",
    "toothbrush": "Toothbrush (Consumer Hygiene Goods)",
    "transistor": "Transistor (Semiconductor Micro-Package)",
    "zipper": "Zipper (Mechanical Closure Fastener)",
    "unknown": "Unknown Product (Low-Confidence / Unsupported Specimen)",
}

# Line routing map for downstream defect inspection engines
CLASS_TO_INSPECTION_LINE: dict[str, str] = {
    "plastic_bottle": "bottle",
    "electrical_cable": "cable",
    "charger_cable": "cable",
    "wooden_spoon": "wood",
    "metal_spoon": "spoon",
    "bottle": "bottle",
    "cable": "cable",
    "wood": "wood",
    "spoon": "spoon",
}


class ClassificationResult(NamedTuple):
    """Product classification results and feature-space metrics."""

    category: str
    confidence: float
    margin: float
    ood_score: float
    ood_threshold: float
    is_ood: bool
    ranked_scores: list[tuple[str, float]]
    detected_object: str = ""
    fine_class: str = ""


_MODEL = None
_CENTROIDS = None
_CLASSES = None
_DEVICE = None


def _load_model():
    """Load and cache the trained EfficientNet-B0 model."""
    global _MODEL, _CENTROIDS, _CLASSES, _DEVICE

    if _MODEL is not None:
        return _MODEL, _CENTROIDS, _CLASSES, _DEVICE

    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not MODEL_PATH.is_file():
        LOGGER.warning("EfficientNet model checkpoint not found at %s. Using fallback.", MODEL_PATH)
        return None, None, None, _DEVICE

    try:
        ckpt = torch.load(MODEL_PATH, map_location=_DEVICE, weights_only=False)
        _CLASSES = ckpt["classes"]
        num_classes = ckpt.get("num_classes", len(_CLASSES))

        # Reconstruct architecture
        base_model = models.efficientnet_b0(weights=None)
        in_features = base_model.classifier[1].in_features
        base_model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.3, inplace=True),
            torch.nn.Linear(in_features, 256),
            torch.nn.SiLU(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(256, num_classes),
        )
        base_model.load_state_dict(ckpt["state_dict"])
        base_model.to(_DEVICE)
        base_model.eval()
        _MODEL = base_model
        LOGGER.info("Loaded EfficientNet-B0 model on %s with classes: %s", _DEVICE, _CLASSES)
    except Exception as exc:
        LOGGER.exception("Failed to load EfficientNet model: %s", exc)
        _MODEL = None

    if CENTROIDS_PATH.is_file():
        try:
            _CENTROIDS = torch.load(CENTROIDS_PATH, map_location=_DEVICE, weights_only=True)
        except Exception:
            _CENTROIDS = None

    return _MODEL, _CENTROIDS, _CLASSES, _DEVICE


def validate_and_preprocess_image(image: np.ndarray | Image.Image) -> torch.Tensor:
    """Validate image format and apply standard training transformations."""
    if isinstance(image, np.ndarray):
        if image.size == 0:
            raise ValueError("Empty image array provided.")
        if image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError(f"Expected 3-channel RGB or 4-channel RGBA image, got shape {image.shape}")
        pil_image = Image.fromarray(image.astype(np.uint8, copy=False)).convert("RGB")
    elif isinstance(image, Image.Image):
        if image.width == 0 or image.height == 0:
            raise ValueError("Zero-dimension PIL Image provided.")
        pil_image = image.convert("RGB")
    else:
        raise TypeError(f"Image must be a numpy.ndarray or PIL.Image, got {type(image)}")

    return IMAGE_TRANSFORM(pil_image)


def classify_product_image(
    image: np.ndarray | Image.Image,
    confidence_threshold: float = 0.50,
) -> ClassificationResult:
    """Classify an industrial specimen image into its fine-grained product category.

    Args:
        image: RGB or RGBA image (NumPy array or PIL.Image).
        confidence_threshold: Minimum confidence required to accept prediction (default 0.50).
                             Images below threshold or with anomalous embeddings return 'Unknown Product'.

    Returns:
        ClassificationResult with predicted category, confidence, ranked scores, and object description.
    """
    model, centroids, classes, device = _load_model()

    # Fallback if model not available
    if model is None or classes is None:
        return ClassificationResult(
            category="bottle",
            confidence=0.50,
            margin=0.0,
            ood_score=0.5,
            ood_threshold=0.5,
            is_ood=True,
            ranked_scores=[("bottle", 0.50)],
            detected_object="Fallback Classifier (Model Unloaded)",
            fine_class="bottle",
        )

    # Validate and prepare tensor
    try:
        tensor = validate_and_preprocess_image(image).unsqueeze(0).to(device)
    except Exception as exc:
        LOGGER.error("Image preprocessing failed: %s", exc)
        return ClassificationResult(
            category="unknown",
            confidence=0.0,
            margin=0.0,
            ood_score=1.0,
            ood_threshold=0.5,
            is_ood=True,
            ranked_scores=[],
            detected_object=f"Invalid Image: {exc}",
            fine_class="unknown",
        )

    with torch.inference_mode():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    # Rank classes by softmax probability
    ranked_indices = np.argsort(probs)[::-1]
    best_idx = int(ranked_indices[0])
    second_idx = int(ranked_indices[1]) if len(ranked_indices) > 1 else best_idx

    best_class = classes[best_idx]
    confidence = float(probs[best_idx])
    margin = float(probs[best_idx] - probs[second_idx])

    # Out-Of-Distribution (OOD) distance metric via prototype centroids
    ood_score = round(1.0 - confidence, 4)
    ood_threshold = 0.50
    is_ood = False

    # Check centroid cosine similarity if available
    if centroids is not None and best_class in centroids:
        with torch.inference_mode():
            feat = model.features(tensor)
            feat = model.avgpool(feat)
            feat = torch.flatten(feat, 1)
            feat_embed = F.normalize(model.classifier[:3](feat), p=2, dim=-1)
            centroid = centroids[best_class].to(device)
            sim = float((feat_embed * centroid).sum().item())
            ood_score = round(max(0.0, 1.0 - sim), 4)

    # Reject ambiguous or low-confidence inputs
    if confidence < confidence_threshold or (margin < 0.08 and confidence < 0.65):
        is_ood = True
        return ClassificationResult(
            category="unknown",
            confidence=round(confidence, 4),
            margin=round(margin, 4),
            ood_score=ood_score,
            ood_threshold=ood_threshold,
            is_ood=True,
            ranked_scores=[(classes[i], round(float(probs[i]), 4)) for i in ranked_indices],
            detected_object="Unknown Product (Low-Confidence / Manual Confirmation Required)",
            fine_class="unknown",
        )

    # Map fine class to downstream inspection line
    inspection_line = CLASS_TO_INSPECTION_LINE.get(best_class, best_class)
    detected_desc = OBJECT_DESCRIPTIONS.get(best_class, best_class.replace("_", " ").title())
    ranked = [(classes[i], round(float(probs[i]), 4)) for i in ranked_indices]

    return ClassificationResult(
        category=inspection_line,
        confidence=round(confidence, 4),
        margin=round(margin, 4),
        ood_score=ood_score,
        ood_threshold=ood_threshold,
        is_ood=is_ood,
        ranked_scores=ranked,
        detected_object=detected_desc,
        fine_class=best_class,
    )
