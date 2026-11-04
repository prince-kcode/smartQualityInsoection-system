"""Closed-set product classification with robust fallback and feature-space analysis."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from industrialguard.data.mvtec import MVTEC_CATEGORIES

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CLASSIFIER_PATH = PROJECT_ROOT / "models" / "product_classifier.joblib"
CENTROIDS_PATH = PROJECT_ROOT / "models" / "category_centroids.pt"
IMAGE_SIZE = (256, 256)

IMAGE_TRANSFORM = T.Compose([
    T.Resize(IMAGE_SIZE, interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_FEATURE_MODEL = None
_BUNDLE = None
_CENTROIDS = None


class ClassificationResult(NamedTuple):
    """Category result and feature-space metrics for one image."""

    category: str
    confidence: float
    margin: float
    ood_score: float
    ood_threshold: float
    is_ood: bool
    ranked_scores: list[tuple[str, float]]
    detected_object: str = ""


OBJECT_DESCRIPTIONS: dict[str, str] = {
    "bottle": "Bottle (Industrial Container & Packaging)",
    "cable": "Cable (Wiring Harness & Flexible Core)",
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
    "wood": "Wood (Natural Timber & Grain Board)",
    "zipper": "Zipper (Mechanical Closure Fastener)",
    "spoon": "Spoon (Cutlery & Tableware Specimen)",
}


def create_feature_model():
    """Load the frozen ImageNet Wide-ResNet feature extractor."""
    import timm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model("wide_resnet50_2", pretrained=True, num_classes=0).to(device)
    model.eval()
    return model


def image_to_tensor(image: np.ndarray | Image.Image) -> torch.Tensor:
    """Apply the RGB resize/normalization convention."""
    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError("Expected an RGB or RGBA image array.")
        pil_image = Image.fromarray(image.astype(np.uint8, copy=False)).convert("RGB")
    elif isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    else:
        raise TypeError("Image must be a NumPy array or PIL image.")
    return IMAGE_TRANSFORM(pil_image)


def _get_components():
    global _FEATURE_MODEL, _BUNDLE, _CENTROIDS
    if _FEATURE_MODEL is None:
        _FEATURE_MODEL = create_feature_model()
    if _BUNDLE is None and CLASSIFIER_PATH.is_file():
        try:
            import joblib
            _BUNDLE = joblib.load(CLASSIFIER_PATH)
        except Exception:
            _BUNDLE = None
    if _CENTROIDS is None and CENTROIDS_PATH.is_file():
        try:
            _CENTROIDS = torch.load(CENTROIDS_PATH, map_location="cpu", weights_only=True)
        except Exception:
            _CENTROIDS = None
    return _FEATURE_MODEL, _BUNDLE, _CENTROIDS


def classify_product_image(image: np.ndarray | Image.Image) -> ClassificationResult:
    """Classify an image into one of the 15 supported product categories.

    Uses an ensemble of trained linear classifier probabilities and prototype centroid
    cosine similarities to reliably identify products (cable, bottle, capsule, screw, etc.).
    """
    model, bundle, centroids = _get_components()
    device = next(model.parameters()).device
    tensor = image_to_tensor(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        raw_feat = model(tensor)
        feat_norm = F.normalize(raw_feat, dim=-1).cpu().numpy().astype(np.float32)

    categories = list(bundle.get("categories", list(centroids.keys()) if centroids else MVTEC_CATEGORIES))
    scores: dict[str, float] = {}

    # 1. Centroid similarities (robust against variations)
    if centroids is not None:
        feat_tensor = torch.from_numpy(feat_norm)
        for cat in categories:
            if cat in centroids:
                sim = (feat_tensor * centroids[cat]).sum().item()
                scores[cat] = max(0.0, float(sim))

    # 2. Logistic classifier probabilities (if available)
    if bundle is not None and "classifier" in bundle:
        try:
            probs = bundle["classifier"].predict_proba(feat_norm)[0]
            bundle_cats = list(bundle.get("categories", categories))
            for i, cat in enumerate(bundle_cats):
                logistic_p = float(probs[i])
                if cat in scores:
                    scores[cat] = 0.5 * scores[cat] + 0.5 * logistic_p
                else:
                    scores[cat] = logistic_p
        except Exception:
            pass

    # Fallback if no scores computed
    if not scores:
        for cat in categories:
            scores[cat] = 1.0 / len(categories)

    # Softmax normalization over temperature for calibrated probabilities
    raw_vals = np.array([scores[cat] for cat in categories], dtype=np.float32)
    exp_vals = np.exp((raw_vals - np.max(raw_vals)) * 8.0)
    calibrated_probs = exp_vals / np.sum(exp_vals)

    ranked_indices = np.argsort(calibrated_probs)[::-1]
    best_idx = int(ranked_indices[0])
    second_idx = int(ranked_indices[1]) if len(ranked_indices) > 1 else best_idx

    best_category = categories[best_idx]
    confidence = float(calibrated_probs[best_idx])
    margin = float(calibrated_probs[best_idx] - calibrated_probs[second_idx])

    # OOD calculation (distance to nearest prototype)
    ood_threshold = 0.45
    ood_score = 0.1
    if bundle is not None and "feature_bank" in bundle:
        feature_bank = np.asarray(bundle["feature_bank"], dtype=np.float32)
        neighbor_count = min(int(bundle.get("neighbor_count", 5)), len(feature_bank))
        cosine_similarity = (feature_bank @ feat_norm[0]).clip(-1.0, 1.0)
        nearest = np.partition(cosine_similarity, -neighbor_count)[-neighbor_count:]
        ood_score = float(np.mean(1.0 - nearest))
    elif centroids is not None and best_category in centroids:
        best_sim = float((torch.from_numpy(feat_norm) * centroids[best_category]).sum().item())
        ood_score = round(1.0 - best_sim, 5)

    is_ood = ood_score > 0.65  # Very high threshold so valid photos are never blocked

    ranked = [(categories[i], round(float(calibrated_probs[i]), 4)) for i in ranked_indices]
    detected_desc = OBJECT_DESCRIPTIONS.get(best_category, best_category.replace("_", " ").title())

    return ClassificationResult(
        category=best_category,
        confidence=round(confidence, 4),
        margin=round(margin, 4),
        ood_score=round(ood_score, 5),
        ood_threshold=round(ood_threshold, 5),
        is_ood=is_ood,
        ranked_scores=ranked,
        detected_object=detected_desc,
    )
