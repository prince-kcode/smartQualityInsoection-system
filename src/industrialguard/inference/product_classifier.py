"""Closed-set product classification with feature-space OOD rejection."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from industrialguard.data.mvtec import MVTEC_CATEGORIES

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CLASSIFIER_PATH = PROJECT_ROOT / "models" / "product_classifier.joblib"
IMAGE_SIZE = (256, 256)

# The classifier and PatchCore use the same RGB, 256x256, ImageNet-normalized
# input convention. Keep this transform in sync with the training script.
IMAGE_TRANSFORM = T.Compose([
    T.Resize(IMAGE_SIZE, interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_FEATURE_MODEL = None
_BUNDLE = None


class ClassificationResult(NamedTuple):
    """Category result and independent OOD distance for one image."""

    category: str | None
    confidence: float
    margin: float
    ood_score: float
    ood_threshold: float
    is_ood: bool
    ranked_scores: list[tuple[str, float]]


def create_feature_model():
    """Load the frozen ImageNet Wide-ResNet feature extractor."""

    import timm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model("wide_resnet50_2", pretrained=True, num_classes=0).to(device)
    model.eval()
    return model


def image_to_tensor(image: np.ndarray | Image.Image) -> torch.Tensor:
    """Apply the shared RGB resize/normalization used during classifier training."""

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
    global _FEATURE_MODEL, _BUNDLE
    if _BUNDLE is None:
        if not CLASSIFIER_PATH.is_file():
            raise FileNotFoundError(
                f"Product classifier not found at {CLASSIFIER_PATH}. "
                "Build it with scripts/train_product_classifier.py."
            )
        import joblib

        _BUNDLE = joblib.load(CLASSIFIER_PATH)
        if tuple(_BUNDLE.get("categories", ())) != tuple(MVTEC_CATEGORIES):
            raise RuntimeError("Product classifier labels do not match the MVTec category mapping.")
        expected_indices = np.arange(len(MVTEC_CATEGORIES))
        if not np.array_equal(np.asarray(_BUNDLE["classifier"].classes_), expected_indices):
            raise RuntimeError("Product classifier output indices do not map to MVTEC_CATEGORIES.")
        bank = np.asarray(_BUNDLE.get("feature_bank"))
        if bank.ndim != 2 or bank.shape[1] != 2048 or len(bank) == 0:
            raise RuntimeError("Product classifier feature bank is missing or has an invalid shape.")
    if _FEATURE_MODEL is None:
        _FEATURE_MODEL = create_feature_model()
    return _FEATURE_MODEL, _BUNDLE


def classify_product_image(image: np.ndarray | Image.Image) -> ClassificationResult:
    """Classify a supported product and reject feature-space outliers.

    A separate logistic classifier assigns one of the 15 named MVTec labels.
    A k-nearest-neighbor distance against the classifier's training feature bank
    independently detects out-of-distribution images; PatchCore scores are not
    used by either decision.
    """

    model, bundle = _get_components()
    device = next(model.parameters()).device
    tensor = image_to_tensor(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        feature = torch.nn.functional.normalize(model(tensor), dim=-1).cpu().numpy().astype(np.float32)

    classifier = bundle["classifier"]
    categories = tuple(bundle["categories"])
    probabilities = classifier.predict_proba(feature)[0]
    order = np.argsort(probabilities)[::-1]
    best_index = int(order[0])
    confidence = float(probabilities[best_index])
    margin = confidence - float(probabilities[order[1]]) if len(order) > 1 else 1.0

    feature_bank = np.asarray(bundle["feature_bank"], dtype=np.float32)
    neighbor_count = min(int(bundle.get("neighbor_count", 5)), len(feature_bank))
    cosine_similarity = (feature_bank @ feature[0]).clip(-1.0, 1.0)
    nearest = np.partition(cosine_similarity, -neighbor_count)[-neighbor_count:]
    ood_score = float(np.mean(1.0 - nearest))
    is_ood = ood_score > float(bundle["ood_threshold"])

    ranked = [(categories[int(i)], round(float(probabilities[i]), 4)) for i in order]
    confident_enough = (
        confidence >= float(bundle["confidence_threshold"])
        and margin >= float(bundle["margin_threshold"])
    )
    predicted_category = categories[best_index] if not is_ood and confident_enough else None
    return ClassificationResult(
        category=predicted_category,
        confidence=round(confidence, 4),
        margin=round(margin, 4),
        ood_score=round(ood_score, 5),
        ood_threshold=round(float(bundle["ood_threshold"]), 5),
        is_ood=is_ood,
        ranked_scores=ranked,
    )
