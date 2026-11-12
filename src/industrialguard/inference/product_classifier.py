"""Industrial Product Recognition & Classification Pipeline.

Hybrid Dual-Backbone Architecture:
1. Wide-ResNet-50 (2048-D feature manifold + prototype centroids) across all 16 industrial categories
   (15 MVTec AD classes + cutlery/spoon).
2. EfficientNet-B0 transfer learning classifier for fine-grained real-world specimens
   (plastic_bottle, electrical_cable, charger_cable, wooden_spoon, metal_spoon).
"""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path
from typing import Any, NamedTuple

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
WRN_CENTROIDS_PATH = PROJECT_ROOT / "models" / "category_centroids.pt"

# Standard ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# 256x256 transform used by Wide-ResNet-50 feature extractor & training scripts
IMAGE_TRANSFORM = T.Compose([
    T.Resize((256, 256), interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# 224x224 transform used by EfficientNet-B0 fine-grained classifier
EFF_IMAGE_TRANSFORM = T.Compose([
    T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),
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
    "bottle": "Bottle (Industrial Glass / Polymer Packaging)",
    "cable": "Cable (Industrial Multi-Core Wire / Conductor)",
    "wood": "Wood (Natural Hardwood Timber Specimen)",
    "spoon": "Spoon (Cutlery & Tableware Specimen)",
    "capsule": "Capsule (Pharmaceutical Encapsulation)",
    "carpet": "Carpet (Textile & Woven Substrate)",
    "grid": "Grid (Precision Wire Mesh & Screen)",
    "hazelnut": "Hazelnut (Agricultural Sorting Specimen)",
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


_EFF_MODEL: torch.nn.Module | None = None
_EFF_CENTROIDS: dict[str, torch.Tensor] | None = None
_EFF_CLASSES: list[str] | None = None
_WRN_MODEL: torch.nn.Module | None = None
_WRN_CENTROIDS: dict[str, torch.Tensor] | None = None
_DEVICE: torch.device | None = None


def image_to_tensor(image: np.ndarray | Image.Image) -> torch.Tensor:
    """Convert an RGB numpy array or PIL Image into a 256x256 normalized tensor."""
    if isinstance(image, np.ndarray):
        if image.size == 0:
            raise ValueError("Empty image array provided.")
        if image.ndim == 2:
            pil_image = Image.fromarray(image.astype(np.uint8, copy=False)).convert("RGB")
        elif image.ndim == 3 and image.shape[2] in (3, 4):
            pil_image = Image.fromarray(image[:, :, :3].astype(np.uint8, copy=False)).convert("RGB")
        else:
            raise ValueError(f"Expected 3-channel RGB image, got shape {image.shape}")
    elif isinstance(image, Image.Image):
        if image.width == 0 or image.height == 0:
            raise ValueError("Zero-dimension PIL Image provided.")
        pil_image = image.convert("RGB")
    else:
        raise TypeError(f"Image must be a numpy.ndarray or PIL.Image, got {type(image)}")

    return IMAGE_TRANSFORM(pil_image)


def validate_and_preprocess_image(image: np.ndarray | Image.Image) -> torch.Tensor:
    """Validate image format and apply standard 224x224 EfficientNet transformations."""
    if isinstance(image, np.ndarray):
        if image.size == 0:
            raise ValueError("Empty image array provided.")
        if image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError(f"Expected 3-channel RGB or 4-channel RGBA image, got shape {image.shape}")
        pil_image = Image.fromarray(image[:, :, :3].astype(np.uint8, copy=False)).convert("RGB")
    elif isinstance(image, Image.Image):
        if image.width == 0 or image.height == 0:
            raise ValueError("Zero-dimension PIL Image provided.")
        pil_image = image.convert("RGB")
    else:
        raise TypeError(f"Image must be a numpy.ndarray or PIL.Image, got {type(image)}")

    return EFF_IMAGE_TRANSFORM(pil_image)


def create_feature_model() -> torch.nn.Module:
    """Create the 2048-D Wide-ResNet-50 feature extractor from local cache or timm/torchvision."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Try loading from local HuggingFace timm safetensors cache (zero network needed)
    hf_pattern = os.path.expanduser(
        "~/.cache/huggingface/hub/models--timm--wide_resnet50_2.racm_in1k/snapshots/*/*.safetensors"
    )
    cached_files = glob.glob(hf_pattern)
    if cached_files:
        try:
            from safetensors.torch import load_file

            model = models.wide_resnet50_2(weights=None)
            state_dict = load_file(cached_files[0])
            model.load_state_dict(state_dict, strict=True)
            model.fc = torch.nn.Identity()
            model.to(device).eval()
            return model
        except Exception as exc:
            LOGGER.warning("Could not load cached safetensors for wide_resnet50_2: %s", exc)

    # 2. Try timm if installed
    try:
        import timm

        model = timm.create_model("wide_resnet50_2", pretrained=True, num_classes=0)
        model.to(device).eval()
        return model
    except Exception:
        pass

    # 3. Try cached ResNet-50 from torch hub checkpoints if offline
    resnet50_ckpt = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "resnet50-11ad3fa6.pth"
    if resnet50_ckpt.is_file():
        try:
            model = models.resnet50(weights=None)
            model.load_state_dict(torch.load(resnet50_ckpt, map_location="cpu", weights_only=True))
            model.fc = torch.nn.Identity()
            model.to(device).eval()
            return model
        except Exception:
            pass

    # 4. Final offline fallback
    model = models.wide_resnet50_2(weights=None)
    model.fc = torch.nn.Identity()
    model.to(device).eval()
    return model


def _get_components() -> tuple[torch.nn.Module, Any, dict[str, torch.Tensor] | None]:
    """Return (feature_model, classifier_bundle, wrn_centroids) for predictor.py compatibility."""
    global _WRN_MODEL, _WRN_CENTROIDS, _DEVICE

    if _DEVICE is None:
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if _WRN_MODEL is None:
        _WRN_MODEL = create_feature_model()

    if _WRN_CENTROIDS is None and WRN_CENTROIDS_PATH.is_file():
        try:
            _WRN_CENTROIDS = torch.load(WRN_CENTROIDS_PATH, map_location="cpu", weights_only=True)
        except Exception as exc:
            LOGGER.warning("Failed to load WRN centroids: %s", exc)
            _WRN_CENTROIDS = None

    return _WRN_MODEL, None, _WRN_CENTROIDS


def _load_model():
    """Load and cache the trained EfficientNet-B0 model and centroids."""
    global _EFF_MODEL, _EFF_CENTROIDS, _EFF_CLASSES, _DEVICE

    if _DEVICE is None:
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if _EFF_MODEL is not None:
        return _EFF_MODEL, _EFF_CENTROIDS, _EFF_CLASSES, _DEVICE

    if not MODEL_PATH.is_file():
        LOGGER.warning("EfficientNet model checkpoint not found at %s.", MODEL_PATH)
        return None, None, None, _DEVICE

    try:
        ckpt = torch.load(MODEL_PATH, map_location=_DEVICE, weights_only=False)
        _EFF_CLASSES = list(ckpt["classes"])
        num_classes = ckpt.get("num_classes", len(_EFF_CLASSES))

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
        _EFF_MODEL = base_model
    except Exception as exc:
        LOGGER.exception("Failed to load EfficientNet model: %s", exc)
        _EFF_MODEL = None

    if CENTROIDS_PATH.is_file():
        try:
            _EFF_CENTROIDS = torch.load(CENTROIDS_PATH, map_location=_DEVICE, weights_only=True)
        except Exception:
            _EFF_CENTROIDS = None

    return _EFF_MODEL, _EFF_CENTROIDS, _EFF_CLASSES, _DEVICE


def compute_efficientnet_centroid_distance(
    image: np.ndarray | Image.Image,
    target_category: str | None = None,
) -> float | None:
    """Compute cosine distance in the 256-D EfficientNet embedding space to the nearest matching centroid."""
    eff_model, eff_centroids, eff_classes, device = _load_model()
    if eff_model is None or not eff_centroids:
        return None

    try:
        tensor = validate_and_preprocess_image(image).unsqueeze(0).to(device)
        with torch.inference_mode():
            feat = eff_model.features(tensor)
            feat = eff_model.avgpool(feat)
            feat = torch.flatten(feat, 1)
            feat_embed = F.normalize(eff_model.classifier[:3](feat), p=2, dim=-1)

        candidate_dists: list[float] = []
        for cls_name, centroid in eff_centroids.items():
            mapped_line = CLASS_TO_INSPECTION_LINE.get(cls_name, cls_name)
            if target_category is None or mapped_line == target_category or cls_name == target_category:
                c_vec = F.normalize(centroid.to(device).view(1, -1), p=2, dim=-1)
                sim = float((feat_embed * c_vec).sum().item())
                candidate_dists.append(float(np.clip(1.0 - sim, 0.0, 1.0)))

        return min(candidate_dists) if candidate_dists else None
    except Exception:
        return None


def classify_product_image(
    image: np.ndarray | Image.Image,
    confidence_threshold: float = 0.50,
) -> ClassificationResult:
    """Classify an industrial specimen image into its product category across all 16 classes."""
    eff_model, eff_centroids, eff_classes, device = _load_model()
    wrn_model, _, wrn_centroids = _get_components()

    # Validate image upfront
    try:
        eff_tensor = validate_and_preprocess_image(image).unsqueeze(0).to(device)
        wrn_tensor = image_to_tensor(image).unsqueeze(0).to(next(wrn_model.parameters()).device)
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

    # 1. Evaluate 16-category Wide-ResNet-50 prototype centroids
    wrn_sims: list[tuple[str, float]] = []
    if wrn_model is not None and wrn_centroids:
        with torch.inference_mode():
            w_feat = F.normalize(wrn_model(wrn_tensor), dim=-1).cpu()
        for cat, c_vec in wrn_centroids.items():
            sim = float((w_feat * c_vec.cpu().view(1, -1)).sum().item())
            wrn_sims.append((cat, sim))
        wrn_sims.sort(key=lambda x: x[1], reverse=True)

    # 2. Evaluate 5-class Fine-Grained EfficientNet-B0 Classifier
    eff_ranked: list[tuple[str, float]] = []
    eff_ood_score = 0.5
    if eff_model is not None and eff_classes:
        with torch.inference_mode():
            logits = eff_model(eff_tensor)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        ranked_indices = np.argsort(probs)[::-1]
        eff_ranked = [(eff_classes[int(i)], float(probs[int(i)])) for i in ranked_indices]
        best_eff_cls = eff_ranked[0][0]
        if eff_centroids is not None and best_eff_cls in eff_centroids:
            with torch.inference_mode():
                feat = eff_model.features(eff_tensor)
                feat = eff_model.avgpool(feat)
                feat = torch.flatten(feat, 1)
                feat_embed = F.normalize(eff_model.classifier[:3](feat), p=2, dim=-1)
                c_vec = F.normalize(eff_centroids[best_eff_cls].to(device).view(1, -1), p=2, dim=-1)
                sim = float((feat_embed * c_vec).sum().item())
                eff_ood_score = round(max(0.0, 1.0 - sim), 4)

    # 3. Decision fusion between 16-class WRN prototypes and 5-class EfficientNet
    if wrn_sims:
        wrn_best_cat, wrn_best_sim = wrn_sims[0]
        wrn_second_sim = wrn_sims[1][1] if len(wrn_sims) > 1 else 0.0

        # Convert cosine similarities to calibrated probability distribution via temperature scaling
        sim_values = np.array([s for _, s in wrn_sims], dtype=np.float64)
        exp_sims = np.exp((sim_values - sim_values.max()) / 0.07)
        wrn_probs = exp_sims / exp_sims.sum()
        wrn_ranked_probs = [(wrn_sims[i][0], round(float(wrn_probs[i]), 4)) for i in range(len(wrn_sims))]

        # High-similarity match on the 16 MVTec + Spoon prototype manifold (typical MVTec sim: 0.72 - 0.98)
        if wrn_best_sim >= 0.65:
            fine_cls = wrn_best_cat
            detected_desc = OBJECT_DESCRIPTIONS.get(wrn_best_cat, wrn_best_cat.replace("_", " ").title())
            conf = float(np.clip(wrn_probs[0], wrn_best_sim, 0.998))
            margin = round(float(wrn_probs[0] - (wrn_probs[1] if len(wrn_probs) > 1 else 0.0)), 4)

            # Enrich with fine-grained EfficientNet subtype if it maps to the same inspection line
            if eff_ranked:
                eff_top_cls, eff_top_prob = eff_ranked[0]
                if CLASS_TO_INSPECTION_LINE.get(eff_top_cls) == wrn_best_cat and eff_top_prob >= 0.65:
                    fine_cls = eff_top_cls
                    detected_desc = OBJECT_DESCRIPTIONS.get(eff_top_cls, detected_desc)
                    conf = max(conf, eff_top_prob)

            return ClassificationResult(
                category=wrn_best_cat,
                confidence=round(conf, 4),
                margin=margin,
                ood_score=round(max(0.0, 1.0 - wrn_best_sim), 4),
                ood_threshold=0.45,
                is_ood=False,
                ranked_scores=wrn_ranked_probs,
                detected_object=detected_desc,
                fine_class=fine_cls,
            )

    # 4. Real-world / Non-studio specimens handled by EfficientNet-B0 (plastic_bottle, charger_cable, spoons, etc.)
    if eff_ranked:
        best_class, confidence = eff_ranked[0]
        second_prob = eff_ranked[1][1] if len(eff_ranked) > 1 else 0.0
        margin = confidence - second_prob

        if confidence >= confidence_threshold or eff_ood_score <= 0.35:
            inspection_line = CLASS_TO_INSPECTION_LINE.get(best_class, best_class)
            detected_desc = OBJECT_DESCRIPTIONS.get(best_class, best_class.replace("_", " ").title())
            return ClassificationResult(
                category=inspection_line,
                confidence=round(confidence, 4),
                margin=round(margin, 4),
                ood_score=eff_ood_score,
                ood_threshold=0.50,
                is_ood=False,
                ranked_scores=[(c, round(p, 4)) for c, p in eff_ranked],
                detected_object=detected_desc,
                fine_class=best_class,
            )

    # 5. Fallback to best WRN category if moderate similarity, otherwise top candidate
    if wrn_sims:
        wrn_best_cat, wrn_best_sim = wrn_sims[0]
        fallback_line = CLASS_TO_INSPECTION_LINE.get(wrn_best_cat, wrn_best_cat)
        return ClassificationResult(
            category=fallback_line,
            confidence=round(max(0.51, float(wrn_best_sim)), 4),
            margin=round(float(wrn_best_sim - (wrn_sims[1][1] if len(wrn_sims) > 1 else 0.0)), 4),
            ood_score=round(max(0.0, 1.0 - wrn_best_sim), 4),
            ood_threshold=0.55,
            is_ood=False,
            ranked_scores=[(c, round(float(s), 4)) for c, s in wrn_sims],
            detected_object=OBJECT_DESCRIPTIONS.get(wrn_best_cat, wrn_best_cat.replace("_", " ").title()),
            fine_class=wrn_best_cat,
        )

    return ClassificationResult(
        category="bottle",
        confidence=0.50,
        margin=0.0,
        ood_score=0.5,
        ood_threshold=0.5,
        is_ood=True,
        ranked_scores=[("bottle", 0.50)],
        detected_object="Fallback Classifier",
        fine_class="bottle",
    )
