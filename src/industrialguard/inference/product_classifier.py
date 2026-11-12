"""Industrial Product Recognition, OOD Gate & Feature Extraction Pipeline.

Hybrid Tri-Backbone Architecture:
1. CLIP ViT-B/32 (Semantic OOD Gate + 16-Category Prototype Bank):
   Rejects any uploaded image that does not belong to the trained industrial dataset
   ("NOT TRAINED FOR THIS DATASET").
2. Wide-ResNet-50 (2048-D feature manifold + prototype centroids) across all 16 industrial categories.
3. EfficientNet-B0 transfer learning classifier for fine-grained real-world specimens
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
CLIP_BANK_PATH = PROJECT_ROOT / "models" / "clip_category_bank.pt"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGE_TRANSFORM = T.Compose([
    T.Resize((256, 256), interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

EFF_IMAGE_TRANSFORM = T.Compose([
    T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

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
    "unknown": "Not Trained For This Dataset (Unsupported Object)",
}

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
_CLIP_MODEL: Any = None
_CLIP_PROC: Any = None
_CLIP_BANK: dict[str, Any] | None = None
_DEVICE: torch.device | None = None


def image_to_tensor(image: np.ndarray | Image.Image | Path | str) -> torch.Tensor:
    """Convert an RGB numpy array, PIL Image, or file path into a 256x256 normalized tensor."""
    if isinstance(image, (Path, str)):
        image = Image.open(image).convert("RGB")
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
        raise TypeError(f"Image must be a numpy.ndarray, PIL.Image, or Path, got {type(image)}")

    return IMAGE_TRANSFORM(pil_image)


def validate_and_preprocess_image(image: np.ndarray | Image.Image | Path | str) -> torch.Tensor:
    """Validate image format and apply standard 224x224 EfficientNet transformations."""
    if isinstance(image, (Path, str)):
        image = Image.open(image).convert("RGB")
    if isinstance(image, np.ndarray):
        if image.size == 0:
            raise ValueError("Empty image array provided.")
        if image.ndim == 2:
            pil_image = Image.fromarray(image.astype(np.uint8, copy=False)).convert("RGB")
        elif image.ndim == 3 and image.shape[2] in (3, 4):
            pil_image = Image.fromarray(image[:, :, :3].astype(np.uint8, copy=False)).convert("RGB")
        else:
            raise ValueError(f"Expected 3-channel RGB or 4-channel RGBA image, got shape {image.shape}")
    elif isinstance(image, Image.Image):
        if image.width == 0 or image.height == 0:
            raise ValueError("Zero-dimension PIL Image provided.")
        pil_image = image.convert("RGB")
    else:
        raise TypeError(f"Image must be a numpy.ndarray, PIL.Image, or Path, got {type(image)}")

    return EFF_IMAGE_TRANSFORM(pil_image)


def create_feature_model() -> torch.nn.Module:
    """Create the 2048-D Wide-ResNet-50 feature extractor from local cache or HuggingFace Hub."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hf_pattern = os.path.expanduser(
        "~/.cache/huggingface/hub/models--timm--wide_resnet50_2.racm_in1k/snapshots/*/*.safetensors"
    )
    cached_files = glob.glob(hf_pattern)
    if not cached_files:
        try:
            from huggingface_hub import hf_hub_download

            downloaded = hf_hub_download(
                repo_id="timm/wide_resnet50_2.racm_in1k",
                filename="model.safetensors",
            )
            if downloaded and os.path.isfile(downloaded):
                cached_files = [downloaded]
        except Exception as exc:
            LOGGER.warning("Could not download wide_resnet50_2.racm_in1k from HuggingFace Hub: %s", exc)

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
            LOGGER.warning("Could not load safetensors for wide_resnet50_2: %s", exc)

    try:
        import timm

        model = timm.create_model("wide_resnet50_2.racm_in1k", pretrained=True, num_classes=0)
        model.to(device).eval()
        return model
    except Exception:
        pass

    try:
        model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.DEFAULT)
        model.fc = torch.nn.Identity()
        model.to(device).eval()
        return model
    except Exception:
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


def _get_clip_components():
    """Load and cache CLIP ViT-B/32 model, processor, and precomputed category/OOD bank."""
    global _CLIP_MODEL, _CLIP_PROC, _CLIP_BANK, _DEVICE

    if _DEVICE is None:
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if _CLIP_BANK is None and CLIP_BANK_PATH.is_file():
        try:
            _CLIP_BANK = torch.load(CLIP_BANK_PATH, map_location="cpu", weights_only=False)
        except Exception as exc:
            LOGGER.warning("Could not load clip_category_bank.pt: %s", exc)
            _CLIP_BANK = None

    if _CLIP_MODEL is None:
        try:
            from transformers import CLIPModel, CLIPProcessor

            try:
                _CLIP_MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True).to(_DEVICE).eval()
                _CLIP_PROC = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
            except Exception:
                _CLIP_MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(_DEVICE).eval()
                _CLIP_PROC = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        except Exception as exc:
            LOGGER.warning("Could not load CLIP model: %s", exc)
            _CLIP_MODEL = None
            _CLIP_PROC = None

    return _CLIP_MODEL, _CLIP_PROC, _CLIP_BANK, _DEVICE


def extract_clip_image_embedding(image: np.ndarray | Image.Image | Path | str) -> torch.Tensor | None:
    """Return normalized [1, 512] CLIP ViT-B/32 image embedding on CPU."""
    clip_model, clip_proc, _, device = _get_clip_components()
    if clip_model is None or clip_proc is None:
        return None
    if isinstance(image, (Path, str)):
        pil_img = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            pil_img = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        else:
            pil_img = Image.fromarray(image[:, :, :3].astype(np.uint8))
    else:
        pil_img = image.convert("RGB")
    inputs = clip_proc(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = clip_model.get_image_features(**inputs)
        if not isinstance(out, torch.Tensor):
            out = getattr(out, "pooler_output", out[0])
        return F.normalize(out.float(), dim=-1).cpu()


def _load_model():
    """Load and cache the trained EfficientNet-B0 model and centroids."""
    global _EFF_MODEL, _EFF_CENTROIDS, _EFF_CLASSES, _DEVICE

    if _DEVICE is None:
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if _EFF_MODEL is not None:
        return _EFF_MODEL, _EFF_CENTROIDS, _EFF_CLASSES, _DEVICE

    if not MODEL_PATH.is_file():
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


def classify_product_image(
    image: np.ndarray | Image.Image,
    confidence_threshold: float = 0.50,
) -> ClassificationResult:
    """Classify an industrial specimen image or reject it if not in the trained dataset."""
    eff_model, eff_centroids, eff_classes, device = _load_model()
    wrn_model, _, wrn_centroids = _get_components()
    _, _, clip_bank, _ = _get_clip_components()

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
            ood_threshold=0.45,
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
    wrn_best_cat = wrn_sims[0][0] if wrn_sims else "bottle"
    wrn_best_sim = wrn_sims[0][1] if wrn_sims else 0.0

    # 2. Evaluate CLIP ViT-B/32 Image Bank & Semantic OOD Gate
    clip_emb = extract_clip_image_embedding(image)
    clip_bank_sims: list[tuple[str, float]] = []
    best_cat_text_sim = 0.0
    best_ood_text_sim = 0.0
    best_ood_label = "Unsupported Object"

    if clip_emb is not None and clip_bank is not None:
        img_bank: dict[str, torch.Tensor] = clip_bank.get("image_bank", {})
        for cat, bank_mat in img_bank.items():
            sims = (clip_emb @ bank_mat.T)[0]
            # Top-3 nearest neighbors mean similarity
            k = min(3, sims.numel())
            top_k_sim = float(torch.topk(sims, k=k).values.mean().item())
            clip_bank_sims.append((cat, top_k_sim))
        clip_bank_sims.sort(key=lambda x: x[1], reverse=True)

        cat_text_embeds: dict[str, torch.Tensor] = clip_bank.get("cat_text_embeds", {})
        cat_text_scores: list[tuple[str, float]] = []
        for cat, t_emb in cat_text_embeds.items():
            s = float((clip_emb @ t_emb.T).item())
            cat_text_scores.append((cat, s))
        cat_text_scores.sort(key=lambda x: x[1], reverse=True)
        if cat_text_scores:
            best_cat_text_sim = cat_text_scores[0][1]

        ood_labels: list[str] = clip_bank.get("ood_labels", [])
        ood_text_embeds: torch.Tensor | None = clip_bank.get("ood_text_embeds")
        if ood_text_embeds is not None and len(ood_labels) > 0:
            ood_sims = (clip_emb @ ood_text_embeds.T)[0]
            max_idx = int(torch.argmax(ood_sims).item())
            best_ood_text_sim = float(ood_sims[max_idx].item())
            best_ood_label = ood_labels[max_idx]

    max_clip_bank_sim = clip_bank_sims[0][1] if clip_bank_sims else 0.0
    best_clip_cat = clip_bank_sims[0][0] if clip_bank_sims else wrn_best_cat

    # 3. Evaluate 5-class Fine-Grained EfficientNet-B0 Classifier
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

    # 4. STRICT OUT-OF-DISTRIBUTION (OOD) REJECTION GATE
    # Reject any uploaded image that does not belong to our 16 trained dataset categories!
    if clip_emb is not None and clip_bank is not None:
        is_semantic_ood = (best_ood_text_sim > best_cat_text_sim + 0.012) and (max_clip_bank_sim < 0.80)
        is_distance_ood = (max_clip_bank_sim < 0.69) and (wrn_best_sim < 0.62)
        if is_semantic_ood or is_distance_ood:
            ood_dist = round(float(1.0 - max_clip_bank_sim), 4)
            return ClassificationResult(
                category="unknown",
                confidence=round(float(max_clip_bank_sim), 4),
                margin=0.0,
                ood_score=ood_dist,
                ood_threshold=0.31,
                is_ood=True,
                ranked_scores=[(c, round(float(s), 4)) for c, s in clip_bank_sims[:5]],
                detected_object=f"Not Trained For This Dataset (Detected Non-Dataset Image: {best_ood_label})",
                fine_class="unknown",
            )

    # 5. In-Distribution Product Classification (16 MVTec + Cutlery classes)
    if wrn_sims and wrn_best_sim >= 0.65:
        sim_values = np.array([s for _, s in wrn_sims], dtype=np.float64)
        exp_sims = np.exp((sim_values - sim_values.max()) / 0.07)
        wrn_probs = exp_sims / exp_sims.sum()
        wrn_ranked_probs = [(wrn_sims[i][0], round(float(wrn_probs[i]), 4)) for i in range(len(wrn_sims))]

        fine_cls = wrn_best_cat
        detected_desc = OBJECT_DESCRIPTIONS.get(wrn_best_cat, wrn_best_cat.replace("_", " ").title())
        conf = float(np.clip(wrn_probs[0], wrn_best_sim, 0.998))
        margin = round(float(wrn_probs[0] - (wrn_probs[1] if len(wrn_probs) > 1 else 0.0)), 4)

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
            ood_threshold=0.38,
            is_ood=False,
            ranked_scores=wrn_ranked_probs,
            detected_object=detected_desc,
            fine_class=fine_cls,
        )

    # 6. Real-world dataset specimens via EfficientNet-B0 + CLIP prototype verification
    if eff_ranked:
        best_class, confidence = eff_ranked[0]
        second_prob = eff_ranked[1][1] if len(eff_ranked) > 1 else 0.0
        margin = confidence - second_prob

        if confidence >= confidence_threshold and (max_clip_bank_sim >= 0.69 or eff_ood_score <= 0.25):
            inspection_line = CLASS_TO_INSPECTION_LINE.get(best_class, best_class)
            detected_desc = OBJECT_DESCRIPTIONS.get(best_class, best_class.replace("_", " ").title())
            return ClassificationResult(
                category=inspection_line,
                confidence=round(confidence, 4),
                margin=round(margin, 4),
                ood_score=eff_ood_score,
                ood_threshold=0.35,
                is_ood=False,
                ranked_scores=[(c, round(p, 4)) for c, p in eff_ranked],
                detected_object=detected_desc,
                fine_class=best_class,
            )

    # 7. Check CLIP bank match if strong
    if clip_bank_sims and max_clip_bank_sim >= 0.74:
        inspection_line = CLASS_TO_INSPECTION_LINE.get(best_clip_cat, best_clip_cat)
        return ClassificationResult(
            category=inspection_line,
            confidence=round(float(max_clip_bank_sim), 4),
            margin=round(float(max_clip_bank_sim - (clip_bank_sims[1][1] if len(clip_bank_sims) > 1 else 0.0)), 4),
            ood_score=round(float(1.0 - max_clip_bank_sim), 4),
            ood_threshold=0.31,
            is_ood=False,
            ranked_scores=[(c, round(float(s), 4)) for c, s in clip_bank_sims[:5]],
            detected_object=OBJECT_DESCRIPTIONS.get(best_clip_cat, best_clip_cat.replace("_", " ").title()),
            fine_class=best_clip_cat,
        )

    return ClassificationResult(
        category="unknown",
        confidence=round(float(max(max_clip_bank_sim, wrn_best_sim)), 4),
        margin=0.0,
        ood_score=round(float(1.0 - max(max_clip_bank_sim, wrn_best_sim)), 4),
        ood_threshold=0.31,
        is_ood=True,
        ranked_scores=[(c, round(float(s), 4)) for c, s in (clip_bank_sims[:5] if clip_bank_sims else wrn_sims[:5])],
        detected_object=f"Not Trained For This Dataset (Detected: {best_ood_label})",
        fine_class="unknown",
    )
