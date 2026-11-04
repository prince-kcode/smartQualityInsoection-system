from __future__ import annotations

import hashlib
import io
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import open_clip
import streamlit as st
import torch
from PIL import Image, ImageDraw
from scipy import ndimage
from torch import nn
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
CLIP_MIN_LABEL_MATCH = 0.32
CLIP_MIN_LOGIT_MARGIN = 0.75
NON_PRODUCT_LABELS = {"person", "animal", "random/noise image", "unclear image"}
logger = logging.getLogger("industrialguard.bottle_yolo")

INDUSTRY_CATEGORY_GROUPS = {
    "Electronics & e-waste": {
        "battery", "cable", "keyboard", "laptop", "mobile_phone", "mouse",
        "pcb1", "pcb2", "pcb3", "pcb4", "printed_circuit_board", "transistor", "wallplugs",
    },
    "Packaging & containers": {"bottle", "can", "plastic_shampoo_bottle", "vial"},
    "Food & agriculture": {
        "cashew", "chewinggum", "fruit_jelly", "fryum", "hazelnut", "macaroni1",
        "macaroni2", "pipe_fryum", "rice", "walnuts",
    },
    "Pharma & healthcare": {"capsule", "pill"},
    "Manufacturing & materials": {
        "candle", "carpet", "fabric", "grid", "leather", "metal_nut", "screw",
        "sheet_metal", "tile", "toothbrush", "wood", "zipper",
    },
}


class MLP(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(512, 384), nn.BatchNorm1d(384), nn.ReLU(), nn.Dropout(.30),
                                 nn.Linear(384, 192), nn.BatchNorm1d(192), nn.ReLU(), nn.Dropout(.20),
                                 nn.Linear(192, classes))

    def forward(self, x):
        return self.net(x)


class BottleGradeMLP(nn.Module):
    """Multi-class bottle defect/quality-grade head for the damage model."""

    def __init__(self, input_dim, classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(.30),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(.20),
            nn.Linear(128, classes),
        )

    def forward(self, x):
        return self.net(x)


class DamageTypeMLP(nn.Module):
    """Broad damage-type clue head over regions localized by the second model."""

    def __init__(self, input_dim, classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(.30),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(.20),
            nn.Linear(128, classes),
        )

    def forward(self, x):
        return self.net(x)


def _decoder_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
    )


class DefectSegmenter(nn.Module):
    """ResNet18 encoder with a U-Net decoder trained on binary defect masks."""

    def __init__(self):
        super().__init__()
        self.encoder = models.resnet18(weights=None)
        self.up3 = nn.Conv2d(512, 256, 1)
        self.dec3 = _decoder_block(512, 256)
        self.up2 = nn.Conv2d(256, 128, 1)
        self.dec2 = _decoder_block(256, 128)
        self.up1 = nn.Conv2d(128, 64, 1)
        self.dec1 = _decoder_block(128, 64)
        self.up0 = nn.Conv2d(64, 64, 1)
        self.dec0 = _decoder_block(128, 64)
        self.final = nn.Sequential(_decoder_block(64, 32), nn.Conv2d(32, 1, 1))

    def forward(self, x):
        target_size = x.shape[-2:]
        e0 = self.encoder.relu(self.encoder.bn1(self.encoder.conv1(x)))
        e1 = self.encoder.layer1(self.encoder.maxpool(e0))
        e2 = self.encoder.layer2(e1)
        e3 = self.encoder.layer3(e2)
        e4 = self.encoder.layer4(e3)
        d = torch.nn.functional.interpolate(self.up3(e4), size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d = self.dec3(torch.cat((d, e3), dim=1))
        d = torch.nn.functional.interpolate(self.up2(d), size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d = self.dec2(torch.cat((d, e2), dim=1))
        d = torch.nn.functional.interpolate(self.up1(d), size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d = self.dec1(torch.cat((d, e1), dim=1))
        d = torch.nn.functional.interpolate(self.up0(d), size=e0.shape[-2:], mode="bilinear", align_corners=False)
        d = self.dec0(torch.cat((d, e0), dim=1))
        d = torch.nn.functional.interpolate(d, size=target_size, mode="bilinear", align_corners=False)
        return self.final(d)


PRODUCT_ALIASES = {
    "bottle": ["a close-up industrial inspection photo of a bottle", "a photo of a plastic bottle", "a photo of a glass bottle"],
    "plastic_shampoo_bottle": ["a plastic shampoo bottle", "an industrial inspection photo of a shampoo bottle",
                               "a plastic shampoo bottle with a surface defect"],
    "cable": ["a close-up industrial inspection photo of a cable", "a photo of a USB charging cable with a connector",
              "a photo of a bundle of electrical wires or cables", "a USB cable or charging cord",
              "a person holding an electronic cable", "a phone charging cable held in a hand",
              "a USB-C charging cord with a connector", "a charging cable and charging connector"],
    "battery": ["a close-up industrial inspection photo of a battery", "a photo of a household battery",
                "a photo of a discarded electronic battery"],
    "keyboard": ["a close-up industrial inspection photo of a keyboard", "a computer keyboard",
                 "a discarded computer keyboard"],
    "laptop": ["a close-up industrial inspection photo of a laptop", "a laptop computer",
               "a discarded laptop computer"],
    "mobile_phone": ["a close-up industrial inspection photo of a mobile phone", "a smartphone",
                     "a discarded mobile phone"],
    "mouse": ["a close-up industrial inspection photo of a computer mouse", "a computer mouse",
              "a discarded computer mouse"],
    "printed_circuit_board": ["a close-up industrial inspection photo of a printed circuit board",
                              "an electronic PCB", "a discarded printed circuit board"],
    "capsule": ["a close-up industrial inspection photo of a capsule", "a capsule pill", "a medicine capsule"],
    "capsules": ["a close-up industrial inspection photo of a capsule", "a capsule pill", "a medicine capsule"],
    "pill": ["a close-up industrial inspection photo of a pill", "a medicine pill"],
    "can": ["a close-up industrial inspection photo of a can", "a metal beverage can"],
    "transistor": ["a close-up industrial inspection photo of a transistor", "an electronic transistor"],
    "pcb1": ["a close-up industrial inspection photo of a printed circuit board", "a printed circuit board"],
    "pcb2": ["a close-up industrial inspection photo of a printed circuit board", "a printed circuit board"],
    "pcb3": ["a close-up industrial inspection photo of a printed circuit board", "a printed circuit board"],
    "pcb4": ["a close-up industrial inspection photo of a printed circuit board", "a printed circuit board"],
    "metal_nut": ["a close-up industrial inspection photo of a metal nut", "a metal hex nut"],
    "wallplugs": ["a close-up industrial inspection photo of wall plugs", "plastic wall anchors"],
    "walnuts": ["a close-up industrial inspection photo of walnuts", "a walnut"],
    "hazelnut": ["a close-up industrial inspection photo of a hazelnut", "a hazelnut"],
    "chewinggum": ["a close-up industrial inspection photo of chewing gum", "a piece of chewing gum"],
    "macaroni1": ["a close-up industrial inspection photo of macaroni", "macaroni pasta"],
    "macaroni2": ["a close-up industrial inspection photo of macaroni", "macaroni pasta"],
    "fruit_jelly": ["a close-up industrial inspection photo of fruit jelly", "a fruit jelly candy"],
    "sheet_metal": ["a close-up industrial inspection photo of sheet metal", "a sheet of metal"],
    "pipe_fryum": ["a close-up industrial inspection photo of pipe fryum", "a pipe-shaped fryum snack"],
}
GENERAL_PROMPTS = {
    "person": ["a photograph of a person", "a portrait of a woman", "a person standing"],
    "animal": ["a photo of a dog", "a photo of a cat", "an animal"],
    "smartphone": ["a smartphone", "a mobile phone"],
    "shoe": ["a shoe", "a sneaker"],
    "clothing": ["a piece of clothing", "a shirt"],
    "charger": ["a phone charger adapter", "a power adapter"],
    "food": ["a plate of food", "a piece of fruit", "a snack"],
    "furniture": ["a chair", "a sofa", "a table"],
    "vehicle": ["a car", "a bicycle", "a motorcycle"],
    "household object": ["a household object", "an everyday object"],
    "random/noise image": ["a random colorful noise pattern", "an abstract pattern of static"],
    "unclear image": ["a blurry photograph of an everyday object", "an unclear image"],
}
OPEN_VOCABULARY_PRODUCT_LABELS = {
    "smartphone", "shoe", "clothing", "charger", "food", "furniture", "vehicle",
    "household object",
}
DEFECT_TYPE_PROMPTS = {
    "scratch": "a close-up product surface with a visible scratch",
    "dent or deformation": "a manufactured product with a dent or deformed area",
    "crack": "a product with a visible crack",
    "hole or puncture": "a product surface with a visible hole or puncture",
    "broken or missing piece": "a broken product with a missing piece",
    "cut or tear": "a product with a cut or tear",
    "wear or abrasion": "a worn product surface with abrasion",
    "stain or contamination": "a product surface with a stain or contamination",
    "other visible damage": "a manufactured product with other visible damage",
}


def _prompt_groups(categories, aliases, general):
    prompts, groups = [], {}
    for category in categories:
        label = category.replace("_", " ").replace("1", "").replace("2", "")
        options = aliases.get(category, [f"a close-up industrial inspection photo of {label}", f"a photo of {label}"])
        groups[category] = list(range(len(prompts), len(prompts) + len(options)))
        prompts.extend(options)
    for category, options in general.items():
        groups[f"general::{category}"] = list(range(len(prompts), len(prompts) + len(options)))
        prompts.extend(options)
    return prompts, groups


def _single_prompt_groups(prompt_map):
    prompts = list(prompt_map.values())
    groups = {name: [index] for index, name in enumerate(prompt_map)}
    return prompts, groups


@st.cache_resource
def load_assets(asset_checkpoint_stamp: tuple = ()):
    model_dir = ROOT / "models"
    required = [model_dir / "product_mlp.pt", model_dir / "neu_defect_mlp.pt",
                model_dir / "resnet18_encoder.pt", model_dir / "normal_feature_banks.npz",
                model_dir / "ood_calibration.json"]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        return None, f"Trained model files are missing: {', '.join(missing)}. Run the training scripts first."
    try:
        encoder = nn.Sequential(*list(models.resnet18(weights=None).children())[:-1])
        enc_bundle = torch.load(required[2], map_location="cpu", weights_only=True)
        encoder.load_state_dict(enc_bundle["state_dict"])
        encoder.to(DEVICE).eval()
        p_bundle = torch.load(required[0], map_location="cpu", weights_only=True)
        product = MLP(len(p_bundle["categories"]))
        product.load_state_dict(p_bundle["state_dict"])
        product.to(DEVICE).eval()
        n_bundle = torch.load(required[1], map_location="cpu", weights_only=True)
        neu = MLP(len(n_bundle["categories"]))
        neu.load_state_dict(n_bundle["state_dict"])
        neu.to(DEVICE).eval()
        bank_file = np.load(required[3])
        banks = {name: bank_file[name] for name in bank_file.files}
        category_path = model_dir / "category_feature_banks.npz"
        category_file = np.load(category_path if category_path.exists() else required[3])
        category_banks = {name: category_file[name] for name in category_file.files}
        calibration = __import__("json").loads(required[4].read_text(encoding="utf-8"))

        defect_segmenter = None
        defect_calibration = None
        defect_path = model_dir / "defect_segmenter.pt"
        if defect_path.exists():
            defect_bundle = torch.load(defect_path, map_location="cpu", weights_only=True)
            defect_segmenter = DefectSegmenter()
            defect_segmenter.load_state_dict(defect_bundle["state_dict"])
            defect_segmenter.to(DEVICE).eval()
            defect_calibration = defect_bundle

        bottle_grade_head = None
        bottle_grade_classes = []
        bottle_grade_thresholds = []
        bottle_grade_path = model_dir / "bottle_damage_mlp.pt"
        if bottle_grade_path.exists():
            bottle_bundle = torch.load(bottle_grade_path, map_location="cpu", weights_only=True)
            bottle_grade_classes = bottle_bundle["classes"]
            bottle_grade_thresholds = bottle_bundle["thresholds"]
            bottle_grade_head = BottleGradeMLP(int(bottle_bundle["input_dim"]), len(bottle_grade_classes))
            bottle_grade_head.load_state_dict(bottle_bundle["state_dict"])
            bottle_grade_head.to(DEVICE).eval()

        # The bottle YOLO head is optional and only applies to the annotated
        # plastic shampoo-bottle dataset. Other categories keep using the
        # general mask-trained damage screen below.
        bottle_yolo = None
        bottle_yolo_path = model_dir / "bottle_defect_yolo.pt"
        if bottle_yolo_path.exists():
            try:
                from ultralytics import YOLO
                bottle_yolo = YOLO(str(bottle_yolo_path))
                bottle_yolo.to(DEVICE)
            except Exception:
                logger.exception("Could not load optional bottle YOLO checkpoint")
                bottle_yolo = None

        damage_type_head = None
        damage_type_classes = []
        damage_type_thresholds = []
        damage_type_path = model_dir / "damage_type_mlp.pt"
        if damage_type_path.exists():
            damage_type_bundle = torch.load(damage_type_path, map_location="cpu", weights_only=True)
            damage_type_classes = damage_type_bundle["classes"]
            damage_type_thresholds = damage_type_bundle["thresholds"]
            damage_type_head = DamageTypeMLP(int(damage_type_bundle["input_dim"]), len(damage_type_classes))
            damage_type_head.load_state_dict(damage_type_bundle["state_dict"])
            damage_type_head.to(DEVICE).eval()

        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32-quickgelu", pretrained="openai", device=DEVICE)
        clip_model.eval()
        tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")
        product_prompts, product_groups = _prompt_groups(p_bundle["categories"], PRODUCT_ALIASES, GENERAL_PROMPTS)
        defect_prompts, defect_groups = _single_prompt_groups(DEFECT_TYPE_PROMPTS)
        with torch.inference_mode():
            product_text = clip_model.encode_text(tokenizer(product_prompts).to(DEVICE)).float()
            product_text /= product_text.norm(dim=-1, keepdim=True)
            defect_text = clip_model.encode_text(tokenizer(defect_prompts).to(DEVICE)).float()
            defect_text /= defect_text.norm(dim=-1, keepdim=True)
        return {"encoder": encoder, "product": product, "product_categories": p_bundle["categories"],
                "neu": neu, "neu_categories": n_bundle["categories"], "banks": banks,
                "category_banks": category_banks,
                "calibration": calibration, "clip": clip_model, "clip_preprocess": clip_preprocess,
                "clip_product_text": product_text, "clip_product_groups": product_groups,
                "clip_defect_text": defect_text, "clip_defect_groups": defect_groups,
                "defect_segmenter": defect_segmenter, "defect_calibration": defect_calibration,
                "bottle_grade_head": bottle_grade_head,
                "bottle_grade_classes": bottle_grade_classes,
                "bottle_grade_thresholds": bottle_grade_thresholds,
                "bottle_yolo": bottle_yolo,
                "damage_type_head": damage_type_head,
                "damage_type_classes": damage_type_classes,
                "damage_type_thresholds": damage_type_thresholds}, None
    except Exception as exc:
        return None, f"Could not load the trained models: {exc}"


def image_transform():
    return transforms.Compose([transforms.Resize(256), transforms.CenterCrop(IMAGE_SIZE),
                               transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def _clip_group_scores(image_features, text_features, groups):
    logits = 100.0 * image_features @ text_features.T
    names = list(groups)
    # logits has shape [batch, prompts]; index prompt positions on axis 1.
    values = torch.stack([logits[0, groups[name]].max() for name in names])
    probabilities = torch.softmax(values, dim=0)
    return {name: float(probabilities[i]) for i, name in enumerate(names)}, {
        name: float(values[i]) for i, name in enumerate(names)}


def _distance(feature, category, assets):
    bank = assets["banks"].get(category)
    if bank is None:
        return float("inf")
    return float(np.sqrt(((bank - feature[None, :]) ** 2).sum(1)).min())


def _category_distance(feature, category, assets):
    bank = assets["category_banks"].get(category)
    if bank is None:
        return float("inf")
    return float(np.sqrt(((bank - feature[None, :]) ** 2).sum(1)).min())


def _screen_for_defects(image, assets):
    model = assets.get("defect_segmenter")
    calibration = assets.get("defect_calibration")
    if model is None or calibration is None:
        return None
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMAGE_SIZE), transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        probability = torch.sigmoid(model(tensor))[0, 0].float().cpu().numpy()
    image_threshold = float(calibration.get("image_score_threshold", 1.0))
    pixel_threshold = float(calibration.get("pixel_threshold", 1.0))
    image_score = float(probability.max())
    raw_mask = probability >= pixel_threshold
    labeled, count = ndimage.label(raw_mask)
    regions = []
    if count:
        sizes = np.bincount(labeled.ravel())
        keep = sizes >= 4
        keep[0] = False
        mask = keep[labeled]
        for component_id in range(1, count + 1):
            if sizes[component_id] < 4:
                continue
            yy, xx = np.where(labeled == component_id)
            regions.append((int(xx.min()), int(yy.min()), int(xx.max()) + 1,
                            int(yy.max()) + 1, int(sizes[component_id])))
        regions.sort(key=lambda region: region[4], reverse=True)
    else:
        mask = np.zeros_like(raw_mask, dtype=bool)
    area_fraction = float(mask.mean())
    flagged = image_score >= image_threshold

    view = transforms.CenterCrop(IMAGE_SIZE)(transforms.Resize(
        256, interpolation=InterpolationMode.BICUBIC)(image.convert("RGB")))
    overlay = np.asarray(view, dtype=np.float32).copy()
    if flagged and mask.any():
        red = np.zeros_like(overlay)
        red[..., 0] = 255
        overlay[mask] = 0.55 * overlay[mask] + 0.45 * red[mask]
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    candidate_type, candidate_score = None, None
    if flagged and mask.any():
        yy, xx = np.where(mask)
        pad_x = max(8, int((xx.max() - xx.min() + 1) * 0.15))
        pad_y = max(8, int((yy.max() - yy.min() + 1) * 0.15))
        crop = view.crop((max(0, int(xx.min()) - pad_x), max(0, int(yy.min()) - pad_y),
                          min(IMAGE_SIZE, int(xx.max()) + pad_x + 1),
                          min(IMAGE_SIZE, int(yy.max()) + pad_y + 1)))
        clip_image = assets["clip_preprocess"](crop).unsqueeze(0).to(DEVICE)
        with torch.inference_mode():
            features = assets["clip"].encode_image(clip_image).float()
            features /= features.norm(dim=-1, keepdim=True)
            labels, _ = _clip_group_scores(features, assets["clip_defect_text"], assets["clip_defect_groups"])
        candidate_type = max(labels, key=labels.get)
        candidate_score = float(labels[candidate_type])

    if area_fraction < 0.002:
        extent = "small"
    elif area_fraction < 0.02:
        extent = "moderate"
    else:
        extent = "large"
    return {"image_score": image_score, "image_threshold": image_threshold,
            "pixel_threshold": pixel_threshold, "flagged": bool(flagged),
            "mask": mask, "area_fraction": area_fraction, "visible_extent": extent,
            "overlay": overlay, "candidate_type": candidate_type,
            "candidate_score": candidate_score, "regions": regions[:8], "view": view}


def _classify_damage_regions(screen, assets):
    head = assets.get("damage_type_head")
    if head is None or screen is None or not screen["flagged"] or not screen["mask"].any():
        return []
    labels = assets["damage_type_classes"]
    thresholds = assets["damage_type_thresholds"]
    guesses = []
    for region_index, (x0, y0, x1, y1, area) in enumerate(screen["regions"][:5], 1):
        pad_x, pad_y = max(8, int((x1 - x0) * 0.25)), max(8, int((y1 - y0) * 0.25))
        crop = screen["view"].crop((max(0, x0 - pad_x), max(0, y0 - pad_y),
                                     min(IMAGE_SIZE, x1 + pad_x), min(IMAGE_SIZE, y1 + pad_y)))
        tensor = image_transform()(crop).unsqueeze(0).to(DEVICE)
        with torch.inference_mode():
            features = assets["encoder"](tensor).flatten(1).float()
            probabilities = torch.softmax(head(features), dim=1)[0].cpu().numpy()
        class_index = int(probabilities.argmax())
        score, limit = float(probabilities[class_index]), float(thresholds[class_index])
        if score >= limit:
            guesses.append({"type": labels[class_index].replace("_", " "),
                            "score": score, "threshold": limit, "region": region_index,
                            "region_area_pixels": area})
    return guesses


def _detect_bottle_yolo(image, assets, confidence_threshold=0.25):
    """Return localized bottle defect and dataset-grade candidates from YOLO."""
    model = assets.get("bottle_yolo")
    if model is None:
        return None
    view = image.convert("RGB")
    try:
        results = model.predict(
            source=np.asarray(view), imgsz=640, conf=0.01, iou=0.6,
            device=0 if DEVICE.type == "cuda" else "cpu", verbose=False,
        )
        result = results[0]
        names = result.names
        predictions = []
        overlay = view.copy()
        draw = ImageDraw.Draw(overlay)
        boxes = result.boxes
        if boxes is not None and len(boxes):
            for coords, score, class_id in zip(
                    boxes.xyxy.detach().cpu().tolist(),
                    boxes.conf.detach().cpu().tolist(),
                    boxes.cls.detach().cpu().to(torch.int64).tolist()):
                label = str(names[int(class_id)])
                threshold = float(confidence_threshold)
                if float(score) < threshold:
                    continue
                x0, y0, x1, y1 = [int(round(value)) for value in coords]
                x0 = max(0, min(overlay.width - 1, x0))
                x1 = max(x0 + 1, min(overlay.width, x1))
                y0 = max(0, min(overlay.height - 1, y0))
                y1 = max(y0 + 1, min(overlay.height, y1))
                if "_" in label:
                    defect_type, grade = label.rsplit("_", 1)
                else:
                    defect_type, grade = label, "unknown"
                grade_text = "marginal" if grade == "marg" else "unacceptable" if grade == "unacc" else grade
                severity = "minor" if grade == "marg" else "major" if grade == "unacc" else "ungraded"
                display_type = {"pob": "product on bottle"}.get(defect_type, defect_type.replace("_", " "))
                text_label = f"{display_type} / {grade_text} {float(score):.0%}"
                draw.rectangle((x0, y0, x1, y1), outline=(235, 48, 52), width=max(2, overlay.width // 350))
                draw.text((x0 + 2, max(0, y0 - 18)), text_label, fill=(235, 48, 52), stroke_width=2,
                          stroke_fill=(255, 255, 255))
                predictions.append({
                    "label": label, "type": display_type, "grade": grade_text,
                    "minor_major": severity, "score": float(score), "threshold": threshold,
                    "box_xyxy": [x0, y0, x1, y1], "region_area_pixels": (x1 - x0) * (y1 - y0),
                })
        predictions.sort(key=lambda item: item["score"], reverse=True)
        return {
            "predictions": predictions, "performed": True,
            "scope": "Plastic shampoo-bottle dataset only",
            "source": "Bottle-specific YOLO defect detector",
            "overlay": np.asarray(overlay), "confidence_threshold": float(confidence_threshold),
            "limitation": "The training source has no clean-bottle examples. A missing detection does not establish that a bottle is defect-free.",
        }
    except Exception as exc:
        logger.exception("Bottle YOLO inference failed")
        return {"predictions": [], "performed": False, "scope": "Plastic shampoo-bottle dataset only",
                "source": "Bottle-specific YOLO defect detector", "error": str(exc)}


def infer_product(image, assets, category_override=None):
    tensor = image_transform()(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        feature_tensor = assets["encoder"](tensor).flatten(1).float()
        mlp_probs = torch.softmax(assets["product"](feature_tensor), dim=1)[0].cpu().numpy()
        clip_image = assets["clip_preprocess"](image.convert("RGB")).unsqueeze(0).to(DEVICE)
        clip_feature = assets["clip"].encode_image(clip_image).float()
        clip_feature /= clip_feature.norm(dim=-1, keepdim=True)
        clip_scores, clip_logits = _clip_group_scores(
            clip_feature, assets["clip_product_text"], assets["clip_product_groups"])
    feature = feature_tensor[0].cpu().numpy()
    categories = assets["product_categories"]
    mlp_idx = int(mlp_probs.argmax())
    mlp_category = categories[mlp_idx]
    mlp_confidence = float(mlp_probs[mlp_idx])
    category_distance_threshold = float(assets["calibration"].get(
        "category_distance_p99", assets["calibration"]["distance_p99"]))
    normal_reference_threshold = float(assets["calibration"].get(
        "normal_reference_distance_p99", assets["calibration"]["distance_p99"]))
    confidence_cutoff = float(assets["calibration"]["confidence_p05"])
    mlp_distance = _category_distance(feature, mlp_category, assets)
    mlp_accepted = mlp_confidence >= confidence_cutoff and mlp_distance <= category_distance_threshold
    top_general = max(GENERAL_PROMPTS, key=lambda label: clip_scores.get(f"general::{label}", 0.0))
    top_general_key = f"general::{top_general}"
    supported_clip = max(categories, key=lambda label: clip_scores.get(label, 0.0))
    clip_confidence = clip_scores.get(supported_clip, 0.0)
    clip_ranked_categories = sorted(
        ((label, clip_scores.get(label, 0.0)) for label in categories),
        key=lambda item: item[1], reverse=True,
    )
    clip_margin = clip_logits.get(supported_clip, 0.0) - clip_logits.get(top_general_key, 0.0)
    person_logit = clip_logits.get("general::person", float("-inf"))
    supported_logit = clip_logits.get(supported_clip, float("-inf"))
    person_dominant = person_logit >= supported_logit + CLIP_MIN_LOGIT_MARGIN
    strong_nonproduct_match = (person_dominant or
                               (top_general in NON_PRODUCT_LABELS
                                and clip_margin <= -CLIP_MIN_LOGIT_MARGIN))

    if category_override:
        selected = category_override
        route = "Manual selection"
        selected_confidence = mlp_confidence if selected == mlp_category else 0.0
        domain_validated = False
    elif strong_nonproduct_match:
        selected = None
        route = "Unsupported or uncertain"
        selected_confidence = max(clip_scores.get(top_general_key, 0.0), mlp_confidence)
        domain_validated = False
    elif mlp_accepted:
        selected = mlp_category
        route = "Industrial MLP"
        selected_confidence = mlp_confidence
        domain_validated = True
    elif (clip_confidence >= CLIP_MIN_LABEL_MATCH
          and clip_margin >= CLIP_MIN_LOGIT_MARGIN
          and clip_logits.get(supported_clip, 0.0) > clip_logits.get(top_general_key, 0.0)):
        selected = supported_clip
        route = "Open-vocabulary match"
        selected_confidence = clip_confidence
        domain_validated = False
    elif (top_general in OPEN_VOCABULARY_PRODUCT_LABELS
          and clip_scores.get(top_general_key, 0.0) >= CLIP_MIN_LABEL_MATCH):
        # This pretrained text-image match can identify familiar products beyond
        # the MLP labels, but its score is not a calibrated class probability.
        selected = top_general
        route = "Open-vocabulary product match"
        selected_confidence = clip_scores.get(top_general_key, 0.0)
        domain_validated = False
    else:
        selected = None
        route = "Unsupported or uncertain"
        selected_confidence = max(clip_scores.get(top_general_key, 0.0), clip_confidence)
        domain_validated = False

    has_reference = bool(selected and selected in assets["banks"])
    distance = _distance(feature, selected, assets) if has_reference else None
    anomaly_score = float(np.clip(distance / max(normal_reference_threshold, 1e-8), 0, 3.0)) if distance is not None else None
    anomaly_flag = bool(distance > normal_reference_threshold) if distance is not None else False
    defect_screen = None
    if selected and route != "Unsupported or uncertain":
        defect_screen = _screen_for_defects(image, assets)
        if defect_screen is not None:
            anomaly_score = defect_screen["image_score"]
            anomaly_flag = defect_screen["flagged"]
    # Bottle detections use their source-specific labels; do not layer broad
    # MVTec crop labels over them as if they were bottle-trained classes.
    damage_type_clues = [] if selected == "plastic_shampoo_bottle" else _classify_damage_regions(defect_screen, assets)
    bottle_damage = None
    if selected == "plastic_shampoo_bottle":
        bottle_damage = _detect_bottle_yolo(image, assets)
    if (selected == "plastic_shampoo_bottle" and bottle_damage is None
            and assets.get("bottle_grade_head") is not None):
        region_confirmed = bool(defect_screen is not None and defect_screen["flagged"]
                                and defect_screen["mask"].any())
        if region_confirmed:
            labels = assets["bottle_grade_classes"]
            thresholds = assets["bottle_grade_thresholds"]
            candidates = []
            for region_index, (x0, y0, x1, y1, area) in enumerate(defect_screen["regions"][:5], 1):
                pad_x, pad_y = max(8, int((x1 - x0) * 0.25)), max(8, int((y1 - y0) * 0.25))
                crop = defect_screen["view"].crop((max(0, x0 - pad_x), max(0, y0 - pad_y),
                                                    min(IMAGE_SIZE, x1 + pad_x),
                                                    min(IMAGE_SIZE, y1 + pad_y)))
                crop_tensor = image_transform()(crop).unsqueeze(0).to(DEVICE)
                with torch.inference_mode():
                    features = assets["encoder"](crop_tensor).flatten(1).float()
                    probabilities = torch.softmax(assets["bottle_grade_head"](features), dim=1)[0].cpu().numpy()
                class_index = int(probabilities.argmax())
                label = labels[class_index]
                score = float(probabilities[class_index])
                limit = float(thresholds[class_index])
                if score < limit:
                    continue
                label_type, grade = label.rsplit("_", 1)
                type_name = {"pob": "product on bottle"}.get(label_type, label_type.replace("_", " "))
                candidates.append({"label": label, "type": type_name,
                                   "grade": "marginal" if grade == "marg" else "unacceptable",
                                   "minor_major": "minor" if grade == "marg" else "major",
                                   "score": score, "threshold": limit, "region": region_index,
                                   "region_area_pixels": area})
            bottle_damage = {"predictions": candidates, "performed": True,
                             "scope": "Plastic shampoo bottle dataset only"}
        else:
            bottle_damage = {"predictions": [], "performed": False,
                             "scope": "Plastic shampoo bottle dataset only"}
    return {
        "category": selected, "route": route, "confidence": selected_confidence,
        "reference_available": has_reference,
        "mlp_category": mlp_category, "mlp_confidence": mlp_confidence,
        "clip_category": supported_clip, "clip_confidence": clip_confidence,
        "clip_margin": clip_margin,
        "clip_top_categories": clip_ranked_categories[:5],
        "mlp_distance": mlp_distance, "general_label": top_general,
        "general_score": clip_scores.get(top_general_key, 0.0),
        "distance": distance, "threshold": normal_reference_threshold,
        "category_distance_threshold": category_distance_threshold, "anomaly_score": anomaly_score,
        "anomaly_flag": anomaly_flag, "domain_validated": domain_validated,
        "defect_screen": defect_screen,
        "damage_type_clues": damage_type_clues,
        "bottle_damage": bottle_damage,
        "neu_category": None, "neu_confidence": None,
    }


def infer_neu(image, assets):
    tensor = image_transform()(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        features = assets["encoder"](tensor).flatten(1).float()
        probs = torch.softmax(assets["neu"](features), dim=1)[0].cpu().numpy()
    idx = int(probs.argmax())
    return {"neu_category": assets["neu_categories"][idx], "neu_confidence": float(probs[idx])}


def _render_defect_assessment(screen, manual=False, bottle_damage=None, damage_type_clues=None,
                              domain_calibrated=True):
    if bottle_damage is not None:
        candidates = bottle_damage["predictions"]
        if not bottle_damage.get("performed"):
            message = bottle_damage.get("error", "Bottle detector is unavailable.")
            st.caption(f"Bottle-specific detector did not return a result: {message}")
        elif candidates:
            display = [f"{item['type'].title()} — {item['minor_major'].title()} "
                       f"(dataset grade: {item['grade']}, model score {item['score']:.1%})"
                       for item in candidates[:3]]
            title = ("Bottle YOLO candidate detections: "
                     if "YOLO" in bottle_damage.get("source", "")
                     else "Bottle-specific defect/type grade clues: ")
            st.info(title + "; ".join(display))
            if bottle_damage.get("overlay") is not None:
                st.image(bottle_damage["overlay"], caption="YOLO candidate defect boxes — review each highlighted region", width=480)
            st.caption("This detector was trained on plastic shampoo bottles only; ‘marginal’ is shown as minor and ‘unacceptable’ as major. The source has no clean-bottle examples, so these are candidate clues and human review is required.")
        else:
            st.caption("The bottle-specific detector found no candidate above its confidence threshold. The source has no clean-bottle examples, so this does not prove the bottle is defect-free.")
    if screen is None:
        st.warning("The defect-localization model is not available; no damage-region result was produced.")
        return
    if not domain_calibrated:
        st.info("The damage model ran as an exploratory screen. Its threshold was calibrated on industrial benchmark images, so treat this score and overlay as a visual clue for this product or scene, not a validated defect decision.")
    if damage_type_clues:
        clues = "; ".join(f"{item['type']} ({item['score']:.1%})" for item in damage_type_clues[:3])
        st.info("Possible damage type clue (trained from MVTec defect regions): " + clues)
        st.caption("These broad labels were mapped from MVTec fault-folder names; they are not severity grades or sale decisions.")
    if screen["flagged"] and screen["mask"].any():
        title = "Candidate visual region" if not domain_calibrated else "Possible damage region highlighted"
        st.warning(f"{title} (screen score {screen['image_score']:.3f}). Review the highlighted area.")
        caption = "Exploratory candidate region — red overlay" if not domain_calibrated else "Candidate damage region — red overlay"
        st.image(screen["overlay"], caption=caption, width=320)
        st.write(f"**Visible region:** {screen['area_fraction']:.2%} of the analyzed view ({screen['visible_extent']} extent band).")
        if screen["candidate_type"]:
            st.info(f"Possible defect clue: {screen['candidate_type']} ({screen['candidate_score']:.1%} text-image match; experimental, not a validated defect classifier).")
        st.write("**Severity:** not graded. The training masks have no minor/major or functional-impact labels.")
        st.write("**Sale suitability:** hold for human inspection; this screen cannot approve the item for sale.")
    elif screen["flagged"]:
        st.warning(f"The image score crossed the review threshold ({screen['image_score']:.3f}), but no region crossed the pixel threshold. Review the image manually.")
        st.write("**Severity:** not determined; no region was localized.")
        st.write("**Sale suitability:** not certified by this visual screen.")
    else:
        threshold_text = "calibrated screen threshold" if domain_calibrated else "screen threshold"
        st.warning(f"No region crossed the {threshold_text} (score {screen['image_score']:.3f}); subtle damage can still be missed.")
        st.write("**Severity:** not determined. A low score is not a minor/defect-free label.")
        st.write("**Sale suitability:** not certified by this visual screen.")
    if manual:
        st.caption("You chose the category manually. The visual screen was run, but the category was not confirmed automatically.")
    st.caption("The area band describes highlighted pixels only. It is not a measure of structural safety or sale grade.")


def make_report(records):
    lines = ["# IndustrialGuard AI — visual screening report", "",
             f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}", "",
             "> Visual screening only. This report is not a safety certification or a reliable sale-clearance decision.", ""]
    for index, record in enumerate(records, 1):
        lines += [f"## Item {index}: {record['name']}", ""]
        if record.get("industry_focus"):
            lines.append(f"- Industry focus: **{record['industry_focus']}**")
        if "error" in record:
            lines += [f"- Analysis status: **Failed** — {record['error']}",
                      "- Defect result and sale suitability: Not available.", ""]
            continue
        result = record["result"]
        if record["mode"] == "NEU steel defect class":
            lines += [f"- Surface class: **{result['neu_category']}**",
                      f"- Class confidence: {result['neu_confidence']:.1%}",
                      "- Sale suitability: Not assessed by this classifier.", ""]
            continue
        if result["category"]:
            lines += [f"- Object category: **{result['category'].replace('_', ' ').title()}** ({result['route']})"]
            if result["route"] in {"Open-vocabulary match", "Open-vocabulary product match"}:
                lines.append(f"- Open-vocabulary label match score: {result['confidence']:.1%} (not a calibrated probability)")
            elif result["route"] == "Industrial MLP":
                lines.append(f"- Product classifier confidence: {result['confidence']:.1%}")
            else:
                lines.append("- Category source: user-selected manual override")
            if result.get("defect_screen") is not None:
                screen = result["defect_screen"]
                lines += [f"- Mask-screen maximum score: {screen['image_score']:.4f} (threshold {screen['image_threshold']:.4f})",
                          f"- Image score crossed review threshold: {'yes' if screen['flagged'] else 'no'}",
                          f"- Damage region highlighted: {'yes' if screen['mask'].any() and screen['flagged'] else 'no; this does not rule out defects'}",
                          f"- Highlighted image area: {screen['area_fraction']:.2%} ({screen['visible_extent']} visual-extent band)",
                          f"- Experimental defect clue: {screen['candidate_type'] or 'not available'}",
                          "- Severity: Not graded; benchmark masks do not contain minor/major or functional-impact labels.",
                          "- Sale suitability: **Human review required**; the AI does not clear the item for sale."]
                type_clues = result.get("damage_type_clues", [])
                if type_clues:
                    lines.append("- MVTec-trained damage type clue: " + ", ".join(
                        f"{item['type']} ({item['score']:.1%})" for item in type_clues[:3]))
                if (result["route"] in {"Open-vocabulary match", "Open-vocabulary product match"}
                        or not result.get("reference_available")):
                    lines.append("- Calibration note: Damage screen ran as an exploratory visual cue; its threshold was calibrated on industrial benchmark images, not this product/style.")
                if result.get("bottle_damage") is not None:
                    guesses = result["bottle_damage"]["predictions"]
                    if not result["bottle_damage"].get("performed"):
                        lines.append("- Bottle-specific detector: Not run successfully — " + str(result["bottle_damage"].get("error", "model unavailable")))
                    else:
                        source_name = result["bottle_damage"].get("source", "Bottle-specific type/grade head")
                        lines.append(f"- {source_name} candidates: " + (
                            ", ".join(f"{g['type']} / {g['minor_major']} (dataset grade {g['grade']}, score {g['score']:.1%})" for g in guesses[:3])
                            if guesses else "No candidate passed the confidence threshold; this does not rule out a defect."))
                        if guesses and guesses[0].get("box_xyxy"):
                            lines.append("- Candidate boxes (x0, y0, x1, y1 pixels): " + "; ".join(
                                f"{g['label']}: {g['box_xyxy']}" for g in guesses[:3]))
                    lines.append("- Scope: Bottle defect types and marginal/unacceptable grades were trained on the Plastic Shampoo Bottle dataset only. It has no clean-bottle examples and cannot determine sale suitability.")
                if result["route"] == "Manual selection":
                    lines.append("- Note: The category was selected manually, so category routing was not automatically confirmed.")
            elif result["route"] == "Manual selection" and not result.get("reference_available"):
                lines += ["- Category source: user-entered or user-selected manual product label.",
                          "- Damage screening: unavailable because the localization model did not return a result.",
                          "- Severity and sale suitability: not determined; human inspection is required."]
            elif result["domain_validated"] or result["route"] == "Manual selection":
                lines += [f"- Normal-reference distance: {result['distance']:.3f} (limit {result['threshold']:.3f})",
                          f"- Image-level anomaly score: {result['anomaly_score']:.2f}",
                          f"- Screening result: {'Possible image-level difference; review required' if result['anomaly_flag'] else 'Not flagged by the screen; this does not rule out a defect'}",
                          "- Defect type/severity: Not determined. This prototype has low measured sensitivity and no validated minor/major labels.",
                          "- Sale suitability: **Human inspection required**; AI screening does not clear the item for sale."]
                if result["route"] == "Manual selection":
                    lines.append("- Note: The selected category was not confirmed automatically; its normal-reference comparison may be out of domain.")
            else:
                lines += ["- Defect type/severity: Not assessed; this image is outside the calibrated industrial-image range.",
                          "- Sale suitability: **Cannot determine** from available product reference data."]
        else:
            lines += [f"- General visual match: **{result['general_label']}** ({result['general_score']:.1%})",
                      "- Industrial defect inspection: Not run; no supported product category was identified.",
                      "- Sale suitability: **Cannot determine**."]
        lines.append("")
    lines += ["## Limitations", "",
              "The object recognizer covers a limited set of trained product labels; it cannot guarantee recognition of every object. The defect segmenter localizes candidate regions from industrial benchmark masks. Its broad damage-type head learns labels mapped from MVTec fault-folder names and may not transfer to unrelated products; the separate bottle head uses plastic shampoo bottle annotations only. Neither model establishes sale suitability. Severity for other product classes and sale suitability require product-specific labels and human review."]
    return "\n".join(lines)


st.set_page_config(page_title="IndustrialGuard AI | Visual Inspection", page_icon="🏭", layout="wide")
st.markdown("""
<style>
.stApp {background:#f4f7fb;} .block-container {max-width:1180px;padding-top:1.4rem;}
.hero {padding:1.35rem 1.7rem;border-radius:18px;color:white;background:linear-gradient(115deg,#142b4a,#137b79);margin-bottom:1rem;}
.hero h1 {margin:0;font-size:2.1rem;} .hero p {opacity:.92;margin:.4rem 0 0;}
.smallnote {color:#526276;font-size:.92rem;} [data-testid="stMetric"] {background:white;border:1px solid #e4ebf2;padding:13px;border-radius:14px;}
.model-step {background:white;border:1px solid #e4ebf2;border-radius:12px;padding:12px 14px;margin:.4rem 0;color:#20344d;}
</style>
<div class="hero"><h1>IndustrialGuard AI</h1><p>Recognize the product, locate possible wear or damage, and create a review report</p></div>
""", unsafe_allow_html=True)

optional_checkpoints = [ROOT / "models" / "product_mlp.pt",
                        ROOT / "models" / "neu_defect_mlp.pt",
                        ROOT / "models" / "resnet18_encoder.pt",
                        ROOT / "models" / "normal_feature_banks.npz",
                        ROOT / "models" / "category_feature_banks.npz",
                        ROOT / "models" / "ood_calibration.json",
                        ROOT / "models" / "defect_segmenter.pt",
                        ROOT / "models" / "bottle_damage_mlp.pt",
                        ROOT / "models" / "bottle_defect_yolo.pt",
                        ROOT / "models" / "damage_type_mlp.pt"]
asset_checkpoint_stamp = tuple(path.stat().st_mtime_ns if path.exists() else 0
                               for path in optional_checkpoints)
assets, error = load_assets(asset_checkpoint_stamp)
if error:
    st.error(error)
    st.stop()

st.subheader("Inspection setup")
st.markdown("<div class='model-step'><b>Model 1 — Product recognition:</b> trained MLP labels familiar industrial products; a visual matcher can suggest familiar additional products.</div><div class='model-step'><b>Model 2 — Wear and damage:</b> a separate image model highlights candidate regions. Outputs on products outside the benchmark image style are exploratory and need human review.</div>", unsafe_allow_html=True)
st.caption("The general damage model localizes candidate regions. A bottle-specific YOLO detector reports type/grade clues only when the product is identified as a plastic shampoo bottle. Neither result certifies sale fitness for general products.")
if assets.get("defect_segmenter") is None:
    st.warning("The wear-and-damage model checkpoint is not loaded yet. Train the second model and refresh this page before analyzing defects.")
mode = st.radio("Analysis mode", ["Object identification + industrial screening", "NEU steel defect class"])
if mode == "NEU steel defect class":
    industry_focus = "Steel surfaces (NEU mode)"
    manual_product_categories = assets["product_categories"]
    st.caption("Industry focus: steel surfaces. This mode uses the separate six-class NEU steel model.")
else:
    industry_options = ["All industries", *INDUSTRY_CATEGORY_GROUPS]
    industry_focus = st.selectbox("Industry focus", industry_options, key="industry_focus")
    if industry_focus == "All industries":
        manual_product_categories = assets["product_categories"]
    else:
        sector_categories = INDUSTRY_CATEGORY_GROUPS[industry_focus]
        manual_product_categories = [category for category in assets["product_categories"]
                                     if category in sector_categories]
    st.caption("Industry focus organizes the manual category list. Automatic detection still checks all trained product labels.")
source = st.radio("Image source", ["Upload images", "Camera"], horizontal=True)
if source == "Upload images":
    files = st.file_uploader("Choose one or more images", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"], accept_multiple_files=True)
    items = [{"name": f.name, "data": f.getvalue()} for f in (files or [])]
else:
    photo = st.camera_input("Take a picture")
    items = [{"name": "camera_capture.jpg", "data": photo.getvalue()}] if photo else []
st.caption(f"Analyze several images together. The trained MLP recognizes its {len(assets['product_categories'])} product labels; a pretrained visual matcher can suggest familiar additional products. Choose Other product (manual) for an item outside those labels. Damage results outside benchmark-style images are exploratory cues and need human review.")
analyze_clicked = st.button("Analyze images", type="primary", disabled=not items, use_container_width=True)


signature_source = mode + "|" + "|".join(
    f"{item['name']}:{hashlib.sha256(item['data']).hexdigest()}" for item in items
)
signature = hashlib.sha256(signature_source.encode()).hexdigest()[:12]
if st.session_state.get("input_signature") != signature:
    st.session_state["input_signature"] = signature
    st.session_state.pop("analysis_records", None)

if analyze_clicked:
    records = []
    progress = st.progress(0, text="Analyzing images…")
    for index, item in enumerate(items):
        try:
            image = Image.open(io.BytesIO(item["data"])).convert("RGB")
            result = infer_neu(image, assets) if mode == "NEU steel defect class" else infer_product(image, assets)
            records.append({"name": item["name"], "data": item["data"], "mode": mode,
                            "industry_focus": industry_focus,
                            "result": result, "automatic_result": result})
        except Exception as exc:
            records.append({"name": item["name"], "data": item["data"], "mode": mode,
                            "industry_focus": industry_focus,
                            "error": f"Image analysis failed: {exc}"})
        progress.progress((index + 1) / len(items), text=f"Analyzed {index + 1} of {len(items)} images")
    progress.empty()
    st.session_state["analysis_records"] = records

records = st.session_state.get("analysis_records", [])
st.subheader("Analysis results")
if not records:
    st.info("Choose one or more images, then press Analyze images.")
for index, record in enumerate(records):
    with st.container(border=True):
        image_col, result_col = st.columns([.85, 1.15], gap="medium")
        with image_col:
            st.caption(record["name"])
            st.image(Image.open(io.BytesIO(record["data"])).convert("RGB"), width="stretch")
        with result_col:
            if "error" in record:
                st.error(record["error"])
                continue
            result = record.get("automatic_result", record["result"])
            if record["mode"] == "NEU steel defect class":
                st.success(f"NEU surface class: {result['neu_category'].replace('_', ' ')}")
                st.metric("Class confidence", f"{result['neu_confidence']:.1%}")
                st.caption("This is a six-class steel-surface model. It does not determine sale suitability.")
                continue

            options = ["Automatic detection", *manual_product_categories, "other_product"]
            override = st.selectbox(
                "Category override for this image", options,
                format_func=lambda value: "Other product (manual)" if value == "other_product" else value.replace("_", " ").title() if value != "Automatic detection" else value,
                key=f"override_{signature}_{index}_{industry_focus}")
            manual_name = ""
            if override == "other_product":
                manual_name = st.text_input("Product name (optional)", key=f"manual_name_{signature}_{index}").strip()
            if override != "Automatic detection":
                image = Image.open(io.BytesIO(record["data"])).convert("RGB")
                result = infer_product(image, assets, category_override=manual_name or override)
            record["result"] = result

            if result["category"]:
                route_text = result["route"].lower()
                st.success(f"Identified as {result['category'].replace('_', ' ').title()} ({route_text})")
                a, b, c = st.columns(3)
                a.metric("Object category", result["category"].replace("_", " ").title())
                if result["route"] == "Open-vocabulary match":
                    b.metric("Open-vocabulary match", f"{result['confidence']:.1%}")
                    c.metric("Damage screen", "Exploratory")
                elif result["route"] == "Open-vocabulary product match":
                    b.metric("Open-vocabulary match", f"{result['confidence']:.1%}")
                    c.metric("Damage screen", "Exploratory")
                elif result["route"] == "Manual selection":
                    b.metric("Category source", "Manual")
                    if result.get("defect_screen") is not None:
                        c.metric("Damage screen score", f"{result['anomaly_score']:.2f}")
                    elif result.get("anomaly_score") is not None:
                        c.metric("Reference difference", f"{result['anomaly_score']:.2f}")
                    else:
                        c.metric("Damage screen", "Unavailable")
                else:
                    b.metric("Classifier confidence", f"{result['confidence']:.1%}")
                    c.metric("Damage screen score" if result.get("defect_screen") is not None else "Reference difference", f"{result['anomaly_score']:.2f}")
                if result["route"] == "Open-vocabulary match":
                    st.info(f"The general visual model matched {result['category'].replace('_', ' ')}; its label-match score is not a calibrated probability. The industrial MLP's closest class was {result['mlp_category'].replace('_', ' ')} ({result['mlp_confidence']:.1%}).")
                    if result.get("defect_screen") is not None:
                        _render_defect_assessment(result["defect_screen"],
                                                   bottle_damage=result.get("bottle_damage"),
                                                   damage_type_clues=result.get("damage_type_clues"),
                                                   domain_calibrated=False)
                    st.warning("Sale suitability: cannot determine from this exploratory screen. Human inspection is required.")
                elif result["route"] == "Open-vocabulary product match":
                    st.info(f"Possible product: {result['category'].replace('_', ' ')} (open-vocabulary label-match score {result['confidence']:.1%}; not a calibrated probability). The industrial MLP's closest class was {result['mlp_category'].replace('_', ' ')} ({result['mlp_confidence']:.1%}).")
                    if result.get("defect_screen") is not None:
                        _render_defect_assessment(result["defect_screen"],
                                                   bottle_damage=result.get("bottle_damage"),
                                                   damage_type_clues=result.get("damage_type_clues"),
                                                   domain_calibrated=False)
                    st.warning("Sale suitability: cannot be determined from this exploratory screen. Human inspection is required.")
                elif result["route"] == "Manual selection":
                    if not result.get("reference_available"):
                        st.info("You chose Other product manually. The damage model can suggest a visual region, but no matching product reference or calibrated category result is available.")
                    else:
                        st.info("You selected this category manually; its normal references were used for comparison.")
                    if result.get("defect_screen") is not None:
                        _render_defect_assessment(result["defect_screen"], manual=True,
                                                   domain_calibrated=result.get("reference_available", False),
                                                   damage_type_clues=result.get("damage_type_clues"),
                                                   bottle_damage=result.get("bottle_damage"))
                    elif result["anomaly_flag"]:
                        st.warning(f"The image differs from normal references (screening score {result['anomaly_score']:.2f}). This may reflect a defect or a different image style; review manually.")
                    else:
                        st.warning(f"The screen did not flag this image (score {result['anomaly_score']:.2f}); it can miss defects. A human check is still needed.")
                else:
                    if result.get("defect_screen") is not None:
                        _render_defect_assessment(result["defect_screen"],
                                                   bottle_damage=result.get("bottle_damage"),
                                                   damage_type_clues=result.get("damage_type_clues"))
                    elif result["anomaly_flag"]:
                        st.warning(f"Possible visual deviation (score {result['anomaly_score']:.2f}). The image-level screen does not localize or reliably name the defect; inspect the image manually.")
                        st.write("**Severity:** uncertain — no validated minor/major labels are available.")
                        st.write("**Sale suitability:** hold for human inspection.")
                    else:
                        st.warning(f"The screen did not flag this image (score {result['anomaly_score']:.2f}); it can miss defects. A human check is still needed.")
                        st.write("**Severity:** not graded by this model.")
                        st.write("**Sale suitability:** cannot be certified by this visual screen.")
                    if result.get("defect_screen") is None:
                        st.caption("This whole-image reference score does not mark defect locations.")
            else:
                st.error("Unsupported or uncertain product image. No industrial defect inspection was run.")
                st.caption(f"General visual match: {result['general_label']} ({result['general_score']:.1%}). Closest industrial MLP class: {result['mlp_category']} ({result['mlp_confidence']:.1%}). Choose a trained category or Other product (manual) if you know the item.")

if records:
    st.divider()
    st.download_button("Download analysis report", data=make_report(records),
                       file_name="industrialguard_inspection_report.md", mime="text/markdown")
    st.caption("The report separates object identification from defect screening. It does not certify an item as fit for sale.")

st.divider()
st.markdown("<p class='smallnote'>Model 1 identifies a product category. Model 2 highlights candidate damage regions and may classify broad damage types from MVTec labels. The optional bottle YOLO detector uses shampoo-bottle box labels only. Other products still need product-specific severity labels and human review; the app does not certify sale suitability.</p>", unsafe_allow_html=True)
