"""Train a binary defect-localization model from public industrial anomaly masks."""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, UnidentifiedImageError
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

from .train import DATA, DEVICE, IMAGE_SIZE, MEAN, OUT, ROOT, STD


SEED = 73
OUTPUT_SIZE = IMAGE_SIZE


def _images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in
                  {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})


def _split_group(rows: list[dict], rng: np.random.Generator) -> None:
    order = rng.permutation(len(rows))
    if len(rows) < 3:
        for i, idx in enumerate(order):
            rows[int(idx)]["split"] = "train" if i < 1 else "test"
        return
    n_train = max(1, int(round(len(rows) * 0.70)))
    n_val = max(1, int(round(len(rows) * 0.15)))
    if n_train + n_val >= len(rows):
        n_train = len(rows) - 2
        n_val = 1
    for i, idx in enumerate(order):
        rows[int(idx)]["split"] = "train" if i < n_train else ("validation" if i < n_train + n_val else "test")


def discover_segmentation_rows() -> list[dict]:
    """Collect real image/mask pairs and source-native normal examples."""
    rows: list[dict] = []
    anomaly_groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    rng = np.random.default_rng(SEED)

    def add(image: Path, mask: Path | None, category: str, dataset: str,
            anomalous: bool, group: str, split: str | None = None,
            bbox_labels: Path | None = None):
        row = {"image": str(image), "mask": str(mask) if mask else None,
               "category": category.lower().replace(" ", "_"), "dataset": dataset,
               "anomalous": int(anomalous), "split": split,
               "bbox_labels": str(bbox_labels) if bbox_labels else None}
        if anomalous and split is None:
            anomaly_groups[(dataset, row["category"], group)].append(row)
        else:
            rows.append(row)

    # MVTec AD: masks are named <image-stem>_mask.png under ground_truth/<fault>.
    mvtec = DATA / "mvtec_ad"
    if mvtec.exists():
        for category in sorted(p for p in mvtec.iterdir() if p.is_dir()):
            for image in _images(category / "train" / "good"):
                add(image, None, category.name, "MVTec AD", False, "", "train")
            test_good = category / "test" / "good"
            for image in _images(test_good):
                add(image, None, category.name, "MVTec AD", False, "", "test")
            for defect in sorted(p for p in (category / "test").iterdir()
                                 if p.is_dir() and p.name.lower() != "good"):
                for image in _images(defect):
                    mask = category / "ground_truth" / defect.name / f"{image.stem}_mask.png"
                    if mask.is_file():
                        add(image, mask, category.name, "MVTec AD", True, defect.name)

    # VisA official CSV supplies the image and pixel-mask paths.
    visa_csv = DATA / "visa" / "split_csv" / "1cls.csv"
    if visa_csv.is_file():
        with visa_csv.open("r", newline="", encoding="utf-8-sig") as stream:
            for item in csv.DictReader(stream):
                image = DATA / "visa" / item["image"]
                is_bad = item["label"].strip().lower() == "anomaly"
                if is_bad:
                    mask = DATA / "visa" / item["mask"] if item.get("mask") else None
                    if mask and mask.is_file():
                        add(image, mask, item["object"], "VisA", True, item["object"])
                else:
                    split = "test" if item["split"].strip().lower() == "test" else "train"
                    add(image, None, item["object"], "VisA", False, "", split)

    # MVTec AD 2 public masks. Private test images are never read.
    ad2 = DATA / "mvtec_ad_2"
    if ad2.exists():
        for category in sorted(p for p in ad2.iterdir() if p.is_dir()):
            for image in _images(category / "train" / "good"):
                add(image, None, category.name, "MVTec AD 2", False, "", "train")
            for image in _images(category / "validation" / "good"):
                add(image, None, category.name, "MVTec AD 2", False, "", "validation")
            for image in _images(category / "test_public" / "good"):
                add(image, None, category.name, "MVTec AD 2", False, "", "test")
            bad_dir = category / "test_public" / "bad"
            for image in _images(bad_dir):
                mask = category / "test_public" / "ground_truth" / "bad" / f"{image.stem}_mask.png"
                if mask.is_file():
                    add(image, mask, category.name, "MVTec AD 2", True, image.stem.split("_")[0])

    # The shampoo-bottle source has box annotations rather than pixel masks.
    # Rasterize each annotated rectangle at native image resolution as a weak
    # localization target; its held-out split is evaluated separately below.
    bottle_root = DATA / "additional" / "Plastic Shampoo Bottle Dataset"
    if bottle_root.exists():
        for folder, split in (("train", "train"), ("valid", "validation"), ("test", "test")):
            image_dir = bottle_root / folder / "images"
            labels_dir = bottle_root / folder / "labels"
            for image in _images(image_dir):
                label_path = labels_dir / f"{image.stem}.txt"
                if not label_path.is_file():
                    raise FileNotFoundError(f"YOLO box labels missing for {image}")
                has_box = any(line.strip() for line in label_path.read_text(encoding="utf-8").splitlines())
                add(image, None, "bottle", "Shampoo Bottle YOLO boxes", has_box,
                    "", split, bbox_labels=label_path)

    for group in anomaly_groups.values():
        _split_group(group, rng)
        rows.extend(group)

    # Hold out a slice of normal source-train images for validation.
    normal_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if not row["anomalous"] and row["split"] == "train":
            normal_groups[(row["dataset"], row["category"])].append(row)
    for group in normal_groups.values():
        if len(group) < 5:
            continue
        indices = rng.permutation(len(group))
        n_val = max(1, int(round(len(group) * 0.10)))
        for idx in indices[:n_val]:
            group[int(idx)]["split"] = "validation"

    return rows


class DefectMaskDataset(Dataset):
    def __init__(self, rows: list[dict], training: bool = False):
        self.rows = rows
        self.training = training
        self.resize_image = transforms.Resize(256, interpolation=InterpolationMode.BICUBIC)
        self.resize_mask = transforms.Resize(256, interpolation=InterpolationMode.NEAREST)
        self.crop = transforms.CenterCrop(OUTPUT_SIZE)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(MEAN, STD)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        try:
            image = Image.open(row["image"]).convert("RGB")
            if row.get("mask"):
                mask = Image.open(row["mask"]).convert("L")
            else:
                mask = Image.new("L", image.size, 0)
                if row.get("bbox_labels"):
                    draw = ImageDraw.Draw(mask)
                    width, height = image.size
                    for line in Path(row["bbox_labels"]).read_text(encoding="utf-8").splitlines():
                        fields = line.split()
                        if len(fields) < 5:
                            continue
                        _, center_x, center_y, box_width, box_height = map(float, fields[:5])
                        x0 = max(0, int(round((center_x - box_width / 2) * width)))
                        y0 = max(0, int(round((center_y - box_height / 2) * height)))
                        x1 = min(width - 1, int(round((center_x + box_width / 2) * width)))
                        y1 = min(height - 1, int(round((center_y + box_height / 2) * height)))
                        draw.rectangle((x0, y0, x1, y1), fill=255)
        except (OSError, UnidentifiedImageError) as exc:
            raise RuntimeError(f"Cannot read image/mask for {row['image']}: {exc}") from exc
        image, mask = self.resize_image(image), self.resize_mask(mask)
        image, mask = self.crop(image), self.crop(mask)
        if self.training:
            if random.random() < 0.5:
                image, mask = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                image, mask = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM), mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            turns = random.randrange(4)
            if turns:
                angle = 90 * turns
                image = image.rotate(angle, resample=Image.Resampling.BICUBIC)
                mask = mask.rotate(angle, resample=Image.Resampling.NEAREST)
        image_tensor = self.normalize(self.to_tensor(image))
        mask_tensor = torch.from_numpy((np.asarray(mask, dtype=np.uint8) > 0).astype(np.float32)).unsqueeze(0)
        weak_target = row["dataset"] == "Shampoo Bottle YOLO boxes" and bool(row["anomalous"])
        return image_tensor, mask_tensor, row["anomalous"], index, weak_target


def _block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
    )


class DefectSegmenter(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = models.resnet18(weights=None)
        self.up3 = nn.Conv2d(512, 256, 1)
        self.dec3 = _block(512, 256)
        self.up2 = nn.Conv2d(256, 128, 1)
        self.dec2 = _block(256, 128)
        self.up1 = nn.Conv2d(128, 64, 1)
        self.dec1 = _block(128, 64)
        self.up0 = nn.Conv2d(64, 64, 1)
        self.dec0 = _block(128, 64)
        self.final = nn.Sequential(_block(64, 32), nn.Conv2d(32, 1, 1))

    def forward(self, x):
        target_size = x.shape[-2:]
        e0 = self.encoder.relu(self.encoder.bn1(self.encoder.conv1(x)))
        e1 = self.encoder.layer1(self.encoder.maxpool(e0))
        e2 = self.encoder.layer2(e1)
        e3 = self.encoder.layer3(e2)
        e4 = self.encoder.layer4(e3)
        d = F.interpolate(self.up3(e4), size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d = self.dec3(torch.cat((d, e3), dim=1))
        d = F.interpolate(self.up2(d), size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d = self.dec2(torch.cat((d, e2), dim=1))
        d = F.interpolate(self.up1(d), size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d = self.dec1(torch.cat((d, e1), dim=1))
        d = F.interpolate(self.up0(d), size=e0.shape[-2:], mode="bilinear", align_corners=False)
        d = self.dec0(torch.cat((d, e0), dim=1))
        d = F.interpolate(d, size=target_size, mode="bilinear", align_corners=False)
        return self.final(d)


def load_pretrained_encoder(segmenter: DefectSegmenter, path: Path) -> None:
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    source = bundle["state_dict"]
    prefixes = {"0": "conv1", "1": "bn1", "4": "layer1", "5": "layer2", "6": "layer3", "7": "layer4"}
    mapped = {}
    for key, value in source.items():
        first, sep, remainder = key.partition(".")
        if first in prefixes and sep:
            mapped[f"{prefixes[first]}.{remainder}"] = value
    result = segmenter.encoder.load_state_dict(mapped, strict=False)
    if not result.missing_keys or any(k.startswith(("conv1", "bn1", "layer1", "layer2", "layer3", "layer4"))
                                        for k in result.missing_keys):
        raise RuntimeError(f"Could not initialize the segmentation encoder: {result.missing_keys[:5]}")


def _loss(logits, masks, bce, weak_targets=None):
    binary = F.binary_cross_entropy_with_logits(
        logits, masks, pos_weight=bce.pos_weight, reduction="none").mean(dim=(1, 2, 3))
    probs = torch.sigmoid(logits)
    intersection = (probs * masks).sum(dim=(1, 2, 3))
    denominator = probs.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0))
    per_image = binary + 0.5 * dice
    if weak_targets is None:
        return per_image.mean()
    weights = torch.where(weak_targets.bool(), 0.25, 1.0).to(per_image.device)
    return (per_image * weights).sum() / weights.sum().clamp_min(1.0)


def evaluate(model, loader, device, bce, collect=False):
    model.eval()
    losses, image_scores, image_labels = [], [], []
    pixel_scores, pixel_labels = [], []
    with torch.inference_mode():
        for images, masks, labels, _, weak_targets in loader:
            images, masks = images.to(device), masks.to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.float16):
                logits = model(images)
                losses.append(float(_loss(logits, masks, bce, weak_targets.to(device)).item()))
            probs = torch.sigmoid(logits.float())
            image_scores.extend(probs.amax(dim=(1, 2, 3)).cpu().numpy().tolist())
            image_labels.extend(torch.as_tensor(labels).cpu().numpy().tolist())
            if collect:
                small_probs = F.interpolate(probs, size=(56, 56), mode="bilinear", align_corners=False)
                small_masks = F.interpolate(masks, size=(56, 56), mode="nearest")
                pixel_scores.append(small_probs.cpu().numpy().reshape(-1))
                pixel_labels.append(small_masks.cpu().numpy().reshape(-1).astype(np.uint8))
    scores = np.asarray(image_scores, dtype=np.float32)
    labels = np.asarray(image_labels, dtype=np.uint8)
    auc = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None
    result = {"loss": float(np.mean(losses)), "image_auc": auc,
              "image_scores": scores, "image_labels": labels}
    if collect:
        result["pixel_scores"] = np.concatenate(pixel_scores) if pixel_scores else np.zeros(0, dtype=np.float32)
        result["pixel_labels"] = np.concatenate(pixel_labels) if pixel_labels else np.zeros(0, dtype=np.uint8)
    return result


def _f1_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    candidates = np.unique(np.concatenate((scores, np.linspace(0.0, 1.0, 301))))
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in candidates:
        prediction = scores >= threshold
        tp = np.sum(prediction & (labels == 1))
        fp = np.sum(prediction & (labels == 0))
        fn = np.sum((~prediction) & (labels == 1))
        value = float(2 * tp / max(1, (2 * tp + fp + fn)))
        if value > best_f1 + 1e-9 or (abs(value - best_f1) <= 1e-9 and threshold < best_threshold):
            best_threshold, best_f1 = float(threshold), value
    return best_threshold, best_f1


def _source_test_metrics(model, rows, device, bce, batch_size, image_threshold, pixel_threshold):
    groups = {
        "Industrial benchmark pixel masks": [r for r in rows if r["dataset"] != "Shampoo Bottle YOLO boxes"],
        "Shampoo bottle bounding boxes": [r for r in rows if r["dataset"] == "Shampoo Bottle YOLO boxes"],
    }
    output = {}
    for name, subset in groups.items():
        if not subset:
            continue
        loader = DataLoader(DefectMaskDataset(subset), batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=device.type == "cuda")
        result = evaluate(model, loader, device, bce, collect=True)
        image_scores, image_labels = result["image_scores"], result["image_labels"]
        image_pred = image_scores >= image_threshold
        pixel_labels, pixel_scores = result["pixel_labels"], result["pixel_scores"]
        pixel_pred = pixel_scores >= pixel_threshold
        intersection = int((pixel_pred & (pixel_labels == 1)).sum())
        union = int((pixel_pred | (pixel_labels == 1)).sum())
        positives, negatives = image_labels == 1, image_labels == 0
        output[name] = {
            "images": len(subset),
            "annotation_type": "YOLO boxes rasterized as weak rectangular targets" if "Shampoo" in name else "pixel masks",
            "image_roc_auc": float(roc_auc_score(image_labels, image_scores)) if len(np.unique(image_labels)) == 2 else None,
            "flag_rate_on_defect_images": float(image_pred[positives].mean()) if positives.any() else None,
            "specificity_on_normal_images": float((~image_pred[negatives]).mean()) if negatives.any() else None,
            "pixel_or_box_region_auc_at_56x56": float(roc_auc_score(pixel_labels, pixel_scores)) if len(np.unique(pixel_labels)) == 2 else None,
            "region_iou_at_pixel_threshold": float(intersection / max(1, union)),
        }
    return output


def _cable_test_metrics(model, rows, device, bce, batch_size, image_threshold, pixel_threshold):
    subset = [row for row in rows if row["dataset"] == "MVTec AD" and row["category"] == "cable"]
    if not subset:
        return None
    loader = DataLoader(DefectMaskDataset(subset), batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=device.type == "cuda")
    result = evaluate(model, loader, device, bce, collect=True)
    scores, labels = result["image_scores"], result["image_labels"]
    flagged = scores >= image_threshold
    positives, negatives = labels == 1, labels == 0
    pixel_labels, pixel_scores = result["pixel_labels"], result["pixel_scores"]
    pixel_pred = pixel_scores >= pixel_threshold
    intersection = int((pixel_pred & (pixel_labels == 1)).sum())
    union = int((pixel_pred | (pixel_labels == 1)).sum())
    return {
        "images": len(subset), "normal_images": int(negatives.sum()),
        "defective_images": int(positives.sum()),
        "image_roc_auc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
        "sensitivity_at_calibrated_threshold": float(flagged[positives].mean()) if positives.any() else None,
        "specificity_at_calibrated_threshold": float((~flagged[negatives]).mean()) if negatives.any() else None,
        "pixel_auc_at_56x56": float(roc_auc_score(pixel_labels, pixel_scores)) if len(np.unique(pixel_labels)) == 2 else None,
        "region_iou_at_calibrated_threshold": float(intersection / max(1, union)),
    }


def main():
    parser = argparse.ArgumentParser(description="Train the IndustrialGuard AI defect-localization model")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Optional small-run cap for debugging only")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Recalibrate the existing checkpoint on validation masks and test it without retraining")
    args = parser.parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = True

    all_rows = discover_segmentation_rows()
    if args.limit:
        for split in ("train", "validation", "test"):
            rows = [r for r in all_rows if r["split"] == split]
            all_rows = [r for r in all_rows if r["split"] != split] + rows[:args.limit]
    train_rows = [r for r in all_rows if r["split"] == "train"]
    val_rows = [r for r in all_rows if r["split"] == "validation"]
    test_rows = [r for r in all_rows if r["split"] == "test"]
    print(json.dumps({"rows": len(all_rows), "train": len(train_rows), "validation": len(val_rows),
                      "test": len(test_rows),
                      "masks": sum(r["anomalous"] for r in all_rows),
                      "device": str(DEVICE)}, indent=2), flush=True)
    for split, subset in (("train", train_rows), ("validation", val_rows), ("test", test_rows)):
        if not subset:
            raise SystemExit(f"No {split} images were found.")
    train_pos = sum(r["anomalous"] for r in train_rows)
    train_neg = len(train_rows) - train_pos
    if train_pos < 5 or train_neg < 5:
        raise SystemExit("Both normal images and masked defective images are needed to train the model.")

    train_set = DefectMaskDataset(train_rows, training=True)
    val_set = DefectMaskDataset(val_rows)
    test_set = DefectMaskDataset(test_rows)
    weights = [0.5 / train_pos if r["anomalous"] else 0.5 / train_neg for r in train_rows]
    samples_per_epoch = 2 * min(train_pos, train_neg)
    sampler = WeightedRandomSampler(weights, num_samples=samples_per_epoch, replacement=True)
    common = {"batch_size": args.batch_size, "num_workers": 4, "pin_memory": DEVICE.type == "cuda"}
    train_loader = DataLoader(train_set, sampler=sampler, drop_last=True, **common)
    val_loader = DataLoader(val_set, shuffle=False, **common)
    test_loader = DataLoader(test_set, shuffle=False, **common)

    model = DefectSegmenter()
    checkpoint_path = OUT / "defect_segmenter.pt"
    if args.evaluate_only:
        if not checkpoint_path.is_file():
            raise SystemExit(f"No trained checkpoint to evaluate: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(DEVICE).eval()
        bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0], device=DEVICE))
        validation = evaluate(model, val_loader, DEVICE, bce, collect=True)
        test = evaluate(model, test_loader, DEVICE, bce, collect=True)
        val_scores, val_labels = validation["image_scores"], validation["image_labels"]
        test_scores, test_labels = test["image_scores"], test["image_labels"]
        threshold, validation_f1 = _f1_threshold(val_scores, val_labels)
        normal_scores = val_scores[val_labels == 0]
        conservative_threshold = float(np.quantile(normal_scores, 0.995))
        high_recall_threshold = float(np.quantile(normal_scores, 0.95))

        def image_metrics(cutoff):
            prediction = test_scores >= cutoff
            tp = int((prediction & (test_labels == 1)).sum())
            fp = int((prediction & (test_labels == 0)).sum())
            fn = int(((~prediction) & (test_labels == 1)).sum())
            tn = int(((~prediction) & (test_labels == 0)).sum())
            return {"threshold": float(cutoff), "sensitivity": tp / max(1, tp + fn),
                    "specificity": tn / max(1, tn + fp), "precision": tp / max(1, tp + fp),
                    "f1": 2 * tp / max(1, 2 * tp + fp + fn)}

        checkpoint["image_score_threshold"] = threshold
        checkpoint["normal_only_p995_threshold"] = conservative_threshold
        checkpoint["threshold_selection"] = "max F1 on held-out validation images containing normal and masked-defect examples"
        checkpoint["weak_bottle_box_loss_weight"] = 0.25
        torch.save(checkpoint, checkpoint_path)

        pixels_y, pixels_p = test["pixel_labels"], test["pixel_scores"]
        pixel_auc = float(roc_auc_score(pixels_y, pixels_p)) if len(np.unique(pixels_y)) == 2 else None
        pixel_threshold = float(checkpoint["pixel_threshold"])
        pixel_pred = pixels_p >= pixel_threshold
        intersection = int((pixel_pred & (pixels_y == 1)).sum())
        union = int((pixel_pred | (pixels_y == 1)).sum())
        metrics = {
            "architecture": checkpoint.get("architecture", "ResNet18 U-Net binary defect segmentation"),
            "best_epoch": checkpoint.get("best_epoch"),
            "split_counts": {split: {"images": len(rows), "normal": sum(not r["anomalous"] for r in rows),
                                      "defective": sum(bool(r["anomalous"]) for r in rows)}
                             for split, rows in (("train", train_rows), ("validation", val_rows), ("test", test_rows))},
            "calibration": {"image_score_threshold": threshold, "validation_image_f1_at_threshold": validation_f1,
                            "normal_only_p995_threshold": conservative_threshold,
                            "normal_only_p95_threshold": high_recall_threshold,
                            "normal_validation_images": int((val_labels == 0).sum()),
                            "defect_validation_images": int((val_labels == 1).sum()),
                            "method": checkpoint["threshold_selection"]},
            "held_out_test": {"images": len(test_rows), "image_roc_auc": test["image_auc"],
                              "at_validation_max_f1_threshold": image_metrics(threshold),
                              "at_normal_only_p995_threshold": image_metrics(conservative_threshold),
                              "at_normal_only_p95_threshold": image_metrics(high_recall_threshold),
                              "mixed_pixel_or_box_region_auc_at_56x56": pixel_auc,
                              "mixed_region_iou_at_calibrated_threshold": float(intersection / max(1, union)),
                              "cable_MVTec_AD": _cable_test_metrics(
                                  model, test_rows, DEVICE, bce, args.batch_size,
                                  threshold, pixel_threshold),
                              "by_annotation_source": _source_test_metrics(
                                  model, test_rows, DEVICE, bce, args.batch_size,
                                  threshold, pixel_threshold)},
            "limitations": "The U-Net localizes binary candidate regions. Shampoo-bottle bounding boxes are weak rectangular targets, downweighted to 0.25 of the per-image loss and evaluated separately from true pixel masks. The attached type and bottle heads have narrower label scopes; no head certifies general severity or sale suitability.",
        }
        report_path = ROOT / "reports" / "defect_segmentation_metrics.json"
        report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        aggregate_path = ROOT / "reports" / "training_metrics.json"
        if aggregate_path.exists():
            aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
            aggregate["defect_segmentation"] = metrics
            aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2), flush=True)
        return

    load_pretrained_encoder(model, OUT / "resnet18_encoder.pt")
    model.to(DEVICE)
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": 1e-5},
        {"params": [p for n, p in model.named_parameters() if not n.startswith("encoder.")], "lr": 3e-4},
    ], weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0], device=DEVICE))
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    best_val, best_state, best_epoch = float("inf"), None, 0
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for step, (images, masks, _, _, weak_targets) in enumerate(train_loader, 1):
            images, masks = images.to(DEVICE, non_blocking=True), masks.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda", dtype=torch.float16):
                logits = model(images)
                loss = _loss(logits, masks, bce, weak_targets.to(DEVICE, non_blocking=True))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        validation = evaluate(model, val_loader, DEVICE, bce)
        print(f"epoch {epoch + 1:02d}/{args.epochs} train_loss={np.mean(losses):.4f} "
              f"val_loss={validation['loss']:.4f} val_image_auc={validation['image_auc']}", flush=True)
        if validation["loss"] < best_val:
            best_val = validation["loss"]
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    calibration = evaluate(model, val_loader, DEVICE, bce, collect=True)
    normal_max = calibration["image_scores"][calibration["image_labels"] == 0]
    normal_pixels = calibration["pixel_scores"][calibration["pixel_labels"] == 0]
    image_threshold, validation_image_f1 = _f1_threshold(
        calibration["image_scores"], calibration["image_labels"])
    normal_only_p995_threshold = float(np.quantile(normal_max, 0.995)) if len(normal_max) else 0.5
    pixel_threshold = float(np.quantile(normal_pixels, 0.999)) if len(normal_pixels) else 0.5
    test = evaluate(model, test_loader, DEVICE, bce, collect=True)
    test_images = test["image_scores"]
    test_labels = test["image_labels"]
    image_pred = test_images >= image_threshold
    sensitivity = float((image_pred & (test_labels == 1)).sum() / max(1, (test_labels == 1).sum()))
    specificity = float(((~image_pred) & (test_labels == 0)).sum() / max(1, (test_labels == 0).sum()))
    pixels_y = test["pixel_labels"]
    pixels_p = test["pixel_scores"]
    pixel_auc = float(roc_auc_score(pixels_y, pixels_p)) if len(np.unique(pixels_y)) == 2 else None
    pixel_pred = pixels_p >= pixel_threshold
    intersection = int((pixel_pred & (pixels_y == 1)).sum())
    union = int((pixel_pred | (pixels_y == 1)).sum())
    pixel_iou = float(intersection / max(1, union))
    source_metrics = _source_test_metrics(model, test_rows, DEVICE, bce, args.batch_size,
                                          image_threshold, pixel_threshold)

    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint = {"state_dict": best_state, "architecture": "ResNet18 U-Net binary defect segmentation",
                  "input_size": IMAGE_SIZE, "mean": MEAN, "std": STD,
                  "weak_bottle_box_loss_weight": 0.25,
                  "image_score_threshold": image_threshold, "pixel_threshold": pixel_threshold,
                  "normal_only_p995_threshold": normal_only_p995_threshold,
                  "threshold_selection": "max F1 on held-out validation images containing normal and masked-defect examples",
                  "validation_image_auc": calibration["image_auc"], "best_epoch": best_epoch}
    torch.save(checkpoint, OUT / "defect_segmenter.pt")
    metrics = {
        "architecture": checkpoint["architecture"], "best_epoch": best_epoch,
        "split_counts": {split: {"images": len(rows), "normal": sum(not r["anomalous"] for r in rows),
                                  "defective": sum(bool(r["anomalous"]) for r in rows)}
                         for split, rows in (("train", train_rows), ("validation", val_rows), ("test", test_rows))},
        "calibration": {"image_score_threshold": image_threshold,
                        "validation_image_f1_at_threshold": validation_image_f1,
                        "normal_only_p995_threshold": normal_only_p995_threshold,
                        "pixel_threshold": pixel_threshold,
                        "normal_validation_images": int((calibration["image_labels"] == 0).sum()),
                        "defect_validation_images": int((calibration["image_labels"] == 1).sum()),
                        "method": checkpoint["threshold_selection"] + "; pixel mask threshold is 99.9th percentile of validation-normal pixel scores"},
        "held_out_test": {"images": len(test_rows), "image_roc_auc": test["image_auc"],
                          "sensitivity_at_calibrated_threshold": sensitivity,
                          "specificity_at_calibrated_threshold": specificity,
                          "mixed_pixel_or_box_region_auc_at_56x56": pixel_auc,
                          "mixed_region_iou_at_calibrated_threshold": pixel_iou,
                          "cable_MVTec_AD": _cable_test_metrics(
                              model, test_rows, DEVICE, bce, args.batch_size,
                              image_threshold, pixel_threshold),
                          "by_annotation_source": source_metrics},
        "limitations": "The U-Net localizes binary candidate regions. Shampoo-bottle bounding boxes are weak rectangular targets, downweighted to 0.25 of the per-image loss and evaluated separately from true pixel masks. The attached type and bottle heads have narrower label scopes; no head certifies general severity or sale suitability.",
    }
    report_path = ROOT / "reports" / "defect_segmentation_metrics.json"
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    aggregate_path = ROOT / "reports" / "training_metrics.json"
    if aggregate_path.exists():
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        aggregate["defect_segmentation"] = metrics
        aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
