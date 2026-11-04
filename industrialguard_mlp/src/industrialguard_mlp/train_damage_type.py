"""Train a defect-region type head from MVTec AD's annotated fault folders."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import models

from .train import (DATA, DEVICE, IMAGE_SIZE, MEAN, OUT, ROOT, STD, build_transform)
from .train_defect_segmenter import discover_segmentation_rows


SEED = 73


def canonical_defect(folder: str) -> str:
    """Map MVTec's category-specific fault-folder names into readable types."""
    name = re.sub(r"[^a-z0-9]+", "_", folder.lower()).strip("_")
    if "scratch" in name:
        return "scratch"
    if "crack" in name:
        return "crack"
    if any(token in name for token in ("hole", "poke", "puncture")):
        return "hole_or_puncture"
    if any(token in name for token in ("cut", "tear", "split")):
        return "cut_or_tear"
    if any(token in name for token in ("broken", "damaged", "defective")):
        return "broken_or_damaged"
    if any(token in name for token in ("missing", "misplaced", "swap", "flip")):
        return "missing_or_misplaced_part"
    if any(token in name for token in ("bent", "fold", "squeeze", "manipulated")):
        return "deformation"
    if name == "combined":
        return "multiple_or_other"
    return "surface_or_contamination"


class DefectRegionDataset(Dataset):
    def __init__(self, rows, transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image_path, mask_path = Path(row["image"]), Path(row["mask"])
        try:
            image = Image.open(image_path).convert("RGB")
            mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        except (OSError, UnidentifiedImageError) as exc:
            raise RuntimeError(f"Cannot read defect image or mask for {image_path}: {exc}") from exc
        yy, xx = np.where(mask)
        if not len(xx):
            raise ValueError(f"The annotated mask is empty: {mask_path}")
        width, height = image.size
        x0, x1 = int(xx.min()), int(xx.max()) + 1
        y0, y1 = int(yy.min()), int(yy.max()) + 1
        pad_x, pad_y = max(4, int((x1 - x0) * 0.30)), max(4, int((y1 - y0) * 0.30))
        crop = image.crop((max(0, x0 - pad_x), max(0, y0 - pad_y),
                           min(width, x1 + pad_x), min(height, y1 + pad_y)))
        defect_folder = mask_path.parent.name
        return self.transform(crop), defect_folder


class DamageTypeMLP(nn.Module):
    def __init__(self, input_dim: int, classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.30),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.20),
            nn.Linear(128, classes),
        )

    def forward(self, x):
        return self.net(x)


def extract(rows, encoder, transform, batch_size=64):
    loader = DataLoader(DefectRegionDataset(rows, transform), batch_size=batch_size,
                        shuffle=False, num_workers=0, pin_memory=DEVICE.type == "cuda")
    feature_rows, folder_names = [], []
    with torch.inference_mode():
        for images, folders in loader:
            images = images.to(DEVICE, non_blocking=True)
            feature_rows.append(encoder(images).flatten(1).float().cpu())
            folder_names.extend(folders)
    return torch.cat(feature_rows).numpy(), folder_names


def thresholds_at_precision(y_true, probabilities, target_precision=0.80, min_samples=5):
    predicted = probabilities.argmax(1)
    limits = []
    for class_index in range(probabilities.shape[1]):
        scores = probabilities[predicted == class_index, class_index]
        selected = 1.01
        for threshold in sorted(set(float(value) for value in scores)):
            accepted = (predicted == class_index) & (probabilities[:, class_index] >= threshold)
            if int(accepted.sum()) < min_samples:
                continue
            if float((y_true[accepted] == class_index).mean()) >= target_precision:
                selected = threshold
                break
        limits.append(selected)
    return limits


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train MVTec defect type clues for localized regions")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    rows = [row for row in discover_segmentation_rows()
            if row["dataset"] == "MVTec AD" and row["anomalous"] and row.get("mask")]
    for row in rows:
        row["defect_type"] = canonical_defect(Path(row["mask"]).parent.name)
    labels = sorted({row["defect_type"] for row in rows})
    to_index = {label: index for index, label in enumerate(labels)}
    splits = {split: [row for row in rows if row["split"] == split]
              for split in ("train", "validation", "test")}
    if any(not splits[split] for split in splits):
        raise RuntimeError("MVTec mask-crop train, validation and test splits must all be nonempty.")
    counts = Counter(row["defect_type"] for row in splits["train"])
    missing = [label for label in labels if counts[label] == 0]
    if missing:
        raise RuntimeError(f"Training split has no examples for: {missing}")

    encoder_bundle = torch.load(OUT / "resnet18_encoder.pt", map_location="cpu", weights_only=True)
    encoder = nn.Sequential(*list(models.resnet18(weights=None).children())[:-1])
    encoder.load_state_dict(encoder_bundle["state_dict"])
    encoder.to(DEVICE).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    transform = build_transform()
    extracted = {split: extract(split_rows, encoder, transform, args.batch_size)
                 for split, split_rows in splits.items()}
    features = {split: extracted[split][0] for split in splits}
    targets = {split: np.asarray([to_index[canonical_defect(folder)]
                                  for folder in extracted[split][1]], dtype=np.int64)
               for split in splits}
    weights = np.clip(len(targets["train"]) / (len(labels) * np.bincount(targets["train"], minlength=len(labels))),
                      0.5, 4.0)
    class_weights = torch.as_tensor(weights, dtype=torch.float32, device=DEVICE)
    model = DamageTypeMLP(features["train"].shape[1], len(labels)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    train_data = TensorDataset(torch.as_tensor(features["train"], dtype=torch.float32),
                               torch.as_tensor(targets["train"], dtype=torch.long))
    loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_x = torch.as_tensor(features["validation"], dtype=torch.float32, device=DEVICE)
    val_y = torch.as_tensor(targets["validation"], dtype=torch.long, device=DEVICE)
    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.inference_mode():
            val_logits = model(val_x)
            val_loss = float(criterion(val_logits, val_y).item())
        print(f"epoch {epoch + 1:02d}/{args.epochs} train_loss={np.mean(losses):.4f} "
              f"val_loss={val_loss:.4f}", flush=True)
        if val_loss < best_loss - 1e-5:
            best_loss, best_epoch, stale = val_loss, epoch + 1, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 8:
                break
    if best_state is None:
        raise RuntimeError("Damage type head training produced no checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        val_prob = torch.softmax(model(val_x), 1).cpu().numpy()
        test_x = torch.as_tensor(features["test"], dtype=torch.float32, device=DEVICE)
        test_prob = torch.softmax(model(test_x), 1).cpu().numpy()
    limits = thresholds_at_precision(targets["validation"], val_prob)
    test_pred = test_prob.argmax(1)
    confidence = test_prob[np.arange(len(test_pred)), test_pred]
    accepted = confidence >= np.asarray(limits)[test_pred]
    report = classification_report(targets["test"], test_pred, labels=np.arange(len(labels)),
                                   target_names=labels, output_dict=True, zero_division=0)
    metrics = {
        "architecture": "DefectTypeMLP on defect-mask-bounded crops using the shared frozen ResNet-18 encoder",
        "source": "MVTec AD test defect folders and pixel masks; custom stratified-by-folder image splits",
        "label_mapping": "scratch, crack, hole/puncture, cut/tear, broken/damaged, missing/misplaced part, deformation, surface/contamination, or multiple/other",
        "counts": {split: {"instances": len(splits[split]),
                           "per_class": dict(Counter(row["defect_type"] for row in splits[split]))}
                   for split in splits},
        "classes": labels, "best_epoch": best_epoch, "validation_loss": best_loss,
        "validation_precision_target": 0.80, "minimum_validation_predictions_per_class": 5,
        "test_accuracy": float(accuracy_score(targets["test"], test_pred)),
        "test_macro_f1": float(f1_score(targets["test"], test_pred, labels=np.arange(len(labels)),
                                         average="macro", zero_division=0)),
        "selective_coverage": float(accepted.mean()),
        "selective_accuracy": float(accuracy_score(targets["test"][accepted], test_pred[accepted])) if accepted.any() else None,
        "per_class": report,
        "thresholds": dict(zip(labels, limits)),
        "confusion_matrix": confusion_matrix(targets["test"], test_pred, labels=np.arange(len(labels))).tolist(),
        "limitation": "Broad types are derived from MVTec folder names, not independently adjudicated severity labels. The classifier is trained only on MVTec defect crops; its clues are not calibrated for unrelated products or camera styles and do not determine sale suitability.",
    }
    checkpoint = {"state_dict": best_state, "classes": labels, "thresholds": limits,
                  "input_dim": int(features["train"].shape[1]), "best_epoch": best_epoch,
                  "preprocessing": {"resize": 256, "center_crop": IMAGE_SIZE, "mean": MEAN, "std": STD},
                  "source": metrics["source"]}
    torch.save(checkpoint, OUT / "damage_type_mlp.pt")
    (ROOT / "reports" / "damage_type_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    aggregate_path = ROOT / "reports" / "training_metrics.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8")) if aggregate_path.exists() else {}
    aggregate["damage_type_head"] = metrics
    aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
