"""Train the bottle-only defect type/quality-grade head from YOLO labels."""
from __future__ import annotations

import ast
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .train import DATA, DEVICE, IMAGE_SIZE, MEAN, OUT, ROOT, STD, build_transform


SEED = 73
PV_ROOT = DATA / "additional" / "Plastic Shampoo Bottle Dataset"


def read_classes() -> list[str]:
    config = PV_ROOT / "data.yaml"
    if not config.is_file():
        raise FileNotFoundError(f"Bottle dataset config missing: {config}")
    for line in config.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("names:"):
            return ast.literal_eval(line.split(":", 1)[1].strip())
    raise ValueError("No class list found in the bottle dataset data.yaml")


class BottleDefectDataset(Dataset):
    def __init__(self, split: str, classes: list[str]):
        self.image_dir = PV_ROOT / split / "images"
        self.label_dir = PV_ROOT / split / "labels"
        self.images = sorted(p for p in self.image_dir.glob("*")
                             if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                             and ".ipynb_checkpoints" not in p.parts)
        self.classes = classes
        self.transform = build_transform()
        if not self.images:
            raise FileNotFoundError(f"No images found in {self.image_dir}")
        self.instances = []
        for image_path in self.images:
            label_path = self.label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(f"YOLO annotation missing for {image_path.name}")
            for line in label_path.read_text(encoding="utf-8").splitlines():
                values = line.split()
                if not values:
                    continue
                label = int(values[0])
                if not 0 <= label < len(self.classes) or len(values) < 5:
                    raise ValueError(f"Invalid YOLO label in {label_path}: {line}")
                box = tuple(float(value) for value in values[1:5])
                self.instances.append((image_path, label, box))
        if not self.instances:
            raise ValueError(f"No defect boxes found in {self.label_dir}")

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        path, label, (center_x, center_y, box_width, box_height) = self.instances[index]
        try:
            image = Image.open(path).convert("RGB")
        except (OSError, UnidentifiedImageError) as exc:
            raise RuntimeError(f"Cannot read {path}: {exc}") from exc
        width, height = image.size
        x0, y0 = (center_x - box_width / 2) * width, (center_y - box_height / 2) * height
        x1, y1 = (center_x + box_width / 2) * width, (center_y + box_height / 2) * height
        pad_x, pad_y = max(4, (x1 - x0) * 0.25), max(4, (y1 - y0) * 0.25)
        crop = image.crop((max(0, int(x0 - pad_x)), max(0, int(y0 - pad_y)),
                           min(width, int(x1 + pad_x)), min(height, int(y1 + pad_y))))
        return self.transform(crop), label, str(path)


class BottleGradeMLP(nn.Module):
    def __init__(self, input_dim: int, classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.30),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.20),
            nn.Linear(128, classes),
        )

    def forward(self, x):
        return self.net(x)


def extract_embeddings(rows: Dataset, encoder, batch_size: int = 64):
    loader = DataLoader(rows, batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=DEVICE.type == "cuda")
    features, labels = [], []
    with torch.inference_mode():
        for images, target, _ in loader:
            images = images.to(DEVICE, non_blocking=True)
            features.append(encoder(images).flatten(1).float().cpu())
            labels.append(target.long())
    return torch.cat(features).numpy(), torch.cat(labels).numpy()


def _precision_thresholds(y: np.ndarray, probabilities: np.ndarray, target_precision=0.90,
                          min_samples=10) -> list[float]:
    predicted = probabilities.argmax(axis=1)
    thresholds = []
    for class_id in range(probabilities.shape[1]):
        class_scores = probabilities[predicted == class_id, class_id]
        candidates = sorted(set(float(x) for x in class_scores))
        selected = 1.01
        for threshold in candidates:
            accepted = (predicted == class_id) & (probabilities[:, class_id] >= threshold)
            if int(accepted.sum()) < min_samples:
                continue
            precision = float((y[accepted] == class_id).mean())
            if precision >= target_precision:
                selected = threshold
                break
        thresholds.append(float(selected))
    return thresholds


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train the shampoo-bottle defect type/grade head")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    classes = read_classes()
    datasets = {split: BottleDefectDataset(split, classes)
                for split in ("train", "valid", "test")}
    encoder_bundle = torch.load(OUT / "resnet18_encoder.pt", map_location="cpu", weights_only=True)
    from torchvision import models

    encoder = nn.Sequential(*list(models.resnet18(weights=None).children())[:-1])
    encoder.load_state_dict(encoder_bundle["state_dict"])
    encoder.to(DEVICE).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    data = {split: extract_embeddings(dataset, encoder, args.batch_size)
            for split, dataset in datasets.items()}
    x_train, y_train = data["train"]
    x_val, y_val = data["valid"]
    x_test, y_test = data["test"]
    counts = np.bincount(y_train, minlength=len(classes))
    if np.any(counts == 0):
        raise RuntimeError("The training split is missing at least one bottle defect/grade label.")
    class_weights = torch.as_tensor(np.clip(len(y_train) / (len(classes) * counts), 0.5, 4.0),
                                    dtype=torch.float32, device=DEVICE)

    model = BottleGradeMLP(x_train.shape[1], len(classes)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
    train_data = TensorDataset(torch.as_tensor(x_train, dtype=torch.float32),
                               torch.as_tensor(y_train, dtype=torch.long))
    loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_x = torch.as_tensor(x_val, dtype=torch.float32, device=DEVICE)
    val_y = torch.as_tensor(y_val, dtype=torch.long, device=DEVICE)
    best_loss, best_epoch, best_state, stale = float("inf"), 0, None, 0

    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        for bx, by in loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.CrossEntropyLoss(weight=class_weights)(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.inference_mode():
            val_logits = model(val_x)
            val_loss = float(nn.CrossEntropyLoss(weight=class_weights)(val_logits, val_y).item())
        print(f"epoch {epoch + 1:02d}/{args.epochs} train_loss={np.mean(epoch_losses):.4f} "
              f"val_loss={val_loss:.4f}", flush=True)
        if val_loss < best_loss - 1e-5:
            best_loss, best_epoch, stale = val_loss, epoch + 1, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 8:
                break

    if best_state is None:
        raise RuntimeError("Bottle defect head training produced no checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        val_probabilities = torch.softmax(model(val_x), dim=1).cpu().numpy()
        test_probabilities = torch.softmax(model(torch.as_tensor(x_test, dtype=torch.float32,
                                                                   device=DEVICE)), dim=1).cpu().numpy()
    thresholds = _precision_thresholds(y_val, val_probabilities)
    prediction = test_probabilities.argmax(axis=1)
    predicted_confidence = test_probabilities[np.arange(len(prediction)), prediction]
    accepted = predicted_confidence >= np.asarray(thresholds)[prediction]
    per_class = {}
    for i, name in enumerate(classes):
        y = y_test == i
        p = prediction == i
        class_accept = accepted & p
        per_class[name] = {
            "test_instances": int(y.sum()), "accepted_threshold": thresholds[i],
            "accepted_predictions": int(class_accept.sum()),
            "selective_precision": float((y_test[class_accept] == i).mean()) if class_accept.any() else None,
            "precision": float(precision_score(y, p, zero_division=0)),
            "recall": float(recall_score(y, p, zero_division=0)),
            "f1": float(f1_score(y, p, zero_division=0)),
        }
    selective_accuracy = float(accuracy_score(y_test[accepted], prediction[accepted])) if accepted.any() else None
    metrics = {
        "architecture": "multi-class defect/grade MLP over defect-region crops and the shared frozen ResNet-18 feature encoder",
        "source": "Plastic Shampoo Bottle Dataset, each YOLO bounding-box crop inherits its defect type and grade",
        "train_instances": len(x_train), "validation_instances": len(x_val), "test_instances": len(x_test),
        "source_images": {split: len(dataset.images) for split, dataset in datasets.items()},
        "instances_per_class_train": dict(zip(classes, counts.astype(int).tolist())),
        "classes": classes, "best_epoch": best_epoch, "validation_loss": best_loss,
        "test_accuracy": float(accuracy_score(y_test, prediction)),
        "test_macro_f1": float(f1_score(y_test, prediction, labels=np.arange(len(classes)), average="macro", zero_division=0)),
        "selective_coverage_at_validation_precision_target": float(accepted.mean()),
        "selective_accuracy": selective_accuracy,
        "validation_precision_target": 0.90,
        "per_class": per_class,
        "thresholds": dict(zip(classes, thresholds)),
        "confusion_matrix": confusion_matrix(y_test, prediction, labels=np.arange(len(classes))).tolist(),
        "limitation": "Type and marginal/unacceptable grades apply only to crops resembling this plastic shampoo-bottle dataset. The source has no clean bottle examples; this head classifies already-localized candidate regions and cannot determine defect absence, safety, or sale suitability.",
    }
    checkpoint = {
        "state_dict": best_state, "classes": classes, "thresholds": thresholds,
        "best_epoch": best_epoch, "input_dim": int(x_train.shape[1]),
        "preprocessing": {"resize": 256, "center_crop": IMAGE_SIZE, "mean": MEAN, "std": STD},
        "source": metrics["source"],
    }
    torch.save(checkpoint, OUT / "bottle_damage_mlp.pt")
    report_path = ROOT / "reports" / "bottle_damage_metrics.json"
    report_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    aggregate_path = ROOT / "reports" / "training_metrics.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8")) if aggregate_path.exists() else {}
    aggregate["bottle_damage_grade_head"] = metrics
    aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
