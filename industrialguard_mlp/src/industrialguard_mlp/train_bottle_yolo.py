"""Train and evaluate a bottle-only YOLO defect detector from official YOLO labels.

This model is intentionally limited to the plastic shampoo-bottle dataset. It
does not replace the general industrial mask segmenter for other products.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

from .train import DATA, ROOT


DATASET_ROOT = DATA / "additional" / "Plastic Shampoo Bottle Dataset"
MODEL_DIR = ROOT / "models"
RUN_DIR = MODEL_DIR / "bottle_yolo_runs"
BEST_CHECKPOINT = MODEL_DIR / "bottle_defect_yolo.pt"
PRETRAINED_CHECKPOINT = MODEL_DIR / "pretrained" / "yolo11n.pt"
REPORT_PATH = ROOT / "reports" / "bottle_yolo_metrics.json"


def read_classes() -> list[str]:
    source = DATASET_ROOT / "data.yaml"
    if not source.is_file():
        raise FileNotFoundError(f"Bottle YOLO dataset configuration is missing: {source}")
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("names:"):
            names = ast.literal_eval(line.split(":", 1)[1].strip())
            if not isinstance(names, list) or not names:
                break
            return [str(name) for name in names]
    raise ValueError(f"No valid class list found in {source}")


def _write_portable_config(classes: list[str]) -> Path:
    # Use an absolute root so Ultralytics cannot prepend its global datasets dir.
    config = {
        "path": str(DATASET_ROOT.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": classes,
    }
    output = ROOT / "reports" / "bottle_yolo_dataset.yaml"
    output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    for split in ("train", "valid", "test"):
        folder = DATASET_ROOT / split / "images"
        if not folder.is_dir() or not any(folder.glob("*")):
            raise FileNotFoundError(f"Bottle YOLO {split} images are missing from {folder}")
    return output


def train(epochs: int = 30, imgsz: int = 640, batch: int = 16, workers: int = 4) -> dict:
    classes = read_classes()
    config = _write_portable_config(classes)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    pretrained = PRETRAINED_CHECKPOINT if PRETRAINED_CHECKPOINT.is_file() else "yolo11n.pt"
    model = YOLO(str(pretrained))
    if isinstance(pretrained, str):
        source_weights = ROOT / "yolo11n.pt"
        if source_weights.is_file():
            PRETRAINED_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_weights, PRETRAINED_CHECKPOINT)
    train_result = model.train(
        data=str(config),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=0 if torch.cuda.is_available() else "cpu",
        workers=workers,
        patience=8,
        project=str(RUN_DIR),
        name="plastic_shampoo_bottle",
        exist_ok=True,
        seed=73,
        deterministic=True,
        cache=False,
        amp=True,
        plots=True,
        verbose=True,
    )
    best = Path(model.trainer.best)
    if not best.is_file():
        raise FileNotFoundError(f"Ultralytics did not save a best checkpoint: {best}")
    shutil.copy2(best, BEST_CHECKPOINT)

    # The test split is evaluated once after model selection on validation.
    evaluated = YOLO(str(BEST_CHECKPOINT))
    test_metrics = evaluated.val(
        data=str(config), split="test", imgsz=imgsz, batch=batch,
        device=0 if torch.cuda.is_available() else "cpu", plots=True,
        project=str(RUN_DIR), name="official_test", exist_ok=True, verbose=True,
    )
    box = test_metrics.box
    history_path = RUN_DIR / "plastic_shampoo_bottle" / "results.csv"
    history = list(csv.DictReader(history_path.open("r", newline="", encoding="utf-8"))) if history_path.exists() else []
    best_row = max(history, key=lambda row: float(row.get("metrics/mAP50-95(B)", 0.0))) if history else {}
    # Ultralytics versions differ in which per-class support attributes they
    # expose. Read the source YOLO labels directly for a stable report.
    label_counts = [0] * len(classes)
    for label_file in sorted((DATASET_ROOT / "test" / "labels").glob("*.txt")):
        for line in label_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                class_id = int(line.split()[0])
                if 0 <= class_id < len(label_counts):
                    label_counts[class_id] += 1
    per_class = {}
    for index, label in enumerate(classes):
        per_class[label] = {
            "map50_95": float(box.maps[index]) if index < len(box.maps) else None,
            "test_instances": label_counts[index],
        }
    result = {
        "architecture": "Ultralytics YOLO11n object detector",
        "scope": "Plastic Shampoo Bottle Dataset only; not a general-purpose defect detector",
        "classes": classes,
        "dataset": {
            "train_images": len(list((DATASET_ROOT / "train" / "images").glob("*"))),
            "validation_images": len(list((DATASET_ROOT / "valid" / "images").glob("*"))),
            "held_out_test_images": len(list((DATASET_ROOT / "test" / "images").glob("*"))),
            "test_split": "official test split; not used for training or checkpoint selection",
        },
        "training": {
            "pretrained_weights": "yolo11n.pt",
            "pretrained_sha256": hashlib.sha256(PRETRAINED_CHECKPOINT.read_bytes()).hexdigest()
            if PRETRAINED_CHECKPOINT.is_file() else None,
            "epochs_requested": epochs,
            "epochs_completed": len(history),
            "best_validation_epoch": int(float(best_row.get("epoch", 0))) if best_row else None,
            "image_size": imgsz,
            "batch_size": batch,
            "device": "cuda:0" if torch.cuda.is_available() else "cpu",
            "best_checkpoint": str(BEST_CHECKPOINT.relative_to(ROOT)),
            "ultralytics_version": __import__("ultralytics").__version__,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        },
        "test_metrics": {
            "precision": float(box.mp),
            "recall": float(box.mr),
            "map50": float(box.map50),
            "map50_95": float(box.map),
            "per_class": per_class,
        },
        "limitation": (
            "Annotations cover five defect types and marginal/unacceptable grades on plastic shampoo bottles. "
            "The model must not be applied as a calibrated defect or sale-grade model to other products. "
            "It has no clean-bottle examples and cannot certify defect absence or sale suitability."
        ),
    }
    REPORT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    aggregate_path = ROOT / "reports" / "training_metrics.json"
    if aggregate_path.exists():
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        aggregate["bottle_yolo_detector"] = result
        aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    train(args.epochs, args.imgsz, args.batch, args.workers)
