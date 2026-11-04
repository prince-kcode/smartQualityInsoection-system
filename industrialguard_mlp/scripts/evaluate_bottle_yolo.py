"""Evaluate the saved bottle YOLO checkpoint and write its model card."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.industrialguard_mlp.train_bottle_yolo import (
    BEST_CHECKPOINT, DATASET_ROOT, MODEL_DIR, REPORT_PATH, ROOT, _write_portable_config, read_classes,
)


def main():
    if not BEST_CHECKPOINT.is_file():
        raise FileNotFoundError(f"Bottle YOLO checkpoint missing: {BEST_CHECKPOINT}")
    classes = read_classes()
    config = _write_portable_config(classes)
    model = YOLO(str(BEST_CHECKPOINT))
    result = model.val(
        data=str(config), split="test", imgsz=640, batch=16,
        device=0 if torch.cuda.is_available() else "cpu", plots=True,
        project=str(MODEL_DIR / "bottle_yolo_runs"), name="official_test_report",
        exist_ok=True, verbose=False,
    )
    box = result.box

    label_counts = [0] * len(classes)
    for label_file in sorted((DATASET_ROOT / "test" / "labels").glob("*.txt")):
        for line in label_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                class_id = int(line.split()[0])
                if 0 <= class_id < len(label_counts):
                    label_counts[class_id] += 1
    per_class = {
        label: {
            "map50_95": float(box.maps[index]) if index < len(box.maps) else None,
            "test_instances": label_counts[index],
        }
        for index, label in enumerate(classes)
    }
    training_csv = MODEL_DIR / "bottle_yolo_runs" / "plastic_shampoo_bottle" / "results.csv"
    history = list(csv.DictReader(training_csv.open("r", newline="", encoding="utf-8"))) if training_csv.exists() else []
    best_row = max(history, key=lambda row: float(row.get("metrics/mAP50-95(B)", 0.0))) if history else {}
    pretrained = MODEL_DIR / "pretrained" / "yolo11n.pt"
    pretrained_hash = hashlib.sha256(pretrained.read_bytes()).hexdigest() if pretrained.exists() else None
    report = {
        "architecture": "Ultralytics YOLO11n object detector",
        "scope": "Plastic shampoo-bottle subtype only; generic bottles and other categories do not use this model",
        "classes": classes,
        "dataset": {
            "train_images": 5535,
            "validation_images": 1582,
            "held_out_test_images": 790,
            "test_instances": int(sum(label_counts)),
            "test_split": "official held-out test split; not used for training or model selection",
        },
        "training": {
            "pretrained_weights": "yolo11n.pt",
            "pretrained_sha256": pretrained_hash,
            "epochs_requested": 30,
            "epochs_completed": len(history),
            "best_validation_epoch": int(float(best_row.get("epoch", 0))) if best_row else None,
            "best_validation_map50": float(best_row.get("metrics/mAP50(B)", 0.0)) if best_row else None,
            "best_validation_map50_95": float(best_row.get("metrics/mAP50-95(B)", 0.0)) if best_row else None,
            "image_size": 640,
            "batch_size": 16,
            "device": "cuda:0" if torch.cuda.is_available() else "cpu",
            "checkpoint": str(BEST_CHECKPOINT.relative_to(ROOT)),
            "ultralytics_version": __import__("ultralytics").__version__,
        },
        "test_metrics": {
            "precision": float(box.mp),
            "recall": float(box.mr),
            "map50": float(box.map50),
            "map50_95": float(box.map),
            "per_class": per_class,
        },
        "app_candidate_confidence_threshold": 0.25,
        "limitation": (
            "Annotations cover five defect types and marginal/unacceptable grades on plastic shampoo bottles. "
            "The source contains no clean shampoo-bottle images. Missing detections do not establish that an item is defect-free, "
            "and the detector is not calibrated for other bottles or products."
        ),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    aggregate_path = ROOT / "reports" / "training_metrics.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8")) if aggregate_path.exists() else {}
    aggregate["bottle_yolo_detector"] = report
    aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
