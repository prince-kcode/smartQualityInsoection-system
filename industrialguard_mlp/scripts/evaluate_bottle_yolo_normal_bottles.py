"""Probe bottle-YOLO false alarms on the separate MVTec normal-bottle set."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models" / "bottle_defect_yolo.pt"
NORMAL_DIR = ROOT / "datasets" / "mvtec_ad" / "bottle" / "test" / "good"
REPORT = ROOT / "reports" / "bottle_yolo_normal_bottle_probe.json"


def main():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Train the bottle YOLO model first: {CHECKPOINT}")
    images = sorted(path for path in NORMAL_DIR.glob("*")
                    if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not images:
        raise FileNotFoundError(f"No held-out MVTec normal bottle images found at {NORMAL_DIR}")

    model = YOLO(str(CHECKPOINT))
    results = model.predict(
        source=[str(path) for path in images], imgsz=640, conf=0.25, iou=0.6,
        device=0 if torch.cuda.is_available() else "cpu", verbose=False,
    )
    image_rows = []
    for path, result in zip(images, results):
        boxes = result.boxes
        scores = boxes.conf.detach().cpu().tolist() if boxes is not None and len(boxes) else []
        labels = boxes.cls.detach().cpu().to(torch.int64).tolist() if boxes is not None and len(boxes) else []
        image_rows.append({
            "image": str(path.relative_to(ROOT)),
            "candidate_count": len(scores),
            "top_score": max(scores) if scores else None,
            "candidate_labels": [str(result.names[int(label)]) for label in labels],
        })
    positives = sum(row["candidate_count"] > 0 for row in image_rows)
    report = {
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "probe_dataset": "MVTec AD bottle/test/good; separate normal bottle photos, different domain from shampoo dataset",
        "confidence_threshold": 0.25,
        "images_tested": len(image_rows),
        "images_with_candidates": positives,
        "candidate_image_rate": positives / len(image_rows),
        "candidate_boxes_total": sum(row["candidate_count"] for row in image_rows),
        "interpretation": "Small exploratory cross-domain negative probe only; not a calibrated clean-shampoo-bottle false-positive rate.",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "images": image_rows,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
