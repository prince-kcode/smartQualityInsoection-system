"""Train a supervised plastic-bottle defect detector on the academic PBD dataset."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data/raw/external/pbd/PBD_Dataset"
DEFAULT_RESULTS = PROJECT_ROOT / "results/retrained/pbd_bottle_detector"
CLASS_NAMES = [
    "background",
    "body_stain",
    "body_concave",
    "body_score",
    "body_squeeze",
    "shoulder_pipe",
    "shoulder_no_burr",
    "shoulder_stain",
]
CLASS_IDS = {
    "body": {"stain": 1, "concave": 2, "score": 3, "squeeze": 4},
    "shoulder": {"pipe": 5, "no_burr": 6, "stain": 7},
}


class PbdDataset(Dataset):
    """Combine PBD body and shoulder COCO annotations into one class-index space."""

    def __init__(self, root: Path, split: str) -> None:
        self.root = Path(root)
        self.items: list[tuple[Path, list[list[float]], list[int]]] = []
        for part, class_map in CLASS_IDS.items():
            base = self.root / part
            annotations = json.loads(
                (base / "annotations" / f"{split}.json").read_text(encoding="utf-8")
            )
            category_map = {
                category["id"]: class_map[category["name"]]
                for category in annotations["categories"]
            }
            by_image: dict[int, list[dict[str, Any]]] = {}
            for annotation in annotations["annotations"]:
                by_image.setdefault(annotation["image_id"], []).append(annotation)
            for image in annotations["images"]:
                file_path = base / f"{split}_images" / image["file_name"]
                boxes: list[list[float]] = []
                labels: list[int] = []
                for annotation in by_image.get(image["id"], []):
                    x, y, width, height = annotation["bbox"]
                    if width <= 0 or height <= 0:
                        continue
                    boxes.append([x, y, x + width, y + height])
                    labels.append(category_map[annotation["category_id"]])
                if not file_path.is_file():
                    raise FileNotFoundError(file_path)
                self.items.append((file_path, boxes, labels))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        path, boxes, labels = self.items[index]
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            width, height = image.size
            array = np.asarray(image, dtype=np.uint8).copy()
        tensor = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": torch.as_tensor(
                [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes],
                dtype=torch.float32,
            ),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
            "size": torch.tensor([height, width], dtype=torch.int64),
        }
        return tensor, target


def collate_fn(batch):
    return tuple(zip(*batch, strict=True))


def box_iou(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.float32)
    left_top = torch.maximum(box[:2], boxes[:, :2])
    right_bottom = torch.minimum(box[2:], boxes[:, 2:])
    intersection = (right_bottom - left_top).clamp(min=0).prod(dim=1)
    box_area = (box[2:] - box[:2]).clamp(min=0).prod()
    areas = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0).prod(dim=1)
    return intersection / (box_area + areas - intersection).clamp(min=1e-8)


def mean_ap50(model, loader, device: torch.device) -> dict[str, Any]:
    detections: dict[int, list[tuple[float, bool]]] = {i: [] for i in range(1, len(CLASS_NAMES))}
    ground_truth_counts = {i: 0 for i in range(1, len(CLASS_NAMES))}
    model.eval()
    with torch.inference_mode():
        for images, targets in loader:
            outputs = model([image.to(device) for image in images])
            for output, target in zip(outputs, targets, strict=True):
                gt_boxes = target["boxes"]
                gt_labels = target["labels"]
                used: dict[int, set[int]] = {i: set() for i in range(1, len(CLASS_NAMES))}
                for class_id in ground_truth_counts:
                    ground_truth_counts[class_id] += int((gt_labels == class_id).sum())
                pred_boxes = output["boxes"].detach().cpu()
                pred_scores = output["scores"].detach().cpu()
                pred_labels = output["labels"].detach().cpu()
                for class_id in ground_truth_counts:
                    indices = torch.where((pred_labels == class_id) & (pred_scores >= 0.05))[0]
                    indices = indices[torch.argsort(pred_scores[indices], descending=True)]
                    class_gt = torch.where(gt_labels == class_id)[0]
                    boxes_for_class = gt_boxes[class_gt]
                    for pred_index in indices.tolist():
                        score = float(pred_scores[pred_index])
                        overlaps = box_iou(pred_boxes[pred_index], boxes_for_class)
                        candidates = torch.argsort(overlaps, descending=True)
                        match = next(
                            (int(candidate) for candidate in candidates
                             if float(overlaps[candidate]) >= 0.5 and int(candidate) not in used[class_id]),
                            None,
                        )
                        if match is None:
                            detections[class_id].append((score, False))
                        else:
                            used[class_id].add(match)
                            detections[class_id].append((score, True))

    per_class: dict[str, Any] = {}
    aps: list[float] = []
    for class_id, name in enumerate(CLASS_NAMES[1:], start=1):
        ranked = sorted(detections[class_id], key=lambda item: item[0], reverse=True)
        tp = np.asarray([int(is_true) for _, is_true in ranked], dtype=np.float64)
        fp = 1.0 - tp
        cumulative_tp = np.cumsum(tp)
        cumulative_fp = np.cumsum(fp)
        positives = ground_truth_counts[class_id]
        if positives and len(ranked):
            recall = cumulative_tp / positives
            precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-12)
            ap = float(np.mean([
                np.max(precision[recall >= threshold], initial=0.0)
                for threshold in np.linspace(0, 1, 101)
            ]))
            final_recall = float(recall[-1])
            final_precision = float(precision[-1])
            aps.append(ap)
        else:
            ap = None
            final_recall = 0.0
            final_precision = 0.0
        per_class[name] = {
            "ground_truth_boxes": positives,
            "detections_at_score_0_05": len(ranked),
            "ap50": ap,
            "precision_all_ranked": final_precision,
            "recall_all_ranked": final_recall,
        }
    return {"map50": float(np.mean(aps)) if aps else 0.0, "per_class": per_class}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-images", type=int, default=0, help="0 means use every training image")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise SystemExit("--epochs and --batch-size must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    train_dataset = PbdDataset(args.dataset_root, "train")
    val_dataset = PbdDataset(args.dataset_root, "val")
    if args.max_train_images:
        selected = random.Random(args.seed).sample(range(len(train_dataset)), min(args.max_train_images, len(train_dataset)))
        train_dataset.items = [train_dataset.items[index] for index in sorted(selected)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
        collate_fn=collate_fn, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=collate_fn, pin_memory=device.type == "cuda",
    )

    from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=weights,
        min_size=640,
        max_size=800,
        box_detections_per_img=100,
    )
    features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(features, len(CLASS_NAMES))
    model.to(device)
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.0025, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.epochs - 1), gamma=0.2)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.results_dir / "pbd_bottle_defect_detector.pt"
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    print(
        f"Training on {device}; train images={len(train_dataset)}, validation images={len(val_dataset)}, "
        f"defect labels={len(CLASS_NAMES) - 1}, epochs={args.epochs}",
        flush=True,
    )
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        image_count = 0
        for images, targets in train_loader:
            batch_images = [image.to(device, non_blocking=True) for image in images]
            batch_targets = [
                {key: value.to(device, non_blocking=True) for key, value in target.items() if key != "size"}
                for target in targets
            ]
            loss_dict = model(batch_images, batch_targets)
            loss = sum(loss_dict.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite detector loss during epoch {epoch + 1}: {loss_dict}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(images)
            image_count += len(images)
        scheduler.step()
        epoch_loss = loss_sum / max(1, image_count)
        history.append({"epoch": epoch + 1, "train_loss": epoch_loss})
        torch.save({
            "state_dict": model.state_dict(),
            "class_names": CLASS_NAMES,
            "min_size": 640,
            "max_size": 800,
            "seed": args.seed,
            "epoch": epoch + 1,
        }, checkpoint)
        print(f"epoch {epoch + 1}/{args.epochs}: train_loss={epoch_loss:.5f}", flush=True)

    model.eval()
    metrics = mean_ap50(model, val_loader, device)
    duration = time.perf_counter() - started
    metrics.update({
        "model": "Faster R-CNN MobileNetV3 FPN",
        "dataset": "PBD plastic bottle body and shoulder",
        "train_images": len(train_dataset),
        "validation_images": len(val_dataset),
        "epochs": args.epochs,
        "seed": args.seed,
        "device": str(device),
        "training_seconds": duration,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "evaluation": "validation split, AP@IoU=0.50; not an independent test set",
    })
    (args.results_dir / "pbd_bottle_evaluation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (args.results_dir / "pbd_bottle_training_curve.csv").open("w", encoding="utf-8") as file:
        file.write("epoch,train_loss\n")
        for row in history:
            file.write(f"{row['epoch']},{row['train_loss']:.8f}\n")
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Checkpoint: {checkpoint.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
