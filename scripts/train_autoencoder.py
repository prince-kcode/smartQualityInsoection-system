"""Train and evaluate a normal-only CNN autoencoder on MVTec AD."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset

from industrialguard.data.mvtec import IMAGE_SUFFIXES, MVTEC_CATEGORIES, audit_category
from industrialguard.models.autoencoder import ConvAutoencoder


IMAGE_SIZE = 256
CATEGORIES = MVTEC_CATEGORIES


class ImageDataset(Dataset):
    """Load RGB images resized to the model input dimensions."""

    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.paths[index]
        with Image.open(path) as opened:
            image = opened.convert("RGB").resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
            )
            array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return tensor, str(path)


def image_files(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def collect_test_items(category_root: Path) -> list[tuple[Path, Path | None, bool]]:
    items: list[tuple[Path, Path | None, bool]] = []
    for path in image_files(category_root / "test" / "good"):
        items.append((path, None, False))
    test_root = category_root / "test"
    gt_root = category_root / "ground_truth"
    for defect_dir in sorted(p for p in test_root.iterdir() if p.is_dir() and p.name != "good"):
        mask_dir = gt_root / defect_dir.name
        for path in image_files(defect_dir):
            mask_path = mask_dir / f"{path.stem}_mask.png"
            if not mask_path.is_file():
                alternate = mask_dir / f"{path.stem}.png"
                mask_path = alternate if alternate.is_file() else mask_path
            if not mask_path.is_file():
                raise FileNotFoundError(f"Missing ground-truth mask for {path}")
            items.append((path, mask_path, True))
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/mvtec_ad"))
    parser.add_argument("--category", choices=CATEGORIES, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--results-dir", type=Path, default=Path("results/retrained/autoencoder"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise SystemExit("--epochs and --batch-size must be positive")
    if not 0.05 <= args.validation_fraction < 0.5:
        raise SystemExit("--validation-fraction must be in [0.05, 0.5)")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = torch.cuda.is_available()

    category_root = args.root / args.category
    audit = audit_category(category_root, args.category)
    if audit.missing_directories or audit.missing_masks or audit.unexpected_mask_groups:
        raise SystemExit(f"Dataset audit failed for {args.category}; run scripts/inspect_dataset.py")

    normal_paths = image_files(category_root / "train" / "good")
    random.Random(args.seed).shuffle(normal_paths)
    validation_count = max(1, round(len(normal_paths) * args.validation_fraction))
    validation_paths = normal_paths[:validation_count]
    training_paths = normal_paths[validation_count:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        ImageDataset(training_paths), batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        ImageDataset(validation_paths), batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=device.type == "cuda",
    )
    model = ConvAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    history: list[dict[str, float]] = []
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    print(
        f"Training {args.category} autoencoder on {device}; "
        f"train={len(training_paths)}, normal validation={len(validation_paths)}"
    )
    started = time.perf_counter()
    best_loss = float("inf")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.results_dir / f"{args.category}_autoencoder.pt"
    for epoch in range(args.epochs):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for images, _ in train_loader:
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                reconstruction = model(images)
                loss = loss_fn(reconstruction, images)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += float(loss.detach()) * images.size(0)
            train_count += images.size(0)

        model.eval()
        validation_loss_sum = 0.0
        validation_count = 0
        with torch.no_grad():
            for images, _ in validation_loader:
                images = images.to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    reconstruction = model(images)
                    loss = loss_fn(reconstruction, images)
                validation_loss_sum += float(loss) * images.size(0)
                validation_count += images.size(0)

        train_loss = train_loss_sum / train_count
        validation_loss = validation_loss_sum / validation_count
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "validation_loss": validation_loss})
        print(f"epoch {epoch + 1:02d}/{args.epochs}: train={train_loss:.6f} val={validation_loss:.6f}")
        if validation_loss < best_loss:
            best_loss = validation_loss
            torch.save(
                {"state_dict": model.state_dict(), "image_size": IMAGE_SIZE, "seed": args.seed},
                checkpoint,
            )

    training_seconds = time.perf_counter() - started
    with (args.results_dir / f"{args.category}_training_curve.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "validation_loss"])
        writer.writeheader()
        writer.writerows(history)

    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["state_dict"])
    model.eval()

    def score_batch(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
        loader = DataLoader(ImageDataset(paths), batch_size=args.batch_size, shuffle=False, num_workers=0)
        scores: list[float] = []
        maps: list[np.ndarray] = []
        with torch.inference_mode():
            for images, _ in loader:
                images = images.to(device)
                recon = model(images)
                anomaly_maps = torch.mean((recon - images).square(), dim=1)
                image_scores = anomaly_maps.mean(dim=(1, 2))
                scores.extend(image_scores.float().cpu().numpy().tolist())
                maps.extend(anomaly_maps.float().cpu().numpy())
        return np.asarray(scores), np.asarray(maps)

    validation_scores, validation_maps = score_batch(validation_paths)
    image_threshold = float(np.quantile(validation_scores, 0.99))
    pixel_threshold = float(np.quantile(validation_maps.reshape(-1), 0.995))

    test_items = collect_test_items(category_root)
    test_paths = [item[0] for item in test_items]
    test_scores, test_maps = score_batch(test_paths)
    labels = np.asarray([item[2] for item in test_items], dtype=np.uint8)
    predictions = test_scores >= image_threshold
    pixel_labels: list[np.ndarray] = []
    pixel_scores: list[np.ndarray] = []
    pixel_predictions: list[np.ndarray] = []
    for index, (_, mask_path, _) in enumerate(test_items):
        if mask_path is None:
            mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        else:
            with Image.open(mask_path) as opened:
                mask = np.asarray(
                    opened.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST),
                    dtype=np.uint8,
                ) > 0
        pixel_labels.append(mask.reshape(-1))
        pixel_scores.append(test_maps[index].reshape(-1))
        pixel_predictions.append((test_maps[index] >= pixel_threshold).reshape(-1))

    y_pixel = np.concatenate(pixel_labels).astype(np.uint8)
    score_pixel = np.concatenate(pixel_scores)
    pred_pixel = np.concatenate(pixel_predictions).astype(np.uint8)
    y_image = labels
    y_score = test_scores
    result = {
        "category": args.category,
        "model": "CNN Autoencoder",
        "seed": args.seed,
        "epochs": args.epochs,
        "train_images": len(training_paths),
        "validation_normal_images": len(validation_paths),
        "test_normal_images": int((labels == 0).sum()),
        "test_defect_images": int((labels == 1).sum()),
        "image_threshold": image_threshold,
        "pixel_threshold": pixel_threshold,
        "image_auroc": float(roc_auc_score(y_image, y_score)),
        "image_average_precision": float(average_precision_score(y_image, y_score)),
        "image_precision": float(precision_score(y_image, predictions, zero_division=0)),
        "image_recall": float(recall_score(y_image, predictions, zero_division=0)),
        "image_f1": float(f1_score(y_image, predictions, zero_division=0)),
        "pixel_auroc": float(roc_auc_score(y_pixel, score_pixel)),
        "pixel_average_precision": float(average_precision_score(y_pixel, score_pixel)),
        "pixel_precision": float(precision_score(y_pixel, pred_pixel, zero_division=0)),
        "pixel_recall": float(recall_score(y_pixel, pred_pixel, zero_division=0)),
        "pixel_f1": float(f1_score(y_pixel, pred_pixel, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_image, predictions, labels=[0, 1]).tolist(),
        "training_seconds": training_seconds,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "device": str(device),
        "input_size": IMAGE_SIZE,
        "threshold_source": "99th percentile of held-out normal validation images; 99.5th percentile of normal validation pixels",
    }
    report_path = args.results_dir / f"{args.category}_evaluation.json"
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Checkpoint: {checkpoint.resolve()}")
    print(f"Evaluation: {report_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
