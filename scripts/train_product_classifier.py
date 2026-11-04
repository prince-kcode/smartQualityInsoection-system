"""Train the 15-category product classifier and calibrate its OOD gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from industrialguard.data.mvtec import MVTEC_CATEGORIES
from industrialguard.inference.product_classifier import (
    IMAGE_TRANSFORM,
    create_feature_model,
)

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/mvtec_ad"))
    parser.add_argument("--ood-dir", type=Path, default=Path("data/raw/external/pbd/PBD_Dataset/body/val_images"),
                        help="Separate unsupported-image calibration set (default: PBD validation images).")
    parser.add_argument("--extra-ood", type=Path, action="append", default=[],
                        help="Additional unsupported holdout test image; repeat for multiple images.")
    parser.add_argument("--output", type=Path, default=Path("models/product_classifier.joblib"))
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def image_paths(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def unique_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"Calibration image not found: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest not in seen:
            seen.add(digest)
            unique.append(path)
    return unique


def extract_features(model, paths: list[Path], batch_size: int) -> np.ndarray:
    device = next(model.parameters()).device
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            tensors = []
            for path in paths[start:start + batch_size]:
                try:
                    with Image.open(path) as image:
                        tensors.append(IMAGE_TRANSFORM(image.convert("RGB")))
                except (OSError, ValueError) as error:
                    raise ValueError(f"Could not decode classifier image {path}: {error}") from error
            batch = torch.stack(tensors).to(device)
            features = torch.nn.functional.normalize(model(batch), dim=-1)
            output.append(features.cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(output, axis=0) if output else np.empty((0, 2048), dtype=np.float32)


def knn_distance(query: np.ndarray, bank: np.ndarray, k: int = 5, batch_size: int = 128) -> np.ndarray:
    k = min(k, len(bank))
    distances = []
    for start in range(0, len(query), batch_size):
        similarity = query[start:start + batch_size] @ bank.T
        nearest = np.partition(similarity, -k, axis=1)[:, -k:]
        distances.extend((1.0 - nearest).mean(axis=1).tolist())
    return np.asarray(distances, dtype=np.float32)


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least one")
    train_paths: list[Path] = []
    labels: list[int] = []
    for class_index, category in enumerate(MVTEC_CATEGORIES):
        files = image_paths(args.root / category / "train" / "good")
        if not files:
            raise SystemExit(f"No normal training images found for category {category}.")
        train_paths.extend(files)
        labels.extend([class_index] * len(files))

    # Reserve half of each MVTec test folder for OOD/category-threshold
    # calibration. The remaining half stays untouched for final evaluation.
    supported_test_cal_paths: list[Path] = []
    supported_test_cal_labels: list[int] = []
    for class_index, category in enumerate(MVTEC_CATEGORIES):
        test_root = args.root / category / "test"
        for group in sorted(path for path in test_root.iterdir() if path.is_dir()):
            files = image_paths(group)
            if len(files) < 2:
                continue
            rng = np.random.default_rng(args.seed + class_index * 1009 + sum(group.name.encode("utf-8")))
            order = rng.permutation(len(files))
            calibration_count = min(len(files) - 1, max(1, len(files) // 2))
            selected = [files[int(i)] for i in order[:calibration_count]]
            supported_test_cal_paths.extend(selected)
            supported_test_cal_labels.extend([class_index] * len(selected))

    ood_pool = unique_paths(image_paths(args.ood_dir))
    if len(ood_pool) < 2:
        raise SystemExit("Need at least two unsupported images to calibrate and independently test the OOD gate.")
    ood_cal_paths, ood_test_paths = train_test_split(
        ood_pool, test_size=0.30, random_state=args.seed,
    )
    ood_cal_paths = list(ood_cal_paths)
    ood_test_paths = unique_paths(list(ood_test_paths) + args.extra_ood)
    if not ood_cal_paths or not ood_test_paths:
        raise SystemExit("No unsupported calibration images found. Provide --ood-dir or --extra-ood.")
    missing = [path for path in ood_test_paths if not path.is_file()]
    if missing:
        raise SystemExit("Unsupported holdout image not found: " + str(missing[0]))

    y = np.asarray(labels, dtype=np.int64)
    train_index, support_cal_index = train_test_split(
        np.arange(len(y)), test_size=0.20, random_state=args.seed, stratify=y,
    )
    print(
        f"Extracting frozen Wide-ResNet features for {len(train_paths)} training images, "
        f"{len(supported_test_cal_paths)} held-out MVTec validation images, "
        f"{len(ood_cal_paths)} OOD calibration images, and "
        f"{len(ood_test_paths)} untouched OOD holdout images.", flush=True,
    )
    model = create_feature_model()
    features = extract_features(model, train_paths, args.batch_size)
    supported_test_cal_features = extract_features(model, supported_test_cal_paths, args.batch_size)
    ood_cal_features = extract_features(model, ood_cal_paths, args.batch_size)
    ood_test_features = extract_features(model, ood_test_paths, args.batch_size)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    classifier = LogisticRegression(max_iter=1500, C=1.0, random_state=args.seed)
    classifier.fit(features[train_index], y[train_index])
    categories = tuple(MVTEC_CATEGORIES)
    if tuple(classifier.classes_.tolist()) != tuple(range(len(categories))):
        raise RuntimeError("Classifier class indices do not map exactly to MVTEC_CATEGORIES.")

    support_normal_cal = features[support_cal_index]
    support_cal = np.concatenate([support_normal_cal, supported_test_cal_features], axis=0)
    support_labels = np.concatenate([y[support_cal_index], np.asarray(supported_test_cal_labels, dtype=np.int64)])
    ood_threshold_scores = knn_distance(support_cal, features[train_index])
    ood_cal_scores = knn_distance(ood_cal_features, features[train_index])
    ood_test_scores = knn_distance(ood_test_features, features[train_index])
    # The gate is calibrated with labeled supported and unsupported validation
    # inputs. Preserve at least 99% of held-out normal products, then reject as
    # many independent OOD examples as the data permits.
    ood_threshold = float(np.quantile(ood_threshold_scores, 0.99, method="higher"))

    support_probabilities = classifier.predict_proba(support_cal)
    support_order = np.argsort(support_probabilities, axis=1)[:, ::-1]
    support_confidence = support_probabilities[np.arange(len(support_cal)), support_order[:, 0]]
    support_margin = support_confidence - support_probabilities[np.arange(len(support_cal)), support_order[:, 1]]
    support_correct = support_order[:, 0] == support_labels
    if not np.any(support_correct):
        raise RuntimeError("Category classifier did not correctly classify the support calibration split.")
    confidence_threshold = max(0.50, float(np.quantile(support_confidence[support_correct], 0.05)))
    margin_threshold = max(0.10, float(np.quantile(support_margin[support_correct], 0.05)))
    support_auto = (
        (ood_threshold_scores <= ood_threshold)
        & (support_confidence >= confidence_threshold)
        & (support_margin >= margin_threshold)
    )

    import joblib

    args.output.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "classifier": classifier,
        "categories": categories,
        "feature_bank": features[train_index].astype(np.float32),
        "neighbor_count": 5,
        "ood_threshold": ood_threshold,
        "confidence_threshold": confidence_threshold,
        "margin_threshold": margin_threshold,
        "preprocessing": "RGB; resize 256x256 bilinear; ToTensor; ImageNet mean/std",
        "backbone": "timm wide_resnet50_2 pretrained ImageNet features",
        "seed": args.seed,
    }
    joblib.dump(bundle, args.output)
    support_accepted = ood_threshold_scores <= ood_threshold
    ood_rejected = ood_cal_scores > ood_threshold
    report = {
        "categories": list(categories),
        "train_images": int(len(train_index)),
        "supported_calibration_images": int(len(support_cal_index)),
        "supported_mvetest_calibration_images": int(len(supported_test_cal_paths)),
        "ood_calibration_images": int(len(ood_cal_paths)),
        "ood_holdout_images": int(len(ood_test_paths)),
        "ood_threshold": ood_threshold,
        "ood_score_definition": "mean cosine distance to five nearest training product embeddings",
        "supported_calibration_accept_rate": float(support_accepted.mean()),
        "ood_calibration_reject_rate": float(ood_rejected.mean()),
        "ood_calibration_directory": str(args.ood_dir),
        "ood_holdout_reject_rate": float(np.mean(ood_test_scores > ood_threshold)),
        "ood_holdout_distance_min": float(ood_test_scores.min()),
        "ood_holdout_distance_max": float(ood_test_scores.max()),
        "ood_calibration_extra_image_count": len(args.extra_ood),
        "category_confidence_threshold": confidence_threshold,
        "category_margin_threshold": margin_threshold,
        "support_calibration_category_accuracy": float(np.mean(support_correct)),
        "support_calibration_auto_category_coverage": float(np.mean(support_auto)),
        "support_calibration_auto_category_accuracy": (
            float(np.mean(support_correct[support_auto])) if np.any(support_auto) else None
        ),
        "support_calibration_uncertain_count": int(np.sum(~support_auto & (ood_threshold_scores <= ood_threshold))),
        "support_calibration_incorrect_auto_count": int(np.sum(support_auto & ~support_correct)),
        "artifact": str(args.output),
        "seed": args.seed,
    }
    report_path = args.output.with_name("product_classifier_calibration.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
