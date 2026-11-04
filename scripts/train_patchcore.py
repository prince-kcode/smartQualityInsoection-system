"""Train and evaluate one category-specific PatchCore baseline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from industrialguard.data.mvtec import MVTEC_CATEGORIES, audit_category


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/raw/mvtec_ad"),
        help="MVTec AD root containing category folders (default: %(default)s)",
    )
    parser.add_argument(
        "--category",
        choices=MVTEC_CATEGORIES,
        required=True,
        help="Category to train as a separate one-class anomaly detector.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/retrained"),
        help="Directory for Anomalib logs and checkpoints (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    if args.num_workers < 0:
        raise SystemExit("--num-workers cannot be negative")

    category_root = args.root / args.category
    audit = audit_category(category_root, args.category)
    if audit.missing_directories:
        missing = ", ".join(audit.missing_directories)
        raise SystemExit(
            f"Dataset layout is incomplete for {args.category} at {category_root}: {missing}. "
            "Run scripts/inspect_dataset.py after extracting MVTec AD."
        )
    if audit.train_good == 0:
        raise SystemExit(f"No defect-free training images found in {category_root / 'train' / 'good'}")
    if audit.missing_masks or audit.unexpected_mask_groups:
        raise SystemExit(
            f"Ground-truth masks are incomplete for {args.category}. "
            "Run scripts/inspect_dataset.py and fix the mask layout before evaluation."
        )

    # Defer heavyweight model imports until the dataset path has been checked.
    from anomalib.data import MVTecAD
    from anomalib.engine import Engine
    from anomalib.models import Patchcore
    from lightning.pytorch import seed_everything

    seed_everything(args.seed, workers=True)
    datamodule = MVTecAD(
        root=str(args.root.resolve()),
        category=args.category,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_workers=args.num_workers,
        # Keep benchmark test images out of validation and threshold selection.
        # Synthetic validation is generated from the training data by Anomalib.
        val_split_mode="synthetic",
        val_split_ratio=0.1,
        seed=args.seed,
    )
    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        pre_trained=True,
        num_neighbors=9,
    )
    engine = Engine(
        default_root_dir=str((args.results_dir / args.category).resolve()),
        max_epochs=1,
        accelerator="auto",
        devices=1,
    )

    print(f"Training PatchCore for {args.category} using {audit.train_good} normal images.")
    engine.fit(model=model, datamodule=datamodule)
    print("Training complete. Evaluating once on the MVTec AD test split.")
    metrics = engine.test(model=model, datamodule=datamodule)
    category_results = args.results_dir / args.category
    category_results.mkdir(parents=True, exist_ok=True)
    serializable = metrics if metrics is not None else []
    (category_results / "test_metrics.json").write_text(
        json.dumps(serializable, indent=2, default=lambda value: float(value) if hasattr(value, "item") else str(value)),
        encoding="utf-8",
    )

    flattened: dict[str, float] = {}
    def collect_metrics(value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, dict):
                    collect_metrics(item)
                elif isinstance(item, (int, float)) or hasattr(item, "item"):
                    try:
                        flattened[str(key).lower().replace("_", "").replace("/", "")] = float(item)
                    except (TypeError, ValueError):
                        pass
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect_metrics(item)
    collect_metrics(metrics)

    def pick_metric(level: str, metric: str) -> float | None:
        return next((val for key, val in flattened.items() if level in key and metric in key), None)
    image_auroc = pick_metric("image", "auroc")
    image_f1 = pick_metric("image", "f1")
    pixel_auroc = pick_metric("pixel", "auroc")
    pixel_f1 = pick_metric("pixel", "f1")
    if None not in (image_auroc, image_f1, pixel_auroc, pixel_f1):
        metrics_csv = Path("reports/retrained_patchcore_metrics.csv")
        metrics_csv.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        if metrics_csv.is_file():
            with metrics_csv.open("r", newline="", encoding="utf-8") as file:
                rows = [row for row in csv.DictReader(file) if row["category"] != args.category]
        rows.append({
            "category": args.category,
            "image_auroc": f"{image_auroc:.6f}",
            "image_f1": f"{image_f1:.6f}",
            "pixel_auroc": f"{pixel_auroc:.6f}",
            "pixel_f1": f"{pixel_f1:.6f}",
            "train_normal_images": str(audit.train_good),
            "seed": str(args.seed),
        })
        with metrics_csv.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["category", "image_auroc", "image_f1", "pixel_auroc", "pixel_f1", "train_normal_images", "seed"])
            writer.writeheader(); writer.writerows(sorted(rows, key=lambda row: row["category"]))
    print(f"Test metrics saved: {(category_results / 'test_metrics.json').resolve()}")
    print(f"Anomalib outputs are under: {(args.results_dir / args.category).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
