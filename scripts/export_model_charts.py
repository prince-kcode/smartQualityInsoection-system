"""Export comparison, ROC, confusion-matrix, and training-curve images."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve
from torch.utils.data import DataLoader

from industrialguard.models.autoencoder import ConvAutoencoder
from industrialguard.data.mvtec import MVTEC_CATEGORIES
from train_autoencoder import ImageDataset, collect_test_items


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/mvtec_ad"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/retrained/autoencoder"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    combined: list[dict[str, object]] = []

    for category in MVTEC_CATEGORIES:
        result_path = args.results_dir / f"{category}_evaluation.json"
        checkpoint_path = args.results_dir / f"{category}_autoencoder.pt"
        if not result_path.is_file() or not checkpoint_path.is_file():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        category_root = args.dataset_root / category
        items = collect_test_items(category_root)
        paths = [item[0] for item in items]
        labels = np.asarray([item[2] for item in items], dtype=np.uint8)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model = ConvAutoencoder().to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        loader = DataLoader(ImageDataset(paths), batch_size=16, shuffle=False, num_workers=0)
        scores: list[float] = []
        with torch.inference_mode():
            for images, _ in loader:
                images = images.to(device)
                anomaly_map = torch.mean((model(images) - images).square(), dim=1)
                scores.extend(anomaly_map.mean(dim=(1, 2)).float().cpu().numpy().tolist())
        scores_array = np.asarray(scores)
        prediction = scores_array >= float(result["image_threshold"])

        fpr, tpr, _ = roc_curve(labels, scores_array)
        fig, ax = plt.subplots(figsize=(6.4, 5.0), constrained_layout=True)
        ax.plot(fpr, tpr, color="#126e82", linewidth=2.5, label=f"CNN autoencoder (AUC {result['image_auroc']:.3f})")
        ax.plot([0, 1], [0, 1], linestyle="--", color="#98a2b3", linewidth=1.2, label="Chance")
        ax.set(xlabel="False positive rate", ylabel="True positive rate", title=f"{category.replace('_', ' ').title()} image ROC")
        ax.legend(loc="lower right", frameon=False)
        ax.grid(alpha=0.2)
        fig.savefig(args.reports_dir / f"{category}_autoencoder_roc.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.0, 4.2), constrained_layout=True)
        ConfusionMatrixDisplay.from_predictions(
            labels, prediction, labels=[0, 1], display_labels=["Normal", "Defective"],
            cmap="Blues", colorbar=False, ax=ax,
        )
        ax.set_title(f"{category.replace('_', ' ').title()} autoencoder")
        fig.savefig(args.reports_dir / f"{category}_autoencoder_confusion.png", dpi=180)
        plt.close(fig)

        curve_path = args.results_dir / f"{category}_training_curve.csv"
        with curve_path.open("r", newline="", encoding="utf-8") as file:
            curve = list(csv.DictReader(file))
        fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
        ax.plot([int(row["epoch"]) for row in curve], [float(row["train_loss"]) for row in curve], label="Training loss", color="#126e82", linewidth=2)
        ax.plot([int(row["epoch"]) for row in curve], [float(row["validation_loss"]) for row in curve], label="Validation loss", color="#e07a35", linewidth=2)
        ax.set(xlabel="Epoch", ylabel="Mean squared error", title=f"{category.replace('_', ' ').title()} autoencoder training")
        ax.legend(frameon=False)
        ax.grid(alpha=0.2)
        fig.savefig(args.reports_dir / f"{category}_autoencoder_training.png", dpi=180)
        plt.close(fig)

        with (args.reports_dir / f"{category}_autoencoder_test_scores.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["image_path", "is_defective", "anomaly_score", "predicted_defective"])
            writer.writerows(
                [str(item[0]), int(labels[index]), f"{scores_array[index]:.9f}", int(prediction[index])]
                for index, item in enumerate(items)
            )

        combined.append(
            {
                "category": category,
                "image_auroc": result["image_auroc"],
                "image_f1": result["image_f1"],
                "pixel_auroc": result["pixel_auroc"],
                "pixel_f1": result["pixel_f1"],
                "training_seconds": result["training_seconds"],
                "checkpoint_bytes": result["checkpoint_bytes"],
            }
        )

    patchcore_csv = args.reports_dir / "retrained_patchcore_metrics.csv"
    patchcore = []
    if patchcore_csv.is_file():
        with patchcore_csv.open("r", newline="", encoding="utf-8") as file:
            patchcore = list(csv.DictReader(file))
    comparison: list[dict[str, str]] = []
    for row in patchcore:
        comparison.append({"category": row["category"], "model": "PatchCore", **{key: row[key] for key in ("image_auroc", "image_f1", "pixel_auroc", "pixel_f1")}})
    for row in combined:
        comparison.append({"category": str(row["category"]), "model": "CNN Autoencoder", **{key: f"{float(row[key]):.6f}" for key in ("image_auroc", "image_f1", "pixel_auroc", "pixel_f1")}})
    output_csv = args.reports_dir / "model_comparison.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["category", "model", "image_auroc", "image_f1", "pixel_auroc", "pixel_f1"])
        writer.writeheader()
        writer.writerows(comparison)

    categories = [
        category for category in MVTEC_CATEGORIES
        if any(row["category"] == category and row["model"] == "PatchCore" for row in comparison)
        and any(row["category"] == category and row["model"] == "CNN Autoencoder" for row in comparison)
    ]
    for metric, label in (("image_auroc", "Image AUROC"), ("pixel_auroc", "Pixel AUROC")):
        fig, ax = plt.subplots(figsize=(max(12, len(categories) * 0.9), 5.2), constrained_layout=True)
        x = np.arange(len(categories))
        width = 0.38
        for offset, model_name, color in ((-width / 2, "PatchCore", "#126e82"), (width / 2, "CNN Autoencoder", "#e07a35")):
            values = []
            for category in categories:
                row = next((item for item in comparison if item["category"] == category and item["model"] == model_name), None)
                values.append(float(row[metric]) if row else 0.0)
            ax.bar(x + offset, values, width=width, label=model_name, color=color)
        ax.set_xticks(x, [category.replace("_", " ").title() for category in categories], rotation=35, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(label)
        ax.set_title(f"Held-out test {label.lower()}")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.2)
        fig.savefig(args.reports_dir / f"comparison_{metric}.png", dpi=180)
        plt.close(fig)

    print(f"Wrote {output_csv.resolve()}")
    print(f"Charts and test score exports are under {args.reports_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
