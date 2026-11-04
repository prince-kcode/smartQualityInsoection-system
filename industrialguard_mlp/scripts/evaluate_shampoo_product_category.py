"""Re-evaluate shampoo subtype recognition on its clean official held-out split."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import models

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.industrialguard_mlp.train import DEVICE, OUT, build_transform, discover_records, extract_embeddings


class ProductMLP(nn.Module):
    def __init__(self, classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 384), nn.BatchNorm1d(384), nn.ReLU(), nn.Dropout(.30),
            nn.Linear(384, 192), nn.BatchNorm1d(192), nn.ReLU(), nn.Dropout(.20),
            nn.Linear(192, classes),
        )

    def forward(self, x):
        return self.net(x)


def main() -> None:
    encoder_bundle = torch.load(OUT / "resnet18_encoder.pt", map_location="cpu", weights_only=True)
    encoder = nn.Sequential(*list(models.resnet18(weights=None).children())[:-1])
    encoder.load_state_dict(encoder_bundle["state_dict"])
    encoder.to(DEVICE).eval()

    product_bundle = torch.load(OUT / "product_mlp.pt", map_location="cpu", weights_only=True)
    product = ProductMLP(len(product_bundle["categories"]))
    product.load_state_dict(product_bundle["state_dict"])
    product.eval()
    categories = product_bundle["categories"]
    subtype_idx = categories.index("plastic_shampoo_bottle")

    category_rows, _, inventory = discover_records()
    rows = [row for row in category_rows
            if row["dataset"] == "Plastic Shampoo Bottle" and row["split"] == "test"]
    if not rows:
        raise RuntimeError("No clean official shampoo-bottle test images were found")
    features = extract_embeddings(rows, encoder, build_transform())
    with torch.inference_mode():
        predicted = product(torch.as_tensor(features)).argmax(1).cpu().numpy()

    true_positives = int(np.sum(predicted == subtype_idx))
    accuracy = true_positives / len(rows)
    report_path = ROOT / "reports" / "training_metrics.json"
    aggregate = json.loads(report_path.read_text(encoding="utf-8"))
    aggregate["datasets"]["Plastic Shampoo Bottle"] = inventory["Plastic Shampoo Bottle"]
    product_metrics = aggregate["product_classifier"]
    previous = product_metrics["plastic_shampoo_bottle_category_test"]
    false_positives = int(previous["shampoo_subtype_false_positives_on_other_held_out_sources"])
    result = {
        "test_images": len(rows),
        "split_method": "official Plastic Shampoo Bottle test split; notebook-checkpoint duplicate paths excluded",
        "category_accuracy": accuracy,
        "category_recall": accuracy,
        "correct_test_images": true_positives,
        "shampoo_subtype_false_positives_on_other_held_out_sources": false_positives,
        "other_held_out_source_images_checked": int(
            previous["other_held_out_source_images_checked"]),
        "subtype_precision_across_available_held_out_sources":
            true_positives / max(1, true_positives + false_positives),
    }
    product_metrics["plastic_shampoo_bottle_category_test"] = result
    report_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    (ROOT / "reports" / "plastic_shampoo_product_category_test.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
