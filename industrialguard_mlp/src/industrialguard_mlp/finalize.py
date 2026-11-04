"""Build inference banks and held-out metrics from the already trained MLP checkpoints."""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from torch import nn
from torchvision import models

from .train import (DATA, DEVICE, IMAGE_SIZE, MEAN, OUT, ROOT, STD, SEED, build_transform,
                    discover_records, extract_embeddings, load_neu_rows)


class MLP(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(512, 384), nn.BatchNorm1d(384), nn.ReLU(), nn.Dropout(.30),
                                 nn.Linear(384, 192), nn.BatchNorm1d(192), nn.ReLU(), nn.Dropout(.20),
                                 nn.Linear(192, classes))

    def forward(self, x):
        return self.net(x)


def load_mlp(path):
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    model = MLP(len(bundle["categories"]))
    model.load_state_dict(bundle["state_dict"])
    return model.eval(), bundle


def nearest_distances(features, predicted, categories, banks):
    distances = []
    for feature, idx in zip(features, predicted):
        bank = banks[categories[int(idx)]]
        distances.append(float(np.sqrt(((bank - feature[None, :]) ** 2).sum(1)).min()))
    return np.asarray(distances, dtype=np.float32)


def main():
    torch.manual_seed(SEED)
    model_dir = OUT
    enc_bundle = torch.load(model_dir / "resnet18_encoder.pt", map_location="cpu", weights_only=True)
    encoder = nn.Sequential(*list(models.resnet18(weights=None).children())[:-1])
    encoder.load_state_dict(enc_bundle["state_dict"])
    encoder.to(DEVICE).eval()
    product, p_bundle = load_mlp(model_dir / "product_mlp.pt")
    neu, n_bundle = load_mlp(model_dir / "neu_defect_mlp.pt")
    categories = p_bundle["categories"]
    category_to_idx = {name: i for i, name in enumerate(categories)}
    category_rows, anomaly_rows, inventory = discover_records()
    train_rows = [r for r in category_rows if r["split"] == "train"]
    val_rows = [r for r in category_rows if r["split"] == "validation" and not r["anomalous"]]
    test_rows = [r for r in anomaly_rows if r["split"] == "test"]
    if not train_rows or not val_rows:
        raise RuntimeError("Product train/validation rows are missing; check the extracted data layout.")

    transform = build_transform()
    category_bank_path = model_dir / "category_feature_banks.npz"
    normal_bank_path = model_dir / "normal_feature_banks.npz"
    category_bank_rows, bank_rows = [], []
    if category_bank_path.exists() and normal_bank_path.exists():
        with np.load(category_bank_path) as archive:
            category_banks = {name: archive[name] for name in archive.files}
        with np.load(normal_bank_path) as archive:
            banks = {name: archive[name] for name in archive.files}
        print(f"Loaded saved category and normal references; embedding {len(val_rows)} validation images and {len(test_rows)} held-out test images", flush=True)
    else:
        # Deterministic, balanced fallback for checkpoints created before the
        # train script started writing separate category/normal banks.
        rng = np.random.default_rng(SEED)
        grouped = defaultdict(list)
        for row in train_rows:
            grouped[row["category"]].append(row)
        for category in categories:
            category_rows_for_bank = grouped.get(category, [])
            if category_rows_for_bank:
                chosen = rng.choice(len(category_rows_for_bank),
                                    size=min(256, len(category_rows_for_bank)), replace=False)
                category_bank_rows.extend(category_rows_for_bank[int(i)] for i in chosen)
            normal_rows_for_bank = [r for r in category_rows_for_bank
                                    if not r["anomalous"] and r["dataset"] != "ETID"]
            if normal_rows_for_bank:
                chosen = rng.choice(len(normal_rows_for_bank),
                                    size=min(256, len(normal_rows_for_bank)), replace=False)
                bank_rows.extend(normal_rows_for_bank[int(i)] for i in chosen)
        print(f"Embedding {len(category_bank_rows)} category references, {len(bank_rows)} normal references, {len(val_rows)} validation images, and {len(test_rows)} held-out test images", flush=True)
        category_bank_x = extract_embeddings(category_bank_rows, encoder, transform)
        bank_x = extract_embeddings(bank_rows, encoder, transform)
        category_bank_lists = defaultdict(list)
        for row, vector in zip(category_bank_rows, category_bank_x):
            category_bank_lists[row["category"]].append(vector.astype(np.float32))
        category_banks = {category: np.stack(vectors) for category, vectors in category_bank_lists.items()}
        np.savez_compressed(category_bank_path, **category_banks)
        bank_lists = defaultdict(list)
        for row, vector in zip(bank_rows, bank_x):
            bank_lists[row["category"]].append(vector.astype(np.float32))
        banks = {category: np.stack(vectors) for category, vectors in bank_lists.items()}
        np.savez_compressed(normal_bank_path, **banks)
    val_x = extract_embeddings(val_rows, encoder, transform)
    test_x = extract_embeddings(test_rows, encoder, transform)

    val_logits = product(torch.as_tensor(val_x)).detach()
    val_probs = torch.softmax(val_logits, 1).numpy()
    val_pred = val_logits.argmax(1).numpy()
    val_true = np.asarray([category_to_idx[r["category"]] for r in val_rows], dtype=np.int64)
    val_dist = nearest_distances(val_x, val_pred, categories, category_banks)
    val_correct = val_pred == val_true
    cutoff = float(np.quantile(val_dist[val_correct] if val_correct.any() else val_dist, .99))
    confidence_cutoff = float(np.quantile(val_probs[np.arange(len(val_pred)), val_pred][val_correct]
                                          if val_correct.any() else val_probs.max(1), .05))
    normal_reference_distances = []
    for row, pred_idx, feature in zip(val_rows, val_pred, val_x):
        predicted_class = categories[int(pred_idx)]
        bank = banks.get(predicted_class)
        if bank is not None and predicted_class == row["category"] and row["dataset"] != "ETID":
            normal_reference_distances.append(
                float(np.sqrt(((bank - feature[None, :]) ** 2).sum(1)).min()))
    normal_reference_cutoff = float(np.quantile(normal_reference_distances or [cutoff], .99))
    (model_dir / "ood_calibration.json").write_text(json.dumps({
        "distance_p99": cutoff, "category_distance_p99": cutoff,
        "normal_reference_distance_p99": normal_reference_cutoff,
        "confidence_p05": confidence_cutoff,
        "calibration_count": len(val_rows), "calibration_correct_count": int(val_correct.sum()),
        "normal_reference_calibration_count": len(normal_reference_distances),
        "method": "category OOD uses category-feature banks; anomaly reference distance uses a separate 99th-percentile cutoff calibrated on correctly routed held-out normal non-ETID images",
    }, indent=2), encoding="utf-8")
    (model_dir / "category_index.json").write_text(json.dumps({"categories": categories,
        "category_to_index": category_to_idx}, indent=2), encoding="utf-8")

    test_logits = product(torch.as_tensor(test_x)).detach()
    test_pred = test_logits.argmax(1).numpy()
    test_true = np.asarray([category_to_idx[r["category"]] for r in test_rows], dtype=np.int64)
    test_dist = np.asarray([
        float(np.sqrt(((banks[row["category"]] - feature[None, :]) ** 2).sum(1)).min())
        for row, feature in zip(test_rows, test_x)
    ], dtype=np.float32)
    defect_y = np.asarray([r["anomalous"] for r in test_rows], dtype=np.int64)
    auc = float(roc_auc_score(defect_y, test_dist)) if len(np.unique(defect_y)) == 2 else None
    legacy_test_categories = sorted({r["category"] for r in test_rows})
    legacy_test_indices = [category_to_idx[name] for name in legacy_test_categories]
    test_report = classification_report(test_true, test_pred, labels=legacy_test_indices,
                                        target_names=legacy_test_categories, output_dict=True, zero_division=0)

    etid_test = [r for r in category_rows if r["dataset"] == "ETID" and r["split"] == "test"]
    etid_x = extract_embeddings(etid_test, encoder, transform)
    etid_pred = product(torch.as_tensor(etid_x)).argmax(1).numpy()
    etid_true = np.asarray([category_to_idx[r["category"]] for r in etid_test], dtype=np.int64)
    etid_names = sorted({r["category"] for r in etid_test})
    etid_indices = [category_to_idx[name] for name in etid_names]
    etid_report = classification_report(etid_true, etid_pred, labels=etid_indices,
                                        target_names=etid_names, output_dict=True, zero_division=0)
    etid_cable = [(row, pred) for row, pred in zip(etid_test, etid_pred) if row["category"] == "cable"]
    etid_metrics = {"test_images": len(etid_test), "category_accuracy": float(accuracy_score(etid_true, etid_pred)),
                    "macro_f1": float(etid_report["macro avg"]["f1-score"]),
                    "classification_report": etid_report,
                    "cable_test_images": len(etid_cable),
                    "cable_accuracy": float(sum(categories[int(pred)] == "cable" for _, pred in etid_cable)
                                             / max(1, len(etid_cable)))}

    shampoo_test = [r for r in category_rows
                    if r["dataset"] == "Plastic Shampoo Bottle" and r["split"] == "test"]
    shampoo_x = extract_embeddings(shampoo_test, encoder, transform)
    shampoo_pred = product(torch.as_tensor(shampoo_x)).argmax(1).numpy()
    shampoo_class_idx = category_to_idx["plastic_shampoo_bottle"]
    shampoo_true_positives = int((shampoo_pred == shampoo_class_idx).sum())
    other_heldout_predictions = np.concatenate((test_pred, etid_pred))
    shampoo_false_positives = int((other_heldout_predictions == shampoo_class_idx).sum())
    shampoo_accuracy = float(accuracy_score(
        np.asarray([category_to_idx[r["category"]] for r in shampoo_test], dtype=np.int64), shampoo_pred))
    shampoo_metrics = {
        "test_images": len(shampoo_test),
        "split_method": "official Plastic Shampoo Bottle test split; hidden notebook checkpoint duplicates excluded",
        "category_accuracy": shampoo_accuracy,
        "category_recall": shampoo_accuracy,
        "shampoo_subtype_false_positives_on_other_held_out_sources": shampoo_false_positives,
        "other_held_out_source_images_checked": int(len(other_heldout_predictions)),
        "subtype_precision_across_available_held_out_sources": float(
            shampoo_true_positives / max(1, shampoo_true_positives + shampoo_false_positives)),
    }

    cable_items = [(i, row) for i, row in enumerate(test_rows)
                   if row["dataset"] == "MVTec AD" and row["category"] == "cable"]
    cable = None
    if cable_items:
        cable_ix = np.asarray([i for i, _ in cable_items])
        cable_y = defect_y[cable_ix]
        cable_d = test_dist[cable_ix]
        cable = {"test_images": int(len(cable_ix)),
                 "category_accuracy": float(accuracy_score(np.full(len(cable_ix), category_to_idx["cable"]), test_pred[cable_ix])),
                 "normal_images": int((cable_y == 0).sum()), "defective_images": int((cable_y == 1).sum()),
                 "image_level_anomaly_roc_auc": float(roc_auc_score(cable_y, cable_d)) if len(np.unique(cable_y)) == 2 else None,
                 "threshold_screen_sensitivity": float(((cable_d > normal_reference_cutoff) & (cable_y == 1)).sum() / max(1, (cable_y == 1).sum())),
                 "threshold_screen_specificity": float(((cable_d <= normal_reference_cutoff) & (cable_y == 0)).sum() / max(1, (cable_y == 0).sum()))}

    neu_rows = load_neu_rows()
    neu_test = [r for r in neu_rows if r["split"] == "test"]
    neu_x = extract_embeddings(neu_test, encoder, transform)
    neu_true = np.asarray([n_bundle["categories"].index(r["category"]) for r in neu_test], dtype=np.int64)
    neu_pred = neu(torch.as_tensor(neu_x)).argmax(1).numpy()
    neu_report = classification_report(neu_true, neu_pred, labels=np.arange(len(n_bundle["categories"])),
                                       target_names=n_bundle["categories"], output_dict=True, zero_division=0)

    metrics = {
        "device": str(DEVICE), "seed": SEED, "image_size": IMAGE_SIZE, "datasets": inventory,
        "product_classifier": {"classes": categories, "train_images": len(train_rows), "validation_images": len(val_rows),
            "best_epoch": p_bundle.get("best_epoch"), "accuracy": float(accuracy_score(val_true, val_pred)),
            "macro_f1": float(classification_report(val_true, val_pred, labels=np.arange(len(categories)), target_names=categories,
                                                      output_dict=True, zero_division=0)["macro avg"]["f1-score"]),
            "held_out_test_images": len(test_rows), "held_out_category_accuracy": float(accuracy_score(test_true, test_pred)),
            "held_out_category_macro_f1": float(test_report["macro avg"]["f1-score"]),
            "classification_report": test_report, "confusion_matrix": confusion_matrix(test_true, test_pred).tolist(),
            "held_out_test_categories": legacy_test_categories,
            "ETID_source_group_held_out_test": etid_metrics,
            "plastic_shampoo_bottle_category_test": shampoo_metrics,
            "distance_reference": "separate category_feature_banks.npz; category-only ETID and shampoo-bottle images excluded from normal anomaly references"},
        "image_level_anomaly_screening": {"held_out_test_images": len(test_rows), "normal_images": int((defect_y == 0).sum()),
            "defect_images": int((defect_y == 1).sum()), "roc_auc": auc, "distance_threshold": normal_reference_cutoff,
            "category_distance_threshold": cutoff,
            "sensitivity": float((((test_dist > normal_reference_cutoff) & (defect_y == 1)).sum()) / max(1, (defect_y == 1).sum())),
            "specificity": float((((test_dist <= normal_reference_cutoff) & (defect_y == 0)).sum()) / max(1, (defect_y == 0).sum())),
            "cable_MVTec_AD": cable,
            "limitation": "Global pooled image embeddings provide image-level screening only; no defect localization is produced."},
        "neu_defect_classifier": {"classes": n_bundle["categories"], "test_images": len(neu_test),
            "accuracy": float(accuracy_score(neu_true, neu_pred)), "macro_f1": float(neu_report["macro avg"]["f1-score"]),
            "classification_report": neu_report, "confusion_matrix": confusion_matrix(neu_true, neu_pred).tolist()},
        "ood_calibration": {"distance_p99": cutoff, "category_distance_p99": cutoff,
            "normal_reference_distance_p99": normal_reference_cutoff, "confidence_p05": confidence_cutoff,
            "normal_validation_count": len(val_rows), "correct_validation_count": int(val_correct.sum()),
            "normal_reference_validation_count": len(normal_reference_distances),
            "category_reference_images_per_category_max": max((len(v) for v in category_banks.values()), default=0),
            "normal_reference_images_per_category_max": max((len(v) for v in banks.values()), default=0)},
    }
    aggregate_path = ROOT / "reports" / "training_metrics.json"
    prior = json.loads(aggregate_path.read_text(encoding="utf-8")) if aggregate_path.exists() else {}
    prior_product = prior.get("product_classifier", {})
    for key in ("selected_training_variant", "validation_tuning"):
        if key in prior_product:
            metrics["product_classifier"][key] = prior_product[key]
    prior.update(metrics)
    aggregate_path.write_text(json.dumps(prior, indent=2), encoding="utf-8")
    print(json.dumps({"product_validation_accuracy": metrics["product_classifier"]["accuracy"],
                      "product_test_accuracy": metrics["product_classifier"]["held_out_category_accuracy"],
                      "product_test_macro_f1": metrics["product_classifier"]["held_out_category_macro_f1"],
                      "ETID_test_accuracy": etid_metrics["category_accuracy"],
                      "ETID_test_macro_f1": etid_metrics["macro_f1"],
                      "ETID_cable_accuracy": etid_metrics["cable_accuracy"],
                      "anomaly_test_auc": auc, "cable": cable,
                      "neu_test_accuracy": metrics["neu_defect_classifier"]["accuracy"],
                      "neu_test_macro_f1": metrics["neu_defect_classifier"]["macro_f1"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
