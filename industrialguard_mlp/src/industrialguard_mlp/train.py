from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets"
OUT = ROOT / "models"
SEED = 73
IMAGE_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def canonical(name: str) -> str:
    name = name.strip().lower().replace(" ", "_")
    return {"capsules": "capsule"}.get(name, name)


def safe_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in EXTS
                  and ".ipynb_checkpoints" not in p.parts)


def discover_records() -> tuple[list[dict], list[dict], dict]:
    """Build image-level labels from the official normal/anomaly folder layouts."""
    category_rows: list[dict] = []
    anomaly_rows: list[dict] = []
    inventory: dict[str, dict] = {}

    def add(path: Path, category: str, split: str, anomalous: bool, dataset: str):
        row = {"path": str(path), "category": canonical(category), "split": split,
               "anomalous": int(anomalous), "dataset": dataset}
        anomaly_rows.append(row)
        if not anomalous:
            category_rows.append(row)

    # MVTec AD: train/good is the only training source; test supplies held-out normal/defect evaluation.
    root = DATA / "mvtec_ad"
    if root.exists():
        for category_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            cat = category_dir.name
            for img in safe_images(category_dir / "train" / "good"):
                add(img, cat, "train", False, "MVTec AD")
            for defect_dir in sorted((category_dir / "test").glob("*")) if (category_dir / "test").exists() else []:
                if defect_dir.is_dir():
                    abnormal = defect_dir.name.lower() != "good"
                    for img in safe_images(defect_dir):
                        add(img, cat, "test", abnormal, "MVTec AD")

    # VisA provides image-level train/test and normal/anomaly labels in its official CSV.
    visa_csv = DATA / "visa" / "split_csv" / "1cls.csv"
    if visa_csv.exists():
        with visa_csv.open("r", newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                rel = row.get("image", "").replace("/", "\\")
                path = DATA / "visa" / Path(rel)
                if not path.is_file():
                    continue
                cat = row.get("object", path.parts[-4])
                label = row.get("label", "normal").lower()
                split = row.get("split", "train").lower()
                add(path, cat, "train" if split == "train" and label == "normal" else "test",
                    label not in {"normal", "good", "0"}, "VisA")

    # MVTec AD 2: normal-only train and validation; public test includes good and bad.
    root = DATA / "mvtec_ad_2"
    if root.exists():
        for category_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            cat = category_dir.name
            for img in safe_images(category_dir / "train" / "good"):
                add(img, cat, "train", False, "MVTec AD 2")
            for img in safe_images(category_dir / "validation" / "good"):
                add(img, cat, "validation", False, "MVTec AD 2")
            test_root = category_dir / "test_public"
            for group in ("good", "bad"):
                for img in safe_images(test_root / group):
                    add(img, cat, "test", group == "bad", "MVTec AD 2")

    # PharmaVision's split folders contain tablet photos under normal/defective.
    # They expand the existing "pill" product class; the archive has no masks,
    # and its per-image folders do not include severity/type grades.
    pv_root = DATA / "additional" / "PharmaVision"
    if pv_root.exists():
        for folder, split in (("Train", "train"), ("Val", "validation"), ("Test", "test")):
            for label in ("normal", "defective"):
                for img in safe_images(pv_root / folder / label):
                    row = {"path": str(img), "category": "pill", "split": split,
                           "anomalous": int(label == "defective"), "dataset": "PharmaVision"}
                    anomaly_rows.append(row)
                    # Unlike the industrial one-class datasets, this labeled
                    # source includes both intact and defective examples for
                    # recognizing the product category in different conditions.
                    category_rows.append(row)

    # Add an explicit product subtype for the YOLO-labeled shampoo-bottle
    # domain. Keeping this separate from generic MVTec ``bottle`` prevents the
    # bottle-only defect detector from being routed to glass/other bottles.
    bottle_root = DATA / "additional" / "Plastic Shampoo Bottle Dataset"
    if bottle_root.exists():
        for folder, split in (("train", "train"), ("valid", "validation"), ("test", "test")):
            for img in safe_images(bottle_root / folder / "images"):
                category_rows.append({
                    "path": str(img), "category": "plastic_shampoo_bottle", "split": split,
                    "anomalous": 0, "dataset": "Plastic Shampoo Bottle",
                    "category_only": True,
                })

    # ETID is category-only data: it shows real e-waste products, but supplies
    # no normal/defect labels. Preserve its source-group-held-out split exactly
    # and never add these photos to the normal-reference anomaly rows.
    etid_manifest = DATA / "additional" / "etid" / "manifest.csv"
    if etid_manifest.exists():
        with etid_manifest.open("r", newline="", encoding="utf-8") as stream:
            for item in csv.DictReader(stream):
                path = ROOT / item["path"]
                if not path.is_file():
                    raise RuntimeError(f"ETID manifest image is missing: {path}")
                category_rows.append({"path": str(path), "category": canonical(item["category"]),
                                      "split": item["split"], "anomalous": 0, "dataset": "ETID",
                                      "category_only": True, "source_group": item["source_group"]})

    # Deterministic stratified validation split from MVTec AD / VisA training normals.
    train_by_class: dict[str, list[dict]] = defaultdict(list)
    for row in category_rows:
        if row["split"] == "train" and row["dataset"] not in {
                "PharmaVision", "ETID", "Plastic Shampoo Bottle"}:
            train_by_class[row["category"]].append(row)
    for cat, rows in train_by_class.items():
        if len(rows) >= 5:
            ix = np.arange(len(rows))
            tr, va = train_test_split(ix, test_size=0.15, random_state=SEED)
            for i in va:
                rows[i]["split"] = "validation"

    for dataset in ("MVTec AD", "VisA", "MVTec AD 2", "PharmaVision"):
        selected = [r for r in anomaly_rows if r["dataset"] == dataset]
        inventory[dataset] = {
            "images": len(selected),
            "normal_images": sum(not r["anomalous"] for r in selected),
            "anomaly_images": sum(r["anomalous"] for r in selected),
            "categories": sorted({r["category"] for r in selected}),
        }
    etid_rows = [row for row in category_rows if row["dataset"] == "ETID"]
    if etid_rows:
        inventory["ETID"] = {
            "images": len(etid_rows),
            "category_only": True,
            "categories": sorted({row["category"] for row in etid_rows}),
            "split_counts": {split: sum(row["split"] == split for row in etid_rows)
                             for split in ("train", "validation", "test")},
            "class_counts_by_split": {
                split: {category: sum(row["split"] == split and row["category"] == category
                                      for row in etid_rows)
                        for category in sorted({row["category"] for row in etid_rows})}
                for split in ("train", "validation", "test")},
            "split_method": "source-group-held-out; category labels only; excluded from anomaly evaluation and normal reference banks",
        }
    bottle_rows = [row for row in category_rows if row["dataset"] == "Plastic Shampoo Bottle"]
    if bottle_rows:
        inventory["Plastic Shampoo Bottle"] = {
            "images": len(bottle_rows), "category_only": True,
            "categories": ["plastic_shampoo_bottle"],
            "split_counts": {split: sum(row["split"] == split for row in bottle_rows)
                             for split in ("train", "validation", "test")},
            "split_method": "official YOLO train/valid/test image splits; category-only labels kept out of defect and normal-reference evaluation",
        }
    return category_rows, anomaly_rows, inventory


class ImageDataset(Dataset):
    def __init__(self, rows: list[dict], transform):
        self.rows, self.transform = rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        try:
            image = Image.open(row["path"]).convert("RGB")
        except (OSError, UnidentifiedImageError) as exc:
            raise RuntimeError(f"Cannot read image {row['path']}: {exc}") from exc
        return self.transform(image), index


def build_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def make_encoder():
    weights = models.ResNet18_Weights.DEFAULT
    backbone = models.resnet18(weights=weights)
    encoder = nn.Sequential(*list(backbone.children())[:-1]).to(DEVICE).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


def extract_embeddings(rows: list[dict], encoder, transform, batch_size=96):
    loader = DataLoader(ImageDataset(rows, transform), batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=DEVICE.type == "cuda")
    feats = torch.empty((len(rows), 512), dtype=torch.float32)
    with torch.inference_mode():
        for batch, indices in loader:
            batch = batch.to(DEVICE, non_blocking=True)
            output = encoder(batch).flatten(1).float().cpu()
            feats[indices] = output
    return feats.numpy()


class MLP(nn.Module):
    def __init__(self, input_dim: int, classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 384), nn.BatchNorm1d(384), nn.ReLU(), nn.Dropout(0.30),
            nn.Linear(384, 192), nn.BatchNorm1d(192), nn.ReLU(), nn.Dropout(0.20),
            nn.Linear(192, classes),
        )

    def forward(self, x):
        return self.net(x)


def fit_mlp(train_x, train_y, val_x, val_y, classes, epochs=45, class_weight_power=0.0):
    torch.manual_seed(SEED)
    model = MLP(train_x.shape[1], classes).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-3)
    if class_weight_power > 0:
        counts = np.bincount(train_y, minlength=classes).astype(np.float32)
        weights = np.power(counts.max() / np.maximum(counts, 1), class_weight_power)
        weights /= weights.mean()
        criterion = nn.CrossEntropyLoss(weight=torch.as_tensor(weights, dtype=torch.float32, device=DEVICE))
    else:
        criterion = nn.CrossEntropyLoss()
    best, best_state, best_epoch = float("inf"), None, 0
    x = torch.as_tensor(train_x, dtype=torch.float32)
    y = torch.as_tensor(train_y, dtype=torch.long)
    ds = torch.utils.data.TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=128, shuffle=True)
    val_x_t = torch.as_tensor(val_x, dtype=torch.float32, device=DEVICE)
    val_y_t = torch.as_tensor(val_y, dtype=torch.long, device=DEVICE)
    for epoch in range(epochs):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(val_x_t), val_y_t).item() if len(val_y) else 0.0
        if val_loss < best:
            best, best_epoch = val_loss, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch+1:02d}/{epochs} validation_loss={val_loss:.4f}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu().eval(), best_epoch


def load_neu_rows():
    root = DATA / "neu_cls" / "NEU-CLS"
    names = {"Cr": "Crazing", "In": "Inclusion", "Pa": "Patches",
             "PS": "Pitted_surface", "RS": "Rolled-in_scale", "Sc": "Scratches"}
    rows = []
    for img in safe_images(root):
        prefix = img.stem.split("_")[0]
        if prefix in names:
            rows.append({"path": str(img), "category": names[prefix], "split": "train",
                         "anomalous": 1, "dataset": "NEU-CLS"})
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["category"]].append(row)
    for group in groups.values():
        indices = np.arange(len(group))
        tr, te = train_test_split(indices, test_size=0.2, random_state=SEED)
        for i in te:
            group[i]["split"] = "test"
        tr, va = train_test_split(tr, test_size=0.15, random_state=SEED)
        for i in va:
            group[i]["split"] = "validation"
    return rows


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Train clean IndustrialGuard MLP models")
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--product-only", action="store_true",
                        help="Retrain product recognition and retain the current NEU steel model")
    args = parser.parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    report_path = ROOT / "reports" / "training_metrics.json"
    prior_report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    category_rows, anomaly_rows, inventory = discover_records()
    print("Dataset inventory:", json.dumps(inventory, indent=2), flush=True)
    if not category_rows:
        raise SystemExit("No normal product images were found in the configured dataset folders.")

    encoder = make_encoder()
    torch.save({"state_dict": encoder.state_dict(), "weights": "ResNet18_Weights.DEFAULT",
                "preprocessing": {"resize": 256, "center_crop": IMAGE_SIZE, "mean": MEAN, "std": STD}},
               OUT / "resnet18_encoder.pt")
    transform = build_transform()
    # Use explicit split labels and only source train-normal images for fitting.
    product_train = [r for r in category_rows if r["split"] == "train"]
    product_val = [r for r in category_rows if r["split"] == "validation"]
    if not product_val:
        raise SystemExit("No held-out normal images were found for product-category validation.")
    all_categories = sorted({r["category"] for r in category_rows})
    category_to_idx = {c: i for i, c in enumerate(all_categories)}
    print(f"Extracting product embeddings: {len(product_train)} train, {len(product_val)} validation, {len(all_categories)} classes", flush=True)
    train_features = extract_embeddings(product_train, encoder, transform, args.batch_size)
    val_features = extract_embeddings(product_val, encoder, transform, args.batch_size)
    train_y = np.array([category_to_idx[r["category"]] for r in product_train])
    val_y = np.array([category_to_idx[r["category"]] for r in product_val])
    # Compare the original loss to a moderate inverse-square-root class weight.
    # Select the final model by held-out validation macro-F1, never ETID test.
    product_candidates = []
    for candidate_name, weight_power in (("unweighted", 0.0), ("inverse_sqrt_class_frequency", 0.5)):
        candidate_model, candidate_epoch = fit_mlp(
            train_features, train_y, val_features, val_y, len(all_categories), args.epochs,
            class_weight_power=weight_power)
        candidate_pred = candidate_model(torch.as_tensor(val_features)).argmax(1).numpy()
        candidate_report = classification_report(val_y, candidate_pred, labels=np.arange(len(all_categories)),
                                                 target_names=all_categories, output_dict=True, zero_division=0)
        product_candidates.append({"name": candidate_name, "class_weight_power": weight_power,
                                   "model": candidate_model, "best_epoch": candidate_epoch,
                                   "validation_accuracy": float(accuracy_score(val_y, candidate_pred)),
                                   "validation_macro_f1": float(candidate_report["macro avg"]["f1-score"]),
                                   "validation_report": candidate_report})
    product_candidates.sort(key=lambda item: (item["validation_macro_f1"], item["validation_accuracy"]), reverse=True)
    chosen_candidate = product_candidates[0]
    product_mlp = chosen_candidate["model"]
    product_best_epoch = chosen_candidate["best_epoch"]
    print("Product classifier validation tuning:", json.dumps([
        {k: item[k] for k in ("name", "class_weight_power", "best_epoch", "validation_accuracy", "validation_macro_f1")}
        for item in product_candidates], indent=2), flush=True)
    product_pred = product_mlp(torch.as_tensor(val_features)).argmax(1).numpy()
    product_report = classification_report(val_y, product_pred, labels=np.arange(len(all_categories)),
                                           target_names=all_categories, output_dict=True, zero_division=0)
    torch.save({"state_dict": product_mlp.state_dict(), "categories": all_categories,
                "input_dim": int(train_features.shape[1]), "best_epoch": product_best_epoch,
                "backbone": "torchvision ResNet18 ImageNet1K_V1 frozen feature encoder",
                "preprocessing": {"resize": 256, "center_crop": IMAGE_SIZE, "mean": MEAN, "std": STD}},
               OUT / "product_mlp.pt")

    # This steel-surface task is separate from product recognition. Keep its
    # tested weights intact when the product category set changes.
    if args.product_only:
        neu_metrics = prior_report.get("neu_defect_classifier")
        if not neu_metrics or not (OUT / "neu_defect_mlp.pt").exists():
            raise RuntimeError("--product-only needs an existing NEU model and its training report")
    else:
        neu_rows = load_neu_rows()
        neu_labels = sorted({r["category"] for r in neu_rows})
        neu_to_idx = {c: i for i, c in enumerate(neu_labels)}
        neu_train = [r for r in neu_rows if r["split"] == "train"]
        neu_val = [r for r in neu_rows if r["split"] == "validation"]
        neu_test = [r for r in neu_rows if r["split"] == "test"]
        print(f"Extracting NEU embeddings: {len(neu_train)} train, {len(neu_val)} validation, {len(neu_test)} test", flush=True)
        neu_train_x = extract_embeddings(neu_train, encoder, transform, args.batch_size)
        neu_val_x = extract_embeddings(neu_val, encoder, transform, args.batch_size)
        neu_test_x = extract_embeddings(neu_test, encoder, transform, args.batch_size)
        neu_train_y = np.array([neu_to_idx[r["category"]] for r in neu_train])
        neu_val_y = np.array([neu_to_idx[r["category"]] for r in neu_val])
        neu_test_y = np.array([neu_to_idx[r["category"]] for r in neu_test])
        neu_mlp, neu_best_epoch = fit_mlp(neu_train_x, neu_train_y, neu_val_x, neu_val_y, len(neu_labels), args.epochs)
        neu_pred = neu_mlp(torch.as_tensor(neu_test_x)).argmax(1).numpy()
        neu_report = classification_report(neu_test_y, neu_pred, labels=np.arange(len(neu_labels)),
                                           target_names=neu_labels, output_dict=True, zero_division=0)
        torch.save({"state_dict": neu_mlp.state_dict(), "categories": neu_labels,
                    "input_dim": int(neu_train_x.shape[1]), "best_epoch": neu_best_epoch,
                    "backbone": "torchvision ResNet18 ImageNet1K_V1 frozen feature encoder",
                    "preprocessing": {"resize": 256, "center_crop": IMAGE_SIZE, "mean": MEAN, "std": STD}},
                   OUT / "neu_defect_mlp.pt")
        neu_metrics = {"classes": neu_labels, "train_images": len(neu_train),
                       "validation_images": len(neu_val), "test_images": len(neu_test),
                       "best_epoch": neu_best_epoch, "accuracy": accuracy_score(neu_test_y, neu_pred),
                       "macro_f1": neu_report["macro avg"]["f1-score"],
                       "classification_report": neu_report,
                       "confusion_matrix": confusion_matrix(neu_test_y, neu_pred).tolist()}

    # Keep category recognition references separate from verified normal
    # references. ETID is unlabeled for defects and must never enter the latter.
    category_banks: dict[str, list] = defaultdict(list)
    banks: dict[str, list] = defaultdict(list)
    for row, feature in zip(product_train, train_features):
        category_banks[row["category"]].append(feature.astype(np.float32))
        if not row["anomalous"] and row["dataset"] not in {"ETID", "Plastic Shampoo Bottle"}:
            banks[row["category"]].append(feature.astype(np.float32))
    category_bank_arrays = {c: np.stack(v) for c, v in category_banks.items()}
    bank_arrays = {c: np.stack(v) for c, v in banks.items()}
    np.savez_compressed(OUT / "category_feature_banks.npz", **category_bank_arrays)
    np.savez_compressed(OUT / "normal_feature_banks.npz", **bank_arrays)
    validation_rows = product_val
    with torch.inference_mode():
        val_logits = product_mlp(torch.as_tensor(val_features))
        val_pred_idx = val_logits.argmax(1).numpy()
        val_softmax = torch.softmax(val_logits, dim=1).numpy()
    val_dist = []
    for row, pred_idx, feature, probs in zip(validation_rows, val_pred_idx, val_features, val_softmax):
        predicted_class = all_categories[pred_idx]
        bank = category_bank_arrays.get(predicted_class, val_features)
        d = np.sqrt(((bank - feature[None, :]) ** 2).sum(1)).min()
        val_dist.append({"true": row["category"], "predicted": predicted_class, "distance": float(d),
                         "confidence": float(probs[pred_idx]), "correct": int(predicted_class == row["category"])})
    # OOD cutoff is calibrated from held-out supported category images, using
    # category-only features rather than normal-product anomaly references.
    normal_correct_val = [x for x, row in zip(val_dist, validation_rows)
                          if x["correct"] and not row["anomalous"]]
    cutoff = float(np.quantile([x["distance"] for x in normal_correct_val] or [x["distance"] for x in val_dist], 0.99))
    confidence_cutoff = float(np.quantile([x["confidence"] for x in normal_correct_val] or [x["confidence"] for x in val_dist], 0.05))
    normal_reference_val_distances = []
    for row, pred_idx, feature in zip(validation_rows, val_pred_idx, val_features):
        predicted_class = all_categories[pred_idx]
        bank = bank_arrays.get(predicted_class)
        if (bank is not None and predicted_class == row["category"]
                and not row["anomalous"]
                and row["dataset"] not in {"ETID", "Plastic Shampoo Bottle"}):
            normal_reference_val_distances.append(
                float(np.sqrt(((bank - feature[None, :]) ** 2).sum(1)).min()))
    normal_reference_cutoff = float(np.quantile(normal_reference_val_distances or [cutoff], 0.99))

    # Evaluate category routing and image-level normal/defect screening on the untouched public test sets.
    test_rows = [r for r in anomaly_rows if r["split"] == "test"]
    print(f"Extracting held-out product test embeddings: {len(test_rows)}", flush=True)
    test_features = extract_embeddings(test_rows, encoder, transform, args.batch_size)
    with torch.inference_mode():
        test_logits = product_mlp(torch.as_tensor(test_features))
        test_probs = torch.softmax(test_logits, dim=1).numpy()
        test_pred = test_logits.argmax(1).numpy()
    test_true = np.array([category_to_idx[r["category"]] for r in test_rows], dtype=np.int64)
    legacy_test_categories = sorted({r["category"] for r in test_rows})
    legacy_test_indices = [category_to_idx[name] for name in legacy_test_categories]
    legacy_test_report = classification_report(
        test_true, test_pred, labels=legacy_test_indices, target_names=legacy_test_categories,
        output_dict=True, zero_division=0)
    test_distances = []
    for row, feature in zip(test_rows, test_features):
        # This benchmark measures anomaly screening within the known category;
        # predicted-class routing is evaluated separately. ETID-only categories
        # deliberately have no normal/anomaly reference bank.
        bank = bank_arrays.get(row["category"])
        if bank is None:
            raise RuntimeError(f"No labeled normal-reference bank for held-out anomaly category {row['category']}")
        test_distances.append(float(np.sqrt(((bank - feature[None, :]) ** 2).sum(1)).min()))
    test_distances = np.asarray(test_distances, dtype=np.float32)
    test_anomaly_y = np.array([r["anomalous"] for r in test_rows], dtype=np.int64)
    test_auc = float(roc_auc_score(test_anomaly_y, test_distances)) if len(np.unique(test_anomaly_y)) == 2 else None
    test_predictions = (test_distances > normal_reference_cutoff).astype(np.int64)

    etid_test_rows = [r for r in category_rows if r["dataset"] == "ETID" and r["split"] == "test"]
    print(f"Extracting source-group-held-out ETID classification images: {len(etid_test_rows)}", flush=True)
    etid_test_features = extract_embeddings(etid_test_rows, encoder, transform, args.batch_size)
    with torch.inference_mode():
        etid_test_logits = product_mlp(torch.as_tensor(etid_test_features))
        etid_test_pred = etid_test_logits.argmax(1).numpy()
    etid_test_true = np.asarray([category_to_idx[r["category"]] for r in etid_test_rows], dtype=np.int64)
    etid_class_names = sorted({row["category"] for row in etid_test_rows})
    etid_class_indices = [category_to_idx[name] for name in etid_class_names]
    etid_test_report = classification_report(
        etid_test_true, etid_test_pred, labels=etid_class_indices,
        target_names=etid_class_names, output_dict=True, zero_division=0)
    etid_cable_rows = [(row, pred) for row, pred in zip(etid_test_rows, etid_test_pred)
                       if row["category"] == "cable"]
    etid_test_metrics = {
        "test_images": len(etid_test_rows),
        "split_method": "source-group-held-out ETID test groups; no training/validation source group overlap",
        "category_accuracy": float(accuracy_score(etid_test_true, etid_test_pred)),
        "macro_f1": float(etid_test_report["macro avg"]["f1-score"]),
        "classification_report": etid_test_report,
        "confusion_matrix": confusion_matrix(etid_test_true, etid_test_pred,
                                                labels=etid_class_indices).tolist(),
        "cable_test_images": len(etid_cable_rows),
        "cable_accuracy": float(sum(all_categories[int(pred)] == "cable"
                                     for _, pred in etid_cable_rows) / max(1, len(etid_cable_rows))),
    }
    shampoo_test_rows = [r for r in category_rows
                         if r["dataset"] == "Plastic Shampoo Bottle" and r["split"] == "test"]
    print(f"Extracting held-out plastic shampoo-bottle category images: {len(shampoo_test_rows)}", flush=True)
    shampoo_test_features = extract_embeddings(shampoo_test_rows, encoder, transform, args.batch_size)
    with torch.inference_mode():
        shampoo_test_pred = product_mlp(torch.as_tensor(shampoo_test_features)).argmax(1).numpy()
    shampoo_test_true = np.asarray([category_to_idx[r["category"]] for r in shampoo_test_rows], dtype=np.int64)
    shampoo_class_idx = category_to_idx["plastic_shampoo_bottle"]
    non_shampoo_test_pred = np.concatenate((test_pred, etid_test_pred))
    non_shampoo_test_count = int((non_shampoo_test_pred == shampoo_class_idx).sum())
    shampoo_true_positives = int((shampoo_test_pred == shampoo_class_idx).sum())
    shampoo_test_accuracy = float(accuracy_score(shampoo_test_true, shampoo_test_pred))
    shampoo_bottle_test_metrics = {
        "test_images": len(shampoo_test_rows),
        "split_method": "official Plastic Shampoo Bottle test split; kept separate from train/validation",
        "category_accuracy": shampoo_test_accuracy,
        "category_recall": shampoo_test_accuracy,
        "shampoo_subtype_false_positives_on_other_held_out_sources": non_shampoo_test_count,
        "other_held_out_source_images_checked": int(len(non_shampoo_test_pred)),
        "subtype_precision_across_available_held_out_sources": float(
            shampoo_true_positives / max(1, shampoo_true_positives + non_shampoo_test_count)),
    }
    cable_idx = category_to_idx.get("cable")
    cable_records = [(r, p, d) for r, p, d in zip(test_rows, test_pred, test_distances)
                     if r["dataset"] == "MVTec AD" and r["category"] == "cable"]
    cable_metrics = None
    if cable_records:
        cable_y = np.array([r["anomalous"] for r, _, _ in cable_records], dtype=np.int64)
        cable_p = np.array([p for _, p, _ in cable_records], dtype=np.int64)
        cable_d = np.array([d for _, _, d in cable_records], dtype=np.float32)
        cable_metrics = {"test_images": len(cable_records),
                         "category_accuracy": float(accuracy_score(np.full(len(cable_records), cable_idx), cable_p)),
                         "normal_images": int((cable_y == 0).sum()), "defective_images": int((cable_y == 1).sum()),
                         "image_level_anomaly_roc_auc": float(roc_auc_score(cable_y, cable_d)) if len(np.unique(cable_y)) == 2 else None,
                         "threshold_screen_sensitivity": float(((cable_d > normal_reference_cutoff) & (cable_y == 1)).sum() / max(1, (cable_y == 1).sum())),
                         "threshold_screen_specificity": float(((cable_d <= normal_reference_cutoff) & (cable_y == 0)).sum() / max(1, (cable_y == 0).sum()))}
    write_json(OUT / "category_index.json", {"categories": all_categories, "category_to_index": category_to_idx})
    write_json(OUT / "ood_calibration.json", {"distance_p99": cutoff,
                                                "category_distance_p99": cutoff,
                                                "normal_reference_distance_p99": normal_reference_cutoff,
                                                "confidence_p05": confidence_cutoff,
                                                "calibration_count": len(val_dist),
                                                "calibration_correct_count": sum(x["correct"] for x in val_dist),
                                                "normal_reference_calibration_count": len(normal_reference_val_distances),
                                                "method": "product category OOD uses category-only feature banks; anomaly reference distance has a separate 99th-percentile cutoff calibrated on correctly routed, held-out normal non-ETID images"})
    combined_metrics = {
        "device": str(DEVICE), "seed": SEED, "image_size": IMAGE_SIZE,
        "datasets": inventory,
        "product_classifier": {"classes": all_categories, "train_images": len(product_train),
                               "validation_images": len(product_val), "best_epoch": product_best_epoch,
                               "selected_training_variant": chosen_candidate["name"],
                               "validation_tuning": [{k: item[k] for k in
                                    ("name", "class_weight_power", "best_epoch", "validation_accuracy", "validation_macro_f1")}
                                    for item in product_candidates],
                               "accuracy": accuracy_score(val_y, product_pred),
                               "macro_f1": product_report["macro avg"]["f1-score"],
                               "classification_report": product_report,
                               "confusion_matrix": confusion_matrix(val_y, product_pred).tolist(),
                               "held_out_test_images": len(test_rows),
                               "held_out_category_accuracy": float(accuracy_score(test_true, test_pred)) if len(test_rows) else None,
                               "held_out_category_macro_f1": legacy_test_report["macro avg"]["f1-score"] if len(test_rows) else None,
                               "held_out_test_categories": legacy_test_categories,
                               "ETID_source_group_held_out_test": etid_test_metrics,
                               "plastic_shampoo_bottle_category_test": shampoo_bottle_test_metrics,
                               "distance_reference": "separate category_feature_banks.npz; category-only ETID and bottle-subtype images do not enter normal anomaly references"},
        "image_level_anomaly_screening": {"held_out_test_images": len(test_rows),
                                           "normal_images": int((test_anomaly_y == 0).sum()),
                                           "defect_images": int((test_anomaly_y == 1).sum()),
                                           "roc_auc": test_auc,
                                           "threshold": normal_reference_cutoff,
                                           "category_distance_threshold": cutoff,
                                           "sensitivity": float(((test_predictions == 1) & (test_anomaly_y == 1)).sum() / max(1, (test_anomaly_y == 1).sum())),
                                           "specificity": float(((test_predictions == 0) & (test_anomaly_y == 0)).sum() / max(1, (test_anomaly_y == 0).sum())),
                                           "cable_MVTec_AD": cable_metrics,
                                           "limitation": "Uses global average pooled image embeddings; local defect localization is not supported."},
         "neu_defect_classifier": neu_metrics,
         "ood_calibration": {"distance_p99": cutoff, "category_distance_p99": cutoff,
                             "normal_reference_distance_p99": normal_reference_cutoff,
                             "confidence_p05": confidence_cutoff,
                            "normal_validation_count": sum(not row["anomalous"] for row in validation_rows),
                            "correct_validation_count": len(normal_correct_val)},
        "important_limit": "The product MLP identifies product categories. Wear/damage localization is handled by a separate segmentation model; the MLP does not localize defects.",
         "dataset_notes": {"PharmaVision": "Used Train/Val/Test normal and defective tablet images as the existing pill product category. The downloaded archive contains only binary normal/defective folders and no per-image severity/type table or pixel masks.",
                          "ETID": "6,269 category-labeled e-waste product photos across battery, cable, keyboard, laptop, mobile phone, mouse, and PCB. Source groups are held out globally across train/validation/test. ETID is category-only: it does not label defects and is excluded from normal anomaly references and defect training.",
                          "Plastic Shampoo Bottle": "Official bottle defect image splits are used to teach a separate plastic_shampoo_bottle product category; these photos include defects and are not treated as normal references. The bottle-specific YOLO detector alone learns the defect boxes/types/grades."}
    }
    # Keep independently trained defect stages and their metrics when the
    # product-category MLP is retrained.
    for key in ("defect_segmentation", "damage_type_head", "bottle_damage_grade_head",
                "bottle_yolo_detector"):
        if key in prior_report:
            combined_metrics[key] = prior_report[key]
    write_json(report_path, combined_metrics)
    print("Saved models and metrics.", flush=True)
    print(json.dumps({"product_accuracy": accuracy_score(val_y, product_pred),
                      "product_macro_f1": product_report["macro avg"]["f1-score"],
                      "ETID_test_accuracy": etid_test_metrics["category_accuracy"],
                      "ETID_test_macro_f1": etid_test_metrics["macro_f1"],
                      "ETID_cable_accuracy": etid_test_metrics["cable_accuracy"],
                      "neu_test_accuracy_retained": neu_metrics.get("accuracy"),
                      "category_count": len(all_categories),
                      "neu_class_count": len(neu_metrics.get("classes", []))}, indent=2), flush=True)


if __name__ == "__main__":
    main()
