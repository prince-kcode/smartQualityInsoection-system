#!/usr/bin/env python3
"""Transfer Learning Training Pipeline for Industrial Product Recognition.

Architecture: EfficientNet-B0 with two-stage fine-tuning
Classes: charger_cable, electrical_cable, metal_spoon, plastic_bottle, wooden_spoon
Input Size: 224x224
Framework: PyTorch / torchvision
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Ensure Matplotlib config dir is local to avoid warnings
os.environ["MPLCONFIGDIR"] = str(Path(__file__).resolve().parents[1] / ".mpl_cache")
Path("reports").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)
Path(".mpl_cache").mkdir(exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "dataset_products_split"
MODEL_SAVE_PATH = PROJECT_ROOT / "models" / "product_classifier_efficientnet.pt"
REPORTS_DIR = PROJECT_ROOT / "reports"

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, val_test_transform


def build_model(num_classes: int) -> nn.Module:
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, 256),
        nn.SiLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes),
    )
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=-1)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels), np.array(all_probs)


def plot_training_curves(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"], "r--s", label="Val Loss", linewidth=2)
    ax1.set_title("Cross-Entropy Loss Curve", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], "b-o", label="Train Acc", linewidth=2)
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], "g--s", label="Val Acc", linewidth=2)
    ax2.set_title("Classification Accuracy Curve", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f" Saved training curves to: {save_path}")


def plot_confusion_matrix(cm, class_names, save_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Industrial Product Recognition Confusion Matrix (Test Set)",
        ylabel="Ground Truth Class",
        xlabel="Predicted Class",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]:d}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold",
                fontsize=11,
            )

    fig.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f" Saved confusion matrix to: {save_path}")


def main():
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    print("=" * 70, flush=True)
    print(" INDUSTRIAL PRODUCT RECOGNITION - EFFICIENTNET-B0 TRAINING")
    print("=" * 70)
    print(f" PyTorch Version : {torch.__version__}")
    print(f" Compute Device  : {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f" Dataset Path    : {DATASET_DIR}")

    train_tf, val_test_tf = get_transforms()

    train_dataset = datasets.ImageFolder(str(DATASET_DIR / "train"), transform=train_tf)
    val_dataset = datasets.ImageFolder(str(DATASET_DIR / "val"), transform=val_test_tf)
    test_dataset = datasets.ImageFolder(str(DATASET_DIR / "test"), transform=val_test_tf)

    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f" Number of Classes: {num_classes}")
    for idx, cname in enumerate(class_names):
        print(f"   [{idx}] {cname}: {train_dataset.targets.count(idx)} train / {val_dataset.targets.count(idx)} val / {test_dataset.targets.count(idx)} test")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Calculate class weights for balanced gradient backprop
    class_counts = [train_dataset.targets.count(i) for i in range(num_classes)]
    total_samples = sum(class_counts)
    class_weights = torch.tensor(
        [total_samples / (num_classes * c) for c in class_counts],
        dtype=torch.float32,
        device=DEVICE,
    )
    print(f" Balanced Class Weights: {[round(w.item(), 3) for w in class_weights]}")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    model = build_model(num_classes).to(DEVICE)

    # -------------------------------------------------------------
    # STAGE 1: Train Head with Frozen Backbone (8 epochs)
    # -------------------------------------------------------------
    print("\n--- STAGE 1: Feature Backbone Frozen, Training Head (8 Epochs) ---")
    for param in model.features.parameters():
        param.requires_grad = False

    optimizer_head = AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_val_acc = 0.0

    stage1_epochs = 8
    for epoch in range(1, stage1_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer_head, DEVICE)
        va_loss, va_acc, _, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        print(f"Epoch {epoch:02d}/{stage1_epochs:02d} [{elapsed:.1f}s] - Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc:.1%} | Val Loss: {va_loss:.4f}, Val Acc: {va_acc:.1%}")

    # -------------------------------------------------------------
    # STAGE 2: Fine-Tuning Top Blocks with Lower LR (15 epochs)
    # -------------------------------------------------------------
    print("\n--- STAGE 2: Unfreezing Top Conv Blocks (features[5:]) & Fine-Tuning ---")
    for i, block in enumerate(model.features):
        if i >= 5:  # Unfreeze higher-level semantic features
            for param in block.parameters():
                param.requires_grad = True
        else:
            for param in block.parameters():
                param.requires_grad = False

    optimizer_ft = AdamW([
        {"params": [p for i, block in enumerate(model.features) if i >= 5 for p in block.parameters()], "lr": 1e-4},
        {"params": model.classifier.parameters(), "lr": 3e-4},
    ], weight_decay=1e-4)

    stage2_epochs = 15
    scheduler = CosineAnnealingLR(optimizer_ft, T_max=stage2_epochs, eta_min=1e-6)

    best_state_dict = None
    patience = 6
    patience_counter = 0

    for epoch in range(1, stage2_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer_ft, DEVICE)
        va_loss, va_acc, _, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        is_best = False
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_val_acc = va_acc
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            is_best = True
            patience_counter = 0
        else:
            patience_counter += 1

        marker = " [BEST MODEL SAVED]" if is_best else ""
        print(f"FT-Epoch {epoch:02d}/{stage2_epochs:02d} [{elapsed:.1f}s] - Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc:.1%} | Val Loss: {va_loss:.4f}, Val Acc: {va_acc:.1%}{marker}")

        if patience_counter >= patience:
            print(f"Early stopping triggered at FT-Epoch {epoch}")
            break

    # Restore best checkpoint
    if best_state_dict is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state_dict.items()})
        print(f"\n Restored best validation checkpoint (Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc:.1%})")

    # -------------------------------------------------------------
    # Save Model Checkpoint
    # -------------------------------------------------------------
    save_payload = {
        "architecture": "efficientnet_b0",
        "state_dict": model.state_dict(),
        "classes": class_names,
        "image_size": IMAGE_SIZE,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "num_classes": num_classes,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
    }
    torch.save(save_payload, MODEL_SAVE_PATH)
    file_size_mb = os.path.getsize(MODEL_SAVE_PATH) / (1024 * 1024)
    print(f" Saved full model checkpoint to: {MODEL_SAVE_PATH} ({file_size_mb:.2f} MB)")

    # -------------------------------------------------------------
    # STEP 4: Evaluation on Unseen Test Set
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print(" EVALUATION ON UNSEEN TEST SET (98 Images)")
    print("=" * 70)

    test_loss, test_acc, test_preds, test_labels, test_probs = evaluate(model, test_loader, criterion, DEVICE)
    print(f" Test Accuracy : {test_acc:.2%}")
    print(f" Test Loss     : {test_loss:.4f}")

    # Classification Report
    report_dict = classification_report(test_labels, test_preds, target_names=class_names, output_dict=True)
    report_str = classification_report(test_labels, test_preds, target_names=class_names, digits=4)
    print("\nClassification Report:\n")
    print(report_str)

    # Confusion Matrix
    cm = confusion_matrix(test_labels, test_preds)
    print("Confusion Matrix:")
    print(cm)

    # Plot curves and matrix
    curves_path = REPORTS_DIR / "product_classifier_training_curves.png"
    cm_path = REPORTS_DIR / "product_classifier_confusion_matrix.png"
    plot_training_curves(history, curves_path)
    plot_confusion_matrix(cm, class_names, cm_path)

    # Save metrics JSON
    metrics_report = {
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "val_best_accuracy": float(best_val_acc),
        "val_best_loss": float(best_val_loss),
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "classes": class_names,
    }
    report_json_path = REPORTS_DIR / "product_classifier_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=2)
    print(f" Saved metrics JSON to: {report_json_path}")

    # Check for any misclassified test samples
    misclassified = []
    for idx, (p, g) in enumerate(zip(test_preds, test_labels)):
        if p != g:
            img_path = test_dataset.samples[idx][0]
            conf = float(test_probs[idx][p])
            misclassified.append({
                "path": img_path,
                "ground_truth": class_names[g],
                "predicted": class_names[p],
                "confidence": conf,
            })

    if misclassified:
        print(f"\n Misclassified Test Samples ({len(misclassified)} total):")
        for m in misclassified:
            print(f"   Image: {os.path.basename(m['path'])} | Truth: {m['ground_truth']} -> Pred: {m['predicted']} (Conf: {m['confidence']:.1%})")
    else:
        print("\n Perfect 100% test set classification! Zero errors across all test specimens.")

    print("\n Training & Evaluation Pipeline Completed Successfully!")


if __name__ == "__main__":
    main()
