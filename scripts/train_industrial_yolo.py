"""Prepare MVTec AD defect bounding boxes and train a balanced industrial YOLO defect detector."""
from __future__ import annotations

import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "mvtec_ad"
OUTPUT_DIR = ROOT / "data" / "processed" / "mvtec_yolo"
MODELS_DIR = ROOT / "models"
TARGET_MODEL = MODELS_DIR / "industrial_defect_yolo.pt"
PRETRAINED_YOLO = ROOT / "industrialguard_mlp" / "yolo11n.pt"


def prepare_dataset():
    print("Preparing balanced YOLO dataset from MVTec AD...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for split in ["train", "val"]:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    random.seed(42)
    categories = sorted([d.name for d in RAW_DIR.iterdir() if d.is_dir()])
    
    train_count = 0
    val_count = 0

    for cat in categories:
        cat_dir = RAW_DIR / cat
        gt_dir = cat_dir / "ground_truth"
        test_dir = cat_dir / "test"
        train_dir = cat_dir / "train"
        
        # 1. Defect images
        if gt_dir.is_dir():
            defect_types = sorted([d.name for d in gt_dir.iterdir() if d.is_dir()])
            for dtype in defect_types:
                masks = sorted(list((gt_dir / dtype).glob("*_mask.png")))
                for mask_path in masks:
                    img_name = mask_path.name.replace("_mask.png", ".png")
                    img_path = test_dir / dtype / img_name
                    if not img_path.exists():
                        continue
                    
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        continue
                    h, w = mask.shape
                    
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8))
                    
                    bboxes = []
                    for i in range(1, num_labels):
                        bx, by, bw, bh, area = stats[i]
                        if area < 15:  # filter noise
                            continue
                        
                        x_center = (bx + bw / 2.0) / w
                        y_center = (by + bh / 2.0) / h
                        norm_w = bw / float(w)
                        norm_h = bh / float(h)
                        
                        x_center = min(max(x_center, 0.0), 1.0)
                        y_center = min(max(y_center, 0.0), 1.0)
                        norm_w = min(max(norm_w, 0.001), 1.0)
                        norm_h = min(max(norm_h, 0.001), 1.0)
                        
                        bboxes.append(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
                    
                    split = "val" if random.random() < 0.15 else "train"
                    dest_stem = f"{cat}_{dtype}_{img_path.stem}"
                    dest_img = OUTPUT_DIR / "images" / split / f"{dest_stem}.png"
                    dest_lbl = OUTPUT_DIR / "labels" / split / f"{dest_stem}.txt"
                    
                    shutil.copy2(img_path, dest_img)
                    dest_lbl.write_text("\n".join(bboxes), encoding="utf-8")
                    
                    if split == "train":
                        train_count += 1
                    else:
                        val_count += 1

        # 2. Test Good images (negative samples)
        good_dir = test_dir / "good"
        if good_dir.is_dir():
            good_imgs = sorted(list(good_dir.glob("*.png")))
            for img_path in good_imgs:
                split = "val" if random.random() < 0.15 else "train"
                dest_stem = f"{cat}_good_test_{img_path.stem}"
                dest_img = OUTPUT_DIR / "images" / split / f"{dest_stem}.png"
                dest_lbl = OUTPUT_DIR / "labels" / split / f"{dest_stem}.txt"
                
                shutil.copy2(img_path, dest_img)
                dest_lbl.write_text("", encoding="utf-8")
                
                if split == "train":
                    train_count += 1
                else:
                    val_count += 1

        # 3. Train Good images (negative samples, up to 40 per category for balance)
        train_good_dir = train_dir / "good"
        if train_good_dir.is_dir():
            train_good_imgs = sorted(list(train_good_dir.glob("*.png")))
            selected_train_goods = train_good_imgs[:40]
            for img_path in selected_train_goods:
                split = "val" if random.random() < 0.15 else "train"
                dest_stem = f"{cat}_good_train_{img_path.stem}"
                dest_img = OUTPUT_DIR / "images" / split / f"{dest_stem}.png"
                dest_lbl = OUTPUT_DIR / "labels" / split / f"{dest_stem}.txt"
                
                shutil.copy2(img_path, dest_img)
                dest_lbl.write_text("", encoding="utf-8")
                
                if split == "train":
                    train_count += 1
                else:
                    val_count += 1

    print(f"Balanced Dataset prepared! Train images: {train_count}, Val images: {val_count}")

    yaml_content = {
        "path": str(OUTPUT_DIR.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "defect"}
    }
    yaml_path = OUTPUT_DIR / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    
    return yaml_path


def train_yolo(yaml_path: Path):
    print("Starting balanced YOLO training on CUDA RTX 4060...")
    import torch
    device = 0 if torch.cuda.is_available() else "cpu"
    
    weights = str(PRETRAINED_YOLO) if PRETRAINED_YOLO.exists() else "yolo11n.pt"
    model = YOLO(weights)
    
    train_res = model.train(
        data=str(yaml_path),
        epochs=15,
        imgsz=640,
        batch=16,
        device=device,
        workers=2,
        project=str(ROOT / "models" / "yolo_runs"),
        name="industrial_defect_balanced",
        exist_ok=True,
        seed=42,
        amp=True,
        verbose=True,
    )
    
    best_pt = Path(model.trainer.best)
    if best_pt.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_pt, TARGET_MODEL)
        print(f"SUCCESS! Balanced Industrial Defect YOLO saved to: {TARGET_MODEL} ({TARGET_MODEL.stat().st_size / (1024*1024):.2f} MB)")
    else:
        last_pt = Path(model.trainer.last)
        if last_pt.exists():
            shutil.copy2(last_pt, TARGET_MODEL)


if __name__ == "__main__":
    yaml_p = prepare_dataset()
    train_yolo(yaml_p)
