"""Clean, verify, deduplicate, and create stratified 70/15/15 train/val/test splits."""
from __future__ import annotations

import hashlib
import random
import shutil
from collections import defaultdict
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_ROOT / "data" / "dataset_products"
SPLIT_DIR = PROJECT_ROOT / "data" / "dataset_products_split"

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def hash_image(path: Path) -> str:
    """Compute sha256 hash of image pixels to detect exact and near-exact duplicates."""
    with Image.open(path) as img:
        rgb = img.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
        return hashlib.sha256(rgb.tobytes()).hexdigest()


def prepare_dataset_splits():
    random.seed(SEED)
    print("=" * 70)
    print("STEP 2: CLEANING, VERIFYING, AND SPLITTING DATASET (70/15/15)")
    print("=" * 70)

    if SPLIT_DIR.exists():
        shutil.rmtree(SPLIT_DIR)

    categories = sorted([d.name for d in SOURCE_DIR.iterdir() if d.is_dir()])
    print(f"Discovered categories ({len(categories)}): {categories}\n")

    summary = {}
    seen_hashes_global = set()

    for cat in categories:
        cat_src = SOURCE_DIR / cat
        all_files = sorted(list(cat_src.glob("*.jpg")) + list(cat_src.glob("*.png")))
        valid_files = []
        seen_hashes = set()

        for f in all_files:
            try:
                # 1. Verify readable image
                with Image.open(f) as img:
                    img.verify()
                with Image.open(f) as img:
                    w, h = img.size
                    if w < 50 or h < 50:
                        continue
                
                # 2. Check for duplicate pixels
                phash = hash_image(f)
                if phash in seen_hashes or phash in seen_hashes_global:
                    continue
                seen_hashes.add(phash)
                seen_hashes_global.add(phash)
                valid_files.append(f)
            except Exception:
                continue

        # Shuffle deterministically
        random.shuffle(valid_files)

        n_total = len(valid_files)
        n_train = int(n_total * TRAIN_RATIO)
        n_val = int(n_total * VAL_RATIO)
        n_test = n_total - n_train - n_val

        train_files = valid_files[:n_train]
        val_files = valid_files[n_train:n_train + n_val]
        test_files = valid_files[n_train + n_val:]

        # Copy to split folders
        for split_name, flist in [("train", train_files), ("val", val_files), ("test", test_files)]:
            split_cat_dir = SPLIT_DIR / split_name / cat
            split_cat_dir.mkdir(parents=True, exist_ok=True)
            for idx, src_p in enumerate(flist):
                dst_p = split_cat_dir / f"{cat}_{idx:03d}.jpg"
                shutil.copy2(src_p, dst_p)

        summary[cat] = {
            "total_cleaned": n_total,
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files),
        }
        print(f"  {cat:20s}: Total={n_total:3d} | Train={len(train_files):3d} | Val={len(val_files):2d} | Test={len(test_files):2d}")

    print("\n" + "=" * 70)
    print("SPLIT SUMMARY:")
    total_train = sum(s["train"] for s in summary.values())
    total_val = sum(s["val"] for s in summary.values())
    total_test = sum(s["test"] for s in summary.values())
    total_all = sum(s["total_cleaned"] for s in summary.values())
    print(f"Total Dataset: {total_all} cleaned images")
    print(f"  - Train Split (70%): {total_train} images")
    print(f"  - Val Split   (15%): {total_val} images")
    print(f"  - Test Split  (15%): {total_test} images")
    print("=" * 70)
    print(f"Dataset successfully prepared in: {SPLIT_DIR}")


if __name__ == "__main__":
    prepare_dataset_splits()
