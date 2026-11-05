"""Automated dataset collector for industrial product identification.

Downloads and curates clean, diverse real-world images for:
- plastic_bottle
- electrical_cable
- charger_cable
- wooden_spoon
- metal_spoon
"""
from __future__ import annotations

import hashlib
import json
import ssl
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from PIL import Image
import io

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET_DIR = PROJECT_ROOT / "data" / "dataset_products"
PBD_DIR = PROJECT_ROOT / "data" / "raw" / "external" / "pbd" / "PBD_Dataset"
MVTEC_DIR = PROJECT_ROOT / "data" / "raw" / "mvtec_ad"

CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE

HEADERS = {
    "User-Agent": "IndustrialProductClassifier/2.0 (research@industrialguard.ai)",
}

CATEGORIES_CONFIG = {
    "wooden_spoon": [
        "wooden spoon",
        "wooden spoons kitchen",
        "wood spoon tableware",
        "wooden ladle spoon",
        "cuillere en bois",
        "holzlöffel",
    ],
    "metal_spoon": [
        "silver spoon cutlery",
        "stainless steel spoon",
        "tablespoon cutlery",
        "teaspoon metal",
        "soup spoon silverware",
        "metal spoon table",
    ],
    "charger_cable": [
        "usb charger cable",
        "usb c cable",
        "charging cable phone",
        "lightning cable",
        "micro usb cable",
        "usb wire cord",
    ],
    "electrical_cable": [
        "electrical power cable",
        "electric wire cord",
        "heavy duty power cable",
        "industrial electrical wire",
        "extension cord cable",
    ],
}


def fetch_wikimedia_urls(query: str, limit: int = 50) -> list[str]:
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": 6,
        "gsrlimit": limit,
        "prop": "imageinfo",
        "iiprop": "url|mime|size",
    }
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=20, context=CTX) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        pages = data.get("query", {}).get("pages", {})
        results = []
        for pid, pdata in pages.items():
            info = pdata.get("imageinfo", [{}])[0]
            u = info.get("url")
            mime = info.get("mime", "")
            size = info.get("size", 0)
            if u and ("jpeg" in mime or "png" in mime or "jpg" in mime) and size > 15000:
                results.append(u)
        return results
    except Exception as e:
        print(f"  [Warning] Query '{query}' failed: {e}")
        return []


def download_and_save_image(url: str, output_path: Path) -> bool:
    if output_path.exists():
        return True
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=15, context=CTX) as resp:
            content = resp.read()
        with Image.open(io.BytesIO(content)) as img:
            # Verify valid RGB image
            rgb = img.convert("RGB")
            w, h = rgb.size
            if w < 100 or h < 100:
                return False
            # Resize maximum dimension to 800px to maintain quality without massive disk bloat
            rgb.thumbnail((800, 800), Image.Resampling.LANCZOS)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            rgb.save(output_path, "JPEG", quality=90)
        return True
    except Exception:
        return False


def collect_local_plastic_bottles(target_folder: Path, count: int = 150) -> int:
    target_folder.mkdir(parents=True, exist_ok=True)
    saved = 0
    if PBD_DIR.exists():
        # Mix body, shoulder, and natural variations
        body_imgs = list((PBD_DIR / "body" / "train_images").glob("*.jpg"))
        shoulder_imgs = list((PBD_DIR / "shoulder" / "train_images").glob("*.jpg"))
        selected = body_imgs[:count // 2] + shoulder_imgs[:count // 2]
        for idx, src in enumerate(selected):
            dst = target_folder / f"pbd_bottle_{idx:03d}.jpg"
            if not dst.exists():
                try:
                    with Image.open(src) as im:
                        im.convert("RGB").save(dst, "JPEG", quality=90)
                    saved += 1
                except Exception:
                    pass
            else:
                saved += 1
    # Also add MVTec normal bottles if available
    mvtec_bottle = MVTEC_DIR / "bottle" / "train" / "good"
    if mvtec_bottle.exists():
        for idx, src in enumerate(list(mvtec_bottle.glob("*.png"))[:30]):
            dst = target_folder / f"mvtec_bottle_{idx:03d}.jpg"
            if not dst.exists():
                try:
                    with Image.open(src) as im:
                        im.convert("RGB").save(dst, "JPEG", quality=90)
                    saved += 1
                except Exception:
                    pass
            else:
                saved += 1
    return saved


def collect_local_electrical_cables(target_folder: Path, count: int = 50) -> int:
    target_folder.mkdir(parents=True, exist_ok=True)
    saved = 0
    mvtec_cable = MVTEC_DIR / "cable" / "train" / "good"
    if mvtec_cable.exists():
        for idx, src in enumerate(list(mvtec_cable.glob("*.png"))[:count]):
            dst = target_folder / f"mvtec_cable_{idx:03d}.jpg"
            if not dst.exists():
                try:
                    with Image.open(src) as im:
                        im.convert("RGB").save(dst, "JPEG", quality=90)
                    saved += 1
                except Exception:
                    pass
            else:
                saved += 1
    return saved


def main():
    print("=" * 70)
    print("INDUSTRIAL PRODUCT DATASET BUILDER")
    print("Building balanced dataset across 5 classes:")
    print("  1. plastic_bottle")
    print("  2. electrical_cable")
    print("  3. charger_cable")
    print("  4. wooden_spoon")
    print("  5. metal_spoon")
    print("=" * 70)

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Plastic Bottles (Local PBD)
    print("\n[Class 1/5] Assembling 'plastic_bottle' from local PBD dataset...")
    bottle_dir = TARGET_DIR / "plastic_bottle"
    n_bottles = collect_local_plastic_bottles(bottle_dir, count=150)
    print(f"  -> Total plastic_bottle images: {len(list(bottle_dir.glob('*.jpg')))}")

    # 2. Electrical Cables (Base MVTec + Online extension)
    print("\n[Class 2/5] Assembling 'electrical_cable'...")
    cable_dir = TARGET_DIR / "electrical_cable"
    n_cables = collect_local_electrical_cables(cable_dir, count=50)
    print(f"  -> Added {n_cables} base cable images from MVTec AD.")

    # 3. Download Web Images for missing categories
    for cat, queries in CATEGORIES_CONFIG.items():
        cat_dir = TARGET_DIR / cat
        cat_dir.mkdir(parents=True, exist_ok=True)
        existing = len(list(cat_dir.glob("*.jpg")))
        print(f"\n[Collecting '{cat}'] Current count: {existing} images")
        
        target_count = 120
        needed = target_count - existing
        if needed <= 0:
            print(f"  -> '{cat}' already has {existing} images. Skipping.")
            continue

        seen_urls = set()
        img_idx = existing
        for q in queries:
            if img_idx >= target_count:
                break
            print(f"  Searching Commons for '{q}'...")
            urls = fetch_wikimedia_urls(q, limit=40)
            for u in urls:
                if u in seen_urls:
                    continue
                seen_urls.add(u)
                out_p = cat_dir / f"{cat}_{img_idx:03d}.jpg"
                if download_and_save_image(u, out_p):
                    img_idx += 1
                    print(f"    Downloaded [{img_idx}/{target_count}]: {out_p.name}")
                    time.sleep(0.15)
                if img_idx >= target_count:
                    break

        print(f"  -> '{cat}' final count: {len(list(cat_dir.glob('*.jpg')))}")

    # Final Summary
    print("\n" + "=" * 70)
    print("DATASET COLLECTION COMPLETE")
    print("Summary of collected images:")
    for d in sorted(TARGET_DIR.iterdir()):
        if d.is_dir():
            count = len(list(d.glob("*.jpg"))) + len(list(d.glob("*.png")))
            print(f"  - {d.name:20s}: {count:4d} images")
    print("=" * 70)


if __name__ == "__main__":
    main()

