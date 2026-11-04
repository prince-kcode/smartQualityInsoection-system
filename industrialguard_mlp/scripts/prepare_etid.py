"""Safely extract and group-split the public E-Tadweer image dataset."""
from __future__ import annotations

import csv
import hashlib
import re
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "datasets" / "additional" / "ETID.zip"
DEST = ROOT / "datasets" / "additional" / "etid"
MANIFEST = DEST / "manifest.csv"
EXPECTED_SHA256 = "518d1f20407f23ae4a09b7cd9da30877799e815907fc602c43ba25c214bc9b63"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
CLASS_MAP = {
    "Battery": "battery",
    "Cable": "cable",
    "Keyboard": "keyboard",
    "Laptop": "laptop",
    "Mobile": "mobile_phone",
    "Mouse": "mouse",
    "PCB": "printed_circuit_board",
}
SOURCE_GROUP = re.compile(r"^(ewaste-mixed-\d+|ewaste-\d+|image_\d{8})", re.IGNORECASE)
TEST_GROUPS = {"ewaste-13", "ewaste-18", "ewaste-8", "ewaste-9", "ewaste-mixed-7",
               "ewaste-mixed-20", "image_20260217"}
VALIDATION_GROUPS = {"ewaste-6", "ewaste-7", "ewaste-mixed-8", "ewaste-mixed-22",
                     "image_20260121"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_group(filename: str) -> str:
    stem = Path(filename).stem
    match = SOURCE_GROUP.match(stem)
    return (match.group(0) if match else stem.split("_")[0]).lower()


def split_groups(rows: list[dict]) -> None:
    all_groups = {row["source_group"] for row in rows}
    if TEST_GROUPS & VALIDATION_GROUPS or not TEST_GROUPS <= all_groups or not VALIDATION_GROUPS <= all_groups:
        raise RuntimeError("ETID source-group split definitions do not match the archive contents")
    for row in rows:
        group = row["source_group"]
        row["split"] = "test" if group in TEST_GROUPS else "validation" if group in VALIDATION_GROUPS else "train"
    for category in CLASS_MAP.values():
        splits = {row["split"] for row in rows if row["category"] == category}
        if splits != {"train", "validation", "test"}:
            raise RuntimeError(f"ETID {category} does not have images in all three group-held-out splits")


def main() -> None:
    if not ARCHIVE.is_file():
        raise SystemExit(f"Missing ETID archive: {ARCHIVE}")
    actual_hash = file_sha256(ARCHIVE)
    if actual_hash != EXPECTED_SHA256:
        raise SystemExit(f"ETID SHA-256 mismatch: expected {EXPECTED_SHA256}, got {actual_hash}")

    DEST.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    expected_root = "E-Tadweer Image Dataset (ETID)"
    with zipfile.ZipFile(ARCHIVE) as archive:
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            if path.is_absolute() or ".." in path.parts:
                raise RuntimeError(f"Unsafe archive path: {info.filename}")
            if info.is_dir() or len(path.parts) != 3 or path.parts[0] != expected_root:
                continue
            source_class, filename = path.parts[1], path.name
            if source_class not in CLASS_MAP or filename.startswith("._") or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            category = CLASS_MAP[source_class]
            output = DEST / category / filename
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists():
                if output.stat().st_size != info.file_size:
                    raise RuntimeError(f"Existing ETID image size differs from the archive: {output}")
            else:
                with archive.open(info) as source, output.open("wb") as target:
                    target.write(source.read())
            rows.append({"path": output.relative_to(ROOT).as_posix(), "category": category,
                         "source_group": source_group(filename), "split": "", "source_class": source_class})

    if len(rows) != 6269:
        raise RuntimeError(f"Expected 6,269 ETID photos after excluding archive metadata; found {len(rows)}")
    split_groups(rows)
    with MANIFEST.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "category", "source_group", "split", "source_class"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["category"], row["path"])))

    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        counts.setdefault(row["category"], {"train": 0, "validation": 0, "test": 0})[row["split"]] += 1
    print(f"Verified archive SHA-256: {actual_hash}")
    print(f"Extracted {len(rows)} images into {DEST}")
    print(f"Group-held-out counts: {counts}")


if __name__ == "__main__":
    main()
