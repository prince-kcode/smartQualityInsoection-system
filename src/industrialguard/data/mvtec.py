"""Audit a local MVTec AD category without changing its files."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
MVTEC_CATEGORIES = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
)


@dataclass(frozen=True)
class CategoryAudit:
    """Counts and structural issues for one MVTec AD category."""

    category: str
    train_good: int
    test_good: int
    defect_images: dict[str, int]
    defect_masks: dict[str, int]
    missing_masks: list[str]
    unexpected_mask_groups: list[str]
    missing_directories: list[str]

    @property
    def is_complete(self) -> bool:
        """Return whether the category has the expected folders and masks."""

        return not (
            self.missing_directories or self.missing_masks or self.unexpected_mask_groups
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly representation including completeness."""

        result = asdict(self)
        result["is_complete"] = self.is_complete
        return result


def _image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def audit_category(category_root: Path, category: str | None = None) -> CategoryAudit:
    """Count MVTec images and check masks for each defective test image.

    The standard category layout contains ``train/good``, ``test/good``,
    ``test/<defect>`` and ``ground_truth/<defect>`` directories. Defect masks
    are matched to test images by filename stem, independent of image suffix.
    """

    category_root = Path(category_root)
    category_name = category or category_root.name
    required = ["train/good", "test/good", "ground_truth"]
    missing_directories = [
        rel for rel in required if not (category_root / rel).is_dir()
    ]

    test_root = category_root / "test"
    ground_truth_root = category_root / "ground_truth"
    defect_dirs = sorted(
        folder
        for folder in test_root.iterdir()
        if folder.is_dir() and folder.name.lower() != "good"
    ) if test_root.is_dir() else []
    mask_dirs = {
        folder.name: folder
        for folder in ground_truth_root.iterdir() if folder.is_dir()
    } if ground_truth_root.is_dir() else {}

    defect_images = {
        folder.name: len(_image_files(folder))
        for folder in defect_dirs
    }
    defect_masks = {
        name: len(_image_files(folder))
        for name, folder in sorted(mask_dirs.items())
    }

    missing_masks: list[str] = []
    for defect_dir in defect_dirs:
        mask_dir = mask_dirs.get(defect_dir.name)
        mask_stems = {
            mask.stem.removesuffix("_mask")
            for mask in _image_files(mask_dir)
        } if mask_dir else set()
        for image in _image_files(defect_dir):
            if image.stem not in mask_stems:
                missing_masks.append(f"{defect_dir.name}/{image.name}")

    defect_names = {folder.name for folder in defect_dirs}
    unexpected_mask_groups = sorted(set(mask_dirs) - defect_names)

    return CategoryAudit(
        category=category_name,
        train_good=len(_image_files(category_root / "train" / "good")),
        test_good=len(_image_files(category_root / "test" / "good")),
        defect_images=defect_images,
        defect_masks=defect_masks,
        missing_masks=missing_masks,
        unexpected_mask_groups=unexpected_mask_groups,
        missing_directories=missing_directories,
    )


def audit_categories(dataset_root: Path, categories: list[str]) -> list[CategoryAudit]:
    """Audit selected categories under a dataset root."""

    root = Path(dataset_root)
    return [audit_category(root / category, category) for category in categories]

