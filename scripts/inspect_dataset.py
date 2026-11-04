"""Inspect the local MVTec AD folder layout and report mask coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from industrialguard.data.mvtec import MVTEC_CATEGORIES, audit_categories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/raw/mvtec_ad"),
        help="MVTec AD root containing category folders (default: %(default)s)",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=list(MVTEC_CATEGORIES),
        help="Category folder names to inspect (default: all 15 MVTec AD categories)",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.root.is_dir():
        print(f"Dataset root not found: {args.root.resolve()}")
        print("Place the extracted MVTec AD dataset at the selected root and retry.")
        return 2

    results = audit_categories(args.root, args.categories)
    if args.json:
        print(json.dumps([result.to_dict() for result in results], indent=2))
    else:
        for result in results:
            print(f"\n{result.category}: {'complete' if result.is_complete else 'needs attention'}")
            print(f"  defect-free train images: {result.train_good}")
            print(f"  defect-free test images:  {result.test_good}")
            print(f"  defective test images:    {sum(result.defect_images.values())}")
            for defect, count in result.defect_images.items():
                print(f"    {defect}: {count} images, {result.defect_masks.get(defect, 0)} masks")
            if result.missing_directories:
                print(f"  missing directories: {', '.join(result.missing_directories)}")
            if result.missing_masks:
                print(f"  images without masks: {len(result.missing_masks)}")
                for relative_path in result.missing_masks[:10]:
                    print(f"    {relative_path}")
                if len(result.missing_masks) > 10:
                    print(f"    ... and {len(result.missing_masks) - 10} more")
            if result.unexpected_mask_groups:
                print(f"  masks without matching defect folder: {', '.join(result.unexpected_mask_groups)}")

    return 0 if all(result.is_complete for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
