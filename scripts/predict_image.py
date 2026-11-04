"""Predict anomalies in an image using a trained category checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from industrialguard.inference.predictor import predict_image
from industrialguard.inference.predictor import SUPPORTED_CATEGORIES
from industrialguard.visualization.heatmap import make_heatmap, overlay_heatmap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Image file to inspect")
    parser.add_argument(
        "--category",
        choices=SUPPORTED_CATEGORIES,
        required=True,
        help="Product category selects its own trained PatchCore checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/predictions"),
        help="Directory for generated heatmap and overlay files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with Image.open(args.image) as opened:
        original = np.asarray(opened.convert("RGB"), dtype=np.uint8)

    result = predict_image(args.image, args.category)
    heatmap = make_heatmap(result.anomaly_map, (original.shape[1], original.shape[0]))
    overlay = overlay_heatmap(original, heatmap)

    output_dir = args.output_dir / args.category
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    Image.fromarray(heatmap).save(output_dir / f"{stem}_heatmap.png")
    Image.fromarray(overlay).save(output_dir / f"{stem}_overlay.png")

    status = "DEFECTIVE" if result.is_anomalous else "NORMAL"
    print(f"category: {result.category}")
    print(f"status: {status}")
    print(f"anomaly score: {result.anomaly_score:.6f} (not a probability)")
    print(f"processing time: {result.processing_seconds:.3f} s")
    print(f"heatmap: {(output_dir / f'{stem}_heatmap.png').resolve()}")
    print(f"overlay: {(output_dir / f'{stem}_overlay.png').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
