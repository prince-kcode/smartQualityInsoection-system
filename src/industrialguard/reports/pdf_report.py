"""Single-page PDF inspection reports built with Pillow."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except OSError:
        return ImageFont.load_default(size=size)


def _fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    fitted = image.copy()
    fitted.thumbnail(size, Image.Resampling.LANCZOS)
    return fitted


def create_inspection_pdf(
    *,
    inspection_id: str,
    created_at: str,
    category: str,
    filename: str,
    status: str,
    anomaly_score: float,
    processing_seconds: float,
    original_rgb: np.ndarray,
    heatmap_rgb: np.ndarray,
) -> bytes:
    """Create a one-page PDF and return its bytes."""

    width, height = 1240, 1754
    page = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(page)
    title_font = _font(38)
    heading_font = _font(25)
    body_font = _font(21)
    small_font = _font(18)

    margin = 72
    draw.text((margin, 56), "IndustrialGuard AI Inspection Report", fill="#172033", font=title_font)
    draw.line((margin, 122, width - margin, 122), fill="#cbd5e1", width=3)

    status_color = "#b42318" if status == "DEFECTIVE" else "#067647"
    rows = [
        ("Inspection ID", inspection_id),
        ("Date and time (UTC)", created_at),
        ("Product category", category.replace("_", " ").title()),
        ("Input image", filename),
        ("Model", "PatchCore with Wide ResNet-50"),
        ("Result", status),
        ("Anomaly score", f"{anomaly_score:.6f} (not a probability)"),
        ("Inference time", f"{processing_seconds:.3f} seconds"),
    ]
    y = 160
    for label, value in rows:
        draw.text((margin, y), f"{label}: ", fill="#475467", font=body_font)
        color = status_color if label == "Result" else "#101828"
        draw.text((margin + 270, y), value, fill=color, font=body_font)
        y += 49

    image_top = 650
    panel_width = (width - 3 * margin) // 2
    panel_height = 760
    for index, (heading, array) in enumerate(
        (("Original image", original_rgb), ("Anomaly heatmap", heatmap_rgb)),
    ):
        x = margin + index * (panel_width + margin)
        draw.text((x, image_top), heading, fill="#172033", font=heading_font)
        image = Image.fromarray(np.asarray(array, dtype=np.uint8).copy(), mode="RGB")
        fitted = _fit(image, (panel_width, panel_height))
        image_x = x + (panel_width - fitted.width) // 2
        draw.rounded_rectangle(
            (x, image_top + 44, x + panel_width, image_top + panel_height + 56),
            radius=12,
            outline="#d0d5dd",
            width=2,
        )
        page.paste(fitted, (image_x, image_top + 50 + (panel_height - fitted.height) // 2))

    draw.line((margin, height - 105, width - margin, height - 105), fill="#cbd5e1", width=2)
    draw.text(
        (margin, height - 78),
        "For review and demonstration; model scores and visualizations require human validation.",
        fill="#667085",
        font=small_font,
    )

    output = io.BytesIO()
    page.save(output, format="PDF", resolution=150.0)
    return output.getvalue()
