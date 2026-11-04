"""Multi-Model Industrial AI Inspection PDF Report Generator."""
from __future__ import annotations

import io
from typing import Optional

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
    recognition_confidence: Optional[float] = None,
    yolo_defect_count: int = 0,
    yolo_annotated_rgb: Optional[np.ndarray] = None,
) -> bytes:
    """Create an industrial quality inspection PDF certificate."""
    width, height = 1240, 1754
    page = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(page)
    title_font = _font(36)
    heading_font = _font(22)
    body_font = _font(19)
    bold_font = _font(20)
    small_font = _font(16)

    margin = 60
    # Header Banner
    draw.rounded_rectangle((margin, 40, width - margin, 120), radius=10, fill="#0f172a")
    draw.text((margin + 24, 56), "🛡️ SMART INDUSTRIAL QUALITY INSPECTION REPORT", fill="#ffffff", font=title_font)

    status_color = "#dc2626" if status == "DEFECTIVE" or "NOT" in status else "#16a34a"
    status_bg = "#fef2f2" if status == "DEFECTIVE" or "NOT" in status else "#f0fdf4"
    verdict_text = "❌ NOT APPROVED TO SELL" if status == "DEFECTIVE" or "NOT" in status else "✅ APPROVED TO SELL"

    # Verdict Box
    draw.rounded_rectangle((margin, 135, width - margin, 215), radius=12, fill=status_bg, outline=status_color, width=3)
    draw.text((margin + 30, 155), f"COMMERCIAL VERDICT: {verdict_text}", fill=status_color, font=title_font)

    conf_str = f" ({recognition_confidence:.1%} confidence)" if recognition_confidence else ""
    rows = [
        ("Inspection ID", inspection_id),
        ("Timestamp (UTC)", created_at),
        ("Model 1 (Recognition)", f"{category.replace('_', ' ').title()}{conf_str}"),
        ("Model 2 (Anomaly Score)", f"{anomaly_score:.5f}"),
        ("YOLO Defect Localizer", f"{yolo_defect_count} defect instance(s) detected"),
        ("Inspection Latency", f"{processing_seconds:.3f} seconds"),
    ]

    y = 235
    for label, value in rows:
        draw.text((margin, y), f"{label}:", fill="#475467", font=bold_font)
        draw.text((margin + 320, y), str(value), fill="#101828", font=body_font)
        y += 38

    # Panels: Original, Model 2 Heatmap, YOLO Localizer
    panels = [
        ("1. Original Part", original_rgb),
        ("2. Model 2 Anomaly Heatmap", heatmap_rgb),
    ]
    if yolo_annotated_rgb is not None:
        panels.append(("3. YOLO Defect Localizer", yolo_annotated_rgb))

    num_panels = len(panels)
    gap = 24
    total_gaps = (num_panels - 1) * gap
    panel_width = (width - 2 * margin - total_gaps) // num_panels
    panel_height = 800
    image_top = 480

    for idx, (heading, array) in enumerate(panels):
        px = margin + idx * (panel_width + gap)
        draw.text((px, image_top), heading, fill="#0f172a", font=heading_font)
        
        image = Image.fromarray(np.asarray(array, dtype=np.uint8).copy(), mode="RGB")
        fitted = _fit(image, (panel_width, panel_height))
        image_x = px + (panel_width - fitted.width) // 2
        
        draw.rounded_rectangle(
            (px, image_top + 36, px + panel_width, image_top + panel_height + 40),
            radius=10,
            outline="#cbd5e1",
            width=2,
        )
        page.paste(fitted, (image_x, image_top + 42 + (panel_height - fitted.height) // 2))

    draw.line((margin, height - 90, width - margin, height - 90), fill="#cbd5e1", width=2)
    draw.text(
        (margin, height - 65),
        "Certified Multi-Model AI Quality Assessment Pipeline: Model 1 (Recognition) + Model 2 (PatchCore) + YOLOv11 Defect Localizer.",
        fill="#64748b",
        font=small_font,
    )

    output = io.BytesIO()
    page.save(output, format="PDF", resolution=150.0)
    return output.getvalue()
