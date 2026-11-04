"""Render anomaly maps as heatmaps and RGB overlays."""

from __future__ import annotations

import cv2
import numpy as np


def make_heatmap(anomaly_map: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    """Convert a numeric anomaly map to an RGB heatmap of ``(width, height)``."""

    anomaly_map = np.asarray(anomaly_map, dtype=np.float32)
    if anomaly_map.ndim != 2 or anomaly_map.size == 0:
        raise ValueError("anomaly_map must be a non-empty 2D array.")
    width, height = output_size
    if width < 1 or height < 1:
        raise ValueError("output_size must contain positive width and height.")
    if not np.isfinite(anomaly_map).all():
        raise ValueError("anomaly_map contains non-finite values.")

    resized = cv2.resize(anomaly_map, (width, height), interpolation=cv2.INTER_CUBIC)
    low = float(resized.min())
    high = float(resized.max())
    if high > low:
        normalized = np.clip((resized - low) / (high - low), 0, 1)
    else:
        normalized = np.zeros_like(resized)
    color_bgr = cv2.applyColorMap(np.round(normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)


def overlay_heatmap(
    image_rgb: np.ndarray,
    heatmap_rgb: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend an RGB heatmap over an RGB image."""

    image = np.asarray(image_rgb)
    heatmap = np.asarray(heatmap_rgb)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image_rgb must be uint8 RGB data.")
    if heatmap.dtype != np.uint8 or heatmap.ndim != 3 or heatmap.shape[2] != 3:
        raise ValueError("heatmap_rgb must be uint8 RGB data.")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1.")
    if heatmap.shape[:2] != image.shape[:2]:
        heatmap = cv2.resize(
            heatmap,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    blended = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return np.asarray(blended, dtype=np.uint8)
