"""Reproducible image-processing views for uploaded or dataset images.

Images entering this module use RGB channel order and uint8 values. These
operations are intended for comparison and education; anomaly inference
should continue to use the unmodified image unless an evaluation validates
another preprocessing path.
"""

from __future__ import annotations

import cv2
import numpy as np


def _validate_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype != np.uint8:
        raise TypeError("Expected a uint8 image with values from 0 to 255.")
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape (height, width, 3).")
    if array.shape[0] < 2 or array.shape[1] < 2:
        raise ValueError("Image dimensions must each be at least 2 pixels.")
    return np.ascontiguousarray(array)


def gaussian_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply Gaussian smoothing while preserving the RGB uint8 format."""

    rgb = _validate_rgb(image)
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd integer greater than or equal to 3.")
    return cv2.GaussianBlur(rgb, (kernel_size, kernel_size), sigmaX=0)


def median_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply median filtering, useful for reducing isolated impulse noise."""

    rgb = _validate_rgb(image)
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd integer greater than or equal to 3.")
    return cv2.medianBlur(rgb, kernel_size)


def clahe_rgb(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Enhance local luminance with CLAHE while retaining RGB output."""

    rgb = _validate_rgb(image)
    if clip_limit <= 0:
        raise ValueError("clip_limit must be greater than 0.")
    if len(tile_grid_size) != 2 or any(size < 1 for size in tile_grid_size):
        raise ValueError("tile_grid_size must contain two positive integers.")

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_lightness = clahe.apply(lightness)
    enhanced_lab = cv2.merge((enhanced_lightness, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)


def canny_edges(
    image: np.ndarray,
    threshold_low: int = 80,
    threshold_high: int = 160,
) -> np.ndarray:
    """Return a single-channel uint8 Canny edge map."""

    rgb = _validate_rgb(image)
    if not (0 <= threshold_low < threshold_high <= 255):
        raise ValueError("Thresholds must satisfy 0 <= threshold_low < threshold_high <= 255.")
    grayscale = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return cv2.Canny(grayscale, threshold_low, threshold_high)


def channel_histograms(image: np.ndarray, bins: int = 256) -> dict[str, np.ndarray]:
    """Return per-channel RGB histograms for comparing image transforms."""

    rgb = _validate_rgb(image)
    if bins < 2 or bins > 256:
        raise ValueError("bins must be between 2 and 256.")
    names = ("red", "green", "blue")
    return {
        name: np.histogram(rgb[:, :, index], bins=bins, range=(0, 256))[0]
        for index, name in enumerate(names)
    }
