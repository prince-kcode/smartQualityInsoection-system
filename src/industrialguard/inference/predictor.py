"""Checkpoint-backed PatchCore prediction for one uploaded image."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from industrialguard.data.mvtec import MVTEC_CATEGORIES


SUPPORTED_CATEGORIES = MVTEC_CATEGORIES


@dataclass(frozen=True)
class PredictionResult:
    """Image-level and pixel-level anomaly outputs for one image."""

    category: str
    anomaly_score: float
    is_anomalous: bool
    anomaly_map: np.ndarray
    processing_seconds: float


@dataclass
class LoadedPredictor:
    """Reusable model and engine for one category."""

    category: str
    model: Any
    engine: Any
    checkpoint: Path


def checkpoint_path(category: str, results_root: Path | None = None) -> Path:
    """Find the trained Anomalib checkpoint for a supported category."""

    if category not in SUPPORTED_CATEGORIES:
        raise ValueError(f"Category must be one of: {', '.join(SUPPORTED_CATEGORIES)}")
    project_root = Path(__file__).resolve().parents[3]
    root = Path(results_root) if results_root is not None else project_root / "results" / "retrained"
    return root / category / "Patchcore" / "MVTecAD" / category / "v0" / "weights" / "lightning" / "model.ckpt"


def load_predictor(category: str, results_root: Path | None = None) -> LoadedPredictor:
    """Load the category-specific PatchCore model and Anomalib engine."""

    weights_path = checkpoint_path(category, results_root)
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"No trained checkpoint for {category} at {weights_path}. "
            f"Run scripts/train_patchcore.py --category {category} first."
        )

    # Import the heavy model stack only when a prediction is requested.
    from anomalib.engine import Engine
    from anomalib.models import Patchcore

    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        pre_trained=True,
        num_neighbors=9,
    )
    engine = Engine(
        default_root_dir=str(weights_path.parents[6]),
        accelerator="auto",
        devices=1,
    )
    return LoadedPredictor(category, model, engine, weights_path)


def predict_image(
    image_path: Path,
    category: str,
    results_root: Path | None = None,
    predictor: LoadedPredictor | None = None,
) -> PredictionResult:
    """Run the matching category model against an image file.

    Prediction is performed by Anomalib's Engine API. The anomaly score is a
    distance-based score, not a calibrated probability.
    """

    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    from anomalib.data import PredictDataset
    loaded = predictor or load_predictor(category, results_root)
    if loaded.category != category:
        raise ValueError(f"Loaded model is for {loaded.category}, not {category}.")
    dataset = PredictDataset(path=str(image_path), image_size=(256, 256))

    started = perf_counter()
    predictions = loaded.engine.predict(
        model=loaded.model,
        dataset=dataset,
        ckpt_path=str(loaded.checkpoint),
        return_predictions=True,
    )
    elapsed = perf_counter() - started
    if not predictions:
        raise RuntimeError("Anomalib returned no prediction for the uploaded image.")

    prediction = predictions[0]
    score_value = getattr(prediction, "pred_score", None)
    label_value = getattr(prediction, "pred_label", None)
    anomaly_map_value = getattr(prediction, "anomaly_map", None)
    if score_value is None or label_value is None or anomaly_map_value is None:
        raise RuntimeError("Anomalib prediction is missing a score, label, or anomaly map.")

    def to_numpy(value: object) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()  # type: ignore[union-attr]
        return np.asarray(value)

    score = float(to_numpy(score_value).reshape(-1)[0])
    label = bool(to_numpy(label_value).reshape(-1)[0])
    anomaly_map = to_numpy(anomaly_map_value).squeeze()
    if anomaly_map.ndim != 2:
        raise RuntimeError(f"Expected a 2D anomaly map, received shape {anomaly_map.shape}.")

    return PredictionResult(
        category=category,
        anomaly_score=score,
        is_anomalous=label,
        anomaly_map=anomaly_map.astype(np.float32, copy=False),
        processing_seconds=elapsed,
    )
