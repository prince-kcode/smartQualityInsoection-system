"""Industrial Quality Inspection - YOLO Defect Localizer Module.

Provides bounding-box defect localization using Ultralytics YOLO models.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Search paths for YOLO weights
DEFAULT_MODEL_PATHS = [
    Path("models/industrial_defect_yolo.pt"),
    Path("models/bottle_defect_yolo.pt"),
    Path("industrialguard_mlp/models/bottle_defect_yolo.pt"),
    Path("industrialguard_mlp/yolo11n.pt"),
]


@dataclass
class DefectBox:
    box: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str
    area: int


@dataclass
class YOLODetectionResult:
    annotated_image: Image.Image
    detections: List[DefectBox] = field(default_factory=list)
    has_defect: bool = False
    defect_count: int = 0
    max_confidence: float = 0.0
    model_name: str = "Unknown"
    status_summary: str = "No defects detected by YOLO"


class IndustrialYOLODetector:
    """YOLO detector for industrial surface & component defects."""

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        self.model = None
        self.model_path = None
        self._load_model(model_path)

    def _load_model(self, model_path: Optional[Union[str, Path]] = None):
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.warning("ultralytics package not installed. YOLO detection disabled.")
            return

        candidates = []
        if model_path:
            candidates.append(Path(model_path))
        candidates.extend(DEFAULT_MODEL_PATHS)

        for p in candidates:
            if p.is_file():
                try:
                    logger.info(f"Loading YOLO model from: {p}")
                    self.model = YOLO(str(p))
                    self.model_path = p
                    break
                except Exception as e:
                    logger.warning(f"Failed to load YOLO model from {p}: {e}")

        if self.model is None:
            logger.warning("No YOLO defect checkpoint found. Running in fallback mode.")

    def detect(
        self,
        image: Union[Image.Image, np.ndarray, str, Path],
        conf: float = 0.20,
        iou: float = 0.45,
        target_category: Optional[str] = None,
    ) -> YOLODetectionResult:
        """Run YOLO defect detection on an input image.

        Args:
            image: Input image (PIL, numpy BGR/RGB, or file path).
            conf: Confidence threshold for bounding boxes.
            iou: NMS IoU threshold.
            target_category: Optional category name to specialize detection.

        Returns:
            YOLODetectionResult containing bounding boxes and annotated PIL image.
        """
        # Convert image to RGB PIL and BGR numpy for OpenCV drawing
        if isinstance(image, (str, Path)):
            pil_img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            if image.ndim == 2:
                pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
            elif image.shape[2] == 3:
                pil_img = Image.fromarray(image)
            else:
                pil_img = Image.fromarray(image[:, :, :3])
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        np_rgb = np.array(pil_img)
        h, w = np_rgb.shape[:2]

        # If model is loaded, run YOLO inference
        if self.model is not None:
            try:
                results = self.model.predict(
                    source=np_rgb,
                    conf=conf,
                    iou=iou,
                    verbose=False,
                    device="cuda:0" if hasattr(self.model, "device") and self.model.device.type == "cuda" else "cpu"
                )
                
                detections = []
                annotated_bgr = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2BGR)
                
                if results and len(results) > 0:
                    r = results[0]
                    boxes = r.boxes
                    if boxes is not None and len(boxes) > 0:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            c = float(box.conf[0].item())
                            cls_id = int(box.cls[0].item())
                            cls_name = r.names.get(cls_id, f"defect_{cls_id}")
                            
                            # Clean up class name for display
                            display_name = cls_name.replace("_", " ").title()
                            if "defect" not in display_name.lower():
                                display_name = f"{display_name} Defect"

                            bx1, by1, bx2, by2 = int(x1), int(y1), int(x2), int(y2)
                            area = (bx2 - bx1) * (by2 - by1)
                            
                            detections.append(
                                DefectBox(
                                    box=(bx1, by1, bx2, by2),
                                    confidence=c,
                                    class_id=cls_id,
                                    class_name=display_name,
                                    area=area,
                                )
                            )
                            
                            # Draw high-visibility industrial bounding box
                            # Crimson red for defects
                            color = (0, 30, 230)
                            cv2.rectangle(annotated_bgr, (bx1, by1), (bx2, by2), color, 3)
                            
                            # Draw label background
                            label_str = f"{display_name} {c*100:.1f}%"
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            scale = 0.55
                            thickness = 2
                            (tw, th), baseline = cv2.getTextSize(label_str, font, scale, thickness)
                            
                            # Top badge
                            badge_y1 = max(0, by1 - th - 10)
                            badge_y2 = by1
                            badge_x2 = min(w, bx1 + tw + 10)
                            cv2.rectangle(annotated_bgr, (bx1, badge_y1), (badge_x2, badge_y2), color, -1)
                            cv2.putText(annotated_bgr, label_str, (bx1 + 5, by1 - 5), font, scale, (255, 255, 255), thickness)

                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                annotated_pil = Image.fromarray(annotated_rgb)

                has_defect = len(detections) > 0
                max_conf = max([d.confidence for d in detections], default=0.0)
                summary = (
                    f"YOLO detected {len(detections)} defect region(s) (Peak conf: {max_conf*100:.1f}%)"
                    if has_defect
                    else "YOLO verified: No defect bounding boxes found"
                )

                return YOLODetectionResult(
                    annotated_image=annotated_pil,
                    detections=detections,
                    has_defect=has_defect,
                    defect_count=len(detections),
                    max_confidence=max_conf,
                    model_name=self.model_path.name if self.model_path else "YOLOv11",
                    status_summary=summary,
                )
            except Exception as e:
                logger.error(f"YOLO inference failed: {e}", exc_info=True)

        # Fallback if no YOLO or inference failure
        return YOLODetectionResult(
            annotated_image=pil_img,
            detections=[],
            has_defect=False,
            defect_count=0,
            max_confidence=0.0,
            model_name="Fallback (YOLO unavailable)",
            status_summary="YOLO detection unavailable in this runtime",
        )


# Global singleton instance for fast reuse in Streamlit
_yolo_instance: Optional[IndustrialYOLODetector] = None


def get_yolo_detector(model_path: Optional[Union[str, Path]] = None) -> IndustrialYOLODetector:
    global _yolo_instance
    if _yolo_instance is None or (model_path and _yolo_instance.model_path != Path(model_path)):
        _yolo_instance = IndustrialYOLODetector(model_path)
    return _yolo_instance
