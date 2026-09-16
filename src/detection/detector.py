"""Wrapper mong quanh Ultralytics YOLO de load model detect 1 lan va tai su dung."""

from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    """1 ket qua phat hien tren 1 frame."""

    xyxy: tuple[float, float, float, float]  # (x1, y1, x2, y2) pixel
    class_id: int
    confidence: float


class VehicleDetector:
    """Load model YOLO26 da fine-tune, dung cho inference tung frame hoac batch."""

    def __init__(self, weights_path: str, conf_threshold: float = 0.35, iou_threshold: float = 0.5):
        self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def predict(self, frame: np.ndarray) -> list[Detection]:
        """Chay detect tren 1 frame BGR (numpy array tu OpenCV), tra ve danh sach Detection."""
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                Detection(
                    xyxy=(x1, y1, x2, y2),
                    class_id=int(box.cls[0]),
                    confidence=float(box.conf[0]),
                )
            )
        return detections
