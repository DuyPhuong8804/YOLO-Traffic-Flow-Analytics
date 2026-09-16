"""Ham ve overlay (bounding box, duong ke ao, bang dem) len frame bang OpenCV."""

import cv2
import numpy as np

from src.common.class_map import CLASS_COLORS_BGR, CLASS_NAMES_VI, VehicleClass
from src.counting.virtual_line import VirtualLine
from src.tracking.types import TrackedObject


def draw_tracked_objects(frame: np.ndarray, objects: list[TrackedObject]) -> np.ndarray:
    for obj in objects:
        x1, y1, x2, y2 = [int(v) for v in obj.xyxy]
        color = CLASS_COLORS_BGR[VehicleClass(obj.class_id)]
        label = f"#{obj.track_id} {CLASS_NAMES_VI[VehicleClass(obj.class_id)]} {obj.confidence:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame


def draw_lines(frame: np.ndarray, lines: list[VirtualLine]) -> np.ndarray:
    for line in lines:
        p1 = tuple(int(v) for v in line.point1)
        p2 = tuple(int(v) for v in line.point2)
        cv2.line(frame, p1, p2, (0, 255, 255), 2)
        cv2.putText(frame, line.name, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    return frame


def draw_counts_panel(frame: np.ndarray, counts: dict[str, int]) -> np.ndarray:
    """Ve 1 bang nho o goc tren-trai hien thi tong so dem theo tung nhom (loai_xe/huong)."""
    x0, y0 = 10, 10
    line_height = 22
    panel_h = line_height * (len(counts) + 1) + 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + 260, y0 + panel_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

    cv2.putText(frame, "THONG KE DEM XE", (x0 + 8, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    for i, (label, value) in enumerate(counts.items(), start=1):
        y = y0 + 18 + i * line_height
        cv2.putText(frame, f"{label}: {value}", (x0 + 8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame
