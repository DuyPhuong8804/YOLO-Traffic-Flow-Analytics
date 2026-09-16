"""
Orchestrator: doc video -> detect+track (VehicleTracker) -> dem qua line (VehicleCounter)
-> ghi DB theo lo (TrafficDatabase) -> (tuy chon) ve overlay de preview.

Day la "trai tim" ket noi tat ca cac module lai voi nhau. Duoc dung boi
src/dashboard/app.py (chay co ve overlay de hien thi video preview truc tiep).
"""

from collections import Counter
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from src.common.class_map import class_id_to_name
from src.common.config_loader import load_yaml
from src.common.draw_utils import draw_counts_panel, draw_lines, draw_tracked_objects
from src.counting.virtual_line import VirtualLine
from src.counting.vehicle_counter import CountEvent, VehicleCounter
from src.storage.database import TrafficDatabase
from src.tracking.tracker import VehicleTracker


@dataclass
class FrameResult:
    frame: np.ndarray
    new_events: list[CountEvent]
    cumulative_counts: dict[str, int]
    fps: float


def _build_lines(camera_cfg: dict) -> list[VirtualLine]:
    return [
        VirtualLine(
            name=ln["name"],
            point1=tuple(ln["point1"]),
            point2=tuple(ln["point2"]),
            direction_a_label=ln["direction_a_label"],
            direction_b_label=ln["direction_b_label"],
        )
        for ln in camera_cfg["lines"]
    ]


class TrafficPipeline:
    """Xu ly 1 video/stream cho 1 camera_id cu the tu dau den cuoi."""

    def __init__(
        self,
        camera_id: str,
        app_config_path: str = "app_config.yaml",
        lines_config_path: str = "counting_lines.yaml",
        video_source_override: str | None = None,
        draw_overlay: bool = True,
    ):
        app_cfg = load_yaml(app_config_path)
        lines_cfg = load_yaml(lines_config_path)
        camera_cfg = lines_cfg["cameras"][camera_id]

        self.camera_id = camera_id
        self.video_source = video_source_override or camera_cfg["video_source"]
        self.draw_overlay = draw_overlay

        self.tracker = VehicleTracker(
            weights_path=app_cfg["model_path"],
            tracker_cfg=app_cfg["tracker"],
            conf_threshold=app_cfg["conf_threshold"],
            iou_threshold=app_cfg["iou_threshold"],
        )
        self.lines = _build_lines(camera_cfg)
        self.counter = VehicleCounter(camera_id=camera_id, lines=self.lines)
        self.db = TrafficDatabase(
            db_path=app_cfg["db_path"],
            batch_size=app_cfg["batch_insert_size"],
            flush_interval_sec=app_cfg["batch_insert_interval_sec"],
        )

        # Bo dem cong don theo (loai_xe, huong) trong suot phien chay, dung de hien
        # thi truc tiep tren overlay/dashboard ma khong can query lai DB moi frame.
        self._cumulative: Counter[str] = Counter()

    def _handle_events(self, events: list[CountEvent]) -> None:
        for ev in events:
            self.db.add_event(ev)
            key = f"{class_id_to_name(ev.class_id)} - {ev.direction}"
            self._cumulative[key] += 1

    def run(self, stop_event: "threading.Event | None" = None):
        """Generator: moi lan yield 1 FrameResult ung voi 1 frame da xu ly xong.

        stop_event (tuy chon): dung de dung vong lap TU BEN NGOAI mot cach AN TOAN
        khi generator nay dang duoc "keo" (drive) boi 1 THREAD KHAC (vd StreamManager
        chay pipeline trong background thread). KHONG duoc goi generator.close() tu
        thread khac generator dang chay - Python generator khong thread-safe theo
        cach do va se nem ValueError("generator already executing"). Thay vao do,
        thread dang chay generator tu kiem tra stop_event moi vong lap.
        """
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            raise RuntimeError(f"Khong mo duoc video source: {self.video_source}")

        prev_time = time.time()
        try:
            while True:
                if stop_event is not None and stop_event.is_set():
                    break

                ok, frame = cap.read()
                if not ok:
                    break

                tracked_objects = self.tracker.update(frame)
                new_events = self.counter.update(tracked_objects)
                self._handle_events(new_events)

                now = time.time()
                fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now

                if self.draw_overlay:
                    frame = draw_tracked_objects(frame, tracked_objects)
                    frame = draw_lines(frame, self.lines)
                    frame = draw_counts_panel(frame, dict(self._cumulative))

                yield FrameResult(
                    frame=frame,
                    new_events=new_events,
                    cumulative_counts=dict(self._cumulative),
                    fps=fps,
                )
        finally:
            cap.release()
            self.db.close()
