import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from .classes import to_project_class
from .config import load_config, resolve
from .counter import CrossingEvent, LineCounter
from .drawing import draw_box, draw_hud, draw_line, ui_scale
from .storage import TrafficDB


def default_weights() -> Path:
    cfg = load_config()["model"]
    trained = resolve(cfg["weights"])
    return trained if trained.exists() else resolve(cfg["fallback_weights"])


def load_model(weights: str | Path | None = None) -> YOLO:
    return YOLO(str(weights or default_weights()))


def open_source(source):
    src = int(source) if str(source).isdigit() else str(source)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được nguồn video: {source}")
    return cap


def read_first_frame(source) -> np.ndarray:
    cap = open_source(source)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Không đọc được khung hình đầu tiên")
    return frame


@dataclass
class FrameResult:
    frame: np.ndarray  # BGR da ve
    frame_idx: int
    total_frames: int
    fps: float
    video_fps: float
    counts_in: Counter
    counts_out: Counter
    new_events: list[CrossingEvent] = field(default_factory=list)
    session_id: int | None = None

    @property
    def by_class(self) -> Counter:
        return self.counts_in + self.counts_out

    @property
    def total(self) -> int:
        return sum(self.by_class.values())

    @property
    def video_seconds(self) -> float:
        return self.frame_idx / self.video_fps

    @property
    def progress(self) -> float:
        return min(self.frame_idx / self.total_frames, 1.0) if self.total_frames > 0 else 0.0


class TrafficPipeline:
    """Video -> YOLO26 detect -> ByteTrack -> dem qua vach -> SQLite."""

    def __init__(self, model: YOLO | None = None, conf=None, iou=None, imgsz=None, tracker=None,
                 db: TrafficDB | None = None, device=None):
        cfg = load_config()["model"]
        self.model = model or load_model()
        self.conf = conf if conf is not None else cfg["conf"]
        self.iou = iou if iou is not None else cfg["iou"]
        self.imgsz = imgsz or cfg["imgsz"]
        self.tracker = tracker or cfg["tracker"]
        self.db = db
        self.device = device

    def run(self, source, line_norm, start_time: datetime | None = None, output_path: str | None = None,
            max_frames: int | None = None):
        cap = open_source(source)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        is_live = str(source).isdigit() or str(source).startswith(("rtsp://", "http://", "https://"))
        start_time = start_time or datetime.now()

        a = (int(line_norm[0][0] * w), int(line_norm[0][1] * h))
        b = (int(line_norm[1][0] * w), int(line_norm[1][1] * h))
        counter = LineCounter(a, b)
        s = ui_scale(w)

        writer = None
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (w, h))

        session_id = None
        if self.db:
            session_id = self.db.start_session(
                source=Path(str(source)).name if not is_live else str(source),
                started_at=start_time, line=line_norm, model=Path(self.model.ckpt_path or "").name,
            )

        names = self.model.names
        # Reset tracker giua cac lan chay de ID khong bi ke thua tu video truoc
        if getattr(self.model, "predictor", None) is not None:
            self.model.predictor.trackers = None
            self.model.predictor = None

        frame_idx, t0, fps, flash = 0, time.perf_counter(), 0.0, 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok or (max_frames and frame_idx >= max_frames):
                    break
                tick = time.perf_counter()
                res = self.model.track(frame, persist=True, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
                                       tracker=self.tracker, device=self.device, verbose=False)[0]

                dets = []
                if res.boxes is not None and len(res.boxes):
                    boxes = res.boxes.xyxy.cpu().numpy()
                    clss = res.boxes.cls.int().cpu().tolist()
                    confs = res.boxes.conf.cpu().tolist()
                    ids = res.boxes.id.int().cpu().tolist() if res.boxes.id is not None else [None] * len(clss)
                    for box, c, cf, tid in zip(boxes, clss, confs, ids):
                        cls = to_project_class(names[c])
                        if cls:
                            dets.append((box, cls, cf, tid))

                tracks = [(tid, cls, ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
                          for box, cls, _, tid in dets if tid is not None]
                events = counter.update(tracks, frame_idx)
                if events:
                    flash = 6
                    if self.db:
                        ts_of = (lambda _: datetime.now()) if is_live else (
                            lambda f: start_time + timedelta(seconds=f / video_fps))
                        self.db.add_events(session_id, [(e.track_id, e.class_name, e.direction, e.frame_idx,
                                                         ts_of(e.frame_idx)) for e in events])

                for box, cls, cf, tid in dets:
                    draw_box(frame, box, cls, tid, cf, tid is not None and counter.is_counted(tid), s)
                draw_line(frame, a, b, flash > 0, s)
                flash = max(flash - 1, 0)

                inst = 1.0 / max(time.perf_counter() - tick, 1e-6)
                fps = inst if fps == 0 else 0.9 * fps + 0.1 * inst
                by_class = counter.by_class()
                draw_hud(frame, by_class, counter.total, fps, s)
                if writer:
                    writer.write(frame)

                frame_idx += 1
                yield FrameResult(frame, frame_idx, total_frames, fps, video_fps, Counter(counter.counts["in"]),
                                  Counter(counter.counts["out"]), events, session_id)
        finally:
            cap.release()
            if writer:
                writer.release()
            if self.db and session_id is not None:
                elapsed = time.perf_counter() - t0
                ended = datetime.now() if is_live else start_time + timedelta(seconds=frame_idx / video_fps)
                self.db.end_session(session_id, ended, frame_idx, frame_idx / elapsed if elapsed else 0)
