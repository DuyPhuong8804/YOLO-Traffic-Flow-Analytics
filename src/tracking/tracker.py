"""
Wrapper quanh tinh nang tracking co san cua Ultralytics (model.track()).

Ultralytics tich hop san 2 tracker: ByteTrack (mac dinh, nhe, khong can model
re-identification rieng) va BoT-SORT. De bai uu tien ByteTrack vi nhe va tich
hop san - chi can truyen tracker="bytetrack.yaml".

ByteTrack hoat dong qua 2 buoc chinh (giup giam that lac ID khi xe may bi che khuat):
    1. Buoc 1: match cac detection CO DO TIN CAY CAO voi track dang co bang IOU.
    2. Buoc 2: match tiep cac detection do tin cay THAP (thuong la vat bi che
       khuat mot phan) voi cac track CHUA duoc match o buoc 1, thay vi bo qua
       luon nhu NMS/tracker thong thuong. Nho vay, xe may bi khuat mot phan
       trong dam dong van co co hoi duoc gan lai dung track_id cu thay vi bi
       tao ID moi (gay dem trung) hoac mat track (gay dem sot).
    3. Kalman filter du doan vi tri track trong cac frame khong co detection
       khop (occlusion ngan), giu track "song" trong track_buffer frame truoc
       khi xoa han - cau hinh qua track_buffer trong configs/app_config.yaml.
"""

import numpy as np
from ultralytics import YOLO

from src.tracking.types import TrackedObject


class VehicleTracker:
    """Chay detect + track dong thoi tren tung frame, tra ve danh sach TrackedObject."""

    def __init__(
        self,
        weights_path: str,
        tracker_cfg: str = "bytetrack.yaml",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.5,
    ):
        self.model = YOLO(weights_path)
        self.tracker_cfg = tracker_cfg
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def update(self, frame: np.ndarray) -> list[TrackedObject]:
        """Xu ly 1 frame moi. persist=True de model nho track state giua cac lan goi."""
        results = self.model.track(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            tracker=self.tracker_cfg,
            persist=True,
            verbose=False,
        )[0]

        tracked = []
        if results.boxes is None or results.boxes.id is None:
            # Chua co track nao duoc gan ID (vd frame dau tien khong detect duoc gi)
            return tracked

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            tracked.append(
                TrackedObject(
                    track_id=int(box.id[0]),
                    class_id=int(box.cls[0]),
                    confidence=float(box.conf[0]),
                    xyxy=(x1, y1, x2, y2),
                )
            )
        return tracked
