"""
VehicleCounter: ket hop ket qua tracking (TrackedObject moi frame) voi 1 hoac nhieu
VirtualLine de sinh ra CountEvent moi khi 1 track cat qua duong ke lan dau tien.

Trang thai duoc quan ly hoan toan trong bo nho (dict), phu hop xu ly video theo
luong (streaming) frame-by-frame ma khong can luu toan bo lich su frame.
"""

from collections import deque
from dataclasses import dataclass, field
import time

from src.counting.virtual_line import VirtualLine, Point
from src.tracking.types import TrackedObject

HISTORY_MAXLEN = 5  # so vi tri tam gan nhat luu cho moi track, dung de tinh vector huong on dinh hon
MIN_HITS_TO_COUNT = 3  # track phai xuat hien >= so frame nay truoc khi duoc phep tinh la 1 luot dem
STALE_AFTER_FRAMES = 150  # so frame vang mat lien tuc truoc khi xoa han state (~5-10s o 15-30 FPS)


@dataclass
class CountEvent:
    """1 su kien dem duoc: 1 track cat qua 1 duong ke theo 1 huong, tai 1 thoi diem."""

    track_id: int
    class_id: int
    camera_id: str
    line_name: str
    direction: str
    timestamp: float = field(default_factory=time.time)


class _TrackState:
    """Trang thai noi bo cho 1 track_id: lich su vi tri tam + so frame da thay + da dem chua."""

    __slots__ = ("centers", "hits", "counted_lines", "missed_frames")

    def __init__(self) -> None:
        self.centers: deque[Point] = deque(maxlen=HISTORY_MAXLEN)
        self.hits = 0
        self.counted_lines: set[str] = set()  # ten cac line da dem track nay roi (moi line dem doc lap)
        self.missed_frames = 0  # so frame lien tuc gan day KHONG thay track nay trong ket qua tracker


class VehicleCounter:
    """Quan ly dem xe cat qua danh sach VirtualLine cho 1 camera cu the."""

    def __init__(self, camera_id: str, lines: list[VirtualLine]):
        self.camera_id = camera_id
        self.lines = lines
        self._states: dict[int, _TrackState] = {}

    def update(self, tracked_objects: list[TrackedObject]) -> list[CountEvent]:
        """Goi 1 lan moi frame voi danh sach TrackedObject hien tai. Tra ve cac CountEvent
        moi sinh ra trong frame nay (thuong rong, chi co gia tri khi vua co xe cat line)."""
        events: list[CountEvent] = []
        seen_ids = set()

        for obj in tracked_objects:
            seen_ids.add(obj.track_id)
            state = self._states.setdefault(obj.track_id, _TrackState())
            state.hits += 1
            state.missed_frames = 0
            curr_center = obj.center

            if state.centers and state.hits >= MIN_HITS_TO_COUNT:
                prev_center = state.centers[-1]
                for line in self.lines:
                    if line.name in state.counted_lines:
                        continue  # track nay da duoc dem qua line nay roi, khong dem lai
                    if line.crossed_by(prev_center, curr_center):
                        direction = line.direction_label(prev_center, curr_center)
                        events.append(
                            CountEvent(
                                track_id=obj.track_id,
                                class_id=obj.class_id,
                                camera_id=self.camera_id,
                                line_name=line.name,
                                direction=direction,
                            )
                        )
                        state.counted_lines.add(line.name)

            state.centers.append(curr_center)

        self._cleanup_stale_tracks(seen_ids)
        return events

    def _cleanup_stale_tracks(self, seen_ids: set[int]) -> None:
        """Tang bo dem vang mat cho cac track khong xuat hien o frame nay, va CHI xoa
        han state khi da vang mat qua STALE_AFTER_FRAMES frame lien tuc.

        Quan trong: KHONG xoa ngay lap tuc khi track vang mat 1 frame, vi ByteTrack
        van giu track_id "song" trong track_buffer frame de cho phep tai xuat hien
        sau occlusion. Neu xoa som, se mat lich su vi tri (centers) truoc luc bi
        khuat -> khong the phat hien duoc truong hop xe "nhay" qua duong ke ngay
        trong luc bi che khuat, dan den dem sot."""
        stale_ids = set(self._states.keys()) - seen_ids
        for tid in stale_ids:
            state = self._states[tid]
            state.missed_frames += 1
            if state.missed_frames > STALE_AFTER_FRAMES:
                del self._states[tid]
