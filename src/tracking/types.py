"""
Dataclass TrackedObject tach rieng khoi tracker.py de cac module chi can KIEU DU LIEU
nay (vehicle_counter.py, draw_utils.py, unit test) khong bi buoc phai import ultralytics/
torch - von la dependency nang va khong can thiet cho logic hinh hoc/dem thuan tuy.
"""

from dataclasses import dataclass


@dataclass
class TrackedObject:
    """1 doi tuong da duoc detect + gan track_id o 1 frame."""

    track_id: int
    class_id: int
    confidence: float
    xyxy: tuple[float, float, float, float]

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
