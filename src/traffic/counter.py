from collections import Counter
from dataclasses import dataclass, field

Point = tuple[float, float]


@dataclass
class CrossingEvent:
    track_id: int
    class_name: str
    direction: str  # "in" | "out"
    frame_idx: int


def _side(a: Point, b: Point, p: Point) -> float:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _segments_intersect(p1: Point, p2: Point, a: Point, b: Point) -> bool:
    d1, d2 = _side(a, b, p1), _side(a, b, p2)
    d3, d4 = _side(p1, p2, a), _side(p1, p2, b)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


@dataclass
class LineCounter:
    """Dem phuong tien khi tam cua track cat qua doan thang AB.

    Moi track_id chi duoc dem mot lan, nen xe dung lai hay rung bbox tren vach
    khong bi dem trung. Chieu "in" la di tu phia trai sang phia phai cua vector AB
    (voi truc y huong xuong nhu toa do anh, tuc la tu tren xuong duoi voi vach ngang).
    """

    a: Point
    b: Point
    counts: dict = field(default_factory=lambda: {"in": Counter(), "out": Counter()})
    _last_pos: dict = field(default_factory=dict)
    _class_votes: dict = field(default_factory=dict)
    _counted: set = field(default_factory=set)

    def update(self, tracks, frame_idx: int) -> list[CrossingEvent]:
        """tracks: iterable (track_id, class_name, (cx, cy))."""
        events = []
        for tid, cls, pos in tracks:
            votes = self._class_votes.setdefault(tid, Counter())
            votes[cls] += 1
            prev = self._last_pos.get(tid)
            self._last_pos[tid] = pos
            if prev is None or tid in self._counted:
                continue
            if _segments_intersect(prev, pos, self.a, self.b):
                direction = "in" if _side(self.a, self.b, pos) > 0 else "out"
                # Lay lop duoc bo phieu nhieu nhat qua cac frame de on dinh phan loai
                final_cls = votes.most_common(1)[0][0]
                self._counted.add(tid)
                self.counts[direction][final_cls] += 1
                events.append(CrossingEvent(tid, final_cls, direction, frame_idx))
        return events

    def is_counted(self, track_id: int) -> bool:
        return track_id in self._counted

    @property
    def total(self) -> int:
        return sum(self.counts["in"].values()) + sum(self.counts["out"].values())

    def by_class(self) -> Counter:
        return self.counts["in"] + self.counts["out"]
