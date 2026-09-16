"""
Unit test cho logic dem qua duong ke ao (src/counting/*). Khong can model/video
thuc, chi mo phong toa do de kiem tra thuat toan hinh hoc + logic chong dem trung/sot.

Chay:
    pytest tests/test_counting_logic.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.counting.virtual_line import VirtualLine, segments_intersect  # noqa: E402
from src.counting.vehicle_counter import VehicleCounter, MIN_HITS_TO_COUNT  # noqa: E402
from src.tracking.types import TrackedObject  # noqa: E402


def _make_tracked(track_id: int, cx: float, cy: float, class_id: int = 0) -> TrackedObject:
    """Tao 1 TrackedObject voi tam bbox tai (cx, cy), kich thuoc bbox co dinh 10x10."""
    return TrackedObject(
        track_id=track_id,
        class_id=class_id,
        confidence=0.9,
        xyxy=(cx - 5, cy - 5, cx + 5, cy + 5),
    )


# ----------------------------- segments_intersect -----------------------------

def test_segments_intersect_crossing():
    assert segments_intersect((0, 0), (10, 10), (0, 10), (10, 0)) is True


def test_segments_intersect_parallel_no_cross():
    assert segments_intersect((0, 0), (10, 0), (0, 5), (10, 5)) is False


def test_segments_intersect_not_reaching():
    # 2 doan thang cung huong nhung khong du dai de cham nhau
    assert segments_intersect((0, 0), (4, 4), (0, 10), (10, 0)) is False


# ----------------------------- VirtualLine.direction_label -----------------------------

def test_direction_label_opposite_for_opposite_movement():
    line = VirtualLine(
        name="l1", point1=(0, 100), point2=(200, 100),
        direction_a_label="xuong", direction_b_label="len",
    )
    down = line.direction_label((50, 90), (50, 110))   # di chuyen tu tren xuong duoi
    up = line.direction_label((50, 110), (50, 90))      # di chuyen tu duoi len tren
    assert down != up
    assert {down, up} == {"xuong", "len"}


# ----------------------------- VehicleCounter -----------------------------

def _horizontal_line() -> VirtualLine:
    return VirtualLine(
        name="line_test", point1=(0, 100), point2=(200, 100),
        direction_a_label="xuong", direction_b_label="len",
    )


def test_counter_emits_exactly_one_event_on_crossing():
    counter = VehicleCounter(camera_id="cam1", lines=[_horizontal_line()])

    # Track di chuyen tu y=80 xuong y=120 qua nhieu frame, cat qua line y=100
    ys = [80, 85, 90, 95, 105, 110]
    all_events = []
    for y in ys:
        events = counter.update([_make_tracked(track_id=1, cx=50, cy=y)])
        all_events.extend(events)

    assert len(all_events) == 1
    assert all_events[0].direction == "xuong"
    assert all_events[0].track_id == 1


def test_counter_does_not_double_count_same_track():
    counter = VehicleCounter(camera_id="cam1", lines=[_horizontal_line()])

    # Xe di qua line roi dao dong qua lai gan line (nhieu do rung do nhieu detect)
    ys = [80, 90, 95, 105, 98, 102, 97, 103]
    all_events = []
    for y in ys:
        events = counter.update([_make_tracked(track_id=1, cx=50, cy=y)])
        all_events.extend(events)

    # Chi duoc tinh la 1 luot dem duy nhat cho track_id=1, du bbox dao dong qua lai
    assert len(all_events) == 1


def test_counter_requires_min_hits_before_counting():
    counter = VehicleCounter(camera_id="cam1", lines=[_horizontal_line()])

    # Track xuat hien lan dau ngay tai vi tri da vuot qua line -> khong co "prev" hop le,
    # kiem tra rang cac frame dau (chua du MIN_HITS_TO_COUNT) khong sinh su kien nham.
    events_frame1 = counter.update([_make_tracked(track_id=2, cx=50, cy=80)])
    assert events_frame1 == []
    assert MIN_HITS_TO_COUNT >= 1


def test_counter_independent_tracks_counted_separately():
    counter = VehicleCounter(camera_id="cam1", lines=[_horizontal_line()])

    all_events = []
    for y in [80, 90, 95, 105, 110]:
        events = counter.update(
            [_make_tracked(track_id=1, cx=30, cy=y), _make_tracked(track_id=2, cx=170, cy=y)]
        )
        all_events.extend(events)

    track_ids_counted = {e.track_id for e in all_events}
    assert track_ids_counted == {1, 2}
    assert len(all_events) == 2


def test_counter_no_event_when_not_crossing():
    counter = VehicleCounter(camera_id="cam1", lines=[_horizontal_line()])

    all_events = []
    for y in [10, 15, 20, 25, 30]:  # khong bao gio den gan line y=100
        events = counter.update([_make_tracked(track_id=1, cx=50, cy=y)])
        all_events.extend(events)

    assert all_events == []
