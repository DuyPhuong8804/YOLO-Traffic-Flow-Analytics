"""Unit test cho TrafficDatabase: batch insert theo nguong so luong va flush thu cong."""

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.class_map import VehicleClass  # noqa: E402
from src.counting.vehicle_counter import CountEvent  # noqa: E402
from src.storage.database import TrafficDatabase  # noqa: E402


def _make_event(track_id: int, class_id: int = VehicleClass.CAR) -> CountEvent:
    return CountEvent(
        track_id=track_id, class_id=class_id, camera_id="cam1",
        line_name="line1", direction="xuong",
    )


def test_flush_writes_buffered_events(tmp_path):
    db_path = tmp_path / "test.db"
    db = TrafficDatabase(str(db_path), batch_size=100, flush_interval_sec=999)

    db.add_event(_make_event(track_id=1))
    db.add_event(_make_event(track_id=2))

    # Chua du batch_size va chua het flush_interval -> van con nam trong buffer, chua xuong DB
    conn = sqlite3.connect(str(db_path))
    count_before = conn.execute("SELECT COUNT(*) FROM traffic_counts").fetchone()[0]
    assert count_before == 0

    db.flush()
    count_after = conn.execute("SELECT COUNT(*) FROM traffic_counts").fetchone()[0]
    assert count_after == 2
    conn.close()
    db.close()


def test_auto_flush_when_batch_size_reached(tmp_path):
    db_path = tmp_path / "test.db"
    db = TrafficDatabase(str(db_path), batch_size=3, flush_interval_sec=999)

    for i in range(3):
        db.add_event(_make_event(track_id=i))

    conn = sqlite3.connect(str(db_path))
    count = conn.execute("SELECT COUNT(*) FROM traffic_counts").fetchone()[0]
    assert count == 3  # da tu dong flush ngay khi buffer dat batch_size
    conn.close()
    db.close()


def test_column_values_stored_correctly(tmp_path):
    db_path = tmp_path / "test.db"
    db = TrafficDatabase(str(db_path), batch_size=1, flush_interval_sec=999)

    db.add_event(_make_event(track_id=42, class_id=VehicleClass.TRUCK))

    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        "SELECT loai_xe, huong_di_chuyen, camera_id, line_name, track_id FROM traffic_counts"
    ).fetchone()
    assert row == ("truck", "xuong", "cam1", "line1", 42)
    conn.close()
    db.close()
