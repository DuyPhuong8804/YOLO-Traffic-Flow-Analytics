"""
Luu tru ket qua dem xe vao SQLite.

De tranh nghen I/O khi xu ly video real-time (moi lan xe cat line phai ghi DB
ngay se lam cham vong lap xu ly frame), du lieu duoc TICH LUY vao buffer trong
bo nho va ghi xuong DB theo LO (batch insert) khi thoa 1 trong 2 dieu kien:
    - so ban ghi tich luy dat nguong batch_insert_size, HOAC
    - da qua batch_insert_interval_sec giay ke tu lan flush truoc.

Schema toi thieu theo yeu cau de bai: id, timestamp, loai_xe, huong_di_chuyen, camera_id.
Bo sung them line_name/track_id de phuc vu debug va mo rong thong ke sau nay.
"""

import sqlite3
import time
from pathlib import Path

from src.common.class_map import class_id_to_name
from src.counting.vehicle_counter import CountEvent

SCHEMA = """
CREATE TABLE IF NOT EXISTS traffic_counts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    loai_xe TEXT NOT NULL,
    huong_di_chuyen TEXT NOT NULL,
    camera_id TEXT NOT NULL,
    line_name TEXT NOT NULL,
    track_id INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_traffic_counts_timestamp ON traffic_counts (timestamp);
CREATE INDEX IF NOT EXISTS idx_traffic_counts_loai_xe ON traffic_counts (loai_xe);
"""


class TrafficDatabase:
    """Ket noi SQLite + buffer batch insert cho cac su kien dem xe (CountEvent)."""

    def __init__(self, db_path: str, batch_size: int = 20, flush_interval_sec: float = 5.0):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")  # cho phep doc (API/dashboard) khong bi block khi dang ghi
        self.conn.executescript(SCHEMA)
        self.conn.commit()

        self.batch_size = batch_size
        self.flush_interval_sec = flush_interval_sec
        self._buffer: list[tuple] = []
        self._last_flush = time.time()

    def add_event(self, event: CountEvent) -> None:
        """Them 1 su kien dem vao buffer. Tu dong flush neu du dieu kien."""
        self._buffer.append(
            (
                event.timestamp,
                class_id_to_name(event.class_id),
                event.direction,
                event.camera_id,
                event.line_name,
                event.track_id,
            )
        )
        should_flush = (
            len(self._buffer) >= self.batch_size
            or (time.time() - self._last_flush) >= self.flush_interval_sec
        )
        if should_flush:
            self.flush()

    def flush(self) -> None:
        """Ghi toan bo buffer hien tai xuong SQLite trong 1 transaction duy nhat."""
        if not self._buffer:
            self._last_flush = time.time()
            return
        self.conn.executemany(
            """INSERT INTO traffic_counts
               (timestamp, loai_xe, huong_di_chuyen, camera_id, line_name, track_id)
               VALUES (?, ?, ?, ?, ?, ?)""",
            self._buffer,
        )
        self.conn.commit()
        self._buffer.clear()
        self._last_flush = time.time()

    def close(self) -> None:
        self.flush()
        self.conn.close()
