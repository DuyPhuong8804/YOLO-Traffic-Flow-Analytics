import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source      TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    frames      INTEGER DEFAULT 0,
    avg_fps     REAL DEFAULT 0,
    line        TEXT,
    model       TEXT
);
CREATE TABLE IF NOT EXISTS vehicle_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    track_id    INTEGER NOT NULL,
    class_name  TEXT NOT NULL,
    direction   TEXT NOT NULL,
    frame_idx   INTEGER NOT NULL,
    ts          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON vehicle_events(ts);
CREATE INDEX IF NOT EXISTS idx_events_session ON vehicle_events(session_id);
"""

INTERVALS = {"minute": "%Y-%m-%d %H:%M:00", "hour": "%Y-%m-%d %H:00:00", "day": "%Y-%m-%d"}


class TrafficDB:
    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path)
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ---------- ghi ----------
    def start_session(self, source: str, started_at: datetime, line=None, model: str = "") -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO sessions(source, started_at, line, model) VALUES (?,?,?,?)",
                (source, started_at.isoformat(sep=" ", timespec="seconds"), str(line), model),
            )
            return cur.lastrowid

    def add_events(self, session_id: int, events: list[tuple]):
        """events: (track_id, class_name, direction, frame_idx, ts: datetime)"""
        if not events:
            return
        with self._conn() as c:
            c.executemany(
                "INSERT INTO vehicle_events(session_id, track_id, class_name, direction, frame_idx, ts)"
                " VALUES (?,?,?,?,?,?)",
                [(session_id, t, cls, d, f, ts.isoformat(sep=" ", timespec="seconds")) for t, cls, d, f, ts in events],
            )

    def end_session(self, session_id: int, ended_at: datetime, frames: int, avg_fps: float):
        with self._conn() as c:
            c.execute(
                "UPDATE sessions SET ended_at=?, frames=?, avg_fps=? WHERE id=?",
                (ended_at.isoformat(sep=" ", timespec="seconds"), frames, round(avg_fps, 2), session_id),
            )

    def delete_session(self, session_id: int):
        with self._conn() as c:
            c.execute("DELETE FROM sessions WHERE id=?", (session_id,))

    # ---------- doc ----------
    def _where(self, start=None, end=None, session_id=None):
        clauses, params = [], []
        if start:
            clauses.append("e.ts >= ?")
            params.append(str(start))
        if end:
            clauses.append("e.ts <= ?")
            params.append(str(end))
        if session_id:
            clauses.append("e.session_id = ?")
            params.append(int(session_id))
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", params

    def _df(self, sql, params=()) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql_query(sql, c, params=params)

    def sessions(self) -> pd.DataFrame:
        return self._df(
            "SELECT s.*, COUNT(e.id) AS vehicles FROM sessions s"
            " LEFT JOIN vehicle_events e ON e.session_id = s.id"
            " GROUP BY s.id ORDER BY s.id DESC"
        )

    def events(self, start=None, end=None, session_id=None, limit: int | None = None) -> pd.DataFrame:
        where, params = self._where(start, end, session_id)
        sql = f"SELECT e.* FROM vehicle_events e{where} ORDER BY e.ts DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        return self._df(sql, params)

    def summary(self, start=None, end=None, session_id=None) -> pd.DataFrame:
        where, params = self._where(start, end, session_id)
        return self._df(
            f"SELECT class_name, direction, COUNT(*) AS count FROM vehicle_events e{where}"
            " GROUP BY class_name, direction",
            params,
        )

    def timeseries(self, interval="hour", start=None, end=None, session_id=None) -> pd.DataFrame:
        fmt = INTERVALS[interval]
        where, params = self._where(start, end, session_id)
        return self._df(
            f"SELECT strftime('{fmt}', e.ts) AS bucket, class_name, COUNT(*) AS count"
            f" FROM vehicle_events e{where} GROUP BY bucket, class_name ORDER BY bucket",
            params,
        )

    def time_range(self):
        with self._conn() as c:
            return c.execute("SELECT MIN(ts), MAX(ts) FROM vehicle_events").fetchone()
