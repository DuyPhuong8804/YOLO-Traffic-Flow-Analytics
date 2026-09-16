"""
Cac truy van thong ke tren bang traffic_counts, dung truc tiep boi dashboard
(khong qua lop API trung gian).

timestamp luu trong DB la Unix epoch (float, UTC). Cac ham o day dung
datetime(timestamp, 'unixepoch', 'localtime') de quy doi ve gio dia phuong
khi group theo gio/ngay.
"""

import sqlite3


def _rows_to_dicts(cursor: sqlite3.Cursor) -> list[dict]:
    cols = [c[0] for c in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


def get_hourly_stats(conn: sqlite3.Connection, limit_hours: int = 24) -> list[dict]:
    """Luu luong theo gio (24h gan nhat mac dinh), tach theo loai xe."""
    cur = conn.execute(
        """
        SELECT strftime('%Y-%m-%d %H:00', timestamp, 'unixepoch', 'localtime') AS hour,
               loai_xe,
               COUNT(*) AS count
        FROM traffic_counts
        WHERE timestamp >= (strftime('%s', 'now', 'localtime') - ? * 3600)
        GROUP BY hour, loai_xe
        ORDER BY hour ASC
        """,
        (limit_hours,),
    )
    return _rows_to_dicts(cur)


def get_daily_stats(conn: sqlite3.Connection, limit_days: int = 30) -> list[dict]:
    """Luu luong theo ngay (30 ngay gan nhat mac dinh), tach theo loai xe."""
    cur = conn.execute(
        """
        SELECT date(timestamp, 'unixepoch', 'localtime') AS date,
               loai_xe,
               COUNT(*) AS count
        FROM traffic_counts
        WHERE timestamp >= (strftime('%s', 'now', 'localtime') - ? * 86400)
        GROUP BY date, loai_xe
        ORDER BY date ASC
        """,
        (limit_days,),
    )
    return _rows_to_dicts(cur)


def get_stats_by_type(conn: sqlite3.Connection) -> list[dict]:
    """Tong luu luong theo tung loai xe (toan bo lich su)."""
    cur = conn.execute(
        """
        SELECT loai_xe, COUNT(*) AS count
        FROM traffic_counts
        GROUP BY loai_xe
        ORDER BY count DESC
        """
    )
    return _rows_to_dicts(cur)


def get_peak_hours(conn: sqlite3.Connection, top_n: int = 5) -> list[dict]:
    """Top N khung gio co luu luong cao nhat (tinh tong tat ca loai xe, toan bo lich su)."""
    cur = conn.execute(
        """
        SELECT strftime('%Y-%m-%d %H:00', timestamp, 'unixepoch', 'localtime') AS hour,
               COUNT(*) AS total_count
        FROM traffic_counts
        GROUP BY hour
        ORDER BY total_count DESC
        LIMIT ?
        """,
        (top_n,),
    )
    return _rows_to_dicts(cur)
