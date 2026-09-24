from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from traffic import api
from traffic.storage import TrafficDB

T0 = datetime(2026, 9, 1, 7, 0, 0)


@pytest.fixture
def db(tmp_path):
    d = TrafficDB(tmp_path / "t.db")
    sid = d.start_session("demo.mp4", T0, line=[[0, 0.5], [1, 0.5]])
    d.add_events(sid, [
        (1, "car", "in", 10, T0 + timedelta(minutes=5)),
        (2, "car", "out", 20, T0 + timedelta(minutes=30)),
        (3, "truck", "in", 30, T0 + timedelta(hours=1, minutes=10)),
        (4, "motorbike", "in", 40, T0 + timedelta(hours=1, minutes=20)),
    ])
    d.end_session(sid, T0 + timedelta(hours=2), frames=100, avg_fps=25.0)
    return d


def test_summary_and_timeseries(db):
    s = db.summary()
    assert s["count"].sum() == 4
    ts = db.timeseries("hour")
    assert set(ts["bucket"]) == {"2026-09-01 07:00:00", "2026-09-01 08:00:00"}


def test_time_filter(db):
    assert db.summary(start=T0 + timedelta(hours=1))["count"].sum() == 2


def test_sessions_and_cascade_delete(db):
    sessions = db.sessions()
    assert sessions.iloc[0]["vehicles"] == 4
    db.delete_session(int(sessions.iloc[0]["id"]))
    assert db.events().empty


def test_api(db, monkeypatch):
    monkeypatch.setattr(api, "get_db", lambda: db)
    client = TestClient(api.app)
    r = client.get("/stats/summary").json()
    assert r["total"] == 4
    assert r["by_class"] == {"car": 2, "truck": 1, "motorbike": 1}
    assert r["by_direction"] == {"in": 3, "out": 1}
    ts = client.get("/stats/timeseries", params={"interval": "hour"}).json()
    assert [row["total"] for row in ts] == [2, 2]
    assert len(client.get("/events", params={"limit": 2}).json()) == 2
    assert client.get("/sessions").json()[0]["vehicles"] == 4
