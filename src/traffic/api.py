"""API thong ke luu luong phuong tien.

Chay: uvicorn traffic.api:app --app-dir src --port 8000   (tai lieu: /docs)
"""
from datetime import datetime
from typing import Literal

from fastapi import FastAPI, HTTPException, Query

from .classes import CLASS_LABELS_VI, CLASS_NAMES
from .config import load_config, resolve
from .storage import TrafficDB

app = FastAPI(
    title="Traffic Flow Analytics API",
    description="Số liệu đếm & phân loại phương tiện (xe con, xe tải, xe máy) từ hệ thống YOLO26 + ByteTrack.",
    version="1.0.0",
)


def get_db() -> TrafficDB:
    return TrafficDB(resolve(load_config()["storage"]["db_path"]))


@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.now().isoformat(timespec="seconds")}


@app.get("/classes")
def classes():
    return [{"id": i, "name": n, "label": CLASS_LABELS_VI[n]} for i, n in enumerate(CLASS_NAMES)]


@app.get("/sessions")
def sessions():
    return get_db().sessions().to_dict(orient="records")


@app.delete("/sessions/{session_id}")
def delete_session(session_id: int):
    db = get_db()
    if session_id not in set(db.sessions()["id"]):
        raise HTTPException(404, "Không tìm thấy phiên")
    db.delete_session(session_id)
    return {"deleted": session_id}


@app.get("/stats/summary")
def summary(start: datetime | None = None, end: datetime | None = None, session_id: int | None = None):
    df = get_db().summary(start, end, session_id)
    by_class = {c: int(df[df.class_name == c]["count"].sum()) for c in CLASS_NAMES}
    by_dir = {d: int(df[df.direction == d]["count"].sum()) for d in ("in", "out")}
    return {"total": int(df["count"].sum()), "by_class": by_class, "by_direction": by_dir}


@app.get("/stats/timeseries")
def timeseries(interval: Literal["minute", "hour", "day"] = "hour", start: datetime | None = None,
               end: datetime | None = None, session_id: int | None = None):
    df = get_db().timeseries(interval, start, end, session_id)
    if df.empty:
        return []
    wide = df.pivot_table(index="bucket", columns="class_name", values="count", fill_value=0)
    wide = wide.reindex(columns=CLASS_NAMES, fill_value=0).reset_index()
    wide["total"] = wide[CLASS_NAMES].sum(axis=1)
    return wide.astype({c: int for c in CLASS_NAMES + ["total"]}).to_dict(orient="records")


@app.get("/events")
def events(start: datetime | None = None, end: datetime | None = None, session_id: int | None = None,
           limit: int = Query(100, ge=1, le=10000)):
    return get_db().events(start, end, session_id, limit).to_dict(orient="records")
