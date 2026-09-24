"""Test tich hop: chay pipeline that tren video mau (bo qua neu thieu weights hoac video)."""
from datetime import datetime

import pytest

from traffic import PROJECT_ROOT
from traffic.pipeline import TrafficPipeline, default_weights, load_model
from traffic.storage import TrafficDB

VIDEO = PROJECT_ROOT / "Data" / "videos" / "demo_traffic_sample.mp4"
T0 = datetime(2026, 9, 1, 7, 0, 0)

pytestmark = pytest.mark.skipif(not VIDEO.exists() or not default_weights().exists(),
                                reason="thieu video mau hoac weights")


def test_pipeline_end_to_end(tmp_path):
    db = TrafficDB(tmp_path / "t.db")
    pipe = TrafficPipeline(model=load_model(), db=db)
    out = tmp_path / "out.mp4"
    last = None
    for last in pipe.run(str(VIDEO), [[0.05, 0.6], [0.95, 0.6]], start_time=T0, output_path=str(out),
                         max_frames=450):
        pass

    assert last.frame_idx == 450
    assert last.total > 0, "15 giay dau video phai co xe qua vach"
    assert out.exists() and out.stat().st_size > 0

    session = db.sessions().iloc[0]
    assert session["frames"] == 450 and session["vehicles"] == last.total
    events = db.events()
    assert events["track_id"].is_unique
    assert set(events["class_name"]) <= {"car", "truck", "motorbike"}
    assert events["ts"].min() >= "2026-09-01 07:00:00" and events["ts"].max() <= "2026-09-01 07:00:15"
