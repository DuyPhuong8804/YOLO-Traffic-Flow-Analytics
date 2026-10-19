"""Do do tre API, dashboard va nhip cap nhat khi phan tich truc tiep -> models/vehicle_yolo26/latency.json.

Ba phan:
  1. API: khoi dong uvicorn trong tien trinh rieng, goi HTTP that (loopback) tung endpoint N lan sau khi khoi dong nhiet.
     Do tren CSDL that (Data/traffic_analytics.db) va hai CSDL tong hop (10.000 / 100.000 su kien) de xem do tre
     tang the nao khi du lieu lon len. CSDL tong hop chi de do quy mo, KHONG phai du lieu giao thong.
  2. Dashboard: chay trang views/dashboard.py bang streamlit.testing.v1.AppTest (khong can trinh duyet), do thoi gian
     moi lan chay lai script (lan dau + cac lan sau). Chi gom phan o phia may chu: doc CSDL, xu ly pandas, dung bieu do;
     KHONG gom thoi gian trinh duyet ve bieu do.
  3. Phan tich truc tiep: thoi gian giua hai khung hinh ket qua lien tiep cua pipeline (doc + detect + track + dem + ve),
     tren mot video that, khong ghi CSDL. Day la nhip ma trang "Phan tich video" nhan duoc khung moi.
Ket qua chi dung cho may phat trien (RTX 3050, Windows 11), khong co mang that giua may khach va may chu.
"""
import argparse
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))
os.chdir(ROOT)

import numpy as np  # noqa: E402
import requests  # noqa: E402

from traffic.classes import CLASS_NAMES  # noqa: E402
from traffic.config import load_config, resolve  # noqa: E402
from traffic.storage import TrafficDB  # noqa: E402

PORT = 8765
WARMUP, N_API, N_DASH = 5, 50, 20
SYNTH_SIZES = (10_000, 100_000)


def pct(ms):
    a = np.asarray(ms, dtype=float)
    return {"n": int(a.size), "min": float(a.min()), "median": float(np.median(a)), "p95": float(np.percentile(a, 95)),
            "max": float(a.max())}


def make_synthetic_db(path: Path, n_events: int, seed=0):
    """CSDL gia: 20 phien, su kien rai deu trong 7 ngay (chi de do quy mo)."""
    rnd = random.Random(seed)
    db = TrafficDB(path)
    base = datetime(2026, 1, 1)
    sids = [db.start_session(f"synthetic_{i}", base + timedelta(days=i % 7), model="synthetic") for i in range(20)]
    batch = []
    for k in range(n_events):
        sid = sids[k % len(sids)]
        ts = base + timedelta(days=(k % len(sids)) % 7, seconds=rnd.randrange(86400))
        batch.append((k, rnd.choice(CLASS_NAMES), rnd.choice(("in", "out")), k, ts))
        if len(batch) == 5000:
            db.add_events(sid, batch)
            batch = []
    if batch:
        db.add_events(sids[0], batch)
    return db


# ---------------------------------------------------------------- phan 1: API
def serve(db_path: str, port: int):
    """Chay API voi CSDL chi dinh (che do con, goi lai chinh script nay)."""
    import uvicorn

    import traffic.api as api
    api.get_db = lambda: TrafficDB(db_path)
    uvicorn.run(api.app, host="127.0.0.1", port=port, log_level="warning")


def measure_api(db_path: Path, session_id: int):
    proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--serve", str(db_path), str(PORT)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{PORT}"
    try:
        for _ in range(100):
            try:
                if requests.get(f"{base}/health", timeout=1).ok:
                    break
            except requests.RequestException:
                time.sleep(0.2)
        else:
            raise RuntimeError("API khong khoi dong duoc")
        endpoints = {
            "/health": "/health",
            "/sessions": "/sessions",
            "/stats/summary": "/stats/summary",
            "/stats/summary?session_id": f"/stats/summary?session_id={session_id}",
            "/stats/timeseries?interval=minute": "/stats/timeseries?interval=minute",
            "/events?limit=100": "/events?limit=100",
        }
        sess = requests.Session()
        out = {}
        for name, url in endpoints.items():
            for _ in range(WARMUP):
                sess.get(base + url, timeout=60).raise_for_status()
            ms = []
            for _ in range(N_API):
                t = time.perf_counter()
                r = sess.get(base + url, timeout=60)
                ms.append((time.perf_counter() - t) * 1000)
                r.raise_for_status()
            out[name] = pct(ms)
            print(f"   {name:36s} median {out[name]['median']:7.1f} ms  p95 {out[name]['p95']:7.1f}  max {out[name]['max']:7.1f}",
                  flush=True)
        return out
    finally:
        proc.terminate()
        proc.wait(10)


# ---------------------------------------------------------------- phan 2: dashboard
def measure_dashboard(db: TrafficDB):
    from streamlit.testing.v1 import AppTest

    import ui
    ui.get_db = lambda: db  # trang dashboard nhap `get_db` tu module ui khi chay
    at = AppTest.from_file(str(ROOT / "app" / "views" / "dashboard.py"), default_timeout=120)
    ms = []
    for i in range(N_DASH + 1):
        t = time.perf_counter()
        at.run()
        ms.append((time.perf_counter() - t) * 1000)
        if at.exception:
            raise RuntimeError(f"Dashboard loi: {at.exception[0].value}")
    return {"lan_dau_ms": ms[0], "cac_lan_sau": pct(ms[1:]), "so_bieu_do_plotly": len(at.get("plotly_chart"))}


# ---------------------------------------------------------------- phan 3: phan tich truc tiep
def measure_live(video: str, max_frames: int):
    from traffic.pipeline import TrafficPipeline
    cfg = load_config()
    pipe = TrafficPipeline()  # khong ghi CSDL
    gaps, last = [], None
    for res in pipe.run(video, cfg["counting"]["line"], max_frames=max_frames):
        now = time.perf_counter()
        if last is not None:
            gaps.append((now - last) * 1000)
        last = now
    gaps = gaps[20:]  # bo 20 khung khoi dong nhiet (giong benchmark.py)
    a = np.asarray(gaps)
    return {"video": Path(video).name, "so_khung_do": int(a.size), **pct(gaps), "p99": float(np.percentile(a, 99)),
            "fps_trung_binh": float(1000 / a.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", nargs=2, metavar=("DB", "PORT"), help="(noi bo) chay API voi CSDL chi dinh")
    ap.add_argument("--video", default="Data/videos/vn_giao_thong_ban_ngay.mp4")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--skip-live", action="store_true")
    args = ap.parse_args()
    if args.serve:
        return serve(args.serve[0], int(args.serve[1]))

    real = TrafficDB(resolve(load_config()["storage"]["db_path"]))
    sess = real.sessions()
    out = {"may": f"{platform.system()} {platform.release()}, Python {platform.python_version()}",
           "so_lan_do_api": N_API, "so_lan_khoi_dong_nhiet": WARMUP, "so_lan_do_dashboard": N_DASH,
           "api": {}, "dashboard": {}}

    dbs = {"csdl_that": (resolve(load_config()["storage"]["db_path"]), int(sess.sort_values("vehicles").iloc[-1]["id"]))}
    tmp = Path(tempfile.mkdtemp(prefix="latency_"))
    for n in SYNTH_SIZES:
        p = tmp / f"synthetic_{n}.db"
        make_synthetic_db(p, n)
        dbs[f"tong_hop_{n}_su_kien"] = (p, 1)

    for name, (path, sid) in dbs.items():
        db = TrafficDB(path)
        n_ev, n_se = len(db.events()), len(db.sessions())
        print(f"== API, {name}: {n_ev} su kien, {n_se} phien (session_id mau = {sid})", flush=True)
        out["api"][name] = {"so_su_kien": n_ev, "so_phien": n_se, "endpoint": measure_api(path, sid)}
        print(f"== Dashboard, {name}", flush=True)
        out["dashboard"][name] = {"so_su_kien": n_ev, **measure_dashboard(db)}
        d = out["dashboard"][name]
        print(f"   lan dau {d['lan_dau_ms']:.0f} ms; cac lan sau median {d['cac_lan_sau']['median']:.0f}  "
              f"p95 {d['cac_lan_sau']['p95']:.0f}  max {d['cac_lan_sau']['max']:.0f} ms", flush=True)

    if not args.skip_live:
        print("== Phan tich truc tiep", flush=True)
        out["phan_tich_truc_tiep"] = measure_live(args.video, args.frames)
        l = out["phan_tich_truc_tiep"]
        print(f"   median {l['median']:.1f}  p95 {l['p95']:.1f}  p99 {l['p99']:.1f}  max {l['max']:.1f} ms/khung")

    shutil.rmtree(tmp, ignore_errors=True)
    path = ROOT / "models" / "vehicle_yolo26" / "latency.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("ghi", path)


if __name__ == "__main__":
    main()
