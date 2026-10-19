"""Kiem tra trung lap / gan trung lap giua train, valid, test -> models/vehicle_yolo26/leakage_check.json.

Ba phep kiem tra:
  1. MD5 noi dung file: phat hien anh giong het nhau.
  2. Ten anh goc Roboflow (so truoc "_jpg.rf."): phat hien cung mot anh goc nam o hai tap.
  3. Hash cam nhan dHash 16x16 (256 bit): phat hien anh gan giong (cung canh, khung ke tiep trong video).
     Voi moi anh test/valid, lay khoang cach Hamming toi anh train gan nhat. Lam moc so sanh bang cung phep do
     giua cac anh train voi nhau (bo qua chinh no).
Neu co anh test gan trung lap train (khoang cach <= --thresh), danh gia lai mAP tren tap test da loai cac anh do.
"""
import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from traffic.config import load_config, resolve  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png"}
HASH_N = 16


def list_images(folder: Path):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTS)


def dhash(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    small = cv2.resize(img, (HASH_N + 1, HASH_N), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).flatten()


def dhash_flip(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    small = cv2.resize(cv2.flip(img, 1), (HASH_N + 1, HASH_N), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).flatten()


def contact_sheet(pairs, path: Path, tile=(320, 240)):
    """Moi hang: anh test (trai) | anh train gan nhat (phai), ghi khoang cach len anh."""
    rows = []
    for q, r, d in pairs:
        a, b = (cv2.resize(cv2.imdecode(np.fromfile(str(x), dtype=np.uint8), cv2.IMREAD_COLOR), tile) for x in (q, r))
        cv2.putText(a, f"d={d}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        rows.append(np.hstack([a, b]))
    cv2.imencode(".jpg", np.vstack(rows))[1].tofile(str(path))


def thumb(path: Path) -> np.ndarray:
    """Anh nho 32x24 lam mo, tru trung binh, chuan hoa: dac trung bo cuc/nen (it phu thuoc xe nho)."""
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    v = cv2.GaussianBlur(cv2.resize(img, (32, 24), interpolation=cv2.INTER_AREA), (5, 5), 0).astype(np.float32).ravel()
    v -= v.mean()
    return v / (np.linalg.norm(v) + 1e-6)


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def source_name(path: Path) -> str:
    return re.split(r"_(?:jpg|jpeg|png)\.rf\.", path.name, maxsplit=1)[0]


def hamming_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a[:, None, :] != b[None, :, :]).sum(axis=2)


def nearest(query: np.ndarray, ref: np.ndarray, skip_self=False):
    d = hamming_matrix(query, ref)
    if skip_self:
        np.fill_diagonal(d, 10 ** 6)
    return d.argmin(axis=1), d.min(axis=1)


def dist_summary(d: np.ndarray, thresholds):
    return {"min": int(d.min()), "p5": float(np.percentile(d, 5)), "median": float(np.median(d)),
            **{f"so_anh_<=_{t}": int((d <= t).sum()) for t in thresholds}}


def eval_without(weights, data_yaml, img_paths, imgsz):
    """Danh gia mAP tren danh sach anh cho truoc (Ultralytics tim nhan bang cach doi /images/ -> /labels/)."""
    import tempfile

    import yaml
    from ultralytics import YOLO
    with tempfile.TemporaryDirectory(prefix="leakage_") as d:
        tmp = Path(d)
        lst = tmp / "test_filtered.txt"
        lst.write_text("\n".join(str(p.resolve()) for p in img_paths), encoding="utf-8")
        cfg = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
        cfg.update({"path": str(tmp), "train": str(lst), "val": str(lst), "test": str(lst)})
        ytmp = tmp / "data.yaml"
        ytmp.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
        m = YOLO(str(weights)).val(data=str(ytmp), split="test", imgsz=imgsz, workers=0, plots=False, verbose=False,
                                   project=str(tmp), name="val")
        return {"so_anh": len(img_paths), "mAP50": float(m.box.map50), "mAP50_95": float(m.box.map),
                "precision": float(m.box.mp), "recall": float(m.box.mr)}


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresh", type=int, default=10, help="khoang cach Hamming (tren 256 bit) coi la gan trung lap")
    ap.add_argument("--weights", default=cfg["model"]["weights"])
    args = ap.parse_args()

    data_yaml = (ROOT / cfg["train"]["data"]).resolve()
    import yaml
    base = ROOT / yaml.safe_load(data_yaml.read_text(encoding="utf-8"))["path"]
    splits = {s: list_images(base / s / "images") for s in ("train", "valid", "test")}
    print({s: len(v) for s, v in splits.items()})

    md5s = {s: [md5(p) for p in v] for s, v in splits.items()}
    srcs = {s: [source_name(p) for p in v] for s, v in splits.items()}
    hashes = {s: np.stack([dhash(p) for p in v]) for s, v in splits.items()}

    out = {"so_anh": {s: len(v) for s, v in splits.items()}, "nguong_gan_trung_lap": args.thresh,
           "hash": f"dHash {HASH_N}x{HASH_N} = {HASH_N * HASH_N} bit"}

    # 1) md5 va 2) ten anh goc
    out["md5_trung"] = {}
    out["ten_goc_trung"] = {}
    for a, b in (("test", "train"), ("valid", "train"), ("test", "valid")):
        out["md5_trung"][f"{a}_voi_{b}"] = len(set(md5s[a]) & set(md5s[b]))
        out["ten_goc_trung"][f"{a}_voi_{b}"] = len(set(srcs[a]) & set(srcs[b]))
    out["md5_trung_trong_train"] = len(md5s["train"]) - len(set(md5s["train"]))

    out["ten_goc_duy_nhat_trong_train"] = len(set(srcs["train"]))
    flip = {s: np.stack([dhash_flip(p) for p in splits[s]]) for s in ("valid", "test")}

    # 3) dHash
    ths = sorted({0, 2, 5, args.thresh, 20, 30})
    _, d_tt = nearest(hashes["train"], hashes["train"], skip_self=True)
    out["moc_train_voi_train"] = dist_summary(d_tt, ths)
    flagged = {}
    for s in ("valid", "test"):
        idx, d = nearest(hashes[s], hashes["train"])
        out[f"{s}_voi_train"] = dist_summary(d, ths)
        _, df = nearest(flip[s], hashes["train"])
        out[f"{s}_lat_ngang_voi_train"] = dist_summary(df, ths)
        if s == "test":
            order = np.argsort(d)[:8]
            sheet = ROOT / "docs" / "images" / "24_cap_anh_test_gan_train_nhat.jpg"
            contact_sheet([(splits[s][i], splits["train"][idx[i]], int(d[i])) for i in order], sheet)
            out["contact_sheet"] = str(sheet.relative_to(ROOT))
        flagged[s] = [{"anh": splits[s][i].name, "anh_train_gan_nhat": splits["train"][idx[i]].name,
                       "khoang_cach": int(d[i])} for i in np.argsort(d) if d[i] <= args.thresh]
    out["anh_nghi_trung"] = flagged
    print(json.dumps({k: v for k, v in out.items() if k != "anh_nghi_trung"}, indent=2, ensure_ascii=False))
    print("Anh nghi trung: valid", len(flagged["valid"]), " test", len(flagged["test"]))
    for r in flagged["test"][:20]:
        print("  ", r)

    # 4) Do giong ve bo cuc/nen (cung camera?): tuong quan anh nho; mAP test tach theo muc giong train
    th = {s: np.stack([thumb(p) for p in splits[s]]) for s in ("train", "test")}
    sim = th["test"] @ th["train"].T
    best = sim.argmax(axis=1)
    cos = sim.max(axis=1)
    sim_tt = th["train"] @ th["train"].T
    np.fill_diagonal(sim_tt, -1)
    out["tuong_quan_bo_cuc"] = {"test_voi_train_trung_vi": float(np.median(cos)), "test_p10": float(np.percentile(cos, 10)),
                                "train_voi_train_trung_vi": float(np.median(sim_tt.max(axis=1)))}
    rng = np.random.default_rng(0)
    pick = rng.choice(len(splits["test"]), size=16, replace=False)
    sheet2 = ROOT / "docs" / "images" / "25_mau_ngau_nhien_test_va_anh_train_gan_nhat.jpg"
    contact_sheet([(splits["test"][i], splits["train"][best[i]], f"{cos[i]:.2f}") for i in pick], sheet2, tile=(256, 192))
    out["contact_sheet_mau_ngau_nhien"] = str(sheet2.relative_to(ROOT))
    out["mau_ngau_nhien_test"] = [splits["test"][i].name for i in pick]
    # tach test thanh nua "giong train nhat" va nua "it giong" theo tuong quan bo cuc
    order = np.argsort(-cos)
    half = len(order) // 2
    near, far = [splits["test"][i] for i in order[:half]], [splits["test"][i] for i in order[half:]]
    out["map_test_nua_giong_train_nhat"] = eval_without(resolve(args.weights), data_yaml, near, cfg["model"]["imgsz"])
    out["map_test_nua_it_giong_train"] = eval_without(resolve(args.weights), data_yaml, far, cfg["model"]["imgsz"])
    print(json.dumps({k: out[k] for k in ("tuong_quan_bo_cuc", "map_test_nua_giong_train_nhat", "map_test_nua_it_giong_train")},
                     indent=2, ensure_ascii=False))

    # Danh gia lai mAP khi loai cac anh test nghi trung
    bad = {r["anh"] for r in flagged["test"]}
    if bad:
        keep = [p for p in splits["test"] if p.name not in bad]
        out["map_test_day_du"] = eval_without(resolve(args.weights), data_yaml, splits["test"], cfg["model"]["imgsz"])
        out["map_test_da_loai_anh_nghi_trung"] = eval_without(resolve(args.weights), data_yaml, keep, cfg["model"]["imgsz"])
        print(json.dumps({k: out[k] for k in ("map_test_day_du", "map_test_da_loai_anh_nghi_trung")},
                         indent=2, ensure_ascii=False))

    path = ROOT / "models" / "vehicle_yolo26" / "leakage_check.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("ghi", path)


if __name__ == "__main__":
    main()
