"""Chuan hoa va kiem tra bo du lieu Vietnamese vehicle v3 (Roboflow) ve 3 lop car / truck / motorbike.

    python scripts/prepare_dataset.py            # chi kiem tra + thong ke
    python scripts/prepare_dataset.py --remap    # anh xa nhan goc 4 lop -> 3 lop (chay lai nhieu lan van an toan)

Nhan goc cua Roboflow: 0=car, 1=motorbike, 2=truck, 3=bus. Bus duoc gop vao truck (nhom xe co lon),
vi de tai chi yeu cau 3 lop va bus chi chiem ~5.6% so box.
"""
import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from traffic.classes import CLASS_NAMES  # noqa: E402

DATASET = ROOT / "Data" / "Vietnamese vehicle.v3-2023-02-01-5-31pm.yolo26"
BACKUP = ROOT / "Data" / "processed_labels_backup"
SPLITS = ["train", "valid", "test"]
RAW_NAMES = ["car", "motorbike", "truck", "bus"]
RAW_TO_FINAL = {0: 0, 1: 2, 2: 1, 3: 1}  # -> index trong CLASS_NAMES


def remap():
    marker = BACKUP / "DONE.marker"
    if not marker.exists():
        for split in SPLITS:
            shutil.copytree(DATASET / split / "labels", BACKUP / split / "labels", dirs_exist_ok=True)
        marker.write_text("backed up\n", encoding="utf-8")
        print(f"Da sao luu nhan goc vao {BACKUP}")

    # Luon doc tu ban sao luu (nhan goc) nen chay lai nhieu lan khong bi anh xa chong
    for split in SPLITS:
        n = 0
        for src in (BACKUP / split / "labels").glob("*.txt"):
            lines = [ln.split() for ln in src.read_text(encoding="utf-8").splitlines() if ln.strip()]
            out = [" ".join([str(RAW_TO_FINAL[int(p[0])]), *p[1:]]) for p in lines]
            (DATASET / split / "labels" / src.name).write_text("\n".join(out) + "\n", encoding="utf-8")
            n += 1
        print(f"[{split}] da anh xa {n} file nhan")

    cfg_path = DATASET / "data.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg.update(nc=len(CLASS_NAMES), names=CLASS_NAMES)
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    for cache in DATASET.glob("*/labels.cache"):
        cache.unlink()


def check() -> bool:
    ok = True
    total = Counter()
    print(f"{'split':<6} {'anh':>5} {'nhan':>5} " + " ".join(f"{c:>9}" for c in CLASS_NAMES) + f" {'tong box':>9}")
    for split in SPLITS:
        imgs = {p.stem for p in (DATASET / split / "images").glob("*") if p.suffix.lower() in {".jpg", ".png", ".jpeg"}}
        labels = list((DATASET / split / "labels").glob("*.txt"))
        cnt = Counter()
        for f in labels:
            for i, ln in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                p = ln.split()
                if not p:
                    continue
                cls, coords = int(p[0]), list(map(float, p[1:]))
                if len(p) != 5 or not 0 <= cls < len(CLASS_NAMES) or not all(0 <= v <= 1 for v in coords):
                    print(f"  LOI {f.name}:{i} -> {ln}")
                    ok = False
                    continue
                cnt[cls] += 1
        missing = imgs - {f.stem for f in labels}
        if missing:
            print(f"  [{split}] {len(missing)} anh khong co file nhan (anh nen, khong co xe)")
        total += cnt
        print(f"{split:<6} {len(imgs):>5} {len(labels):>5} " + " ".join(f"{cnt[i]:>9}" for i in range(len(CLASS_NAMES)))
              + f" {sum(cnt.values()):>9}")
    print(f"{'tong':<6} {'':>5} {'':>5} " + " ".join(f"{total[i]:>9}" for i in range(len(CLASS_NAMES)))
          + f" {sum(total.values()):>9}")
    print("Kiem tra nhan: " + ("OK" if ok else "CO LOI"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--remap", action="store_true", help="Anh xa nhan goc 4 lop ve 3 lop")
    args = ap.parse_args()
    if not DATASET.exists():
        sys.exit(f"Khong tim thay {DATASET}. Tai bo du lieu tu Roboflow (xem README) va giai nen vao Data/.")
    if args.remap:
        remap()
    sys.exit(0 if check() else 1)


if __name__ == "__main__":
    main()
