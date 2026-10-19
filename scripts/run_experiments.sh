#!/usr/bin/env bash
# Chuoi thi nghiem tang mAP@0.5:0.95: cung cong thuc augmentation, doi model (n/s) va imgsz (640/960).
# Moi run luu vao models/exp_<ten>/, danh gia tren tap test o DUNG imgsz luc train.
set -u
cd "$(dirname "$0")/.."
PY=venv/Scripts/python
COMMON="--epochs 150 --patience 30 --close-mosaic 20 --scale 0.7 --mixup 0.1"
run() {  # ten model imgsz batch
  echo "=== $(date) START $1 ==="
  $PY -u scripts/train.py --name "exp_$1" --model "$2" --imgsz "$3" --batch "$4" $COMMON > "logs/train_exp_$1.log" 2>&1 \
    && $PY -u scripts/evaluate.py --weights "models/exp_$1/weights/best.pt" --imgsz "$3" > "logs/eval_exp_$1.log" 2>&1
  local rc=$?
  echo "=== $(date) DONE $1 (exit $rc) ==="
}
run n640_aug yolo26n.pt 640 16
run s640     yolo26s.pt 640 8
run n960     yolo26n.pt 960 8
run s960     yolo26s.pt 960 4
