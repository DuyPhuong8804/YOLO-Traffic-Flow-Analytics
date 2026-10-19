"""Do lai cac so lieu trong bao cao khong nam trong file ket qua nao khac (in ra man hinh):
  - ma tran nham lan tren tap test (conf 0.25) -> nhom nham lop, FP nen, FN
  - so xe dem duoc theo lop tren 7 video (khop bang Tuan 8)
  - so xe may phat hien moi khung o imgsz 640 va 1280 tren video ket xe
  - ti le xe may trong nhan test duoc YOLO26n COCO bat o IoU >= 0.1 va >= 0.5
  - thong ke CSDL demo (Data/traffic_analytics.db)
"""
import os, sys, sqlite3
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; os.chdir(ROOT); sys.path.insert(0,str(ROOT/"src"))
import cv2, numpy as np
from ultralytics import YOLO
from traffic.config import load_config, resolve
from traffic.pipeline import TrafficPipeline
TMP=os.environ["TMP"]
def iou(a,b):
    x1=np.maximum(a[:,None,0],b[None,:,0]);y1=np.maximum(a[:,None,1],b[None,:,1]);x2=np.minimum(a[:,None,2],b[None,:,2]);y2=np.minimum(a[:,None,3],b[None,:,3])
    i=np.clip(x2-x1,0,None)*np.clip(y2-y1,0,None);aa=(a[:,2]-a[:,0])*(a[:,3]-a[:,1]);bb=(b[:,2]-b[:,0])*(b[:,3]-b[:,1]);return i/(aa[:,None]+bb[None]-i+1e-9)
def main():
    cfg=load_config(); m=YOLO(str(resolve(cfg["model"]["weights"])))
    # confusion matrix (plots=True so matrix is filled)
    v=m.val(data=cfg["train"]["data"],split="test",imgsz=640,plots=True,verbose=False,workers=0,project=TMP+"/v",name="cm2",exist_ok=True)
    cm=v.confusion_matrix.matrix.astype(int); print("CM rows=pred cols=true [car,truck,motorbike,background]\n",cm,flush=True)
    # per-video counts by class
    for p in sorted(Path("Data/videos").glob("*.mp4")):
        pipe=TrafficPipeline(model=m); last=None
        for r in pipe.run(str(p),cfg["counting"]["line"]): last=r
        c=last.by_class; print("COUNT",p.stem,sum(c.values()),dict(c),flush=True)
    # motorbike dets per frame, 640 vs 1280 on ket_xe
    for sz in (640,1280):
        n=[]
        for r in m.predict("Data/videos/vn_ket_xe_xe_may_o_to.mp4",imgsz=sz,conf=0.35,iou=0.5,stream=True,verbose=False):
            n.append(int((r.boxes.cls==2).sum()))
        print("MOTO/frame imgsz",sz,"mean",round(float(np.mean(n)),2),"median",float(np.median(n)),"max",max(n),"frame0",n[0],flush=True)
    # COCO baseline IoU for motorbike GT
    coco=YOLO("yolo26n.pt"); D=Path("Data/Vietnamese vehicle.v3-2023-02-01-5-31pm.yolo26/test")
    best=[]
    for img in sorted((D/"images").glob("*")):
        lab=D/"labels"/(img.stem+".txt")
        im=cv2.imread(str(img)); h,w=im.shape[:2]
        gt=[l.split() for l in lab.read_text().splitlines() if l.strip()]
        gt=np.array([[float(x) for x in g[1:5]] for g in gt if g[0]=="2"]) if any(g[0]=="2" for g in gt) else np.zeros((0,4))
        if not len(gt): continue
        g=np.stack([(gt[:,0]-gt[:,2]/2)*w,(gt[:,1]-gt[:,3]/2)*h,(gt[:,0]+gt[:,2]/2)*w,(gt[:,1]+gt[:,3]/2)*h],1)
        r=coco.predict(im,imgsz=640,conf=0.1,iou=0.7,verbose=False)[0]
        pb=r.boxes.xyxy.cpu().numpy()[(r.boxes.cls==3).cpu().numpy()]
        best+=list(iou(g,pb).max(1)) if len(pb) else [0.0]*len(g)
    best=np.array(best); print("COCO moto GT n",len(best),"IoU>=0.1",round(float((best>=0.1).mean()),3),">=0.5",round(float((best>=0.5).mean()),3),flush=True)
    # DB
    db=sqlite3.connect("Data/traffic_analytics.db"); print("DB tables",db.execute("select name from sqlite_master where type='table'").fetchall())
    print("DB class/dir",db.execute("select class_name,direction,count(*) from vehicle_events group by 1,2").fetchall())
    print("DB sessions",db.execute("select count(*) from sessions").fetchall())
if __name__=="__main__": main()
