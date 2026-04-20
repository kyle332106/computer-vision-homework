from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.io_utils import imread


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="runs/eval/rtsp_probe.jpg")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    img = imread(args.image)
    if img is None:
        raise SystemExit(f"image not found: {args.image}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    models = [
        ("old", "models/yolo26_plate.pt"),
        ("combined", "models/yolo26_plate_combined.pt"),
    ]
    for name, path in models:
        model = YOLO(path)
        print(f"=== {name} ===")
        for imgsz in (1280, 1536, 1920):
            res = model.predict(
                rgb,
                imgsz=imgsz,
                conf=0.15,
                iou=0.4,
                verbose=False,
                device=args.device,
            )[0]
            confs = [round(float(b.conf[0]), 2) for b in res.boxes]
            print(f"imgsz={imgsz} count={len(confs)} confs={confs}")
            for i, b in enumerate(res.boxes[:6]):
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                print(f"  #{i} bbox=({x1},{y1},{x2},{y2})")
        print()


if __name__ == "__main__":
    main()
