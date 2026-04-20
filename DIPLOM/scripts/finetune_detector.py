"""Дообучение уже натренированного yolo26_plate.pt на combined датасете
(AUTO.RIA + keremberke HF parking). Цель — закрыть domain-gap на парковочных
ракурсах при сохранении качества на близких планах.

Использование:
    python scripts/finetune_detector.py --epochs 15 --batch 16
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--data", default="D:/Новая папка/alpr_data/yolo_plates_combined/data.yaml")
    ap.add_argument("--base", default="models/yolo26_plate.pt")
    ap.add_argument("--name", default="ua_plates_combined")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    base = root / args.base
    if not base.exists():
        sys.exit(f"базовые веса не найдены: {base}")

    print(f"[finetune] base={base}")
    print(f"[finetune] data={args.data}")
    model = YOLO(str(base))
    save_dir = root / "runs" / "detect" / args.name
    weights_dir = save_dir / "weights"

    try:
        model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=0,
            project=str(root / "runs" / "detect"),
            name=args.name,
            patience=6,
            save=True,
            plots=True,
            cache=False,
            deterministic=True,
            seed=0,
            close_mosaic=3,
            # для parking-kadrov нужна широкая аугментация масштаба и ракурсов
            degrees=8, translate=0.15, scale=0.5,
            hsv_h=0.02, hsv_s=0.6, hsv_v=0.4,
            fliplr=0.0, mosaic=1.0,
            exist_ok=True,
            lr0=0.005,        # меньше — fine-tune, не переобучение
        )
    except Exception as exc:
        if not (weights_dir / "best.pt").exists():
            raise
        print(f"[warn] train interrupted after best.pt was saved: {exc!r}")

    metrics = model.val(data=args.data)
    print("\n=== Метрики (combined val) ===")
    print(f"mAP50    : {metrics.box.map50:.4f}")
    print(f"mAP50-95 : {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall   : {metrics.box.mr:.4f}")

    best = weights_dir / "best.pt"
    target = root / "models" / "yolo26_plate_combined.pt"
    target.parent.mkdir(exist_ok=True)
    shutil.copy2(best, target)
    print(f"\nsaved → {target} ({target.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
