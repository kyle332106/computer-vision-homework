"""Fine-tune YOLO26n на AUTO.RIA UA — эквивалент 02_train_detector.ipynb в CLI-форме.

Использование:
    python scripts/train_detector.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import yaml
from ultralytics import YOLO


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    cfg = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    data_yaml = Path(cfg["yolo_dataset_dir"]) / "data.yaml"
    if not data_yaml.exists():
        sys.exit(f"data.yaml не найден: {data_yaml} — сначала convert_to_yolo.py")

    base = root / "models" / "yolo26n.pt"
    base_arg = str(base) if base.exists() else "yolo26n.pt"

    model = YOLO(base_arg)
    det = cfg["detector"]

    results = model.train(
        data=str(data_yaml),
        epochs=det["epochs"],
        imgsz=det["imgsz"],
        batch=det["batch"],
        device=det["device"],
        project=str(root / "runs" / "detect"),
        name="ua_plates_yolo26n",
        patience=10,
        save=True,
        plots=True,
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.3,
        degrees=5, translate=0.1, scale=0.3,
        fliplr=0.0,
        mosaic=1.0, mixup=0.0,
        exist_ok=True,
    )

    metrics = model.val(data=str(data_yaml))
    print("\n=== Метрики ===")
    print(f"mAP50    : {metrics.box.map50:.4f}")
    print(f"mAP50-95 : {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall   : {metrics.box.mr:.4f}")

    best = root / "runs" / "detect" / "ua_plates_yolo26n" / "weights" / "best.pt"
    target = root / "models" / "yolo26_plate.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, target)
    print(f"\nБест-веса → {target} ({target.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
