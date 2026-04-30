"""Нарезать OCR-кропы из AUTO.RIA с real GT текстом.

Использует ocr_train.csv / ocr_val.csv, которые генерирует convert_to_yolo.py
(поля: image, x1, y1, x2, y2, text).

Кропы кладутся в data/ocr_crops/{train,val}/ с labels_{split}.csv (file, text).
Этот формат напрямую ест train-цикл в notebooks/02b_train_crnn_ocr.ipynb.

Использование:
    python scripts/prepare_ocr_crops.py \
        --yolo-root C:/alpr_data/yolo_plates \
        --images-root C:/alpr_data/autoriaNumberplateDataset-2026-01-13 \
        --out data/ocr_crops
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import preprocess
from src.io_utils import imread, imwrite


def tight_crop_by_edges(crop_bgr: np.ndarray, margin: float = 0.05) -> np.ndarray:
    """Обрезать рамку и фон: ищем "текстовую полосу" по проекции Canny-краёв.

    Fallback, когда 4-corner rectify не срабатывает: сумма edge-интенсивности
    вдоль строк/колонок → обрезка до активной области.
    """
    if crop_bgr.size == 0:
        return crop_bgr
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    h, w = edges.shape
    # Строка считается "text-line", если на ней > thresh % пикселей — edge.
    row_sum = edges.sum(axis=1) / max(255 * w, 1)
    col_sum = edges.sum(axis=0) / max(255 * h, 1)
    row_thr = max(0.03, row_sum.mean() * 0.6)
    col_thr = max(0.02, col_sum.mean() * 0.3)
    rows = np.where(row_sum > row_thr)[0]
    cols = np.where(col_sum > col_thr)[0]
    if len(rows) < 3 or len(cols) < 3:
        return crop_bgr
    y1, y2 = rows[0], rows[-1]
    x1, x2 = cols[0], cols[-1]
    # небольшой отступ, чтобы не срезать крайние символы
    mh = int((y2 - y1) * margin)
    mw = int((x2 - x1) * margin)
    y1 = max(0, y1 - mh); y2 = min(h, y2 + mh)
    x1 = max(0, x1 - mw); x2 = min(w, x2 + mw)
    if (y2 - y1) < h * 0.2 or (x2 - x1) < w * 0.3:
        return crop_bgr  # слишком маленький — не доверяем
    return crop_bgr[y1:y2, x1:x2]


def process(split: str, ocr_csv: Path, images_root: Path, out_dir: Path, rectify: bool, override_rows: list | None = None, src_split_hint: str | None = None) -> int:
    split_dir = out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    if override_rows is not None:
        rows = override_rows
    else:
        if not ocr_csv.exists():
            print(f"[skip] {split}: {ocr_csv} не найден")
            return 0
        with open(ocr_csv, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    out_rows: list[tuple[str, str, str]] = []
    for i, row in enumerate(tqdm(rows, desc=f"crop {split}")):
        image_name = row["image"]
        text = row["text"].strip()
        rname = row.get("region_name", "").strip()
        if not text:
            continue
        # Источник картинки: либо указан explicit (при merge), либо по split
        candidates = []
        if src_split_hint:
            candidates.append(images_root / src_split_hint / image_name)
        if row.get("_src_split"):
            candidates.append(images_root / row["_src_split"] / image_name)
        candidates.extend([images_root / "train" / image_name,
                           images_root / "val" / image_name,
                           images_root / image_name])
        src_path = next((c for c in candidates if c.exists()), None)
        if src_path is None:
            continue
        img = imread(src_path)
        if img is None:
            continue
        x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
        crop = img[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            continue
        if rectify:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            corners = preprocess.find_plate_corners(rgb)
            if corners is not None:
                rect = preprocess.rectify_plate(rgb, corners, out_size=(260, 80))
                crop = cv2.cvtColor(rect, cv2.COLOR_RGB2BGR)
            else:
                crop = tight_crop_by_edges(crop)
        out_name = f"{Path(image_name).stem}_{i}.jpg"
        imwrite(split_dir / out_name, crop)
        out_rows.append((out_name, text, rname))

    csv_out = out_dir / f"labels_{split}.csv"
    with open(csv_out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(("file", "text", "kind"))
        w.writerows(out_rows)
    print(f"[{split}] {len(out_rows)} кропов → {csv_out}")
    return len(out_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo-root", required=True, help="результат convert_to_yolo.py")
    ap.add_argument("--images-root", required=True, help="корень AUTO.RIA с train/ и val/")
    ap.add_argument("--out", default="data/ocr_crops")
    ap.add_argument("--rectify", action="store_true",
                    help="Применить Harris+Homography rectify (ДЗ6+7); fallback — tight-crop по краям")
    ap.add_argument("--resplit", type=float, default=None,
                    help="Объединить train+val AUTO.RIA и сделать random split (например 0.15 = 85%% train / 15%% val)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    yolo_root = Path(args.yolo_root)
    images_root = Path(args.images_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.resplit is None:
        n_train = process("train", yolo_root / "ocr_train.csv", images_root, out_dir, args.rectify, src_split_hint="train")
        n_val = process("val", yolo_root / "ocr_val.csv", images_root, out_dir, args.rectify, src_split_hint="val")
    else:
        import random
        all_rows = []
        for split_name in ("train", "val"):
            csv_path = yolo_root / f"ocr_{split_name}.csv"
            if not csv_path.exists():
                continue
            with open(csv_path, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    row["_src_split"] = split_name
                    all_rows.append(row)
        random.Random(args.seed).shuffle(all_rows)
        n_val_new = int(len(all_rows) * args.resplit)
        val_rows = all_rows[:n_val_new]
        train_rows = all_rows[n_val_new:]
        print(f"[resplit] merged={len(all_rows)}, new train={len(train_rows)}, new val={len(val_rows)} ({args.resplit:.0%})")
        n_train = process("train", Path(), images_root, out_dir, args.rectify, override_rows=train_rows)
        n_val = process("val", Path(), images_root, out_dir, args.rectify, override_rows=val_rows)
    print(f"\nDone. train={n_train}, val={n_val}")


if __name__ == "__main__":
    main()
