"""AUTO.RIA Numberplate Dataset (VIA JSON polygons) → YOLO + OCR labels.

Использование:
    python scripts/convert_to_yolo.py \
        --source C:/alpr_data/autoriaNumberplateDataset-2026-01-13 \
        --dest   C:/alpr_data/yolo_plates \
        --subset 4000

Что делает:
  • Парсит via_region_data.json из train/ и val/
  • Берёт только regions с class='numberplate' (игнорирует emptyPlate, пустые)
  • Полигоны → AABB (axis-aligned bounding box) в YOLO-нормализованном формате
  • Копирует изображения, создаёт <imgname>.txt рядом (YOLO-разметка)
  • ДОП: пишет ocr_<split>.csv с полями (image, x1, y1, x2, y2, text)
         для обучения CRNN+CTC на реальных кропах с GT-текстом
  • Генерирует data.yaml с одним классом license_plate
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.io_utils import imread


PLATE_CLASSES = {"numberplate", "numberplate "}   # некоторые записи с хвостовым пробелом
EXCLUDE_CLASSES = {"emptyPlate", "emptyPlate "}    # нечитаемые/заблюренные — не нужны
TEXT_KEYS = ("np", "description", "text", "plate_number", "number")   # np — фактическое поле AUTO.RIA


def _is_plate(attrs: dict) -> bool:
    """Считаем регион номером, если:
      class='numberplate', ИЛИ
      присутствует поле `np` (текст номера — значит это точно numberplate), ИЛИ
      label='numberplate' и нет противоречащего class='emptyPlate'.
    Явно исключаем emptyPlate (заблюренные/нечитаемые)."""
    cls = attrs.get("class", "")
    if isinstance(cls, str) and cls.strip() in EXCLUDE_CLASSES:
        return False
    if isinstance(cls, str) and cls.strip() in PLATE_CLASSES:
        return True
    if attrs.get("np"):
        return True
    label = attrs.get("label", "")
    if isinstance(label, str) and label.strip() in PLATE_CLASSES:
        return True
    return False


def _extract_text(region_attrs: dict) -> str:
    for key in TEXT_KEYS:
        val = region_attrs.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _extract_region_name(region_attrs: dict) -> str:
    val = region_attrs.get("region_name", "")
    return val.strip() if isinstance(val, str) else ""


def parse_via_json(json_path: Path) -> dict[str, list[tuple[int, int, int, int, str, str]]]:
    """Вернёт: filename -> [(x1, y1, x2, y2, text, region_name), ...] в пикселях.

    text — строка GT-номера (из поля `np`), либо "" если в разметке не указан.
    region_name — формат номера (eu-ua-2015, su, eu-ua-2004, ...) для per-kind метрик.
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("_via_img_metadata", data)
    out: dict[str, list[tuple[int, int, int, int, str, str]]] = {}

    for rec in meta.values():
        filename = rec.get("filename")
        regions = rec.get("regions") or []
        if isinstance(regions, dict):
            regions = list(regions.values())

        boxes = []
        for r in regions:
            attrs = r.get("region_attributes") or {}
            if not _is_plate(attrs):
                continue
            shape = r.get("shape_attributes") or {}
            if shape.get("name") != "polygon":
                continue
            xs = shape.get("all_points_x") or []
            ys = shape.get("all_points_y") or []
            if len(xs) < 3 or len(ys) < 3:
                continue
            x1, y1 = min(xs), min(ys)
            x2, y2 = max(xs), max(ys)
            if x2 <= x1 or y2 <= y1:
                continue
            text = _extract_text(attrs)
            rname = _extract_region_name(attrs)
            boxes.append((x1, y1, x2, y2, text, rname))

        if boxes:
            out[filename] = boxes
    return out


def box_to_yolo(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> tuple[float, float, float, float]:
    cx = (x1 + x2) / 2 / w
    cy = (y1 + y2) / 2 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return cx, cy, bw, bh


def process_split(
    split_name: str,
    src_dir: Path,
    dst_img_dir: Path,
    dst_lbl_dir: Path,
    dst_root: Path,
    subset: int | None,
    seed: int = 42,
    ocr_only: bool = False,
) -> tuple[int, int]:
    json_path = src_dir / "via_region_data.json"
    if not json_path.exists():
        print(f"[skip] {split_name}: нет {json_path.name}")
        return 0, 0

    print(f"[{split_name}] парсинг {json_path.name}...")
    ann = parse_via_json(json_path)
    print(f"[{split_name}] изображений с аннотациями: {len(ann)}")

    items = []
    missing = 0
    for fname, boxes in ann.items():
        img_path = src_dir / fname
        if not img_path.exists():
            missing += 1
            continue
        items.append((img_path, boxes))
    if missing:
        print(f"[{split_name}] не найдено файлов картинок: {missing}")

    if subset and len(items) > subset:
        random.Random(seed).shuffle(items)
        items = items[:subset]
    print(f"[{split_name}] итоговый размер: {len(items)}")

    if not ocr_only:
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    ocr_rows: list[tuple[str, int, int, int, int, str, str]] = []
    kept = 0
    text_count = 0
    for img_path, boxes in tqdm(items, desc=f"convert {split_name}"):
        img = imread(img_path) if not ocr_only else None
        if not ocr_only and img is None:
            continue

        lines = []
        for (x1, y1, x2, y2, text, rname) in boxes:
            if not ocr_only:
                h, w = img.shape[:2]
                cx, cy, bw, bh = box_to_yolo(x1, y1, x2, y2, w, h)
                if not (0 < bw <= 1 and 0 < bh <= 1):
                    continue
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            if text:
                ocr_rows.append((img_path.name, x1, y1, x2, y2, text, rname))
                text_count += 1

        if ocr_only:
            continue
        if not lines:
            continue
        stem = img_path.stem
        shutil.copy2(img_path, dst_img_dir / img_path.name)
        (dst_lbl_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        kept += 1

    ocr_csv = dst_root / f"ocr_{split_name}.csv"
    with open(ocr_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(("image", "x1", "y1", "x2", "y2", "text", "region_name"))
        w.writerows(ocr_rows)

    print(f"[{split_name}] изображений: {kept}, bbox с текстом: {text_count} → {ocr_csv.name}")
    return kept, text_count


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="autoriaNumberplateDataset-YYYY-MM-DD/")
    ap.add_argument("--dest", required=True, help="куда сложить yolo-датасет")
    ap.add_argument("--subset", type=int, default=4000, help="train subset size (val — все)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ocr-only", action="store_true",
                    help="Только ocr_{split}.csv, не копировать картинки и не писать YOLO-метки")
    args = ap.parse_args()

    src = Path(args.source)
    dst = Path(args.dest)
    dst.mkdir(parents=True, exist_ok=True)

    n_train, t_train = process_split(
        "train", src / "train",
        dst / "images/train", dst / "labels/train", dst,
        subset=args.subset, seed=args.seed, ocr_only=args.ocr_only,
    )
    n_val, t_val = process_split(
        "val", src / "val",
        dst / "images/val", dst / "labels/val", dst,
        subset=None, seed=args.seed, ocr_only=args.ocr_only,
    )

    if not args.ocr_only:
        data_yaml = dst / "data.yaml"
        data_yaml.write_text(
            f"path: {dst.resolve().as_posix()}\n"
            "train: images/train\n"
            "val: images/val\n"
            "names:\n"
            "  0: license_plate\n",
            encoding="utf-8",
        )
        print(f"data.yaml -> {data_yaml}")
    print(f"\nDone. train images={n_train} OCR-labels={t_train}, val images={n_val} OCR-labels={t_val}")
    if t_train == 0 and t_val == 0:
        print("⚠ В VIA JSON не найдено текстовых меток — CRNN надо обучать на synthetic_ocr/ или размечать вручную.")


if __name__ == "__main__":
    main()
