"""Конвертация Zenodo synthetic_ua в формат labels_{split}.csv для CRNN.

Zenodo-датасет использует НАСТОЯЩИЙ UA-plate-шрифт. Имя файла = сам номер.
Это закрывает главный domain-gap нашей мультишрифтовой синтетики.

Использование:
    python scripts/prepare_zenodo_ocr.py
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

from tqdm import tqdm


def process(src_images: Path, dst_images: Path, split_name: str) -> list[tuple[str, str, str]]:
    if not src_images.exists():
        return []
    dst_images.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, str, str]] = []
    for img in tqdm(sorted(src_images.glob("*.png")), desc=f"zenodo {split_name}"):
        text = img.stem.upper()
        if not (2 <= len(text) <= 10):
            continue
        dst = dst_images / img.name
        if not dst.exists():
            shutil.copy2(img, dst)
        rows.append((img.name, text, "zenodo_ua"))
    return rows


def main() -> None:
    src = Path("D:/Новая папка/alpr_data/synthetic_ua")
    dst = Path(__file__).resolve().parent.parent / "data" / "zenodo_ocr"

    splits = {
        "train": src / "train" / "images",
        "val": src / "valid" / "images",
    }
    for split, src_dir in splits.items():
        dst_images = dst / "images" / split
        rows = process(src_dir, dst_images, split)
        csv_out = dst / f"labels_{split}.csv"
        with open(csv_out, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(("file", "text", "kind"))
            w.writerows(rows)
        print(f"[{split}] {len(rows)} → {csv_out}")


if __name__ == "__main__":
    main()
