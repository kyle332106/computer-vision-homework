"""Объединить AUTO.RIA + keremberke HF plate dataset → yolo_plates_combined/.

keremberke/license-plate-object-detection — 8823 изображений с COCO-разметкой,
парковочные + street-view ракурсы. Дополняет AUTO.RIA (close-up car photos).

Использование:
    python scripts/prepare_combined_detector.py
"""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

from tqdm import tqdm


SRC_HF = Path("D:/Новая папка/alpr_data/hf_plates")
SRC_AUTORIA = Path("D:/Новая папка/alpr_data/yolo_plates")
DST = Path("D:/Новая папка/alpr_data/yolo_plates_combined")


def extract_hf_zips():
    """Распакуем train.zip/valid.zip/test.zip в подпапки, если ещё не распакованы."""
    for split in ("train", "valid", "test"):
        out = SRC_HF / split
        if out.exists() and any(out.iterdir()):
            print(f"[hf] {split}/ уже распакован")
            continue
        zf = SRC_HF / "data" / f"{split}.zip"
        if not zf.exists():
            print(f"[hf] нет {zf}")
            continue
        print(f"[hf] распаковываю {zf.name}...")
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zf) as z:
            z.extractall(out)


def coco_to_yolo(coco_json: Path, dst_lbl_dir: Path, dst_img_dir: Path, src_img_dir: Path, id_prefix: str) -> int:
    """Parse COCO JSON → write YOLO .txt labels. Copy images with unique prefix."""
    data = json.loads(coco_json.read_text(encoding="utf-8"))
    images = {img["id"]: img for img in data["images"]}
    # все классы → 0 (single-class detector)
    anns_by_img: dict[int, list] = {}
    for ann in data["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    dst_lbl_dir.mkdir(parents=True, exist_ok=True)
    dst_img_dir.mkdir(parents=True, exist_ok=True)

    kept = 0
    for img_id, img_info in tqdm(images.items(), desc=f"coco→yolo {id_prefix}"):
        w = img_info["width"]
        h = img_info["height"]
        src_path = src_img_dir / img_info["file_name"]
        if not src_path.exists():
            continue
        anns = anns_by_img.get(img_id, [])
        if not anns:
            continue
        # Unique name — избегаем коллизий с AUTO.RIA
        new_name = f"{id_prefix}_{img_info['file_name']}"
        new_stem = Path(new_name).stem

        lines = []
        for a in anns:
            x, y, bw, bh = a["bbox"]      # COCO: [x,y,w,h] absolute pixels
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw/w:.6f} {bh/h:.6f}")
        if not lines:
            continue

        shutil.copy2(src_path, dst_img_dir / new_name)
        (dst_lbl_dir / f"{new_stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        kept += 1
    print(f"  kept {kept} images+labels")
    return kept


def copy_autoria_split(split: str, dst_img_dir: Path, dst_lbl_dir: Path) -> int:
    src_img = SRC_AUTORIA / "images" / split
    src_lbl = SRC_AUTORIA / "labels" / split
    if not src_img.exists():
        print(f"[autoria] нет {src_img}")
        return 0
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for img in tqdm(list(src_img.iterdir()), desc=f"copy autoria {split}"):
        if img.is_file():
            shutil.copy2(img, dst_img_dir / img.name)
            lbl = src_lbl / f"{img.stem}.txt"
            if lbl.exists():
                shutil.copy2(lbl, dst_lbl_dir / lbl.name)
                n += 1
    return n


def main():
    extract_hf_zips()

    DST.mkdir(parents=True, exist_ok=True)

    # 1. Скопировать AUTO.RIA как есть (4000 train + 335 val)
    n_aut_tr = copy_autoria_split("train", DST / "images/train", DST / "labels/train")
    n_aut_vl = copy_autoria_split("val", DST / "images/val", DST / "labels/val")

    # 2. Добавить keremberke (train → train, valid → val)
    n_hf_tr = coco_to_yolo(
        SRC_HF / "train/_annotations.coco.json",
        DST / "labels/train", DST / "images/train",
        SRC_HF / "train", "hf",
    )
    n_hf_vl = coco_to_yolo(
        SRC_HF / "valid/_annotations.coco.json",
        DST / "labels/val", DST / "images/val",
        SRC_HF / "valid", "hf",
    )

    # 3. data.yaml
    data_yaml = DST / "data.yaml"
    data_yaml.write_text(
        f"path: {DST.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: license_plate\n",
        encoding="utf-8",
    )
    print(f"\nDone. train={n_aut_tr + n_hf_tr} (autoria={n_aut_tr} + hf={n_hf_tr})")
    print(f"     val={n_aut_vl + n_hf_vl} (autoria={n_aut_vl} + hf={n_hf_vl})")
    print(f"data.yaml → {data_yaml}")


if __name__ == "__main__":
    main()
