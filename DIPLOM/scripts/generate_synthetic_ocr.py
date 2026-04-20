"""Генерация синтетического датасета для CRNN-OCR — все форматы UA + именные.

Почему это нужно:
  AUTO.RIA (реальный) — в основном стандартные UA-номера, один шрифт.
  Zenodo synthetic — тоже один формат.
  Для "любой формат/шрифт/именные" нужен свой генератор.

Что покрываем:
  1. Standard UA (новый стандарт, белый фон)           AA1234BB
  2. UA diplomatic (красный фон)                        001CD001
  3. UA спецтехника (синий фон, "С")                   СН-0001-АА
  4. UA такси (жёлтый фон)                              AA1234TAX
  5. UA military (зелёный)                              ВТ12345
  6. Named / vanity — произвольные латиница+кириллица+цифры+тире 2–8 символов
  7. Иностранные (белый, разные форматы)               ABC-123-DE, 123ABC456

Шрифты — все .ttf/.otf из системной папки Windows (C:/Windows/Fonts)
плюс любые из `DIPLOM/data/fonts/` если присутствуют.

Использование:
    python scripts/generate_synthetic_ocr.py --out data/synthetic_ocr --n 30000
"""

from __future__ import annotations

import argparse
import csv
import random
import string
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.io_utils import imwrite


LATIN = string.ascii_uppercase
CYRILLIC = "АБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯ"
DIGITS = string.digits

# Палитры (R, G, B) фон + цвет текста — под каждый тип номера.
PALETTES: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    "standard": ((250, 250, 250), (20, 20, 20)),
    "diplomat": ((190, 30, 30), (245, 245, 245)),
    "spec":     ((30, 60, 180), (245, 245, 245)),
    "taxi":     ((240, 210, 50), (20, 20, 20)),
    "military": ((40, 90, 40), (245, 245, 245)),
    "named":    ((250, 250, 250), (20, 20, 20)),
    "foreign":  ((250, 250, 250), (20, 20, 20)),
}

KIND_WEIGHTS = {
    "standard": 0.35,
    "named":    0.25,
    "foreign":  0.15,
    "diplomat": 0.08,
    "spec":     0.07,
    "taxi":     0.05,
    "military": 0.05,
}


# ---------------------------------------------------------------------------
# Генерация строк
# ---------------------------------------------------------------------------

def _gen_standard() -> str:
    # AA 1234 BB — 2 буквы, 4 цифры, 2 буквы (кириллица UA на официальных)
    ua_letters = "АВСЕНІКМОРТХ"
    return (
        "".join(random.choices(ua_letters, k=2))
        + "".join(random.choices(DIGITS, k=4))
        + "".join(random.choices(ua_letters, k=2))
    )


def _gen_diplomat() -> str:
    # три цифры страны + CD/CC + три цифры авто, латиница
    return (
        "".join(random.choices(DIGITS, k=3))
        + random.choice(["CD", "CC", "CT"])
        + "".join(random.choices(DIGITS, k=3))
    )


def _gen_spec() -> str:
    # С + 4 цифры + 2 буквы (например, полиция, ДСНС)
    return "С" + "".join(random.choices(DIGITS, k=4)) + "".join(random.choices("АВГДМН", k=2))


def _gen_taxi() -> str:
    letters = "АВСЕНІКМОРТХ"
    return (
        "".join(random.choices(letters, k=2))
        + "".join(random.choices(DIGITS, k=4))
        + "".join(random.choices(letters, k=2))
    )


def _gen_military() -> str:
    return "".join(random.choices("ВТ", k=2)) + "".join(random.choices(DIGITS, k=5))


def _gen_named() -> str:
    # именной/vanity: 2–8 символов, смесь букв + цифр + опц. тире
    length = random.randint(2, 8)
    alphabet = LATIN + CYRILLIC + DIGITS
    s = "".join(random.choices(alphabet, k=length))
    if length > 4 and random.random() < 0.2:
        i = random.randint(2, length - 2)
        s = s[:i] + "-" + s[i:]
    return s


def _gen_foreign() -> str:
    patterns = [
        lambda: "".join(random.choices(LATIN, k=3)) + "-" + "".join(random.choices(DIGITS, k=3)),
        lambda: "".join(random.choices(DIGITS, k=3)) + "".join(random.choices(LATIN, k=2)) + "".join(random.choices(DIGITS, k=3)),
        lambda: "".join(random.choices(LATIN, k=2)) + "-" + "".join(random.choices(DIGITS, k=4)),
        lambda: "".join(random.choices(LATIN, k=4)) + "".join(random.choices(DIGITS, k=3)),
    ]
    return random.choice(patterns)()


GENERATORS = {
    "standard": _gen_standard,
    "diplomat": _gen_diplomat,
    "spec":     _gen_spec,
    "taxi":     _gen_taxi,
    "military": _gen_military,
    "named":    _gen_named,
    "foreign":  _gen_foreign,
}


def pick_kind() -> str:
    r = random.random()
    cum = 0.0
    for k, w in KIND_WEIGHTS.items():
        cum += w
        if r <= cum:
            return k
    return "standard"


# ---------------------------------------------------------------------------
# Рендер одной "таблички"
# ---------------------------------------------------------------------------

def load_fonts(custom_dir: Path | None) -> list[Path]:
    fonts: list[Path] = []
    win_fonts = Path("C:/Windows/Fonts")
    if win_fonts.exists():
        # берём разнообразные шрифты для максимального покрытия
        candidates = [
            "arial.ttf", "arialbd.ttf", "calibri.ttf", "calibrib.ttf",
            "consola.ttf", "consolab.ttf", "cour.ttf", "courbd.ttf",
            "times.ttf", "timesbd.ttf", "verdana.ttf", "verdanab.ttf",
            "impact.ttf", "tahoma.ttf", "tahomabd.ttf",
            "framd.ttf", "georgia.ttf", "trebuc.ttf",
        ]
        for name in candidates:
            p = win_fonts / name
            if p.exists():
                fonts.append(p)
    if custom_dir and custom_dir.exists():
        fonts.extend(custom_dir.glob("*.ttf"))
        fonts.extend(custom_dir.glob("*.otf"))
    if not fonts:
        raise RuntimeError("не найдено ни одного шрифта .ttf")
    return fonts


def render_plate(text: str, kind: str, font_path: Path, out_h: int = 64) -> np.ndarray:
    bg, fg = PALETTES[kind]
    # подбираем размер шрифта по высоте изображения
    font_size = random.randint(int(out_h * 0.55), int(out_h * 0.75))
    try:
        font = ImageFont.truetype(str(font_path), size=font_size)
    except Exception:
        font = ImageFont.load_default()

    # оценим ширину
    tmp = Image.new("RGB", (10, 10), bg)
    d = ImageDraw.Draw(tmp)
    try:
        bbox = d.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        tw, th = d.textsize(text, font=font)

    padding = max(8, out_h // 4)
    W = tw + padding * 2
    H = out_h
    img = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(img)
    draw.text(((W - tw) // 2 - (bbox[0] if 'bbox' in locals() else 0),
               (H - th) // 2 - (bbox[1] if 'bbox' in locals() else 0)),
              text, font=font, fill=fg)

    arr = np.array(img)
    arr = _augment(arr)
    return arr


def _augment(img: np.ndarray) -> np.ndarray:
    """Лёгкие аугментации — шум, блюр, лёгкий перспективный перекос."""
    h, w = img.shape[:2]

    # лёгкая перспектива
    if random.random() < 0.6:
        dx = int(w * random.uniform(-0.03, 0.03))
        dy = int(h * random.uniform(-0.05, 0.05))
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([
            [0 + max(0, dx), 0 + max(0, dy)],
            [w + min(0, dx), 0 + max(0, -dy)],
            [w + min(0, -dx), h + min(0, dy)],
            [0 + max(0, -dx), h + min(0, -dy)],
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, M, (w, h), borderValue=tuple(int(c) for c in img[0, 0].tolist()))

    # блюр
    if random.random() < 0.4:
        k = random.choice([3, 3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    # шум
    if random.random() < 0.5:
        sigma = random.uniform(2, 12)
        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # яркость/контраст
    if random.random() < 0.6:
        alpha = random.uniform(0.7, 1.3)
        beta = random.uniform(-25, 25)
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic_ocr")
    ap.add_argument("--n", type=int, default=20000, help="всего примеров")
    ap.add_argument("--train-frac", type=float, default=0.9)
    ap.add_argument("--fonts-dir", default="data/fonts", help="доп. шрифты (если есть)")
    ap.add_argument("--out-h", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    out = Path(args.out)
    img_train = out / "images/train"
    img_val = out / "images/val"
    img_train.mkdir(parents=True, exist_ok=True)
    img_val.mkdir(parents=True, exist_ok=True)

    fonts = load_fonts(Path(args.fonts_dir))
    print(f"[fonts] найдено {len(fonts)} шрифтов")

    label_rows = {"train": [], "val": []}
    n_train = int(args.n * args.train_frac)

    for i in tqdm(range(args.n), desc="render"):
        kind = pick_kind()
        text = GENERATORS[kind]()
        font_path = random.choice(fonts)
        try:
            img = render_plate(text, kind, font_path, out_h=args.out_h)
        except Exception as exc:
            print(f"[skip] {text} / {font_path.name}: {exc}")
            continue

        split = "train" if i < n_train else "val"
        fname = f"{i:07d}_{kind}.png"
        out_path = (img_train if split == "train" else img_val) / fname
        imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        label_rows[split].append((fname, text, kind))

    for split, rows in label_rows.items():
        with open(out / f"labels_{split}.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(("file", "text", "kind"))
            w.writerows(rows)

    print(f"\nDone. train={len(label_rows['train'])}, val={len(label_rows['val'])}")
    print(f"Labels: {out}/labels_train.csv, {out}/labels_val.csv")


if __name__ == "__main__":
    main()
