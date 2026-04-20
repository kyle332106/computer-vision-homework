"""Классический детектор номерных знаков на фильтрах курса.

Цепочка использует исключительно техники из ДЗ/lesson'ов:
  1. Gray-world (ДЗ2) — нормализация цветового баланса
  2. Grayscale + Unsharp mask (ДЗ3) — подчёркивание контура символов
  3. Sobel/Canny (ДЗ4) — вертикальные градиенты (плашка = плотный лес вертикальных штрихов)
  4. Морфологическое closing горизонтальным ядром — фузит символы в сплошную "кляксу" плашки
  5. Otsu (ДЗ8) — бинаризация
  6. Contours → прямоугольники → фильтр по aspect ratio и размеру

Результат — список (x1, y1, x2, y2) plate-кандидатов, которые идут в OCR.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import preprocess


def enhance(img_rgb: np.ndarray) -> np.ndarray:
    """ДЗ2 + ДЗ3: color balance + sharpening для подчёркивания границ символов."""
    balanced = preprocess.gray_world(img_rgb)
    sharp = preprocess.unsharp_mask(balanced, sigma=1.2, amount=1.0)
    return sharp


def find_candidates(
    img_rgb: np.ndarray,
    min_aspect: float = 1.8,
    max_aspect: float = 6.5,
    min_h: int = 12,
    min_w: int = 40,
    max_w_frac: float = 0.5,
    max_h_frac: float = 0.3,
) -> list[tuple[int, int, int, int, float]]:
    """Вернуть список (x1, y1, x2, y2, score) plate-кандидатов.

    score — "plate-likeness" от 0 до 1, собран из aspect-ratio fit + edge-density.
    """
    H, W = img_rgb.shape[:2]

    # 1. Enhance (ДЗ2 + ДЗ3)
    enhanced = enhance(img_rgb)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)

    # 2. Вертикальные градиенты (ДЗ4): Sobel по X — ловим штрихи символов
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_x = np.absolute(sobel_x)
    sobel_x = cv2.normalize(sobel_x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 3. Otsu на градиент-мапе (ДЗ8) — выделяем сильные edges
    _, edges = cv2.threshold(sobel_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. Морфологическое closing горизонтальным rect — фузит символы в плашку
    #    Размер ядра масштабируется от ширины кадра: кератин ~3% ширины
    kw = max(15, int(W * 0.012))
    kh = max(3, int(H * 0.006))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    # ещё одно opening маленьким ядром — убрать тонкие шумные линии
    closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    # 5. Contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[tuple[int, int, int, int, float]] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h < min_h or w < min_w:
            continue
        if w > W * max_w_frac or h > H * max_h_frac:
            continue
        aspect = w / max(h, 1)
        if not (min_aspect <= aspect <= max_aspect):
            continue
        # edge density внутри bbox — у реальных плашек высокая
        roi = edges[y:y + h, x:x + w]
        density = float(roi.mean()) / 255.0
        if density < 0.15:
            continue
        # солидность контура относительно bbox — отсекаем L-образные/ломаные
        area = cv2.contourArea(cnt)
        solidity = area / max(w * h, 1)
        if solidity < 0.25:
            continue
        # score = combination: aspect в sweet-spot 2.5..4.5 + высокая плотность + solidity
        aspect_fit = 1.0 - abs(aspect - 3.5) / 3.5
        score = 0.4 * max(0, aspect_fit) + 0.4 * density + 0.2 * solidity
        candidates.append((x, y, x + w, y + h, score))

    return candidates


def merge_with_yolo(
    yolo_boxes: list[tuple[int, int, int, int, float]],
    classical_boxes: list[tuple[int, int, int, int, float]],
    iou_thr: float = 0.3,
) -> list[tuple[int, int, int, int, float, str]]:
    """NMS-объединение. Возвращает список (x1,y1,x2,y2,conf,source).

    source ∈ {'yolo', 'cv', 'both'}. YOLO при перекрытии доминирует (обычно точнее).
    """
    def _iou(a, b):
        xa = max(a[0], b[0]); ya = max(a[1], b[1])
        xb = min(a[2], b[2]); yb = min(a[3], b[3])
        inter = max(0, xb - xa) * max(0, yb - ya)
        u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / u if u > 0 else 0

    out: list[tuple[int, int, int, int, float, str]] = []
    classical_used = [False] * len(classical_boxes)
    for yb in yolo_boxes:
        matched = False
        for i, cb in enumerate(classical_boxes):
            if classical_used[i]:
                continue
            if _iou(yb, cb) > iou_thr:
                classical_used[i] = True
                matched = True
                break
        out.append((*yb, "both" if matched else "yolo"))
    for i, cb in enumerate(classical_boxes):
        if not classical_used[i]:
            out.append((*cb, "cv"))
    return out
