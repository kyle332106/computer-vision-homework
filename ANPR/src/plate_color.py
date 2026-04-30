"""Определение цвета номерного знака в контексте правил, а не сырых RGB-оттенков.

Конечные классы — не "любой наблюдаемый цвет пикселей", а допустимые типы
фона номерных знаков, с которыми имеет смысл работать в ALPR-домене:
  белый, жёлтый, красный, синий, зелёный, неизвестно
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PlateColorInfo:
    key: str
    name: str
    context: str


PLATE_COLOR_RULES = {
    "white": PlateColorInfo("white", "белый", "обычный гражданский / базовый тип"),
    "yellow": PlateColorInfo("yellow", "жёлтый", "такси / коммерческий спецтип"),
    "red": PlateColorInfo("red", "красный", "дипломатический"),
    "blue": PlateColorInfo("blue", "синий", "ведомственный / специальный"),
    "green": PlateColorInfo("green", "зелёный", "военный"),
    "unknown": PlateColorInfo("unknown", "неизвестно", "тип фона не определён"),
}


def _dominant_background_mask(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Оценка маски фона плашки.

    Берём не весь центр номера, а в первую очередь фоновые полосы по краям
    таблички: верх/низ и узкие боковые зоны. Так меньше шанс, что тёмные
    символы потянут фон в "серый".
    """
    h, w = rgb.shape[:2]
    if h < 8 or w < 16:
        return rgb, np.ones((h, w), dtype=bool)

    # Обрезаем внешнюю рамку, чтобы не путать цвет автомобиля/бампера с цветом номера.
    top = int(h * 0.10)
    bottom = int(h * 0.90)
    left = int(w * 0.08)
    right = int(w * 0.92)
    roi = rgb[top:bottom, left:right]
    hh, ww = roi.shape[:2]

    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]

    # Фон плашки почти всегда хорошо виден в верхней/нижней полосе и по краям.
    border_mask = np.zeros((hh, ww), dtype=bool)
    band_h = max(2, int(hh * 0.22))
    band_w = max(2, int(ww * 0.10))
    border_mask[:band_h, :] = True
    border_mask[-band_h:, :] = True
    border_mask[:, :band_w] = True
    border_mask[:, -band_w:] = True

    border_v = v[border_mask]
    border_s = s[border_mask]
    bright_thr = float(np.percentile(border_v, 35))
    sat_thr = float(np.percentile(border_s, 60))

    # Символы обычно темнее; оставляем светлый или насыщенный фон в рамках border_mask.
    mask = border_mask & ((v >= bright_thr) | (s >= sat_thr))
    return roi, mask


def classify_plate_color(rectified_rgb: np.ndarray) -> tuple[str, float]:
    """Совместимый wrapper: вернуть только (название_цвета, confidence)."""
    info, conf = classify_plate_color_info(rectified_rgb)
    return info.name, conf


def classify_plate_color_info(rectified_rgb: np.ndarray) -> tuple[PlateColorInfo, float]:
    """Вернуть (PlateColorInfo, confidence)."""
    if rectified_rgb is None or rectified_rgb.size == 0:
        return PLATE_COLOR_RULES["unknown"], 0.0

    roi, mask = _dominant_background_mask(rectified_rgb)
    if mask.sum() < 20:
        return PLATE_COLOR_RULES["unknown"], 0.0

    pixels = roi[mask]
    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    h = hsv[:, 0].astype(np.float32) * 2.0   # OpenCV hue: 0..179 -> degrees
    s = hsv[:, 1].astype(np.float32)
    v = hsv[:, 2].astype(np.float32)

    mean_s = float(np.mean(s))
    mean_v = float(np.mean(v))
    hue = float(np.median(h))
    p70_v = float(np.percentile(v, 70))

    colored = s >= 55
    ratios = {
        "yellow": float(np.mean(colored & (h >= 20) & (h <= 75) & (v > 120))),
        "red": float(np.mean(colored & ((h <= 15) | (h >= 340)))),
        "blue": float(np.mean(colored & (h >= 180) & (h <= 260))),
        "green": float(np.mean(colored & (h > 75) & (h < 170))),
    }
    best_color = max(ratios, key=ratios.get)
    best_ratio = ratios[best_color]

    # Низкая насыщенность → achromatic colors
    if mean_s < 28:
        if mean_v > 190:
            return PLATE_COLOR_RULES["white"], 0.95
        if mean_v > 120:
            return PLATE_COLOR_RULES["white"], 0.76
        return PLATE_COLOR_RULES["unknown"], 0.35
    if mean_s < 55 and mean_v > 165:
        return PLATE_COLOR_RULES["white"], 0.82

    # Цветной фон имеет смысл только если этот оттенок занимает заметную долю плашки.
    # Это отсекает флаг UA, блики и цвет кузова по краям бокса.
    if best_ratio >= 0.35:
        conf = min(0.96, 0.72 + best_ratio * 0.6)
        return PLATE_COLOR_RULES[best_color], conf

    # Если явного доминирующего цветного фона нет, но фон светлый — это почти
    # всегда белая/светлая плашка, а не "синяя/зелёная".
    if mean_v > 150 and p70_v > 170:
        return PLATE_COLOR_RULES["white"], 0.78

    # Fallback: только если оттенок устойчиво тянет к одному из допустимых
    # доменных цветов. "Серый" и "чёрный" не выдаём как конечные классы.
    if 15 < hue < 45:
        return PLATE_COLOR_RULES["yellow"], 0.70
    if 260 < hue < 340:
        return PLATE_COLOR_RULES["blue"], 0.55
    if mean_v > 110:
        return PLATE_COLOR_RULES["white"], 0.45
    return PLATE_COLOR_RULES["unknown"], 0.35
