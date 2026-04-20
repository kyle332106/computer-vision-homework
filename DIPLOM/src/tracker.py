"""Обёртки над KCF и CSRT с возможностью сравнить их — как в ДЗ10.

Идея: YOLO детектирует раз в N кадров (тяжёлая операция),
между детекциями используется лёгкий трекер для плавности real-time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np


def make_tracker(kind: str = "csrt") -> cv2.Tracker:
    """Создать трекер по имени. kind: 'kcf' | 'csrt'."""
    kind = kind.lower()
    legacy = getattr(cv2, "legacy", None)
    if kind == "kcf":
        if hasattr(cv2, "TrackerKCF_create"):
            return cv2.TrackerKCF_create()
        if legacy is not None and hasattr(legacy, "TrackerKCF_create"):
            return legacy.TrackerKCF_create()
    if kind == "csrt":
        if hasattr(cv2, "TrackerCSRT_create"):
            return cv2.TrackerCSRT_create()
        if legacy is not None and hasattr(legacy, "TrackerCSRT_create"):
            return legacy.TrackerCSRT_create()
    raise ValueError(f"unknown tracker: {kind}")


def has_tracker(kind: str = "csrt") -> bool:
    try:
        _ = make_tracker(kind)
        return True
    except Exception:
        return False


@dataclass
class TrackedPlate:
    bbox: tuple[int, int, int, int]      # (x, y, w, h)
    text: str = ""
    tracker: cv2.Tracker | None = None
    text_votes: dict[str, float] = field(default_factory=dict)
    color_votes: dict[str, float] = field(default_factory=dict)
    age: int = 0                         # сколько кадров уже трекается
    last_detect_age: int = 0             # сколько кадров назад был последний YOLO-detect


@dataclass
class TrackerStats:
    """Статистика для сравнения трекеров (ДЗ10 стиль)."""
    name: str
    frames: int = 0
    successes: int = 0
    total_time: float = 0.0
    ious: list[float] = field(default_factory=list)

    @property
    def fps(self) -> float:
        return self.frames / self.total_time if self.total_time > 0 else 0.0

    @property
    def success_rate(self) -> float:
        return self.successes / self.frames if self.frames else 0.0

    @property
    def mean_iou(self) -> float:
        return float(np.mean(self.ious)) if self.ious else 0.0


def iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    """IoU для xywh-боксов."""
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def benchmark_trackers(
    frames: list[np.ndarray],
    init_bbox: tuple[int, int, int, int],
    gt_bboxes: list[tuple[int, int, int, int]] | None = None,
    kinds: tuple[str, ...] = ("kcf", "csrt"),
) -> dict[str, TrackerStats]:
    """Прогнать набор трекеров на одной последовательности кадров.

    frames[0] — кадр инициализации (bbox = init_bbox).
    frames[1:] — последующие кадры.
    gt_bboxes[i] — эталонный bbox для frames[i+1] (если есть).
    """
    results: dict[str, TrackerStats] = {}
    for kind in kinds:
        tracker = make_tracker(kind)
        tracker.init(frames[0], init_bbox)
        stats = TrackerStats(name=kind)

        for i, frame in enumerate(frames[1:]):
            t0 = time.perf_counter()
            ok, bbox = tracker.update(frame)
            stats.total_time += time.perf_counter() - t0
            stats.frames += 1
            if ok:
                stats.successes += 1
                if gt_bboxes is not None and i < len(gt_bboxes):
                    stats.ious.append(iou(tuple(map(int, bbox)), gt_bboxes[i]))
        results[kind] = stats
    return results
