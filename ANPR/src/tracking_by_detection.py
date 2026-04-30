"""Lightweight SORT / ByteTrack-style tracking-by-detection for ANPR.

The tracker keeps stable IDs for license-plate detections. It uses the core
ideas from SORT (Kalman prediction + IoU assignment) and ByteTrack (first match
high-confidence detections, then recover tracks with low-confidence detections).
No external tracker package is required; scipy is used for Hungarian assignment
when available, with a greedy fallback otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrackDetection:
    bbox: tuple[int, int, int, int]  # xyxy
    score: float
    index: int = -1


@dataclass
class TrackResult:
    track_id: int
    bbox: tuple[int, int, int, int]  # xyxy
    score: float
    detection_index: int = -1
    hits: int = 0
    age: int = 0
    time_since_update: int = 0


def xyxy_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _bbox_to_z(bbox: tuple[float, float, float, float]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    scale = w * h
    ratio = w / h
    return np.array([[cx], [cy], [scale], [ratio]], dtype=float)


def _x_to_bbox(x: np.ndarray) -> tuple[float, float, float, float]:
    cx = float(x[0, 0])
    cy = float(x[1, 0])
    scale = max(1.0, float(x[2, 0]))
    ratio = max(0.01, float(x[3, 0]))
    w = np.sqrt(scale * ratio)
    h = scale / max(w, 1.0)
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def _linear_assignment(cost_matrix: np.ndarray) -> list[tuple[int, int]]:
    if cost_matrix.size == 0:
        return []
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(cost_matrix)
        return list(zip(rows.tolist(), cols.tolist()))
    except Exception:
        matches: list[tuple[int, int]] = []
        used_rows: set[int] = set()
        used_cols: set[int] = set()
        flat = [
            (float(cost_matrix[r, c]), r, c)
            for r in range(cost_matrix.shape[0])
            for c in range(cost_matrix.shape[1])
        ]
        for _, row, col in sorted(flat):
            if row in used_rows or col in used_cols:
                continue
            matches.append((row, col))
            used_rows.add(row)
            used_cols.add(col)
        return matches


class KalmanBoxTrack:
    """SORT-like constant-velocity Kalman track for one plate bbox."""

    def __init__(self, detection: TrackDetection, track_id: int):
        self.track_id = track_id
        self.score = detection.score
        self.detection_index = detection.index

        self.x = np.zeros((7, 1), dtype=float)
        self.x[:4] = _bbox_to_z(tuple(map(float, detection.bbox)))

        self.f = np.eye(7, dtype=float)
        for i in range(3):
            self.f[i, i + 4] = 1.0
        self.h = np.zeros((4, 7), dtype=float)
        self.h[:4, :4] = np.eye(4, dtype=float)

        self.p = np.eye(7, dtype=float) * 10.0
        self.p[4:, 4:] *= 1000.0
        self.r = np.eye(4, dtype=float)
        self.r[2:, 2:] *= 10.0
        self.q = np.eye(7, dtype=float) * 0.01
        self.q[4:, 4:] *= 0.01

        self.age = 0
        self.hits = 1
        self.hit_streak = 1
        self.time_since_update = 0

    def predict(self) -> tuple[float, float, float, float]:
        if self.x[2] + self.x[6] <= 0:
            self.x[6] = 0
        self.x = self.f @ self.x
        self.p = self.f @ self.p @ self.f.T + self.q
        self.age += 1
        self.time_since_update += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        return self.bbox

    def update(self, detection: TrackDetection) -> None:
        z = _bbox_to_z(tuple(map(float, detection.bbox)))
        y = z - self.h @ self.x
        s = self.h @ self.p @ self.h.T + self.r
        k = self.p @ self.h.T @ np.linalg.inv(s)
        self.x = self.x + k @ y
        self.p = (np.eye(7) - k @ self.h) @ self.p

        self.score = detection.score
        self.detection_index = detection.index
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return _x_to_bbox(self.x)

    def result(self) -> TrackResult:
        x1, y1, x2, y2 = self.bbox
        return TrackResult(
            track_id=self.track_id,
            bbox=(int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
            score=self.score,
            detection_index=self.detection_index,
            hits=self.hits,
            age=self.age,
            time_since_update=self.time_since_update,
        )


class ByteTrackPlateTracker:
    """Small ByteTrack/SORT-style tracker for license plates."""

    def __init__(
        self,
        high_thresh: float = 0.45,
        low_thresh: float = 0.10,
        match_iou: float = 0.30,
        low_match_iou: float = 0.20,
        max_age: int = 5,
        min_hits: int = 1,
        use_low_confidence: bool = True,
    ):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_iou = match_iou
        self.low_match_iou = low_match_iou
        self.max_age = max_age
        self.min_hits = min_hits
        self.use_low_confidence = use_low_confidence
        self.tracks: list[KalmanBoxTrack] = []
        self.frame_count = 0
        self._next_track_id = 1

    def reset(self) -> None:
        self.tracks = []
        self.frame_count = 0
        self._next_track_id = 1

    def update(self, detections: list[TrackDetection]) -> list[TrackResult]:
        self.frame_count += 1
        for track in self.tracks:
            track.predict()

        high = [d for d in detections if d.score >= self.high_thresh]
        low = [
            d for d in detections
            if self.low_thresh <= d.score < self.high_thresh
        ]

        unmatched_tracks = list(range(len(self.tracks)))
        unmatched_high = list(range(len(high)))

        matches, unmatched_tracks, unmatched_high = self._match(
            unmatched_tracks,
            high,
            self.match_iou,
        )
        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(high[det_idx])

        if self.use_low_confidence and low and unmatched_tracks:
            low_matches, unmatched_tracks, _ = self._match(
                unmatched_tracks,
                low,
                self.low_match_iou,
            )
            for track_idx, det_idx in low_matches:
                self.tracks[track_idx].update(low[det_idx])

        for det_idx in unmatched_high:
            self.tracks.append(KalmanBoxTrack(high[det_idx], self._next_track_id))
            self._next_track_id += 1

        self.tracks = [
            track for track in self.tracks
            if track.time_since_update <= self.max_age
        ]

        results: list[TrackResult] = []
        for track in self.tracks:
            if track.hits >= self.min_hits and track.time_since_update <= self.max_age:
                results.append(track.result())
        return results

    def _match(
        self,
        track_indices: list[int],
        detections: list[TrackDetection],
        iou_threshold: float,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not track_indices or not detections:
            return [], track_indices, list(range(len(detections)))

        iou_matrix = np.zeros((len(track_indices), len(detections)), dtype=float)
        for row, track_idx in enumerate(track_indices):
            track_box = self.tracks[track_idx].bbox
            for col, det in enumerate(detections):
                iou_matrix[row, col] = xyxy_iou(track_box, tuple(map(float, det.bbox)))

        matches: list[tuple[int, int]] = []
        used_rows: set[int] = set()
        used_cols: set[int] = set()
        for row, col in _linear_assignment(1.0 - iou_matrix):
            if iou_matrix[row, col] < iou_threshold:
                continue
            matches.append((track_indices[row], col))
            used_rows.add(row)
            used_cols.add(col)

        unmatched_tracks = [
            track_idx for row, track_idx in enumerate(track_indices)
            if row not in used_rows
        ]
        unmatched_dets = [
            det_idx for det_idx in range(len(detections))
            if det_idx not in used_cols
        ]
        return matches, unmatched_tracks, unmatched_dets
