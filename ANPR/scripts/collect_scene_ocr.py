"""Сбор scene-specific OCR датасета из RTSP по consensus label.

Идея: прогоняем текущий ALPR-pipeline по потоку, группируем детекции по позиции
в кадре, копим vote'ы по тексту. Сохраняем только те группы, где один и тот же
текст стабильно подтверждается много раз. Это даёт clean pseudo-labels для
дообучения CRNN именно под конкретную камеру/сцену.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.pipeline import ALPRPipeline, PipelineConfig


@dataclass
class Group:
    bbox: tuple[int, int, int, int]
    votes: dict[str, float] = field(default_factory=dict)
    samples: list[tuple[float, Path, str]] = field(default_factory=list)
    hits: int = 0


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    xa = max(a[0], b[0]); ya = max(a[1], b[1])
    xb = min(a[2], b[2]); yb = min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    ua = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    ub = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    union = ua + ub - inter
    return inter / union if union > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--weights", default="models/yolo26_plate_combined.pt")
    ap.add_argument("--out", default="data/scene_ocr")
    ap.add_argument("--min-hits", type=int, default=8)
    ap.add_argument("--min-vote-share", type=float, default=0.75)
    ap.add_argument("--max-per-group", type=int, default=24)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    out_root = root / args.out
    if out_root.exists():
        shutil.rmtree(out_root)
    tmp_dir = out_root / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    pipe = ALPRPipeline(PipelineConfig(
        detector_weights=args.weights,
        imgsz=args.imgsz,
        conf=args.conf,
        classical_backend="cpu",
        classical_min_ocr_conf=0.8,
        use_tracker=False,
    ))

    cap = cv2.VideoCapture(args.url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.url)
    if not cap.isOpened():
        raise SystemExit("RTSP open failed")

    groups: list[Group] = []
    processed = 0
    for frame_idx in range(args.frames):
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detections = pipe.process_frame(frame_rgb)
        processed += 1
        for det in detections:
            if det.rectified is None or det.rectified.size == 0 or not det.text:
                continue
            weight = det.ocr_conf + det.conf + (0.3 if det.source == "both" else 0.0)
            match = None
            best_iou = 0.0
            for g in groups:
                cur_iou = iou(det.bbox, g.bbox)
                if cur_iou > best_iou and cur_iou >= 0.25:
                    best_iou = cur_iou
                    match = g
            if match is None:
                match = Group(det.bbox)
                groups.append(match)
            match.hits += 1
            match.votes[det.text] = match.votes.get(det.text, 0.0) + weight
            sample_path = tmp_dir / f"f{frame_idx:04d}_{len(match.samples):03d}.png"
            cv2.imwrite(str(sample_path), cv2.cvtColor(det.rectified, cv2.COLOR_RGB2BGR))
            match.samples.append((weight, sample_path, det.text))

    cap.release()
    print(f"processed frames: {processed}, groups: {len(groups)}")

    train_img = out_root / "images" / "train"
    val_img = out_root / "images" / "val"
    train_img.mkdir(parents=True, exist_ok=True)
    val_img.mkdir(parents=True, exist_ok=True)
    train_rows: list[tuple[str, str, str]] = []
    val_rows: list[tuple[str, str, str]] = []

    rng = random.Random(0)
    kept_groups = 0
    for gi, g in enumerate(groups):
        total_vote = sum(g.votes.values())
        if g.hits < args.min_hits or total_vote <= 0:
            continue
        text, top_vote = max(g.votes.items(), key=lambda kv: kv[1])
        share = top_vote / total_vote
        if share < args.min_vote_share:
            continue
        chosen = [(w, p, t) for (w, p, t) in g.samples if t == text]
        chosen.sort(key=lambda x: -x[0])
        chosen = chosen[:args.max_per_group]
        if len(chosen) < args.min_hits:
            continue
        kept_groups += 1
        rng.shuffle(chosen)
        split_idx = max(1, int(round(len(chosen) * 0.2)))
        val_set = set(p for _, p, _ in chosen[:split_idx])
        print(f"group {gi}: text={text} hits={g.hits} share={share:.2f} keep={len(chosen)}")
        for si, (_, src, _) in enumerate(chosen):
            dst_dir = val_img if src in val_set else train_img
            new_name = f"scene_g{gi:02d}_{si:03d}.png"
            shutil.copy2(src, dst_dir / new_name)
            row = (new_name, text, "scene_rtsp")
            if src in val_set:
                val_rows.append(row)
            else:
                train_rows.append(row)

    with open(out_root / "labels_train.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(("file", "text", "kind")); w.writerows(train_rows)
    with open(out_root / "labels_val.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(("file", "text", "kind")); w.writerows(val_rows)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"kept groups: {kept_groups}")
    print(f"scene train={len(train_rows)} val={len(val_rows)} -> {out_root}")


if __name__ == "__main__":
    main()
