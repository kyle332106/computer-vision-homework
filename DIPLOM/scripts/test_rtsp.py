"""Быстрая проверка RTSP-потока + сырой ALPR-пайплайн.

Использование:
    python scripts/test_rtsp.py "rtsp://user:pass@host:port/path" --frames 50
    python scripts/test_rtsp.py "rtsp://..." --save-frame check.jpg

Флаги:
    --no-alpr   — просто проверить, что поток открывается
    --frames N  — сколько кадров прогнать через pipeline (по умолчанию 30)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.pipeline import ALPRPipeline, PipelineConfig


def open_stream(url: str) -> cv2.VideoCapture:
    # CAP_FFMPEG — самый надёжный backend для RTSP на Windows
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        # fallback — автовыбор
        cap = cv2.VideoCapture(url)
    # уменьшить задержку
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url", help="RTSP URL (rtsp://user:pass@host:port/path)")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--save-frame", default=None, help="сохранить первый кадр и выйти")
    ap.add_argument("--no-alpr", action="store_true", help="без детекции, просто проверить поток")
    ap.add_argument("--imgsz", type=int, default=1280, help="YOLO imgsz (дефолт 1280; для 2K+ камер 1536-1920)")
    ap.add_argument("--conf", type=float, default=0.2, help="YOLO confidence threshold")
    ap.add_argument("--weights", default="models/yolo26_plate.pt", help="веса детектора YOLO")
    ap.add_argument("--classical-backend", choices=["cpu", "nn"], default="cpu",
                    help="backend для classical plate finder")
    ap.add_argument("--classical-ocr-thr", type=float, default=0.8,
                    help="мин. OCR-confidence для classical-only кандидатов")
    ap.add_argument("--detect-every-n", type=int, default=5,
                    help="полная детекция раз в N кадров; между ними трекер")
    args = ap.parse_args()

    print(f"[rtsp] открываю {args.url.split('@')[-1] if '@' in args.url else args.url}")
    cap = open_stream(args.url)
    if not cap.isOpened():
        sys.exit("НЕ удалось открыть поток. Проверьте URL/креды/доступ к сети.")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    print(f"[rtsp] resolution={w}x{h}  declared_fps={fps:.1f}")

    if args.save_frame:
        ok, frame = cap.read()
        if not ok:
            cap.release()
            sys.exit("не смог прочитать первый кадр")
        cv2.imwrite(args.save_frame, frame)
        print(f"[rtsp] первый кадр → {args.save_frame}")
        cap.release()
        return

    pipeline = None
    if not args.no_alpr:
        pipeline = ALPRPipeline(PipelineConfig(
            detector_weights=args.weights,
            imgsz=args.imgsz,
            conf=args.conf,
            classical_backend=args.classical_backend,
            classical_min_ocr_conf=args.classical_ocr_thr,
            detect_every_n=args.detect_every_n,
        ))
        _ = pipeline.detector  # форс-загрузка
        print(
            f"[alpr] OCR backend = {pipeline.cfg.ocr_backend}  "
            f"imgsz={args.imgsz}  conf={args.conf}  "
            f"weights={args.weights}  classical={args.classical_backend}"
        )

    t0 = time.perf_counter()
    read_fails = 0
    detected_total = 0
    for i in range(args.frames):
        ok, frame = cap.read()
        if not ok:
            read_fails += 1
            if read_fails > 5:
                print("[rtsp] слишком много подряд неудачных read(), прерываю")
                break
            continue
        read_fails = 0
        if pipeline is None:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dets = pipeline.process_stream_frame(frame_rgb)
        if dets:
            detected_total += len(dets)
            texts = ", ".join(
                f"{d.text or '?'} [{getattr(d, 'plate_color', 'неизвестно')}]"
                for d in dets
            )
            print(f"  frame {i:3d}  plates={len(dets)}  [{texts}]")
    dt = time.perf_counter() - t0

    cap.release()
    eff_fps = args.frames / dt if dt > 0 else 0
    print(f"\n[summary] кадров прогнано: {args.frames}, эффективный FPS: {eff_fps:.2f}")
    if pipeline is not None:
        print(f"[summary] всего детекций номеров: {detected_total}")


if __name__ == "__main__":
    main()
