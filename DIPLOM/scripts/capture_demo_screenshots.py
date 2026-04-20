"""Захват свежих скриншотов для demo_artifacts из RTSP-потока."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.pipeline import ALPRPipeline, PipelineConfig, draw_detections


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url", help="RTSP URL")
    ap.add_argument("--frames", type=int, default=30, help="сколько кадров пропустить до сохранения")
    ap.add_argument("--output-dir", default="demo_artifacts/screenshots", help="папка для сохранения")
    ap.add_argument("--imgsz", type=int, default=1536, help="YOLO imgsz")
    ap.add_argument("--conf", type=float, default=0.1, help="confidence threshold")
    ap.add_argument("--weights", default="models/yolo26_plate_combined.pt", help="веса детектора")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[capture] открываю {args.url.split('@')[-1] if '@' in args.url else args.url}")
    cap = cv2.VideoCapture(args.url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.url)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    if not cap.isOpened():
        sys.exit("Не удалось открыть RTSP-поток.")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    print(f"[capture] resolution={w}x{h}  fps={fps:.1f}")

    # Инициализируем пайплайн
    pipeline = ALPRPipeline(PipelineConfig(
        detector_weights=args.weights,
        imgsz=args.imgsz,
        conf=args.conf,
        use_classical_finder=False,
        classical_min_ocr_conf=0.95,
        use_tracker=False,
        final_min_area=800,
        min_plate_distance=150,  # мин. расстояние между номерами на разных машинах
    ))
    print(f"[capture] пайплайн инициализирован (conf={args.conf}, imgsz={args.imgsz})")

    clean_frame = None
    merged_frame = None
    frame_count = 0

    for i in range(args.frames):
        ok, frame_bgr = cap.read()
        if not ok:
            print(f"[capture] ошибка чтения кадра {i}")
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if i == 0:
            # Сохраняем первый чистый кадр
            clean_frame = frame_rgb.copy()
            clean_path = output_dir / "01_rtsp_clean.jpg"
            clean_bgr = cv2.cvtColor(clean_frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(clean_path), clean_bgr)
            print(f"[capture] ✓ сохранён чистый кадр → {clean_path}")

        # Обработка
        detections = pipeline.process_stream_frame(frame_rgb)

        # Если есть детекции, сохраняем с отрисовкой
        if detections and merged_frame is None:
            merged_frame = draw_detections(frame_rgb.copy(), detections)
            merged_path = output_dir / "02_rtsp_merged.jpg"
            merged_bgr = cv2.cvtColor(merged_frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(merged_path), merged_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # Выведем информацию о каждой детекции для отладки
            print(f"[capture] ✓ сохранён merged кадр с {len(detections)} детекциями:")
            for j, d in enumerate(detections, 1):
                print(f"    #{j}: text='{d.text}' raw_text='{d.raw_text}' bbox={d.bbox} source={d.source}")

        if i % 10 == 0:
            print(f"[capture] обработано {i} кадров...")

        frame_count = i + 1
        if clean_frame is not None and merged_frame is not None:
            print(f"[capture] оба кадра получены, выход")
            break

    cap.release()

    if clean_frame is not None:
        print(f"[capture] ✓ готово (обработано {frame_count} кадров)")
    else:
        print(f"[capture] ⚠ не удалось захватить кадры")


if __name__ == "__main__":
    main()
