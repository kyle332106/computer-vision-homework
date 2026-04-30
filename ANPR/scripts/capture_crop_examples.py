"""Захват примеров кропов номерных знаков для демонстрации препроцессинга."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.pipeline import ALPRPipeline, PipelineConfig


def main():
    url = "rtsp://admin:qwer0987@192.168.110.206:554/snl/live/1/1"
    
    print("[crops] открываю поток")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(url)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    if not cap.isOpened():
        sys.exit("Не удалось открыть RTSP-поток.")

    pipeline = ALPRPipeline(PipelineConfig(
        detector_weights="models/yolo26_plate_combined.pt",
        imgsz=1536,
        conf=0.1,
        use_classical_finder=False,
    ))

    output_dir = Path("demo_artifacts/screenshots")
    output_dir.mkdir(parents=True, exist_ok=True)

    captured = False
    
    for i in range(50):
        ok, frame_bgr = cap.read()
        if not ok:
            break
        
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detections = pipeline.process_stream_frame(frame_rgb)
        
        # Берём первый номер с высокой уверенностью
        for d in detections:
            if d.ocr_conf > 0.95 and len(d.text) >= 6 and not captured:
                # Сохраняем исходный crop
                crop_path = output_dir / "03_crop_raw.png"
                crop_bgr = cv2.cvtColor(d.crop, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(crop_path), crop_bgr)
                print(f"[crops] ✓ исходный crop ({d.crop.shape}) → {crop_path}")
                print(f"         текст: {d.text}, conf: {d.ocr_conf:.3f}")
                
                # Сохраняем препроцессированный crop (выравненный)
                if d.rectified is not None:
                    rect_path = output_dir / "04_crop_preprocessed.png"
                    rect_bgr = cv2.cvtColor(d.rectified, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(rect_path), rect_bgr)
                    print(f"[crops] ✓ препроцессированный ({d.rectified.shape}) → {rect_path}")
                
                captured = True
                break
        
        if captured:
            break
        
        if i % 10 == 0:
            print(f"[crops] кадр {i}...")

    cap.release()
    
    if not captured:
        print("[crops] ⚠ не удалось захватить примеры кропов")
    else:
        print("[crops] готово!")


if __name__ == "__main__":
    main()
