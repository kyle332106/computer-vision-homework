"""Диагностический скрипт для проверки детекций из RTSP."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.pipeline import ALPRPipeline, PipelineConfig


def main():
    url = "rtsp://admin:qwer0987@192.168.110.206:554/snl/live/1/1"
    
    print("[diag] открываю поток")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(url)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    if not cap.isOpened():
        sys.exit("Не удалось открыть RTSP-поток.")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    print(f"[diag] resolution={w}x{h}")

    pipeline = ALPRPipeline(PipelineConfig(
        detector_weights="models/yolo26_plate_combined.pt",
        imgsz=1536,
        conf=0.1,
        use_classical_finder=False,
    ))

    for i in range(5):
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detections = pipeline.process_stream_frame(frame_rgb)
        
        print(f"\n[frame {i}] {len(detections)} детекций:")
        for j, d in enumerate(detections):
            print(f"  #{j+1}: text='{d.text}' raw_text='{d.raw_text}' ocr_conf={d.ocr_conf:.3f} source={d.source} bbox={d.bbox}")

    cap.release()


if __name__ == "__main__":
    main()
