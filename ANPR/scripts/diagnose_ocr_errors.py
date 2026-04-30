"""Диагностика OCR ошибок на текущем фрейме."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.pipeline import ALPRPipeline, PipelineConfig


def main():
    url = "rtsp://admin:qwer0987@192.168.110.206:554/snl/live/1/1"
    
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        sys.exit("Не удалось открыть поток")

    pipeline = ALPRPipeline(PipelineConfig(
        detector_weights="models/yolo26_plate_combined.pt",
        imgsz=1536,
        conf=0.1,
        use_classical_finder=False,
    ))

    ok, frame_bgr = cap.read()
    if ok:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detections = pipeline.process_stream_frame(frame_rgb)
        
        print(f"\n{'='*70}")
        print(f"ДЕТЕКЦИИ: {len(detections)}")
        print(f"{'='*70}\n")
        
        for i, d in enumerate(detections, 1):
            print(f"[Детекция #{i}]")
            print(f"  Текст распознанный:   {repr(d.text)}")
            print(f"  Текст сырой (raw):    {repr(d.raw_text)}")
            print(f"  OCR confidence:       {d.ocr_conf:.4f}")
            print(f"  Источник:             {d.source}")
            print(f"  BBox:                 {d.bbox}")
            print(f"  Crop shape:           {d.crop.shape if d.crop is not None else 'None'}")
            
            if d.rectified is not None:
                print(f"  Rectified shape:      {d.rectified.shape}")
                # Сохраняем для анализа
                rect_bgr = cv2.cvtColor(d.rectified, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"det_{i}_rectified.png", rect_bgr)
                print(f"  → сохранён в det_{i}_rectified.png")
            
            print()

    cap.release()


if __name__ == "__main__":
    main()
